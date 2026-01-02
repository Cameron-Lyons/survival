use crate::constants::{COX_MAX_ITER, PARALLEL_THRESHOLD_SMALL};
use crate::utilities::numpy_utils::{extract_optional_vec_f64, extract_vec_f64, extract_vec_i32};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
pub struct CVResult {
    #[pyo3(get)]
    pub fold_scores: Vec<f64>,
    #[pyo3(get)]
    pub mean_score: f64,
    #[pyo3(get)]
    pub std_score: f64,
    #[pyo3(get)]
    pub fold_coefficients: Vec<Vec<f64>>,
}
#[pymethods]
impl CVResult {
    #[new]
    fn new(
        fold_scores: Vec<f64>,
        mean_score: f64,
        std_score: f64,
        fold_coefficients: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            fold_scores,
            mean_score,
            std_score,
            fold_coefficients,
        }
    }
}
pub struct CVConfig {
    pub n_folds: usize,
    pub shuffle: bool,
    pub seed: Option<u64>,
}
impl Default for CVConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            shuffle: true,
            seed: None,
        }
    }
}
fn simple_shuffle(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let mut state = seed.wrapping_add(i as u64);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}
fn create_folds(n: usize, n_folds: usize, shuffle: bool, seed: Option<u64>) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        let seed = seed.unwrap_or(42);
        simple_shuffle(&mut indices, seed);
    }
    let fold_size = n / n_folds;
    let remainder = n % n_folds;
    let mut folds = Vec::with_capacity(n_folds);
    let mut start = 0;
    for i in 0..n_folds {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(indices[start..end].to_vec());
        start = end;
    }
    folds
}
pub fn cv_cox(
    time: &[f64],
    status: &[i32],
    covariates: &Array2<f64>,
    weights: Option<&[f64]>,
    config: &CVConfig,
) -> Result<CVResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::coxfit6::{CoxFitBuilder, Method as CoxMethod};
    use ndarray::Array1;
    let n = time.len();
    let nvar = covariates.nrows();
    let default_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&default_weights);
    let folds = create_folds(n, config.n_folds, config.shuffle, config.seed);
    let results: Vec<(f64, Vec<f64>)> = (0..config.n_folds)
        .into_par_iter()
        .map(|fold_idx| {
            let test_indices = &folds[fold_idx];
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            let train_n = train_indices.len();
            let test_n = test_indices.len();
            let train_time: Vec<f64> = train_indices.iter().map(|&i| time[i]).collect();
            let train_status: Vec<i32> = train_indices.iter().map(|&i| status[i]).collect();
            let train_weights: Vec<f64> = train_indices.iter().map(|&i| weights[i]).collect();
            let mut train_covariates = Array2::zeros((train_n, nvar));
            for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
                for var in 0..nvar {
                    train_covariates[[new_idx, var]] = covariates[[var, orig_idx]];
                }
            }
            let mut sorted_indices: Vec<usize> = (0..train_n).collect();
            sorted_indices.sort_by(|&a, &b| {
                train_time[b]
                    .partial_cmp(&train_time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sorted_time: Vec<f64> = sorted_indices.iter().map(|&i| train_time[i]).collect();
            let sorted_status: Vec<i32> = sorted_indices.iter().map(|&i| train_status[i]).collect();
            let sorted_weights: Vec<f64> =
                sorted_indices.iter().map(|&i| train_weights[i]).collect();
            let mut sorted_covariates = Array2::zeros((train_n, nvar));
            for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
                for var in 0..nvar {
                    sorted_covariates[[new_idx, var]] = train_covariates[[orig_idx, var]];
                }
            }
            let time_arr = Array1::from_vec(sorted_time);
            let status_arr = Array1::from_vec(sorted_status);
            let weights_arr = Array1::from_vec(sorted_weights);
            let beta = match CoxFitBuilder::new(time_arr, status_arr, sorted_covariates)
                .weights(weights_arr)
                .method(CoxMethod::Breslow)
                .max_iter(COX_MAX_ITER)
                .eps(1e-9)
                .toler(1e-9)
                .build()
            {
                Ok(mut fit) => {
                    if fit.fit().is_ok() {
                        let (b, _, _, _, _, _, _, _) = fit.results();
                        b
                    } else {
                        vec![0.0; nvar]
                    }
                }
                Err(_) => vec![0.0; nvar],
            };
            let test_time: Vec<f64> = test_indices.iter().map(|&i| time[i]).collect();
            let test_status: Vec<i32> = test_indices.iter().map(|&i| status[i]).collect();
            let mut test_covariates = Array2::zeros((nvar, test_n));
            for (new_idx, &orig_idx) in test_indices.iter().enumerate() {
                for var in 0..nvar {
                    test_covariates[[var, new_idx]] = covariates[[var, orig_idx]];
                }
            }
            let linear_predictor: Vec<f64> = (0..test_n)
                .map(|i| {
                    (0..nvar)
                        .map(|var| beta[var] * test_covariates[[var, i]])
                        .sum()
                })
                .collect();

            let (concordant, discordant, tied) = if test_n > PARALLEL_THRESHOLD_SMALL {
                (0..test_n)
                    .into_par_iter()
                    .filter(|&i| test_status[i] == 1)
                    .map(|i| {
                        let mut c = 0.0;
                        let mut d = 0.0;
                        let mut t = 0.0;
                        for j in 0..test_n {
                            if i != j && test_time[j] > test_time[i] {
                                if linear_predictor[i] > linear_predictor[j] {
                                    c += 1.0;
                                } else if linear_predictor[i] < linear_predictor[j] {
                                    d += 1.0;
                                } else {
                                    t += 1.0;
                                }
                            }
                        }
                        (c, d, t)
                    })
                    .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
            } else {
                let mut concordant = 0.0;
                let mut discordant = 0.0;
                let mut tied = 0.0;
                for i in 0..test_n {
                    if test_status[i] != 1 {
                        continue;
                    }
                    for j in 0..test_n {
                        if i != j && test_time[j] > test_time[i] {
                            if linear_predictor[i] > linear_predictor[j] {
                                concordant += 1.0;
                            } else if linear_predictor[i] < linear_predictor[j] {
                                discordant += 1.0;
                            } else {
                                tied += 1.0;
                            }
                        }
                    }
                }
                (concordant, discordant, tied)
            };
            let total = concordant + discordant + tied;
            let c_index = if total > 0.0 {
                (concordant + 0.5 * tied) / total
            } else {
                0.5
            };
            (c_index, beta)
        })
        .collect();
    let (fold_scores, fold_coefficients): (Vec<f64>, Vec<Vec<f64>>) = results.into_iter().unzip();
    let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance = fold_scores
        .iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>()
        / (fold_scores.len() - 1) as f64;
    let std_score = variance.sqrt();
    Ok(CVResult {
        fold_scores,
        mean_score,
        std_score,
        fold_coefficients,
    })
}
#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, n_folds=None, shuffle=None, seed=None))]
pub fn cv_cox_concordance(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    covariates: Vec<Vec<f64>>,
    weights: Option<&Bound<'_, PyAny>>,
    n_folds: Option<usize>,
    shuffle: Option<bool>,
    seed: Option<u64>,
) -> PyResult<CVResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let weights = extract_optional_vec_f64(weights)?;
    let n = time.len();
    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = CVConfig {
        n_folds: n_folds.unwrap_or(5),
        shuffle: shuffle.unwrap_or(true),
        seed,
    };
    let weights_ref = weights.as_deref();
    cv_cox(&time, &status, &cov_array, weights_ref, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}
pub fn cv_survreg(
    time: &[f64],
    status: &[f64],
    covariates: &Array2<f64>,
    distribution: &str,
    config: &CVConfig,
) -> Result<CVResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::survreg6::survreg;
    let n = time.len();
    let nvar = covariates.ncols();
    let cov_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nvar).map(|j| covariates[[j, i]]).collect())
        .collect();
    let folds = create_folds(n, config.n_folds, config.shuffle, config.seed);
    let results: Vec<(f64, Vec<f64>)> = (0..config.n_folds)
        .into_par_iter()
        .filter_map(|fold_idx| {
            let test_indices = &folds[fold_idx];
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();
            let train_time: Vec<f64> = train_indices.iter().map(|&i| time[i]).collect();
            let train_status: Vec<f64> = train_indices.iter().map(|&i| status[i]).collect();
            let train_covariates: Vec<Vec<f64>> =
                train_indices.iter().map(|&i| cov_vecs[i].clone()).collect();
            let fit_result = survreg(
                train_time,
                train_status,
                train_covariates,
                None,
                None,
                None,
                None,
                Some(distribution),
                Some(COX_MAX_ITER),
                Some(1e-5),
                Some(1e-9),
            )
            .ok()?;
            let test_time: Vec<f64> = test_indices.iter().map(|&i| time[i]).collect();
            let test_status: Vec<f64> = test_indices.iter().map(|&i| status[i]).collect();
            let test_covariates: Vec<Vec<f64>> =
                test_indices.iter().map(|&i| cov_vecs[i].clone()).collect();
            let test_fit = survreg(
                test_time,
                test_status,
                test_covariates,
                None,
                None,
                Some(fit_result.coefficients.clone()),
                None,
                Some(distribution),
                Some(1),
                Some(1e-5),
                Some(1e-9),
            )
            .ok()?;
            Some((test_fit.log_likelihood, fit_result.coefficients))
        })
        .collect();
    if results.is_empty() {
        return Err("All CV folds failed".into());
    }
    let (fold_scores, fold_coefficients): (Vec<f64>, Vec<Vec<f64>>) = results.into_iter().unzip();
    let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let variance = fold_scores
        .iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>()
        / (fold_scores.len() - 1) as f64;
    let std_score = variance.sqrt();
    Ok(CVResult {
        fold_scores,
        mean_score,
        std_score,
        fold_coefficients,
    })
}
#[pyfunction]
#[pyo3(signature = (time, status, covariates, distribution=None, n_folds=None, shuffle=None, seed=None))]
pub fn cv_survreg_loglik(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    covariates: Vec<Vec<f64>>,
    distribution: Option<&str>,
    n_folds: Option<usize>,
    shuffle: Option<bool>,
    seed: Option<u64>,
) -> PyResult<CVResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_f64(status)?;
    let n = time.len();
    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = CVConfig {
        n_folds: n_folds.unwrap_or(5),
        shuffle: shuffle.unwrap_or(true),
        seed,
    };
    let dist = distribution.unwrap_or("weibull");
    cv_survreg(&time, &status, &cov_array, dist, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}
