use crate::constants::{
    COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE, DEFAULT_MAX_ITER,
    PARALLEL_THRESHOLD_SMALL,
};
use crate::internal::numpy_utils::{extract_optional_vec_f64, extract_vec_f64, extract_vec_i32};
use crate::internal::statistical::lcg64_shuffle_per_index_seed;
use crate::internal::validation::{
    validate_binary_f64, validate_binary_i32, validate_finite, validate_non_negative,
};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
pub(crate) struct CVConfig {
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

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn validate_cv_config(n_folds: usize, n_obs: usize) -> PyResult<()> {
    if n_folds < 2 || n_folds > n_obs {
        return Err(value_error(
            "n_folds must be between 2 and the number of observations",
        ));
    }
    Ok(())
}

fn validate_time_status_i32(time: &[f64], status: &[i32]) -> PyResult<()> {
    if time.is_empty() || status.len() != time.len() {
        return Err(value_error(
            "time and status must have the same non-zero length",
        ));
    }
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

fn validate_time_status_f64(time: &[f64], status: &[f64]) -> PyResult<()> {
    if time.is_empty() || status.len() != time.len() {
        return Err(value_error(
            "time and status must have the same non-zero length",
        ));
    }
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_finite(status, "status")?;
    validate_binary_f64(status, "status")?;
    Ok(())
}

fn validate_covariates(covariates: &[Vec<f64>], n_obs: usize) -> PyResult<usize> {
    if covariates.len() != n_obs {
        return Err(value_error("covariates length must match time length"));
    }
    let nvar = covariates.first().map_or(0, Vec::len);
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != nvar {
            return Err(value_error(format!(
                "covariates row {row_idx} length must match the first row"
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "covariates contains non-finite value {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }
    Ok(nvar)
}

fn validate_weights(weights: Option<&[f64]>, n_obs: usize) -> PyResult<()> {
    let Some(weights) = weights else {
        return Ok(());
    };
    if weights.len() != n_obs {
        return Err(value_error("weights length must match time length"));
    }
    validate_finite(weights, "weights")?;
    validate_non_negative(weights, "weights")?;
    Ok(())
}

fn create_folds(n: usize, n_folds: usize, shuffle: bool, seed: Option<u64>) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        let seed = seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
        lcg64_shuffle_per_index_seed(&mut indices, seed);
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

fn covariate_rows_for_indices(
    covariates: &Array2<f64>,
    nvar: usize,
    indices: &[usize],
) -> Vec<Vec<f64>> {
    let mut rows = Vec::with_capacity(indices.len());
    for &obs_idx in indices {
        let mut row = Vec::with_capacity(nvar);
        for var in 0..nvar {
            row.push(covariates[[var, obs_idx]]);
        }
        rows.push(row);
    }
    rows
}

pub(crate) fn cv_cox(
    time: &[f64],
    status: &[i32],
    covariates: &Array2<f64>,
    weights: Option<&[f64]>,
    config: &CVConfig,
) -> Result<CVResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::cox_optimizer::{CoxFitBuilder, Method as CoxMethod};
    use ndarray::Array1;
    let n = time.len();
    let nvar = covariates.nrows();
    let folds = create_folds(n, config.n_folds, config.shuffle, config.seed);
    let results: Vec<(f64, Vec<f64>)> = (0..config.n_folds)
        .into_par_iter()
        .map(|fold_idx| {
            let test_indices = &folds[fold_idx];
            let test_n = test_indices.len();
            let train_n = n - test_n;
            let mut sorted_train_indices = Vec::with_capacity(train_n);
            for (idx, fold) in folds.iter().enumerate() {
                if idx != fold_idx {
                    sorted_train_indices.extend_from_slice(fold);
                }
            }
            sorted_train_indices.sort_by(|&a, &b| time[b].total_cmp(&time[a]));

            let mut sorted_time = Vec::with_capacity(train_n);
            let mut sorted_status = Vec::with_capacity(train_n);
            let mut sorted_weights = weights.map(|_| Vec::with_capacity(train_n));
            let mut sorted_covariates = Array2::zeros((train_n, nvar));
            for (new_idx, &orig_idx) in sorted_train_indices.iter().enumerate() {
                sorted_time.push(time[orig_idx]);
                sorted_status.push(status[orig_idx]);
                if let (Some(input_weights), Some(sorted_weights)) =
                    (weights, sorted_weights.as_mut())
                {
                    sorted_weights.push(input_weights[orig_idx]);
                }
                for var in 0..nvar {
                    sorted_covariates[[new_idx, var]] = covariates[[var, orig_idx]];
                }
            }
            let time_arr = Array1::from_vec(sorted_time);
            let status_arr = Array1::from_vec(sorted_status);
            let mut builder = CoxFitBuilder::new(time_arr, status_arr, sorted_covariates)
                .method(CoxMethod::Breslow)
                .max_iter(COX_MAX_ITER)
                .eps(COX_CONVERGENCE_TOLERANCE)
                .toler(COX_RANK_TOLERANCE);
            if let Some(sorted_weights) = sorted_weights {
                builder = builder.weights(Array1::from_vec(sorted_weights));
            }
            let beta = match builder.build() {
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
            let linear_predictor: Vec<f64> = test_indices
                .iter()
                .map(|&orig_idx| {
                    let mut eta = 0.0;
                    for var in 0..nvar {
                        eta += beta[var] * covariates[[var, orig_idx]];
                    }
                    eta
                })
                .collect();

            let (concordant, discordant, tied) = if test_n > PARALLEL_THRESHOLD_SMALL {
                (0..test_n)
                    .into_par_iter()
                    .filter(|&i| status[test_indices[i]] == 1)
                    .map(|i| {
                        let mut c = 0.0;
                        let mut d = 0.0;
                        let mut t = 0.0;
                        let orig_i = test_indices[i];
                        for j in 0..test_n {
                            let orig_j = test_indices[j];
                            if i != j && time[orig_j] > time[orig_i] {
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
                    let orig_i = test_indices[i];
                    if status[orig_i] != 1 {
                        continue;
                    }
                    for j in 0..test_n {
                        let orig_j = test_indices[j];
                        if i != j && time[orig_j] > time[orig_i] {
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
    validate_time_status_i32(&time, &status)?;
    validate_weights(weights.as_deref(), n)?;
    let n_folds = n_folds.unwrap_or_else(|| 5.min(n));
    validate_cv_config(n_folds, n)?;
    let nvar = validate_covariates(&covariates, n)?;
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = CVConfig {
        n_folds,
        shuffle: shuffle.unwrap_or(true),
        seed,
    };
    let weights_ref = weights.as_deref();
    cv_cox(&time, &status, &cov_array, weights_ref, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}
pub(crate) fn cv_survreg(
    time: &[f64],
    status: &[f64],
    covariates: &Array2<f64>,
    distribution: &str,
    config: &CVConfig,
) -> Result<CVResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::parametric_survival::survreg;
    let n = time.len();
    let nvar = covariates.nrows();
    let folds = create_folds(n, config.n_folds, config.shuffle, config.seed);
    let results: Vec<(f64, Vec<f64>)> = (0..config.n_folds)
        .into_par_iter()
        .filter_map(|fold_idx| {
            let test_indices = &folds[fold_idx];
            let mut train_indices = Vec::with_capacity(n - test_indices.len());
            for (idx, fold) in folds.iter().enumerate() {
                if idx != fold_idx {
                    train_indices.extend_from_slice(fold);
                }
            }
            let train_time: Vec<f64> = train_indices.iter().map(|&i| time[i]).collect();
            let train_status: Vec<f64> = train_indices.iter().map(|&i| status[i]).collect();
            let train_covariates = covariate_rows_for_indices(covariates, nvar, &train_indices);
            let fit_result = survreg(
                train_time,
                train_status,
                train_covariates,
                None,
                None,
                None,
                None,
                Some(distribution),
                Some(DEFAULT_MAX_ITER),
                Some(1e-5),
                Some(1e-9),
                None,
                None,
                None,
            )
            .ok()?;
            let test_time: Vec<f64> = test_indices.iter().map(|&i| time[i]).collect();
            let test_status: Vec<f64> = test_indices.iter().map(|&i| status[i]).collect();
            let test_covariates = covariate_rows_for_indices(covariates, nvar, test_indices);
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
                None,
                None,
                None,
            )
            .ok()?;
            Some((test_fit.log_likelihood, fit_result.coefficients))
        })
        .collect();
    if results.is_empty() {
        return Err("All CV folds failed".into());
    }
    if results.len() < 2 {
        return Err("At least two CV folds must complete successfully".into());
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
    validate_time_status_f64(&time, &status)?;
    let n_folds = n_folds.unwrap_or_else(|| 5.min(n));
    validate_cv_config(n_folds, n)?;
    let nvar = validate_covariates(&covariates, n)?;
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = CVConfig {
        n_folds,
        shuffle: shuffle.unwrap_or(true),
        seed,
    };
    let dist = distribution.unwrap_or("weibull");
    cv_survreg(&time, &status, &cov_array, dist, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_test_data(n: usize) -> (Vec<f64>, Vec<i32>, Array2<f64>) {
        let time: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let mut covariates = Array2::zeros((1, n));
        for i in 0..n {
            covariates[[0, i]] = (i as f64) / (n as f64);
        }
        (time, status, covariates)
    }

    #[test]
    fn cv_cox_k3() {
        let (time, status, covariates) = make_test_data(30);
        let config = CVConfig {
            n_folds: 3,
            shuffle: true,
            seed: Some(42),
        };
        let result = cv_cox(&time, &status, &covariates, None, &config).unwrap();
        assert_eq!(result.fold_scores.len(), 3);
    }

    #[test]
    fn cv_cox_k5() {
        let (time, status, covariates) = make_test_data(30);
        let config = CVConfig {
            n_folds: 5,
            shuffle: true,
            seed: Some(42),
        };
        let result = cv_cox(&time, &status, &covariates, None, &config).unwrap();
        assert_eq!(result.fold_scores.len(), 5);
    }

    #[test]
    fn cv_cox_reproducibility() {
        let (time, status, covariates) = make_test_data(30);
        let config = CVConfig {
            n_folds: 5,
            shuffle: true,
            seed: Some(123),
        };
        let result1 = cv_cox(&time, &status, &covariates, None, &config).unwrap();
        let result2 = cv_cox(&time, &status, &covariates, None, &config).unwrap();
        for i in 0..result1.fold_scores.len() {
            assert!((result1.fold_scores[i] - result2.fold_scores[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn cv_cox_unweighted_matches_unit_weights() {
        let (time, status, covariates) = make_test_data(30);
        let weights = vec![1.0; time.len()];
        let config = CVConfig {
            n_folds: 5,
            shuffle: true,
            seed: Some(123),
        };

        let unweighted = cv_cox(&time, &status, &covariates, None, &config).unwrap();
        let weighted = cv_cox(&time, &status, &covariates, Some(&weights), &config).unwrap();

        assert_eq!(unweighted.fold_scores.len(), weighted.fold_scores.len());
        for (unweighted_score, weighted_score) in
            unweighted.fold_scores.iter().zip(&weighted.fold_scores)
        {
            assert!((unweighted_score - weighted_score).abs() < 1e-12);
        }
        assert!((unweighted.mean_score - weighted.mean_score).abs() < 1e-12);
        assert!((unweighted.std_score - weighted.std_score).abs() < 1e-12);
        assert_eq!(
            unweighted.fold_coefficients.len(),
            weighted.fold_coefficients.len()
        );
        for (unweighted_coef, weighted_coef) in unweighted
            .fold_coefficients
            .iter()
            .zip(&weighted.fold_coefficients)
        {
            assert_eq!(unweighted_coef.len(), weighted_coef.len());
            for (unweighted_value, weighted_value) in unweighted_coef.iter().zip(weighted_coef) {
                assert!((unweighted_value - weighted_value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cv_cox_different_seeds() {
        let (time, status, covariates) = make_test_data(30);
        let config1 = CVConfig {
            n_folds: 5,
            shuffle: true,
            seed: Some(42),
        };
        let config2 = CVConfig {
            n_folds: 5,
            shuffle: true,
            seed: Some(999),
        };
        let result1 = cv_cox(&time, &status, &covariates, None, &config1).unwrap();
        let result2 = cv_cox(&time, &status, &covariates, None, &config2).unwrap();
        let all_same = result1
            .fold_scores
            .iter()
            .zip(&result2.fold_scores)
            .all(|(a, b)| (a - b).abs() < 1e-10);
        assert!(!all_same);
    }

    #[test]
    fn cv_survreg_uses_covariate_rows_as_variable_count() {
        let n = 30;
        let time: Vec<f64> = (1..=n as i64).map(|i| i as f64).collect();
        let status: Vec<f64> = vec![1.0; n];
        let mut covariates = Array2::zeros((1, n));
        for i in 0..n {
            covariates[[0, i]] = i as f64 / n as f64;
        }
        let config = CVConfig {
            n_folds: 3,
            shuffle: true,
            seed: Some(42),
        };

        let result = cv_survreg(&time, &status, &covariates, "weibull", &config).unwrap();

        assert_eq!(result.fold_scores.len(), 3);
        assert_eq!(result.fold_coefficients.len(), 3);
        assert!(
            result
                .fold_coefficients
                .iter()
                .all(|coefficients| !coefficients.is_empty())
        );
    }

    #[test]
    fn public_crossval_helpers_validate_inputs() {
        pyo3::Python::initialize();
        assert!(
            validate_cv_config(1, 10)
                .expect_err("single fold should fail")
                .to_string()
                .contains("n_folds must be between 2")
        );
        assert!(
            validate_cv_config(11, 10)
                .expect_err("too many folds should fail")
                .to_string()
                .contains("n_folds must be between 2")
        );
        assert!(
            validate_time_status_i32(&[1.0, f64::NAN], &[1, 0])
                .expect_err("non-finite time should fail")
                .to_string()
                .contains("time contains non-finite")
        );
        assert!(
            validate_time_status_i32(&[1.0, 2.0], &[1, 2])
                .expect_err("non-binary status should fail")
                .to_string()
                .contains("status values must be 0 or 1")
        );
        assert!(
            validate_covariates(&[vec![0.1], vec![0.2, 0.3]], 2)
                .expect_err("ragged covariates should fail")
                .to_string()
                .contains("covariates row 1 length")
        );
        assert!(
            validate_weights(Some(&[1.0, f64::INFINITY]), 2)
                .expect_err("non-finite weights should fail")
                .to_string()
                .contains("weights contains non-finite")
        );
    }
}
