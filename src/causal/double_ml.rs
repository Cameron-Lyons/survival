#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DoubleMLConfig {
    #[pyo3(get, set)]
    pub n_folds: usize,
    #[pyo3(get, set)]
    pub n_rep: usize,
    #[pyo3(get, set)]
    pub score: String,
    #[pyo3(get, set)]
    pub trimming_threshold: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl DoubleMLConfig {
    #[new]
    #[pyo3(signature = (
        n_folds=5,
        n_rep=1,
        score=None,
        trimming_threshold=0.01,
        seed=None
    ))]
    pub fn new(
        n_folds: usize,
        n_rep: usize,
        score: Option<String>,
        trimming_threshold: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }
        Ok(Self {
            n_folds,
            n_rep,
            score: score.unwrap_or_else(|| "ATE".to_string()),
            trimming_threshold,
            seed,
        })
    }
}

fn create_folds(n: usize, n_folds: usize, rng: &mut fastrand::Rng) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut indices);

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

fn fit_nuisance_model(
    x: &[Vec<f64>],
    y: &[f64],
    train_idx: &[usize],
    _test_idx: &[usize],
) -> Vec<f64> {
    let n_train = train_idx.len();
    let n_features = x[0].len();

    let mut x_mean = vec![0.0; n_features];
    let mut y_mean = 0.0;

    for &i in train_idx {
        for (j, &xij) in x[i].iter().enumerate() {
            x_mean[j] += xij;
        }
        y_mean += y[i];
    }

    for m in &mut x_mean {
        *m /= n_train as f64;
    }
    y_mean /= n_train as f64;

    let mut xtx = vec![vec![0.0; n_features]; n_features];
    let mut xty = vec![0.0; n_features];

    for &i in train_idx {
        let xi_centered: Vec<f64> = x[i]
            .iter()
            .zip(x_mean.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let yi_centered = y[i] - y_mean;

        for j in 0..n_features {
            for k in 0..n_features {
                xtx[j][k] += xi_centered[j] * xi_centered[k];
            }
            xty[j] += xi_centered[j] * yi_centered;
        }
    }

    #[allow(clippy::needless_range_loop)]
    for j in 0..n_features {
        xtx[j][j] += 0.01;
    }

    let mut beta = vec![0.0; n_features];
    #[allow(clippy::needless_range_loop)]
    for j in 0..n_features {
        if xtx[j][j].abs() > 1e-10 {
            beta[j] = xty[j] / xtx[j][j];
        }
    }

    x.iter()
        .map(|xi| {
            let pred: f64 = xi
                .iter()
                .zip(x_mean.iter())
                .zip(beta.iter())
                .map(|((&xij, &mj), &bj)| (xij - mj) * bj)
                .sum();
            pred + y_mean
        })
        .collect()
}

fn fit_propensity_model(x: &[Vec<f64>], d: &[i32], train_idx: &[usize], trimming: f64) -> Vec<f64> {
    let n_train = train_idx.len();
    let n_features = x[0].len();

    let mut x_mean = vec![0.0; n_features];
    let mut d_mean = 0.0;

    for &i in train_idx {
        for (j, &xij) in x[i].iter().enumerate() {
            x_mean[j] += xij;
        }
        d_mean += d[i] as f64;
    }

    for m in &mut x_mean {
        *m /= n_train as f64;
    }
    d_mean /= n_train as f64;

    let mut beta = vec![0.0; n_features];
    for j in 0..n_features {
        let mut num = 0.0;
        let mut denom = 0.0;

        for &i in train_idx {
            let xij_centered = x[i][j] - x_mean[j];
            let di = d[i] as f64;
            num += xij_centered * (di - d_mean);
            denom += xij_centered * xij_centered;
        }

        if denom.abs() > 1e-10 {
            beta[j] = num / denom;
        }
    }

    x.iter()
        .map(|xi| {
            let logit: f64 = xi
                .iter()
                .zip(x_mean.iter())
                .zip(beta.iter())
                .map(|((&xij, &mj), &bj)| (xij - mj) * bj)
                .sum();
            let prob = 1.0 / (1.0 + (-logit).exp());
            prob.clamp(trimming, 1.0 - trimming)
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DoubleMLResult {
    #[pyo3(get)]
    pub ate: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub pvalue: f64,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub scores: Vec<f64>,
}

#[pymethods]
impl DoubleMLResult {
    fn __repr__(&self) -> String {
        format!(
            "DoubleMLResult(ATE={:.4}, SE={:.4}, p={:.4})",
            self.ate, self.se, self.pvalue
        )
    }

    fn is_significant(&self, alpha: f64) -> bool {
        self.pvalue < alpha
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    treatment,
    outcome,
    time,
    event,
    config=None
))]
pub fn double_ml_survival(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    outcome: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<DoubleMLConfig>,
) -> PyResult<DoubleMLResult> {
    let config = config.unwrap_or_else(|| DoubleMLConfig::new(5, 1, None, 0.01, None).unwrap());

    let n = covariates.len();
    if n == 0 || treatment.len() != n || outcome.len() != n || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let mut all_scores = Vec::new();

    for _rep in 0..config.n_rep {
        let folds = create_folds(n, config.n_folds, &mut rng);

        let mut y_residuals = vec![0.0; n];
        let mut d_residuals = vec![0.0; n];

        for (fold_idx, test_fold) in folds.iter().enumerate() {
            let train_idx: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();

            let y_pred = fit_nuisance_model(&covariates, &outcome, &train_idx, test_fold);
            let d_float: Vec<f64> = treatment.iter().map(|&d| d as f64).collect();
            let d_pred = fit_nuisance_model(&covariates, &d_float, &train_idx, test_fold);

            for &i in test_fold {
                y_residuals[i] = outcome[i] - y_pred[i];
                d_residuals[i] = treatment[i] as f64 - d_pred[i];
            }
        }

        let propensity = fit_propensity_model(
            &covariates,
            &treatment,
            &(0..n).collect::<Vec<_>>(),
            config.trimming_threshold,
        );

        let scores: Vec<f64> = (0..n)
            .map(|i| {
                let weight = if treatment[i] == 1 {
                    1.0 / propensity[i]
                } else {
                    -1.0 / (1.0 - propensity[i])
                };
                y_residuals[i] * weight
            })
            .collect();

        all_scores.extend(scores);
    }

    let ate = all_scores.iter().sum::<f64>() / all_scores.len() as f64;

    let var: f64 =
        all_scores.iter().map(|&s| (s - ate).powi(2)).sum::<f64>() / (all_scores.len() - 1) as f64;
    let se = (var / all_scores.len() as f64).sqrt();

    let z = ate / se.max(1e-10);
    let pvalue = 2.0 * (1.0 - normal_cdf(z.abs()));

    let ci_lower = ate - 1.96 * se;
    let ci_upper = ate + 1.96 * se;

    Ok(DoubleMLResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        pvalue,
        n_obs: n,
        scores: all_scores,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CATEResult {
    #[pyo3(get)]
    pub cate_estimates: Vec<f64>,
    #[pyo3(get)]
    pub cate_se: Vec<f64>,
    #[pyo3(get)]
    pub group_labels: Vec<String>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
}

#[pymethods]
impl CATEResult {
    fn __repr__(&self) -> String {
        format!("CATEResult(n_groups={})", self.group_labels.len())
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    treatment,
    outcome,
    time,
    event,
    group_variable,
    config=None
))]
pub fn double_ml_cate(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    outcome: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    group_variable: Vec<i32>,
    config: Option<DoubleMLConfig>,
) -> PyResult<CATEResult> {
    let config = config.unwrap_or_else(|| DoubleMLConfig::new(5, 1, None, 0.01, None).unwrap());

    let n = covariates.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Inputs must not be empty",
        ));
    }

    let mut unique_groups: Vec<i32> = group_variable.clone();
    unique_groups.sort();
    unique_groups.dedup();

    let mut cate_estimates = Vec::new();
    let mut cate_se = Vec::new();
    let mut group_labels = Vec::new();
    let mut group_sizes = Vec::new();

    for group in unique_groups {
        let indices: Vec<usize> = group_variable
            .iter()
            .enumerate()
            .filter(|(_, g)| **g == group)
            .map(|(i, _)| i)
            .collect();

        if indices.len() < 10 {
            continue;
        }

        let group_cov: Vec<Vec<f64>> = indices.iter().map(|&i| covariates[i].clone()).collect();
        let group_treat: Vec<i32> = indices.iter().map(|&i| treatment[i]).collect();
        let group_outcome: Vec<f64> = indices.iter().map(|&i| outcome[i]).collect();
        let group_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let group_event: Vec<i32> = indices.iter().map(|&i| event[i]).collect();

        if let Ok(result) = double_ml_survival(
            group_cov,
            group_treat,
            group_outcome,
            group_time,
            group_event,
            Some(config.clone()),
        ) {
            cate_estimates.push(result.ate);
            cate_se.push(result.se);
            group_labels.push(format!("group_{}", group));
            group_sizes.push(indices.len());
        }
    }

    Ok(CATEResult {
        cate_estimates,
        cate_se,
        group_labels,
        group_sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = DoubleMLConfig::new(1, 1, None, 0.01, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_folds() {
        let mut rng = fastrand::Rng::new();
        rng.seed(42);
        let folds = create_folds(100, 5, &mut rng);
        assert_eq!(folds.len(), 5);
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }
}
