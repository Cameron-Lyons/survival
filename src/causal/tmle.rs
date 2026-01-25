#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone)]
#[pyclass]
pub struct TMLEConfig {
    #[pyo3(get, set)]
    pub n_folds: usize,
    #[pyo3(get, set)]
    pub trimming: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl TMLEConfig {
    #[new]
    #[pyo3(signature = (
        n_folds=5,
        trimming=0.01,
        max_iter=100,
        tol=1e-6,
        seed=None
    ))]
    pub fn new(
        n_folds: usize,
        trimming: f64,
        max_iter: usize,
        tol: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }
        Ok(Self {
            n_folds,
            trimming,
            max_iter,
            tol,
            seed,
        })
    }
}

fn logistic_regression_predict(x: &[Vec<f64>], coeffs: &[f64], intercept: f64) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            let logit: f64 = xi
                .iter()
                .zip(coeffs.iter())
                .map(|(&xij, &c)| xij * c)
                .sum::<f64>()
                + intercept;
            1.0 / (1.0 + (-logit).exp())
        })
        .collect()
}

fn fit_logistic(x: &[Vec<f64>], y: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, f64) {
    let n = x.len();
    let n_features = x[0].len();

    let mut coeffs = vec![0.0; n_features];
    let mut intercept = 0.0;

    for _ in 0..max_iter {
        let preds = logistic_regression_predict(x, &coeffs, intercept);

        let mut grad = vec![0.0; n_features];
        let mut grad_intercept = 0.0;

        for i in 0..n {
            let error = y[i] - preds[i];
            grad_intercept += error;
            for j in 0..n_features {
                grad[j] += error * x[i][j];
            }
        }

        let learning_rate = 0.1 / n as f64;
        let mut max_change: f64 = 0.0;

        for j in 0..n_features {
            let change = learning_rate * grad[j];
            coeffs[j] += change;
            max_change = max_change.max(change.abs());
        }
        let intercept_change = learning_rate * grad_intercept;
        intercept += intercept_change;
        max_change = max_change.max(intercept_change.abs());

        if max_change < tol {
            break;
        }
    }

    (coeffs, intercept)
}

fn fit_linear(x: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let n_features = x[0].len();

    let mut x_mean = vec![0.0; n_features];
    let mut y_mean = 0.0;

    for i in 0..n {
        for j in 0..n_features {
            x_mean[j] += x[i][j];
        }
        y_mean += y[i];
    }

    for m in &mut x_mean {
        *m /= n as f64;
    }
    y_mean /= n as f64;

    let mut coeffs = vec![0.0; n_features];
    for j in 0..n_features {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..n {
            let xij_centered = x[i][j] - x_mean[j];
            num += xij_centered * (y[i] - y_mean);
            denom += xij_centered * xij_centered;
        }

        if denom.abs() > 1e-10 {
            coeffs[j] = num / denom;
        }
    }

    let intercept = y_mean
        - x_mean
            .iter()
            .zip(coeffs.iter())
            .map(|(&m, &c)| m * c)
            .sum::<f64>();

    (coeffs, intercept)
}

fn linear_predict(x: &[Vec<f64>], coeffs: &[f64], intercept: f64) -> Vec<f64> {
    x.iter()
        .map(|xi| {
            xi.iter()
                .zip(coeffs.iter())
                .map(|(&xij, &c)| xij * c)
                .sum::<f64>()
                + intercept
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TMLEResult {
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
    pub influence_function: Vec<f64>,
    #[pyo3(get)]
    pub n_obs: usize,
}

#[pymethods]
impl TMLEResult {
    fn __repr__(&self) -> String {
        format!(
            "TMLEResult(ATE={:.4}, SE={:.4}, p={:.4})",
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
    config=None
))]
pub fn tmle_ate(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    outcome: Vec<f64>,
    config: Option<TMLEConfig>,
) -> PyResult<TMLEResult> {
    let config = config.unwrap_or_else(|| TMLEConfig::new(5, 0.01, 100, 1e-6, None).unwrap());

    let n = covariates.len();
    if n == 0 || treatment.len() != n || outcome.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let treatment_float: Vec<f64> = treatment.iter().map(|&t| t as f64).collect();
    let (ps_coeffs, ps_intercept) =
        fit_logistic(&covariates, &treatment_float, config.max_iter, config.tol);
    let mut propensity = logistic_regression_predict(&covariates, &ps_coeffs, ps_intercept);

    for p in &mut propensity {
        *p = p.clamp(config.trimming, 1.0 - config.trimming);
    }

    let (q_coeffs, q_intercept) = fit_linear(&covariates, &outcome);
    let q_initial = linear_predict(&covariates, &q_coeffs, q_intercept);

    let mut q1 = q_initial.clone();
    let mut q0 = q_initial.clone();

    let h1: Vec<f64> = propensity.iter().map(|&p| 1.0 / p).collect();
    let h0: Vec<f64> = propensity.iter().map(|&p| -1.0 / (1.0 - p)).collect();

    let clever_covariate: Vec<f64> = treatment
        .iter()
        .zip(h1.iter().zip(h0.iter()))
        .map(|(&t, (&h1i, &h0i))| if t == 1 { h1i } else { h0i })
        .collect();

    let residuals: Vec<f64> = outcome
        .iter()
        .zip(q_initial.iter())
        .map(|(&y, &q)| y - q)
        .collect();

    let epsilon: f64 = {
        let num: f64 = clever_covariate
            .iter()
            .zip(residuals.iter())
            .map(|(&c, &r)| c * r)
            .sum();
        let denom: f64 = clever_covariate.iter().map(|&c| c * c).sum();
        if denom.abs() > 1e-10 {
            num / denom
        } else {
            0.0
        }
    };

    for i in 0..n {
        q1[i] = q_initial[i] + epsilon * h1[i];
        q0[i] = q_initial[i] + epsilon * h0[i];
    }

    let ate: f64 = (q1.iter().sum::<f64>() - q0.iter().sum::<f64>()) / n as f64;

    let influence_function: Vec<f64> = (0..n)
        .map(|i| {
            let d = if treatment[i] == 1 {
                (outcome[i] - q1[i]) / propensity[i]
            } else {
                -(outcome[i] - q0[i]) / (1.0 - propensity[i])
            };
            d + q1[i] - q0[i] - ate
        })
        .collect();

    let var: f64 = influence_function
        .iter()
        .map(|&if_i| if_i * if_i)
        .sum::<f64>()
        / (n * (n - 1)) as f64;
    let se = var.sqrt();

    let z = ate / se.max(1e-10);
    let pvalue = 2.0 * (1.0 - normal_cdf(z.abs()));

    let ci_lower = ate - 1.96 * se;
    let ci_upper = ate + 1.96 * se;

    Ok(TMLEResult {
        ate,
        se,
        ci_lower,
        ci_upper,
        pvalue,
        influence_function,
        n_obs: n,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TMLESurvivalResult {
    #[pyo3(get)]
    pub survival_diff: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub se: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub rmst_diff: f64,
    #[pyo3(get)]
    pub rmst_se: f64,
}

#[pymethods]
impl TMLESurvivalResult {
    fn __repr__(&self) -> String {
        format!(
            "TMLESurvivalResult(n_times={}, rmst_diff={:.4})",
            self.time_points.len(),
            self.rmst_diff
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    treatment,
    time,
    event,
    time_points=None,
    config=None
))]
pub fn tmle_survival(
    covariates: Vec<Vec<f64>>,
    treatment: Vec<i32>,
    time: Vec<f64>,
    event: Vec<i32>,
    time_points: Option<Vec<f64>>,
    config: Option<TMLEConfig>,
) -> PyResult<TMLESurvivalResult> {
    let config = config.unwrap_or_else(|| TMLEConfig::new(5, 0.01, 100, 1e-6, None).unwrap());

    let n = covariates.len();
    if n == 0 || treatment.len() != n || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_points =
        time_points.unwrap_or_else(|| (1..=10).map(|i| max_time * i as f64 / 10.0).collect());

    let treatment_float: Vec<f64> = treatment.iter().map(|&t| t as f64).collect();
    let (ps_coeffs, ps_intercept) =
        fit_logistic(&covariates, &treatment_float, config.max_iter, config.tol);
    let mut propensity = logistic_regression_predict(&covariates, &ps_coeffs, ps_intercept);

    for p in &mut propensity {
        *p = p.clamp(config.trimming, 1.0 - config.trimming);
    }

    let mut survival_diff = Vec::new();
    let mut se_vec = Vec::new();
    let mut ci_lower = Vec::new();
    let mut ci_upper = Vec::new();

    for &t in &time_points {
        let outcomes: Vec<f64> = time
            .iter()
            .zip(event.iter())
            .map(|(&ti, &ei)| {
                if ti > t {
                    1.0
                } else if ei == 1 {
                    0.0
                } else {
                    0.5
                }
            })
            .collect();

        let (q_coeffs, q_intercept) = fit_linear(&covariates, &outcomes);
        let q_initial = linear_predict(&covariates, &q_coeffs, q_intercept);

        let h1: Vec<f64> = propensity.iter().map(|&p| 1.0 / p).collect();
        let h0: Vec<f64> = propensity.iter().map(|&p| -1.0 / (1.0 - p)).collect();

        let clever_covariate: Vec<f64> = treatment
            .iter()
            .zip(h1.iter().zip(h0.iter()))
            .map(|(&tr, (&h1i, &h0i))| if tr == 1 { h1i } else { h0i })
            .collect();

        let residuals: Vec<f64> = outcomes
            .iter()
            .zip(q_initial.iter())
            .map(|(&y, &q)| y - q)
            .collect();

        let epsilon: f64 = {
            let num: f64 = clever_covariate
                .iter()
                .zip(residuals.iter())
                .map(|(&c, &r)| c * r)
                .sum();
            let denom: f64 = clever_covariate.iter().map(|&c| c * c).sum();
            if denom.abs() > 1e-10 {
                num / denom
            } else {
                0.0
            }
        };

        let q1: Vec<f64> = (0..n).map(|i| q_initial[i] + epsilon * h1[i]).collect();
        let q0: Vec<f64> = (0..n).map(|i| q_initial[i] + epsilon * h0[i]).collect();

        let s1 = q1.iter().sum::<f64>() / n as f64;
        let s0 = q0.iter().sum::<f64>() / n as f64;
        let diff = s1 - s0;

        let influence: Vec<f64> = (0..n)
            .map(|i| {
                let d = if treatment[i] == 1 {
                    (outcomes[i] - q1[i]) / propensity[i]
                } else {
                    -(outcomes[i] - q0[i]) / (1.0 - propensity[i])
                };
                d + q1[i] - q0[i] - diff
            })
            .collect();

        let var: f64 =
            influence.iter().map(|&if_i| if_i * if_i).sum::<f64>() / (n * (n - 1)) as f64;
        let se = var.sqrt();

        survival_diff.push(diff);
        se_vec.push(se);
        ci_lower.push(diff - 1.96 * se);
        ci_upper.push(diff + 1.96 * se);
    }

    let mut rmst_diff = 0.0;
    for i in 1..time_points.len() {
        let dt = time_points[i] - time_points[i - 1];
        let avg_diff = (survival_diff[i] + survival_diff[i - 1]) / 2.0;
        rmst_diff += dt * avg_diff;
    }

    let rmst_se = se_vec.iter().map(|&s| s * s).sum::<f64>().sqrt() / time_points.len() as f64;

    Ok(TMLESurvivalResult {
        survival_diff,
        time_points,
        se: se_vec,
        ci_lower,
        ci_upper,
        rmst_diff,
        rmst_se,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = TMLEConfig::new(1, 0.01, 100, 1e-6, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_tmle_ate() {
        let covariates = vec![
            vec![1.0, 2.0],
            vec![1.5, 2.5],
            vec![2.0, 3.0],
            vec![2.5, 3.5],
            vec![3.0, 4.0],
        ];
        let treatment = vec![0, 1, 0, 1, 1];
        let outcome = vec![1.0, 2.0, 1.5, 2.5, 3.0];

        let result = tmle_ate(covariates, treatment, outcome, None).unwrap();
        assert!(result.se > 0.0);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
    }
}
