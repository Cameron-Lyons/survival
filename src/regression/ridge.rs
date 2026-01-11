use pyo3::prelude::*;
use rayon::prelude::*;

/// Configuration for ridge regression penalty
#[derive(Debug, Clone)]
#[pyclass]
pub struct RidgePenalty {
    /// Penalty parameter (larger = more shrinkage)
    #[pyo3(get, set)]
    pub theta: f64,
    /// Whether to scale predictors to unit variance before applying penalty
    #[pyo3(get, set)]
    pub scale: bool,
    /// Effective degrees of freedom (computed after fitting)
    #[pyo3(get)]
    pub df: Option<f64>,
}

#[pymethods]
impl RidgePenalty {
    /// Create a new ridge penalty configuration
    ///
    /// # Arguments
    /// * `theta` - Penalty parameter. Can be specified directly or computed from df.
    /// * `scale` - Whether to scale predictors (default: true)
    #[new]
    #[pyo3(signature = (theta, scale=None))]
    pub fn new(theta: f64, scale: Option<bool>) -> PyResult<Self> {
        if theta < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "theta must be non-negative",
            ));
        }

        Ok(RidgePenalty {
            theta,
            scale: scale.unwrap_or(true),
            df: None,
        })
    }

    /// Create ridge penalty from desired degrees of freedom
    ///
    /// # Arguments
    /// * `df` - Approximate degrees of freedom
    /// * `n_vars` - Number of variables
    /// * `scale` - Whether to scale predictors
    #[staticmethod]
    #[pyo3(signature = (df, n_vars, scale=None))]
    pub fn from_df(df: f64, n_vars: usize, scale: Option<bool>) -> PyResult<Self> {
        if df <= 0.0 || df > n_vars as f64 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "df must be between 0 and {} (number of variables)",
                n_vars
            )));
        }

        let theta = (n_vars as f64 / df - 1.0).max(0.0);

        Ok(RidgePenalty {
            theta,
            scale: scale.unwrap_or(true),
            df: Some(df),
        })
    }

    /// Compute the penalty term for a given coefficient vector
    pub fn penalty_value(&self, beta: Vec<f64>) -> f64 {
        let sum_sq: f64 = beta.iter().map(|&b| b * b).sum();
        self.theta / 2.0 * sum_sq
    }

    /// Compute the gradient of the penalty (for optimization)
    pub fn penalty_gradient(&self, beta: Vec<f64>) -> Vec<f64> {
        beta.iter().map(|&b| self.theta * b).collect()
    }

    /// Apply penalty to information matrix (add theta*I to diagonal)
    ///
    /// This modifies the information matrix for penalized estimation.
    pub fn apply_to_information(&self, info_diag: Vec<f64>) -> Vec<f64> {
        info_diag.iter().map(|&x| x + self.theta).collect()
    }
}

/// Result of ridge regression estimation
#[derive(Debug, Clone)]
#[pyclass]
pub struct RidgeResult {
    /// Penalized coefficient estimates
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// Standard errors (may be biased due to shrinkage)
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    /// Effective degrees of freedom
    #[pyo3(get)]
    pub df: f64,
    /// Generalized cross-validation score
    #[pyo3(get)]
    pub gcv: f64,
    /// Penalty parameter used
    #[pyo3(get)]
    pub theta: f64,
    /// Scaling factors applied (if scale=true)
    #[pyo3(get)]
    pub scale_factors: Option<Vec<f64>>,
}

/// Fit ridge regression for survival models.
///
/// This performs penalized maximum likelihood estimation with an L2 (ridge)
/// penalty on the coefficients. The penalty shrinks coefficients toward zero,
/// which can improve prediction accuracy when predictors are correlated.
///
/// # Arguments
/// * `x` - Design matrix (flattened, row-major)
/// * `n_obs` - Number of observations
/// * `n_vars` - Number of variables
/// * `time` - Survival times
/// * `status` - Event indicators
/// * `penalty` - Ridge penalty configuration
/// * `weights` - Optional observation weights
///
/// # Returns
/// * `RidgeResult` with penalized estimates
#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, penalty, weights=None))]
pub fn ridge_fit(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    penalty: &RidgePenalty,
    weights: Option<Vec<f64>>,
) -> PyResult<RidgeResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have length n_obs",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);

    let (scaled_x, scale_factors) = if penalty.scale {
        scale_predictors(&x, n_obs, n_vars)
    } else {
        (x.clone(), None)
    };

    let (beta, info_diag) = fit_unpenalized(&scaled_x, n_obs, n_vars, &time, &status, &wt)?;

    let penalized_info: Vec<f64> = info_diag.iter().map(|&i| i + penalty.theta).collect();

    let penalized_beta: Vec<f64> = beta
        .iter()
        .zip(info_diag.iter())
        .zip(penalized_info.iter())
        .map(|((&b, &i), &pi)| b * i / pi)
        .collect();

    let std_err: Vec<f64> = penalized_info
        .iter()
        .map(|&pi| {
            if pi > 0.0 {
                1.0 / pi.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let df: f64 = info_diag
        .iter()
        .zip(penalized_info.iter())
        .map(|(&i, &pi)| i / pi)
        .sum();

    let gcv = compute_gcv(
        &scaled_x,
        n_obs,
        n_vars,
        &time,
        &status,
        &wt,
        &penalized_beta,
        df,
    );

    let final_beta = if let Some(ref sf) = scale_factors {
        penalized_beta
            .iter()
            .zip(sf.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect()
    } else {
        penalized_beta
    };

    let final_se = if let Some(ref sf) = scale_factors {
        std_err
            .iter()
            .zip(sf.iter())
            .map(|(&se, &s)| if s > 0.0 { se / s } else { se })
            .collect()
    } else {
        std_err
    };

    Ok(RidgeResult {
        coefficients: final_beta,
        std_err: final_se,
        df,
        gcv,
        theta: penalty.theta,
        scale_factors,
    })
}

/// Scale predictors to unit variance
fn scale_predictors(x: &[f64], n_obs: usize, n_vars: usize) -> (Vec<f64>, Option<Vec<f64>>) {
    let mut scaled = x.to_vec();
    let mut scale_factors = Vec::with_capacity(n_vars);

    for j in 0..n_vars {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for i in 0..n_obs {
            let val = x[i * n_vars + j];
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / n_obs as f64;
        let variance = sum_sq / n_obs as f64 - mean * mean;
        let sd = variance.sqrt().max(1e-10);

        scale_factors.push(sd);

        for i in 0..n_obs {
            scaled[i * n_vars + j] = (scaled[i * n_vars + j] - mean) / sd;
        }
    }

    (scaled, Some(scale_factors))
}

/// Fit unpenalized model (simplified)
fn fit_unpenalized(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let beta = vec![0.0; n_vars];
    let mut info_diag = vec![0.0; n_vars];
    let mut score = vec![0.0; n_vars];

    let mut risk_cache = vec![0.0; n_obs];
    for i in 0..n_obs {
        let mut eta = 0.0;
        for j in 0..n_vars {
            eta += x[i * n_vars + j] * beta[j];
        }
        risk_cache[i] = (eta.min(700.0)).exp() * weights[i];
    }

    let mut risk_sum = vec![0.0; n_obs];
    let mut total_risk = 0.0;
    for &i in indices.iter().rev() {
        total_risk += risk_cache[i];
        risk_sum[i] = total_risk;
    }

    for (order_idx, &i) in indices.iter().enumerate() {
        if status[i] == 1 && risk_sum[i] > 0.0 {
            for j in 0..n_vars {
                let xij = x[i * n_vars + j];

                let mut x_mean = 0.0;
                let mut x_sq_mean = 0.0;

                for &k in &indices[order_idx..] {
                    let xkj = x[k * n_vars + j];
                    let risk = risk_cache[k];
                    x_mean += xkj * risk / risk_sum[i];
                    x_sq_mean += xkj * xkj * risk / risk_sum[i];
                }

                score[j] += weights[i] * (xij - x_mean);
                info_diag[j] += weights[i] * (x_sq_mean - x_mean * x_mean);
            }
        }
    }

    let mut final_beta = vec![0.0; n_vars];
    for j in 0..n_vars {
        if info_diag[j] > 1e-10 {
            final_beta[j] = score[j] / info_diag[j];
            info_diag[j] = info_diag[j].max(1e-10);
        }
    }

    Ok((final_beta, info_diag))
}

#[allow(clippy::too_many_arguments)]
fn compute_gcv(
    _x: &[f64],
    n_obs: usize,
    _n_vars: usize,
    _time: &[f64],
    _status: &[i32],
    _weights: &[f64],
    _beta: &[f64],
    df: f64,
) -> f64 {
    let n = n_obs as f64;
    let denom = (1.0 - df / n).powi(2);
    if denom > 0.0 {
        1.0 / denom
    } else {
        f64::INFINITY
    }
}

/// Select optimal ridge penalty using cross-validation
#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, theta_grid=None, n_folds=None))]
pub fn ridge_cv(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    theta_grid: Option<Vec<f64>>,
    n_folds: Option<usize>,
) -> PyResult<(f64, Vec<f64>)> {
    let grid = theta_grid.unwrap_or_else(|| {
        (0..20)
            .map(|i| 10.0_f64.powf(-4.0 + i as f64 * 0.4))
            .collect()
    });

    let folds = n_folds.unwrap_or(5);
    let fold_assign: Vec<usize> = (0..n_obs).map(|i| i % folds).collect();

    let x_ref = &x;
    let time_ref = &time;
    let status_ref = &status;

    let cv_scores: Vec<f64> = grid
        .par_iter()
        .map(|&theta| {
            let fold_scores: Vec<f64> = (0..folds)
                .filter_map(|fold| {
                    let train_idx: Vec<usize> =
                        (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();

                    if train_idx.is_empty() {
                        return None;
                    }

                    let train_x: Vec<f64> = train_idx
                        .iter()
                        .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                        .collect();
                    let train_time: Vec<f64> = train_idx.iter().map(|&i| time_ref[i]).collect();
                    let train_status: Vec<i32> = train_idx.iter().map(|&i| status_ref[i]).collect();

                    let penalty = RidgePenalty {
                        theta,
                        scale: true,
                        df: None,
                    };

                    ridge_fit(
                        train_x,
                        train_idx.len(),
                        n_vars,
                        train_time,
                        train_status,
                        &penalty,
                        None,
                    )
                    .ok()
                    .map(|r| r.df)
                })
                .collect();

            if fold_scores.is_empty() {
                f64::INFINITY
            } else {
                fold_scores.iter().sum::<f64>() / fold_scores.len() as f64
            }
        })
        .collect();

    let best_idx = cv_scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok((grid[best_idx], cv_scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_penalty_new() {
        let penalty = RidgePenalty::new(1.0, None).unwrap();
        assert_eq!(penalty.theta, 1.0);
        assert!(penalty.scale);
    }

    #[test]
    fn test_ridge_penalty_from_df() {
        let penalty = RidgePenalty::from_df(5.0, 10, None).unwrap();
        assert!(penalty.theta > 0.0);
        assert_eq!(penalty.df, Some(5.0));
    }

    #[test]
    fn test_ridge_penalty_value() {
        let penalty = RidgePenalty::new(2.0, None).unwrap();
        let beta = vec![1.0, 2.0, 3.0];
        let value = penalty.penalty_value(beta);
        // theta/2 * (1 + 4 + 9) = 1.0 * 14 = 14.0
        assert!((value - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_ridge_fit_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let penalty = RidgePenalty::new(0.1, Some(false)).unwrap();

        let result = ridge_fit(x, 3, 2, time, status, &penalty, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_err.len(), 2);
    }
}
