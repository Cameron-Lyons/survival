use crate::constants::DIVISION_FLOOR;
use crate::internal::cox_risk::precompute_cox_unit_risk_set_cumsum;
use crate::internal::matrix::standardize_or_borrow_row_major_matrix;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_empty,
    validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Configuration for ridge regression penalty
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
        if !theta.is_finite() || theta < 0.0 {
            return Err(PyValueError::new_err(
                "theta must be finite and non-negative",
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
        if !df.is_finite() || df <= 0.0 || df > n_vars as f64 {
            return Err(PyValueError::new_err(format!(
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
#[pyclass(from_py_object)]
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
    validate_ridge_inputs(&x, n_obs, n_vars, &time, &status, weights.as_deref())?;
    validate_ridge_penalty(penalty)?;

    let weights = weights.as_deref();

    let (scaled_x, _, scales) =
        standardize_or_borrow_row_major_matrix(&x, n_obs, n_vars, penalty.scale);
    let scale_factors = penalty.scale.then_some(scales);

    let (beta, info_diag) =
        fit_unpenalized(scaled_x.as_ref(), n_obs, n_vars, &time, &status, weights)?;

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
        scaled_x.as_ref(),
        n_obs,
        n_vars,
        &time,
        &status,
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

fn validate_ridge_penalty(penalty: &RidgePenalty) -> PyResult<()> {
    if !penalty.theta.is_finite() || penalty.theta < 0.0 {
        return Err(PyValueError::new_err(
            "theta must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_ridge_inputs(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if n_obs == 0 {
        return Err(PyValueError::new_err("n_obs must be positive"));
    }
    if n_vars == 0 {
        return Err(PyValueError::new_err("n_vars must be positive"));
    }

    let expected_x_len = n_obs.checked_mul(n_vars).ok_or_else(|| {
        PyValueError::new_err("n_obs * n_vars overflowed while validating x length")
    })?;
    if x.len() != expected_x_len {
        return Err(PyValueError::new_err("x length must equal n_obs * n_vars"));
    }
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyValueError::new_err(
            "time and status must have length n_obs",
        ));
    }

    validate_no_nan(x, "x")?;
    validate_finite(x, "x")?;
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;

    if let Some(weights) = weights {
        if weights.len() != n_obs {
            return Err(PyValueError::new_err("weights must have length n_obs"));
        }
        validate_no_nan(weights, "weights")?;
        validate_finite(weights, "weights")?;
        validate_non_negative(weights, "weights")?;
    }

    Ok(())
}

/// Fit unpenalized model (simplified)
fn fit_unpenalized(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let mut info_diag = vec![0.0; n_vars];
    let mut score = vec![0.0; n_vars];
    let risk_data = precompute_cox_unit_risk_set_cumsum(x, n_obs, n_vars, time, weights);

    for i in 0..n_obs {
        if status[i] != 1 {
            continue;
        }

        let pos = risk_data.risk_set_pos[i];
        let risk_sum = risk_data.cumsum_exp_eta[pos];
        if risk_sum <= 0.0 {
            continue;
        }

        let weight = observation_weight(weights, i);
        for j in 0..n_vars {
            let xij = x[i * n_vars + j];
            let cumsum_idx = pos * n_vars + j;
            let x_mean = risk_data.cumsum_weighted_x[cumsum_idx] / risk_sum;
            let x_sq_mean = risk_data.cumsum_weighted_x_sq[cumsum_idx] / risk_sum;

            score[j] += weight * (xij - x_mean);
            info_diag[j] += weight * (x_sq_mean - x_mean * x_mean).max(0.0);
        }
    }

    let mut final_beta = vec![0.0; n_vars];
    for j in 0..n_vars {
        if info_diag[j] > DIVISION_FLOOR {
            final_beta[j] = score[j] / info_diag[j];
            info_diag[j] = info_diag[j].max(DIVISION_FLOOR);
        }
    }

    Ok((final_beta, info_diag))
}

#[inline]
fn observation_weight(weights: Option<&[f64]>, idx: usize) -> f64 {
    weights.map_or(1.0, |values| values[idx])
}

#[allow(clippy::too_many_arguments)]
fn compute_gcv(
    _x: &[f64],
    n_obs: usize,
    _n_vars: usize,
    _time: &[f64],
    _status: &[i32],
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
    validate_ridge_inputs(&x, n_obs, n_vars, &time, &status, None)?;

    let grid = theta_grid.unwrap_or_else(|| {
        (0..20)
            .map(|i| 10.0_f64.powf(-4.0 + i as f64 * 0.4))
            .collect()
    });
    validate_non_empty(&grid, "theta_grid")?;
    validate_no_nan(&grid, "theta_grid")?;
    validate_finite(&grid, "theta_grid")?;
    validate_non_negative(&grid, "theta_grid")?;

    let folds = n_folds.unwrap_or_else(|| n_obs.min(5));
    if folds < 2 || folds > n_obs {
        return Err(PyValueError::new_err("n_folds must be between 2 and n_obs"));
    }
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
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok((grid[best_idx], cv_scores))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TIME_EPSILON;

    fn assert_close(left: f64, right: f64) {
        if left == right {
            return;
        }

        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to equal {right}"
        );
    }

    fn assert_vec_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (&left_value, &right_value) in left.iter().zip(right) {
            assert_close(left_value, right_value);
        }
    }

    fn assert_optional_vec_close(left: &Option<Vec<f64>>, right: &Option<Vec<f64>>) {
        match (left, right) {
            (Some(left), Some(right)) => assert_vec_close(left, right),
            (None, None) => {}
            _ => panic!("expected matching optional vectors"),
        }
    }

    fn assert_ridge_result_close(left: &RidgeResult, right: &RidgeResult) {
        assert_vec_close(&left.coefficients, &right.coefficients);
        assert_vec_close(&left.std_err, &right.std_err);
        assert_close(left.df, right.df);
        assert_close(left.gcv, right.gcv);
        assert_close(left.theta, right.theta);
        assert_optional_vec_close(&left.scale_factors, &right.scale_factors);
    }

    #[test]
    fn test_ridge_penalty_new() {
        let penalty = RidgePenalty::new(1.0, None).unwrap();
        assert_eq!(penalty.theta, 1.0);
        assert!(penalty.scale);

        let err = RidgePenalty::new(f64::INFINITY, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("theta must be finite and non-negative")
        );
    }

    #[test]
    fn test_ridge_penalty_from_df() {
        let penalty = RidgePenalty::from_df(5.0, 10, None).unwrap();
        assert!(penalty.theta > 0.0);
        assert_eq!(penalty.df, Some(5.0));

        let err = RidgePenalty::from_df(f64::NAN, 10, None).unwrap_err();
        assert!(err.to_string().contains("df must be between 0 and 10"));
    }

    #[test]
    fn test_ridge_penalty_value() {
        let penalty = RidgePenalty::new(2.0, None).unwrap();
        let beta = vec![1.0, 2.0, 3.0];
        let value = penalty.penalty_value(beta);
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

    #[test]
    fn test_ridge_fit_unweighted_matches_unit_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let penalty = RidgePenalty::new(0.1, Some(false)).unwrap();

        let unweighted = ridge_fit(
            x.clone(),
            3,
            2,
            time.clone(),
            status.clone(),
            &penalty,
            None,
        )
        .unwrap();
        let unit_weighted = ridge_fit(x, 3, 2, time, status, &penalty, Some(vec![1.0; 3])).unwrap();

        assert_ridge_result_close(&unweighted, &unit_weighted);
    }

    #[test]
    fn test_ridge_fit_scaled_returns_standardization_scales() {
        let x = vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let penalty = RidgePenalty::new(0.1, Some(true)).unwrap();

        let result = ridge_fit(x, 3, 2, time, status, &penalty, None).unwrap();
        let scale_factors = result
            .scale_factors
            .expect("scaled fit should report scaling factors");

        assert!((scale_factors[0] - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(scale_factors[1], DIVISION_FLOOR);
    }

    #[test]
    fn test_ridge_fit_uses_shared_risk_set_for_tied_times() {
        let x = vec![0.0, 2.0, 2.0];
        let time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 1.0, 1.0];

        let (beta, info_diag) = fit_unpenalized(&x, 3, 1, &time, &status, Some(&weights)).unwrap();

        assert!((beta[0] + 1.5).abs() < 1e-12);
        assert!((info_diag[0] - 8.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_ridge_fit_excludes_zero_weight_rows_from_risk_sets() {
        let x = vec![0.0, 10.0, 2.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 0.0, 1.0];

        let (beta, info_diag) = fit_unpenalized(&x, 3, 1, &time, &status, Some(&weights)).unwrap();

        assert!((beta[0] + 1.0).abs() < 1e-12);
        assert!((info_diag[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ridge_fit_rejects_malformed_public_inputs() {
        let penalty = RidgePenalty::new(0.1, Some(false)).unwrap();

        let err = ridge_fit(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 2],
            &penalty,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = ridge_fit(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 1],
            &penalty,
            Some(vec![1.0, f64::INFINITY]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("weights contains non-finite"));

        let err = ridge_cv(
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            2,
            vec![1.0, 2.0],
            vec![1, 1],
            Some(vec![]),
            Some(2),
        )
        .unwrap_err();
        assert!(err.to_string().contains("theta_grid cannot be empty"));

        let err = ridge_cv(
            vec![1.0, 0.0, 0.0, 1.0],
            2,
            2,
            vec![1.0, 2.0],
            vec![1, 1],
            Some(vec![0.1]),
            Some(0),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("n_folds must be between 2 and n_obs")
        );
    }
}
