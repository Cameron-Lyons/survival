use super::coxph::{CoxPHFit, coxph_fit, coxph_penalized_fit};
use super::coxph_penalty::CoxPenaltyDiagnostics;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_empty,
    validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RidgePenalty {
    /// Fixed theta, or the initial controller value for a df-selected penalty.
    #[pyo3(get, set)]
    pub theta: f64,
    #[pyo3(get, set)]
    pub scale: bool,
    #[pyo3(get)]
    pub df: Option<f64>,
}

#[pymethods]
impl RidgePenalty {
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

    #[staticmethod]
    #[pyo3(signature = (df, n_vars, scale=None))]
    pub fn from_df(df: f64, n_vars: usize, scale: Option<bool>) -> PyResult<Self> {
        if n_vars == 0 || !df.is_finite() || df < 0.0 || df > n_vars as f64 {
            return Err(PyValueError::new_err(format!(
                "df must be between 0 and {} (number of variables)",
                n_vars
            )));
        }

        Ok(RidgePenalty {
            // This is the initial search value, not a data-independent solution.
            theta: 1.0,
            scale: scale.unwrap_or(true),
            df: Some(df),
        })
    }

    pub fn penalty_value(&self, beta: Vec<f64>) -> f64 {
        let sum_sq: f64 = beta.iter().map(|&b| b * b).sum();
        self.theta / 2.0 * sum_sq
    }

    pub fn penalty_gradient(&self, beta: Vec<f64>) -> Vec<f64> {
        beta.iter().map(|&b| self.theta * b).collect()
    }

    pub fn apply_to_information(&self, info_diag: Vec<f64>) -> Vec<f64> {
        info_diag.iter().map(|&x| x + self.theta).collect()
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RidgeResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    #[pyo3(get)]
    pub df: f64,
    #[pyo3(get)]
    pub gcv: f64,
    #[pyo3(get)]
    pub theta: f64,
    /// Unweighted sample standard deviations used in the penalty, if scaled.
    #[pyo3(get)]
    pub scale_factors: Option<Vec<f64>>,
}

/// Fit a joint Efron Cox model with a diagonal ridge penalty.
///
/// Coefficients and standard errors are in the original covariate units. The
/// GCV score is `(-2 * log_likelihood / n_obs) / (1 - df / n_obs)^2` and is
/// infinite when the effective degrees of freedom reach the observation count.
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
    validate_ridge_penalty(penalty, n_vars)?;

    let (fit, diagnostics, scale_factors, theta) =
        fit_ridge_model(&x, n_obs, n_vars, time, status, penalty, weights)?;
    let df = diagnostics.term_df[0];
    let gcv = partial_likelihood_gcv(fit.log_likelihood[1], n_obs, df);
    let std_err = fit
        .information_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| row[i].max(0.0).sqrt())
        .collect();

    Ok(RidgeResult {
        coefficients: fit.coefficients[0].clone(),
        std_err,
        df,
        gcv,
        theta,
        scale_factors,
    })
}

type RidgeFitDiagnostics = (CoxPHFit, CoxPenaltyDiagnostics, Option<Vec<f64>>, f64);

fn ridge_scale_variances(x: &[f64], n_obs: usize, n_vars: usize) -> Vec<f64> {
    // A fixed origin prevents a large common offset from corrupting the
    // sample mean, and keeps truly constant columns exactly unpenalized.
    let origins = &x[..n_vars];
    let mut means = vec![0.0; n_vars];
    for row in x.chunks_exact(n_vars) {
        for ((mean, &value), &origin) in means.iter_mut().zip(row).zip(origins) {
            *mean += (value - origin) / n_obs as f64;
        }
    }
    let mut variances = vec![0.0; n_vars];
    if n_obs > 1 {
        for row in x.chunks_exact(n_vars) {
            for (j, variance) in variances.iter_mut().enumerate() {
                *variance += ((row[j] - origins[j]) - means[j]).powi(2) / (n_obs - 1) as f64;
            }
        }
    }
    variances
}

#[allow(clippy::too_many_arguments)]
fn fit_ridge_model(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    penalty: &RidgePenalty,
    weights: Option<Vec<f64>>,
) -> PyResult<RidgeFitDiagnostics> {
    let variances = penalty
        .scale
        .then(|| ridge_scale_variances(x, n_obs, n_vars));
    let covariates = x.chunks_exact(n_vars).map(|row| row.to_vec()).collect();
    let scales = variances.clone().unwrap_or_else(|| vec![1.0; n_vars]);
    let groups = vec![(0..n_vars).collect()];
    let (fit, diagnostics, theta) = if let Some(target_df) = penalty.df {
        let (fit, diagnostics, selection) = super::coxph_ridge::coxph_ridge_fit(
            time,
            status,
            covariates,
            scales,
            groups,
            vec![None],
            vec![Some(target_df)],
            vec![0.1],
            None,
            weights,
            None,
            None,
            Some(50),
            None,
            None,
            Some("efron"),
            None,
            None,
            None,
        )?;
        (fit, diagnostics, selection.fitted_theta[0])
    } else {
        let diagonal = scales
            .into_iter()
            .map(|scale| penalty.theta * scale)
            .collect();
        let (fit, diagnostics) = coxph_penalized_fit(
            time,
            status,
            covariates,
            diagonal,
            groups,
            None,
            weights,
            None,
            None,
            Some(50),
            None,
            None,
            Some("efron"),
            None,
            None,
        )?;
        (fit, diagnostics, penalty.theta)
    };
    let scale_factors = variances.map(|values| values.into_iter().map(f64::sqrt).collect());
    Ok((fit, diagnostics, scale_factors, theta))
}

fn partial_likelihood_gcv(log_likelihood: f64, n_obs: usize, df: f64) -> f64 {
    let n = n_obs as f64;
    if df >= n {
        f64::INFINITY
    } else {
        (-2.0 * log_likelihood / n) / (1.0 - df / n).powi(2)
    }
}

fn validate_ridge_penalty(penalty: &RidgePenalty, n_vars: usize) -> PyResult<()> {
    if !penalty.theta.is_finite() || penalty.theta < 0.0 {
        return Err(PyValueError::new_err(
            "theta must be finite and non-negative",
        ));
    }
    if let Some(df) = penalty.df
        && (!df.is_finite() || df < 0.0 || df > n_vars as f64)
    {
        return Err(PyValueError::new_err(format!(
            "df must be between 0 and {n_vars} (number of variables)"
        )));
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
        if weights.iter().all(|&weight| weight == 0.0) {
            return Err(PyValueError::new_err(
                "weights must include at least one positive value",
            ));
        }
    }

    Ok(())
}

/// Select theta by deterministic cross-validated Cox partial likelihood.
///
/// Scores are `-2 / n_obs` times the sum over folds of the full-data minus
/// training-data log likelihood, evaluated at that fold's training coefficients.
/// Holding out a row removes it from both event contributions and training risk sets.
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
    struct Fold {
        x: Vec<f64>,
        time: Vec<f64>,
        status: Vec<i32>,
    }
    let training: Vec<Fold> = (0..folds)
        .map(|fold| {
            let indices: Vec<usize> = (0..n_obs).filter(|&i| i % folds != fold).collect();
            Fold {
                x: indices
                    .iter()
                    .flat_map(|&i| x[i * n_vars..(i + 1) * n_vars].iter().copied())
                    .collect(),
                time: indices.iter().map(|&i| time[i]).collect(),
                status: indices.iter().map(|&i| status[i]).collect(),
            }
        })
        .collect();
    let full_covariates: Vec<Vec<f64>> = x.chunks_exact(n_vars).map(|row| row.to_vec()).collect();
    let cv_scores: PyResult<Vec<f64>> = grid
        .par_iter()
        .map(|&theta| {
            let penalty = RidgePenalty {
                theta,
                scale: true,
                df: None,
            };
            let mut score = 0.0;
            for fold in &training {
                let (fitted, _, _, _) = fit_ridge_model(
                    &fold.x,
                    fold.time.len(),
                    n_vars,
                    fold.time.clone(),
                    fold.status.clone(),
                    &penalty,
                    None,
                )?;
                let evaluated = coxph_fit(
                    time.clone(),
                    status.clone(),
                    full_covariates.clone(),
                    None,
                    None,
                    None,
                    Some(fitted.coefficients[0].clone()),
                    Some(0),
                    None,
                    None,
                    Some("efron"),
                    None,
                    None,
                )?;
                score -= 2.0 * (evaluated.log_likelihood[1] - fitted.log_likelihood[1]);
            }
            Ok(score / n_obs as f64)
        })
        .collect();
    let cv_scores = cv_scores?;

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

        assert_eq!(scale_factors[0], 1.0);
        assert_eq!(scale_factors[1], 0.0);
    }

    #[test]
    fn test_ridge_fit_uses_shared_risk_set_for_tied_times() {
        let x = vec![0.0, 2.0, 2.0];
        let time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 1.0, 1.0];

        let penalty = RidgePenalty::new(0.5, Some(false)).unwrap();
        let fitted = ridge_fit(
            x.clone(),
            3,
            1,
            time,
            status.clone(),
            &penalty,
            Some(weights.clone()),
        )
        .unwrap();
        let exactly_tied = ridge_fit(
            x,
            3,
            1,
            vec![1.0, 1.0, 2.0],
            status,
            &penalty,
            Some(weights),
        )
        .unwrap();
        assert_ridge_result_close(&fitted, &exactly_tied);
    }

    #[test]
    fn test_ridge_fit_excludes_zero_weight_rows_from_risk_sets() {
        let x = vec![0.0, 10.0, 2.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 0.0, 1.0];

        let penalty = RidgePenalty::new(0.5, Some(false)).unwrap();
        let fitted = ridge_fit(x, 3, 1, time, status, &penalty, Some(weights)).unwrap();
        let removed = ridge_fit(
            vec![0.0, 2.0],
            2,
            1,
            vec![1.0, 3.0],
            vec![1, 0],
            &penalty,
            None,
        )
        .unwrap();
        assert_vec_close(&fitted.coefficients, &removed.coefficients);
        assert_vec_close(&fitted.std_err, &removed.std_err);
        assert_close(fitted.df, removed.df);
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

        let err = ridge_fit(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 1],
            &penalty,
            Some(vec![0.0, 0.0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least one positive"));

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

    #[test]
    fn standalone_ridge_matches_r_fixtures_in_original_units() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/cox_ridge_df_r_reference.json"
        ))
        .unwrap();
        let mut checked = 0;
        for (name, case) in reference["cases"].as_object().unwrap() {
            let theta = case["standalone_theta"].as_f64();
            let target = case["standalone_df"].as_f64();
            if theta.is_none() && target.is_none() {
                continue;
            }
            let rows: Vec<Vec<f64>> = serde_json::from_value(case["model_matrix"].clone()).unwrap();
            let n_obs = rows.len();
            let n_vars = rows[0].len();
            let time = serde_json::from_value(case["data"]["time"].clone()).unwrap();
            let status = serde_json::from_value(case["data"]["event"].clone()).unwrap();
            let weights = case["weighted"]
                .as_bool()
                .unwrap()
                .then(|| serde_json::from_value(case["data"]["w"].clone()).unwrap());
            let scaled = case["standalone_scale"].as_bool().unwrap();
            let penalty = if let Some(theta) = theta {
                RidgePenalty::new(theta, Some(scaled)).unwrap()
            } else {
                RidgePenalty::from_df(target.unwrap(), n_vars, Some(scaled)).unwrap()
            };
            let fitted = ridge_fit(
                rows.into_iter().flatten().collect(),
                n_obs,
                n_vars,
                time,
                status,
                &penalty,
                weights,
            )
            .unwrap();
            let expected_theta = case["applied_theta"]
                .as_object()
                .unwrap()
                .values()
                .next()
                .unwrap()
                .as_f64()
                .unwrap();
            let expected_df = case["df"].as_f64().unwrap();
            let expected_ell = case["log_likelihood"][1].as_f64().unwrap();
            let close = |actual: f64, expected: f64| {
                assert!(
                    (actual - expected).abs() < 2e-6 * (1.0 + expected.abs()),
                    "{name}: {actual} != {expected}"
                );
            };
            close(fitted.theta, expected_theta);
            close(fitted.df, expected_df);
            close(
                fitted.gcv,
                partial_likelihood_gcv(expected_ell, n_obs, expected_df),
            );
            for (actual, expected) in fitted
                .coefficients
                .iter()
                .zip(case["coefficients"].as_array().unwrap())
            {
                close(*actual, expected.as_f64().unwrap());
            }
            for (actual, expected) in fitted
                .std_err
                .iter()
                .zip(case["std_err"].as_array().unwrap())
            {
                close(*actual, expected.as_f64().unwrap());
            }
            if scaled {
                for (actual, expected) in fitted
                    .scale_factors
                    .unwrap()
                    .iter()
                    .zip(case["scale_factors"].as_array().unwrap())
                {
                    close(*actual, expected.as_f64().unwrap());
                }
            } else {
                assert!(fitted.scale_factors.is_none());
            }
            checked += 1;
        }
        assert!(checked >= 8);
    }

    #[test]
    fn selected_df_is_a_data_dependent_request_and_zero_target_is_valid() {
        let penalty = RidgePenalty::from_df(0.5, 2, None).unwrap();
        assert_eq!(penalty.theta, 1.0);
        assert_eq!(penalty.df, Some(0.5));
        assert!(RidgePenalty::from_df(0.0, 2, None).is_ok());
        assert!(RidgePenalty::from_df(0.0, 0, None).is_err());
        assert!(RidgePenalty::from_df(2.1, 2, None).is_err());
    }

    #[test]
    fn ridge_variances_preserve_constants_and_common_offsets() {
        let x: Vec<f64> = (0..10).flat_map(|i| [i as f64, 1.1]).collect();
        let shifted: Vec<f64> = (0..10).flat_map(|i| [i as f64 + 1e12, 1.1]).collect();
        let variances = ridge_scale_variances(&x, 10, 2);
        assert_eq!(variances[1], 0.0);
        assert_eq!(variances, ridge_scale_variances(&shifted, 10, 2));
        assert_close(variances[0], 55.0 / 6.0);
    }

    fn brute_partial_loglik(
        x: &[f64],
        n_vars: usize,
        time: &[f64],
        status: &[i32],
        beta: &[f64],
    ) -> f64 {
        let lp: Vec<f64> = x
            .chunks_exact(n_vars)
            .map(|row| {
                row.iter()
                    .zip(beta)
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum()
            })
            .collect();
        let shift = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut times: Vec<f64> = time
            .iter()
            .zip(status)
            .filter_map(|(&time, &event)| (event == 1).then_some(time))
            .collect();
        times.sort_by(f64::total_cmp);
        times.dedup();
        let mut result = 0.0;
        for event_time in times {
            let risk: f64 = time
                .iter()
                .zip(&lp)
                .filter(|(time, _)| **time >= event_time)
                .map(|(_, &eta)| (eta - shift).exp())
                .sum();
            let deaths: Vec<usize> = time
                .iter()
                .zip(status)
                .enumerate()
                .filter_map(|(i, (&time, &event))| (time == event_time && event == 1).then_some(i))
                .collect();
            let death_risk: f64 = deaths.iter().map(|&i| (lp[i] - shift).exp()).sum();
            result += deaths.iter().map(|&i| lp[i] - shift).sum::<f64>();
            for step in 0..deaths.len() {
                result -= (risk - step as f64 / deaths.len() as f64 * death_risk).ln();
            }
        }
        result
    }

    #[test]
    fn cv_scores_use_held_out_risk_sets_and_actual_partial_likelihood() {
        let n = 24;
        let p = 2;
        let x: Vec<f64> = (0..n)
            .flat_map(|i| [(i as f64 * 0.7).sin(), (i as f64 * 0.3).cos()])
            .collect();
        let time: Vec<f64> = (0..n).map(|i| (i % 7 + 1) as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| i32::from(i % 5 != 0)).collect();
        let grid = vec![0.1, 1.0, 10.0];
        let (_, actual) = ridge_cv(
            x.clone(),
            n,
            p,
            time.clone(),
            status.clone(),
            Some(grid.clone()),
            Some(3),
        )
        .unwrap();
        for (&theta, score) in grid.iter().zip(actual) {
            let mut expected = 0.0;
            for fold in 0..3 {
                let indices: Vec<usize> = (0..n).filter(|i| i % 3 != fold).collect();
                let train_x: Vec<f64> = indices
                    .iter()
                    .flat_map(|&i| x[i * p..(i + 1) * p].iter().copied())
                    .collect();
                let train_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
                let train_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
                let fitted = ridge_fit(
                    train_x.clone(),
                    indices.len(),
                    p,
                    train_time.clone(),
                    train_status.clone(),
                    &RidgePenalty::new(theta, None).unwrap(),
                    None,
                )
                .unwrap();
                expected -= 2.0
                    * (brute_partial_loglik(&x, p, &time, &status, &fitted.coefficients)
                        - brute_partial_loglik(
                            &train_x,
                            p,
                            &train_time,
                            &train_status,
                            &fitted.coefficients,
                        ));
            }
            assert!((score - expected / n as f64).abs() < 1e-10);
        }
    }

    #[test]
    fn cv_propagates_training_fit_errors() {
        let error = ridge_cv(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            6,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1; 6],
            Some(vec![f64::MAX]),
            Some(2),
        )
        .unwrap_err();
        assert!(error.to_string().contains("penalty values must be finite"));
    }

    #[test]
    fn predictive_cv_selects_an_interior_penalty_for_censored_signal() {
        let mut state = 81_273_u64;
        let mut uniform = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let n = 80;
        let p = 8;
        let mut x = Vec::with_capacity(n * p);
        let mut time = Vec::with_capacity(n);
        let mut status = Vec::with_capacity(n);
        for _ in 0..n {
            let row: Vec<f64> = (0..p)
                .map(|_| (0..12).map(|_| uniform()).sum::<f64>() - 6.0)
                .collect();
            let event_time = -uniform().ln() * (-1.4 * row[0] + 0.4 * row[1]).exp();
            let censor_time = -uniform().ln() * 2.0;
            time.push(event_time.min(censor_time));
            status.push(i32::from(event_time <= censor_time));
            x.extend(row);
        }
        let grid = vec![0.001, 0.1, 1.0, 10.0, 100.0, 1000.0];
        let (theta, scores) = ridge_cv(x, n, p, time, status, Some(grid), Some(5)).unwrap();
        assert_eq!(theta, 10.0);
        assert!(scores[3] + 0.05 < scores[0]);
        assert!(scores[3] + 0.5 < scores[5]);
    }
}
