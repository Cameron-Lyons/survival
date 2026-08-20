use crate::constants::{DIVISION_FLOOR, same_time};
use crate::internal::matrix::{matrix_inverse, regularized_lu_solve};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_empty,
    validate_non_negative,
};
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RidgePenalty {
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
    #[pyo3(get)]
    pub scale_factors: Option<Vec<f64>>,
}

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

    let unit_weights;
    let weights = match weights.as_deref() {
        Some(values) => values,
        None => {
            unit_weights = vec![1.0; n_obs];
            &unit_weights
        }
    };
    let scale_factors = penalty
        .scale
        .then(|| ridge_scale_factors(&x, n_obs, n_vars));
    let penalty_diagonal: Vec<f64> = match scale_factors.as_ref() {
        Some(scales) => scales
            .iter()
            .map(|scale| penalty.theta * scale * scale)
            .collect(),
        None => vec![penalty.theta; n_vars],
    };
    let order = descending_time_order(&time);
    let (coefficients, stats) = fit_penalized_cox(
        &x,
        n_obs,
        n_vars,
        &time,
        &status,
        weights,
        &order,
        &penalty_diagonal,
    )?;
    let mut penalized_information = stats.information.clone();
    for column in 0..n_vars {
        penalized_information[column * n_vars + column] += penalty_diagonal[column];
    }
    let penalized_information = Array2::from_shape_vec((n_vars, n_vars), penalized_information)
        .map_err(|_| PyValueError::new_err("failed to construct ridge information matrix"))?;
    let inverse = ridge_information_inverse(&penalized_information)?;
    let std_err = (0..n_vars)
        .map(|column| inverse[[column, column]].max(0.0).sqrt())
        .collect();
    let df = (0..n_vars)
        .map(|row| {
            (0..n_vars)
                .map(|column| inverse[[row, column]] * stats.information[column * n_vars + row])
                .sum::<f64>()
        })
        .sum();

    let gcv = compute_gcv(&x, n_obs, n_vars, &time, &status, &coefficients, df);

    Ok(RidgeResult {
        coefficients,
        std_err,
        df,
        gcv,
        theta: penalty.theta,
        scale_factors,
    })
}

struct PenalizedCoxStats {
    log_likelihood: f64,
    score: Vec<f64>,
    information: Vec<f64>,
}

fn ridge_scale_factors(x: &[f64], n_obs: usize, n_vars: usize) -> Vec<f64> {
    (0..n_vars)
        .map(|column| {
            let mean = (0..n_obs).map(|row| x[row * n_vars + column]).sum::<f64>() / n_obs as f64;
            let divisor = n_obs.saturating_sub(1).max(1) as f64;
            (((0..n_obs)
                .map(|row| {
                    let centered = x[row * n_vars + column] - mean;
                    centered * centered
                })
                .sum::<f64>()
                / divisor)
                .sqrt())
            .max(DIVISION_FLOOR)
        })
        .collect()
}

fn ridge_information_inverse(information: &Array2<f64>) -> PyResult<Array2<f64>> {
    if let Some(inverse) = matrix_inverse(information) {
        return Ok(inverse);
    }
    if information
        .iter()
        .all(|value| value.abs() <= DIVISION_FLOOR)
    {
        return Ok(Array2::zeros(information.dim()));
    }

    let n = information.nrows();
    let mut inverse = Array2::zeros((n, n));
    for column in 0..n {
        let mut unit = Array1::zeros(n);
        unit[column] = 1.0;
        let solution = regularized_lu_solve(information, &unit)
            .map_err(|_| PyValueError::new_err("ridge information matrix is singular"))?;
        for row in 0..n {
            inverse[[row, column]] = solution[row];
        }
    }
    Ok(inverse)
}

fn descending_time_order(time: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&left, &right| time[right].total_cmp(&time[left]));
    order
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn penalized_cox_stats(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    order: &[usize],
    beta: &[f64],
) -> PenalizedCoxStats {
    let eta: Vec<f64> = (0..n_obs)
        .map(|row| {
            (0..n_vars)
                .map(|column| x[row * n_vars + column] * beta[column])
                .sum()
        })
        .collect();
    let shift = eta
        .iter()
        .zip(weights)
        .filter_map(|(&value, &weight)| (weight > 0.0).then_some(value))
        .fold(f64::NEG_INFINITY, f64::max);
    let shift = if shift.is_finite() { shift } else { 0.0 };
    let mut risk_sum = 0.0;
    let mut risk_x = vec![0.0; n_vars];
    let mut risk_xx = vec![0.0; n_vars * n_vars];
    let mut score = vec![0.0; n_vars];
    let mut information = vec![0.0; n_vars * n_vars];
    let mut log_likelihood = 0.0;
    let mut group_start = 0;

    while group_start < n_obs {
        let group_time = time[order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < n_obs && same_time(time[order[group_end]], group_time) {
            group_end += 1;
        }

        for &row in &order[group_start..group_end] {
            let risk = weights[row] * (eta[row] - shift).exp();
            risk_sum += risk;
            for left in 0..n_vars {
                let x_left = x[row * n_vars + left];
                risk_x[left] += risk * x_left;
                for right in left..n_vars {
                    risk_xx[left * n_vars + right] += risk * x_left * x[row * n_vars + right];
                }
            }
        }

        let mut death_weight = 0.0;
        let mut death_eta = 0.0;
        let mut death_x = vec![0.0; n_vars];
        for &row in &order[group_start..group_end] {
            if status[row] == 0 || weights[row] == 0.0 {
                continue;
            }
            death_weight += weights[row];
            death_eta += weights[row] * eta[row];
            for column in 0..n_vars {
                death_x[column] += weights[row] * x[row * n_vars + column];
            }
        }

        if death_weight > 0.0 && risk_sum > 0.0 {
            log_likelihood += death_eta - death_weight * (risk_sum.ln() + shift);
            for left in 0..n_vars {
                let mean_left = risk_x[left] / risk_sum;
                score[left] += death_x[left] - death_weight * mean_left;
                for right in left..n_vars {
                    let increment = death_weight
                        * (risk_xx[left * n_vars + right] / risk_sum
                            - mean_left * risk_x[right] / risk_sum);
                    information[left * n_vars + right] += increment;
                    if right != left {
                        information[right * n_vars + left] += increment;
                    }
                }
            }
        }
        group_start = group_end;
    }

    PenalizedCoxStats {
        log_likelihood,
        score,
        information,
    }
}

fn penalized_log_likelihood(
    stats: &PenalizedCoxStats,
    beta: &[f64],
    penalty_diagonal: &[f64],
) -> f64 {
    stats.log_likelihood
        - 0.5
            * beta
                .iter()
                .zip(penalty_diagonal)
                .map(|(&coefficient, &penalty)| penalty * coefficient * coefficient)
                .sum::<f64>()
}

#[allow(clippy::too_many_arguments)]
fn fit_penalized_cox(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    order: &[usize],
    penalty_diagonal: &[f64],
) -> PyResult<(Vec<f64>, PenalizedCoxStats)> {
    let mut beta = vec![0.0; n_vars];
    let mut stats = penalized_cox_stats(x, n_obs, n_vars, time, status, weights, order, &beta);
    let mut objective = penalized_log_likelihood(&stats, &beta, penalty_diagonal);

    for _ in 0..50 {
        let mut system = stats.information.clone();
        let mut gradient = stats.score.clone();
        for column in 0..n_vars {
            system[column * n_vars + column] += penalty_diagonal[column];
            gradient[column] -= penalty_diagonal[column] * beta[column];
        }
        let system = Array2::from_shape_vec((n_vars, n_vars), system)
            .map_err(|_| PyValueError::new_err("failed to construct ridge Newton system"))?;
        let delta = regularized_lu_solve(&system, &Array1::from_vec(gradient))
            .map_err(|_| PyValueError::new_err("ridge Newton system is singular"))?;
        if delta
            .iter()
            .fold(0.0_f64, |largest, &value| largest.max(value.abs()))
            < 1e-10
        {
            break;
        }

        let mut step = 1.0;
        let mut accepted = None;
        while step >= 1.0 / 1024.0 {
            let candidate: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(&coefficient, &change)| coefficient + step * change)
                .collect();
            let candidate_stats =
                penalized_cox_stats(x, n_obs, n_vars, time, status, weights, order, &candidate);
            let candidate_objective =
                penalized_log_likelihood(&candidate_stats, &candidate, penalty_diagonal);
            let objective_tolerance = 1e-12 * (1.0 + objective.abs());
            if candidate_objective + objective_tolerance >= objective {
                accepted = Some((candidate, candidate_stats, candidate_objective));
                break;
            }
            step *= 0.5;
        }
        let Some((candidate, candidate_stats, candidate_objective)) = accepted else {
            break;
        };
        let improvement = candidate_objective - objective;
        beta = candidate;
        stats = candidate_stats;
        objective = candidate_objective;
        if improvement <= 1e-13 * (1.0 + objective.abs()) {
            break;
        }
    }
    Ok((beta, stats))
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

        assert!((scale_factors[0] - 1.0).abs() < 1e-12);
        assert_eq!(scale_factors[1], DIVISION_FLOOR);
    }

    #[test]
    fn test_ridge_fit_matches_weighted_reference() {
        let x = vec![
            0.2, 1.2, 0.5, 0.7, 0.8, 1.5, 1.0, 0.2, 1.4, 1.1, 1.8, 0.4, 2.1, 1.8, 2.5, 0.9,
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let weights = vec![1.0, 1.5, 0.5, 2.0, 1.0, 1.2, 0.8, 1.0];

        let unscaled = ridge_fit(
            x.clone(),
            8,
            2,
            time.clone(),
            status.clone(),
            &RidgePenalty::new(0.2, Some(false)).unwrap(),
            Some(weights.clone()),
        )
        .unwrap();
        let scaled = ridge_fit(
            x,
            8,
            2,
            time,
            status,
            &RidgePenalty::new(0.2, Some(true)).unwrap(),
            Some(weights),
        )
        .unwrap();

        for (&actual, expected) in unscaled
            .coefficients
            .iter()
            .zip([-3.17278910503507, 0.0281103888695528])
        {
            assert!((actual - expected).abs() < 5e-9);
        }
        for (&actual, expected) in scaled
            .coefficients
            .iter()
            .zip([-3.81767201833844, 0.113752966479022])
        {
            assert!(
                (actual - expected).abs() < 5e-9,
                "expected {expected}, got {actual}"
            );
        }
        assert!((unscaled.df - 1.37336701637676).abs() < 5e-9);
        assert!((scaled.df - 1.52004107720074).abs() < 5e-9);
    }

    #[test]
    fn test_ridge_fit_uses_shared_risk_set_for_tied_times() {
        let x = vec![0.0, 2.0, 2.0];
        let time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 1.0, 1.0];
        let order = descending_time_order(&time);
        let stats = penalized_cox_stats(&x, 3, 1, &time, &status, &weights, &order, &[0.0]);

        assert!((stats.score[0] + 4.0 / 3.0).abs() < 1e-12);
        assert!((stats.information[0] - 8.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_ridge_fit_excludes_zero_weight_rows_from_risk_sets() {
        let x = vec![0.0, 10.0, 2.0];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 0];
        let weights = vec![1.0, 0.0, 1.0];
        let order = descending_time_order(&time);
        let stats = penalized_cox_stats(&x, 3, 1, &time, &status, &weights, &order, &[0.0]);

        assert!((stats.score[0] + 1.0).abs() < 1e-12);
        assert!((stats.information[0] - 1.0).abs() < 1e-12);
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
