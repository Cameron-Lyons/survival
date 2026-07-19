use crate::constants::{DIVISION_FLOOR, TIME_EPSILON};
use crate::internal::simd::{sum_f64, weighted_squared_diff_sum};
use pyo3::prelude::*;
use rayon::prelude::*;

const SIMD_THRESHOLD: usize = 64;

/// Compute inverse-probability-of-censoring weighted Brier scores.
///
/// `survival_predictions` is a flattened row-major matrix (`observation x
/// evaluation time`) containing predictions for the survival probability
/// `P(T > t)`. Censoring is estimated once with a weighted Kaplan-Meier curve.
/// Observed events are removed before censorings at a tied time, matching
/// `survival::brier`'s default tie handling.
pub(crate) fn ipcw_brier_scores(
    observed_times: &[f64],
    status: &[u8],
    survival_predictions: &[f64],
    evaluation_times: &[f64],
    case_weights: Option<&[f64]>,
) -> Result<Vec<f64>, String> {
    let n = observed_times.len();
    if status.len() != n {
        return Err("observed_times and status must have the same length".to_string());
    }
    let n_times = evaluation_times.len();
    let expected_predictions = n
        .checked_mul(n_times)
        .ok_or_else(|| "survival prediction shape is too large".to_string())?;
    if survival_predictions.len() != expected_predictions {
        return Err(format!(
            "survival_predictions has {} values but expected {expected_predictions} for shape ({n}, {n_times})",
            survival_predictions.len()
        ));
    }
    if n == 0 {
        return if evaluation_times.is_empty() {
            Ok(Vec::new())
        } else {
            Err("cannot score evaluation times without observations".to_string())
        };
    }
    if case_weights.is_some_and(|weights| weights.len() != n) {
        return Err("case_weights must have the same length as observations".to_string());
    }
    if observed_times.iter().any(|time| !time.is_finite()) {
        return Err("observed_times must be finite".to_string());
    }
    if evaluation_times.iter().any(|time| !time.is_finite()) {
        return Err("evaluation_times must be finite".to_string());
    }
    if let Some((idx, value)) = status
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| *value > 1)
    {
        return Err(format!(
            "status must contain only 0/1 values; got {value} at index {idx}"
        ));
    }

    if case_weights.is_some_and(|weights| {
        weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    }) {
        return Err("case_weights must be finite and non-negative".to_string());
    }
    let weight_at = |row_idx: usize| case_weights.map_or(1.0, |weights| weights[row_idx]);
    let total_case_weight = case_weights.map_or(n as f64, |weights| weights.iter().sum());
    if !total_case_weight.is_finite() || total_case_weight <= 0.0 {
        return Err("case_weights must have positive sum".to_string());
    }

    if let Some((prediction_idx, value)) = survival_predictions
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        let row_idx = prediction_idx / n_times;
        let time_idx = prediction_idx % n_times;
        return Err(format!(
            "survival prediction at row {row_idx}, column {time_idx} must be finite and between 0 and 1; got {value}"
        ));
    }

    // Shift censorings just after deaths at the same time. Besides defining
    // the tie order, using the adjusted times throughout reproduces the
    // interval convention used by R's brier implementation.
    let mut unique_times = observed_times.to_vec();
    unique_times.sort_by(f64::total_cmp);
    unique_times.dedup_by(|left, right| (*left - *right).abs() < TIME_EPSILON);
    let censor_shift = unique_times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|difference| *difference > 0.0)
        .reduce(f64::min)
        .map_or(0.0, |difference| difference / 2.0);
    let adjusted_times: Vec<f64> = observed_times
        .iter()
        .zip(status)
        .map(|(&time, &event)| {
            if event == 0 {
                time + censor_shift
            } else {
                time
            }
        })
        .collect();

    // Build G(t), the censoring survival curve, on the adjusted time scale.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        adjusted_times[left]
            .total_cmp(&adjusted_times[right])
            .then_with(|| left.cmp(&right))
    });
    let mut risk_weight = total_case_weight;
    let mut censoring_survival = 1.0;
    let mut censor_times = Vec::new();
    let mut censor_survival_after = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        let block_start = pos;
        let block_time = adjusted_times[order[pos]];
        pos += 1;
        while pos < n && (adjusted_times[order[pos]] - block_time).abs() < TIME_EPSILON {
            pos += 1;
        }

        let mut event_weight = 0.0;
        let mut censor_weight = 0.0;
        for &row_idx in &order[block_start..pos] {
            if status[row_idx] == 1 {
                event_weight += weight_at(row_idx);
            } else {
                censor_weight += weight_at(row_idx);
            }
        }
        if censor_weight > 0.0 {
            if risk_weight <= 0.0 {
                censoring_survival = 0.0;
            } else {
                censoring_survival *= (1.0 - censor_weight / risk_weight).clamp(0.0, 1.0);
            }
            censor_times.push(block_time);
            censor_survival_after.push(censoring_survival);
        }
        risk_weight = (risk_weight - event_weight - censor_weight).max(0.0);
    }

    let censor_survival_at = |time: f64| {
        let index = censor_times.partition_point(|&censor_time| censor_time <= time + TIME_EPSILON);
        if index == 0 {
            1.0
        } else {
            censor_survival_after[index - 1]
        }
    };

    let mut scores = Vec::with_capacity(evaluation_times.len());
    for (time_idx, &evaluation_time) in evaluation_times.iter().enumerate() {
        let mut weighted_loss = 0.0;
        let mut scoring_weight = 0.0;

        for row_idx in 0..n {
            let observed_time = adjusted_times[row_idx];
            let event = status[row_idx] == 1;
            if !event && observed_time + TIME_EPSILON < evaluation_time {
                continue;
            }

            let censoring_probability = censor_survival_at(observed_time.min(evaluation_time));
            let case_weight = weight_at(row_idx);
            if case_weight > 0.0 && censoring_probability <= DIVISION_FLOOR {
                return Err(format!(
                    "censoring survival is zero at evaluation time {evaluation_time}"
                ));
            }
            let weight = if case_weight == 0.0 {
                0.0
            } else {
                case_weight / censoring_probability
            };
            let target_survival = if event && observed_time <= evaluation_time + TIME_EPSILON {
                0.0
            } else {
                1.0
            };
            let residual = survival_predictions[row_idx * n_times + time_idx] - target_survival;
            weighted_loss += weight * residual * residual;
            scoring_weight += weight;
        }

        if scoring_weight <= 0.0 {
            return Err(format!(
                "no observable outcomes remain at evaluation time {evaluation_time}"
            ));
        }
        scores.push(weighted_loss / scoring_weight);
    }
    Ok(scores)
}

fn validate_predictions(predictions: &[f64]) -> bool {
    predictions.iter().all(|&p| (0.0..=1.0).contains(&p))
}

pub fn compute_brier(
    predictions: &[f64],
    outcomes: &[i32],
    weights: Option<&[f64]>,
) -> Option<f64> {
    let n = predictions.len();
    if n != outcomes.len() {
        return None;
    }
    if weights.is_some_and(|w| w.len() != n) {
        return None;
    }
    if n == 0 {
        return Some(0.0);
    }

    if !validate_predictions(predictions) {
        return None;
    }

    let outcomes_f64: Vec<f64> = outcomes.iter().map(|&x| x as f64).collect();

    if n >= SIMD_THRESHOLD {
        match weights {
            Some(w) => {
                let score = weighted_squared_diff_sum(predictions, &outcomes_f64, w);
                let total_weight = sum_f64(w);
                if total_weight > 0.0 {
                    Some(score / total_weight)
                } else {
                    Some(0.0)
                }
            }
            None => {
                let score = crate::internal::simd::squared_diff_sum(predictions, &outcomes_f64);
                Some(score / n as f64)
            }
        }
    } else {
        let mut score = 0.0;
        let mut total_weight = 0.0;
        for i in 0..n {
            let pred = predictions[i];
            let obs = outcomes_f64[i];
            let w = weights.map_or(1.0, |ws| ws[i]);
            score += w * (pred - obs).powi(2);
            total_weight += w;
        }
        if total_weight > 0.0 {
            Some(score / total_weight)
        } else {
            Some(0.0)
        }
    }
}

fn compute_brier_column(
    predictions: &[Vec<f64>],
    outcomes: &[i32],
    column_index: usize,
    weights: Option<&[f64]>,
) -> Option<f64> {
    let n = outcomes.len();
    if predictions.len() != n || weights.is_some_and(|w| w.len() != n) {
        return None;
    }
    if n == 0 {
        return Some(0.0);
    }

    let mut score = 0.0;
    let mut total_weight = 0.0;

    for (row_index, row) in predictions.iter().enumerate() {
        let pred = row[column_index];
        if !(0.0..=1.0).contains(&pred) {
            return None;
        }
        let weight = weights.map_or(1.0, |w| w[row_index]);
        let diff = pred - outcomes[row_index] as f64;
        score += weight * diff * diff;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        Some(score / total_weight)
    } else {
        Some(0.0)
    }
}

#[pyfunction]
#[pyo3(signature = (predictions, outcomes, weights=None))]
pub fn brier(
    predictions: Vec<f64>,
    outcomes: Vec<i32>,
    weights: Option<Vec<f64>>,
) -> PyResult<f64> {
    let n = predictions.len();
    if n != outcomes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "predictions and outcomes must have the same length",
        ));
    }
    if n == 0 {
        return Ok(0.0);
    }

    if !validate_predictions(&predictions) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "predictions must be between 0 and 1",
        ));
    }

    let outcomes_f64: Vec<f64> = outcomes.iter().map(|&x| x as f64).collect();

    if n >= SIMD_THRESHOLD {
        match weights {
            Some(ref w) => {
                if w.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "weights must have the same length as predictions",
                    ));
                }
                let score = weighted_squared_diff_sum(&predictions, &outcomes_f64, w);
                let total_weight = sum_f64(w);
                if total_weight > 0.0 {
                    Ok(score / total_weight)
                } else {
                    Ok(0.0)
                }
            }
            None => {
                let score = crate::internal::simd::squared_diff_sum(&predictions, &outcomes_f64);
                Ok(score / n as f64)
            }
        }
    } else {
        let weights = if let Some(w) = weights {
            if w.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "weights must have the same length as predictions",
                ));
            }
            w
        } else {
            vec![1.0; n]
        };

        let mut score = 0.0;
        let mut total_weight = 0.0;
        for i in 0..n {
            let pred = predictions[i];
            let obs = outcomes_f64[i];
            let w = weights[i];
            score += w * (pred - obs).powi(2);
            total_weight += w;
        }
        if total_weight > 0.0 {
            Ok(score / total_weight)
        } else {
            Ok(0.0)
        }
    }
}

#[pyfunction]
#[pyo3(signature = (predictions, outcomes, times, weights=None))]
pub fn integrated_brier(
    predictions: Vec<Vec<f64>>,
    outcomes: Vec<i32>,
    times: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<f64> {
    if predictions.is_empty() {
        return Ok(0.0);
    }
    let n_obs = predictions.len();
    let n_times = predictions[0].len();
    if n_times != times.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "number of time points must match number of prediction columns",
        ));
    }
    if n_obs != outcomes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "predictions and outcomes must have the same number of observations",
        ));
    }
    for pred_row in &predictions {
        if pred_row.len() != n_times {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all prediction rows must have the same length",
            ));
        }
    }
    if let Some(w) = weights.as_ref()
        && w.len() != n_obs
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must have the same length as predictions",
        ));
    }
    let mut time_intervals = Vec::with_capacity(n_times);
    for i in 0..n_times {
        let interval_width = if i == 0 {
            if n_times > 1 {
                times[1] - times[0]
            } else {
                1.0
            }
        } else if i == n_times - 1 {
            times[i] - times[i - 1]
        } else {
            (times[i + 1] - times[i - 1]) / 2.0
        };
        time_intervals.push(interval_width);
    }
    let total_time: f64 = time_intervals.iter().sum();
    let weights_ref = weights.as_deref();
    let result = time_intervals
        .par_iter()
        .enumerate()
        .map(|(t_idx, &interval)| {
            compute_brier_column(&predictions, &outcomes, t_idx, weights_ref)
                .map(|score| score * interval)
                .ok_or("invalid prediction value")
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b));
    match result {
        Ok(integrated_score) => {
            if total_time > 0.0 {
                Ok(integrated_score / total_time)
            } else {
                Ok(0.0)
            }
        }
        Err(_) => Err(pyo3::exceptions::PyValueError::new_err(
            "predictions must be between 0 and 1",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipcw_brier_matches_hand_calculated_censoring_weights() {
        let scores = ipcw_brier_scores(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 0, 1, 0],
            &[0.1, 0.4, 0.7, 0.8],
            &[2.5],
            None,
        )
        .expect("valid IPCW score should be computed");

        // R's default tie handling moves the censoring at time 2 to 2.5.
        // At that boundary all four rows remain observable and G(2.5)=2/3.
        assert!((scores[0] - 0.13545454545454547).abs() < 1e-12);
    }

    #[test]
    fn ipcw_brier_is_zero_for_perfect_survival_predictions() {
        let scores = ipcw_brier_scores(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 0, 1, 0],
            &[0.0, 1.0, 1.0, 1.0],
            &[2.5],
            None,
        )
        .expect("perfect observable predictions should be scored");

        assert_eq!(scores, vec![0.0]);
    }

    #[test]
    fn ipcw_brier_groups_near_tied_event_and_censor_times() {
        let predictions = [0.2, 0.4, 0.6, 0.8];
        let exact = ipcw_brier_scores(
            &[1.0, 2.0, 2.0, 3.0],
            &[1, 1, 0, 0],
            &predictions,
            &[2.0],
            None,
        )
        .expect("exact ties should be scored");
        let near = ipcw_brier_scores(
            &[1.0, 2.0, 2.0 + TIME_EPSILON / 2.0, 3.0],
            &[1, 1, 0, 0],
            &predictions,
            &[2.0],
            None,
        )
        .expect("near ties should be scored consistently");

        assert!((exact[0] - 0.1).abs() < 1e-12);
        assert!((near[0] - exact[0]).abs() < 1e-12);
    }

    #[test]
    fn ipcw_brier_validates_flat_prediction_shape_and_values() {
        let shape_error = ipcw_brier_scores(&[1.0, 2.0], &[1, 0], &[0.2], &[1.0], None)
            .expect_err("a truncated prediction matrix should fail");
        assert!(shape_error.contains("expected 2 for shape (2, 1)"));

        let value_error = ipcw_brier_scores(&[1.0, 2.0], &[1, 0], &[0.2, f64::NAN], &[1.0], None)
            .expect_err("non-finite survival predictions should fail");
        assert!(value_error.contains("row 1, column 0"));
    }

    #[test]
    fn compute_brier_rejects_mismatched_weights() {
        assert_eq!(compute_brier(&[0.2, 0.8], &[0, 1], Some(&[1.0])), None);
    }

    #[test]
    fn integrated_brier_uses_weighted_columns_without_scratch_vectors() {
        let result = integrated_brier(
            vec![vec![0.1, 0.2, 0.4], vec![0.8, 0.7, 0.6]],
            vec![0, 1],
            vec![1.0, 2.0, 4.0],
            Some(vec![1.0, 3.0]),
        )
        .unwrap();

        assert!((result - 0.10416666666666667).abs() < 1e-12);
    }

    #[test]
    fn integrated_brier_rejects_mismatched_weights() {
        Python::initialize();
        let err = integrated_brier(
            vec![vec![0.1, 0.2], vec![0.8, 0.7]],
            vec![0, 1],
            vec![1.0, 2.0],
            Some(vec![1.0]),
        )
        .expect_err("mismatched weights should fail validation");

        assert!(
            err.to_string()
                .contains("weights must have the same length")
        );
    }
}
