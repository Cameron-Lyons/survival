use crate::utilities::simd::{sum_f64, weighted_squared_diff_sum};
use pyo3::prelude::*;
use rayon::prelude::*;

const SIMD_THRESHOLD: usize = 64;

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
                let score = crate::utilities::simd::squared_diff_sum(predictions, &outcomes_f64);
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
                let score = crate::utilities::simd::squared_diff_sum(&predictions, &outcomes_f64);
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
            let preds_at_t: Vec<f64> = predictions.iter().map(|row| row[t_idx]).collect();
            compute_brier(&preds_at_t, &outcomes, weights_ref)
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
