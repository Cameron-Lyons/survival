use pyo3::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct RCLLResult {
    pub rcll: f64,
    pub mean_rcll: f64,
    pub n_events: usize,
    pub n_censored: usize,
    pub event_contribution: f64,
    pub censored_contribution: f64,
}

impl fmt::Display for RCLLResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RCLLResult(rcll={:.4}, mean={:.4}, n_events={}, n_censored={})",
            self.rcll, self.mean_rcll, self.n_events, self.n_censored
        )
    }
}

#[pymethods]
impl RCLLResult {
    #[new]
    fn new(
        rcll: f64,
        mean_rcll: f64,
        n_events: usize,
        n_censored: usize,
        event_contribution: f64,
        censored_contribution: f64,
    ) -> Self {
        Self {
            rcll,
            mean_rcll,
            n_events,
            n_censored,
            event_contribution,
            censored_contribution,
        }
    }
}

const MIN_PROB: f64 = 1e-7;

fn compute_density_from_survival(survival_probs: &[f64], times: &[f64], event_time: f64) -> f64 {
    if times.is_empty() || survival_probs.is_empty() {
        return MIN_PROB;
    }

    let idx = times
        .iter()
        .position(|&t| t >= event_time)
        .unwrap_or(times.len() - 1);

    if idx == 0 {
        let s_t = survival_probs[0];
        (1.0 - s_t).max(MIN_PROB)
    } else {
        let s_prev = survival_probs[idx - 1];
        let s_curr = survival_probs[idx];
        (s_prev - s_curr).max(MIN_PROB)
    }
}

fn get_survival_at_time(survival_probs: &[f64], times: &[f64], t: f64) -> f64 {
    if times.is_empty() || survival_probs.is_empty() {
        return 1.0;
    }

    if t < times[0] {
        return 1.0;
    }

    if t >= times[times.len() - 1] {
        return survival_probs[survival_probs.len() - 1].max(MIN_PROB);
    }

    let mut left = 0;
    let mut right = times.len();

    while left < right {
        let mid = (left + right) / 2;
        if times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 {
        1.0
    } else {
        survival_probs[left - 1].max(MIN_PROB)
    }
}

pub fn compute_rcll(
    survival_predictions: &[Vec<f64>],
    prediction_times: &[f64],
    event_times: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
) -> RCLLResult {
    let n = event_times.len();

    if n == 0 || survival_predictions.is_empty() || prediction_times.is_empty() {
        return RCLLResult {
            rcll: 0.0,
            mean_rcll: 0.0,
            n_events: 0,
            n_censored: 0,
            event_contribution: 0.0,
            censored_contribution: 0.0,
        };
    }

    let default_weights: Vec<f64> = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    let mut total_loss = 0.0;
    let mut total_weight = 0.0;
    let mut n_events = 0usize;
    let mut n_censored = 0usize;
    let mut event_contribution = 0.0;
    let mut censored_contribution = 0.0;

    for i in 0..n {
        let surv_probs = &survival_predictions[i];
        let t_i = event_times[i];
        let delta_i = status[i];
        let weight = w[i];

        let loss = if delta_i == 1 {
            let f_t = compute_density_from_survival(surv_probs, prediction_times, t_i);
            let contrib = -f_t.ln();
            n_events += 1;
            event_contribution += weight * contrib;
            contrib
        } else {
            let s_t = get_survival_at_time(surv_probs, prediction_times, t_i);
            let contrib = -s_t.ln();
            n_censored += 1;
            censored_contribution += weight * contrib;
            contrib
        };

        total_loss += weight * loss;
        total_weight += weight;
    }

    let mean_rcll = if total_weight > 0.0 {
        total_loss / total_weight
    } else {
        0.0
    };

    RCLLResult {
        rcll: total_loss,
        mean_rcll,
        n_events,
        n_censored,
        event_contribution,
        censored_contribution,
    }
}

pub fn compute_rcll_single_time(
    survival_probs: &[f64],
    event_times: &[f64],
    status: &[i32],
    prediction_time: f64,
    weights: Option<&[f64]>,
) -> RCLLResult {
    let n = event_times.len();

    if n == 0 || survival_probs.len() != n {
        return RCLLResult {
            rcll: 0.0,
            mean_rcll: 0.0,
            n_events: 0,
            n_censored: 0,
            event_contribution: 0.0,
            censored_contribution: 0.0,
        };
    }

    let default_weights: Vec<f64> = vec![1.0; n];
    let w = weights.unwrap_or(&default_weights);

    let mut total_loss = 0.0;
    let mut total_weight = 0.0;
    let mut n_events = 0usize;
    let mut n_censored = 0usize;
    let mut event_contribution = 0.0;
    let mut censored_contribution = 0.0;

    for i in 0..n {
        let s_i = survival_probs[i].clamp(MIN_PROB, 1.0 - MIN_PROB);
        let t_i = event_times[i];
        let delta_i = status[i];
        let weight = w[i];

        let loss = if delta_i == 1 && t_i <= prediction_time {
            let f_t = (1.0 - s_i).max(MIN_PROB);
            let contrib = -f_t.ln();
            n_events += 1;
            event_contribution += weight * contrib;
            contrib
        } else if delta_i == 0 || t_i > prediction_time {
            let contrib = -s_i.ln();
            n_censored += 1;
            censored_contribution += weight * contrib;
            contrib
        } else {
            0.0
        };

        total_loss += weight * loss;
        total_weight += weight;
    }

    let mean_rcll = if total_weight > 0.0 {
        total_loss / total_weight
    } else {
        0.0
    };

    RCLLResult {
        rcll: total_loss,
        mean_rcll,
        n_events,
        n_censored,
        event_contribution,
        censored_contribution,
    }
}

#[pyfunction]
#[pyo3(signature = (survival_predictions, prediction_times, event_times, status, weights=None))]
pub fn rcll(
    survival_predictions: Vec<Vec<f64>>,
    prediction_times: Vec<f64>,
    event_times: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
) -> PyResult<RCLLResult> {
    let n = event_times.len();

    if n != status.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "event_times and status must have the same length",
        ));
    }

    if survival_predictions.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "survival_predictions must have one row per observation",
        ));
    }

    if let Some(ref w) = weights
        && w.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must have the same length as event_times",
        ));
    }

    for (i, row) in survival_predictions.iter().enumerate() {
        if row.len() != prediction_times.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "survival_predictions row {} has {} elements, expected {}",
                i,
                row.len(),
                prediction_times.len()
            )));
        }
    }

    Ok(compute_rcll(
        &survival_predictions,
        &prediction_times,
        &event_times,
        &status,
        weights.as_deref(),
    ))
}

#[pyfunction]
#[pyo3(signature = (survival_probs, event_times, status, prediction_time, weights=None))]
pub fn rcll_single_time(
    survival_probs: Vec<f64>,
    event_times: Vec<f64>,
    status: Vec<i32>,
    prediction_time: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<RCLLResult> {
    let n = event_times.len();

    if n != status.len() || n != survival_probs.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "survival_probs, event_times, and status must have the same length",
        ));
    }

    if let Some(ref w) = weights
        && w.len() != n
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must have the same length as event_times",
        ));
    }

    if prediction_time <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prediction_time must be positive",
        ));
    }

    Ok(compute_rcll_single_time(
        &survival_probs,
        &event_times,
        &status,
        prediction_time,
        weights.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rcll_basic() {
        let survival_predictions = vec![
            vec![0.95, 0.85, 0.70, 0.50],
            vec![0.90, 0.75, 0.55, 0.35],
            vec![0.98, 0.92, 0.80, 0.65],
        ];
        let prediction_times = vec![1.0, 2.0, 3.0, 4.0];
        let event_times = vec![2.5, 1.5, 4.0];
        let status = vec![1, 1, 0];

        let result = compute_rcll(
            &survival_predictions,
            &prediction_times,
            &event_times,
            &status,
            None,
        );

        assert!(result.mean_rcll > 0.0);
        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_censored, 1);
    }

    #[test]
    fn test_rcll_single_time() {
        let survival_probs = vec![0.8, 0.6, 0.9, 0.7];
        let event_times = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 0];
        let prediction_time = 5.0;

        let result = compute_rcll_single_time(
            &survival_probs,
            &event_times,
            &status,
            prediction_time,
            None,
        );

        assert!(result.mean_rcll > 0.0);
        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_censored, 2);
    }

    #[test]
    fn test_rcll_perfect_prediction() {
        let survival_probs = vec![0.99, 0.99, 0.01, 0.01];
        let event_times = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![0, 0, 1, 1];
        let prediction_time = 5.0;

        let result = compute_rcll_single_time(
            &survival_probs,
            &event_times,
            &status,
            prediction_time,
            None,
        );

        assert!(result.mean_rcll < 1.0);
    }

    #[test]
    fn test_rcll_empty_input() {
        let result = compute_rcll(&[], &[], &[], &[], None);

        assert_eq!(result.rcll, 0.0);
        assert_eq!(result.n_events, 0);
        assert_eq!(result.n_censored, 0);
    }

    #[test]
    fn test_rcll_weighted() {
        let survival_probs = vec![0.8, 0.6, 0.9];
        let event_times = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 0];
        let weights = vec![2.0, 1.0, 1.0];
        let prediction_time = 5.0;

        let result = compute_rcll_single_time(
            &survival_probs,
            &event_times,
            &status,
            prediction_time,
            Some(&weights),
        );

        assert!(result.mean_rcll > 0.0);
    }

    #[test]
    fn test_survival_at_time() {
        let survival_probs = vec![0.9, 0.8, 0.7, 0.6];
        let times = vec![1.0, 2.0, 3.0, 4.0];

        assert!((get_survival_at_time(&survival_probs, &times, 0.5) - 1.0).abs() < 1e-10);
        assert!((get_survival_at_time(&survival_probs, &times, 1.5) - 0.9).abs() < 1e-10);
        assert!((get_survival_at_time(&survival_probs, &times, 2.5) - 0.8).abs() < 1e-10);
        assert!((get_survival_at_time(&survival_probs, &times, 5.0) - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_density_from_survival() {
        let survival_probs = vec![0.9, 0.8, 0.7, 0.6];
        let times = vec![1.0, 2.0, 3.0, 4.0];

        let density = compute_density_from_survival(&survival_probs, &times, 2.0);
        assert!((density - 0.1).abs() < 1e-10);
    }
}
