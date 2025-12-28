use crate::utilities::validation::{
    clamp_probability, validate_length, validate_no_nan, validate_non_empty, validate_non_negative,
};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvFitKMOutput {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub estimate: Vec<f64>,
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, weights=None, entry_times=None, position=None, reverse=None, computation_type=None))]
pub fn survfitkm(
    time: Vec<f64>,
    status: Vec<f64>,
    weights: Option<Vec<f64>>,
    entry_times: Option<Vec<f64>>,
    position: Option<Vec<i32>>,
    reverse: Option<bool>,
    computation_type: Option<i32>,
) -> PyResult<SurvFitKMOutput> {
    validate_non_empty(&time, "time")?;
    validate_length(time.len(), status.len(), "status")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&time, "time")?;
    validate_no_nan(&status, "status")?;

    let weights = match weights {
        Some(w) => {
            validate_length(time.len(), w.len(), "weights")?;
            validate_non_negative(&w, "weights")?;
            w
        }
        None => vec![1.0; time.len()],
    };

    let position = match position {
        Some(p) => {
            validate_length(time.len(), p.len(), "position")?;
            p
        }
        None => vec![0; time.len()],
    };

    if let Some(ref entry) = entry_times {
        validate_length(time.len(), entry.len(), "entry_times")?;
    }

    let _reverse = reverse.unwrap_or(false);
    let _computation_type = computation_type.unwrap_or(0);

    Ok(compute_survfitkm(
        &time,
        &status,
        &weights,
        entry_times.as_deref(),
        &position,
        _reverse,
        _computation_type,
    ))
}

pub fn compute_survfitkm(
    time: &[f64],
    status: &[f64],
    weights: &[f64],
    _entry_times: Option<&[f64]>,
    position: &[i32],
    _reverse: bool,
    _computation_type: i32,
) -> SurvFitKMOutput {
    let mut unique_times = Vec::new();
    for (&t, &s) in time.iter().zip(status) {
        if s > 0.0 && !unique_times.contains(&t) {
            unique_times.push(t);
        }
    }
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let dtime = unique_times;

    let ntime = dtime.len();
    let mut n_risk = vec![0.0; ntime];
    let mut n_event = vec![0.0; ntime];
    let mut n_censor = vec![0.0; ntime];
    let mut estimate = vec![1.0; ntime];
    let mut std_err = vec![0.0; ntime];

    let mut current_risk = weights.iter().sum();
    let mut current_estimate = 1.0;
    let mut cumulative_variance = 0.0;

    for (i, &t) in dtime.iter().enumerate().rev() {
        let mut censored = 0.0;
        let mut weighted_events = 0.0;
        let weighted_risk = current_risk;

        for (j, (&time_j, &status_j)) in time.iter().zip(status).enumerate() {
            if (time_j - t).abs() < 1e-9 {
                if status_j > 0.0 {
                    weighted_events += weights[j];
                } else if position[j] & 2 != 0 {
                    censored += 1.0;
                }
            }
        }

        if i < ntime - 1 {
            current_risk -= weighted_events + censored;
        }

        n_risk[i] = weighted_risk;
        n_event[i] = weighted_events;
        n_censor[i] = censored;

        if weighted_risk > 0.0 && weighted_events > 0.0 {
            let hazard = weighted_events / weighted_risk;
            current_estimate *= 1.0 - hazard;
            cumulative_variance += hazard / (weighted_risk - weighted_events);
        }

        estimate[i] = current_estimate;
        std_err[i] = (current_estimate * current_estimate * cumulative_variance).sqrt();
    }

    let z = 1.96;
    let conf_lower: Vec<f64> = estimate
        .iter()
        .zip(std_err.iter())
        .map(|(&s, &se)| {
            if s <= 0.0 || s >= 1.0 || se <= 0.0 {
                clamp_probability(s)
            } else {
                let log_s = s.ln();
                let log_se = se / s;
                clamp_probability((log_s - z * log_se).exp())
            }
        })
        .collect();

    let conf_upper: Vec<f64> = estimate
        .iter()
        .zip(std_err.iter())
        .map(|(&s, &se)| {
            if s <= 0.0 || s >= 1.0 || se <= 0.0 {
                clamp_probability(s)
            } else {
                let log_s = s.ln();
                let log_se = se / s;
                clamp_probability((log_s + z * log_se).exp())
            }
        })
        .collect();

    SurvFitKMOutput {
        time: dtime,
        n_risk,
        n_event,
        n_censor,
        estimate,
        std_err,
        conf_lower,
        conf_upper,
    }
}

#[pymodule]
#[pyo3(name = "survfitkm")]
fn survfitkm_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_class::<SurvFitKMOutput>()?;
    Ok(())
}
