use pyo3::prelude::*;

/// Output from Kaplan-Meier survival estimation.
#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvFitKMOutput {
    /// Unique event times
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Number at risk at each time point
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    /// Number of events at each time point
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    /// Number censored at each time point
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    /// Survival probability estimate
    #[pyo3(get)]
    pub estimate: Vec<f64>,
    /// Standard error of the estimate
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    /// Lower confidence bound (95% CI)
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    /// Upper confidence bound (95% CI)
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
) -> SurvFitKMOutput {
    let weights = weights.unwrap_or_else(|| vec![1.0; time.len()]);
    let position = position.unwrap_or_else(|| vec![0; time.len()]);
    let _reverse = reverse.unwrap_or(false);
    let _computation_type = computation_type.unwrap_or(0);

    survfitkm_internal(
        &time,
        &status,
        &weights,
        entry_times.as_deref(),
        &position,
        _reverse,
        _computation_type,
    )
}

pub fn survfitkm_internal(
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
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    // Calculate 95% confidence intervals using log transformation (more accurate for survival)
    // CI: S(t) * exp(±z * se / S(t)) where z = 1.96 for 95% CI
    let z = 1.96;
    let conf_lower: Vec<f64> = estimate
        .iter()
        .zip(std_err.iter())
        .map(|(&s, &se)| {
            if s <= 0.0 || s >= 1.0 || se <= 0.0 {
                s
            } else {
                let log_s = s.ln();
                let log_se = se / s;
                (log_s - z * log_se).exp().max(0.0)
            }
        })
        .collect();

    let conf_upper: Vec<f64> = estimate
        .iter()
        .zip(std_err.iter())
        .map(|(&s, &se)| {
            if s <= 0.0 || s >= 1.0 || se <= 0.0 {
                s
            } else {
                let log_s = s.ln();
                let log_se = se / s;
                (log_s + z * log_se).exp().min(1.0)
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

#[allow(dead_code)]
fn process_entry_times(entry_times: Option<&[f64]>, position: &[i32]) -> Vec<f64> {
    let mut entry_vec = Vec::new();
    if let Some(entries) = entry_times {
        for (&time, &pos) in entries.iter().zip(position) {
            if pos & 1 != 0 {
                entry_vec.push(time);
            }
        }
    }
    entry_vec
}

#[pymodule]
#[pyo3(name = "survfitkm")]
fn survfitkm_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_class::<SurvFitKMOutput>()?;
    Ok(())
}
