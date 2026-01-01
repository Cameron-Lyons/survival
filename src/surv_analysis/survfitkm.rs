use crate::utilities::validation::{
    clamp_probability, validate_length, validate_no_nan, validate_non_empty, validate_non_negative,
};
use pyo3::prelude::*;

#[derive(Debug, Clone, Default)]
#[pyclass]
pub struct SurvfitKMOptions {
    #[pyo3(get, set)]
    pub weights: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub entry_times: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub position: Option<Vec<i32>>,
    #[pyo3(get, set)]
    pub reverse: Option<bool>,
    #[pyo3(get, set)]
    pub computation_type: Option<i32>,
}

#[pymethods]
impl SurvfitKMOptions {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_weights(mut self_: PyRefMut<'_, Self>, weights: Vec<f64>) -> PyRefMut<'_, Self> {
        self_.weights = Some(weights);
        self_
    }

    pub fn with_entry_times(
        mut self_: PyRefMut<'_, Self>,
        entry_times: Vec<f64>,
    ) -> PyRefMut<'_, Self> {
        self_.entry_times = Some(entry_times);
        self_
    }

    pub fn with_position(mut self_: PyRefMut<'_, Self>, position: Vec<i32>) -> PyRefMut<'_, Self> {
        self_.position = Some(position);
        self_
    }

    pub fn with_reverse(mut self_: PyRefMut<'_, Self>, reverse: bool) -> PyRefMut<'_, Self> {
        self_.reverse = Some(reverse);
        self_
    }

    pub fn with_computation_type(
        mut self_: PyRefMut<'_, Self>,
        computation_type: i32,
    ) -> PyRefMut<'_, Self> {
        self_.computation_type = Some(computation_type);
        self_
    }
}

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
    _position: &[i32],
    _reverse: bool,
    _computation_type: i32,
) -> SurvFitKMOutput {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let estimated_events = (n / 10).max(16);
    let mut event_times = Vec::with_capacity(estimated_events);
    let mut n_risk_vec = Vec::with_capacity(estimated_events);
    let mut n_event_vec = Vec::with_capacity(estimated_events);
    let mut n_censor_vec = Vec::with_capacity(estimated_events);
    let mut estimate_vec = Vec::with_capacity(estimated_events);
    let mut std_err_vec = Vec::with_capacity(estimated_events);
    let mut current_risk: f64 = weights.iter().sum();
    let mut current_estimate = 1.0;
    let mut cumulative_variance = 0.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut weighted_events = 0.0;
        let mut weighted_censor = 0.0;
        let mut j = i;
        while j < n && (time[indices[j]] - current_time).abs() < 1e-9 {
            let idx = indices[j];
            if status[idx] > 0.0 {
                weighted_events += weights[idx];
            } else {
                weighted_censor += weights[idx];
            }
            j += 1;
        }
        if weighted_events > 0.0 {
            let risk_at_time = current_risk;
            event_times.push(current_time);
            n_risk_vec.push(risk_at_time);
            n_event_vec.push(weighted_events);
            n_censor_vec.push(weighted_censor);
            if risk_at_time > 0.0 {
                let hazard = weighted_events / risk_at_time;
                current_estimate *= 1.0 - hazard;
                if risk_at_time > weighted_events {
                    cumulative_variance +=
                        weighted_events / (risk_at_time * (risk_at_time - weighted_events));
                }
            }
            estimate_vec.push(current_estimate);
            let se = current_estimate * cumulative_variance.sqrt();
            std_err_vec.push(se);
        }
        current_risk -= weighted_events + weighted_censor;
        i = j;
    }
    let z = 1.96;
    let (conf_lower, conf_upper): (Vec<f64>, Vec<f64>) = estimate_vec
        .iter()
        .zip(std_err_vec.iter())
        .map(|(&s, &se)| {
            if s <= 0.0 || s >= 1.0 || se <= 0.0 {
                (clamp_probability(s), clamp_probability(s))
            } else {
                let log_s = s.ln();
                let log_se = se / s;
                (
                    clamp_probability((log_s - z * log_se).exp()),
                    clamp_probability((log_s + z * log_se).exp()),
                )
            }
        })
        .unzip();
    SurvFitKMOutput {
        time: event_times,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_censor: n_censor_vec,
        estimate: estimate_vec,
        std_err: std_err_vec,
        conf_lower,
        conf_upper,
    }
}

#[pyfunction]
pub fn survfitkm_with_options(
    time: Vec<f64>,
    status: Vec<f64>,
    options: Option<&SurvfitKMOptions>,
) -> PyResult<SurvFitKMOutput> {
    let opts = options.cloned().unwrap_or_default();
    survfitkm(
        time,
        status,
        opts.weights,
        opts.entry_times,
        opts.position,
        opts.reverse,
        opts.computation_type,
    )
}

#[pymodule]
#[pyo3(name = "survfitkm")]
fn survfitkm_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_with_options, &m)?)?;
    m.add_class::<SurvFitKMOutput>()?;
    m.add_class::<SurvfitKMOptions>()?;
    Ok(())
}
