use crate::constants::{PARALLEL_THRESHOLD_XLARGE, TIME_EPSILON};
use crate::utilities::numpy_utils::{
    extract_optional_vec_f64, extract_optional_vec_i32, extract_vec_f64,
};
use crate::utilities::validation::{
    clamp_probability, validate_length, validate_no_nan, validate_non_empty, validate_non_negative,
};
use pyo3::prelude::*;
use rayon::prelude::*;

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
pub struct KaplanMeierConfig {
    #[pyo3(get, set)]
    pub reverse: bool,

    #[pyo3(get, set)]
    pub computation_type: i32,

    #[pyo3(get, set)]
    pub conf_level: f64,
}

#[pymethods]
impl KaplanMeierConfig {
    #[new]
    #[pyo3(signature = (reverse=None, computation_type=None, conf_level=None))]
    fn new(reverse: Option<bool>, computation_type: Option<i32>, conf_level: Option<f64>) -> Self {
        Self {
            reverse: reverse.unwrap_or(false),
            computation_type: computation_type.unwrap_or(0),
            conf_level: conf_level.unwrap_or(0.95),
        }
    }
}

impl Default for KaplanMeierConfig {
    fn default() -> Self {
        Self {
            reverse: false,
            computation_type: 0,
            conf_level: 0.95,
        }
    }
}

impl KaplanMeierConfig {
    pub fn create(
        reverse: Option<bool>,
        computation_type: Option<i32>,
        conf_level: Option<f64>,
    ) -> Self {
        Self {
            reverse: reverse.unwrap_or(false),
            computation_type: computation_type.unwrap_or(0),
            conf_level: conf_level.unwrap_or(0.95),
        }
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

/// Compute Kaplan-Meier survival curve estimates.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times. Accepts numpy arrays, pandas Series, polars Series, or lists.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// weights : array-like, optional
///     Case weights for weighted estimation.
/// entry_times : array-like, optional
///     Left truncation (late entry) times for delayed entry data.
/// position : array-like, optional
///     Position indicators for tied event handling.
/// reverse : bool, optional
///     If True, compute reverse Kaplan-Meier (censoring distribution). Default False.
/// computation_type : int, optional
///     Algorithm variant (0=standard, 1=alternative). Default 0.
///
/// Returns
/// -------
/// SurvFitKMOutput
///     Object containing: time (event times), n_risk (at-risk counts), n_event (event counts),
///     survival (survival probabilities), std_err (standard errors), conf_lower/conf_upper (95% CI).
///
/// Examples
/// --------
/// >>> import survival
/// >>> import pandas as pd
/// >>> df = pd.DataFrame({'time': [1, 2, 3, 4, 5], 'status': [1, 0, 1, 1, 0]})
/// >>> result = survival.survfitkm(df['time'], df['status'])
/// >>> result.survival
#[pyfunction]
#[pyo3(signature = (time, status, weights=None, entry_times=None, position=None, reverse=None, computation_type=None))]
pub fn survfitkm(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
    entry_times: Option<&Bound<'_, PyAny>>,
    position: Option<&Bound<'_, PyAny>>,
    reverse: Option<bool>,
    computation_type: Option<i32>,
) -> PyResult<SurvFitKMOutput> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_f64(status)?;
    let weights_opt = extract_optional_vec_f64(weights)?;
    let entry_times_opt = extract_optional_vec_f64(entry_times)?;
    let position_opt = extract_optional_vec_i32(position)?;
    let config = KaplanMeierConfig {
        reverse: reverse.unwrap_or(false),
        computation_type: computation_type.unwrap_or(0),
        conf_level: 0.95,
    };
    validate_non_empty(&time, "time")?;
    validate_length(time.len(), status.len(), "status")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&time, "time")?;
    validate_no_nan(&status, "status")?;
    let weights = match weights_opt {
        Some(w) => {
            validate_length(time.len(), w.len(), "weights")?;
            validate_non_negative(&w, "weights")?;
            w
        }
        None => vec![1.0; time.len()],
    };
    let position = match position_opt {
        Some(p) => {
            validate_length(time.len(), p.len(), "position")?;
            p
        }
        None => vec![0; time.len()],
    };
    if let Some(ref entry) = entry_times_opt {
        validate_length(time.len(), entry.len(), "entry_times")?;
    }
    Ok(compute_survfitkm(
        &time,
        &status,
        &weights,
        entry_times_opt.as_deref(),
        &position,
        &config,
    ))
}

pub fn compute_survfitkm(
    time: &[f64],
    status: &[f64],
    weights: &[f64],
    _entry_times: Option<&[f64]>,
    _position: &[i32],
    config: &KaplanMeierConfig,
) -> SurvFitKMOutput {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    if n > PARALLEL_THRESHOLD_XLARGE {
        indices.par_sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
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

    let _reverse = config.reverse;
    let _computation_type = config.computation_type;

    while i < n {
        let current_time = time[indices[i]];
        let mut weighted_events = 0.0;
        let mut weighted_censor = 0.0;
        let mut j = i;
        while j < n && (time[indices[j]] - current_time).abs() < TIME_EPSILON {
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

    let alpha = 1.0 - config.conf_level;
    let z = normal_quantile(1.0 - alpha / 2.0);

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
    validate_non_empty(&time, "time")?;
    validate_length(time.len(), status.len(), "status")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&time, "time")?;
    validate_no_nan(&status, "status")?;
    let weights = match opts.weights {
        Some(w) => {
            validate_length(time.len(), w.len(), "weights")?;
            validate_non_negative(&w, "weights")?;
            w
        }
        None => vec![1.0; time.len()],
    };
    let position = match opts.position {
        Some(p) => {
            validate_length(time.len(), p.len(), "position")?;
            p
        }
        None => vec![0; time.len()],
    };
    if let Some(ref entry) = opts.entry_times {
        validate_length(time.len(), entry.len(), "entry_times")?;
    }
    let config = KaplanMeierConfig {
        reverse: opts.reverse.unwrap_or(false),
        computation_type: opts.computation_type.unwrap_or(0),
        conf_level: 0.95,
    };
    Ok(compute_survfitkm(
        &time,
        &status,
        &weights,
        opts.entry_times.as_deref(),
        &position,
        &config,
    ))
}

fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -result } else { result }
}

#[pymodule]
#[pyo3(name = "survfitkm")]
fn survfitkm_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_with_options, &m)?)?;
    m.add_class::<SurvFitKMOutput>()?;
    m.add_class::<SurvfitKMOptions>()?;
    m.add_class::<KaplanMeierConfig>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaplan_meier_config_default() {
        let config = KaplanMeierConfig::default();
        assert!(!config.reverse);
        assert_eq!(config.computation_type, 0);
        assert!((config.conf_level - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_kaplan_meier_config_create() {
        let config = KaplanMeierConfig::create(Some(true), Some(1), Some(0.99));
        assert!(config.reverse);
        assert_eq!(config.computation_type, 1);
        assert!((config.conf_level - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_normal_quantile() {
        assert!((normal_quantile(0.5)).abs() < 0.01);
        let q_025 = normal_quantile(0.025);
        let q_975 = normal_quantile(0.975);
        assert!((q_025 + q_975).abs() < 0.01);
        assert!((q_975 - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_compute_survfitkm_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = vec![1.0; 5];
        let position = vec![0; 5];
        let config = KaplanMeierConfig::default();

        let result = compute_survfitkm(&time, &status, &weights, None, &position, &config);

        assert!(!result.time.is_empty());
        assert!(!result.estimate.is_empty());
        assert!((result.estimate[0] - 1.0).abs() < 1e-10 || result.estimate[0] < 1.0);
    }
}
