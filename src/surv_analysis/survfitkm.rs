use crate::constants::{
    DEFAULT_CONFIDENCE_LEVEL, PARALLEL_THRESHOLD_XLARGE, TIME_EPSILON, exp_ci, normal_ci,
};
use crate::internal::numpy_utils::{
    extract_optional_vec_f64, extract_optional_vec_i32, extract_vec_f64,
};
use crate::internal::statistical::normal_inverse_cdf;
use crate::internal::validation::{
    clamp_probability, validate_binary_f64, validate_finite, validate_length, validate_no_nan,
    validate_non_empty, validate_non_negative,
};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
#[pyclass(from_py_object)]
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
    #[pyo3(get, set)]
    pub conf_level: Option<f64>,
    #[pyo3(get, set)]
    pub conf_type: Option<String>,
}

#[pymethods]
impl SurvfitKMOptions {
    #[new]
    #[pyo3(signature = (weights=None, entry_times=None, position=None, reverse=None, computation_type=None, conf_level=None, conf_type=None))]
    pub fn new(
        weights: Option<Vec<f64>>,
        entry_times: Option<Vec<f64>>,
        position: Option<Vec<i32>>,
        reverse: Option<bool>,
        computation_type: Option<i32>,
        conf_level: Option<f64>,
        conf_type: Option<String>,
    ) -> Self {
        Self {
            weights,
            entry_times,
            position,
            reverse,
            computation_type,
            conf_level,
            conf_type,
        }
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

    pub fn with_conf_level(mut self_: PyRefMut<'_, Self>, conf_level: f64) -> PyRefMut<'_, Self> {
        self_.conf_level = Some(conf_level);
        self_
    }

    pub fn with_conf_type(mut self_: PyRefMut<'_, Self>, conf_type: String) -> PyRefMut<'_, Self> {
        self_.conf_type = Some(conf_type);
        self_
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct KaplanMeierConfig {
    #[pyo3(get, set)]
    pub reverse: bool,

    #[pyo3(get, set)]
    pub computation_type: i32,

    #[pyo3(get, set)]
    pub conf_level: f64,

    #[pyo3(get, set)]
    pub conf_type: String,
}

#[pymethods]
impl KaplanMeierConfig {
    #[new]
    #[pyo3(signature = (reverse=None, computation_type=None, conf_level=None, conf_type=None))]
    fn new(
        reverse: Option<bool>,
        computation_type: Option<i32>,
        conf_level: Option<f64>,
        conf_type: Option<String>,
    ) -> Self {
        Self {
            reverse: reverse.unwrap_or(false),
            computation_type: computation_type.unwrap_or(0),
            conf_level: conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL),
            conf_type: conf_type.unwrap_or_else(|| "log".to_string()),
        }
    }
}

impl Default for KaplanMeierConfig {
    fn default() -> Self {
        Self {
            reverse: false,
            computation_type: 0,
            conf_level: DEFAULT_CONFIDENCE_LEVEL,
            conf_type: "log".to_string(),
        }
    }
}

impl KaplanMeierConfig {
    pub fn create(
        reverse: Option<bool>,
        computation_type: Option<i32>,
        conf_level: Option<f64>,
        conf_type: Option<String>,
    ) -> Self {
        Self {
            reverse: reverse.unwrap_or(false),
            computation_type: computation_type.unwrap_or(0),
            conf_level: conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL),
            conf_type: conf_type.unwrap_or_else(|| "log".to_string()),
        }
    }
}

fn validate_conf_level(conf_level: f64) -> PyResult<()> {
    if !(0.0..1.0).contains(&conf_level) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "conf_level must be between 0 and 1",
        ));
    }
    Ok(())
}

fn normalize_conf_type(conf_type: Option<&str>) -> PyResult<String> {
    let normalized = conf_type
        .unwrap_or("log")
        .to_ascii_lowercase()
        .replace('_', "-");
    match normalized.as_str() {
        "plain" | "log" | "logit" | "arcsin" | "none" => Ok(normalized),
        "log-log" | "loglog" => Ok("log-log".to_string()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "conf_type must be 'plain', 'log', 'log-log', 'logit', 'arcsin', or 'none'",
        )),
    }
}

fn compute_confidence_interval(survival: f64, std_err: f64, z: f64, conf_type: &str) -> (f64, f64) {
    if std_err <= 0.0 || survival <= 0.0 || survival >= 1.0 {
        let bounded = clamp_probability(survival);
        return (bounded, bounded);
    }

    match conf_type {
        "plain" => {
            let (lower, upper) = normal_ci(survival, std_err, z);
            (clamp_probability(lower), clamp_probability(upper))
        }
        "log" => {
            let log_survival = survival.ln();
            let log_std_err = std_err / survival;
            let (lower, upper) = exp_ci(log_survival, log_std_err, z);
            (clamp_probability(lower), clamp_probability(upper))
        }
        "log-log" => {
            let log_survival = survival.ln();
            let transformed_std_err = z * (std_err / survival) / log_survival;
            let log_neg_log_survival = (-log_survival).ln();
            (
                clamp_probability((-((log_neg_log_survival - transformed_std_err).exp())).exp()),
                clamp_probability((-((log_neg_log_survival + transformed_std_err).exp())).exp()),
            )
        }
        "logit" => {
            let logit_survival = (survival / (1.0 - survival)).ln();
            let transformed_std_err = z * std_err / (survival * (1.0 - survival));
            (
                clamp_probability(1.0 - 1.0 / (1.0 + (logit_survival - transformed_std_err).exp())),
                clamp_probability(1.0 - 1.0 / (1.0 + (logit_survival + transformed_std_err).exp())),
            )
        }
        "arcsin" => {
            let angle = survival.sqrt().asin();
            let transformed_std_err = 0.5 * z * std_err / (survival * (1.0 - survival)).sqrt();
            (
                clamp_probability((angle - transformed_std_err).max(0.0).sin().powi(2)),
                clamp_probability(
                    (angle + transformed_std_err)
                        .min(std::f64::consts::FRAC_PI_2)
                        .sin()
                        .powi(2),
                ),
            )
        }
        _ => unreachable!("conf_type is validated before confidence intervals are computed"),
    }
}

fn validate_entry_times(time: &[f64], entry_times: &[f64]) -> PyResult<()> {
    validate_length(time.len(), entry_times.len(), "entry_times")?;
    validate_no_nan(entry_times, "entry_times")?;
    validate_finite(entry_times, "entry_times")?;
    validate_non_negative(entry_times, "entry_times")?;

    for (idx, (&entry_time, &exit_time)) in entry_times.iter().zip(time.iter()).enumerate() {
        if entry_time >= exit_time - TIME_EPSILON {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "entry_times must be less than time for observation {}",
                idx
            )));
        }
    }
    Ok(())
}

fn sorted_indices_by(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    if values.len() > PARALLEL_THRESHOLD_XLARGE {
        indices.par_sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));
    } else {
        indices.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));
    }
    indices
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
    pub cumhaz: Vec<f64>,
    #[pyo3(get)]
    pub std_chaz: Vec<f64>,
    #[pyo3(get)]
    pub conf_lower: Vec<f64>,
    #[pyo3(get)]
    pub conf_upper: Vec<f64>,
}

#[pymethods]
impl SurvFitKMOutput {
    #[getter]
    pub fn cumulative_hazard(&self) -> Vec<f64> {
        self.cumhaz.clone()
    }

    #[getter]
    pub fn cumulative_hazard_std_err(&self) -> Vec<f64> {
        self.std_chaz.clone()
    }
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
/// conf_type : str, optional
///     Confidence interval transform: plain, log, log-log, logit, arcsin, or none.
///
/// Returns
/// -------
/// SurvFitKMOutput
///     Object containing: time (event/censor times), n_risk (at-risk counts), n_event (event counts),
///     estimate (survival probabilities), std_err (standard errors), cumhaz/std_chaz, and
///     conf_lower/conf_upper intervals.
///
/// Examples
/// --------
/// >>> import survival
/// >>> import pandas as pd
/// >>> df = pd.DataFrame({'time': [1, 2, 3, 4, 5], 'status': [1, 0, 1, 1, 0]})
/// >>> result = survival.survfitkm(df['time'], df['status'])
/// >>> result.estimate
#[pyfunction]
#[pyo3(signature = (time, status, weights=None, entry_times=None, position=None, reverse=None, computation_type=None, conf_level=None, conf_type=None))]
#[allow(clippy::too_many_arguments)]
pub fn survfitkm(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
    entry_times: Option<&Bound<'_, PyAny>>,
    position: Option<&Bound<'_, PyAny>>,
    reverse: Option<bool>,
    computation_type: Option<i32>,
    conf_level: Option<f64>,
    conf_type: Option<String>,
) -> PyResult<SurvFitKMOutput> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_f64(status)?;
    let weights_opt = extract_optional_vec_f64(weights)?;
    let entry_times_opt = extract_optional_vec_f64(entry_times)?;
    let position_opt = extract_optional_vec_i32(position)?;
    let config = KaplanMeierConfig {
        reverse: reverse.unwrap_or(false),
        computation_type: computation_type.unwrap_or(0),
        conf_level: conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL),
        conf_type: normalize_conf_type(conf_type.as_deref())?,
    };
    validate_conf_level(config.conf_level)?;
    validate_non_empty(&time, "time")?;
    validate_length(time.len(), status.len(), "status")?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&status, "status")?;
    validate_finite(&status, "status")?;
    validate_binary_f64(&status, "status")?;
    let weights = match weights_opt {
        Some(w) => {
            validate_length(time.len(), w.len(), "weights")?;
            validate_no_nan(&w, "weights")?;
            validate_finite(&w, "weights")?;
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
        validate_entry_times(&time, entry)?;
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
    entry_times: Option<&[f64]>,
    _position: &[i32],
    config: &KaplanMeierConfig,
) -> SurvFitKMOutput {
    let n = time.len();
    let indices = sorted_indices_by(time);
    let entry_indices = entry_times.map(sorted_indices_by);
    let mut entry_cursor = 0;
    let estimated_events = (n / 10).max(16);
    let mut event_times = Vec::with_capacity(estimated_events);
    let mut n_risk_vec = Vec::with_capacity(estimated_events);
    let mut n_event_vec = Vec::with_capacity(estimated_events);
    let mut n_censor_vec = Vec::with_capacity(estimated_events);
    let mut estimate_vec = Vec::with_capacity(estimated_events);
    let mut std_err_vec = Vec::with_capacity(estimated_events);
    let mut cumhaz_vec = Vec::with_capacity(estimated_events);
    let mut std_chaz_vec = Vec::with_capacity(estimated_events);
    let mut current_risk: f64 = if entry_times.is_some() {
        0.0
    } else {
        weights.iter().sum()
    };
    let mut current_estimate = 1.0;
    let mut cumulative_variance = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut cumulative_hazard_variance: f64 = 0.0;
    let mut i = 0;

    while i < n {
        let current_time = time[indices[i]];
        if let (Some(entry), Some(sorted_entries)) = (entry_times, entry_indices.as_ref()) {
            while entry_cursor < n
                && entry[sorted_entries[entry_cursor]] < current_time - TIME_EPSILON
            {
                current_risk += weights[sorted_entries[entry_cursor]];
                entry_cursor += 1;
            }
        }
        let mut weighted_events = 0.0;
        let mut weighted_censor = 0.0;
        let mut j = i;
        while j < n && (time[indices[j]] - current_time).abs() < TIME_EPSILON {
            let idx = indices[j];
            let is_event = if config.reverse {
                status[idx] <= 0.0
            } else {
                status[idx] > 0.0
            };
            if is_event {
                weighted_events += weights[idx];
            } else {
                weighted_censor += weights[idx];
            }
            j += 1;
        }
        if weighted_events > 0.0 || weighted_censor > 0.0 {
            let risk_at_time = current_risk;
            event_times.push(current_time);
            n_risk_vec.push(risk_at_time);
            n_event_vec.push(weighted_events);
            n_censor_vec.push(weighted_censor);
            if weighted_events > 0.0 && risk_at_time > 0.0 {
                let hazard = weighted_events / risk_at_time;
                cumulative_hazard += hazard;
                cumulative_hazard_variance += weighted_events / (risk_at_time * risk_at_time);
                current_estimate *= 1.0 - hazard;
                if risk_at_time > weighted_events {
                    cumulative_variance +=
                        weighted_events / (risk_at_time * (risk_at_time - weighted_events));
                }
            }
            estimate_vec.push(current_estimate);
            let se = current_estimate * cumulative_variance.sqrt();
            std_err_vec.push(se);
            cumhaz_vec.push(cumulative_hazard);
            std_chaz_vec.push(cumulative_hazard_variance.sqrt());
        }
        current_risk -= weighted_events + weighted_censor;
        i = j;
    }

    let alpha = 1.0 - config.conf_level;
    let z = normal_inverse_cdf(1.0 - alpha / 2.0);

    let (conf_lower, conf_upper): (Vec<f64>, Vec<f64>) = if config.conf_type == "none" {
        (vec![], vec![])
    } else {
        estimate_vec
            .iter()
            .zip(std_err_vec.iter())
            .map(|(&s, &se)| compute_confidence_interval(s, se, z, &config.conf_type))
            .unzip()
    };
    SurvFitKMOutput {
        time: event_times,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_censor: n_censor_vec,
        estimate: estimate_vec,
        std_err: std_err_vec,
        cumhaz: cumhaz_vec,
        std_chaz: std_chaz_vec,
        conf_lower,
        conf_upper,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, options=None))]
pub fn survfitkm_with_options(
    time: Vec<f64>,
    status: Vec<f64>,
    options: Option<&SurvfitKMOptions>,
) -> PyResult<SurvFitKMOutput> {
    let opts = options.cloned().unwrap_or_default();
    validate_non_empty(&time, "time")?;
    validate_length(time.len(), status.len(), "status")?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&status, "status")?;
    validate_finite(&status, "status")?;
    validate_binary_f64(&status, "status")?;
    let weights = match opts.weights {
        Some(w) => {
            validate_length(time.len(), w.len(), "weights")?;
            validate_no_nan(&w, "weights")?;
            validate_finite(&w, "weights")?;
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
        validate_entry_times(&time, entry)?;
    }
    let config = KaplanMeierConfig {
        reverse: opts.reverse.unwrap_or(false),
        computation_type: opts.computation_type.unwrap_or(0),
        conf_level: opts.conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL),
        conf_type: normalize_conf_type(opts.conf_type.as_deref())?,
    };
    validate_conf_level(config.conf_level)?;
    Ok(compute_survfitkm(
        &time,
        &status,
        &weights,
        opts.entry_times.as_deref(),
        &position,
        &config,
    ))
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
        assert_eq!(config.conf_type, "log");
    }

    #[test]
    fn test_kaplan_meier_config_create() {
        let config =
            KaplanMeierConfig::create(Some(true), Some(1), Some(0.99), Some("plain".to_string()));
        assert!(config.reverse);
        assert_eq!(config.computation_type, 1);
        assert!((config.conf_level - 0.99).abs() < 1e-10);
        assert_eq!(config.conf_type, "plain");
    }

    #[test]
    fn test_normal_quantile() {
        assert!((normal_inverse_cdf(0.5)).abs() < 0.01);
        let q_025 = normal_inverse_cdf(0.025);
        let q_975 = normal_inverse_cdf(0.975);
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
        assert_eq!(result.time, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.n_event, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(result.cumhaz.len(), result.time.len());
        assert_eq!(result.std_chaz.len(), result.time.len());
        assert!((result.estimate[0] - 1.0).abs() < 1e-10 || result.estimate[0] < 1.0);
    }

    #[test]
    fn test_compute_survfitkm_groups_near_tied_times() {
        let time = vec![1.0 + TIME_EPSILON / 2.0, 2.0, 1.0];
        let status = vec![1.0, 0.0, 1.0];
        let weights = vec![1.0; 3];
        let position = vec![0; 3];
        let config = KaplanMeierConfig::default();

        let result = compute_survfitkm(&time, &status, &weights, None, &position, &config);

        assert_eq!(result.time, vec![1.0, 2.0]);
        assert_eq!(result.n_risk, vec![3.0, 1.0]);
        assert_eq!(result.n_event, vec![2.0, 0.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0]);
        assert!((result.estimate[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_binary_f64_rejects_non_binary_values() {
        pyo3::Python::initialize();

        let err = validate_binary_f64(&[0.0, 0.5, 1.0], "status")
            .expect_err("non-binary status should be rejected");

        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn test_compute_survfitkm_delayed_entry() {
        let entry_times = vec![0.0, 0.0, 1.0, 2.0, 3.0];
        let time = vec![2.0, 4.0, 3.0, 5.0, 5.0];
        let status = vec![1.0, 0.0, 1.0, 1.0, 0.0];
        let weights = vec![1.0; 5];
        let position = vec![0; 5];
        let config = KaplanMeierConfig::default();

        let result = compute_survfitkm(
            &time,
            &status,
            &weights,
            Some(&entry_times),
            &position,
            &config,
        );

        assert_eq!(result.time, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.n_risk, vec![3.0, 3.0, 3.0, 2.0]);
        assert_eq!(result.n_event, vec![1.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 0.0, 1.0, 1.0]);
        assert!((result.estimate[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result.estimate[1] - 4.0 / 9.0).abs() < 1e-10);
        assert!((result.estimate[2] - 4.0 / 9.0).abs() < 1e-10);
        assert!((result.estimate[3] - 2.0 / 9.0).abs() < 1e-10);
        assert!((result.cumhaz[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((result.cumhaz[1] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result.cumhaz[2] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result.cumhaz[3] - 7.0 / 6.0).abs() < 1e-10);
        assert!((result.std_chaz[0] - (1.0_f64 / 9.0).sqrt()).abs() < 1e-10);
        assert!((result.std_chaz[1] - (2.0_f64 / 9.0).sqrt()).abs() < 1e-10);
        assert!((result.std_chaz[2] - (2.0_f64 / 9.0).sqrt()).abs() < 1e-10);
        assert!((result.std_chaz[3] - (2.0_f64 / 9.0 + 1.0 / 4.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_survfitkm_reverse_censoring_distribution() {
        let time = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let status = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = vec![1.0; 5];
        let position = vec![0; 5];
        let config = KaplanMeierConfig::create(Some(true), None, None, None);

        let result = compute_survfitkm(&time, &status, &weights, None, &position, &config);

        assert_eq!(result.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.n_risk, vec![5.0, 4.0, 2.0, 1.0]);
        assert_eq!(result.n_event, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(result.n_censor, vec![1.0, 1.0, 0.0, 1.0]);
        assert!((result.estimate[0] - 1.0).abs() < 1e-10);
        assert!((result.estimate[1] - 0.75).abs() < 1e-10);
        assert!((result.estimate[2] - 0.375).abs() < 1e-10);
        assert!((result.estimate[3] - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_validate_entry_times_rejects_non_finite_values() {
        pyo3::Python::initialize();
        let time = vec![1.0, 2.0];
        let entry_times = vec![0.0, f64::INFINITY];

        let err = validate_entry_times(&time, &entry_times)
            .expect_err("non-finite entry times should be rejected");

        assert!(err.to_string().contains("entry_times contains non-finite"));
    }
}
