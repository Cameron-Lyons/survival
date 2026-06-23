use crate::constants::{
    DEFAULT_CONFIDENCE_LEVEL, PARALLEL_THRESHOLD_XLARGE, TIME_EPSILON, exp_ci, normal_ci,
};
use crate::internal::numpy_utils::{
    extract_optional_vec_f64, extract_optional_vec_i32, extract_vec_f64, extract_vec_i32,
};
use crate::internal::statistical::normal_inverse_cdf;
use crate::internal::validation::{
    clamp_probability, validate_binary_f64, validate_binary_i32, validate_finite, validate_length,
    validate_no_nan, validate_non_empty, validate_non_negative,
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
    #[pyo3(get, set)]
    pub timefix: Option<bool>,
}

#[pymethods]
impl SurvfitKMOptions {
    #[new]
    #[pyo3(signature = (weights=None, entry_times=None, position=None, reverse=None, computation_type=None, conf_level=None, conf_type=None, timefix=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: Option<Vec<f64>>,
        entry_times: Option<Vec<f64>>,
        position: Option<Vec<i32>>,
        reverse: Option<bool>,
        computation_type: Option<i32>,
        conf_level: Option<f64>,
        conf_type: Option<String>,
        timefix: Option<bool>,
    ) -> Self {
        Self {
            weights,
            entry_times,
            position,
            reverse,
            computation_type,
            conf_level,
            conf_type,
            timefix,
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

    pub fn with_timefix(mut self_: PyRefMut<'_, Self>, timefix: bool) -> PyRefMut<'_, Self> {
        self_.timefix = Some(timefix);
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
    ) -> PyResult<Self> {
        build_kaplan_meier_config(reverse, computation_type, conf_level, conf_type)
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
    ) -> PyResult<Self> {
        build_kaplan_meier_config(reverse, computation_type, conf_level, conf_type)
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

fn build_kaplan_meier_config(
    reverse: Option<bool>,
    computation_type: Option<i32>,
    conf_level: Option<f64>,
    conf_type: Option<String>,
) -> PyResult<KaplanMeierConfig> {
    let conf_level = conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    validate_conf_level(conf_level)?;

    Ok(KaplanMeierConfig {
        reverse: reverse.unwrap_or(false),
        computation_type: computation_type.unwrap_or(0),
        conf_level,
        conf_type: normalize_conf_type(conf_type.as_deref())?,
    })
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

fn validate_entry_times(time: &[f64], entry_times: &[f64], timefix: bool) -> PyResult<()> {
    validate_length(time.len(), entry_times.len(), "entry_times")?;
    validate_no_nan(entry_times, "entry_times")?;
    validate_finite(entry_times, "entry_times")?;
    validate_non_negative(entry_times, "entry_times")?;

    for (idx, (&entry_time, &exit_time)) in entry_times.iter().zip(time.iter()).enumerate() {
        let invalid = if timefix {
            entry_time >= exit_time - TIME_EPSILON
        } else {
            entry_time >= exit_time
        };
        if invalid {
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

fn entry_before_time(entry_time: f64, time: f64, timefix: bool) -> bool {
    if timefix {
        entry_time < time - TIME_EPSILON
    } else {
        entry_time < time
    }
}

fn same_survfit_time(left: f64, right: f64, timefix: bool) -> bool {
    if timefix {
        (left - right).abs() < TIME_EPSILON
    } else {
        left == right
    }
}

fn validate_counting_survfit_table_inputs(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    id: &[i32],
    weights: &[f64],
    timefix: bool,
) -> PyResult<()> {
    validate_non_empty(start, "start")?;
    validate_length(start.len(), stop.len(), "stop")?;
    validate_length(start.len(), status.len(), "status")?;
    validate_length(start.len(), id.len(), "id")?;
    validate_length(start.len(), weights.len(), "weights")?;
    validate_no_nan(start, "start")?;
    validate_finite(start, "start")?;
    validate_non_negative(start, "start")?;
    validate_no_nan(stop, "stop")?;
    validate_finite(stop, "stop")?;
    validate_non_negative(stop, "stop")?;
    validate_binary_i32(status, "status")?;
    validate_no_nan(weights, "weights")?;
    validate_finite(weights, "weights")?;
    validate_non_negative(weights, "weights")?;

    for (idx, (&entry_time, &exit_time)) in start.iter().zip(stop.iter()).enumerate() {
        let invalid = if timefix {
            entry_time >= exit_time - TIME_EPSILON
        } else {
            entry_time >= exit_time
        };
        if invalid {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "start must be less than stop for observation {}",
                idx
            )));
        }
    }
    Ok(())
}

fn counting_survfit_positions(start: &[f64], stop: &[f64], id: &[i32], timefix: bool) -> Vec<i32> {
    let n = stop.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        id[a]
            .cmp(&id[b])
            .then_with(|| stop[a].total_cmp(&stop[b]))
            .then_with(|| a.cmp(&b))
    });

    let mut positions = vec![0; n];
    for (sorted_idx, &row_idx) in order.iter().enumerate() {
        let current_id = id[row_idx];
        let previous_row = sorted_idx
            .checked_sub(1)
            .and_then(|previous_idx| order.get(previous_idx))
            .copied();
        let next_row = order.get(sorted_idx + 1).copied();

        let mut first = previous_row.is_none_or(|previous| id[previous] != current_id);
        if let Some(previous) = previous_row
            && !first
        {
            first = entry_before_time(stop[previous], start[row_idx], timefix);
        }

        let mut last = next_row.is_none_or(|next| id[next] != current_id);
        if let Some(next) = next_row
            && !last
        {
            last = entry_before_time(stop[row_idx], start[next], timefix);
        }

        positions[row_idx] = if first { 1 } else { 0 } + if last { 2 } else { 0 };
    }
    positions
}

fn counting_survfit_times(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    positions: &[i32],
    include_entry: bool,
    timefix: bool,
) -> Vec<f64> {
    let n = stop.len();
    let sort_stop = sorted_indices_by(stop);
    if !include_entry {
        let mut times = Vec::new();
        for &idx in &sort_stop {
            let should_include = positions[idx] > 1 || status[idx] > 0 || times.is_empty();
            let is_new_time = times
                .last()
                .is_none_or(|&previous| !same_survfit_time(stop[idx], previous, timefix));
            if should_include && is_new_time {
                times.push(stop[idx]);
            }
        }
        return times;
    }

    let sort_start = sorted_indices_by(start);
    let mut times = vec![start[sort_start[0]]];
    let mut current = times[0];
    let mut entry_cursor = 1;
    for &stop_idx in &sort_stop {
        while entry_cursor < n
            && entry_before_time(start[sort_start[entry_cursor]], stop[stop_idx], timefix)
        {
            let start_idx = sort_start[entry_cursor];
            if positions[start_idx] & 1 != 0
                && !same_survfit_time(start[start_idx], current, timefix)
            {
                current = start[start_idx];
                times.push(current);
            }
            entry_cursor += 1;
        }

        if (positions[stop_idx] > 1 || status[stop_idx] > 0)
            && !same_survfit_time(stop[stop_idx], current, timefix)
        {
            current = stop[stop_idx];
            times.push(current);
        }
    }
    times
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CountingSurvfitTables {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_event_count: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub n_censor_count: Vec<f64>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvfitCurveResult {
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
    #[pyo3(get)]
    pub cumhaz: Vec<f64>,
    #[pyo3(get)]
    pub std_chaz: Vec<f64>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<f64>>,
}

pub(crate) fn compute_counting_survfit_tables(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    id: &[i32],
    weights: &[f64],
    include_entry: bool,
    timefix: bool,
) -> CountingSurvfitTables {
    let positions = counting_survfit_positions(start, stop, id, timefix);
    let times = counting_survfit_times(start, stop, status, &positions, include_entry, timefix);
    let sort_start = sorted_indices_by(start);
    let sort_stop = sorted_indices_by(stop);

    let mut n_risk = vec![0.0; times.len()];
    let mut n_event = vec![0.0; times.len()];
    let mut n_event_count = vec![0.0; times.len()];
    let mut n_censor = vec![0.0; times.len()];
    let mut n_censor_count = vec![0.0; times.len()];
    let mut n_enter = include_entry.then(|| vec![0.0; times.len()]);

    let mut stop_cursor = stop.len();
    let mut start_cursor = start.len();
    let mut weighted_risk = 0.0;
    for time_idx in (0..times.len()).rev() {
        let current_time = times[time_idx];
        let mut event_weight = 0.0;
        let mut event_count = 0.0;
        let mut censor_weight = 0.0;
        let mut censor_count = 0.0;

        while stop_cursor > 0
            && !entry_before_time(stop[sort_stop[stop_cursor - 1]], current_time, timefix)
        {
            let row_idx = sort_stop[stop_cursor - 1];
            weighted_risk += weights[row_idx];
            if status[row_idx] > 0 {
                event_count += 1.0;
                event_weight += weights[row_idx];
            } else if positions[row_idx] & 2 != 0 {
                censor_count += 1.0;
                censor_weight += weights[row_idx];
            }
            stop_cursor -= 1;
        }

        let mut enter_weight = 0.0;
        while start_cursor > 0
            && !entry_before_time(start[sort_start[start_cursor - 1]], current_time, timefix)
        {
            let row_idx = sort_start[start_cursor - 1];
            weighted_risk -= weights[row_idx];
            if include_entry
                && positions[row_idx] & 1 != 0
                && same_survfit_time(start[row_idx], current_time, timefix)
            {
                enter_weight += weights[row_idx];
            }
            start_cursor -= 1;
        }

        n_risk[time_idx] = weighted_risk.max(0.0);
        n_event[time_idx] = event_weight;
        n_event_count[time_idx] = event_count;
        n_censor[time_idx] = censor_weight;
        n_censor_count[time_idx] = censor_count;
        if let Some(ref mut entries) = n_enter {
            entries[time_idx] = enter_weight;
        }
    }

    CountingSurvfitTables {
        time: times,
        n_risk,
        n_event,
        n_event_count,
        n_censor,
        n_censor_count,
        n_enter,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_survfit_curve_table_inputs(
    time: &[f64],
    n_risk: &[f64],
    n_event: &[f64],
    n_event_count: &[f64],
    n_censor: &[f64],
    n_censor_count: &[f64],
    n_enter: Option<&[f64]>,
    stype: i32,
    ctype: i32,
) -> PyResult<()> {
    validate_non_empty(time, "time")?;
    validate_length(time.len(), n_risk.len(), "n_risk")?;
    validate_length(time.len(), n_event.len(), "n_event")?;
    validate_length(time.len(), n_event_count.len(), "n_event_count")?;
    validate_length(time.len(), n_censor.len(), "n_censor")?;
    validate_length(time.len(), n_censor_count.len(), "n_censor_count")?;
    if let Some(entries) = n_enter {
        validate_length(time.len(), entries.len(), "n_enter")?;
        validate_no_nan(entries, "n_enter")?;
        validate_finite(entries, "n_enter")?;
        validate_non_negative(entries, "n_enter")?;
    }
    for (values, field) in [
        (time, "time"),
        (n_risk, "n_risk"),
        (n_event, "n_event"),
        (n_event_count, "n_event_count"),
        (n_censor, "n_censor"),
        (n_censor_count, "n_censor_count"),
    ] {
        validate_no_nan(values, field)?;
        validate_finite(values, field)?;
        validate_non_negative(values, field)?;
    }
    if stype != 1 && stype != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "stype must be 1 or 2",
        ));
    }
    if ctype != 1 && ctype != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ctype must be 1 or 2",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_survfit_curve_from_tables(
    time: &[f64],
    n_risk: &[f64],
    n_event: &[f64],
    n_event_count: &[f64],
    n_censor: &[f64],
    n_censor_count: &[f64],
    n_enter: Option<&[f64]>,
    reverse: bool,
    stype: i32,
    ctype: i32,
    conf_level: f64,
    conf_type: &str,
) -> SurvfitCurveResult {
    let mut current_survival = 1.0;
    let mut greenwood_variance = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut cumulative_hazard_variance = 0.0;
    let alpha = 1.0 - conf_level;
    let z = normal_inverse_cdf(1.0 - alpha / 2.0);

    let mut estimate = Vec::with_capacity(time.len());
    let mut std_err = Vec::with_capacity(time.len());
    let mut cumhaz = Vec::with_capacity(time.len());
    let mut std_chaz = Vec::with_capacity(time.len());
    let mut conf_lower = if conf_type == "none" {
        Vec::new()
    } else {
        Vec::with_capacity(time.len())
    };
    let mut conf_upper = if conf_type == "none" {
        Vec::new()
    } else {
        Vec::with_capacity(time.len())
    };

    for idx in 0..time.len() {
        let event_weight = if reverse { n_censor[idx] } else { n_event[idx] };
        let event_count = if reverse {
            n_censor_count[idx]
        } else {
            n_event_count[idx]
        };
        let risk_for_curve = if reverse {
            n_risk[idx] - n_event[idx]
        } else {
            n_risk[idx]
        };

        if event_weight > 0.0 && event_count > 0.0 && risk_for_curve > 0.0 {
            if ctype == 1 {
                cumulative_hazard += event_weight / risk_for_curve;
                cumulative_hazard_variance += event_weight / (risk_for_curve * risk_for_curve);
            } else {
                for step in 0..event_count as usize {
                    let denominator = risk_for_curve - step as f64 * event_weight / event_count;
                    if denominator > 0.0 {
                        cumulative_hazard += event_weight / (event_count * denominator);
                        cumulative_hazard_variance +=
                            event_weight / (event_count * denominator * denominator);
                    }
                }
            }
        }

        let (survival, survival_se) = if stype == 1 {
            if event_weight > 0.0 && event_count > 0.0 && risk_for_curve > 0.0 {
                current_survival *= ((risk_for_curve - event_weight) / risk_for_curve).max(0.0);
                if risk_for_curve > event_weight {
                    greenwood_variance +=
                        event_weight / (risk_for_curve * (risk_for_curve - event_weight));
                }
            }
            (
                current_survival,
                current_survival * greenwood_variance.max(0.0_f64).sqrt(),
            )
        } else {
            let survival = (-cumulative_hazard).exp();
            (
                survival,
                survival * cumulative_hazard_variance.max(0.0_f64).sqrt(),
            )
        };

        estimate.push(survival);
        std_err.push(survival_se);
        cumhaz.push(cumulative_hazard);
        std_chaz.push(cumulative_hazard_variance.max(0.0_f64).sqrt());
        if conf_type != "none" {
            let (lower, upper) = compute_confidence_interval(survival, survival_se, z, conf_type);
            conf_lower.push(lower);
            conf_upper.push(upper);
        }
    }

    SurvfitCurveResult {
        time: time.to_vec(),
        n_risk: n_risk.to_vec(),
        n_event: n_event.to_vec(),
        n_censor: n_censor.to_vec(),
        estimate,
        std_err,
        conf_lower,
        conf_upper,
        cumhaz,
        std_chaz,
        n_enter: n_enter.map(|values| values.to_vec()),
    }
}

#[pyfunction]
#[pyo3(signature = (time, n_risk, n_event, n_event_count, n_censor, n_censor_count, n_enter=None, reverse=false, stype=1, ctype=1, conf_level=None, conf_type=None))]
#[allow(clippy::too_many_arguments)]
pub fn survfit_curve_from_tables(
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_event_count: Vec<f64>,
    n_censor: Vec<f64>,
    n_censor_count: Vec<f64>,
    n_enter: Option<Vec<f64>>,
    reverse: bool,
    stype: i32,
    ctype: i32,
    conf_level: Option<f64>,
    conf_type: Option<String>,
) -> PyResult<SurvfitCurveResult> {
    let conf_level = conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    validate_conf_level(conf_level)?;
    let conf_type = normalize_conf_type(conf_type.as_deref())?;
    validate_survfit_curve_table_inputs(
        &time,
        &n_risk,
        &n_event,
        &n_event_count,
        &n_censor,
        &n_censor_count,
        n_enter.as_deref(),
        stype,
        ctype,
    )?;
    Ok(compute_survfit_curve_from_tables(
        &time,
        &n_risk,
        &n_event,
        &n_event_count,
        &n_censor,
        &n_censor_count,
        n_enter.as_deref(),
        reverse,
        stype,
        ctype,
        conf_level,
        &conf_type,
    ))
}

#[pyfunction]
#[pyo3(signature = (start, stop, status, id, weights=None, include_entry=false, timefix=None))]
pub fn counting_survfit_tables(
    start: &Bound<'_, PyAny>,
    stop: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    id: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
    include_entry: bool,
    timefix: Option<bool>,
) -> PyResult<CountingSurvfitTables> {
    let start = extract_vec_f64(start)?;
    let stop = extract_vec_f64(stop)?;
    let status = extract_vec_i32(status)?;
    let id = extract_vec_i32(id)?;
    let weights = match extract_optional_vec_f64(weights)? {
        Some(w) => w,
        None => vec![1.0; start.len()],
    };
    let timefix = timefix.unwrap_or(true);
    validate_counting_survfit_table_inputs(&start, &stop, &status, &id, &weights, timefix)?;
    Ok(compute_counting_survfit_tables(
        &start,
        &stop,
        &status,
        &id,
        &weights,
        include_entry,
        timefix,
    ))
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
    pub n_event_count: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub n_censor_count: Vec<f64>,
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
#[pyo3(signature = (time, status, weights=None, entry_times=None, position=None, reverse=None, computation_type=None, conf_level=None, conf_type=None, timefix=None))]
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
    timefix: Option<bool>,
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
    let timefix = timefix.unwrap_or(true);
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
        validate_entry_times(&time, entry, timefix)?;
    }
    Ok(compute_survfitkm_with_timefix(
        &time,
        &status,
        &weights,
        entry_times_opt.as_deref(),
        &position,
        &config,
        timefix,
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
    compute_survfitkm_with_timefix(time, status, weights, entry_times, _position, config, true)
}

pub fn compute_survfitkm_with_timefix(
    time: &[f64],
    status: &[f64],
    weights: &[f64],
    entry_times: Option<&[f64]>,
    _position: &[i32],
    config: &KaplanMeierConfig,
    timefix: bool,
) -> SurvFitKMOutput {
    let n = time.len();
    let indices = sorted_indices_by(time);
    let entry_indices = entry_times.map(sorted_indices_by);
    let mut entry_cursor = 0;
    let estimated_events = (n / 10).max(16);
    let mut event_times = Vec::with_capacity(estimated_events);
    let mut n_risk_vec = Vec::with_capacity(estimated_events);
    let mut n_event_vec = Vec::with_capacity(estimated_events);
    let mut n_event_count_vec = Vec::with_capacity(estimated_events);
    let mut n_censor_vec = Vec::with_capacity(estimated_events);
    let mut n_censor_count_vec = Vec::with_capacity(estimated_events);
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
                && entry_before_time(entry[sorted_entries[entry_cursor]], current_time, timefix)
            {
                current_risk += weights[sorted_entries[entry_cursor]];
                entry_cursor += 1;
            }
        }
        let mut weighted_events = 0.0;
        let mut event_count = 0.0;
        let mut weighted_censor = 0.0;
        let mut censor_count = 0.0;
        let mut j = i;
        while j < n && same_survfit_time(time[indices[j]], current_time, timefix) {
            let idx = indices[j];
            let is_event = if config.reverse {
                status[idx] <= 0.0
            } else {
                status[idx] > 0.0
            };
            if is_event {
                weighted_events += weights[idx];
                event_count += 1.0;
            } else {
                weighted_censor += weights[idx];
                censor_count += 1.0;
            }
            j += 1;
        }
        if weighted_events > 0.0 || weighted_censor > 0.0 {
            let risk_at_time = current_risk;
            event_times.push(current_time);
            n_risk_vec.push(risk_at_time);
            n_event_vec.push(weighted_events);
            n_event_count_vec.push(event_count);
            n_censor_vec.push(weighted_censor);
            n_censor_count_vec.push(censor_count);
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
        n_event_count: n_event_count_vec,
        n_censor: n_censor_vec,
        n_censor_count: n_censor_count_vec,
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
    let timefix = opts.timefix.unwrap_or(true);
    if let Some(ref entry) = opts.entry_times {
        validate_entry_times(&time, entry, timefix)?;
    }
    let config = KaplanMeierConfig {
        reverse: opts.reverse.unwrap_or(false),
        computation_type: opts.computation_type.unwrap_or(0),
        conf_level: opts.conf_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL),
        conf_type: normalize_conf_type(opts.conf_type.as_deref())?,
    };
    validate_conf_level(config.conf_level)?;
    Ok(compute_survfitkm_with_timefix(
        &time,
        &status,
        &weights,
        opts.entry_times.as_deref(),
        &position,
        &config,
        timefix,
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
            KaplanMeierConfig::create(Some(true), Some(1), Some(0.99), Some("plain".to_string()))
                .unwrap();
        assert!(config.reverse);
        assert_eq!(config.computation_type, 1);
        assert!((config.conf_level - 0.99).abs() < 1e-10);
        assert_eq!(config.conf_type, "plain");
    }

    #[test]
    fn test_kaplan_meier_config_validates_confidence_options() {
        assert!(KaplanMeierConfig::new(None, None, Some(1.0), None).is_err());
        assert!(KaplanMeierConfig::new(None, None, Some(f64::NAN), None).is_err());
        assert!(KaplanMeierConfig::new(None, None, None, Some("weird".to_string())).is_err());
        assert!(KaplanMeierConfig::create(None, None, Some(-0.1), None).is_err());

        let config = KaplanMeierConfig::new(None, None, None, Some("log_log".to_string())).unwrap();
        assert_eq!(config.conf_type, "log-log");
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
        assert_eq!(result.n_event_count, vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        assert_eq!(result.n_censor_count, vec![0.0, 1.0, 0.0, 1.0, 0.0]);
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
        assert_eq!(result.n_event_count, vec![2.0, 0.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0]);
        assert_eq!(result.n_censor_count, vec![0.0, 1.0]);
        assert!((result.estimate[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_survfitkm_exact_timefix_false() {
        let time = vec![1.0, 1.0 + 5e-10, 2.0];
        let status = vec![1.0, 1.0, 0.0];
        let weights = vec![1.0; 3];
        let position = vec![0; 3];
        let config = KaplanMeierConfig::default();

        let result = compute_survfitkm_with_timefix(
            &time, &status, &weights, None, &position, &config, false,
        );

        assert_eq!(result.time, time);
        assert_eq!(result.n_risk, vec![3.0, 2.0, 1.0]);
        assert_eq!(result.n_event, vec![1.0, 1.0, 0.0]);
        assert_eq!(result.n_event_count, vec![1.0, 1.0, 0.0]);
        assert_eq!(result.n_censor, vec![0.0, 0.0, 1.0]);
        assert_eq!(result.n_censor_count, vec![0.0, 0.0, 1.0]);
        assert!((result.estimate[0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((result.estimate[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((result.estimate[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_survfitkm_reports_unweighted_counts_for_weighted_ties() {
        let time = vec![1.0, 1.0, 2.0];
        let status = vec![1.0, 1.0, 1.0];
        let weights = vec![2.0, 1.0, 1.0];
        let position = vec![0; 3];
        let config = KaplanMeierConfig::default();

        let result = compute_survfitkm(&time, &status, &weights, None, &position, &config);

        assert_eq!(result.time, vec![1.0, 2.0]);
        assert_eq!(result.n_event, vec![3.0, 1.0]);
        assert_eq!(result.n_event_count, vec![2.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 0.0]);
        assert_eq!(result.n_censor_count, vec![0.0, 0.0]);
    }

    #[test]
    fn test_survfit_curve_from_tables_uses_unweighted_counts_for_fh2() {
        let result = compute_survfit_curve_from_tables(
            &[1.0, 2.0],
            &[4.0, 1.0],
            &[3.0, 1.0],
            &[2.0, 1.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            None,
            false,
            2,
            2,
            DEFAULT_CONFIDENCE_LEVEL,
            "log",
        );

        let first_hazard = 3.0 / (2.0 * 4.0) + 3.0 / (2.0 * 2.5);
        assert!((result.cumhaz[0] - first_hazard).abs() < 1e-10);
        assert!((result.cumhaz[1] - (first_hazard + 1.0)).abs() < 1e-10);
        assert!((result.estimate[0] - (-first_hazard).exp()).abs() < 1e-10);
        assert!((result.estimate[1] - (-(first_hazard + 1.0)).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_curve_from_tables_handles_reverse_counts() {
        let result = compute_survfit_curve_from_tables(
            &[1.0, 2.0],
            &[3.0, 2.0],
            &[1.0, 0.0],
            &[1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
            Some(&[0.0, 0.0]),
            true,
            1,
            1,
            DEFAULT_CONFIDENCE_LEVEL,
            "none",
        );

        assert_eq!(result.n_event, vec![1.0, 0.0]);
        assert_eq!(result.n_censor, vec![0.0, 1.0]);
        assert_eq!(result.n_enter, Some(vec![0.0, 0.0]));
        assert_eq!(result.conf_lower, Vec::<f64>::new());
        assert_eq!(result.conf_upper, Vec::<f64>::new());
        assert!((result.estimate[0] - 1.0).abs() < 1e-10);
        assert!((result.estimate[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_counting_survfit_tables_with_id_entry_counts() {
        let start = vec![0.0, 10.0, 25.0, 0.0, 5.0];
        let stop = vec![10.0, 20.0, 30.0, 15.0, 25.0];
        let status = vec![0, 0, 1, 1, 0];
        let id = vec![0, 0, 0, 1, 2];
        let weights = vec![1.0; 5];

        let tables =
            compute_counting_survfit_tables(&start, &stop, &status, &id, &weights, true, true);

        assert_eq!(tables.time, vec![0.0, 5.0, 15.0, 20.0, 25.0, 30.0]);
        assert_eq!(tables.n_risk, vec![0.0, 2.0, 3.0, 2.0, 1.0, 1.0]);
        assert_eq!(tables.n_event, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(tables.n_event_count, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        assert_eq!(tables.n_censor, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(tables.n_censor_count, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(tables.n_enter, Some(vec![2.0, 1.0, 0.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_counting_survfit_tables_honor_exact_timefix_false() {
        let start = vec![0.0, 0.0, 1.0, 1.0 + 5e-10];
        let stop = vec![1.0, 1.0 + 5e-10, 2.0, 2.0];
        let status = vec![1, 1, 0, 0];
        let id = vec![0, 1, 2, 3];
        let weights = vec![1.0; 4];

        let default_tables =
            compute_counting_survfit_tables(&start, &stop, &status, &id, &weights, false, true);
        let exact_tables =
            compute_counting_survfit_tables(&start, &stop, &status, &id, &weights, false, false);

        assert_eq!(default_tables.time, vec![1.0, 2.0]);
        assert_eq!(default_tables.n_risk, vec![2.0, 2.0]);
        assert_eq!(default_tables.n_event, vec![2.0, 0.0]);
        assert_eq!(exact_tables.time, vec![1.0, 1.0 + 5e-10, 2.0]);
        assert_eq!(exact_tables.n_risk, vec![2.0, 2.0, 2.0]);
        assert_eq!(exact_tables.n_event, vec![1.0, 1.0, 0.0]);
        assert_eq!(exact_tables.n_censor, vec![0.0, 0.0, 2.0]);
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
        assert_eq!(result.n_event_count, vec![1.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(result.n_censor_count, vec![0.0, 0.0, 1.0, 1.0]);
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
        let config = KaplanMeierConfig::create(Some(true), None, None, None).unwrap();

        let result = compute_survfitkm(&time, &status, &weights, None, &position, &config);

        assert_eq!(result.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.n_risk, vec![5.0, 4.0, 2.0, 1.0]);
        assert_eq!(result.n_event, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(result.n_event_count, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(result.n_censor, vec![1.0, 1.0, 0.0, 1.0]);
        assert_eq!(result.n_censor_count, vec![1.0, 1.0, 0.0, 1.0]);
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

        let err = validate_entry_times(&time, &entry_times, true)
            .expect_err("non-finite entry times should be rejected");

        assert!(err.to_string().contains("entry_times contains non-finite"));
    }
}
