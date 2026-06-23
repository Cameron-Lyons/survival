use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

use crate::constants::same_time;

const PY_EXP_CLAMP_MIN: f64 = -745.0;
const PY_EXP_CLAMP_MAX: f64 = 709.0;

type CoxBaselineCurveOutput = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<PyValueError, _>(message.into())
}

fn validate_finite_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_finite_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "{name} must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_step_times(name: &str, times: &[f64]) -> PyResult<()> {
    validate_finite_slice(name, times)?;
    if times.windows(2).any(|window| window[1] < window[0]) {
        return Err(value_error(format!("{name} must be sorted ascending")));
    }
    Ok(())
}

fn step_value_at_with_initial(
    times: &[f64],
    values: &[f64],
    requested_time: f64,
    initial: f64,
) -> f64 {
    let pos = times.partition_point(|&time| time <= requested_time);
    if pos == 0 { initial } else { values[pos - 1] }
}

fn requested_times_are_sorted(values: &[f64]) -> bool {
    values.windows(2).all(|window| window[1] >= window[0])
}

fn step_values_at_sorted_requests(
    times: &[f64],
    values: &[f64],
    requested_times: &[f64],
    initial: f64,
) -> Vec<f64> {
    let mut cursor = 0;
    let mut output = Vec::with_capacity(requested_times.len());
    for &requested_time in requested_times {
        while cursor < times.len() && times[cursor] <= requested_time {
            cursor += 1;
        }
        output.push(if cursor == 0 {
            initial
        } else {
            values[cursor - 1]
        });
    }
    output
}

#[pyfunction]
pub fn step_values_at(
    times: Vec<f64>,
    values: Vec<f64>,
    requested_times: Vec<f64>,
    initial: f64,
) -> PyResult<Vec<f64>> {
    if times.len() != values.len() {
        return Err(value_error("times and values must have the same length"));
    }
    validate_step_times("times", &times)?;
    validate_finite_slice("values", &values)?;
    validate_finite_slice("requested_times", &requested_times)?;
    if !initial.is_finite() {
        return Err(value_error("initial must be finite"));
    }
    if requested_times_are_sorted(&requested_times) {
        return Ok(step_values_at_sorted_requests(
            &times,
            &values,
            &requested_times,
            initial,
        ));
    }
    Ok(requested_times
        .iter()
        .map(|&requested_time| step_value_at_with_initial(&times, &values, requested_time, initial))
        .collect())
}

type ConditionedCoxSurvfitOutput = (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>);

#[pyfunction]
pub fn condition_cox_survfit_curves(
    times: Vec<f64>,
    cumhaz: Vec<Vec<f64>>,
    t0: f64,
    include_time0: bool,
    filter_start_time: bool,
    time_epsilon: f64,
) -> PyResult<ConditionedCoxSurvfitOutput> {
    validate_step_times("times", &times)?;
    validate_matrix_nonnegative_finite("cumhaz", &cumhaz)?;
    if cumhaz.iter().any(|row| row.len() != times.len()) {
        return Err(value_error(
            "cumhaz rows must have the same length as times",
        ));
    }
    if !t0.is_finite() {
        return Err(value_error("t0 must be finite"));
    }
    if !time_epsilon.is_finite() || time_epsilon < 0.0 {
        return Err(value_error("time_epsilon must be non-negative and finite"));
    }

    let start_idx = if filter_start_time {
        let cutoff = t0 - time_epsilon;
        let idx = times.partition_point(|&time| time < cutoff);
        if idx == times.len() {
            return Err(value_error("start_time argument has removed all endpoints"));
        }
        idx
    } else {
        0
    };
    let start_pos = times.partition_point(|&time| time < t0);
    let mut kept_times = times[start_idx..].to_vec();

    let mut conditioned_cumhaz = Vec::with_capacity(cumhaz.len());
    let mut conditioned_surv = Vec::with_capacity(cumhaz.len());
    for row in &cumhaz {
        let start_hazard = if start_pos > 0 {
            row[start_pos - 1]
        } else {
            0.0
        };
        let mut curve_hazards: Vec<f64> = row[start_idx..]
            .iter()
            .map(|&hazard| (hazard - start_hazard).max(0.0))
            .collect();
        let mut curve_survival: Vec<f64> = curve_hazards
            .iter()
            .map(|&hazard| safe_exp(-hazard).clamp(0.0, 1.0))
            .collect();
        if include_time0
            && kept_times
                .first()
                .is_none_or(|&first_time| (first_time - t0).abs() >= time_epsilon)
        {
            curve_hazards.insert(0, 0.0);
            curve_survival.insert(0, 1.0);
        }
        conditioned_cumhaz.push(curve_hazards);
        conditioned_surv.push(curve_survival);
    }

    if include_time0
        && kept_times
            .first()
            .is_none_or(|&first_time| (first_time - t0).abs() >= time_epsilon)
    {
        kept_times.insert(0, t0);
    }

    Ok((kept_times, conditioned_surv, conditioned_cumhaz))
}

fn validate_nonnegative_cumhaz_slice(name: &str, values: &[f64]) -> PyResult<()> {
    let mut previous = 0.0;
    for (idx, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "{name} must be non-negative; got {value} at index {idx}"
            )));
        }
        if value < previous {
            return Err(value_error(format!(
                "{name} must be non-decreasing; got {value} at index {idx}"
            )));
        }
        previous = value;
    }
    Ok(())
}

fn validate_optional_count_vector(
    name: &str,
    values: Option<Vec<f64>>,
    n_times: usize,
) -> PyResult<Vec<f64>> {
    match values {
        Some(values) => {
            if values.len() != n_times {
                return Err(value_error(format!(
                    "{name} must have the same length as time"
                )));
            }
            validate_nonnegative_finite_slice(name, &values)?;
            Ok(values)
        }
        None => Ok(vec![0.0; n_times]),
    }
}

fn validate_matrix_width(
    name: &str,
    matrix: &[Vec<f64>],
    n_rows: usize,
    n_cols: usize,
) -> PyResult<()> {
    if matrix.len() != n_rows {
        return Err(value_error(format!("{name} length must match time length")));
    }
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != n_cols {
            return Err(value_error(format!(
                "{name} row {row_idx} has length {}, expected {n_cols}",
                row.len()
            )));
        }
    }
    Ok(())
}

fn validate_survival_matrix(name: &str, matrix: &[Vec<f64>]) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "{name} contains non-finite value at row {row_idx}, column {col_idx}"
                )));
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(value_error(format!(
                    "{name} values must be between 0 and 1; got {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_matrix_nonnegative_finite(name: &str, matrix: &[Vec<f64>]) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "{name} contains non-finite value at row {row_idx}, column {col_idx}"
                )));
            }
            if value < 0.0 {
                return Err(value_error(format!(
                    "{name} must be non-negative; got {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_matrix_nonnegative_or_infinite(name: &str, matrix: &[Vec<f64>]) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            if value.is_nan() {
                return Err(value_error(format!(
                    "{name} contains non-finite value at row {row_idx}, column {col_idx}"
                )));
            }
            if value < 0.0 {
                return Err(value_error(format!(
                    "{name} must be non-negative; got {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_transition_hazards(
    transition_hazards: &[Vec<Vec<f64>>],
    n_times: usize,
) -> PyResult<usize> {
    if transition_hazards.is_empty() {
        return Err(value_error("transition_hazards cannot be empty"));
    }
    if transition_hazards.len() != n_times {
        return Err(value_error(
            "transition_hazards length must match time length",
        ));
    }
    let n_states = transition_hazards[0].len();
    if n_states == 0 {
        return Err(value_error(
            "transition_hazards must have at least one state",
        ));
    }
    for (time_idx, haz) in transition_hazards.iter().enumerate() {
        if haz.len() != n_states {
            return Err(value_error(
                "Each time point must have n_states x n_states transition matrix",
            ));
        }
        for (row_idx, row) in haz.iter().enumerate() {
            if row.len() != n_states {
                return Err(value_error("Transition matrices must be square"));
            }
            let mut outgoing = 0.0;
            for (col_idx, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(value_error(format!(
                        "transition_hazards contains non-finite value at time {time_idx}, row {row_idx}, column {col_idx}"
                    )));
                }
                if row_idx != col_idx {
                    if value < 0.0 {
                        return Err(value_error(format!(
                            "transition_hazards off-diagonal entries must be non-negative; got {value} at time {time_idx}, row {row_idx}, column {col_idx}"
                        )));
                    }
                    outgoing += value;
                }
            }
            if outgoing > 1.0 + 1e-12 {
                return Err(value_error(format!(
                    "transition_hazards outgoing row sums must be less than or equal to 1; got {outgoing} at time {time_idx}, row {row_idx}"
                )));
            }
        }
    }
    Ok(n_states)
}

fn validate_matrix_result_parts(
    time: &[f64],
    surv: &[Vec<f64>],
    cumhaz: &[Vec<f64>],
    std_err: Option<&[Vec<f64>]>,
    n_risk: &[f64],
    n_event: &[f64],
    n_states: usize,
) -> PyResult<()> {
    if n_states == 0 {
        return Err(value_error("n_states must be positive"));
    }
    validate_step_times("time", time)?;
    validate_matrix_width("surv", surv, time.len(), n_states)?;
    validate_matrix_width("cumhaz", cumhaz, time.len(), n_states)?;
    validate_survival_matrix("surv", surv)?;
    validate_matrix_nonnegative_or_infinite("cumhaz", cumhaz)?;
    if let Some(std_err) = std_err {
        validate_matrix_width("std_err", std_err, time.len(), n_states)?;
        validate_matrix_nonnegative_finite("std_err", std_err)?;
    }
    if !n_risk.is_empty() {
        if n_risk.len() != time.len() {
            return Err(value_error("n_risk must have the same length as time"));
        }
        validate_nonnegative_finite_slice("n_risk", n_risk)?;
    }
    if !n_event.is_empty() {
        if n_event.len() != time.len() {
            return Err(value_error("n_event must have the same length as time"));
        }
        validate_nonnegative_finite_slice("n_event", n_event)?;
    }
    Ok(())
}

fn scaled_hazard_increment(events: f64, scaled_risk_sum: f64, risk_scale: f64) -> f64 {
    if events > 0.0 && scaled_risk_sum > 0.0 {
        events / scaled_risk_sum * risk_scale
    } else {
        0.0
    }
}

fn safe_exp(value: f64) -> f64 {
    value.clamp(PY_EXP_CLAMP_MIN, PY_EXP_CLAMP_MAX).exp()
}

fn step_value_at(times: &[f64], values: &[f64], time: f64) -> f64 {
    let idx = times.partition_point(|value| *value <= time);
    if idx == 0 { 0.0 } else { values[idx - 1] }
}

fn basehaz_with_entry_times(
    time: &[f64],
    status: &[i32],
    entry: &[f64],
    weights: &[f64],
    risk_scores: &[f64],
    risk_scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut event_times: Vec<f64> = time
        .iter()
        .zip(status.iter())
        .filter_map(|(&event_time, &event)| (event == 1).then_some(event_time))
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup_by(|a, b| same_time(*a, *b));

    let mut entry_order: Vec<usize> = (0..n).collect();
    entry_order.sort_by(|&a, &b| entry[a].total_cmp(&entry[b]).then_with(|| a.cmp(&b)));
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));
    let mut event_order: Vec<usize> = (0..n).filter(|&idx| status[idx] == 1).collect();
    event_order.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));

    let mut active = vec![false; n];
    let mut entry_pos = 0;
    let mut stop_pos = 0;
    let mut event_pos = 0;
    let mut risk_sum = 0.0;
    let mut cum_hazard = 0.0;
    let mut hazard = Vec::with_capacity(event_times.len());

    for &event_time in &event_times {
        while entry_pos < entry_order.len() && entry[entry_order[entry_pos]] < event_time {
            let idx = entry_order[entry_pos];
            if !active[idx] {
                active[idx] = true;
                risk_sum += risk_scores[idx];
            }
            entry_pos += 1;
        }
        while stop_pos < stop_order.len() && time[stop_order[stop_pos]] < event_time {
            let idx = stop_order[stop_pos];
            if active[idx] {
                active[idx] = false;
                risk_sum -= risk_scores[idx];
            }
            stop_pos += 1;
        }

        while event_pos < event_order.len()
            && time[event_order[event_pos]] < event_time
            && !same_time(time[event_order[event_pos]], event_time)
        {
            event_pos += 1;
        }
        let mut events = 0.0;
        while event_pos < event_order.len() && same_time(time[event_order[event_pos]], event_time) {
            events += weights[event_order[event_pos]];
            event_pos += 1;
        }

        cum_hazard += scaled_hazard_increment(events, risk_sum, risk_scale);
        hazard.push(cum_hazard);
    }

    (event_times, hazard)
}

fn validate_sorted_step_times(name: &str, times: &[f64], stratum: i32) -> PyResult<()> {
    for window in times.windows(2) {
        if window[1] < window[0] {
            return Err(value_error(format!(
                "{name} must be sorted within each stratum; stratum {stratum} has {} before {}",
                window[1], window[0]
            )));
        }
    }
    Ok(())
}

fn validate_non_decreasing_step_values(name: &str, values: &[f64], stratum: i32) -> PyResult<()> {
    for window in values.windows(2) {
        if window[1] < window[0] {
            return Err(value_error(format!(
                "{name} must be non-decreasing within each stratum; stratum {stratum} has {} before {}",
                window[1], window[0]
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvfitMatrixResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub surv: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_states: usize,
}

#[pymethods]
impl SurvfitMatrixResult {
    #[new]
    #[pyo3(signature = (time, surv, cumhaz, std_err=None, n_risk=vec![], n_event=vec![], n_states=1))]
    pub fn new(
        time: Vec<f64>,
        surv: Vec<Vec<f64>>,
        cumhaz: Vec<Vec<f64>>,
        std_err: Option<Vec<Vec<f64>>>,
        n_risk: Vec<f64>,
        n_event: Vec<f64>,
        n_states: usize,
    ) -> PyResult<Self> {
        validate_matrix_result_parts(
            &time,
            &surv,
            &cumhaz,
            std_err.as_deref(),
            &n_risk,
            &n_event,
            n_states,
        )?;
        Ok(Self {
            time,
            surv,
            cumhaz,
            std_err,
            n_risk,
            n_event,
            n_states,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvfitMatrixResult(n_times={}, n_states={}, has_stderr={})",
            self.time.len(),
            self.n_states,
            self.std_err.is_some()
        )
    }

    pub fn get_surv_at_state(&self, state: usize) -> PyResult<Vec<f64>> {
        if state >= self.n_states {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "State index {} out of range (n_states={})",
                state, self.n_states
            )));
        }
        self.surv
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.get(state).copied().ok_or_else(|| {
                    value_error(format!(
                        "surv row {row_idx} has length {}, cannot access state {state}",
                        row.len()
                    ))
                })
            })
            .collect()
    }

    pub fn get_cumhaz_at_state(&self, state: usize) -> PyResult<Vec<f64>> {
        if state >= self.n_states {
            return Err(PyErr::new::<PyIndexError, _>(format!(
                "State index {} out of range (n_states={})",
                state, self.n_states
            )));
        }
        self.cumhaz
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                row.get(state).copied().ok_or_else(|| {
                    value_error(format!(
                        "cumhaz row {row_idx} has length {}, cannot access state {state}",
                        row.len()
                    ))
                })
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (time, hazard, n_risk=None, n_event=None))]
pub fn survfit_from_hazard(
    time: Vec<f64>,
    hazard: Vec<f64>,
    n_risk: Option<Vec<f64>>,
    n_event: Option<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.len() != hazard.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and hazard must have the same length",
        ));
    }
    validate_step_times("time", &time)?;
    validate_nonnegative_finite_slice("hazard", &hazard)?;

    let n = time.len();
    let mut cumhaz = Vec::with_capacity(n);
    let mut surv = Vec::with_capacity(n);

    let mut cum = 0.0;
    for &h in &hazard {
        cum += h;
        cumhaz.push(vec![cum]);
        surv.push(vec![(-cum).exp()]);
    }

    let n_risk_vec = validate_optional_count_vector("n_risk", n_risk, n)?;
    let n_event_vec = validate_optional_count_vector("n_event", n_event, n)?;

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_states: 1,
    })
}

#[pyfunction]
#[pyo3(signature = (time, cumhaz, n_risk=None, n_event=None))]
pub fn survfit_from_cumhaz(
    time: Vec<f64>,
    cumhaz: Vec<f64>,
    n_risk: Option<Vec<f64>>,
    n_event: Option<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.len() != cumhaz.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and cumhaz must have the same length",
        ));
    }
    validate_step_times("time", &time)?;
    validate_nonnegative_cumhaz_slice("cumhaz", &cumhaz)?;

    let n = time.len();
    let surv: Vec<Vec<f64>> = cumhaz.iter().map(|&h| vec![(-h).exp()]).collect();
    let cumhaz_matrix: Vec<Vec<f64>> = cumhaz.iter().map(|&h| vec![h]).collect();

    let n_risk_vec = validate_optional_count_vector("n_risk", n_risk, n)?;
    let n_event_vec = validate_optional_count_vector("n_event", n_event, n)?;

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz: cumhaz_matrix,
        std_err: None,
        n_risk: n_risk_vec,
        n_event: n_event_vec,
        n_states: 1,
    })
}

#[pyfunction]
pub fn survfit_from_matrix(
    time: Vec<f64>,
    hazard_matrix: Vec<Vec<f64>>,
) -> PyResult<SurvfitMatrixResult> {
    if time.is_empty() || hazard_matrix.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and hazard_matrix cannot be empty",
        ));
    }

    let n_times = time.len();
    let n_states = hazard_matrix[0].len();
    if n_states == 0 {
        return Err(value_error("hazard_matrix must have at least one column"));
    }
    validate_step_times("time", &time)?;

    validate_matrix_width("hazard_matrix", &hazard_matrix, n_times, n_states)?;
    validate_matrix_nonnegative_finite("hazard_matrix", &hazard_matrix)?;

    let mut cumhaz = Vec::with_capacity(n_times);
    let mut surv = Vec::with_capacity(n_times);

    let mut cum = vec![0.0; n_states];

    for row in &hazard_matrix {
        for (j, &h) in row.iter().enumerate() {
            cum[j] += h;
        }
        cumhaz.push(cum.clone());
        surv.push(cum.iter().map(|&c| (-c).exp()).collect());
    }

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: vec![0.0; n_times],
        n_event: vec![0.0; n_times],
        n_states,
    })
}

#[pyfunction]
pub fn survfit_multistate(
    time: Vec<f64>,
    transition_hazards: Vec<Vec<Vec<f64>>>,
    initial_state: usize,
) -> PyResult<SurvfitMatrixResult> {
    if time.is_empty() || transition_hazards.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and transition_hazards cannot be empty",
        ));
    }

    let n_times = time.len();
    validate_step_times("time", &time)?;

    let n_states = validate_transition_hazards(&transition_hazards, n_times)?;

    if initial_state >= n_states {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "initial_state must be less than n_states",
        ));
    }

    let mut prob = vec![0.0; n_states];
    prob[initial_state] = 1.0;

    let mut surv = Vec::with_capacity(n_times);
    let mut cumhaz = Vec::with_capacity(n_times);

    for haz_matrix in &transition_hazards {
        let mut new_prob = vec![0.0; n_states];

        for i in 0..n_states {
            let mut out_rate = 0.0;
            for j in 0..n_states {
                if i != j {
                    out_rate += haz_matrix[i][j];
                    new_prob[j] += prob[i] * haz_matrix[i][j];
                }
            }
            new_prob[i] += prob[i] * (1.0 - out_rate).max(0.0);
        }

        prob = new_prob;
        surv.push(prob.clone());

        let ch: Vec<f64> = prob
            .iter()
            .map(|&p| if p > 0.0 { -p.ln() } else { f64::INFINITY })
            .collect();
        cumhaz.push(ch);
    }

    Ok(SurvfitMatrixResult {
        time,
        surv,
        cumhaz,
        std_err: None,
        n_risk: vec![0.0; n_times],
        n_event: vec![0.0; n_times],
        n_states,
    })
}

#[pyfunction]
#[pyo3(signature = (base_times, base_hazards, linear_predictors, center=0.0, base_strata=None, curve_strata=None, requested_times=None))]
pub fn cox_survfit_from_baseline(
    base_times: Vec<f64>,
    base_hazards: Vec<f64>,
    linear_predictors: Vec<f64>,
    center: f64,
    base_strata: Option<Vec<i32>>,
    curve_strata: Option<Vec<i32>>,
    requested_times: Option<Vec<f64>>,
) -> PyResult<CoxBaselineCurveOutput> {
    if base_times.len() != base_hazards.len() {
        return Err(value_error(
            "base_times and base_hazards must have the same length",
        ));
    }
    validate_finite_slice("base_times", &base_times)?;
    validate_nonnegative_finite_slice("base_hazards", &base_hazards)?;
    validate_finite_slice("linear_predictors", &linear_predictors)?;
    if !center.is_finite() {
        return Err(value_error("center must be finite"));
    }

    let base_strata_values = match base_strata {
        Some(values) => {
            if values.len() != base_times.len() {
                return Err(value_error(
                    "base_strata must have the same length as base_times",
                ));
            }
            values
        }
        None => vec![0; base_times.len()],
    };
    let curve_strata_values = match curve_strata {
        Some(values) => {
            if values.len() != linear_predictors.len() {
                return Err(value_error(
                    "curve_strata must have the same length as linear_predictors",
                ));
            }
            values
        }
        None => vec![0; linear_predictors.len()],
    };

    let mut baselines: BTreeMap<i32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for ((&time, &hazard), &stratum) in base_times
        .iter()
        .zip(base_hazards.iter())
        .zip(base_strata_values.iter())
    {
        let (times, hazards) = baselines.entry(stratum).or_default();
        times.push(time);
        hazards.push(hazard);
    }
    for (&stratum, (times, hazards)) in &baselines {
        validate_sorted_step_times("base_times", times, stratum)?;
        validate_non_decreasing_step_values("base_hazards", hazards, stratum)?;
    }

    let mut selected_strata = curve_strata_values.clone();
    selected_strata.sort_unstable();
    selected_strata.dedup();
    for stratum in &selected_strata {
        if !baselines.contains_key(stratum) {
            return Err(value_error(format!(
                "curve_strata contains stratum {stratum} with no baseline hazard"
            )));
        }
    }

    let output_times = match requested_times {
        Some(values) => {
            validate_finite_slice("requested_times", &values)?;
            values
        }
        None => {
            let mut values = Vec::new();
            for stratum in selected_strata {
                if let Some((times, _)) = baselines.get(&stratum) {
                    values.extend(times.iter().copied());
                }
            }
            values.sort_by(|left, right| left.total_cmp(right));
            values.dedup_by(|left, right| same_time(*left, *right));
            values
        }
    };

    let output_times_sorted = requested_times_are_sorted(&output_times);
    let mut survival_curves = Vec::with_capacity(linear_predictors.len());
    let mut cumulative_hazards = Vec::with_capacity(linear_predictors.len());
    for (&linear_predictor, &stratum) in linear_predictors.iter().zip(curve_strata_values.iter()) {
        let risk_multiplier = safe_exp(linear_predictor - center);
        let empty_times: &[f64] = &[];
        let empty_hazards: &[f64] = &[];
        let (times, hazards) = baselines
            .get(&stratum)
            .map(|(times, hazards)| (times.as_slice(), hazards.as_slice()))
            .unwrap_or((empty_times, empty_hazards));
        let mut curve = Vec::with_capacity(output_times.len());
        let mut cumhaz = Vec::with_capacity(output_times.len());
        if output_times_sorted {
            let mut cursor = 0;
            for &time in &output_times {
                while cursor < times.len() && times[cursor] <= time {
                    cursor += 1;
                }
                let base_hazard = if cursor == 0 {
                    0.0
                } else {
                    hazards[cursor - 1]
                };
                let hazard = base_hazard * risk_multiplier;
                cumhaz.push(hazard);
                curve.push(safe_exp(-hazard).clamp(0.0, 1.0));
            }
        } else {
            for &time in &output_times {
                let hazard = step_value_at(times, hazards, time) * risk_multiplier;
                cumhaz.push(hazard);
                curve.push(safe_exp(-hazard).clamp(0.0, 1.0));
            }
        }
        survival_curves.push(curve);
        cumulative_hazards.push(cumhaz);
    }

    Ok((output_times, survival_curves, cumulative_hazards))
}

#[pyfunction]
#[pyo3(signature = (time, status, linear_predictors, centered, entry_times=None, weights=None))]
pub fn basehaz(
    time: Vec<f64>,
    status: Vec<i32>,
    linear_predictors: Vec<f64>,
    centered: bool,
    entry_times: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time must not be empty",
        ));
    }
    if status.len() != n || linear_predictors.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and linear_predictors must have the same length",
        ));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time contains non-finite value at index {}",
                idx
            )));
        }
    }
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "status must contain only 0/1 values; got {} at index {}",
                value, idx
            )));
        }
    }
    for (idx, &value) in linear_predictors.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "linear_predictors contains non-finite value at index {}",
                idx
            )));
        }
    }
    if let Some(values) = entry_times.as_ref() {
        if values.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "entry_times must have the same length as time",
            ));
        }
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "entry_times contains non-finite value at index {}",
                    idx
                )));
            }
        }
        for (idx, (&start, &stop)) in values.iter().zip(time.iter()).enumerate() {
            if start >= stop {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "entry_times[{}] must be less than time[{}]",
                    idx, idx
                )));
            }
        }
    }
    if let Some(values) = weights.as_ref()
        && values.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must have the same length as time",
        ));
    }
    if let Some(values) = weights.as_ref() {
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights contains non-finite value at index {}",
                    idx
                )));
            }
            if value < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights must be non-negative; got {} at index {}",
                    value, idx
                )));
            }
        }
        if values.iter().all(|&value| value == 0.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must include at least one positive value",
            ));
        }
    }
    let weights = weights.unwrap_or_else(|| vec![1.0; n]);

    let center = if centered {
        linear_predictors.iter().sum::<f64>() / n as f64
    } else {
        0.0
    };

    let max_shifted_lp = linear_predictors
        .iter()
        .zip(weights.iter())
        .filter_map(|(&lp, &weight)| (weight > 0.0).then_some(lp - center))
        .fold(f64::NEG_INFINITY, f64::max);
    let risk_scale = (-max_shifted_lp).exp();
    let risk_scores: Vec<f64> = linear_predictors
        .iter()
        .zip(weights.iter())
        .map(|(&lp, &weight)| {
            if weight == 0.0 {
                0.0
            } else {
                weight * (lp - center - max_shifted_lp).exp()
            }
        })
        .collect();

    if let Some(entry) = entry_times {
        return Ok(basehaz_with_entry_times(
            &time,
            &status,
            &entry,
            &weights,
            &risk_scores,
            risk_scale,
        ));
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut unique_times = Vec::new();
    let mut hazard = Vec::new();

    let mut cumulative_risk: Vec<f64> = vec![0.0; n];
    let mut running_sum = 0.0;
    for i in (0..n).rev() {
        running_sum += risk_scores[indices[i]];
        cumulative_risk[i] = running_sum;
    }

    let mut i = 0;
    let mut cum_hazard = 0.0;

    while i < n {
        let group_start = i;
        let current_time = time[indices[i]];
        let mut events = 0.0;
        let mut has_event = false;

        while i < n && same_time(time[indices[i]], current_time) {
            if status[indices[i]] == 1 {
                has_event = true;
                events += weights[indices[i]];
            }
            i += 1;
        }

        if !has_event {
            continue;
        }

        let risk_sum = cumulative_risk[group_start];
        cum_hazard += scaled_hazard_increment(events, risk_sum, risk_scale);

        unique_times.push(current_time);
        hazard.push(cum_hazard);
    }

    Ok((unique_times, hazard))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survfit_from_hazard() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hazard = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let result = survfit_from_hazard(time, hazard, None, None).unwrap();

        assert_eq!(result.time.len(), 5);
        assert_eq!(result.n_states, 1);
        assert!((result.surv[0][0] - (-0.1_f64).exp()).abs() < 1e-10);
        assert!((result.surv[4][0] - (-0.5_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_from_hazard_rejects_unsorted_time() {
        let err = survfit_from_hazard(vec![2.0, 1.0], vec![0.1, 0.1], None, None)
            .expect_err("unsorted time grid should fail");

        assert!(err.to_string().contains("time must be sorted ascending"));
    }

    #[test]
    fn test_survfit_from_cumhaz() {
        let time = vec![1.0, 2.0, 3.0];
        let cumhaz = vec![0.1, 0.3, 0.6];
        let result = survfit_from_cumhaz(time, cumhaz, None, None).unwrap();

        assert_eq!(result.time.len(), 3);
        assert!((result.surv[2][0] - (-0.6_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_from_cumhaz_rejects_unsorted_time() {
        let err = survfit_from_cumhaz(vec![2.0, 1.0], vec![0.1, 0.2], None, None)
            .expect_err("unsorted time grid should fail");

        assert!(err.to_string().contains("time must be sorted ascending"));
    }

    #[test]
    fn test_survfit_from_matrix() {
        let time = vec![1.0, 2.0, 3.0];
        let hazard_matrix = vec![vec![0.1, 0.05], vec![0.1, 0.05], vec![0.1, 0.05]];
        let result = survfit_from_matrix(time, hazard_matrix).unwrap();

        assert_eq!(result.n_states, 2);
        assert!((result.cumhaz[2][0] - 0.3).abs() < 1e-10);
        assert!((result.cumhaz[2][1] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_survfit_from_matrix_rejects_unsorted_time() {
        let err = survfit_from_matrix(vec![2.0, 1.0, 3.0], vec![vec![0.1], vec![0.1], vec![0.1]])
            .expect_err("unsorted time grid should fail");

        assert!(err.to_string().contains("time must be sorted ascending"));
    }

    #[test]
    fn test_survfit_multistate_preserves_probability_mass() {
        let result =
            survfit_multistate(vec![1.0], vec![vec![vec![0.0, 0.25], vec![0.10, 0.0]]], 0).unwrap();

        assert_eq!(result.n_states, 2);
        assert_eq!(result.surv[0], vec![0.75, 0.25]);
        assert!((result.surv[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_survfit_multistate_rejects_unsorted_time() {
        let err = survfit_multistate(
            vec![2.0, 1.0],
            vec![
                vec![vec![0.0, 0.25], vec![0.10, 0.0]],
                vec![vec![0.0, 0.20], vec![0.05, 0.0]],
            ],
            0,
        )
        .expect_err("unsorted time grid should fail");

        assert!(err.to_string().contains("time must be sorted ascending"));
    }

    #[test]
    fn test_step_values_at_uses_right_continuous_steps() {
        let sorted = step_values_at(
            vec![1.0, 3.0, 5.0],
            vec![10.0, 30.0, 50.0],
            vec![0.5, 1.0, 4.0, 6.0],
            0.0,
        )
        .unwrap();
        assert_eq!(sorted, vec![0.0, 10.0, 30.0, 50.0]);

        let unsorted = step_values_at(
            vec![1.0, 3.0, 5.0],
            vec![10.0, 30.0, 50.0],
            vec![6.0, 0.5, 3.0],
            -1.0,
        )
        .unwrap();
        assert_eq!(unsorted, vec![50.0, -1.0, 30.0]);
    }

    #[test]
    fn test_condition_cox_survfit_curves_rebases_cumulative_hazard() {
        let (time, surv, cumhaz) = condition_cox_survfit_curves(
            vec![1.0, 3.0, 5.0],
            vec![vec![0.2, 0.6, 1.1], vec![0.0, 0.4, 0.8]],
            2.5,
            true,
            true,
            1e-9,
        )
        .unwrap();

        assert_eq!(time, vec![2.5, 3.0, 5.0]);
        assert_eq!(cumhaz[0][0], 0.0);
        assert!((cumhaz[0][1] - 0.4).abs() < 1e-12);
        assert!((cumhaz[0][2] - 0.9).abs() < 1e-12);
        assert_eq!(cumhaz[1], vec![0.0, 0.4, 0.8]);
        assert_eq!(surv[0][0], 1.0);
        assert!((surv[0][1] - (-0.4_f64).exp()).abs() < 1e-12);
        assert!((surv[1][2] - (-0.8_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_survfit_matrix_result_constructor_validates_shape() {
        let result =
            SurvfitMatrixResult::new(vec![1.0], vec![], vec![vec![0.1]], None, vec![], vec![], 1);

        assert!(result.is_err());

        let result = SurvfitMatrixResult::new(
            vec![1.0],
            vec![vec![1.2]],
            vec![vec![0.1]],
            None,
            vec![],
            vec![],
            1,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_survfit_matrix_result_constructor_rejects_unsorted_time() {
        let result = SurvfitMatrixResult::new(
            vec![2.0, 1.0],
            vec![vec![0.9], vec![0.8]],
            vec![vec![0.1], vec![0.2]],
            None,
            vec![],
            vec![],
            1,
        );

        assert!(
            result
                .expect_err("unsorted time grid should fail")
                .to_string()
                .contains("time must be sorted ascending")
        );
    }

    #[test]
    fn test_survfit_matrix_accessors_reject_ragged_rows() {
        let result = SurvfitMatrixResult {
            time: vec![1.0],
            surv: vec![vec![]],
            cumhaz: vec![vec![]],
            std_err: None,
            n_risk: vec![],
            n_event: vec![],
            n_states: 1,
        };

        assert!(result.get_surv_at_state(0).is_err());
        assert!(result.get_cumhaz_at_state(0).is_err());
    }

    #[test]
    fn test_survfit_hazard_builders_reject_invalid_values() {
        assert!(survfit_from_hazard(vec![1.0], vec![f64::NAN], None, None).is_err());
        assert!(survfit_from_matrix(vec![1.0], vec![vec![-0.1]]).is_err());
        assert!(survfit_from_matrix(vec![1.0, 2.0], vec![vec![0.1]]).is_err());
        assert!(survfit_from_matrix(vec![1.0], vec![vec![]]).is_err());
        assert!(
            survfit_multistate(vec![1.0], vec![vec![vec![0.0, -0.1], vec![0.0, 0.0]]], 0).is_err()
        );
        assert!(survfit_multistate(vec![1.0], vec![vec![]], 0).is_err());
        assert!(
            survfit_multistate(
                vec![1.0],
                vec![vec![
                    vec![0.0, 0.8, 0.3],
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 0.0, 0.0]
                ]],
                0
            )
            .is_err()
        );
    }

    #[test]
    fn test_cox_survfit_from_baseline_steps_by_stratum() {
        let (times, curves, cumhaz) = cox_survfit_from_baseline(
            vec![1.0, 3.0, 2.0, 4.0],
            vec![0.2, 0.5, 0.1, 0.4],
            vec![0.0, 2.0_f64.ln()],
            0.0,
            Some(vec![0, 0, 1, 1]),
            Some(vec![1, 0]),
            None,
        )
        .unwrap();

        assert_eq!(times, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cumhaz[0], vec![0.0, 0.1, 0.1, 0.4]);
        assert_eq!(cumhaz[1], vec![0.4, 0.4, 1.0, 1.0]);
        for (survival, hazard) in curves[0].iter().zip(cumhaz[0].iter()) {
            assert!((*survival - (-*hazard).exp()).abs() < 1e-12);
        }
    }

    #[test]
    fn test_cox_survfit_from_baseline_preserves_unsorted_requested_times() {
        let (times, curves, cumhaz) = cox_survfit_from_baseline(
            vec![1.0, 3.0],
            vec![0.2, 0.5],
            vec![0.0],
            0.0,
            None,
            None,
            Some(vec![4.0, 0.5, 2.0]),
        )
        .unwrap();

        assert_eq!(times, vec![4.0, 0.5, 2.0]);
        assert_eq!(cumhaz[0], vec![0.5, 0.0, 0.2]);
        assert_eq!(curves[0][1], 1.0);
        assert!((curves[0][0] - (-0.5_f64).exp()).abs() < 1e-12);
        assert!((curves[0][2] - (-0.2_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_cox_survfit_from_baseline_dedupes_near_tied_default_times() {
        let (times, curves, cumhaz) = cox_survfit_from_baseline(
            vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0],
            vec![0.2, 0.25, 0.6],
            vec![0.0],
            0.0,
            Some(vec![0, 0, 0]),
            Some(vec![0]),
            None,
        )
        .unwrap();

        assert_eq!(times, vec![1.0, 2.0]);
        assert_eq!(cumhaz[0], vec![0.2, 0.6]);
        assert!((curves[0][0] - (-0.2_f64).exp()).abs() < 1e-12);
        assert!((curves[0][1] - (-0.6_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_cox_survfit_from_baseline_rejects_invalid_baseline_inputs() {
        assert!(
            cox_survfit_from_baseline(
                vec![1.0, 2.0],
                vec![0.2, 0.1],
                vec![0.0],
                0.0,
                None,
                None,
                None,
            )
            .is_err()
        );

        assert!(
            cox_survfit_from_baseline(
                vec![1.0],
                vec![0.2],
                vec![0.0],
                0.0,
                Some(vec![0]),
                Some(vec![1]),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn test_basehaz() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let lp = vec![0.0, 0.1, -0.1, 0.2, 0.0];
        let (times, haz) = basehaz(time, status, lp, true, None, None).unwrap();

        assert_eq!(times.len(), 3);
        assert_eq!(haz.len(), 3);
        assert!(haz[0] > 0.0);
        assert!(haz[1] > haz[0]);
        assert!(haz[2] > haz[1]);
    }

    #[test]
    fn test_basehaz_uses_scaled_risk_scores_for_large_linear_predictors() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];
        let lp = vec![710.0, 709.0, 708.0];
        let (times, haz) = basehaz(time, status, lp, false, None, None).unwrap();

        let expected_first = (-710.0_f64).exp() / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());
        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        assert!(haz[0].is_finite());
        assert!(haz[0] > 0.0);
        assert!((haz[0] - expected_first).abs() <= expected_first * 1e-12);
        assert!(haz[1] > haz[0]);
        assert!(haz[2] > haz[1]);
    }

    #[test]
    fn test_basehaz_counts_same_time_censors_in_event_risk_set() {
        let (times, hazard) = basehaz(
            vec![2.0, 2.0, 3.0],
            vec![0, 1, 1],
            vec![0.0, 0.0, 0.0],
            false,
            None,
            None,
        )
        .unwrap();

        assert_eq!(times, vec![2.0, 3.0]);
        assert!((hazard[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((hazard[1] - (1.0 / 3.0 + 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_basehaz_same_time_censors_match_weighted_scan_reference() {
        let time = vec![2.0, 2.0, 2.0, 3.0];
        let status = vec![0, 1, 0, 1];
        let lp: Vec<f64> = vec![0.0, 0.3, -0.2, 0.1];
        let weights = vec![2.0, 1.5, 0.5, 1.0];

        let risk_at_two = weights[0] * lp[0].exp()
            + weights[1] * lp[1].exp()
            + weights[2] * lp[2].exp()
            + weights[3] * lp[3].exp();
        let risk_at_three = weights[3] * lp[3].exp();
        let expected = [
            weights[1] / risk_at_two,
            weights[1] / risk_at_two + weights[3] / risk_at_three,
        ];

        let (times, hazard) = basehaz(time, status, lp, false, None, Some(weights)).unwrap();

        assert_eq!(times, vec![2.0, 3.0]);
        for (actual, expected) in hazard.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_basehaz_with_entry_times_uses_delayed_entry_risk_sets() {
        let time = vec![2.0, 2.0, 4.0, 5.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 0, 1, 0];
        let lp = vec![0.0; time.len()];
        let entry = vec![0.0, 0.0, 1.5, 2.5, 0.0, 3.0];
        let (times, haz) = basehaz(time, status, lp, false, Some(entry), None).unwrap();

        assert_eq!(times, vec![2.0, 4.0, 5.0]);
        assert!((haz[0] - 0.5).abs() < 1e-12);
        assert!((haz[1] - 0.75).abs() < 1e-12);
        assert!((haz[2] - (0.75 + 1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_basehaz_with_entry_times_matches_weighted_scan_reference() {
        let time = vec![2.0, 2.0, 4.0, 5.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 0, 1, 0];
        let lp: Vec<f64> = vec![0.0, 0.2, -0.1, 0.4, 0.1, -0.3];
        let entry = vec![0.0, 0.0, 1.5, 2.5, 0.0, 3.0];
        let weights = vec![1.0, 2.0, 0.5, 1.5, 1.0, 0.25];

        let mut expected_times: Vec<f64> = time
            .iter()
            .zip(status.iter())
            .filter_map(|(&event_time, &event)| (event == 1).then_some(event_time))
            .collect();
        expected_times.sort_by(f64::total_cmp);
        expected_times.dedup_by(|left, right| same_time(*left, *right));

        let mut cumulative = 0.0;
        let expected_hazard: Vec<f64> = expected_times
            .iter()
            .map(|&event_time| {
                let events = time
                    .iter()
                    .zip(status.iter())
                    .zip(weights.iter())
                    .filter_map(|((&stop, &event), &weight)| {
                        (event == 1 && same_time(stop, event_time)).then_some(weight)
                    })
                    .sum::<f64>();
                let risk_sum = (0..time.len())
                    .filter(|&idx| entry[idx] < event_time && time[idx] >= event_time)
                    .map(|idx| weights[idx] * lp[idx].exp())
                    .sum::<f64>();
                if risk_sum > 0.0 {
                    cumulative += events / risk_sum;
                }
                cumulative
            })
            .collect();

        let (times, hazard) = basehaz(time, status, lp, false, Some(entry), Some(weights)).unwrap();

        assert_eq!(times, expected_times);
        assert_eq!(hazard.len(), expected_hazard.len());
        for (actual, expected) in hazard.iter().zip(expected_hazard.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
