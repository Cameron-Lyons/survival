use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

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

fn validate_matrix_finite(name: &str, matrix: &[Vec<f64>]) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "{name} contains non-finite value at row {row_idx}, column {col_idx}"
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

fn validate_transition_hazards(transition_hazards: &[Vec<Vec<f64>>]) -> PyResult<()> {
    let n_states = transition_hazards[0].len();
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
            for (col_idx, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(value_error(format!(
                        "transition_hazards contains non-finite value at time {time_idx}, row {row_idx}, column {col_idx}"
                    )));
                }
                if row_idx != col_idx && value < 0.0 {
                    return Err(value_error(format!(
                        "transition_hazards off-diagonal entries must be non-negative; got {value} at time {time_idx}, row {row_idx}, column {col_idx}"
                    )));
                }
            }
        }
    }
    Ok(())
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
    validate_finite_slice("time", time)?;
    validate_matrix_width("surv", surv, time.len(), n_states)?;
    validate_matrix_width("cumhaz", cumhaz, time.len(), n_states)?;
    validate_matrix_finite("surv", surv)?;
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
    validate_finite_slice("time", &time)?;
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
    validate_finite_slice("time", &time)?;
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
    validate_finite_slice("time", &time)?;

    if hazard_matrix.len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_matrix rows must match time length",
        ));
    }

    for row in &hazard_matrix {
        if row.len() != n_states {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All rows in hazard_matrix must have the same number of columns",
            ));
        }
    }
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
    let n_states = transition_hazards[0].len();
    if n_states == 0 {
        return Err(value_error(
            "transition_hazards must have at least one state",
        ));
    }
    validate_finite_slice("time", &time)?;

    if transition_hazards.len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "transition_hazards length must match time length",
        ));
    }
    validate_transition_hazards(&transition_hazards)?;

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
        let mut event_times: Vec<f64> = time
            .iter()
            .zip(status.iter())
            .filter_map(|(&event_time, &event)| (event == 1).then_some(event_time))
            .collect();
        event_times.sort_by(|a, b| a.total_cmp(b));
        event_times.dedup_by(|a, b| (*a - *b).abs() < crate::constants::TIME_EPSILON);

        let mut hazard = Vec::with_capacity(event_times.len());
        let mut cum_hazard = 0.0;
        for event_time in &event_times {
            let events = time
                .iter()
                .zip(status.iter())
                .zip(weights.iter())
                .filter_map(|((&stop, &event), &weight)| {
                    (event == 1 && (stop - event_time).abs() < crate::constants::TIME_EPSILON)
                        .then_some(weight)
                })
                .sum::<f64>();
            let risk_sum: f64 = (0..n)
                .filter(|&idx| entry[idx] < *event_time && time[idx] >= *event_time)
                .map(|idx| risk_scores[idx])
                .sum();
            cum_hazard += scaled_hazard_increment(events, risk_sum, risk_scale);
            hazard.push(cum_hazard);
        }
        return Ok((event_times, hazard));
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
        let idx = indices[i];
        if status[idx] == 0 {
            i += 1;
            continue;
        }

        let current_time = time[idx];
        let mut events = 0.0;
        let start_i = i;

        while i < n && (time[indices[i]] - current_time).abs() < crate::constants::TIME_EPSILON {
            if status[indices[i]] == 1 {
                events += weights[indices[i]];
            }
            i += 1;
        }

        let risk_sum = cumulative_risk[start_i];
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
    fn test_survfit_from_cumhaz() {
        let time = vec![1.0, 2.0, 3.0];
        let cumhaz = vec![0.1, 0.3, 0.6];
        let result = survfit_from_cumhaz(time, cumhaz, None, None).unwrap();

        assert_eq!(result.time.len(), 3);
        assert!((result.surv[2][0] - (-0.6_f64).exp()).abs() < 1e-10);
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
    fn test_survfit_matrix_result_constructor_validates_shape() {
        let result =
            SurvfitMatrixResult::new(vec![1.0], vec![], vec![vec![0.1]], None, vec![], vec![], 1);

        assert!(result.is_err());
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
        assert!(
            survfit_multistate(vec![1.0], vec![vec![vec![0.0, -0.1], vec![0.0, 0.0]]], 0).is_err()
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
}
