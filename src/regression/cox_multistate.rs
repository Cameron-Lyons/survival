use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MultiStateCoxStack {
    #[pyo3(get)]
    pub start: Option<Vec<f64>>,
    #[pyo3(get)]
    pub stop: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub strata: Vec<usize>,
    #[pyo3(get)]
    pub source_rows: Vec<usize>,
    #[pyo3(get)]
    pub transition_indices: Vec<usize>,
}

fn validate_parallel_length(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must have length {expected}; got {actual}"
        )));
    }
    Ok(())
}

/// Expand multi-state histories into transition-stratified Cox rows.
///
/// Each observed interval is copied once for every modeled transition whose
/// source matches the row's current state. Covariates are placed in a
/// transition-specific block so the ordinary stratified Cox optimizer can fit
/// the same separate coefficients and baselines as a single-formula
/// multi-state model.
#[pyfunction]
#[pyo3(signature = (start, stop, event, current_state, covariates, transitions, strata=None))]
pub fn cox_multistate_stack(
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    event: Vec<i32>,
    current_state: Vec<usize>,
    covariates: Vec<Vec<f64>>,
    transitions: Vec<Vec<usize>>,
    strata: Option<Vec<usize>>,
) -> PyResult<MultiStateCoxStack> {
    let n = stop.len();
    validate_parallel_length("event", event.len(), n)?;
    validate_parallel_length("current_state", current_state.len(), n)?;
    validate_parallel_length("covariates", covariates.len(), n)?;
    if let Some(values) = start.as_ref() {
        validate_parallel_length("start", values.len(), n)?;
    }
    if let Some(values) = strata.as_ref() {
        validate_parallel_length("strata", values.len(), n)?;
    }

    let width = covariates.first().map_or(0, Vec::len);
    if covariates.iter().any(|row| row.len() != width) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covariates must be rectangular",
        ));
    }
    if transitions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "multi-state Cox fitting requires at least one observed transition",
        ));
    }
    if transitions.iter().any(|transition| transition.len() != 2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each transition must contain source and target state indices",
        ));
    }
    for (transition_idx, transition) in transitions.iter().enumerate() {
        if transition[0] == transition[1] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transition {transition_idx} must change state"
            )));
        }
        if transitions[..transition_idx]
            .iter()
            .any(|prior| prior == transition)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transition {transition_idx} duplicates an earlier transition"
            )));
        }
    }

    let user_strata = strata.as_deref();
    let user_strata_count = user_strata
        .and_then(|values| values.iter().max().copied())
        .map_or(1, |maximum| maximum + 1);
    let expanded_width = width.checked_mul(transitions.len()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expanded Cox design width is too large")
    })?;
    let capacity = n.checked_mul(transitions.len()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("expanded Cox row count is too large")
    })?;

    let mut expanded_start = start.as_ref().map(|_| Vec::with_capacity(capacity));
    let mut expanded_stop = Vec::with_capacity(capacity);
    let mut expanded_status = Vec::with_capacity(capacity);
    let mut expanded_covariates = Vec::with_capacity(capacity);
    let mut expanded_strata = Vec::with_capacity(capacity);
    let mut source_rows = Vec::with_capacity(capacity);
    let mut transition_indices = Vec::with_capacity(capacity);

    for (transition_idx, transition) in transitions.iter().enumerate() {
        let source = transition[0];
        let target = transition[1];
        let column_start = transition_idx * width;
        for row_idx in 0..n {
            if current_state[row_idx] != source {
                continue;
            }
            if let (Some(output), Some(input)) = (expanded_start.as_mut(), start.as_ref()) {
                output.push(input[row_idx]);
            }
            expanded_stop.push(stop[row_idx]);
            expanded_status.push(i32::from(event[row_idx] == (target + 1) as i32));
            let mut row = vec![0.0; expanded_width];
            row[column_start..column_start + width].copy_from_slice(&covariates[row_idx]);
            expanded_covariates.push(row);
            expanded_strata.push(
                transition_idx * user_strata_count
                    + user_strata.map_or(0, |values| values[row_idx]),
            );
            source_rows.push(row_idx);
            transition_indices.push(transition_idx);
        }
    }

    Ok(MultiStateCoxStack {
        start: expanded_start,
        stop: expanded_stop,
        status: expanded_status,
        covariates: expanded_covariates,
        strata: expanded_strata,
        source_rows,
        transition_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacks_competing_risks_in_transition_major_order() {
        let result = cox_multistate_stack(
            None,
            vec![1.0, 2.0, 3.0],
            vec![2, 3, 0],
            vec![0, 0, 0],
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![vec![0, 1], vec![0, 2]],
            None,
        )
        .unwrap();

        assert_eq!(result.stop, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(result.status, vec![1, 0, 0, 0, 1, 0]);
        assert_eq!(result.strata, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(result.source_rows, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(result.transition_indices, vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(
            result.covariates,
            vec![
                vec![1.0, 0.0],
                vec![2.0, 0.0],
                vec![3.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 2.0],
                vec![0.0, 3.0],
            ]
        );
    }

    #[test]
    fn stacks_only_outgoing_rows_and_crosses_user_strata() {
        let result = cox_multistate_stack(
            Some(vec![0.0, 1.0, 0.0]),
            vec![1.0, 2.0, 2.0],
            vec![2, 0, 0],
            vec![0, 1, 0],
            vec![vec![1.0], vec![2.0], vec![3.0]],
            vec![vec![0, 1], vec![1, 2]],
            Some(vec![0, 1, 1]),
        )
        .unwrap();

        assert_eq!(result.start, Some(vec![0.0, 0.0, 1.0]));
        assert_eq!(result.stop, vec![1.0, 2.0, 2.0]);
        assert_eq!(result.status, vec![1, 0, 0]);
        assert_eq!(result.strata, vec![0, 1, 3]);
        assert_eq!(result.source_rows, vec![0, 2, 1]);
    }
}
