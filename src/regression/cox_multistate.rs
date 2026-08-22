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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MultiStateCoxCurve {
    #[pyo3(get)]
    pub pstate: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MultiStateCoxCurves {
    #[pyo3(get)]
    pub pstate: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<Vec<f64>>>,
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
/// separate, omitted, or shared coefficients and baselines. When maps are not
/// supplied, each transition receives its own coefficient block and baseline.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    start,
    stop,
    event,
    current_state,
    covariates,
    transitions,
    strata=None,
    coefficient_map=None,
    baseline_map=None,
))]
pub fn cox_multistate_stack(
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    event: Vec<i32>,
    current_state: Vec<usize>,
    covariates: Vec<Vec<f64>>,
    transitions: Vec<Vec<usize>>,
    strata: Option<Vec<usize>>,
    coefficient_map: Option<Vec<Vec<i64>>>,
    baseline_map: Option<Vec<usize>>,
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

    let coefficient_map = match coefficient_map {
        Some(values) => values,
        None => {
            let expanded_width = width.checked_mul(transitions.len()).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("expanded Cox design width is too large")
            })?;
            i64::try_from(expanded_width).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("expanded Cox design width is too large")
            })?;
            (0..transitions.len())
                .map(|transition_idx| {
                    (0..width)
                        .map(|column| (transition_idx * width + column) as i64)
                        .collect()
                })
                .collect()
        }
    };
    validate_parallel_length("coefficient_map", coefficient_map.len(), transitions.len())?;
    let mut largest_coefficient = None;
    for (transition_idx, map) in coefficient_map.iter().enumerate() {
        if map.len() != width {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coefficient_map row {transition_idx} must have length {width}; got {}",
                map.len()
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for &coefficient in map {
            if coefficient < -1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "coefficient_map values must be -1 or non-negative",
                ));
            }
            if coefficient >= 0 {
                let coefficient = coefficient as usize;
                if !seen.insert(coefficient) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "coefficient_map row {transition_idx} maps multiple inputs to coefficient {coefficient}"
                    )));
                }
                largest_coefficient = Some(
                    largest_coefficient
                        .map_or(coefficient, |largest: usize| largest.max(coefficient)),
                );
            }
        }
    }
    let expanded_width = largest_coefficient.map_or(0, |largest| largest + 1);
    if expanded_width > 0 {
        let mut present = vec![false; expanded_width];
        for &coefficient in coefficient_map.iter().flatten() {
            if coefficient >= 0 {
                present[coefficient as usize] = true;
            }
        }
        if let Some(missing) = present.iter().position(|value| !value) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coefficient_map does not contain coefficient {missing}"
            )));
        }
    }

    let baseline_map = baseline_map.unwrap_or_else(|| (0..transitions.len()).collect());
    validate_parallel_length("baseline_map", baseline_map.len(), transitions.len())?;
    let baseline_count = baseline_map
        .iter()
        .max()
        .copied()
        .map_or(0, |largest| largest + 1);
    let mut present_baselines = vec![false; baseline_count];
    for &baseline in &baseline_map {
        present_baselines[baseline] = true;
    }
    if let Some(missing) = present_baselines.iter().position(|value| !value) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "baseline_map does not contain baseline {missing}"
        )));
    }

    let user_strata = strata.as_deref();
    let user_strata_count = user_strata
        .and_then(|values| values.iter().max().copied())
        .map_or(1, |maximum| maximum + 1);
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
            for (column, &coefficient) in coefficient_map[transition_idx].iter().enumerate() {
                if coefficient >= 0 {
                    row[coefficient as usize] = covariates[row_idx][column];
                }
            }
            expanded_covariates.push(row);
            expanded_strata.push(
                baseline_map[transition_idx] * user_strata_count
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

fn identity_matrix(size: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; size]; size];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    result
}

fn matrix_product(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let size = left.len();
    let mut result = vec![vec![0.0; size]; size];
    for (left_row, result_row) in left.iter().zip(result.iter_mut()) {
        for (middle, &left_value) in left_row.iter().enumerate() {
            if left_value == 0.0 {
                continue;
            }
            for (column, result_value) in result_row.iter_mut().enumerate() {
                *result_value += left_value * right[middle][column];
            }
        }
    }
    result
}

fn matrix_infinity_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn matrix_exponential(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let size = matrix.len();
    let norm = matrix_infinity_norm(matrix);
    if norm == 0.0 {
        return identity_matrix(size);
    }
    let squarings = if norm <= 0.5 {
        0
    } else {
        (norm / 0.5).log2().ceil() as u32
    };
    let scale = 2.0_f64.powi(-(squarings as i32));
    let scaled = matrix
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut result = identity_matrix(size);
    let mut term = identity_matrix(size);
    for order in 1..=128 {
        term = matrix_product(&term, &scaled);
        let divisor = order as f64;
        for row in &mut term {
            for value in row {
                *value /= divisor;
            }
        }
        for (result_row, term_row) in result.iter_mut().zip(&term) {
            for (result_value, term_value) in result_row.iter_mut().zip(term_row) {
                *result_value += term_value;
            }
        }
        if matrix_infinity_norm(&term) <= 1e-16 * matrix_infinity_norm(&result).max(1.0) {
            break;
        }
    }
    for _ in 0..squarings {
        result = matrix_product(&result, &result);
    }
    result
}

fn validate_curve_structure(
    hazard_increments: &[Vec<f64>],
    transitions: &[Vec<usize>],
    p0: &[f64],
) -> PyResult<()> {
    let transition_count = transitions.len();
    if transitions.iter().any(|transition| transition.len() != 2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each transition must contain source and target state indices",
        ));
    }
    if p0.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "p0 must contain at least one state probability",
        ));
    }
    if p0.iter().any(|value| !value.is_finite() || *value < 0.0)
        || (p0.iter().sum::<f64>() - 1.0).abs() > 1e-8
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "p0 must contain non-negative finite probabilities that sum to 1",
        ));
    }
    for (transition_idx, transition) in transitions.iter().enumerate() {
        if transition[0] >= p0.len() || transition[1] >= p0.len() || transition[0] == transition[1]
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transition {transition_idx} contains invalid state indices"
            )));
        }
    }
    if hazard_increments
        .iter()
        .any(|row| row.len() != transition_count)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hazard_increments must have one column per transition",
        ));
    }
    if hazard_increments
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hazard_increments must contain non-negative finite values",
        ));
    }
    Ok(())
}

fn validate_curve_risk(risk: &[f64], transition_count: usize) -> PyResult<()> {
    validate_parallel_length("risk", risk.len(), transition_count)?;
    if risk.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "risk must contain non-negative finite values",
        ));
    }
    Ok(())
}

fn compute_multistate_curve(
    hazard_increments: &[Vec<f64>],
    transitions: &[Vec<usize>],
    risk: &[f64],
    p0: &[f64],
    exponential: bool,
) -> MultiStateCoxCurve {
    let transition_count = transitions.len();
    let state_count = p0.len();
    let mut probabilities = p0.to_vec();
    let mut cumulative_hazard = vec![0.0; transition_count];
    let mut pstate = Vec::with_capacity(hazard_increments.len());
    let mut cumhaz = Vec::with_capacity(hazard_increments.len());
    for increments in hazard_increments {
        let mut generator = vec![vec![0.0; state_count]; state_count];
        for (transition_idx, transition) in transitions.iter().enumerate() {
            let increment = increments[transition_idx] * risk[transition_idx];
            cumulative_hazard[transition_idx] += increment;
            generator[transition[0]][transition[1]] += increment;
            generator[transition[0]][transition[0]] -= increment;
        }
        let update = if exponential {
            matrix_exponential(&generator)
        } else {
            let mut direct = identity_matrix(state_count);
            for (direct_row, generator_row) in direct.iter_mut().zip(generator) {
                for (direct_value, generator_value) in direct_row.iter_mut().zip(generator_row) {
                    *direct_value += generator_value;
                }
            }
            direct
        };
        let mut next = vec![0.0; state_count];
        for (source, probability) in probabilities.iter().enumerate() {
            for (target, value) in next.iter_mut().enumerate() {
                *value += probability * update[source][target];
            }
        }
        probabilities = next;
        pstate.push(probabilities.clone());
        cumhaz.push(cumulative_hazard.clone());
    }

    MultiStateCoxCurve { pstate, cumhaz }
}

/// Apply direct or matrix-exponential transition updates to a starting state mixture.
#[pyfunction]
#[pyo3(signature = (hazard_increments, transitions, risk, p0, exponential=true))]
pub fn cox_multistate_curve(
    hazard_increments: Vec<Vec<f64>>,
    transitions: Vec<Vec<usize>>,
    risk: Vec<f64>,
    p0: Vec<f64>,
    exponential: bool,
) -> PyResult<MultiStateCoxCurve> {
    validate_curve_structure(&hazard_increments, &transitions, &p0)?;
    validate_curve_risk(&risk, transitions.len())?;
    Ok(compute_multistate_curve(
        &hazard_increments,
        &transitions,
        &risk,
        &p0,
        exponential,
    ))
}

/// Apply multi-state transition updates to multiple covariate-profile risk vectors.
#[pyfunction]
#[pyo3(signature = (hazard_increments, transitions, risks, p0, exponential=true))]
pub fn cox_multistate_curves(
    hazard_increments: Vec<Vec<f64>>,
    transitions: Vec<Vec<usize>>,
    risks: Vec<Vec<f64>>,
    p0: Vec<f64>,
    exponential: bool,
) -> PyResult<MultiStateCoxCurves> {
    validate_curve_structure(&hazard_increments, &transitions, &p0)?;
    if risks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "risks must contain at least one covariate profile",
        ));
    }
    for risk in &risks {
        validate_curve_risk(risk, transitions.len())?;
    }

    let curves = risks
        .iter()
        .map(|risk| {
            compute_multistate_curve(&hazard_increments, &transitions, risk, &p0, exponential)
        })
        .collect::<Vec<_>>();
    let (pstate, cumhaz) = curves
        .into_iter()
        .map(|curve| (curve.pstate, curve.cumhaz))
        .unzip();
    Ok(MultiStateCoxCurves { pstate, cumhaz })
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
            None,
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
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.start, Some(vec![0.0, 0.0, 1.0]));
        assert_eq!(result.stop, vec![1.0, 2.0, 2.0]);
        assert_eq!(result.status, vec![1, 0, 0]);
        assert_eq!(result.strata, vec![0, 1, 3]);
        assert_eq!(result.source_rows, vec![0, 2, 1]);
    }

    #[test]
    fn stacks_with_omitted_and_shared_coefficients_and_baselines() {
        let result = cox_multistate_stack(
            None,
            vec![1.0, 2.0],
            vec![2, 3],
            vec![0, 0],
            vec![vec![1.0, 10.0], vec![2.0, 20.0]],
            vec![vec![0, 1], vec![0, 2]],
            Some(vec![0, 1]),
            Some(vec![vec![0, -1], vec![0, 1]]),
            Some(vec![0, 0]),
        )
        .unwrap();

        assert_eq!(result.strata, vec![0, 1, 0, 1]);
        assert_eq!(
            result.covariates,
            vec![
                vec![1.0, 0.0],
                vec![2.0, 0.0],
                vec![1.0, 10.0],
                vec![2.0, 20.0],
            ]
        );
    }

    #[test]
    fn direct_curve_applies_transition_increment_matrix() {
        let result = cox_multistate_curve(
            vec![vec![0.1, 0.2]],
            vec![vec![0, 1], vec![0, 2]],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            false,
        )
        .unwrap();

        assert_eq!(result.pstate, vec![vec![0.7, 0.1, 0.2]]);
        assert_eq!(result.cumhaz, vec![vec![0.1, 0.2]]);
    }

    #[test]
    fn exponential_curve_preserves_probability_mass() {
        let result = cox_multistate_curve(
            vec![vec![0.1, 0.2]],
            vec![vec![0, 1], vec![0, 2]],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            true,
        )
        .unwrap();

        let remaining = (-0.3_f64).exp();
        assert!((result.pstate[0][0] - remaining).abs() < 1e-14);
        assert!((result.pstate[0][1] - (1.0 - remaining) / 3.0).abs() < 1e-14);
        assert!((result.pstate[0][2] - 2.0 * (1.0 - remaining) / 3.0).abs() < 1e-14);
        assert!((result.pstate[0].iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn batched_curves_match_individual_profile_updates() {
        let hazard_increments = vec![vec![0.1, 0.2], vec![0.05, 0.0]];
        let transitions = vec![vec![0, 1], vec![0, 2]];
        let risks = vec![vec![1.0, 1.0], vec![2.0, 0.5]];
        let p0 = vec![1.0, 0.0, 0.0];
        let result = cox_multistate_curves(
            hazard_increments.clone(),
            transitions.clone(),
            risks.clone(),
            p0.clone(),
            true,
        )
        .unwrap();

        for (profile, risk) in risks.into_iter().enumerate() {
            let individual = cox_multistate_curve(
                hazard_increments.clone(),
                transitions.clone(),
                risk,
                p0.clone(),
                true,
            )
            .unwrap();
            assert_eq!(result.pstate[profile], individual.pstate);
            assert_eq!(result.cumhaz[profile], individual.cumhaz);
        }
    }
}
