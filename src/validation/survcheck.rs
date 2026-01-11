use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Result of survival data validation
#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvCheckResult {
    /// Number of subjects in the data
    #[pyo3(get)]
    pub n_subjects: usize,
    /// Number of transitions observed
    #[pyo3(get)]
    pub n_transitions: usize,
    /// Number of subjects with issues
    #[pyo3(get)]
    pub n_problems: usize,
    /// IDs of subjects with overlapping intervals
    #[pyo3(get)]
    pub overlap_ids: Vec<i64>,
    /// IDs of subjects with gaps in time
    #[pyo3(get)]
    pub gap_ids: Vec<i64>,
    /// IDs of subjects with teleportation (state change without time)
    #[pyo3(get)]
    pub teleport_ids: Vec<i64>,
    /// IDs of subjects with invalid transitions
    #[pyo3(get)]
    pub invalid_ids: Vec<i64>,
    /// Transition counts: from_state -> to_state -> count
    #[pyo3(get)]
    pub transitions: HashMap<String, usize>,
    /// Flags for each observation (0=ok, 1=overlap, 2=gap, 3=teleport, 4=invalid)
    #[pyo3(get)]
    pub flags: Vec<i32>,
    /// Whether the data passed all checks
    #[pyo3(get)]
    pub is_valid: bool,
    /// Human-readable messages about issues found
    #[pyo3(get)]
    pub messages: Vec<String>,
}

/// Check survival data for consistency.
///
/// Validates multi-state survival data for common issues:
/// - Overlapping time intervals for the same subject
/// - Gaps in time sequences
/// - Teleportation (state changes without elapsed time)
/// - Invalid state transitions
///
/// # Arguments
/// * `id` - Subject identifiers
/// * `time1` - Start times for each interval
/// * `time2` - End times for each interval
/// * `status` - Event/state indicator at end of interval
/// * `istate` - Optional initial state at start of interval
///
/// # Returns
/// * `SurvCheckResult` with detailed validation results
#[pyfunction]
#[pyo3(signature = (id, time1, time2, status, istate=None))]
pub fn survcheck(
    id: Vec<i64>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    istate: Option<Vec<i32>>,
) -> PyResult<SurvCheckResult> {
    let n = id.len();

    if time1.len() != n || time2.len() != n || status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let initial_state = istate.unwrap_or_else(|| vec![0; n]);
    if initial_state.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "istate must have same length as other inputs",
        ));
    }

    if n == 0 {
        return Ok(SurvCheckResult {
            n_subjects: 0,
            n_transitions: 0,
            n_problems: 0,
            overlap_ids: vec![],
            gap_ids: vec![],
            teleport_ids: vec![],
            invalid_ids: vec![],
            transitions: HashMap::new(),
            flags: vec![],
            is_valid: true,
            messages: vec![],
        });
    }

    let mut subject_obs: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &subj_id) in id.iter().enumerate() {
        subject_obs.entry(subj_id).or_default().push(i);
    }

    let n_subjects = subject_obs.len();
    let mut flags = vec![0i32; n];
    let mut overlap_ids = HashSet::new();
    let mut gap_ids = HashSet::new();
    let mut teleport_ids = HashSet::new();
    let mut invalid_ids = HashSet::new();
    let mut transitions: HashMap<String, usize> = HashMap::new();
    let mut messages = Vec::new();
    let mut n_transitions = 0;

    for (&subj_id, indices) in &subject_obs {
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_by(|&a, &b| {
            time1[a]
                .partial_cmp(&time1[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut prev_end: Option<f64> = None;
        let mut prev_state: Option<i32> = None;

        for &idx in &sorted_indices {
            let t1 = time1[idx];
            let t2 = time2[idx];
            let state = status[idx];
            let istate_val = initial_state[idx];

            if t2 < t1 {
                flags[idx] = 4;
                invalid_ids.insert(subj_id);
                messages.push(format!(
                    "Subject {}: time2 ({}) < time1 ({}) at observation {}",
                    subj_id, t2, t1, idx
                ));
            }

            if let Some(pe) = prev_end {
                if t1 < pe {
                    flags[idx] = 1;
                    overlap_ids.insert(subj_id);
                    messages.push(format!(
                        "Subject {}: overlapping intervals at observation {}",
                        subj_id, idx
                    ));
                } else if t1 > pe + 1e-10 {
                    flags[idx] = 2;
                    gap_ids.insert(subj_id);
                    messages.push(format!(
                        "Subject {}: gap from {} to {} at observation {}",
                        subj_id, pe, t1, idx
                    ));
                }
            }

            if let Some(ps) = prev_state
                && let Some(pe) = prev_end
                && (t1 - pe).abs() < 1e-10
                && istate_val != ps
            {
                flags[idx] = 3;
                teleport_ids.insert(subj_id);
                messages.push(format!(
                    "Subject {}: teleport from state {} to {} at time {} (observation {})",
                    subj_id, ps, istate_val, t1, idx
                ));
            }

            let trans_key = format!("{} -> {}", istate_val, state);
            *transitions.entry(trans_key).or_insert(0) += 1;
            n_transitions += 1;

            prev_end = Some(t2);
            prev_state = Some(state);
        }
    }

    let n_problems = overlap_ids.len() + gap_ids.len() + teleport_ids.len() + invalid_ids.len();
    let is_valid = n_problems == 0;

    if is_valid {
        messages.push(format!(
            "Data passed all checks: {} subjects, {} transitions",
            n_subjects, n_transitions
        ));
    }

    Ok(SurvCheckResult {
        n_subjects,
        n_transitions,
        n_problems,
        overlap_ids: overlap_ids.into_iter().collect(),
        gap_ids: gap_ids.into_iter().collect(),
        teleport_ids: teleport_ids.into_iter().collect(),
        invalid_ids: invalid_ids.into_iter().collect(),
        transitions,
        flags,
        is_valid,
        messages,
    })
}

/// Simplified check for standard (non-multi-state) survival data
#[pyfunction]
pub fn survcheck_simple(time: Vec<f64>, status: Vec<i32>) -> PyResult<SurvCheckResult> {
    let n = time.len();

    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let mut messages = Vec::new();
    let mut flags = vec![0i32; n];
    let mut invalid_count = 0;

    for (i, &t) in time.iter().enumerate() {
        if t < 0.0 {
            flags[i] = 4;
            invalid_count += 1;
            messages.push(format!("Observation {}: negative time ({})", i, t));
        }
    }

    for (i, &s) in status.iter().enumerate() {
        if !(0..=1).contains(&s) {
            if flags[i] == 0 {
                flags[i] = 4;
                invalid_count += 1;
            }
            messages.push(format!(
                "Observation {}: invalid status value ({}), expected 0 or 1",
                i, s
            ));
        }
    }

    for (i, &t) in time.iter().enumerate() {
        if t.is_nan() {
            if flags[i] == 0 {
                flags[i] = 4;
                invalid_count += 1;
            }
            messages.push(format!("Observation {}: time is NaN", i));
        }
    }

    let is_valid = invalid_count == 0;

    if is_valid {
        messages.push(format!("Data passed all checks: {} observations", n));
    }

    let n_events = status.iter().filter(|&&s| s == 1).count();
    let mut transitions = HashMap::new();
    transitions.insert("0 -> 0".to_string(), n - n_events);
    transitions.insert("0 -> 1".to_string(), n_events);

    Ok(SurvCheckResult {
        n_subjects: n,
        n_transitions: n,
        n_problems: invalid_count,
        overlap_ids: vec![],
        gap_ids: vec![],
        teleport_ids: vec![],
        invalid_ids: (0..n)
            .filter(|&i| flags[i] != 0)
            .map(|i| i as i64)
            .collect(),
        transitions,
        flags,
        is_valid,
        messages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survcheck_valid_data() {
        let id = vec![1, 1, 2, 2];
        let time1 = vec![0.0, 10.0, 0.0, 5.0];
        let time2 = vec![10.0, 20.0, 5.0, 15.0];
        let status = vec![0, 1, 0, 1];

        let result = survcheck(id, time1, time2, status, None).unwrap();
        assert!(result.is_valid);
        assert_eq!(result.n_subjects, 2);
    }

    #[test]
    fn test_survcheck_overlap() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0]; // Second starts before first ends
        let time2 = vec![10.0, 15.0];
        let status = vec![0, 1];

        let result = survcheck(id, time1, time2, status, None).unwrap();
        assert!(!result.is_valid);
        assert!(!result.overlap_ids.is_empty());
    }

    #[test]
    fn test_survcheck_gap() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 15.0]; // Gap between 10 and 15
        let time2 = vec![10.0, 20.0];
        let status = vec![0, 1];

        let result = survcheck(id, time1, time2, status, None).unwrap();
        assert!(!result.is_valid);
        assert!(!result.gap_ids.is_empty());
    }

    #[test]
    fn test_survcheck_simple() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 1];

        let result = survcheck_simple(time, status).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_survcheck_simple_negative_time() {
        let time = vec![-1.0, 2.0, 3.0];
        let status = vec![1, 0, 1];

        let result = survcheck_simple(time, status).unwrap();
        assert!(!result.is_valid);
    }
}
