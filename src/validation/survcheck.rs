use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

fn sorted_issue_ids(ids: &HashSet<i64>) -> Vec<i64> {
    let mut values: Vec<i64> = ids.iter().copied().collect();
    values.sort_unstable();
    values
}

fn sorted_issue_rows(rows: &HashSet<usize>) -> Vec<usize> {
    let mut values: Vec<usize> = rows.iter().copied().collect();
    values.sort_unstable();
    values
}

fn initial_state_at(istate: Option<&[i32]>, idx: usize) -> i32 {
    istate.map_or(0, |values| values[idx])
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvCheckResult {
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub n_observations: usize,
    #[pyo3(get)]
    pub n_transitions: usize,
    #[pyo3(get)]
    pub n_problems: usize,
    #[pyo3(get)]
    pub overlap_ids: Vec<i64>,
    #[pyo3(get)]
    pub overlap_rows: Vec<usize>,
    #[pyo3(get)]
    pub gap_ids: Vec<i64>,
    #[pyo3(get)]
    pub gap_rows: Vec<usize>,
    #[pyo3(get)]
    pub jump_ids: Vec<i64>,
    #[pyo3(get)]
    pub jump_rows: Vec<usize>,
    #[pyo3(get)]
    pub teleport_ids: Vec<i64>,
    #[pyo3(get)]
    pub teleport_rows: Vec<usize>,
    #[pyo3(get)]
    pub invalid_ids: Vec<i64>,
    #[pyo3(get)]
    pub invalid_rows: Vec<usize>,
    #[pyo3(get)]
    pub transitions: HashMap<String, usize>,
    #[pyo3(get)]
    pub flags: Vec<i32>,
    #[pyo3(get)]
    pub current_states: Vec<i32>,
    #[pyo3(get)]
    pub is_valid: bool,
    #[pyo3(get)]
    pub messages: Vec<String>,
}

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

    let istate = istate.as_deref();
    if istate.is_some_and(|values| values.len() != n) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "istate must have same length as other inputs",
        ));
    }

    if n == 0 {
        return Ok(SurvCheckResult {
            n_subjects: 0,
            n_observations: 0,
            n_transitions: 0,
            n_problems: 0,
            overlap_ids: vec![],
            overlap_rows: vec![],
            gap_ids: vec![],
            gap_rows: vec![],
            jump_ids: vec![],
            jump_rows: vec![],
            teleport_ids: vec![],
            teleport_rows: vec![],
            invalid_ids: vec![],
            invalid_rows: vec![],
            transitions: HashMap::new(),
            flags: vec![],
            current_states: vec![],
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
    let mut overlap_rows = HashSet::new();
    let mut gap_ids = HashSet::new();
    let mut gap_rows = HashSet::new();
    let mut jump_ids = HashSet::new();
    let mut jump_rows = HashSet::new();
    let mut teleport_ids = HashSet::new();
    let mut teleport_rows = HashSet::new();
    let mut invalid_ids = HashSet::new();
    let mut invalid_rows = HashSet::new();
    let mut transitions: HashMap<String, usize> = HashMap::new();
    let mut current_states = vec![0i32; n];
    let mut messages = Vec::new();
    let mut n_transitions = 0;

    for (&subj_id, indices) in &subject_obs {
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_by(|&a, &b| {
            time2[a]
                .total_cmp(&time2[b])
                .then_with(|| time1[a].total_cmp(&time1[b]))
        });

        let mut prev_end: Option<f64> = None;
        let mut expected_state: Option<i32> = None;

        for &idx in &sorted_indices {
            let t1 = time1[idx];
            let t2 = time2[idx];
            let state = status[idx];
            let declared_state = istate.map_or_else(
                || expected_state.unwrap_or(0),
                |values| initial_state_at(Some(values), idx),
            );
            let current_state = expected_state.unwrap_or(declared_state);
            current_states[idx] = current_state;

            if !t1.is_finite() || !t2.is_finite() {
                flags[idx] = 4;
                invalid_ids.insert(subj_id);
                invalid_rows.insert(idx);
                messages.push(format!(
                    "Subject {}: non-finite time at observation {}",
                    subj_id, idx
                ));
                continue;
            }

            if t2 < t1 {
                flags[idx] = 4;
                invalid_ids.insert(subj_id);
                invalid_rows.insert(idx);
                messages.push(format!(
                    "Subject {}: time2 ({}) < time1 ({}) at observation {}",
                    subj_id, t2, t1, idx
                ));
                continue;
            }

            if let Some(previous_end) = prev_end {
                let mismatch = expected_state.is_some_and(|value| declared_state != value);
                if t1 < previous_end - 1e-10 {
                    flags[idx] = 1;
                    overlap_ids.insert(subj_id);
                    overlap_rows.insert(idx);
                    messages.push(format!(
                        "Subject {}: overlapping intervals at observation {}",
                        subj_id, idx
                    ));
                } else if t1 > previous_end + 1e-10 {
                    if mismatch {
                        flags[idx] = 5;
                        jump_ids.insert(subj_id);
                        jump_rows.insert(idx);
                        messages.push(format!(
                            "Subject {}: state jump across gap at observation {}",
                            subj_id, idx
                        ));
                    } else {
                        flags[idx] = 2;
                        gap_ids.insert(subj_id);
                        gap_rows.insert(idx);
                        messages.push(format!(
                            "Subject {}: gap from {} to {} at observation {}",
                            subj_id, previous_end, t1, idx
                        ));
                    }
                } else if mismatch {
                    flags[idx] = 3;
                    teleport_ids.insert(subj_id);
                    teleport_rows.insert(idx);
                    messages.push(format!(
                        "Subject {}: state teleport at time {} (observation {})",
                        subj_id, t1, idx
                    ));
                }
            }

            if state > 0 {
                let trans_key = format!("{} -> {}", current_state, state);
                *transitions.entry(trans_key).or_insert(0) += 1;
                n_transitions += 1;
            }

            prev_end = Some(t2);
            expected_state = Some(if state > 0 { state } else { current_state });
        }
    }

    let mut problem_ids = HashSet::new();
    problem_ids.extend(overlap_ids.iter().copied());
    problem_ids.extend(gap_ids.iter().copied());
    problem_ids.extend(jump_ids.iter().copied());
    problem_ids.extend(teleport_ids.iter().copied());
    problem_ids.extend(invalid_ids.iter().copied());

    let n_problems = problem_ids.len();
    let is_valid = n_problems == 0;

    if is_valid {
        messages.push(format!(
            "Data passed all checks: {} subjects, {} transitions",
            n_subjects, n_transitions
        ));
    }

    Ok(SurvCheckResult {
        n_subjects,
        n_observations: n,
        n_transitions,
        n_problems,
        overlap_ids: sorted_issue_ids(&overlap_ids),
        overlap_rows: sorted_issue_rows(&overlap_rows),
        gap_ids: sorted_issue_ids(&gap_ids),
        gap_rows: sorted_issue_rows(&gap_rows),
        jump_ids: sorted_issue_ids(&jump_ids),
        jump_rows: sorted_issue_rows(&jump_rows),
        teleport_ids: sorted_issue_ids(&teleport_ids),
        teleport_rows: sorted_issue_rows(&teleport_rows),
        invalid_ids: sorted_issue_ids(&invalid_ids),
        invalid_rows: sorted_issue_rows(&invalid_rows),
        transitions,
        flags,
        current_states,
        is_valid,
        messages,
    })
}

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
        if !t.is_finite() {
            if flags[i] == 0 {
                flags[i] = 4;
                invalid_count += 1;
            }
            messages.push(format!("Observation {}: time is non-finite", i));
        }
    }

    let is_valid = invalid_count == 0;

    if is_valid {
        messages.push(format!("Data passed all checks: {} observations", n));
    }

    let n_events = status.iter().filter(|&&s| s == 1).count();
    let mut transitions = HashMap::new();
    if n_events > 0 {
        transitions.insert("0 -> 1".to_string(), n_events);
    }

    let invalid_rows: Vec<usize> = (0..n).filter(|&i| flags[i] != 0).collect();

    Ok(SurvCheckResult {
        n_subjects: n,
        n_observations: n,
        n_transitions: n_events,
        n_problems: invalid_count,
        overlap_ids: vec![],
        overlap_rows: vec![],
        gap_ids: vec![],
        gap_rows: vec![],
        jump_ids: vec![],
        jump_rows: vec![],
        teleport_ids: vec![],
        teleport_rows: vec![],
        invalid_ids: (0..n)
            .filter(|&i| flags[i] != 0)
            .map(|i| i as i64)
            .collect(),
        invalid_rows,
        transitions,
        flags,
        current_states: vec![0; n],
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
    fn test_survcheck_default_istate_matches_explicit_zero_state() {
        let id = vec![1, 1, 2, 2];
        let time1 = vec![0.0, 10.0, 0.0, 8.0];
        let time2 = vec![10.0, 20.0, 5.0, 12.0];
        let status = vec![0, 1, 0, 1];
        let explicit_zero = vec![0; id.len()];

        let default = survcheck(
            id.clone(),
            time1.clone(),
            time2.clone(),
            status.clone(),
            None,
        )
        .unwrap();
        let explicit = survcheck(id, time1, time2, status, Some(explicit_zero)).unwrap();

        assert_eq!(default.n_subjects, explicit.n_subjects);
        assert_eq!(default.n_observations, explicit.n_observations);
        assert_eq!(default.n_transitions, explicit.n_transitions);
        assert_eq!(default.n_problems, explicit.n_problems);
        assert_eq!(default.overlap_ids, explicit.overlap_ids);
        assert_eq!(default.overlap_rows, explicit.overlap_rows);
        assert_eq!(default.gap_ids, explicit.gap_ids);
        assert_eq!(default.gap_rows, explicit.gap_rows);
        assert_eq!(default.jump_ids, explicit.jump_ids);
        assert_eq!(default.jump_rows, explicit.jump_rows);
        assert_eq!(default.teleport_ids, explicit.teleport_ids);
        assert_eq!(default.teleport_rows, explicit.teleport_rows);
        assert_eq!(default.invalid_ids, explicit.invalid_ids);
        assert_eq!(default.invalid_rows, explicit.invalid_rows);
        assert_eq!(default.transitions, explicit.transitions);
        assert_eq!(default.flags, explicit.flags);
        assert_eq!(default.current_states, explicit.current_states);
        assert_eq!(default.is_valid, explicit.is_valid);
        assert_eq!(default.messages, explicit.messages);
    }

    #[test]
    fn test_survcheck_overlap() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![10.0, 15.0];
        let status = vec![0, 1];

        let result = survcheck(id, time1, time2, status, None).unwrap();
        assert!(!result.is_valid);
        assert!(!result.overlap_ids.is_empty());
    }

    #[test]
    fn test_survcheck_gap() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 15.0];
        let time2 = vec![10.0, 20.0];
        let status = vec![0, 1];

        let result = survcheck(id, time1, time2, status, None).unwrap();
        assert!(!result.is_valid);
        assert!(!result.gap_ids.is_empty());
    }

    #[test]
    fn test_survcheck_counts_unique_problem_subjects() {
        let id = vec![2, 2, 1, 1, 1];
        let time1 = vec![0.0, 5.0, 0.0, 5.0, 20.0];
        let time2 = vec![10.0, 15.0, 10.0, 15.0, 25.0];
        let status = vec![0, 1, 0, 1, 0];

        let result = survcheck(id, time1, time2, status, None).unwrap();

        assert_eq!(result.n_problems, 2);
        assert_eq!(result.overlap_ids, vec![1, 2]);
        assert_eq!(result.gap_ids, vec![1]);
    }

    #[test]
    fn test_survcheck_propagates_states_and_counts_only_events() {
        let result = survcheck(
            vec![1, 1, 1],
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![1, 0, 1],
            None,
        )
        .unwrap();

        assert_eq!(result.current_states, vec![0, 1, 1]);
        assert_eq!(result.n_transitions, 2);
        assert_eq!(result.transitions.get("0 -> 1"), Some(&1));
        assert_eq!(result.transitions.get("1 -> 1"), Some(&1));
        assert!(result.teleport_rows.is_empty());
    }

    #[test]
    fn test_survcheck_distinguishes_jump_and_teleport_rows() {
        let jump = survcheck(
            vec![1, 1, 2, 2],
            vec![0.0, 1.0, 0.0, 2.0],
            vec![1.0, 2.0, 1.0, 3.0],
            vec![2, 3, 2, 3],
            Some(vec![1, 2, 1, 1]),
        )
        .unwrap();
        assert_eq!(jump.jump_ids, vec![2]);
        assert_eq!(jump.jump_rows, vec![3]);
        assert_eq!(jump.flags[3], 5);

        let teleport = survcheck(
            vec![1, 1],
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![2, 3],
            Some(vec![1, 1]),
        )
        .unwrap();
        assert_eq!(teleport.teleport_ids, vec![1]);
        assert_eq!(teleport.teleport_rows, vec![1]);
        assert_eq!(teleport.flags[1], 3);
    }

    #[test]
    fn test_survcheck_non_finite_times_are_invalid_without_cascading() {
        let result = survcheck(
            vec![1, 1, 1],
            vec![0.0, f64::NAN, 1.0],
            vec![1.0, 2.0, 2.0],
            vec![0, 1, 0],
            None,
        )
        .unwrap();

        assert!(!result.is_valid);
        assert_eq!(result.invalid_ids, vec![1]);
        assert_eq!(result.overlap_ids, Vec::<i64>::new());
        assert_eq!(result.gap_ids, Vec::<i64>::new());
        assert_eq!(result.flags, vec![0, 4, 0]);
        assert!(
            result
                .messages
                .iter()
                .any(|message| message.contains("non-finite time"))
        );
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

    #[test]
    fn test_survcheck_simple_infinite_time_is_invalid() {
        let result = survcheck_simple(vec![1.0, f64::INFINITY], vec![1, 0]).unwrap();

        assert!(!result.is_valid);
        assert_eq!(result.invalid_ids, vec![1]);
        assert_eq!(result.flags, vec![0, 4]);
    }
}
