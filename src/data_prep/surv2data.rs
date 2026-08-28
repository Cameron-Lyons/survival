use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::constants::TIME_EPSILON;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_no_nan};

const NO_NEXT_ROW: usize = usize::MAX;

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Surv2DataResult {
    #[pyo3(get)]
    pub id: Vec<i32>,
    #[pyo3(get)]
    pub time1: Vec<f64>,
    #[pyo3(get)]
    pub time2: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub row_index: Vec<usize>,
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Surv2TimelineResult {
    #[pyo3(get)]
    pub row_index: Vec<usize>,
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub stop: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub istate: Vec<Option<i32>>,
}

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct FromTimelineRowsResult {
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub stop: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub istate: Vec<i32>,
    #[pyo3(get)]
    pub static_row: Vec<usize>,
    #[pyo3(get)]
    pub dynamic_row: Vec<usize>,
    #[pyo3(get)]
    pub removed_row: Vec<usize>,
}

fn validate_parallel_len(field: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        return Err(PyValueError::new_err(format!(
            "{field} must have the same length as id"
        )));
    }
    Ok(())
}

#[pyfunction]
pub fn from_timeline_rows(
    id: Vec<usize>,
    time: Vec<f64>,
    status: Vec<i32>,
) -> PyResult<FromTimelineRowsResult> {
    from_timeline_rows_with_repeated(id, time, status, true)
}

pub(crate) fn from_timeline_rows_with_repeated(
    id: Vec<usize>,
    time: Vec<f64>,
    status: Vec<i32>,
    repeated: bool,
) -> PyResult<FromTimelineRowsResult> {
    from_timeline_rows_with_policy(id, time, status, repeated, false)
}

pub(crate) fn from_timeline_rows_with_policy(
    id: Vec<usize>,
    time: Vec<f64>,
    mut status: Vec<i32>,
    repeated: bool,
    first_only: bool,
) -> PyResult<FromTimelineRowsResult> {
    let n = id.len();
    validate_parallel_len("time", time.len(), n)?;
    validate_parallel_len("status", status.len(), n)?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;

    let mut group_index: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (row, &subject) in id.iter().enumerate() {
        let index = match group_index.get(&subject) {
            Some(&index) => index,
            None => {
                let index = groups.len();
                group_index.insert(subject, index);
                groups.push(Vec::new());
                index
            }
        };
        groups[index].push(row);
    }

    let capacity = n.saturating_sub(groups.len());
    let mut next_row = vec![NO_NEXT_ROW; n];
    let mut has_initial_state = None;
    let mut state_by_row = Vec::new();
    let mut seen_events = HashSet::new();
    group_index.clear();
    for mut rows in groups {
        rows.sort_unstable_by(|&left, &right| {
            time[left]
                .total_cmp(&time[right])
                .then_with(|| left.cmp(&right))
        });
        if rows.windows(2).any(|pair| time[pair[0]] == time[pair[1]]) {
            return Err(PyValueError::new_err("duplicated time for an id"));
        }
        if first_only {
            seen_events.clear();
            for &row in &rows {
                let event = status[row];
                if event != 0 && !seen_events.insert(event) {
                    status[row] = 0;
                }
            }
        }
        let first_row = rows[0];
        let current_has_state = status[first_row] != 0;
        if has_initial_state.is_some_and(|expected| expected != current_has_state) {
            return Err(PyValueError::new_err(
                "everyone or no one should have an initial state",
            ));
        }
        if has_initial_state.is_none() {
            has_initial_state = Some(current_has_state);
            if current_has_state {
                state_by_row.resize(n, 0);
            }
        }
        if rows.len() < 2 {
            continue;
        }
        group_index.insert(id[first_row], first_row);
        if current_has_state {
            let mut current_state = status[first_row];
            for pair in rows.windows(2) {
                let row = pair[0];
                let next = pair[1];
                if status[row] != 0 {
                    current_state = status[row];
                }
                next_row[row] = next;
                state_by_row[row] = current_state;
            }
        } else {
            for pair in rows.windows(2) {
                next_row[pair[0]] = pair[1];
            }
        }
    }

    let has_initial_state = has_initial_state.unwrap_or(false);
    let mut result = FromTimelineRowsResult {
        start: Vec::with_capacity(capacity),
        stop: Vec::with_capacity(capacity),
        status: Vec::with_capacity(capacity),
        istate: Vec::with_capacity(if has_initial_state { capacity } else { 0 }),
        static_row: Vec::with_capacity(capacity),
        dynamic_row: Vec::with_capacity(capacity),
        removed_row: Vec::new(),
    };
    for (row, next) in next_row.into_iter().enumerate() {
        let Some(&first_row) = group_index.get(&id[row]) else {
            result.removed_row.push(row);
            continue;
        };
        if next != NO_NEXT_ROW {
            result.start.push(time[row]);
            result.stop.push(time[next]);
            let event = status[next];
            let current_state = has_initial_state.then(|| state_by_row[row]);
            result.status.push(
                if !first_only && !repeated && current_state == Some(event) {
                    0
                } else {
                    event
                },
            );
            if let Some(state) = current_state {
                result.istate.push(state);
            }
            result.static_row.push(first_row);
            result.dynamic_row.push(row);
        }
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (id, time, status, repeated=false))]
pub fn surv2data_timeline(
    id: Vec<i64>,
    time: Vec<f64>,
    status: Vec<Option<i32>>,
    repeated: bool,
) -> PyResult<Surv2TimelineResult> {
    surv2data_timeline_with_policy(id, time, status, repeated, false)
}

pub(crate) fn surv2data_timeline_with_policy(
    id: Vec<i64>,
    time: Vec<f64>,
    status: Vec<Option<i32>>,
    repeated: bool,
    first_only: bool,
) -> PyResult<Surv2TimelineResult> {
    surv2data_timeline_with_options(id, time, status, repeated, first_only, true)
}

pub(crate) fn surv2data_timeline_with_options(
    id: Vec<i64>,
    time: Vec<f64>,
    mut status: Vec<Option<i32>>,
    repeated: bool,
    first_only: bool,
    multistate: bool,
) -> PyResult<Surv2TimelineResult> {
    let n = id.len();
    if time.len() != n || status.len() != n {
        return Err(PyValueError::new_err(
            "id, time, and status must have the same length",
        ));
    }
    validate_no_nan(&time, "time")?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&left, &right| {
        id[left]
            .cmp(&id[right])
            .then_with(|| time[left].total_cmp(&time[right]))
            .then_with(|| left.cmp(&right))
    });

    let mut next_row = vec![NO_NEXT_ROW; n];
    let mut n_intervals = 0;
    let mut has_initial_state = None;
    let mut seen_events = HashSet::new();
    for (position, &current) in order.iter().enumerate() {
        if position == 0 || id[order[position - 1]] != id[current] {
            if first_only {
                seen_events.clear();
            }
            if multistate {
                let current_has_state = status[current].is_some_and(|value| value != 0);
                if has_initial_state.is_some_and(|expected| expected != current_has_state) {
                    return Err(PyValueError::new_err(
                        "everyone or no one should have an initial state",
                    ));
                }
                has_initial_state = Some(current_has_state);
            }
        }
        if first_only {
            let event = status[current].unwrap_or(0);
            if event != 0 && !seen_events.insert(event) {
                status[current] = Some(0);
            }
        }

        let Some(&next) = order.get(position + 1) else {
            continue;
        };
        if id[current] != id[next] {
            continue;
        }
        if time[current] == time[next] {
            return Err(PyValueError::new_err(
                "duplicated time values for a single id",
            ));
        }
        next_row[current] = next;
        n_intervals += 1;
    }

    let has_initial_state = multistate && has_initial_state.unwrap_or(false);
    let mut state_by_row = if has_initial_state {
        vec![0; n]
    } else {
        Vec::new()
    };
    if has_initial_state {
        let mut current_id = None;
        let mut current_state = 0;
        for &row in &order {
            if current_id != Some(id[row]) {
                current_id = Some(id[row]);
                current_state = status[row].unwrap_or(0);
            } else if let Some(state) = status[row].filter(|&value| value != 0) {
                current_state = state;
            }
            state_by_row[row] = current_state;
        }
    }

    let mut result = Surv2TimelineResult {
        row_index: Vec::with_capacity(n_intervals),
        start: Vec::with_capacity(n_intervals),
        stop: Vec::with_capacity(n_intervals),
        status: Vec::with_capacity(n_intervals),
        istate: Vec::with_capacity(n_intervals),
    };
    for (row_index, next) in next_row.into_iter().enumerate() {
        if next == NO_NEXT_ROW {
            continue;
        }
        let istate = has_initial_state.then(|| state_by_row[row_index]);
        let mut event = status[next].unwrap_or(0);
        if multistate && !first_only && !repeated && istate == Some(event) {
            event = 0;
        }
        result.row_index.push(row_index);
        result.start.push(time[row_index]);
        result.stop.push(time[next]);
        result.status.push(event);
        result.istate.push(istate);
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (id, time, event_time=None, event_status=None))]
pub fn surv2data(
    id: Vec<i32>,
    time: Vec<f64>,
    event_time: Option<Vec<f64>>,
    event_status: Option<Vec<i32>>,
) -> PyResult<Surv2DataResult> {
    let n = id.len();
    if time.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "time must have same length as id",
        ));
    }
    match (&event_time, &event_status) {
        (Some(etimes), Some(estatus)) => {
            if etimes.len() != n {
                return Err(PyErr::new::<PyValueError, _>(
                    "event_time must have same length as id",
                ));
            }
            if estatus.len() != n {
                return Err(PyErr::new::<PyValueError, _>(
                    "event_status must have same length as id",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                "event_time and event_status must both be provided or both be None",
            ));
        }
    }

    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    if let Some(etimes) = &event_time {
        validate_no_nan(etimes, "event_time")?;
        validate_finite(etimes, "event_time")?;
    }
    if let Some(estatus) = &event_status {
        validate_binary_i32(estatus, "event_status")?;
    }

    if n == 0 {
        return Ok(Surv2DataResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
            row_index: Vec::new(),
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| match id[a].cmp(&id[b]) {
        std::cmp::Ordering::Equal => time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)),
        other => other,
    });

    let mut subject_event: HashMap<i32, (f64, i32)> = HashMap::new();
    if let (Some(etimes), Some(estatus)) = (&event_time, &event_status) {
        for i in 0..n {
            let subj_id = id[i];
            if etimes[i] + TIME_EPSILON < time[i] {
                return Err(PyValueError::new_err(format!(
                    "event_time must be >= time for each row; got event_time {} before time {} at index {}",
                    etimes[i], time[i], i
                )));
            }
            match subject_event.get(&subj_id) {
                Some(&(event_time, event_status))
                    if (event_time - etimes[i]).abs() > TIME_EPSILON
                        || event_status != estatus[i] =>
                {
                    return Err(PyValueError::new_err(format!(
                        "event_time/event_status must be constant within id; id {} has conflicting event metadata at index {}",
                        subj_id, i
                    )));
                }
                Some(_) => {}
                None => {
                    subject_event.insert(subj_id, (etimes[i], estatus[i]));
                }
            }
        }
    }

    let mut result = Surv2DataResult {
        id: Vec::with_capacity(n),
        time1: Vec::with_capacity(n),
        time2: Vec::with_capacity(n),
        status: Vec::with_capacity(n),
        row_index: Vec::with_capacity(n),
    };

    let mut i = 0;
    while i < n {
        let start_idx = indices[i];
        let current_id = id[start_idx];

        let mut subject_times: Vec<(f64, usize)> = Vec::new();
        let mut j = i;
        while j < n && id[indices[j]] == current_id {
            subject_times.push((time[indices[j]], indices[j]));
            j += 1;
        }

        for pair in subject_times.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(PyValueError::new_err(
                    "duplicated time values for a single id",
                ));
            }
        }

        let (subj_event_time, subj_event_status) = subject_event
            .get(&current_id)
            .copied()
            .unwrap_or((f64::INFINITY, 0));

        for k in 0..subject_times.len() {
            let (t1, orig_idx) = subject_times[k];

            let t2 = if k + 1 < subject_times.len() {
                subject_times[k + 1].0
            } else if subj_event_time > t1 {
                subj_event_time
            } else {
                t1
            };

            if t2 <= t1 {
                continue;
            }

            let interval_status = if k + 1 >= subject_times.len() {
                subj_event_status
            } else {
                0
            };

            result.id.push(current_id);
            result.time1.push(t1);
            result.time2.push(t2);
            result.status.push(interval_status);
            result.row_index.push(orig_idx + 1);
        }

        i = j;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::{index_permutations, initialize_python};

    #[test]
    fn test_surv2data_basic() {
        let id = vec![1, 1, 1];
        let time = vec![0.0, 5.0, 10.0];
        let event_time = Some(vec![15.0, 15.0, 15.0]);
        let event_status = Some(vec![1, 1, 1]);

        let result = surv2data(id, time, event_time, event_status).unwrap();

        assert_eq!(result.id.len(), 3);
        assert_eq!(result.time1, vec![0.0, 5.0, 10.0]);
        assert_eq!(result.time2, vec![5.0, 10.0, 15.0]);
        assert_eq!(result.status, vec![0, 0, 1]);
    }

    #[test]
    fn test_surv2data_multiple_subjects() {
        let id = vec![1, 1, 2, 2];
        let time = vec![0.0, 5.0, 0.0, 3.0];
        let event_time = Some(vec![10.0, 10.0, 8.0, 8.0]);
        let event_status = Some(vec![1, 1, 0, 0]);

        let result = surv2data(id, time, event_time, event_status).unwrap();

        assert_eq!(result.id.len(), 4);
    }

    #[test]
    fn test_surv2data_no_event_info() {
        let id = vec![1, 1, 1];
        let time = vec![0.0, 5.0, 10.0];

        let result = surv2data(id, time, None, None).unwrap();

        assert!(result.id.len() >= 2);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 5.0);
    }

    #[test]
    fn test_surv2data_is_invariant_to_input_order() {
        let base_id = [1, 1, 2, 2];
        let base_time = [0.0, 5.0, 0.0, 3.0];
        let base_event_time = [10.0, 10.0, 8.0, 8.0];
        let base_event_status = [1, 1, 0, 0];
        let expected = vec![
            (1, 0.0, 5.0, 0),
            (1, 5.0, 10.0, 1),
            (2, 0.0, 3.0, 0),
            (2, 3.0, 8.0, 0),
        ];

        for permutation in index_permutations(base_id.len()) {
            let id: Vec<i32> = permutation.iter().map(|&i| base_id[i]).collect();
            let time: Vec<f64> = permutation.iter().map(|&i| base_time[i]).collect();
            let event_time: Vec<f64> = permutation.iter().map(|&i| base_event_time[i]).collect();
            let event_status: Vec<i32> =
                permutation.iter().map(|&i| base_event_status[i]).collect();

            let result = surv2data(
                id.clone(),
                time.clone(),
                Some(event_time),
                Some(event_status),
            )
            .unwrap();
            let intervals: Vec<(i32, f64, f64, i32)> = result
                .id
                .iter()
                .zip(&result.time1)
                .zip(&result.time2)
                .zip(&result.status)
                .map(|(((id, time1), time2), status)| (*id, *time1, *time2, *status))
                .collect();

            assert_eq!(intervals, expected);
            assert_eq!(result.row_index.len(), result.id.len());
            for idx in 0..result.row_index.len() {
                let original = result.row_index[idx] - 1;
                assert_eq!(id[original], result.id[idx]);
                assert_eq!(time[original], result.time1[idx]);
            }
        }
    }

    #[test]
    fn test_surv2data_rejects_mismatched_inputs() {
        assert!(surv2data(vec![1], vec![], None, None).is_err());
        assert!(surv2data(vec![1], vec![0.0], Some(vec![]), Some(vec![1])).is_err());
        assert!(surv2data(vec![1], vec![0.0], Some(vec![1.0]), Some(vec![])).is_err());
        assert!(surv2data(vec![1], vec![0.0], Some(vec![1.0]), None).is_err());
    }

    #[test]
    fn test_surv2data_rejects_malformed_values() {
        initialize_python();

        let err = surv2data(vec![1], vec![f64::NAN], None, None).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err =
            surv2data(vec![1], vec![0.0], Some(vec![f64::INFINITY]), Some(vec![1])).unwrap_err();
        assert!(err.to_string().contains("event_time contains non-finite"));

        let err = surv2data(vec![1], vec![0.0], Some(vec![1.0]), Some(vec![2])).unwrap_err();
        assert!(
            err.to_string()
                .contains("event_status must contain only 0/1 values")
        );

        let err = surv2data(
            vec![1, 1],
            vec![0.0, 1.0],
            Some(vec![3.0, 4.0]),
            Some(vec![1, 1]),
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("event_time/event_status must be constant within id")
        );

        let err = surv2data(vec![1], vec![2.0], Some(vec![1.0]), Some(vec![1])).unwrap_err();
        assert!(err.to_string().contains("event_time must be >= time"));

        let err = surv2data(vec![1, 1], vec![2.0, 2.0], None, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("duplicated time values for a single id")
        );
    }

    #[test]
    fn surv2data_timeline_builds_original_row_order_and_suppresses_stutter() {
        let result = surv2data_timeline(
            vec![1, 2, 1, 1, 2],
            vec![0.0, 0.0, 5.0, 2.0, 3.0],
            vec![Some(1), Some(1), Some(3), Some(2), None],
            false,
        )
        .unwrap();

        assert_eq!(result.row_index, vec![0, 1, 3]);
        assert_eq!(result.start, vec![0.0, 0.0, 2.0]);
        assert_eq!(result.stop, vec![2.0, 3.0, 5.0]);
        assert_eq!(result.status, vec![2, 0, 3]);
        assert_eq!(result.istate, vec![Some(1), Some(1), Some(2)]);

        let stutter = surv2data_timeline(
            vec![1, 1, 1],
            vec![0.0, 1.0, 2.0],
            vec![Some(1), Some(1), Some(2)],
            false,
        )
        .unwrap();
        let repeated = surv2data_timeline(
            vec![1, 1, 1],
            vec![0.0, 1.0, 2.0],
            vec![Some(1), Some(1), Some(2)],
            true,
        )
        .unwrap();
        assert_eq!(stutter.status, vec![0, 2]);
        assert_eq!(repeated.status, vec![1, 2]);
    }

    #[test]
    fn surv2data_timeline_validates_lengths_missing_times_and_duplicates() {
        assert!(surv2data_timeline(vec![1], vec![], vec![Some(1)], false).is_err());
        assert!(surv2data_timeline(vec![1], vec![f64::NAN], vec![Some(1)], false).is_err());
        assert!(
            surv2data_timeline(
                vec![1, 1],
                vec![f64::INFINITY, f64::INFINITY],
                vec![Some(1), Some(2)],
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn surv2data_timeline_skips_singletons_and_preserves_input_order() {
        let result = surv2data_timeline(
            vec![9, 1, 2, 1, 8, 2],
            vec![5.0, 2.0, 3.0, 0.0, 1.0, 0.0],
            vec![Some(1), Some(2), Some(3), Some(1), Some(1), Some(1)],
            false,
        )
        .unwrap();

        assert_eq!(result.row_index, vec![3, 5]);
        assert_eq!(result.start, vec![0.0, 0.0]);
        assert_eq!(result.stop, vec![2.0, 3.0]);
        assert_eq!(result.status, vec![2, 3]);
        assert_eq!(result.istate, vec![Some(1), Some(1)]);
    }

    #[test]
    fn surv2data_timeline_carries_states_across_censored_rows() {
        let stutter = surv2data_timeline(
            vec![1, 1, 1, 1],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![Some(1), None, Some(1), Some(2)],
            false,
        )
        .unwrap();
        let repeated = surv2data_timeline(
            vec![1, 1, 1, 1],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![Some(1), None, Some(1), Some(2)],
            true,
        )
        .unwrap();

        assert_eq!(stutter.status, vec![0, 0, 2]);
        assert_eq!(stutter.istate, vec![Some(1), Some(1), Some(1)]);
        assert_eq!(repeated.status, vec![0, 1, 2]);
        assert_eq!(repeated.istate, vec![Some(1), Some(1), Some(1)]);
    }

    #[test]
    fn surv2data_timeline_handles_absent_and_mixed_initial_states() {
        let absent = surv2data_timeline(
            vec![1, 1, 1, 2, 2, 2],
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![Some(0), Some(1), Some(1), None, Some(2), Some(2)],
            false,
        )
        .unwrap();

        assert_eq!(absent.status, vec![1, 1, 2, 2]);
        assert_eq!(absent.istate, vec![None, None, None, None]);

        let mixed = surv2data_timeline(
            vec![1, 1, 2, 2],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![Some(1), Some(2), Some(0), Some(2)],
            false,
        )
        .unwrap_err();
        assert!(
            mixed
                .to_string()
                .contains("everyone or no one should have an initial state")
        );
    }

    #[test]
    fn surv2data_timeline_can_retain_only_each_subjects_first_event() {
        let result = surv2data_timeline_with_policy(
            vec![1, 2, 1, 1, 2, 1, 2, 2],
            vec![2.0, 3.0, 0.0, 3.0, 0.0, 1.0, 2.0, 1.0],
            vec![
                Some(1),
                Some(2),
                Some(1),
                Some(2),
                Some(1),
                Some(2),
                Some(1),
                Some(2),
            ],
            true,
            true,
        )
        .unwrap();

        assert_eq!(result.row_index, vec![0, 2, 4, 5, 6, 7]);
        assert_eq!(result.status, vec![0, 2, 2, 0, 0, 0]);
        assert_eq!(
            result.istate,
            vec![Some(2), Some(1), Some(1), Some(2), Some(2), Some(2)]
        );
    }

    #[test]
    fn surv2data_timeline_supports_ordinary_event_histories() {
        let mixed_first_status = surv2data_timeline_with_options(
            vec![1, 1, 2, 2],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![Some(1), None, Some(0), Some(1)],
            true,
            false,
            false,
        )
        .unwrap();

        assert_eq!(mixed_first_status.row_index, vec![0, 2]);
        assert_eq!(mixed_first_status.status, vec![0, 1]);
        assert_eq!(mixed_first_status.istate, vec![None, None]);

        let first_only = surv2data_timeline_with_options(
            vec![1, 1, 1, 1, 1, 1],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![Some(1), Some(0), Some(1), None, Some(2), Some(2)],
            true,
            true,
            false,
        )
        .unwrap();

        assert_eq!(first_only.status, vec![0, 0, 0, 2, 0]);
        assert_eq!(first_only.istate, vec![None; 5]);
    }

    #[test]
    fn from_timeline_rows_restores_input_order_and_tracks_removed_rows() {
        let result = from_timeline_rows(
            vec![0, 1, 0, 1, 0, 2],
            vec![0.0, 3.0, 4.0, 0.0, 2.0, 1.0],
            vec![1, 2, 3, 1, 2, 1],
        )
        .unwrap();

        assert_eq!(result.start, vec![0.0, 0.0, 2.0]);
        assert_eq!(result.stop, vec![2.0, 3.0, 4.0]);
        assert_eq!(result.status, vec![2, 2, 3]);
        assert_eq!(result.istate, vec![1, 1, 2]);
        assert_eq!(result.static_row, vec![0, 3, 0]);
        assert_eq!(result.dynamic_row, vec![0, 3, 4]);
        assert_eq!(result.removed_row, vec![5]);
    }

    #[test]
    fn from_timeline_rows_carries_states_through_censored_rows() {
        let result = from_timeline_rows(
            vec![0, 1, 0, 1, 0, 1],
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            vec![1, 2, 0, 0, 2, 3],
        )
        .unwrap();

        assert_eq!(result.start, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(result.stop, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(result.status, vec![0, 0, 2, 3]);
        assert_eq!(result.istate, vec![1, 2, 1, 2]);
        assert_eq!(result.static_row, vec![0, 1, 0, 1]);
        assert_eq!(result.dynamic_row, vec![0, 1, 2, 3]);
    }

    #[test]
    fn from_timeline_rows_suppresses_repeated_current_states() {
        let id = vec![0, 0, 0, 0];
        let time = vec![0.0, 1.0, 2.0, 3.0];
        let status = vec![1, 0, 1, 2];

        let suppressed =
            from_timeline_rows_with_repeated(id.clone(), time.clone(), status.clone(), false)
                .unwrap();
        let retained = from_timeline_rows(id, time, status).unwrap();

        assert_eq!(suppressed.status, vec![0, 0, 2]);
        assert_eq!(suppressed.istate, vec![1, 1, 1]);
        assert_eq!(retained.status, vec![0, 1, 2]);
        assert_eq!(retained.istate, vec![1, 1, 1]);

        let transition_back = from_timeline_rows_with_repeated(
            vec![0, 0, 0, 0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![1, 2, 0, 1],
            false,
        )
        .unwrap();
        assert_eq!(transition_back.status, vec![2, 0, 1]);
        assert_eq!(transition_back.istate, vec![1, 2, 2]);
    }

    #[test]
    fn from_timeline_rows_can_retain_only_each_subjects_first_event() {
        let result = from_timeline_rows_with_policy(
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            vec![1, 1, 2, 2, 1, 1, 2, 2],
            true,
            true,
        )
        .unwrap();

        assert_eq!(result.status, vec![2, 2, 0, 0, 0, 0]);
        assert_eq!(result.istate, vec![1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn timeline_rows_validate_inputs() {
        assert!(from_timeline_rows(vec![0], vec![], vec![1]).is_err());
        assert!(from_timeline_rows(vec![0], vec![f64::NAN], vec![1]).is_err());
    }

    #[test]
    fn from_timeline_rows_handles_absent_and_mixed_initial_states() {
        let absent = from_timeline_rows_with_repeated(
            vec![0, 0, 0, 1, 1, 1, 2],
            vec![0.0, 1.0, 2.0, 0.0, 2.0, 4.0, 0.0],
            vec![0, 0, 1, 0, 2, 0, 0],
            false,
        )
        .unwrap();

        assert_eq!(absent.start, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(absent.stop, vec![1.0, 2.0, 2.0, 4.0]);
        assert_eq!(absent.status, vec![0, 1, 2, 0]);
        assert!(absent.istate.is_empty());
        assert_eq!(absent.static_row, vec![0, 0, 3, 3]);
        assert_eq!(absent.dynamic_row, vec![0, 1, 3, 4]);
        assert_eq!(absent.removed_row, vec![6]);

        let mixed =
            from_timeline_rows(vec![0, 0, 1, 1], vec![0.0, 1.0, 0.0, 1.0], vec![0, 1, 1, 2])
                .unwrap_err();
        assert!(
            mixed
                .to_string()
                .contains("everyone or no one should have an initial state")
        );

        let singleton_mixed =
            from_timeline_rows(vec![0, 0, 1], vec![0.0, 1.0, 0.0], vec![0, 1, 1]).unwrap_err();
        assert!(
            singleton_mixed
                .to_string()
                .contains("everyone or no one should have an initial state")
        );
    }

    #[test]
    fn timeline_rows_reject_duplicate_times_within_a_subject() {
        let partial_tie =
            from_timeline_rows(vec![0, 0, 0], vec![0.0, 1.0, 1.0], vec![1, 2, 3]).unwrap_err();
        assert!(
            partial_tie
                .to_string()
                .contains("duplicated time for an id")
        );

        let all_tied = from_timeline_rows(vec![0, 0], vec![1.0, 1.0], vec![1, 2]).unwrap_err();
        assert!(all_tied.to_string().contains("duplicated time for an id"));
    }
}
