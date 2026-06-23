//! Convert between timeline (wide) and interval (long) data formats
//!
//! Timeline format: one row per subject, multiple columns for different time points
//! Interval format: multiple rows per subject, with (time1, time2) columns

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::constants::TIME_EPSILON;
use crate::internal::validation::{
    validate_finite, validate_no_nan, validate_non_overlapping_intervals_i32,
};

/// Result of converting to timeline (wide) format
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct TimelineResult {
    /// Subject identifiers (one per row)
    #[pyo3(get)]
    pub id: Vec<i32>,
    /// State at each time point for each subject (subjects x time_points)
    #[pyo3(get)]
    pub states: Vec<Vec<i32>>,
    /// Time points (column headers)
    #[pyo3(get)]
    pub time_points: Vec<f64>,
}

/// Result of converting from timeline to interval format
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct IntervalResult {
    /// Subject identifiers
    #[pyo3(get)]
    pub id: Vec<i32>,
    /// Start time of each interval
    #[pyo3(get)]
    pub time1: Vec<f64>,
    /// End time of each interval
    #[pyo3(get)]
    pub time2: Vec<f64>,
    /// State/status for each interval
    #[pyo3(get)]
    pub status: Vec<i32>,
}

fn validate_timeline_times(field: &'static str, values: &[f64]) -> PyResult<()> {
    validate_no_nan(values, field)?;
    validate_finite(values, field)?;
    Ok(())
}

fn validate_increasing_time_points(time_points: &[f64]) -> PyResult<()> {
    validate_timeline_times("time_points", time_points)?;
    for (index, window) in time_points.windows(2).enumerate() {
        if window[1] <= window[0] + TIME_EPSILON {
            return Err(PyValueError::new_err(format!(
                "time_points must be strictly increasing; got {} then {} at positions {} and {}",
                window[0],
                window[1],
                index,
                index + 1
            )));
        }
    }
    Ok(())
}

fn sorted_unique_time_points(time1: &[f64], time2: &[f64]) -> Vec<f64> {
    let mut times = Vec::with_capacity(time1.len() * 2);
    times.extend_from_slice(time1);
    times.extend_from_slice(time2);
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup_by(|a, b| (*a - *b).abs() <= TIME_EPSILON);
    times
}

/// Convert interval data to timeline (wide) format
///
/// Creates a grid where each row is a subject and each column is a time point.
/// The value at each cell is the state/status at that time.
///
/// # Arguments
/// * `id` - Subject identifiers
/// * `time1` - Start times of intervals
/// * `time2` - End times of intervals
/// * `status` - State/status for each interval
/// * `time_points` - Optional: specific time points to use as columns
///
/// # Returns
/// TimelineResult with subjects as rows and time points as columns
#[pyfunction]
#[pyo3(signature = (id, time1, time2, status, time_points=None))]
pub fn to_timeline(
    id: Vec<i32>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    time_points: Option<Vec<f64>>,
) -> PyResult<TimelineResult> {
    let n = id.len();
    if time1.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "time1 must have same length as id",
        ));
    }
    if time2.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "time2 must have same length as id",
        ));
    }
    if status.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "status must have same length as id",
        ));
    }

    validate_timeline_times("time1", &time1)?;
    validate_timeline_times("time2", &time2)?;
    for (index, (&start, &stop)) in time1.iter().zip(time2.iter()).enumerate() {
        if start > stop + TIME_EPSILON {
            return Err(PyValueError::new_err(format!(
                "time1 must be <= time2; got time1 {} and time2 {} at index {}",
                start, stop, index
            )));
        }
    }
    validate_non_overlapping_intervals_i32(&id, &time1, &time2, TIME_EPSILON)?;

    if n == 0 {
        return Ok(TimelineResult {
            id: Vec::new(),
            states: Vec::new(),
            time_points: Vec::new(),
        });
    }

    let mut unique_ids: Vec<i32> = Vec::new();
    let mut seen_ids: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for &subj_id in &id {
        if seen_ids.insert(subj_id) {
            unique_ids.push(subj_id);
        }
    }

    let times: Vec<f64> = match time_points {
        Some(tp) => {
            validate_increasing_time_points(&tp)?;
            tp
        }
        None => sorted_unique_time_points(&time1, &time2),
    };

    let num_subjects = unique_ids.len();
    let num_times = times.len();

    let id_to_row: HashMap<i32, usize> = unique_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut states: Vec<Vec<i32>> = vec![vec![0; num_times]; num_subjects];

    for i in 0..n {
        let subj_id = id[i];
        let row = id_to_row[&subj_id];
        let t1 = time1[i];
        let t2 = time2[i];
        let stat = status[i];

        for (col, &t) in times.iter().enumerate() {
            if t >= t1 && t < t2 {
                states[row][col] = stat;
            }
        }
        for (col, &t) in times.iter().enumerate() {
            if (t - t2).abs() < 1e-9 {
                states[row][col] = stat;
            }
        }
    }

    Ok(TimelineResult {
        id: unique_ids,
        states,
        time_points: times,
    })
}

/// Convert timeline (wide) format to interval (long) format
///
/// Takes a grid where each row is a subject and each column is a time point,
/// and converts it back to interval format with (time1, time2) pairs.
///
/// # Arguments
/// * `id` - Subject identifiers (one per row)
/// * `states` - State matrix (subjects x time_points)
/// * `time_points` - Time point values for each column
///
/// # Returns
/// IntervalResult with (time1, time2) intervals for each state change
#[pyfunction]
pub fn from_timeline(
    id: Vec<i32>,
    states: Vec<Vec<i32>>,
    time_points: Vec<f64>,
) -> PyResult<IntervalResult> {
    if states.len() != id.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "states must have one row per id, got {} rows and {} ids",
            states.len(),
            id.len()
        )));
    }

    validate_increasing_time_points(&time_points)?;

    if id.is_empty() || time_points.is_empty() {
        return Ok(IntervalResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
        });
    }

    let num_subjects = id.len();
    let num_times = time_points.len();
    for (row, subject_states) in states.iter().enumerate() {
        if subject_states.len() != num_times {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "states row {} has length {} but expected {}",
                row,
                subject_states.len(),
                num_times
            )));
        }
    }

    let mut result = IntervalResult {
        id: Vec::new(),
        time1: Vec::new(),
        time2: Vec::new(),
        status: Vec::new(),
    };

    for subj_idx in 0..num_subjects {
        let subj_id = id[subj_idx];
        let subj_states = &states[subj_idx];

        for t in 0..num_times.saturating_sub(1) {
            let t1 = time_points[t];
            let t2 = time_points[t + 1];
            let status = subj_states[t];

            result.id.push(subj_id);
            result.time1.push(t1);
            result.time2.push(t2);
            result.status.push(status);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn test_to_timeline_basic() {
        let id = vec![1, 1, 2];
        let time1 = vec![0.0, 5.0, 0.0];
        let time2 = vec![5.0, 10.0, 10.0];
        let status = vec![0, 1, 0];

        let result = to_timeline(id, time1, time2, status, None).unwrap();

        assert_eq!(result.id, vec![1, 2]);
        assert!(result.time_points.len() >= 3);
    }

    #[test]
    fn test_to_timeline_preserves_precise_generated_time_points() {
        let result = to_timeline(
            vec![1, 1],
            vec![0.0, 1.0004],
            vec![1.0004, 1.0008],
            vec![0, 1],
            None,
        )
        .unwrap();

        assert_eq!(result.time_points, vec![0.0, 1.0004, 1.0008]);
        assert_eq!(result.states, vec![vec![0, 1, 1]]);
    }

    #[test]
    fn test_from_timeline_basic() {
        let id = vec![1, 2];
        let states = vec![vec![0, 0, 1], vec![0, 1, 1]];
        let time_points = vec![0.0, 5.0, 10.0];

        let result = from_timeline(id, states, time_points).unwrap();

        assert_eq!(result.id.len(), 4);
    }

    #[test]
    fn test_roundtrip() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![5.0, 10.0];
        let status = vec![0, 1];

        let timeline = to_timeline(
            id.clone(),
            time1.clone(),
            time2.clone(),
            status.clone(),
            Some(vec![0.0, 5.0, 10.0]),
        )
        .unwrap();

        let intervals = from_timeline(
            timeline.id.clone(),
            timeline.states.clone(),
            timeline.time_points.clone(),
        )
        .unwrap();

        assert_eq!(intervals.id.len(), 2);
    }

    #[test]
    fn test_to_timeline_length_mismatch_is_error() {
        assert!(to_timeline(vec![1], vec![], vec![1.0], vec![0], None).is_err());
        assert!(to_timeline(vec![1], vec![0.0], vec![], vec![0], None).is_err());
        assert!(to_timeline(vec![1], vec![0.0], vec![1.0], vec![], None).is_err());
    }

    #[test]
    fn test_to_timeline_rejects_malformed_times() {
        initialize_python();

        let err = to_timeline(vec![1], vec![f64::NAN], vec![1.0], vec![0], None).unwrap_err();
        assert!(err.to_string().contains("time1 contains NaN"));

        let err = to_timeline(vec![1], vec![0.0], vec![f64::INFINITY], vec![0], None).unwrap_err();
        assert!(err.to_string().contains("time2 contains non-finite"));

        let err = to_timeline(vec![1], vec![2.0], vec![1.0], vec![0], None).unwrap_err();
        assert!(err.to_string().contains("time1 must be <= time2"));

        let err =
            to_timeline(vec![1], vec![0.0], vec![1.0], vec![0], Some(vec![0.0, 0.0])).unwrap_err();
        assert!(
            err.to_string()
                .contains("time_points must be strictly increasing")
        );

        let err =
            to_timeline(vec![1, 1], vec![0.0, 4.0], vec![5.0, 6.0], vec![0, 1], None).unwrap_err();
        assert!(
            err.to_string()
                .contains("intervals must not overlap within id")
        );
    }

    #[test]
    fn test_from_timeline_shape_mismatch_is_error() {
        assert!(from_timeline(vec![1, 2], vec![vec![0]], vec![0.0]).is_err());
        assert!(from_timeline(vec![1], vec![vec![0, 1]], vec![0.0]).is_err());
    }

    #[test]
    fn test_from_timeline_rejects_malformed_time_points() {
        initialize_python();

        let err = from_timeline(vec![1], vec![vec![0]], vec![f64::NAN]).unwrap_err();
        assert!(err.to_string().contains("time_points contains NaN"));

        let err = from_timeline(vec![1], vec![vec![0, 1]], vec![1.0, 0.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time_points must be strictly increasing")
        );
    }
}
