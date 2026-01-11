//! Convert between timeline (wide) and interval (long) data formats
//!
//! Timeline format: one row per subject, multiple columns for different time points
//! Interval format: multiple rows per subject, with (time1, time2) columns

use pyo3::prelude::*;
use std::collections::{BTreeSet, HashMap};

/// Result of converting to timeline (wide) format
#[pyclass]
#[derive(Clone)]
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
#[pyclass]
#[derive(Clone)]
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
) -> TimelineResult {
    let n = id.len();
    if n == 0 {
        return TimelineResult {
            id: Vec::new(),
            states: Vec::new(),
            time_points: Vec::new(),
        };
    }

    let mut unique_ids: Vec<i32> = Vec::new();
    let mut seen_ids: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for &subj_id in &id {
        if seen_ids.insert(subj_id) {
            unique_ids.push(subj_id);
        }
    }

    let times: Vec<f64> = match time_points {
        Some(tp) => tp,
        None => {
            let mut all_times: BTreeSet<i64> = BTreeSet::new();
            for i in 0..n {
                all_times.insert((time1[i] * 1000.0) as i64);
                all_times.insert((time2[i] * 1000.0) as i64);
            }
            all_times.iter().map(|&t| t as f64 / 1000.0).collect()
        }
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

    TimelineResult {
        id: unique_ids,
        states,
        time_points: times,
    }
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
pub fn from_timeline(id: Vec<i32>, states: Vec<Vec<i32>>, time_points: Vec<f64>) -> IntervalResult {
    if id.is_empty() || states.is_empty() || time_points.is_empty() {
        return IntervalResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
        };
    }

    let num_subjects = id.len();
    let num_times = time_points.len();

    let mut result = IntervalResult {
        id: Vec::new(),
        time1: Vec::new(),
        time2: Vec::new(),
        status: Vec::new(),
    };

    for subj_idx in 0..num_subjects {
        let subj_id = id[subj_idx];
        let subj_states = &states[subj_idx];

        if subj_states.len() != num_times {
            continue;
        }

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

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_timeline_basic() {
        let id = vec![1, 1, 2];
        let time1 = vec![0.0, 5.0, 0.0];
        let time2 = vec![5.0, 10.0, 10.0];
        let status = vec![0, 1, 0];

        let result = to_timeline(id, time1, time2, status, None);

        assert_eq!(result.id, vec![1, 2]);
        assert!(result.time_points.len() >= 3); // At least 0, 5, 10
    }

    #[test]
    fn test_from_timeline_basic() {
        let id = vec![1, 2];
        let states = vec![vec![0, 0, 1], vec![0, 1, 1]];
        let time_points = vec![0.0, 5.0, 10.0];

        let result = from_timeline(id, states, time_points);

        assert_eq!(result.id.len(), 4); // 2 subjects x 2 intervals each
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
        );

        let intervals = from_timeline(
            timeline.id.clone(),
            timeline.states.clone(),
            timeline.time_points.clone(),
        );

        assert_eq!(intervals.id.len(), 2);
    }
}
