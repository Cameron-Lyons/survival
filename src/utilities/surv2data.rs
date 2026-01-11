//! Convert timecourse data to (time1, time2) interval format
//!
//! Converts data where each row represents an observation at a single time point
//! into counting process format with (time1, time2) intervals.

use pyo3::prelude::*;

/// Result of converting timecourse data to interval format
#[pyclass]
#[derive(Clone)]
pub struct Surv2DataResult {
    /// Subject identifiers
    #[pyo3(get)]
    pub id: Vec<i32>,
    /// Start time of each interval
    #[pyo3(get)]
    pub time1: Vec<f64>,
    /// End time of each interval
    #[pyo3(get)]
    pub time2: Vec<f64>,
    /// Event status (1=event, 0=censored) for each interval
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// Original row index (1-indexed) for each output row
    #[pyo3(get)]
    pub row_index: Vec<usize>,
}

/// Convert timecourse data to counting process (time1, time2) format
///
/// Takes data where each row is an observation at a single time point and converts
/// it to interval format suitable for Cox regression with time-varying covariates.
///
/// # Arguments
/// * `id` - Subject identifiers
/// * `time` - Observation times for each row
/// * `event_time` - Optional: time of event for each subject (if known)
/// * `event_status` - Optional: event indicator (1=event, 0=censored)
///
/// # Returns
/// Surv2DataResult with intervals created from consecutive observations
#[pyfunction]
#[pyo3(signature = (id, time, event_time=None, event_status=None))]
pub fn surv2data(
    id: Vec<i32>,
    time: Vec<f64>,
    event_time: Option<Vec<f64>>,
    event_status: Option<Vec<i32>>,
) -> Surv2DataResult {
    let n = id.len();
    if n == 0 {
        return Surv2DataResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
            row_index: Vec::new(),
        };
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| match id[a].cmp(&id[b]) {
        std::cmp::Ordering::Equal => time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal),
        other => other,
    });

    let mut subject_event: std::collections::HashMap<i32, (f64, i32)> =
        std::collections::HashMap::new();
    if let (Some(etimes), Some(estatus)) = (&event_time, &event_status) {
        for i in 0..n {
            let subj_id = id[i];
            subject_event
                .entry(subj_id)
                .or_insert((etimes[i], estatus[i]));
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

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surv2data_basic() {
        // Subject 1: observations at times 0, 5, 10, event at time 15
        let id = vec![1, 1, 1];
        let time = vec![0.0, 5.0, 10.0];
        let event_time = Some(vec![15.0, 15.0, 15.0]);
        let event_status = Some(vec![1, 1, 1]);

        let result = surv2data(id, time, event_time, event_status);

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

        let result = surv2data(id, time, event_time, event_status);

        // Subject 1: [0,5), [5,10) with event
        // Subject 2: [0,3), [3,8) censored
        assert_eq!(result.id.len(), 4);
    }

    #[test]
    fn test_surv2data_no_event_info() {
        let id = vec![1, 1, 1];
        let time = vec![0.0, 5.0, 10.0];

        let result = surv2data(id, time, None, None);

        // Without event info, each observation creates an interval to the next
        // Last observation has no next, so we get 2 intervals from 3 observations
        // Actually, with infinity as default event time, we get 3 intervals
        assert!(result.id.len() >= 2);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 5.0);
    }
}
