//! Condense (shorten) a survival dataset by merging adjacent intervals
//!
//! This is the inverse operation of survsplit - it merges adjacent censored intervals
//! that have the same covariate values.

use pyo3::prelude::*;

/// Result of condensing survival data
#[pyclass]
#[derive(Clone)]
pub struct CondenseResult {
    /// Subject identifiers for each output row
    #[pyo3(get)]
    pub id: Vec<i32>,
    /// Start time of each interval
    #[pyo3(get)]
    pub time1: Vec<f64>,
    /// End time of each interval
    #[pyo3(get)]
    pub time2: Vec<f64>,
    /// Event status (1=event, 0=censored)
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// Maps each output row to original input row indices
    #[pyo3(get)]
    pub row_map: Vec<Vec<usize>>,
}

/// Condense a survival dataset by merging adjacent censored intervals
///
/// Merges consecutive intervals for the same subject where:
/// - The intervals are adjacent (time2\[i\] == time1\[i+1\])
/// - All intermediate intervals are censored (status=0)
/// - The final interval inherits the status of the last merged interval
///
/// # Arguments
/// * `id` - Subject identifiers
/// * `time1` - Start times of intervals
/// * `time2` - End times of intervals
/// * `status` - Event status (1=event, 0=censored)
///
/// # Returns
/// A CondenseResult with merged intervals
#[pyfunction]
#[pyo3(signature = (id, time1, time2, status))]
pub fn survcondense(
    id: Vec<i32>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
) -> CondenseResult {
    let n = id.len();
    if n == 0 {
        return CondenseResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
            row_map: Vec::new(),
        };
    }

    // Create indices sorted by (id, time1)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| match id[a].cmp(&id[b]) {
        std::cmp::Ordering::Equal => time1[a]
            .partial_cmp(&time1[b])
            .unwrap_or(std::cmp::Ordering::Equal),
        other => other,
    });

    let mut result = CondenseResult {
        id: Vec::with_capacity(n),
        time1: Vec::with_capacity(n),
        time2: Vec::with_capacity(n),
        status: Vec::with_capacity(n),
        row_map: Vec::with_capacity(n),
    };

    let mut i = 0;
    while i < n {
        let idx = indices[i];
        let current_id = id[idx];
        let current_start = time1[idx];
        let mut current_end = time2[idx];
        let mut current_status = status[idx];
        let mut row_indices = vec![idx + 1];

        let mut j = i + 1;
        while j < n {
            let next_idx = indices[j];

            if id[next_idx] != current_id {
                break;
            }

            let gap = (time1[next_idx] - current_end).abs();
            if gap > 1e-9 {
                break;
            }

            if current_status != 0 {
                break;
            }

            current_end = time2[next_idx];
            current_status = status[next_idx];
            row_indices.push(next_idx + 1);
            j += 1;
        }

        result.id.push(current_id);
        result.time1.push(current_start);
        result.time2.push(current_end);
        result.status.push(current_status);
        result.row_map.push(row_indices);

        i = j;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survcondense_basic() {
        // Subject 1: intervals [0,5), [5,10), [10,15) all censored
        // Should merge into [0,15)
        let id = vec![1, 1, 1];
        let time1 = vec![0.0, 5.0, 10.0];
        let time2 = vec![5.0, 10.0, 15.0];
        let status = vec![0, 0, 0];

        let result = survcondense(id, time1, time2, status);

        assert_eq!(result.id.len(), 1);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 15.0);
        assert_eq!(result.status[0], 0);
        assert_eq!(result.row_map[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_survcondense_with_event() {
        // Subject 1: [0,5) censored, [5,10) event
        // Should merge into [0,10) with event
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![5.0, 10.0];
        let status = vec![0, 1];

        let result = survcondense(id, time1, time2, status);

        assert_eq!(result.id.len(), 1);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 10.0);
        assert_eq!(result.status[0], 1);
    }

    #[test]
    fn test_survcondense_event_stops_merge() {
        // Subject 1: [0,5) event, [5,10) censored
        // Should NOT merge - event stops the chain
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![5.0, 10.0];
        let status = vec![1, 0];

        let result = survcondense(id, time1, time2, status);

        assert_eq!(result.id.len(), 2);
    }

    #[test]
    fn test_survcondense_multiple_subjects() {
        let id = vec![1, 1, 2, 2];
        let time1 = vec![0.0, 5.0, 0.0, 3.0];
        let time2 = vec![5.0, 10.0, 3.0, 8.0];
        let status = vec![0, 0, 0, 1];

        let result = survcondense(id, time1, time2, status);

        assert_eq!(result.id.len(), 2);
        assert_eq!(result.id, vec![1, 2]);
    }
}
