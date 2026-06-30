//! Condense (shorten) a survival dataset by merging adjacent intervals
//!
//! This is the inverse operation of survsplit - it merges adjacent censored intervals
//! that have the same covariate values.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constants::TIME_EPSILON;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_overlapping_intervals_i32,
};

/// Result of condensing survival data
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
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
) -> PyResult<CondenseResult> {
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

    validate_no_nan(&time1, "time1")?;
    validate_finite(&time1, "time1")?;
    validate_no_nan(&time2, "time2")?;
    validate_finite(&time2, "time2")?;
    validate_binary_i32(&status, "status")?;
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
        return Ok(CondenseResult {
            id: Vec::new(),
            time1: Vec::new(),
            time2: Vec::new(),
            status: Vec::new(),
            row_map: Vec::new(),
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| match id[a].cmp(&id[b]) {
        std::cmp::Ordering::Equal => time1[a].total_cmp(&time1[b]).then_with(|| a.cmp(&b)),
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
            if gap > TIME_EPSILON {
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

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::{index_permutations, initialize_python};

    #[test]
    fn test_survcondense_basic() {
        let id = vec![1, 1, 1];
        let time1 = vec![0.0, 5.0, 10.0];
        let time2 = vec![5.0, 10.0, 15.0];
        let status = vec![0, 0, 0];

        let result = survcondense(id, time1, time2, status).unwrap();

        assert_eq!(result.id.len(), 1);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 15.0);
        assert_eq!(result.status[0], 0);
        assert_eq!(result.row_map[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_survcondense_with_event() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![5.0, 10.0];
        let status = vec![0, 1];

        let result = survcondense(id, time1, time2, status).unwrap();

        assert_eq!(result.id.len(), 1);
        assert_eq!(result.time1[0], 0.0);
        assert_eq!(result.time2[0], 10.0);
        assert_eq!(result.status[0], 1);
    }

    #[test]
    fn test_survcondense_event_stops_merge() {
        let id = vec![1, 1];
        let time1 = vec![0.0, 5.0];
        let time2 = vec![5.0, 10.0];
        let status = vec![1, 0];

        let result = survcondense(id, time1, time2, status).unwrap();

        assert_eq!(result.id.len(), 2);
    }

    #[test]
    fn test_survcondense_multiple_subjects() {
        let id = vec![1, 1, 2, 2];
        let time1 = vec![0.0, 5.0, 0.0, 3.0];
        let time2 = vec![5.0, 10.0, 3.0, 8.0];
        let status = vec![0, 0, 0, 1];

        let result = survcondense(id, time1, time2, status).unwrap();

        assert_eq!(result.id.len(), 2);
        assert_eq!(result.id, vec![1, 2]);
    }

    #[test]
    fn test_survcondense_is_invariant_to_input_order() {
        let base_id = [1, 1, 1, 2, 2];
        let base_time1 = [0.0, 2.0, 4.0, 0.0, 3.0];
        let base_time2 = [2.0, 4.0, 6.0, 3.0, 5.0];
        let base_status = [0, 0, 1, 0, 0];
        let expected = vec![(1, 0.0, 6.0, 1), (2, 0.0, 5.0, 0)];

        for permutation in index_permutations(base_id.len()) {
            let id: Vec<i32> = permutation.iter().map(|&i| base_id[i]).collect();
            let time1: Vec<f64> = permutation.iter().map(|&i| base_time1[i]).collect();
            let time2: Vec<f64> = permutation.iter().map(|&i| base_time2[i]).collect();
            let status: Vec<i32> = permutation.iter().map(|&i| base_status[i]).collect();

            let result =
                survcondense(id.clone(), time1.clone(), time2.clone(), status.clone()).unwrap();
            let condensed: Vec<(i32, f64, f64, i32)> = result
                .id
                .iter()
                .zip(&result.time1)
                .zip(&result.time2)
                .zip(&result.status)
                .map(|(((id, time1), time2), status)| (*id, *time1, *time2, *status))
                .collect();

            assert_eq!(condensed, expected);

            let mut seen = vec![false; id.len()];
            for row_indices in &result.row_map {
                for &row in row_indices {
                    let original = row - 1;
                    assert!(!seen[original]);
                    seen[original] = true;
                }
            }
            assert!(seen.into_iter().all(|covered| covered));
        }
    }

    #[test]
    fn test_survcondense_rejects_mismatched_inputs() {
        assert!(survcondense(vec![1], vec![], vec![1.0], vec![0]).is_err());
        assert!(survcondense(vec![1], vec![0.0], vec![], vec![0]).is_err());
        assert!(survcondense(vec![1], vec![0.0], vec![1.0], vec![]).is_err());
    }

    #[test]
    fn test_survcondense_rejects_malformed_values() {
        initialize_python();

        let err = survcondense(vec![1], vec![f64::NAN], vec![1.0], vec![0]).unwrap_err();
        assert!(err.to_string().contains("time1 contains NaN"));

        let err = survcondense(vec![1], vec![0.0], vec![f64::INFINITY], vec![0]).unwrap_err();
        assert!(err.to_string().contains("time2 contains non-finite"));

        let err = survcondense(vec![1], vec![2.0], vec![1.0], vec![0]).unwrap_err();
        assert!(err.to_string().contains("time1 must be <= time2"));

        let err = survcondense(vec![1], vec![0.0], vec![1.0], vec![2]).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = survcondense(vec![1, 1], vec![0.0, 4.0], vec![5.0, 6.0], vec![0, 0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("intervals must not overlap within id")
        );
    }
}
