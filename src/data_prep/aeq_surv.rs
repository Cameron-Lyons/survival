use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Result of adjusting near ties in survival times
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AeqSurvResult {
    /// Adjusted survival times with near-ties resolved
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Number of values that were adjusted
    #[pyo3(get)]
    pub adjusted_count: usize,
    /// Indices of values that were adjusted
    #[pyo3(get)]
    pub adjusted_indices: Vec<usize>,
}

/// Adjudicate near ties in survival times.
///
/// This function handles floating-point precision issues that can cause
/// survival times that should be equal to be treated as different.
/// It compares values and replaces near-ties with the smaller value.
///
/// # Arguments
/// * `time` - Vector of survival times
/// * `tolerance` - Absolute/relative tolerance for considering values as tied
///   (default: `sqrt(f64::EPSILON)`)
///
/// # Returns
/// * `AeqSurvResult` containing adjusted times and adjustment info
#[pyfunction]
#[pyo3(signature = (time, tolerance=None))]
pub fn aeq_surv(time: Vec<f64>, tolerance: Option<f64>) -> PyResult<AeqSurvResult> {
    let n = time.len();
    for (idx, value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "time values must be finite, got non-finite value at index {}",
                idx
            )));
        }
    }

    if let Some(tol) = tolerance
        && !tol.is_finite()
    {
        return Err(PyErr::new::<PyValueError, _>("tolerance must be finite"));
    }

    if n == 0 {
        return Ok(AeqSurvResult {
            time: vec![],
            adjusted_count: 0,
            adjusted_indices: vec![],
        });
    }

    let tol = tolerance.unwrap_or_else(|| f64::EPSILON.sqrt());
    if tol <= 0.0 {
        return Ok(AeqSurvResult {
            time,
            adjusted_count: 0,
            adjusted_indices: vec![],
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut unique_times = Vec::with_capacity(n);
    for &idx in &indices {
        let value = time[idx];
        if unique_times
            .last()
            .is_none_or(|previous| value != *previous)
        {
            unique_times.push(value);
        }
    }

    if unique_times.len() <= 1 {
        return Ok(AeqSurvResult {
            time,
            adjusted_count: 0,
            adjusted_indices: vec![],
        });
    }

    let mean_abs =
        unique_times.iter().map(|value| value.abs()).sum::<f64>() / unique_times.len() as f64;
    let mut cuts = Vec::with_capacity(unique_times.len());
    cuts.push(unique_times[0]);
    for pair in unique_times.windows(2) {
        let delta = pair[1] - pair[0];
        let tied = delta <= tol || (mean_abs > 0.0 && delta / mean_abs <= tol);
        if !tied {
            cuts.push(pair[1]);
        }
    }

    if cuts.len() == unique_times.len() {
        return Ok(AeqSurvResult {
            time,
            adjusted_count: 0,
            adjusted_indices: vec![],
        });
    }

    let mut adjusted_time = time.clone();
    let mut adjusted_indices = Vec::new();

    for (idx, value) in time.iter().copied().enumerate() {
        let cut_idx = match cuts.binary_search_by(|cut| cut.total_cmp(&value)) {
            Ok(found) => found,
            Err(insert_pos) => insert_pos.saturating_sub(1),
        };
        let adjusted_value = cuts[cut_idx];
        if adjusted_value != value {
            adjusted_time[idx] = adjusted_value;
            adjusted_indices.push(idx);
        }
    }

    let adjusted_count = adjusted_indices.len();

    Ok(AeqSurvResult {
        time: adjusted_time,
        adjusted_count,
        adjusted_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aeq_surv_no_ties() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = aeq_surv(time.clone(), None).unwrap();
        assert_eq!(result.time, time);
        assert_eq!(result.adjusted_count, 0);
    }

    #[test]
    fn test_aeq_surv_near_ties() {
        let time = vec![1.0, 1.0 + 1e-10, 2.0, 3.0];
        let result = aeq_surv(time, Some(1e-8)).unwrap();
        assert_eq!(result.adjusted_count, 1);
        assert!((result.time[0] - result.time[1]).abs() < 1e-15);
    }

    #[test]
    fn test_aeq_surv_matches_r_adjacent_cutpoints() {
        let result = aeq_surv(vec![1.0, 1.0 + 9e-9, 1.0 + 18e-9], Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1.0, 1.0, 1.0]);
        assert_eq!(result.adjusted_indices, vec![1, 2]);
    }

    #[test]
    fn test_aeq_surv_matches_r_relative_tolerance() {
        let result = aeq_surv(vec![1e9, 1e9 + 1.0, 1e9 + 20.0], Some(1e-8)).unwrap();
        assert_eq!(result.time, vec![1e9, 1e9, 1e9 + 20.0]);
        assert_eq!(result.adjusted_indices, vec![1]);
    }

    #[test]
    fn test_aeq_surv_nonpositive_tolerance_is_noop() {
        let time = vec![1.0, 1.0 + 1e-10];
        let negative = aeq_surv(time.clone(), Some(-1.0)).unwrap();
        let zero = aeq_surv(time.clone(), Some(0.0)).unwrap();

        assert_eq!(negative.time, time);
        assert_eq!(negative.adjusted_count, 0);
        assert_eq!(zero.time, time);
        assert_eq!(zero.adjusted_count, 0);
    }

    #[test]
    fn test_aeq_surv_empty() {
        let time: Vec<f64> = vec![];
        let result = aeq_surv(time, None).unwrap();
        assert_eq!(result.time.len(), 0);
        assert_eq!(result.adjusted_count, 0);
    }

    #[test]
    fn test_aeq_surv_all_same() {
        let time = vec![1.0, 1.0, 1.0, 1.0];
        let result = aeq_surv(time, None).unwrap();
        assert_eq!(result.adjusted_count, 0);
    }

    #[test]
    fn test_aeq_surv_rejects_nonfinite_values_and_tolerance() {
        assert!(aeq_surv(vec![1.0, f64::NAN], None).is_err());
        assert!(aeq_surv(vec![1.0], Some(f64::INFINITY)).is_err());
    }
}
