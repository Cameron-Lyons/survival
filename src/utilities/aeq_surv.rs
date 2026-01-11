use pyo3::prelude::*;

/// Result of adjusting near ties in survival times
#[derive(Debug, Clone)]
#[pyclass]
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
/// * `tolerance` - Tolerance for considering values as tied (default: 1e-8 * range)
///
/// # Returns
/// * `AeqSurvResult` containing adjusted times and adjustment info
#[pyfunction]
#[pyo3(signature = (time, tolerance=None))]
pub fn aeq_surv(time: Vec<f64>, tolerance: Option<f64>) -> PyResult<AeqSurvResult> {
    let n = time.len();
    if n == 0 {
        return Ok(AeqSurvResult {
            time: vec![],
            adjusted_count: 0,
            adjusted_indices: vec![],
        });
    }

    let tol = tolerance.unwrap_or_else(|| {
        let min_val = time.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        if range > 0.0 { range * 1e-8 } else { 1e-8 }
    });

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut adjusted_time = time.clone();
    let mut adjusted_indices = Vec::new();

    let mut i = 0;
    while i < n {
        let base_val = adjusted_time[indices[i]];
        let mut j = i + 1;

        while j < n {
            let current_val = adjusted_time[indices[j]];
            if (current_val - base_val).abs() <= tol {
                // Only count as adjusted if the value actually changes
                if current_val != base_val {
                    adjusted_time[indices[j]] = base_val;
                    adjusted_indices.push(indices[j]);
                }
                j += 1;
            } else {
                break;
            }
        }
        i = j;
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
        assert_eq!(result.adjusted_count, 0); // Already exactly tied
    }
}
