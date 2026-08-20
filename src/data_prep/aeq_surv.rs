use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AeqSurvResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub adjusted_count: usize,
    #[pyo3(get)]
    pub adjusted_indices: Vec<usize>,
}

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

/// Collapse nearly equal values across parallel time vectors using a fixed absolute tolerance.
///
/// This matches the endpoint normalization used by the R-compatible model wrappers: values are
/// ordered across all vectors, each run is anchored at its first value, and later values strictly
/// within `tolerance` of that anchor are replaced by it.
#[pyfunction]
#[pyo3(signature = (vectors, tolerance))]
pub fn timefix_vectors(mut vectors: Vec<Vec<f64>>, tolerance: f64) -> PyResult<Vec<Vec<f64>>> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(PyValueError::new_err(
            "tolerance must be a non-negative finite value",
        ));
    }

    let point_count = vectors.iter().map(Vec::len).sum();
    let mut points = Vec::with_capacity(point_count);
    for (vector_idx, vector) in vectors.iter().enumerate() {
        for (row_idx, &value) in vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "time values must be finite, got non-finite value in vector {vector_idx} at index {row_idx}"
                )));
            }
            points.push((value, vector_idx, row_idx));
        }
    }
    points.sort_unstable_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("time values were validated as finite")
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut cursor = 0;
    while cursor < points.len() {
        let base = points[cursor].0;
        let mut scan = cursor + 1;
        while scan < points.len() && points[scan].0 - base < tolerance {
            let (_, vector_idx, row_idx) = points[scan];
            vectors[vector_idx][row_idx] = base;
            scan += 1;
        }
        cursor = scan;
    }

    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_timefix_vectors(mut vectors: Vec<Vec<f64>>, tolerance: f64) -> Vec<Vec<f64>> {
        let mut points = vectors
            .iter()
            .enumerate()
            .flat_map(|(vector_idx, vector)| {
                vector
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(row_idx, value)| (value, vector_idx, row_idx))
            })
            .collect::<Vec<_>>();
        points.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap()
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        let mut cursor = 0;
        while cursor < points.len() {
            let base = points[cursor].0;
            let mut scan = cursor + 1;
            while scan < points.len() && points[scan].0 - base < tolerance {
                let (_, vector_idx, row_idx) = points[scan];
                vectors[vector_idx][row_idx] = base;
                scan += 1;
            }
            cursor = scan;
        }
        vectors
    }

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

    #[test]
    fn timefix_vectors_matches_fixed_anchor_semantics() {
        let result =
            timefix_vectors(vec![vec![0.0, 1.0 + 5e-10], vec![1.0, 1.0 + 1.5e-9]], 1e-9).unwrap();

        assert_eq!(result, vec![vec![0.0, 1.0], vec![1.0, 1.0 + 1.5e-9]]);
    }

    #[test]
    fn timefix_vectors_matches_reference_across_deterministic_fixtures() {
        let mut seed = 0x5eed_cafe_d00d_f00d_u64;
        for fixture in 0..500 {
            let vector_count = fixture % 5;
            let mut vectors = Vec::with_capacity(vector_count);
            for vector_idx in 0..vector_count {
                let row_count = (fixture * 7 + vector_idx * 11) % 31;
                let mut vector = Vec::with_capacity(row_count);
                for row_idx in 0..row_count {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    let bucket = ((seed >> 32) % 41) as f64 - 20.0;
                    let jitter = ((row_idx + vector_idx) % 4) as f64 * 4e-10;
                    vector.push(bucket + jitter);
                }
                vectors.push(vector);
            }
            let expected = reference_timefix_vectors(vectors.clone(), 1e-9);
            assert_eq!(timefix_vectors(vectors, 1e-9).unwrap(), expected);
        }
    }

    #[test]
    fn timefix_vectors_validates_public_inputs() {
        assert!(timefix_vectors(vec![vec![f64::NAN]], 1e-9).is_err());
        assert!(timefix_vectors(vec![vec![1.0]], f64::INFINITY).is_err());
        assert!(timefix_vectors(vec![vec![1.0]], -1.0).is_err());
    }
}
