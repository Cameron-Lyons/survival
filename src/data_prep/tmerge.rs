use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constants::TIME_EPSILON;

fn length_error(name: &str, expected_name: &str, actual: usize, expected: usize) -> PyErr {
    PyErr::new::<PyValueError, _>(format!(
        "{} must have same length as {}, got {} and {}",
        name, expected_name, actual, expected
    ))
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{} values must be finite, got non-finite value at index {}",
                name, index
            )));
        }
    }
    Ok(())
}

fn validate_newx_values(values: &[f64]) -> PyResult<()> {
    for (index, &value) in values.iter().enumerate() {
        if value.is_infinite() {
            return Err(PyValueError::new_err(format!(
                "newx values may be finite or NaN, got infinite value at index {}",
                index
            )));
        }
    }
    Ok(())
}

fn validate_non_decreasing_id(name: &str, id: &[i32]) -> PyResult<()> {
    for (index, window) in id.windows(2).enumerate() {
        if window[1] < window[0] {
            return Err(PyValueError::new_err(format!(
                "{} must be sorted in non-decreasing order; got {} before {} at positions {} and {}",
                name,
                window[0],
                window[1],
                index,
                index + 1
            )));
        }
    }
    Ok(())
}

fn validate_sorted_id_time(
    id_name: &str,
    id: &[i32],
    time_name: &str,
    time: &[f64],
) -> PyResult<()> {
    validate_non_decreasing_id(id_name, id)?;
    for (index, ((&previous_id, &current_id), (&previous_time, &current_time))) in id
        .iter()
        .zip(id.iter().skip(1))
        .zip(time.iter().zip(time.iter().skip(1)))
        .enumerate()
    {
        if current_id == previous_id && current_time + TIME_EPSILON < previous_time {
            return Err(PyValueError::new_err(format!(
                "{} must be non-decreasing within {}; got {} before {} at positions {} and {}",
                time_name,
                id_name,
                previous_time,
                current_time,
                index,
                index + 1
            )));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn tmerge(
    id: Vec<i32>,
    time1: Vec<f64>,
    newx: Vec<f64>,
    nid: Vec<i32>,
    ntime: Vec<f64>,
    x: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let n1 = id.len();
    if time1.len() != n1 {
        return Err(length_error("time1", "id", time1.len(), n1));
    }
    if newx.len() != n1 {
        return Err(length_error("newx", "id", newx.len(), n1));
    }

    let n2 = nid.len();
    if ntime.len() != n2 {
        return Err(length_error("ntime", "nid", ntime.len(), n2));
    }
    if x.len() != n2 {
        return Err(length_error("x", "nid", x.len(), n2));
    }
    validate_finite_values("time1", &time1)?;
    validate_newx_values(&newx)?;
    validate_finite_values("ntime", &ntime)?;
    validate_finite_values("x", &x)?;
    validate_sorted_id_time("id", &id, "time1", &time1)?;
    validate_sorted_id_time("nid", &nid, "ntime", &ntime)?;

    let mut result = newx;
    let mut k = 0;
    let mut current_id: Option<i32> = None;
    let mut csum = 0.0;
    let mut has_one = false;
    for i in 0..n1 {
        let row_id = id[i];
        if current_id != Some(row_id) {
            current_id = Some(row_id);
            csum = 0.0;
            has_one = false;
            while k < n2 && nid[k] < row_id {
                k += 1;
            }
        }
        let start_time = time1[i];
        let mut local_k = k;
        while local_k < n2 && nid[local_k] == row_id && ntime[local_k] <= start_time {
            csum += x[local_k];
            has_one = true;
            local_k += 1;
        }
        k = local_k;
        if has_one {
            result[i] = if result[i].is_nan() {
                csum
            } else {
                result[i] + csum
            };
        }
    }
    Ok(result)
}
#[pyfunction]
pub fn tmerge2(
    id: Vec<i32>,
    time1: Vec<f64>,
    nid: Vec<i32>,
    ntime: Vec<f64>,
) -> PyResult<Vec<usize>> {
    let n1 = id.len();
    if time1.len() != n1 {
        return Err(length_error("time1", "id", time1.len(), n1));
    }

    let n2 = nid.len();
    if ntime.len() != n2 {
        return Err(length_error("ntime", "nid", ntime.len(), n2));
    }
    validate_finite_values("time1", &time1)?;
    validate_finite_values("ntime", &ntime)?;
    validate_sorted_id_time("id", &id, "time1", &time1)?;
    validate_sorted_id_time("nid", &nid, "ntime", &ntime)?;

    let mut result = vec![0; n1];
    let mut k = 0;
    for i in 0..n1 {
        let current_id = id[i];
        let start_time = time1[i];
        result[i] = 0;
        while k < n2 && nid[k] < current_id {
            k += 1;
        }
        let mut last_valid = 0;
        let mut local_k = k;
        while local_k < n2 && nid[local_k] == current_id && ntime[local_k] <= start_time {
            last_valid = local_k + 1;
            local_k += 1;
        }
        result[i] = last_valid;
    }
    Ok(result)
}
#[pyfunction]
pub fn tmerge3(id: Vec<i32>, miss: Vec<bool>) -> PyResult<Vec<usize>> {
    let n = id.len();
    if miss.len() != n {
        return Err(length_error("miss", "id", miss.len(), n));
    }
    validate_non_decreasing_id("id", &id)?;

    let mut result = vec![0; n];
    let mut last_good = 0;
    let mut current_id: Option<i32> = None;
    for (i, (&current, is_missing)) in id.iter().zip(miss).enumerate() {
        if current_id != Some(current) {
            current_id = Some(current);
            last_good = 0;
        }
        if is_missing {
            result[i] = last_good;
        } else {
            result[i] = i + 1;
            last_good = i + 1;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn tmerge_single_id_cumulative_sum() {
        let result = tmerge(
            vec![1, 1],
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![1],
            vec![0.5],
            vec![10.0],
        )
        .unwrap();
        assert!((result[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge_multiple_ids_dont_mix() {
        let result = tmerge(
            vec![1, 2],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1, 2],
            vec![0.5, 0.5],
            vec![10.0, 20.0],
        )
        .unwrap();
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge_nan_replacement() {
        let result = tmerge(
            vec![1, 1],
            vec![1.0, 2.0],
            vec![f64::NAN, f64::NAN],
            vec![1],
            vec![0.5],
            vec![5.0],
        )
        .unwrap();
        assert!((result[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge2_basic_matching() {
        let result = tmerge2(vec![1, 1], vec![1.0, 2.0], vec![1, 1], vec![0.5, 1.5]).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn tmerge2_no_matches() {
        let result = tmerge2(vec![1], vec![0.0], vec![2], vec![1.0]).unwrap();
        assert_eq!(result[0], 0);
    }

    #[test]
    fn tmerge3_no_missing() {
        let result = tmerge3(vec![1, 1, 1], vec![false, false, false]).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn tmerge3_missing_uses_last_good() {
        let result = tmerge3(vec![1, 1, 1], vec![false, true, false]).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 3);
    }

    #[test]
    fn tmerge3_id_transition_resets() {
        let result = tmerge3(vec![1, 2, 2], vec![false, true, false]).unwrap();
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 3);
    }

    #[test]
    fn tmerge_accumulates_within_subject_and_resets_at_boundaries() {
        let result = tmerge(
            vec![1, 1, 2, 2, 2],
            vec![1.0, 3.0, 0.5, 1.5, 3.0],
            vec![f64::NAN, 10.0, f64::NAN, 1.0, f64::NAN],
            vec![1, 1, 2, 2, 2],
            vec![0.5, 2.5, 0.25, 1.0, 2.0],
            vec![2.0, 3.0, 5.0, 7.0, 11.0],
        )
        .unwrap();

        assert_eq!(result.len(), 5);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10);
        assert!((result[2] - 5.0).abs() < 1e-10);
        assert!((result[3] - 13.0).abs() < 1e-10);
        assert!((result[4] - 23.0).abs() < 1e-10);
    }

    #[test]
    fn tmerge2_indices_are_subject_local_and_monotone() {
        let result = tmerge2(
            vec![1, 1, 2, 2],
            vec![1.0, 3.0, 0.5, 3.0],
            vec![1, 1, 2, 2, 2],
            vec![0.5, 2.5, 0.25, 1.0, 2.0],
        )
        .unwrap();

        assert_eq!(result, vec![1, 2, 3, 5]);
        assert!(result[0] <= result[1]);
        assert!(result[2] <= result[3]);
    }

    #[test]
    fn tmerge_handles_negative_ids_without_sentinel_collision() {
        let result = tmerge(
            vec![-1, -1],
            vec![1.0, 2.0],
            vec![f64::NAN, f64::NAN],
            vec![-1],
            vec![0.5],
            vec![4.0],
        )
        .unwrap();
        assert_eq!(result, vec![4.0, 4.0]);
    }

    #[test]
    fn length_mismatches_are_value_errors() {
        assert!(tmerge(vec![1], vec![], vec![0.0], vec![], vec![], vec![]).is_err());
        assert!(tmerge2(vec![1], vec![], vec![], vec![]).is_err());
        assert!(tmerge3(vec![1], vec![]).is_err());
    }

    #[test]
    fn tmerge_rejects_nonfinite_values_and_unsorted_inputs() {
        initialize_python();

        let err = tmerge(vec![1], vec![f64::NAN], vec![0.0], vec![], vec![], vec![]).unwrap_err();
        assert!(err.to_string().contains("time1 values must be finite"));

        let err = tmerge(
            vec![1],
            vec![1.0],
            vec![f64::INFINITY],
            vec![],
            vec![],
            vec![],
        )
        .unwrap_err();
        assert!(err.to_string().contains("newx values may be finite or NaN"));

        let err = tmerge(
            vec![2, 1],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![],
            vec![],
            vec![],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("id must be sorted in non-decreasing order")
        );

        let err = tmerge(
            vec![1, 1],
            vec![2.0, 1.0],
            vec![0.0, 0.0],
            vec![],
            vec![],
            vec![],
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("time1 must be non-decreasing within id")
        );

        let err = tmerge(
            vec![1],
            vec![1.0],
            vec![0.0],
            vec![1],
            vec![f64::INFINITY],
            vec![1.0],
        )
        .unwrap_err();
        assert!(err.to_string().contains("ntime values must be finite"));

        let err = tmerge(
            vec![1],
            vec![1.0],
            vec![0.0],
            vec![1],
            vec![0.5],
            vec![f64::NAN],
        )
        .unwrap_err();
        assert!(err.to_string().contains("x values must be finite"));
    }

    #[test]
    fn tmerge2_and_tmerge3_reject_unsorted_inputs() {
        initialize_python();

        let err = tmerge2(vec![1, 1], vec![2.0, 1.0], vec![], vec![]).unwrap_err();
        assert!(
            err.to_string()
                .contains("time1 must be non-decreasing within id")
        );

        let err = tmerge2(vec![1], vec![1.0], vec![2, 1], vec![0.5, 0.5]).unwrap_err();
        assert!(
            err.to_string()
                .contains("nid must be sorted in non-decreasing order")
        );

        let err = tmerge3(vec![2, 1], vec![false, false]).unwrap_err();
        assert!(
            err.to_string()
                .contains("id must be sorted in non-decreasing order")
        );
    }
}
