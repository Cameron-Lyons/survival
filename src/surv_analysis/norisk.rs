use pyo3::prelude::*;

use crate::internal::validation::validate_binary_i32;

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn validate_same_length(n: usize, actual: usize, name: &str) -> PyResult<()> {
    if actual != n {
        return Err(value_error(format!(
            "{name} length must match time1 length ({actual} != {n})"
        )));
    }
    Ok(())
}

fn validate_finite(values: &[f64], name: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} must contain only finite values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_sort_indices(values: &[i32], n: usize, name: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 || value as usize >= n {
            return Err(value_error(format!(
                "{name} index out of bounds at position {idx}: {value}"
            )));
        }
    }
    Ok(())
}

fn validate_strata_boundaries(values: &[i32], n: usize) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 || value as usize > n {
            return Err(value_error(format!(
                "strata values must be between 0 and {n}; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_norisk_inputs(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    sort1: &[i32],
    sort2: &[i32],
    strata: &[i32],
) -> PyResult<()> {
    let n = time1.len();
    validate_same_length(n, time2.len(), "time2")?;
    validate_same_length(n, status.len(), "status")?;
    validate_same_length(n, sort1.len(), "sort1")?;
    validate_same_length(n, sort2.len(), "sort2")?;
    validate_finite(time1, "time1")?;
    validate_finite(time2, "time2")?;
    validate_binary_i32(status, "status")?;
    validate_sort_indices(sort1, n, "sort1")?;
    validate_sort_indices(sort2, n, "sort2")?;
    validate_strata_boundaries(strata, n)
}

#[pyfunction]
pub fn norisk(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    sort1: Vec<i32>,
    sort2: Vec<i32>,
    strata: Vec<i32>,
) -> PyResult<Vec<i32>> {
    validate_norisk_inputs(&time1, &time2, &status, &sort1, &sort2, &strata)?;
    let time1_slice = &time1;
    let time2_slice = &time2;
    let status_slice = &status;
    let sort1_slice = &sort1;
    let sort2_slice = &sort2;
    let strata_slice = &strata;
    let n = time1_slice.len();
    let mut notused = vec![0; n];
    let mut ndeath = 0;
    let mut istrat = 0;
    let mut j = 0;
    for (i, &sort2_i) in sort2_slice.iter().enumerate() {
        let p2 = sort2_i as usize;
        let dtime = time2_slice[p2];
        if i == strata_slice.get(istrat).copied().unwrap_or(n as i32) as usize {
            while j < i {
                let p1 = sort1_slice[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
            ndeath = 0;
            istrat += 1;
        } else {
            while j < i && time1_slice[sort1_slice[j] as usize] >= dtime {
                let p1 = sort1_slice[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
        }
        ndeath += status_slice[p2];
        if j < n {
            let p1 = sort1_slice[j] as usize;
            notused[p1] = ndeath;
        }
    }
    while j < n {
        let p1 = sort1_slice[j] as usize;
        notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
        j += 1;
    }
    Ok(notused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norisk_rejects_mismatched_lengths() {
        let err = match norisk(
            vec![0.0, 1.0],
            vec![1.0],
            vec![1, 0],
            vec![0, 1],
            vec![0, 1],
            vec![],
        ) {
            Ok(_) => panic!("mismatched time2 length should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("time2 length"));
    }

    #[test]
    fn norisk_rejects_negative_sort_index() {
        let err = match norisk(vec![0.0], vec![1.0], vec![1], vec![-1], vec![0], vec![]) {
            Ok(_) => panic!("negative sort index should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("sort1 index out of bounds"));
    }

    #[test]
    fn norisk_rejects_non_binary_status() {
        let err = match norisk(vec![0.0], vec![1.0], vec![2], vec![0], vec![0], vec![]) {
            Ok(_) => panic!("non-binary status should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("status must contain only 0/1"));
    }
}
