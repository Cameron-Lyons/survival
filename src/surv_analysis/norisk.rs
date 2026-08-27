use pyo3::prelude::*;

use crate::internal::validation::{
    PermutationIndexError, validate_binary_i32, validate_zero_based_i32_permutation,
};

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
    match validate_zero_based_i32_permutation(values, n) {
        Ok(()) => Ok(()),
        Err(PermutationIndexError::Negative { position, value }) => Err(value_error(format!(
            "{name} index out of bounds at position {position}: {value}"
        ))),
        Err(PermutationIndexError::OutOfBounds { position, value }) => Err(value_error(format!(
            "{name} index out of bounds at position {position}: {value}"
        ))),
        Err(PermutationIndexError::Duplicate { position, value }) => Err(value_error(format!(
            "{name} must be a permutation of 0..{n}; duplicate index {value} at position {position}"
        ))),
    }
}

fn normalize_strata_boundaries(values: &mut Vec<i32>, n: usize) -> PyResult<()> {
    if values.len() == n && values.iter().all(|&value| value == 0 || value == 1) {
        let mut boundary_count = 0;
        for position in 0..n {
            if values[position] == 1 {
                values[boundary_count] = position as i32;
                boundary_count += 1;
            }
        }
        values.truncate(boundary_count);
        return Ok(());
    }

    let mut previous = None;
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 || value as usize > n {
            return Err(value_error(format!(
                "strata values must be between 0 and {n}; got {value} at index {idx}"
            )));
        }
        if let Some(previous_value) = previous
            && value <= previous_value
        {
            return Err(value_error(format!(
                "strata values must be strictly increasing; got {value} after {previous_value} at index {idx}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_norisk_inputs(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    sort1: &[i32],
    sort2: &[i32],
    strata: &mut Vec<i32>,
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
    normalize_strata_boundaries(strata, n)
}

#[pyfunction]
pub fn norisk(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    sort1: Vec<i32>,
    sort2: Vec<i32>,
    mut strata: Vec<i32>,
) -> PyResult<Vec<i32>> {
    validate_norisk_inputs(&time1, &time2, &status, &sort1, &sort2, &mut strata)?;
    let n = time1.len();
    let mut notused = vec![0; n];
    let mut ndeath = 0;
    let mut istrat = 0;
    let mut j = 0;
    for (i, &sort2_i) in sort2.iter().enumerate() {
        let p2 = sort2_i as usize;
        let dtime = time2[p2];
        if i == strata.get(istrat).copied().unwrap_or(n as i32) as usize {
            while j < i {
                let p1 = sort1[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
            ndeath = 0;
            istrat += 1;
        } else {
            while j < i && time1[sort1[j] as usize] >= dtime {
                let p1 = sort1[j] as usize;
                notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
                j += 1;
            }
        }
        ndeath += status[p2];
        if j < n {
            let p1 = sort1[j] as usize;
            notused[p1] = ndeath;
        }
    }
    while j < n {
        let p1 = sort1[j] as usize;
        notused[p1] = if ndeath > notused[p1] { 1 } else { 0 };
        j += 1;
    }
    Ok(notused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn norisk_rejects_mismatched_lengths() {
        initialize_python();

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
        initialize_python();

        let err = match norisk(vec![0.0], vec![1.0], vec![1], vec![-1], vec![0], vec![]) {
            Ok(_) => panic!("negative sort index should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("sort1 index out of bounds"));
    }

    #[test]
    fn norisk_rejects_duplicate_sort_index() {
        initialize_python();

        let err = match norisk(
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![1, 0],
            vec![0, 0],
            vec![0, 1],
            vec![],
        ) {
            Ok(_) => panic!("duplicate sort1 index should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("sort1 must be a permutation"));
    }

    #[test]
    fn norisk_rejects_non_binary_status() {
        initialize_python();

        let err = match norisk(vec![0.0], vec![1.0], vec![2], vec![0], vec![0], vec![]) {
            Ok(_) => panic!("non-binary status should fail"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn norisk_rejects_unordered_strata_boundaries() {
        initialize_python();

        let err = match norisk(
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![1, 0, 1],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![2, 1],
        ) {
            Ok(_) => panic!("unordered strata boundaries should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("strata values must be strictly increasing")
        );
    }

    #[test]
    fn norisk_normalizes_marker_style_strata_vector() {
        initialize_python();

        let args = || {
            (
                vec![0.0, 1.0, 2.0],
                vec![1.0, 2.0, 3.0],
                vec![1, 0, 1],
                vec![0, 1, 2],
                vec![0, 1, 2],
            )
        };
        let (time1, time2, status, sort1, sort2) = args();
        let markers = norisk(time1, time2, status, sort1, sort2, vec![1, 0, 0])
            .expect("marker-style strata should remain accepted");
        let (time1, time2, status, sort1, sort2) = args();
        let boundaries = norisk(time1, time2, status, sort1, sort2, vec![0])
            .expect("boundary-style strata should remain accepted");

        assert_eq!(markers, boundaries);
        assert_eq!(markers, vec![0, 1, 1]);
    }

    #[test]
    fn norisk_normalizes_multiple_stratum_markers() {
        initialize_python();

        let args = || {
            (
                vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
                vec![1, 0, 0, 0, 1, 0],
                vec![2, 1, 0, 5, 4, 3],
                vec![0, 1, 2, 3, 4, 5],
            )
        };
        let (time1, time2, status, sort1, sort2) = args();
        let markers = norisk(time1, time2, status, sort1, sort2, vec![1, 0, 0, 1, 0, 0])
            .expect("marker-style strata should remain accepted");
        let (time1, time2, status, sort1, sort2) = args();
        let boundaries = norisk(time1, time2, status, sort1, sort2, vec![0, 3])
            .expect("boundary-style strata should remain accepted");

        assert_eq!(markers, boundaries);
    }
}
