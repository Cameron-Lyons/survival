use crate::internal::validation::{
    PermutationIndexError, ValidationError, validate_length, validate_zero_based_i32_permutation,
    validate_zero_based_usize_permutation,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn validation_err_to_pyresult<T>(result: Result<T, ValidationError>) -> PyResult<T> {
    result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub(crate) fn validate_concordance_inputs(
    time_data_len: usize,
    n: usize,
    indices_len: usize,
    weights_len: usize,
) -> PyResult<()> {
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }
    validation_err_to_pyresult(validate_length(2 * n, time_data_len, "time_data"))?;
    validation_err_to_pyresult(validate_length(n, indices_len, "indices"))?;
    validation_err_to_pyresult(validate_length(n, weights_len, "weights"))?;
    Ok(())
}

pub(crate) fn validate_extended_concordance_inputs(
    time_data_len: usize,
    n: usize,
    indices_len: usize,
    weights_len: usize,
    time_weights_len: usize,
    sort_stop_len: usize,
) -> PyResult<()> {
    validate_concordance_inputs(time_data_len, n, indices_len, weights_len)?;
    validation_err_to_pyresult(validate_length(n, time_weights_len, "time_weights"))?;
    validation_err_to_pyresult(validate_length(n, sort_stop_len, "sort_stop"))?;
    Ok(())
}

pub(crate) fn validate_non_negative_i32_indices(values: &[i32], field: &str) -> PyResult<()> {
    if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| **value < 0) {
        return Err(PyRuntimeError::new_err(format!(
            "{field} contains negative value {value} at index {index}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_i32_tree_indices(values: &[i32], ntree: i32, field: &str) -> PyResult<()> {
    if ntree <= 0 {
        return Err(PyRuntimeError::new_err("ntree must be positive"));
    }

    validate_non_negative_i32_indices(values, field)?;
    let ntree = ntree as usize;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| **value as usize >= ntree)
    {
        return Err(PyRuntimeError::new_err(format!(
            "{field} value {value} at index {index} is outside ntree {ntree}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_i32_permutation_indices(
    values: &[i32],
    n: usize,
    field: &str,
) -> PyResult<()> {
    match validate_zero_based_i32_permutation(values, n) {
        Ok(()) => Ok(()),
        Err(PermutationIndexError::Negative { position, value }) => {
            Err(PyRuntimeError::new_err(format!(
                "{field} contains negative value {value} at index {position}"
            )))
        }
        Err(PermutationIndexError::OutOfBounds { position, value }) => {
            Err(PyRuntimeError::new_err(format!(
                "{field} value {value} at index {position} is outside observation count {n}"
            )))
        }
        Err(PermutationIndexError::Duplicate { position, value }) => {
            Err(PyRuntimeError::new_err(format!(
                "{field} must be a permutation; duplicate index {value} at position {position}"
            )))
        }
    }
}

pub(crate) fn validate_usize_permutation_indices(
    values: &[usize],
    n: usize,
    field: &str,
) -> PyResult<()> {
    match validate_zero_based_usize_permutation(values, n) {
        Ok(()) => Ok(()),
        Err(PermutationIndexError::OutOfBounds { position, value }) => {
            Err(PyRuntimeError::new_err(format!(
                "{field} value {value} at index {position} is outside observation count {n}"
            )))
        }
        Err(PermutationIndexError::Duplicate { position, value }) => {
            Err(PyRuntimeError::new_err(format!(
                "{field} must be a permutation; duplicate index {value} at position {position}"
            )))
        }
        Err(PermutationIndexError::Negative { .. }) => {
            unreachable!("usize indices are never negative")
        }
    }
}

pub(crate) fn build_concordance_result(
    py: Python<'_>,
    count: &[f64],
    imat: Option<&[f64]>,
    resid: Option<&[f64]>,
    n: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let concordant = count[0];
    let discordant = count[1];
    let tied_x = count[2];
    let tied_y = count[3];
    let tied_xy = count.get(4).copied().unwrap_or(0.0);
    let variance = count.get(5).copied();
    let total_pairs = concordant + discordant + tied_x + tied_y + tied_xy;
    let concordance_index = if total_pairs > 0.0 {
        (concordant + 0.5 * (tied_x + tied_y + tied_xy)) / total_pairs
    } else {
        0.0
    };
    let dict = PyDict::new(py);
    dict.set_item("concordant", concordant)?;
    dict.set_item("discordant", discordant)?;
    dict.set_item("tied_x", tied_x)?;
    dict.set_item("tied_y", tied_y)?;
    dict.set_item("tied_xy", tied_xy)?;
    dict.set_item("concordance_index", concordance_index)?;
    dict.set_item("total_pairs", total_pairs)?;
    if let Some(v) = variance {
        dict.set_item("variance", v)?;
    }
    if let Some(imat_data) = imat {
        dict.set_item("information_matrix", imat_data.to_vec())?;
    }
    if let Some(resid_data) = resid {
        dict.set_item("residuals", resid_data.to_vec())?;
    }
    if let Some(n_obs) = n {
        dict.set_item("n_observations", n_obs)?;
    }
    Ok(dict.into())
}
#[inline]
pub(crate) fn walkup_binary_tree(nwt: &[f64], twt: &[f64], index: usize, ntree: usize) -> [f64; 3] {
    let mut sums = [0.0; 3];
    if index >= ntree {
        return sums;
    }
    sums[2] = nwt[index];
    let right_child = 2 * index + 2;
    if right_child < ntree {
        sums[0] += twt[right_child];
    }
    let left_child = 2 * index + 1;
    if left_child < ntree {
        sums[1] += twt[left_child];
    }
    let mut current = index;
    while current > 0 {
        let parent = (current - 1) / 2;
        let parent_twt = twt[parent];
        let current_twt = twt[current];
        if current % 2 == 1 {
            sums[0] += parent_twt - current_twt;
        } else {
            sums[1] += parent_twt - current_twt;
        }
        current = parent;
    }
    sums
}
#[inline]
pub(crate) fn add_to_binary_tree(nwt: &mut [f64], twt: &mut [f64], index: usize, wt: f64) {
    nwt[index] += wt;
    let mut current = index;
    while current > 0 {
        let parent = (current - 1) / 2;
        twt[parent] += wt;
        current = parent;
    }
    twt[0] += wt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walkup_binary_tree_index_ge_ntree_returns_zeros() {
        let nwt = vec![0.0; 4];
        let twt = vec![0.0; 4];
        let result = walkup_binary_tree(&nwt, &twt, 5, 4);
        assert!(result[0].abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
    }

    #[test]
    fn walkup_binary_tree_root_node_small_tree() {
        let ntree = 3;
        let mut nwt = vec![0.0; ntree];
        let mut twt = vec![0.0; ntree];
        nwt[0] = 1.0;
        twt[0] = 3.0;
        twt[1] = 1.5;
        twt[2] = 1.5;
        let result = walkup_binary_tree(&nwt, &twt, 0, ntree);
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 1.5).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn add_to_binary_tree_single_add() {
        let ntree = 4;
        let mut nwt = vec![0.0; ntree];
        let mut twt = vec![0.0; ntree];
        add_to_binary_tree(&mut nwt, &mut twt, 2, 5.0);
        assert!((nwt[2] - 5.0).abs() < 1e-10);
        assert!((twt[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn add_to_binary_tree_multiple_adds() {
        let ntree = 7;
        let mut nwt = vec![0.0; ntree];
        let mut twt = vec![0.0; ntree];
        add_to_binary_tree(&mut nwt, &mut twt, 3, 2.0);
        add_to_binary_tree(&mut nwt, &mut twt, 4, 3.0);
        assert!((nwt[3] - 2.0).abs() < 1e-10);
        assert!((nwt[4] - 3.0).abs() < 1e-10);
        assert!((twt[1] - 5.0).abs() < 1e-10);
        assert!((twt[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn walkup_binary_tree_after_adds() {
        let ntree = 7;
        let mut nwt = vec![0.0; ntree];
        let mut twt = vec![0.0; ntree];
        add_to_binary_tree(&mut nwt, &mut twt, 3, 2.0);
        add_to_binary_tree(&mut nwt, &mut twt, 4, 3.0);
        add_to_binary_tree(&mut nwt, &mut twt, 5, 1.0);
        let result = walkup_binary_tree(&nwt, &twt, 4, ntree);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!(result[0] > 0.0);
        assert!(result[1] > 0.0);
        assert!((result[0] - 7.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn validate_i32_tree_indices_rejects_invalid_values() {
        assert!(validate_i32_tree_indices(&[0, 1], 2, "indices").is_ok());

        let negative = validate_i32_tree_indices(&[0, -1], 2, "indices")
            .expect_err("negative rank should fail");
        assert!(negative.to_string().contains("negative value"));

        let out_of_bounds = validate_i32_tree_indices(&[0, 2], 2, "indices")
            .expect_err("rank outside tree should fail");
        assert!(out_of_bounds.to_string().contains("outside ntree"));
    }

    #[test]
    fn validate_order_indices_rejects_invalid_values() {
        assert!(validate_i32_permutation_indices(&[1, 0], 2, "sort_stop").is_ok());
        assert!(validate_usize_permutation_indices(&[1, 0], 2, "sort_stop").is_ok());

        let negative = validate_i32_permutation_indices(&[0, -1], 2, "sort_stop")
            .expect_err("negative order index should fail");
        assert!(negative.to_string().contains("negative value"));

        let i32_out_of_bounds = validate_i32_permutation_indices(&[0, 2], 2, "sort_stop")
            .expect_err("order index outside observations should fail");
        assert!(
            i32_out_of_bounds
                .to_string()
                .contains("outside observation count")
        );

        let usize_out_of_bounds = validate_usize_permutation_indices(&[0, 2], 2, "sort_stop")
            .expect_err("order index outside observations should fail");
        assert!(
            usize_out_of_bounds
                .to_string()
                .contains("outside observation count")
        );

        let i32_duplicate = validate_i32_permutation_indices(&[0, 0], 2, "sort_stop")
            .expect_err("duplicate order index should fail");
        assert!(i32_duplicate.to_string().contains("must be a permutation"));

        let usize_duplicate = validate_usize_permutation_indices(&[0, 0], 2, "sort_stop")
            .expect_err("duplicate order index should fail");
        assert!(
            usize_duplicate
                .to_string()
                .contains("must be a permutation")
        );
    }
}
