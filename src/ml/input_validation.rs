use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::config_validation::ensure_positive_usize;

pub(crate) fn validate_training_shape(
    x_len: usize,
    n_obs: usize,
    n_vars: usize,
    time_len: usize,
    status_len: usize,
) -> PyResult<()> {
    ensure_positive_usize("n_obs", n_obs)?;
    let expected = n_obs
        .checked_mul(n_vars)
        .ok_or_else(|| PyValueError::new_err("n_obs * n_vars overflows usize"))?;
    if x_len != expected {
        return Err(PyValueError::new_err("x length must equal n_obs * n_vars"));
    }
    if time_len != n_obs || status_len != n_obs {
        return Err(PyValueError::new_err(
            "time and status must have length n_obs",
        ));
    }
    Ok(())
}

pub(crate) fn validate_prediction_shape(x_len: usize, n_new: usize, n_vars: usize) -> PyResult<()> {
    let expected = n_new
        .checked_mul(n_vars)
        .ok_or_else(|| PyValueError::new_err("n_new * n_vars overflows usize"))?;
    if x_len != expected {
        return Err(PyValueError::new_err("x_new dimensions don't match"));
    }
    // Zero-feature trees can have an empty input for any row count. Their
    // outputs still need one entry per row for curves or optional event times.
    let output_row_size = size_of::<Vec<f64>>().max(size_of::<Option<f64>>());
    if n_new > (isize::MAX as usize) / output_row_size {
        return Err(PyValueError::new_err(
            "n_new is too large for prediction output",
        ));
    }
    Ok(())
}
