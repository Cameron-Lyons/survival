use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::internal::validation::{validate_binary_i32, validate_finite};

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

fn validate_training_targets(time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_finite(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

pub(crate) fn validate_training_values(x: &[f64], time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_finite(x, "x")?;
    validate_training_targets(time, status)
}

pub(crate) fn validate_deep_training_values(
    x: &[f64],
    time: &[f64],
    status: &[i32],
) -> PyResult<()> {
    // Training stores features in f32 tensors; inference instead uses f64.
    // Check the converted value once, allowing ordinary rounding and underflow.
    for (index, &value) in x.iter().enumerate() {
        if !(value as f32).is_finite() {
            return Err(PyValueError::new_err(format!(
                "x must contain finite values after f32 conversion; got {value} at index {index}"
            )));
        }
    }
    validate_training_targets(time, status)
}

pub(crate) fn validate_prediction_input(x: &[f64], n_new: usize, n_vars: usize) -> PyResult<()> {
    let expected = n_new
        .checked_mul(n_vars)
        .ok_or_else(|| PyValueError::new_err("n_new * n_vars overflows usize"))?;
    if x.len() != expected {
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
    validate_finite(x, "x_new")?;
    Ok(())
}
