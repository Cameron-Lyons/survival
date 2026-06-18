use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::PARALLEL_THRESHOLD_MEDIUM;
use crate::internal::matrix::invert_flat_square_matrix_with_fallback;
use crate::internal::statistical::{lower_incomplete_gamma, normal_cdf};

fn diagnostic_value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn validate_finite_slice(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(diagnostic_value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_binary_event(event: &[i32]) -> PyResult<()> {
    for (idx, &value) in event.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(diagnostic_value_error(format!(
                "event must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_positive_finite_scalar(name: &str, value: f64) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(diagnostic_value_error(format!(
            "{name} must be a finite positive value"
        )));
    }
    Ok(())
}

fn validate_nonnegative_finite_scalar(name: &str, value: f64) -> PyResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(diagnostic_value_error(format!(
            "{name} must be a finite non-negative value"
        )));
    }
    Ok(())
}

fn validate_optional_positive_finite_scalar(name: &str, value: Option<f64>) -> PyResult<()> {
    if let Some(value) = value {
        validate_positive_finite_scalar(name, value)?;
    }
    Ok(())
}

fn validate_cox_diagnostic_inputs(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    coefficients: &[f64],
) -> PyResult<()> {
    let n = time.len();
    if n < 2 {
        return Err(diagnostic_value_error(
            "time must contain at least two observations",
        ));
    }
    if n_covariates == 0 {
        return Err(diagnostic_value_error("n_covariates must be positive"));
    }
    if event.len() != n {
        return Err(diagnostic_value_error(format!(
            "event has {} rows but time has {}",
            event.len(),
            n
        )));
    }
    let expected_covariates = n
        .checked_mul(n_covariates)
        .ok_or_else(|| diagnostic_value_error("n * n_covariates is too large"))?;
    if covariates.len() != expected_covariates {
        return Err(diagnostic_value_error(format!(
            "covariates must have length n * n_covariates ({expected_covariates}); got {}",
            covariates.len()
        )));
    }
    if coefficients.len() != n_covariates {
        return Err(diagnostic_value_error(format!(
            "coefficients must have length n_covariates ({n_covariates}); got {}",
            coefficients.len()
        )));
    }

    validate_finite_slice("time", time)?;
    validate_binary_event(event)?;
    validate_finite_slice("covariates", covariates)?;
    validate_finite_slice("coefficients", coefficients)?;
    Ok(())
}

include!("dfbeta.rs");
include!("leverage.rs");
include!("schoenfeld.rs");
include!("influence.rs");
include!("tests.rs");
