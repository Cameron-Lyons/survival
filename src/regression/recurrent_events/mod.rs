use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::{DIVISION_FLOOR, exp_ci_bounds_95};
use crate::internal::statistical::{chi2_cdf, ln_gamma, normal_cdf};

fn recurrent_value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn covariate_matrix_or_intercept(covariates: Vec<f64>, n: usize) -> PyResult<(usize, Vec<f64>)> {
    if n == 0 {
        return Err(recurrent_value_error(
            "at least one observation is required",
        ));
    }
    if covariates.is_empty() {
        return Ok((1, vec![1.0; n]));
    }
    if !covariates.len().is_multiple_of(n) {
        return Err(recurrent_value_error(format!(
            "covariates length {} must be a multiple of the number of observations {}",
            covariates.len(),
            n
        )));
    }
    if covariates.iter().any(|value| !value.is_finite()) {
        return Err(recurrent_value_error(
            "covariates must contain only finite values",
        ));
    }

    Ok((covariates.len() / n, covariates))
}

fn sorted_unique_i32(values: &[i32]) -> Vec<i32> {
    let mut unique_values = values.to_vec();
    unique_values.sort_unstable();
    unique_values.dedup();
    unique_values
}

fn index_by_i32(values: &[i32]) -> std::collections::HashMap<i32, usize> {
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| (value, idx))
        .collect()
}

fn validate_recurrent_lengths(n: usize, lengths: &[usize]) -> PyResult<()> {
    if n == 0 {
        return Err(recurrent_value_error(
            "at least one observation is required",
        ));
    }
    if lengths.iter().any(|&len| len != n) {
        return Err(recurrent_value_error(
            "All input vectors must have the same length",
        ));
    }
    Ok(())
}

fn validate_recurrent_solver_controls(max_iter: usize, tol: f64) -> PyResult<()> {
    if max_iter == 0 {
        return Err(recurrent_value_error("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(recurrent_value_error("tol must be a positive finite value"));
    }
    Ok(())
}

fn validate_finite_nonnegative(values_name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(recurrent_value_error(format!(
                "{values_name} must contain only finite values; found non-finite value at row {idx}"
            )));
        }
        if value < 0.0 {
            return Err(recurrent_value_error(format!(
                "{values_name} values must be non-negative; found negative value at row {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_binary_events(event: &[i32]) -> PyResult<()> {
    for (idx, &value) in event.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(recurrent_value_error(format!(
                "event must contain only 0/1 values; found {value} at row {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_event_counts(event: &[i32]) -> PyResult<()> {
    for (idx, &value) in event.iter().enumerate() {
        if value < 0 {
            return Err(recurrent_value_error(format!(
                "event counts must be non-negative; found {value} at row {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_counting_process_inputs(start: &[f64], stop: &[f64], event: &[i32]) -> PyResult<()> {
    validate_finite_nonnegative("start", start)?;
    validate_finite_nonnegative("stop", stop)?;
    validate_binary_events(event)?;

    for (idx, (&start_time, &stop_time)) in start.iter().zip(stop.iter()).enumerate() {
        if stop_time <= start_time {
            return Err(recurrent_value_error(format!(
                "stop must be greater than start at row {idx}"
            )));
        }
    }

    Ok(())
}

fn validate_time_event_inputs(time: &[f64], event: &[i32]) -> PyResult<()> {
    validate_finite_nonnegative("time", time)?;
    validate_binary_events(event)
}

fn validate_event_numbers(event_number: &[i32]) -> PyResult<()> {
    let n = event_number.len();
    for (idx, &value) in event_number.iter().enumerate() {
        if value <= 0 || value as usize > n {
            return Err(recurrent_value_error(format!(
                "event_number values must be between 1 and the number of observations; found {value} at row {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_offset(offset: &[f64], n: usize) -> PyResult<()> {
    validate_recurrent_lengths(n, &[offset.len()])?;
    for (idx, &value) in offset.iter().enumerate() {
        if !value.is_finite() {
            return Err(recurrent_value_error(format!(
                "offset must contain only finite values; found non-finite value at row {idx}"
            )));
        }
    }
    Ok(())
}

include!("pwp.rs");
include!("wlw.rs");
include!("negative_binomial.rs");
include!("anderson_gill.rs");
include!("tests.rs");
