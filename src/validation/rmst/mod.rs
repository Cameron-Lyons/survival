use crate::constants::{
    DEFAULT_CONFIDENCE_LEVEL, PARALLEL_THRESHOLD_XLARGE, clamped_normal_ci, exp_ci, normal_ci,
    z_score_for_confidence,
};
use crate::internal::statistical::{chi2_cdf, normal_cdf as norm_cdf};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_non_negative};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

fn validate_time_status(time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

fn validate_time_and_nonnegative_status(time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    for (index, &value) in status.iter().enumerate() {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "status must contain non-negative event codes; got {value} at index {index}"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_time_horizon(value: f64, field: &'static str) -> PyResult<()> {
    validate_finite(&[value], field)?;
    if value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{field} must be non-negative"
        )));
    }
    Ok(())
}

fn validate_probability_open(value: f64, field: &'static str) -> PyResult<()> {
    validate_finite(&[value], field)?;
    if !(value > 0.0 && value < 1.0) {
        return Err(PyValueError::new_err(format!(
            "{field} must be greater than 0 and less than 1"
        )));
    }
    Ok(())
}

include!("core.rs");
include!("quantiles.rs");
include!("threshold.rs");
include!("tests.rs");
