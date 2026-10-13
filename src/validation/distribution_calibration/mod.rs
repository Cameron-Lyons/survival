use crate::constants::{clamped_normal_ci_95, same_time};
use crate::internal::statistical::chi2_sf;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_non_negative, validate_probability_slice,
};
use crate::simd_ops::{dot_product_simd, mean_simd, subtract_scalar_simd, sum_of_squares_simd};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_survival_probabilities(values: &[f64], field: &str) -> PyResult<()> {
    validate_probability_slice(values, field)
}

fn validate_calibration_observations(time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

fn validate_time_point(time_point: f64) -> PyResult<()> {
    validate_finite(&[time_point], "time_point")?;
    if time_point < 0.0 {
        return Err(PyValueError::new_err("time_point must be non-negative"));
    }
    Ok(())
}

fn validate_prediction_times(prediction_times: &[f64]) -> PyResult<()> {
    validate_finite(prediction_times, "prediction_times")?;
    validate_non_negative(prediction_times, "prediction_times")?;
    for (index, pair) in prediction_times.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(PyValueError::new_err(format!(
                "prediction_times must be sorted in nondecreasing order; index {} has {} before {}",
                index + 1,
                pair[1],
                pair[0]
            )));
        }
    }
    Ok(())
}

fn validate_bandwidth(bandwidth: Option<f64>) -> PyResult<()> {
    if let Some(value) = bandwidth {
        validate_finite(&[value], "bandwidth")?;
        if value <= 0.0 {
            return Err(PyValueError::new_err("bandwidth must be positive"));
        }
    }
    Ok(())
}

#[inline]
fn at_or_before_time_point(time: f64, time_point: f64) -> bool {
    time <= time_point || same_time(time, time_point)
}

#[inline]
fn after_time_point(time: f64, time_point: f64) -> bool {
    time > time_point && !same_time(time, time_point)
}

#[inline]
fn event_at_or_before_time_point(time: f64, status: i32, time_point: f64) -> bool {
    status == 1 && at_or_before_time_point(time, time_point)
}

#[inline]
fn observed_survival_at_time_point(time: f64, status: i32, time_point: f64) -> Option<f64> {
    if event_at_or_before_time_point(time, status, time_point) {
        Some(0.0)
    } else if after_time_point(time, time_point) {
        Some(1.0)
    } else {
        None
    }
}

include!("dcal.rs");
include!("one_calibration.rs");
include!("plot.rs");
include!("brier.rs");
include!("multi_time.rs");
include!("smoothed.rs");
include!("tests.rs");
