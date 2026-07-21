use crate::constants::{
    IPCW_SURVIVAL_FLOOR, PARALLEL_THRESHOLD_LARGE, clamped_normal_ci_95, normal_ci_95, same_time,
};
use crate::internal::fenwick::FenwickTree;
use crate::internal::statistical::{compute_censoring_km, km_step_prob_at, normal_cdf};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_empty,
    validate_non_negative,
};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

#[inline]
fn at_or_before_tau(time: f64, tau: f64) -> bool {
    time <= tau || same_time(time, tau)
}

#[inline]
fn after_event_time(candidate: f64, event_time: f64) -> bool {
    candidate > event_time && !same_time(candidate, event_time)
}

fn validate_uno_time_status(time: &[f64], status: &[i32]) -> PyResult<()> {
    validate_non_empty(time, "time")?;
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    Ok(())
}

fn validate_uno_risk_score(risk_score: &[f64], field: &'static str) -> PyResult<()> {
    validate_no_nan(risk_score, field)?;
    validate_finite(risk_score, field)?;
    Ok(())
}

fn validate_uno_tau(tau: Option<f64>) -> PyResult<()> {
    if let Some(value) = tau {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tau must be finite",
            ));
        }
        if value < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "tau must be non-negative",
            ));
        }
    }
    Ok(())
}

include!("uno.rs");
include!("comparison.rs");
include!("decomposition.rs");
include!("gonen_heller.rs");
include!("tests.rs");
