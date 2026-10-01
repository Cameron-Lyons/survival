//! Fit-local state for user penalty functions and outer-loop controllers.

use super::control::{ControlInput, ControlState};
use super::penalty::PenaltyValue;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatRows, FloatVec};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// R-style `pfun` and `cfun`. The penalty uses positive derivatives; the
/// controller returns a mapping whose `theta` selects the next inner fit.
#[derive(Debug, Clone)]
pub struct ControlledPenalty {
    pub pfun: Arc<Py<PyAny>>,
    pub cfun: Arc<Py<PyAny>>,
    pub diag: bool,
    pub sparse: bool,
    pub needs_df: bool,
}

/// Opaque controller state belongs to one fit, never to the reusable penalty
/// or the numerical fit result. Numeric history columns are parsed once.
pub(crate) struct CallbackState {
    value: Py<PyAny>,
    pub columns: Vec<String>,
}

fn failure(name: &str, error: PyErr) -> SurvivalError {
    SurvivalError::computation(format!("{name} callback failed: {error}"))
}

fn number(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(value) = value.extract::<f64>() {
        return Ok(value);
    }
    match value.extract::<FloatVec>()?.as_ref() {
        [value] => Ok(*value),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "expected one number",
        )),
    }
}

fn optional<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match value.get_item(name) {
        Ok(value) if !value.is_none() => Ok(Some(value)),
        Ok(_) => Ok(None),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(value.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

fn controller_state(
    value: Bound<'_, PyAny>,
    initial: bool,
) -> PyResult<(ControlState, CallbackState)> {
    let theta = number(&value.get_item("theta")?)?;
    if !theta.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must be finite",
        ));
    }
    let done = match optional(&value, "done")? {
        Some(done) => done.extract::<bool>()?,
        None if initial => false,
        None => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "controller must return done",
            ));
        }
    };
    let history = optional(&value, "history")?
        .map(|v| v.extract::<FloatRows>().map(FloatRows::into_inner))
        .transpose()?
        .unwrap_or_default();
    let columns: Vec<String> = optional(&value, "columns")?
        .map(|v| v.extract())
        .transpose()?
        .unwrap_or_default();
    if history.iter().any(|row| row.len() != columns.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "history rows must match columns",
        ));
    }
    let state = ControlState {
        theta,
        done,
        history,
        c_loglik: optional(&value, "c_loglik")?
            .map(|v| number(&v))
            .transpose()?,
        half: optional(&value, "half")?.map(|v| v.extract()).transpose()?,
        theta_history_index: None,
    };
    Ok((
        state,
        CallbackState {
            value: value.unbind(),
            columns,
        },
    ))
}

impl ControlledPenalty {
    pub(crate) fn initial(&self, eps2: f64) -> SurvivalResult<(ControlState, CallbackState)> {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("iter", 0)?;
            info.set_item("eps2", eps2)?;
            controller_state(self.cfun.bind(py).call1((py.None(), info))?, true)
        })
        .map_err(|error| failure("penalty controller", error))
    }

    pub(crate) fn update(
        &self,
        old: &CallbackState,
        input: ControlInput<'_>,
    ) -> SurvivalResult<(ControlState, CallbackState)> {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("iter", input.iter)?;
            info.set_item("coef", FloatVec(input.coef.to_vec()))?;
            for (key, value) in [
                ("plik", input.plik),
                ("loglik", input.loglik),
                ("neff", input.neff),
                ("df", input.df),
                ("trH", input.trh),
            ] {
                info.set_item(key, value)?;
            }
            controller_state(self.cfun.bind(py).call1((old.value.bind(py), info))?, false)
        })
        .map_err(|error| failure("penalty controller", error))
    }

    pub(crate) fn evaluate(
        &self,
        coef: &[f64],
        theta: f64,
        neff: f64,
        which: i32,
    ) -> SurvivalResult<PenaltyValue> {
        Python::attach(|py| {
            let value = self
                .pfun
                .bind(py)
                .call1((FloatVec(coef.to_vec()), theta, neff))?;
            let penalty = number(&value.get_item("penalty")?)?;
            let flag = value.get_item("flag")?.extract::<bool>()?;
            let mut centered = coef.to_vec();
            if let Some(recenter) = optional(&value, "recenter")? {
                let recenter = if let Ok(value) = recenter.extract::<f64>() {
                    vec![value]
                } else {
                    recenter.extract::<FloatVec>()?.into_inner()
                };
                if recenter.len() != 1 && recenter.len() != coef.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "recenter must have one value or one per coefficient",
                    ));
                }
                for (i, b) in centered.iter_mut().enumerate() {
                    *b -= recenter[i % recenter.len()];
                }
            }
            let (first, second) = if flag {
                (Vec::new(), Vec::new())
            } else {
                let mut first = value.get_item("first")?.extract::<FloatVec>()?.into_inner();
                // R recycles dense derivatives during subassignment; its
                // sparse callback requires one entry per frailty coefficient.
                if which == 2 && !first.is_empty() && coef.len().is_multiple_of(first.len()) {
                    first = first.iter().copied().cycle().take(coef.len()).collect();
                }
                (
                    first,
                    value
                        .get_item("second")?
                        .extract::<FloatVec>()?
                        .into_inner(),
                )
            };
            Ok(PenaltyValue {
                coef: centered,
                first,
                second,
                penalty,
                flag,
            })
        })
        .map_err(|error| failure("penalty", error))
    }
}
