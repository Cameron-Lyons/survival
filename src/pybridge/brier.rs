//! Python entry point of the IPCW Brier score (`validation::brier`).

use crate::validation::brier::{BrierInput, BrierResult, brier};
use pyo3::prelude::*;

/// `brier(time, status, times, phat, weights=None, ties=True, efron=False,
/// timefix=True, start=None)`: R's `brier(fit, times, ties, efron)` for a
/// Cox model, `phat[i][j]` being the model's predicted probability that
/// subject `j` has had the event by `times[i]`; `start` holds the entry
/// times of (start, stop] data.
#[pyfunction(name = "brier")]
#[pyo3(signature = (time, status, times, phat, weights=None, ties=true, efron=false, timefix=true, start=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn brier_py(
    py: Python<'_>,
    time: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    phat: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    ties: bool,
    efron: bool,
    timefix: bool,
    start: Option<Vec<f64>>,
) -> PyResult<BrierResult> {
    Ok(py.detach(|| {
        brier(&BrierInput {
            start: start.as_deref(),
            time: &time,
            status: &status,
            weights: weights.as_deref(),
            times: &times,
            phat: &phat,
            ties,
            efron,
            timefix,
        })
    })?)
}
