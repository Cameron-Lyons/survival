//! Python entry point of the IPCW Brier score (`validation::brier`).

use crate::data_types::{FloatMatrix, FloatVec, IntVec};
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
    time: FloatVec,
    status: IntVec,
    times: FloatVec,
    phat: FloatMatrix,
    weights: Option<FloatVec>,
    ties: bool,
    efron: bool,
    timefix: bool,
    start: Option<FloatVec>,
) -> PyResult<BrierResult> {
    Ok(py.detach(|| {
        brier(BrierInput {
            start: start.as_deref(),
            time: &time,
            status: &status,
            weights: weights.as_deref(),
            times: &times,
            phat: phat.outer_iter().map(|row| row.to_vec()).collect(),
            ties,
            efron,
            timefix,
        })
    })?)
}
