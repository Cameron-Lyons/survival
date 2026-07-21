use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::error::Error;

use crate::internal::validation::{validate_finite, validate_no_nan, validate_non_negative};
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct SurvFitAJ {
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_event: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_censor: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pstate: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_chaz: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub std_auc: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub influence: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub n_transition: Vec<Vec<f64>>,
}
#[derive(Debug)]
struct SurvFitAJComputed {
    pub n_risk: Array2<f64>,
    pub n_event: Array2<f64>,
    pub n_censor: Array2<f64>,
    pub pstate: Array2<f64>,
    pub cumhaz: Array2<f64>,
    pub std_err: Option<Array2<f64>>,
    pub std_chaz: Option<Array2<f64>>,
    pub std_auc: Option<Array2<f64>>,
    pub influence: Option<Array2<f64>>,
    pub n_enter: Option<Array2<f64>>,
    pub n_transition: Array2<f64>,
}
impl SurvFitAJComputed {
    fn into_python_result(self) -> SurvFitAJ {
        let array2_to_vec = |arr: Array2<f64>| -> Vec<Vec<f64>> {
            arr.outer_iter().map(|row| row.to_vec()).collect()
        };
        let option_array2_to_vec =
            |opt: Option<Array2<f64>>| -> Option<Vec<Vec<f64>>> { opt.map(array2_to_vec) };
        SurvFitAJ {
            n_risk: array2_to_vec(self.n_risk),
            n_event: array2_to_vec(self.n_event),
            n_censor: array2_to_vec(self.n_censor),
            pstate: array2_to_vec(self.pstate),
            cumhaz: array2_to_vec(self.cumhaz),
            std_err: option_array2_to_vec(self.std_err),
            std_chaz: option_array2_to_vec(self.std_chaz),
            std_auc: option_array2_to_vec(self.std_auc),
            influence: option_array2_to_vec(self.influence),
            n_enter: option_array2_to_vec(self.n_enter),
            n_transition: array2_to_vec(self.n_transition),
        }
    }
}

pub(crate) struct SurvFitAJData<'a> {
    pub y: &'a [f64],
    pub sort1: &'a [usize],
    pub sort2: &'a [usize],
    pub utime: &'a [f64],
    pub cstate: &'a [usize],
    pub wt: &'a [f64],
    pub grp: &'a [usize],
    pub position: &'a [usize],
}

pub(crate) struct SurvFitAJParams<'a> {
    pub ngrp: usize,
    pub p0: &'a [f64],
    pub i0: &'a [f64],
    pub sefit: i32,
    pub entry: bool,
    pub hindx: &'a Array2<usize>,
    pub trmat: &'a Array2<usize>,
    pub t0: f64,
}

struct SurvFitAJCounts {
    n_risk: Array2<f64>,
    n_event: Array2<f64>,
    n_censor: Array2<f64>,
    n_enter: Option<Array2<f64>>,
    n_transition: Array2<f64>,
}

struct SurvFitAJEstimates {
    pstate: Array2<f64>,
    cumhaz: Array2<f64>,
    std_err: Option<Array2<f64>>,
    std_chaz: Option<Array2<f64>>,
    std_auc: Option<Array2<f64>>,
    influence: Option<Array2<f64>>,
}

#[inline]
fn observation_start(y: &[f64], idx: usize) -> f64 {
    y[idx * 3]
}

#[inline]
fn observation_stop(y: &[f64], idx: usize) -> f64 {
    y[idx * 3 + 1]
}

#[inline]
fn observation_target(y: &[f64], idx: usize) -> Option<usize> {
    (y[idx * 3 + 2] as usize).checked_sub(1)
}

fn compute_survfitaj_counts(
    data: &SurvFitAJData<'_>,
    params: &SurvFitAJParams<'_>,
) -> SurvFitAJCounts {
    let ntime = data.utime.len();
    let nused = data.sort1.len();
    let nstate = params.p0.len();
    let nhaz = params.trmat.nrows();
    let mut n_risk = Array2::zeros((ntime, 2 * nstate));
    let mut n_event = Array2::zeros((ntime, nstate));
    let mut n_censor = Array2::zeros((ntime, 2 * nstate));
    let mut n_transition = Array2::zeros((ntime, 2 * nhaz));
    let mut n_enter = params.entry.then(|| Array2::zeros((ntime, 2 * nstate)));
    let mut running_risk = Array1::zeros(2 * nstate);
    let mut start_cursor = nused;
    let mut stop_cursor = nused;

    for time_idx in (0..ntime).rev() {
        let current_time = data.utime[time_idx];
        while start_cursor > 0 {
            let idx = data.sort1[start_cursor - 1];
            if observation_start(data.y, idx) < current_time {
                break;
            }
            let from = data.cstate[idx];
            running_risk[from] -= data.wt[idx];
            running_risk[from + nstate] -= 1.0;
            if params.entry
                && data.position[idx] & 1 != 0
                && let Some(ref mut enter) = n_enter
            {
                enter[[time_idx, from]] += data.wt[idx];
                enter[[time_idx, from + nstate]] += 1.0;
            }
            start_cursor -= 1;
        }

        while stop_cursor > 0 {
            let idx = data.sort2[stop_cursor - 1];
            if observation_stop(data.y, idx) < current_time {
                break;
            }
            let from = data.cstate[idx];
            running_risk[from] += data.wt[idx];
            running_risk[from + nstate] += 1.0;
            if let Some(target) = observation_target(data.y, idx) {
                let transition = params.hindx[[from, target]];
                n_transition[[time_idx, transition]] += data.wt[idx];
                n_transition[[time_idx, transition + nhaz]] += 1.0;
                n_event[[time_idx, target]] += data.wt[idx];
            } else if data.position[idx] > 1 {
                n_censor[[time_idx, from]] += data.wt[idx];
                n_censor[[time_idx, from + nstate]] += 1.0;
            }
            stop_cursor -= 1;
        }
        n_risk.row_mut(time_idx).assign(&running_risk);
    }

    SurvFitAJCounts {
        n_risk,
        n_event,
        n_censor,
        n_enter,
        n_transition,
    }
}

fn advance_survfitaj_estimates(
    counts: &SurvFitAJCounts,
    params: &SurvFitAJParams<'_>,
    time_idx: usize,
    phat: &mut Array1<f64>,
    phat_before: &mut Array1<f64>,
    cumulative_hazard: &mut Array1<f64>,
) -> Result<(), Box<dyn Error>> {
    let mut frozen = false;
    for transition in 0..params.trmat.nrows() {
        let transition_weight = counts.n_transition[[time_idx, transition]];
        if transition_weight == 0.0 {
            continue;
        }
        if !frozen {
            phat_before.assign(phat);
            frozen = true;
        }
        let from = params.trmat[[transition, 0]];
        let to = params.trmat[[transition, 1]];
        let risk = counts.n_risk[[time_idx, from]];
        if risk <= 0.0 {
            return Err(format!(
                "non-positive weighted risk set for transition {from}->{to} at output row {time_idx}"
            )
            .into());
        }
        let hazard = transition_weight / risk;
        cumulative_hazard[transition] += hazard;
        if from != to {
            phat[from] -= phat_before[from] * hazard;
            phat[to] += phat_before[from] * hazard;
        }
    }
    Ok(())
}

fn validate_survfitaj_transition_risk(
    counts: &SurvFitAJCounts,
    params: &SurvFitAJParams<'_>,
) -> Result<(), Box<dyn Error>> {
    let mut outgoing = vec![0.0; params.p0.len()];
    for time_idx in 0..counts.n_risk.nrows() {
        outgoing.fill(0.0);
        for transition in 0..params.trmat.nrows() {
            let weight = counts.n_transition[[time_idx, transition]];
            if weight > 0.0 {
                outgoing[params.trmat[[transition, 0]]] += weight;
            }
        }
        for (state, &weight) in outgoing.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let risk = counts.n_risk[[time_idx, state]];
            if risk <= 0.0 {
                return Err(format!(
                    "non-positive weighted risk set for state {state} at output row {time_idx}"
                )
                .into());
            }
            let tolerance = 64.0 * f64::EPSILON * risk.abs().max(weight.abs()).max(1.0);
            if weight > risk + tolerance {
                return Err(format!(
                    "outgoing transition weight {weight} exceeds risk weight {risk} for state {state} at output row {time_idx}"
                )
                .into());
            }
        }
    }
    Ok(())
}

fn compute_survfitaj_estimates(
    data: &SurvFitAJData<'_>,
    params: &SurvFitAJParams<'_>,
    counts: &SurvFitAJCounts,
) -> Result<SurvFitAJEstimates, Box<dyn Error>> {
    let ntime = data.utime.len();
    let nused = data.sort1.len();
    let nstate = params.p0.len();
    let nhaz = params.trmat.nrows();
    let mut pstate = Array2::zeros((ntime, nstate));
    let mut cumhaz = Array2::zeros((ntime, nhaz));
    let mut phat = Array1::from_vec(params.p0.to_vec());
    let mut phat_before = Array1::zeros(nstate);
    let mut cumulative_hazard = Array1::zeros(nhaz);
    validate_survfitaj_transition_risk(counts, params)?;

    if params.sefit == 0 {
        for time_idx in 0..ntime {
            advance_survfitaj_estimates(
                counts,
                params,
                time_idx,
                &mut phat,
                &mut phat_before,
                &mut cumulative_hazard,
            )?;
            pstate.row_mut(time_idx).assign(&phat);
            cumhaz.row_mut(time_idx).assign(&cumulative_hazard);
        }
        return Ok(SurvFitAJEstimates {
            pstate,
            cumhaz,
            std_err: None,
            std_chaz: None,
            std_auc: None,
            influence: None,
        });
    }

    // R matrices are column-major: i0 is grouped by state, with all groups
    // for state 0 first. Preserve that layout at the Python boundary.
    let mut influence_state = Array2::zeros((params.ngrp, nstate));
    for state in 0..nstate {
        for group in 0..params.ngrp {
            influence_state[[group, state]] = params.i0[group + state * params.ngrp];
        }
    }
    let mut influence_auc = Array2::<f64>::zeros((params.ngrp, nstate));
    let mut influence_hazard = Array2::<f64>::zeros((params.ngrp, nhaz));
    let mut group_risk = Array2::<f64>::zeros((params.ngrp, nstate));
    let mut influence_before = Array2::<f64>::zeros((params.ngrp, nstate));
    let mut hazard_increment = Array2::<f64>::zeros((nstate, nstate));
    let mut se_state = Array1::<f64>::zeros(nstate);
    let mut se_hazard = Array1::<f64>::zeros(nhaz);
    let mut se_auc = Array1::<f64>::zeros(nstate);
    for state in 0..nstate {
        se_state[state] = influence_state
            .column(state)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
    }

    let mut std_err = Array2::zeros((ntime, nstate));
    let mut std_chaz = Array2::zeros((ntime, nhaz));
    let mut std_auc = Array2::zeros((ntime, nstate));
    let mut saved_influence =
        (params.sefit > 1).then(|| Array2::zeros((params.ngrp * nstate, ntime)));
    let mut start_cursor = 0;
    let mut stop_cursor = 0;

    for time_idx in 0..ntime {
        let current_time = data.utime[time_idx];
        let delta = if time_idx == 0 {
            current_time - params.t0
        } else {
            current_time - data.utime[time_idx - 1]
        };
        for state in 0..nstate {
            for group in 0..params.ngrp {
                influence_auc[[group, state]] += delta * influence_state[[group, state]];
            }
            se_auc[state] = influence_auc
                .column(state)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
        }

        while start_cursor < nused {
            let idx = data.sort1[start_cursor];
            if observation_start(data.y, idx) >= current_time {
                break;
            }
            group_risk[[data.grp[idx], data.cstate[idx]]] += data.wt[idx];
            start_cursor += 1;
        }
        while stop_cursor < nused {
            let idx = data.sort2[stop_cursor];
            if observation_stop(data.y, idx) >= current_time {
                break;
            }
            group_risk[[data.grp[idx], data.cstate[idx]]] -= data.wt[idx];
            stop_cursor += 1;
        }

        hazard_increment.fill(0.0);
        let mut event_count = 0usize;
        for &idx in &data.sort2[stop_cursor..] {
            let stop = observation_stop(data.y, idx);
            if stop > current_time {
                break;
            }
            let Some(to) = observation_target(data.y, idx) else {
                continue;
            };
            event_count += 1;
            let from = data.cstate[idx];
            let transition = params.hindx[[from, to]];
            let risk = counts.n_risk[[time_idx, from]];
            influence_hazard[[data.grp[idx], transition]] += data.wt[idx] / risk;
            if from != to {
                let increment = data.wt[idx] / risk;
                hazard_increment[[from, from]] -= increment;
                hazard_increment[[from, to]] += increment;
            }
        }

        if event_count > 0 {
            influence_before.assign(&influence_state);
            for from in 0..nstate {
                for to in 0..nstate {
                    let increment = hazard_increment[[from, to]];
                    if increment == 0.0 {
                        continue;
                    }
                    for group in 0..params.ngrp {
                        influence_state[[group, to]] += influence_before[[group, from]] * increment;
                    }
                }
            }

            for &idx in &data.sort2[stop_cursor..] {
                let stop = observation_stop(data.y, idx);
                if stop > current_time {
                    break;
                }
                let Some(to) = observation_target(data.y, idx) else {
                    continue;
                };
                let from = data.cstate[idx];
                if from != to {
                    let term = data.wt[idx] * phat[from] / counts.n_risk[[time_idx, from]];
                    influence_state[[data.grp[idx], from]] -= term;
                    influence_state[[data.grp[idx], to]] += term;
                }
            }

            for transition in 0..nhaz {
                let transition_weight = counts.n_transition[[time_idx, transition]];
                if transition_weight == 0.0 {
                    continue;
                }
                let from = params.trmat[[transition, 0]];
                let to = params.trmat[[transition, 1]];
                let risk = counts.n_risk[[time_idx, from]];
                let hazard = transition_weight / risk;
                let scaled_hazard = hazard / risk;
                for group in 0..params.ngrp {
                    let at_risk = group_risk[[group, from]];
                    if at_risk > 0.0 {
                        influence_hazard[[group, transition]] -= at_risk * scaled_hazard;
                        if from != to {
                            let term = at_risk * phat[from] * scaled_hazard;
                            influence_state[[group, from]] += term;
                            influence_state[[group, to]] -= term;
                        }
                    }
                }
            }

            for state in 0..nstate {
                se_state[state] = influence_state
                    .column(state)
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
            }
            for transition in 0..nhaz {
                se_hazard[transition] = influence_hazard
                    .column(transition)
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
            }
        }

        advance_survfitaj_estimates(
            counts,
            params,
            time_idx,
            &mut phat,
            &mut phat_before,
            &mut cumulative_hazard,
        )?;
        pstate.row_mut(time_idx).assign(&phat);
        cumhaz.row_mut(time_idx).assign(&cumulative_hazard);
        std_err.row_mut(time_idx).assign(&se_state);
        std_chaz.row_mut(time_idx).assign(&se_hazard);
        std_auc.row_mut(time_idx).assign(&se_auc);
        if let Some(ref mut influence) = saved_influence {
            for state in 0..nstate {
                for group in 0..params.ngrp {
                    influence[[group + state * params.ngrp, time_idx]] =
                        influence_state[[group, state]];
                }
            }
        }
    }

    Ok(SurvFitAJEstimates {
        pstate,
        cumhaz,
        std_err: Some(std_err),
        std_chaz: Some(std_chaz),
        std_auc: Some(std_auc),
        influence: saved_influence,
    })
}

fn compute_survfitaj(
    data: &SurvFitAJData<'_>,
    params: &SurvFitAJParams<'_>,
) -> Result<SurvFitAJComputed, Box<dyn Error>> {
    let counts = compute_survfitaj_counts(data, params);
    let estimates = compute_survfitaj_estimates(data, params, &counts)?;
    Ok(SurvFitAJComputed {
        n_risk: counts.n_risk,
        n_event: counts.n_event,
        n_censor: counts.n_censor,
        pstate: estimates.pstate,
        cumhaz: estimates.cumhaz,
        std_err: estimates.std_err,
        std_chaz: estimates.std_chaz,
        std_auc: estimates.std_auc,
        influence: estimates.influence,
        n_enter: counts.n_enter,
        n_transition: counts.n_transition,
    })
}
/// Compute the low-level Aalen--Johansen tables used by multistate survfit.
///
/// `y` is a row-major sequence of `(start, stop, status)` triples. A positive
/// status is the one-based destination state. `sort1` and `sort2` contain the
/// same observation subset in ascending start- and stop-time order. The
/// transition lookup is an `nstate x nstate` matrix: entries below `nhaz`
/// select a row of `trmat`, while `nhaz` marks an absent transition. Initial
/// influence values use R's column-major `group + state * ngrp` layout.
/// `sefit=0` skips uncertainty, `sefit=1` returns standard errors, and values
/// greater than one also return the grouped state-probability influence.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn survfitaj(
    y: Vec<f64>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    utime: Vec<f64>,
    cstate: Vec<usize>,
    wt: Vec<f64>,
    grp: Vec<usize>,
    ngrp: usize,
    p0: Vec<f64>,
    i0: Vec<f64>,
    sefit: i32,
    entry: bool,
    position: Vec<usize>,
    hindx: Vec<Vec<usize>>,
    trmat: Vec<Vec<usize>>,
    t0: f64,
) -> PyResult<SurvFitAJ> {
    validate_survfitaj_inputs(
        &y, &sort1, &sort2, &utime, &cstate, &wt, &grp, ngrp, &p0, &i0, sefit, &position, &hindx,
        entry, &trmat, t0,
    )?;

    let hindx_array = matrix_from_rows(hindx, "hindx")?;
    let trmat_array = transition_matrix_from_rows(trmat)?;
    let data = SurvFitAJData {
        y: &y,
        sort1: &sort1,
        sort2: &sort2,
        utime: &utime,
        cstate: &cstate,
        wt: &wt,
        grp: &grp,
        position: &position,
    };
    let fit_params = SurvFitAJParams {
        ngrp,
        p0: &p0,
        i0: &i0,
        sefit,
        entry,
        hindx: &hindx_array,
        trmat: &trmat_array,
        t0,
    };
    let result = compute_survfitaj(&data, &fit_params).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("survfitaj failed: {}", e))
    })?;
    Ok(result.into_python_result())
}

fn matrix_from_rows(rows: Vec<Vec<usize>>, name: &'static str) -> PyResult<Array2<usize>> {
    let n_rows = rows.len();
    let n_cols = rows
        .first()
        .ok_or_else(|| {
            PyValueError::new_err(format!("Invalid {name} array: matrix cannot be empty"))
        })?
        .len();
    if n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "Invalid {name} array: matrix must have at least one column"
        )));
    }
    if let Some((row_idx, row)) = rows.iter().enumerate().find(|(_, row)| row.len() != n_cols) {
        return Err(PyValueError::new_err(format!(
            "Invalid {name} array: row {row_idx} has {} columns, expected {n_cols}",
            row.len()
        )));
    }

    Array2::from_shape_vec((n_rows, n_cols), rows.into_iter().flatten().collect())
        .map_err(|e| PyValueError::new_err(format!("Invalid {name} array: {e}")))
}

fn transition_matrix_from_rows(rows: Vec<Vec<usize>>) -> PyResult<Array2<usize>> {
    if rows.is_empty() {
        Ok(Array2::zeros((0, 2)))
    } else {
        matrix_from_rows(rows, "trmat")
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_survfitaj_inputs(
    y: &[f64],
    sort1: &[usize],
    sort2: &[usize],
    utime: &[f64],
    cstate: &[usize],
    wt: &[f64],
    grp: &[usize],
    ngrp: usize,
    p0: &[f64],
    i0: &[f64],
    sefit: i32,
    position: &[usize],
    hindx: &[Vec<usize>],
    entry: bool,
    trmat: &[Vec<usize>],
    t0: f64,
) -> PyResult<()> {
    if !y.len().is_multiple_of(3) {
        return Err(PyValueError::new_err(
            "y length must be a multiple of 3 (start, stop, status/state)",
        ));
    }
    let n_obs = y.len() / 3;
    if n_obs == 0 {
        return Err(PyValueError::new_err(
            "survfitaj requires at least one observation",
        ));
    }
    if sort1.is_empty() || sort2.is_empty() {
        return Err(PyValueError::new_err("sort1 and sort2 cannot be empty"));
    }
    if sort1.len() != sort2.len() {
        return Err(PyValueError::new_err(
            "sort1 and sort2 must have equal length",
        ));
    }
    if cstate.len() != n_obs || wt.len() != n_obs || grp.len() != n_obs || position.len() != n_obs {
        return Err(PyValueError::new_err(
            "cstate, wt, grp, and position must have length equal to y.len() / 3",
        ));
    }
    if p0.is_empty() {
        return Err(PyValueError::new_err(
            "p0 must contain at least one state probability",
        ));
    }
    if ngrp == 0 {
        return Err(PyValueError::new_err("ngrp must be positive"));
    }
    if sefit < 0 {
        return Err(PyValueError::new_err("sefit must be non-negative"));
    }
    let expected_i0_len = ngrp.checked_mul(p0.len()).ok_or_else(|| {
        PyValueError::new_err("ngrp * number of states exceeds addressable input size")
    })?;
    if sefit > 0 && i0.len() != expected_i0_len {
        return Err(PyValueError::new_err(
            "i0 length must equal ngrp * number of states when sefit > 0",
        ));
    }
    if utime.is_empty() {
        return Err(PyValueError::new_err("utime cannot be empty"));
    }
    if !t0.is_finite() {
        return Err(PyValueError::new_err("t0 must be finite"));
    }

    validate_no_nan(y, "y")?;
    validate_finite(y, "y")?;
    validate_no_nan(utime, "utime")?;
    validate_finite(utime, "utime")?;
    validate_no_nan(wt, "wt")?;
    validate_finite(wt, "wt")?;
    validate_no_nan(p0, "p0")?;
    validate_finite(p0, "p0")?;
    validate_non_negative(p0, "p0")?;
    validate_no_nan(i0, "i0")?;
    validate_finite(i0, "i0")?;

    if let Some((idx, weight)) = wt
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| *weight <= 0.0)
    {
        return Err(PyValueError::new_err(format!(
            "wt must contain only positive values; found {weight} at index {idx}"
        )));
    }
    if let Some((idx, times)) = utime
        .windows(2)
        .enumerate()
        .find(|(_, times)| times[0] >= times[1])
    {
        return Err(PyValueError::new_err(format!(
            "utime must be strictly increasing; indices {idx} and {} contain {} and {}",
            idx + 1,
            times[0],
            times[1]
        )));
    }
    if t0 > utime[0] {
        return Err(PyValueError::new_err(format!(
            "t0 must not be later than the first output time {}; got {t0}",
            utime[0]
        )));
    }
    let p0_sum: f64 = p0.iter().sum();
    if (p0_sum - 1.0).abs() > 1e-8 {
        return Err(PyValueError::new_err(format!(
            "p0 probabilities must sum to 1; got {p0_sum}"
        )));
    }

    let nstate = p0.len();
    if let Some((idx, state)) = cstate
        .iter()
        .copied()
        .enumerate()
        .find(|(_, state)| *state >= nstate)
    {
        return Err(PyValueError::new_err(format!(
            "cstate value {state} at index {idx} is out of range for {nstate} states"
        )));
    }
    if let Some((idx, group)) = grp
        .iter()
        .copied()
        .enumerate()
        .find(|(_, group)| *group >= ngrp)
    {
        return Err(PyValueError::new_err(format!(
            "grp value {group} at index {idx} is out of range for {ngrp} groups"
        )));
    }

    let sort1_members = validate_survfitaj_sort(sort1, "sort1", n_obs, y, 0)?;
    let sort2_members = validate_survfitaj_sort(sort2, "sort2", n_obs, y, 1)?;
    if sort1_members != sort2_members {
        return Err(PyValueError::new_err(
            "sort1 and sort2 must contain the same observation indices",
        ));
    }

    validate_usize_matrix_shape(hindx, "hindx")?;
    if hindx.len() != nstate || hindx[0].len() != nstate {
        return Err(PyValueError::new_err(format!(
            "Invalid hindx array: expected a {nstate} by {nstate} matrix"
        )));
    }
    if !trmat.is_empty() {
        validate_usize_matrix_shape(trmat, "trmat")?;
    }
    if !trmat.is_empty() && trmat[0].len() != 2 {
        return Err(PyValueError::new_err(
            "Invalid trmat array: matrix must have exactly 2 columns",
        ));
    }
    let nhaz = trmat.len();

    let mut transition_seen = vec![vec![false; nstate]; nstate];
    for (transition, row) in trmat.iter().enumerate() {
        let from = row[0];
        let to = row[1];
        if from >= nstate || to >= nstate {
            return Err(PyValueError::new_err(format!(
                "trmat transition {from}->{to} at row {transition} is out of range for {nstate} states"
            )));
        }
        if transition_seen[from][to] {
            return Err(PyValueError::new_err(format!(
                "trmat contains duplicate transition {from}->{to}"
            )));
        }
        transition_seen[from][to] = true;
        if hindx[from][to] != transition {
            return Err(PyValueError::new_err(format!(
                "hindx entry for transition {from}->{to} must be {transition}; got {}",
                hindx[from][to]
            )));
        }
    }

    // nhaz is the unsigned sentinel for an absent transition. This mirrors
    // the upstream native routine's -1 sentinel without exposing signed
    // hazard indices in the Python API.
    for (from, row) in hindx.iter().enumerate() {
        for (to, &transition) in row.iter().enumerate() {
            if transition > nhaz {
                return Err(PyValueError::new_err(format!(
                    "hindx hazard index {transition} at row {from}, column {to} exceeds the absent-transition sentinel {nhaz}"
                )));
            }
            if transition < nhaz && (trmat[transition][0] != from || trmat[transition][1] != to) {
                return Err(PyValueError::new_err(format!(
                    "hindx hazard index {transition} at row {from}, column {to} maps to a different trmat transition"
                )));
            }
        }
    }

    for obs in 0..n_obs {
        let start = y[obs * 3];
        let stop = y[obs * 3 + 1];
        let state = y[obs * 3 + 2];
        if start >= stop {
            return Err(PyValueError::new_err(format!(
                "y start time must be less than stop time at observation {obs}"
            )));
        }
        if state < 0.0 || state.fract() != 0.0 {
            return Err(PyValueError::new_err(format!(
                "y state/status value must be a non-negative integer at observation {obs}"
            )));
        }
        if sort1_members[obs] && state as usize > 0 {
            let to_state = state as usize - 1;
            if to_state >= nstate {
                return Err(PyValueError::new_err(format!(
                    "y state/status value {} at observation {} is out of range for {} states",
                    state, obs, nstate
                )));
            }
            let current_state = cstate[obs];
            let transition = hindx[current_state][to_state];
            if transition == nhaz {
                return Err(PyValueError::new_err(format!(
                    "hindx marks observed transition {current_state}->{to_state} as absent"
                )));
            }
        }
        if position[obs] > 3 {
            return Err(PyValueError::new_err(format!(
                "position value {} at observation {obs} is not a valid 0..3 bitmask",
                position[obs]
            )));
        }
    }

    for &obs in sort2 {
        let stop = observation_stop(y, obs);
        if stop >= utime[0]
            && (observation_target(y, obs).is_some() || position[obs] > 1)
            && utime
                .binary_search_by(|time| time.total_cmp(&stop))
                .is_err()
        {
            return Err(PyValueError::new_err(format!(
                "utime must contain stop time {stop} for observation {obs}"
            )));
        }
    }
    if entry {
        for &obs in sort1 {
            let start = observation_start(y, obs);
            if start >= utime[0]
                && position[obs] & 1 != 0
                && utime
                    .binary_search_by(|time| time.total_cmp(&start))
                    .is_err()
            {
                return Err(PyValueError::new_err(format!(
                    "utime must contain entry time {start} for observation {obs} when entry is enabled"
                )));
            }
        }
    }

    Ok(())
}

fn validate_survfitaj_sort(
    sort: &[usize],
    name: &'static str,
    n_obs: usize,
    y: &[f64],
    time_column: usize,
) -> PyResult<Vec<bool>> {
    let mut members = vec![false; n_obs];
    for (position, &obs) in sort.iter().enumerate() {
        if obs >= n_obs {
            return Err(PyValueError::new_err(format!(
                "sort index {obs} at {name} position {position} is out of range for {n_obs} observations"
            )));
        }
        if members[obs] {
            return Err(PyValueError::new_err(format!(
                "{name} contains duplicate observation index {obs}"
            )));
        }
        members[obs] = true;
    }
    if let Some(position) = sort
        .windows(2)
        .position(|pair| y[pair[0] * 3 + time_column] > y[pair[1] * 3 + time_column])
    {
        return Err(PyValueError::new_err(format!(
            "{name} must order observations by ascending {} time; positions {position} and {} are out of order",
            if time_column == 0 { "start" } else { "stop" },
            position + 1
        )));
    }
    Ok(members)
}

fn validate_usize_matrix_shape(matrix: &[Vec<usize>], name: &'static str) -> PyResult<()> {
    let n_cols = matrix
        .first()
        .ok_or_else(|| {
            PyValueError::new_err(format!("Invalid {name} array: matrix cannot be empty"))
        })?
        .len();
    if n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "Invalid {name} array: matrix must have at least one column"
        )));
    }
    if let Some((row_idx, row)) = matrix
        .iter()
        .enumerate()
        .find(|(_, row)| row.len() != n_cols)
    {
        return Err(PyValueError::new_err(format!(
            "Invalid {name} array: row {row_idx} has {} columns, expected {n_cols}",
            row.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const R_TOLERANCE: f64 = 1e-12;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= R_TOLERANCE,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_matrix_close(actual: &[Vec<f64>], expected: &[&[f64]]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual_value, &expected_value) in actual_row.iter().zip(*expected_row) {
                assert_close(actual_value, expected_value);
            }
        }
    }

    fn two_state_hazard_index() -> Vec<Vec<usize>> {
        // One transition means 1 is the unsigned absent-transition sentinel.
        vec![vec![1, 0], vec![1, 1]]
    }

    fn minimal_survfitaj(p0: Vec<f64>) -> PyResult<SurvFitAJ> {
        survfitaj(
            vec![0.0, 1.0, 2.0, 0.0, 2.0, 0.0],
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            vec![0, 0],
            vec![1.0, 1.0],
            vec![0, 1],
            2,
            p0,
            vec![0.0; 4],
            0,
            false,
            vec![3, 3],
            two_state_hazard_index(),
            vec![vec![0, 1]],
            0.0,
        )
    }

    #[test]
    fn survfitaj_rejects_non_normalized_initial_state_distribution() {
        let err = minimal_survfitaj(vec![0.6, 0.6]).expect_err("non-normalized p0 should fail");

        assert!(err.to_string().contains("p0 probabilities must sum to 1"));
    }

    #[test]
    fn survfitaj_accepts_normalized_initial_state_distribution() {
        let result = minimal_survfitaj(vec![1.0, 0.0]).unwrap();

        assert_eq!(result.pstate.len(), 2);
        assert_eq!(result.pstate[0].len(), 2);
    }

    #[test]
    fn survfitaj_matches_r_for_index_zero_and_censor_only_uncertainty() {
        // Reference: R survival 3.8-6, native Csurvfitaj.
        let result = survfitaj(
            vec![0.0, 1.0, 2.0, 0.0, 2.0, 0.0],
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            vec![0, 0],
            vec![1.0, 1.0],
            vec![0, 1],
            2,
            vec![1.0, 0.0],
            vec![0.0; 4],
            3,
            false,
            vec![3, 3],
            two_state_hazard_index(),
            vec![vec![0, 1]],
            0.0,
        )
        .unwrap();

        assert_matrix_close(
            &result.n_risk,
            &[&[2.0, 0.0, 2.0, 0.0], &[1.0, 0.0, 1.0, 0.0]],
        );
        assert_matrix_close(&result.n_event, &[&[0.0, 1.0], &[0.0, 0.0]]);
        assert_matrix_close(
            &result.n_censor,
            &[&[0.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 1.0, 0.0]],
        );
        assert_matrix_close(&result.pstate, &[&[0.5, 0.5], &[0.5, 0.5]]);
        assert_matrix_close(&result.cumhaz, &[&[0.5], &[0.5]]);
        assert_matrix_close(&result.n_transition, &[&[1.0, 1.0], &[0.0, 0.0]]);

        let standard_error = 0.353_553_390_593_273_8;
        assert_matrix_close(
            result.std_err.as_deref().unwrap(),
            &[
                &[standard_error, standard_error],
                &[standard_error, standard_error],
            ],
        );
        assert_matrix_close(
            result.std_chaz.as_deref().unwrap(),
            &[&[standard_error], &[standard_error]],
        );
        assert_matrix_close(
            result.std_auc.as_deref().unwrap(),
            &[&[0.0, 0.0], &[standard_error, standard_error]],
        );
        assert_matrix_close(
            result.influence.as_deref().unwrap(),
            &[
                &[-0.25, -0.25],
                &[0.25, 0.25],
                &[0.25, 0.25],
                &[-0.25, -0.25],
            ],
        );

        let std_err = result.std_err.as_deref().unwrap();
        let influence = result.influence.as_deref().unwrap();
        for time_idx in 0..result.pstate.len() {
            for state in 0..2 {
                let variance: f64 = (0..2)
                    .map(|group| influence[group + state * 2][time_idx].powi(2))
                    .sum();
                assert_close(std_err[time_idx][state].powi(2), variance);
            }
        }
    }

    #[test]
    fn survfitaj_freezes_state_probabilities_across_competing_ties() {
        let result = survfitaj(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            vec![1.0, 2.0],
            vec![0; 4],
            vec![1.0; 4],
            vec![0, 1, 2, 3],
            4,
            vec![1.0, 0.0, 0.0],
            vec![0.0; 12],
            0,
            false,
            vec![3; 4],
            vec![vec![2, 0, 1], vec![2; 3], vec![2; 3]],
            vec![vec![0, 1], vec![0, 2]],
            0.0,
        )
        .unwrap();

        assert_matrix_close(
            &result.n_risk,
            &[
                &[4.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                &[2.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            ],
        );
        assert_matrix_close(&result.n_event, &[&[0.0, 1.0, 1.0], &[0.0; 3]]);
        assert_matrix_close(&result.pstate, &[&[0.5, 0.25, 0.25], &[0.5, 0.25, 0.25]]);
        assert_matrix_close(&result.cumhaz, &[&[0.25, 0.25], &[0.25, 0.25]]);
        assert_matrix_close(&result.n_transition, &[&[1.0, 1.0, 1.0, 1.0], &[0.0; 4]]);
    }

    #[test]
    fn survfitaj_does_not_move_new_probability_mass_twice_at_one_time() {
        let result = survfitaj(
            vec![0.0, 1.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0],
            vec![0, 1, 2, 3],
            vec![0, 2, 1, 3],
            vec![1.0, 2.0],
            vec![0, 0, 1, 1],
            vec![1.0; 4],
            vec![0, 1, 2, 3],
            4,
            vec![0.5, 0.5, 0.0],
            vec![0.0; 12],
            0,
            false,
            vec![3; 4],
            vec![vec![2, 0, 2], vec![2, 2, 1], vec![2; 3]],
            vec![vec![0, 1], vec![1, 2]],
            0.0,
        )
        .unwrap();

        assert_matrix_close(&result.pstate, &[&[0.25, 0.5, 0.25], &[0.25, 0.5, 0.25]]);
        for row in &result.pstate {
            assert_close(row.iter().sum(), 1.0);
        }
    }

    #[test]
    fn survfitaj_ignores_selected_boundaries_before_first_output() {
        let result = survfitaj(
            vec![-2.0, -1.0, 2.0, -1.0, 1.0, 2.0, -2.0, 2.0, 0.0],
            vec![0, 2, 1],
            vec![0, 1, 2],
            vec![1.0, 2.0],
            vec![0, 0, 0],
            vec![1.0; 3],
            vec![0, 1, 2],
            3,
            vec![1.0, 0.0],
            vec![0.0; 6],
            0,
            true,
            vec![3; 3],
            two_state_hazard_index(),
            vec![vec![0, 1]],
            0.0,
        )
        .unwrap();

        assert_matrix_close(
            &result.n_risk,
            &[&[2.0, 0.0, 2.0, 0.0], &[1.0, 0.0, 1.0, 0.0]],
        );
        assert_matrix_close(&result.n_event, &[&[0.0, 1.0], &[0.0, 0.0]]);
        assert_matrix_close(&result.pstate, &[&[0.5, 0.5], &[0.5, 0.5]]);
        assert_matrix_close(result.n_enter.as_deref().unwrap(), &[&[0.0; 4], &[0.0; 4]]);
    }

    #[test]
    fn survfitaj_ignores_transition_metadata_for_excluded_rows() {
        let result = survfitaj(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 1.0],
            vec![0],
            vec![0],
            vec![1.0],
            vec![0, 1],
            vec![1.0, 1.0],
            vec![0, 0],
            1,
            vec![1.0, 0.0],
            Vec::new(),
            0,
            false,
            vec![3, 3],
            two_state_hazard_index(),
            vec![vec![0, 1]],
            0.0,
        )
        .unwrap();

        assert_matrix_close(&result.pstate, &[&[0.0, 1.0]]);
        assert_matrix_close(&result.n_transition, &[&[1.0, 1.0]]);
    }

    #[test]
    fn survfitaj_matches_r_for_entry_counts_and_negative_times() {
        let result = survfitaj(
            vec![-2.0, 1.0, 2.0, -1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            vec![0; 3],
            vec![1.0; 3],
            vec![0, 1, 2],
            3,
            vec![1.0, 0.0],
            vec![0.0; 6],
            3,
            true,
            vec![3; 3],
            two_state_hazard_index(),
            vec![vec![0, 1]],
            -2.0,
        )
        .unwrap();

        assert_matrix_close(
            &result.n_risk,
            &[
                &[0.0, 0.0, 0.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0],
                &[2.0, 0.0, 2.0, 0.0],
                &[3.0, 0.0, 3.0, 0.0],
                &[2.0, 0.0, 2.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0],
            ],
        );
        assert_matrix_close(
            result.n_enter.as_deref().unwrap(),
            &[
                &[1.0, 0.0, 1.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0],
                &[1.0, 0.0, 1.0, 0.0],
                &[0.0; 4],
                &[0.0; 4],
                &[0.0; 4],
            ],
        );
        assert_matrix_close(
            &result.pstate,
            &[
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[1.0, 0.0],
                &[2.0 / 3.0, 1.0 / 3.0],
                &[2.0 / 3.0, 1.0 / 3.0],
                &[2.0 / 3.0, 1.0 / 3.0],
            ],
        );
        let standard_error = 0.272_165_526_975_908_7;
        assert_matrix_close(
            result.std_auc.as_deref().unwrap(),
            &[
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0],
                &[standard_error, standard_error],
                &[2.0 * standard_error, 2.0 * standard_error],
            ],
        );
    }

    #[test]
    fn survfitaj_supports_zero_hazards_and_column_major_initial_influence() {
        let result = survfitaj(
            vec![-2.0, -1.0, 0.0, -2.0, 1.0, 0.0],
            vec![0, 1],
            vec![0, 1],
            vec![-1.0, 1.0],
            vec![0, 1],
            vec![1.0, 1.0],
            vec![0, 1],
            2,
            vec![0.6, 0.4],
            vec![0.1, -0.1, -0.1, 0.1],
            3,
            false,
            vec![3, 3],
            vec![vec![0; 2], vec![0; 2]],
            vec![],
            -2.0,
        )
        .unwrap();

        assert_matrix_close(&result.pstate, &[&[0.6, 0.4], &[0.6, 0.4]]);
        assert_matrix_close(&result.cumhaz, &[&[], &[]]);
        assert_matrix_close(&result.n_transition, &[&[], &[]]);
        let initial_se = 0.141_421_356_237_309_53;
        assert_matrix_close(
            result.std_err.as_deref().unwrap(),
            &[&[initial_se, initial_se], &[initial_se, initial_se]],
        );
        assert_matrix_close(result.std_chaz.as_deref().unwrap(), &[&[], &[]]);
        assert_matrix_close(
            result.std_auc.as_deref().unwrap(),
            &[
                &[initial_se, initial_se],
                &[3.0 * initial_se, 3.0 * initial_se],
            ],
        );
        assert_matrix_close(
            result.influence.as_deref().unwrap(),
            &[&[0.1, 0.1], &[-0.1, -0.1], &[-0.1, -0.1], &[0.1, 0.1]],
        );
    }

    #[test]
    fn survfitaj_rejects_inconsistent_transition_and_sort_metadata() {
        let base = || {
            (
                vec![0.0, 1.0, 2.0, 0.0, 2.0, 0.0],
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 2.0],
                vec![0, 0],
                vec![1.0, 1.0],
                vec![0, 1],
                2,
                vec![1.0, 0.0],
                vec![0.0; 4],
                0,
                false,
                vec![3, 3],
                two_state_hazard_index(),
                vec![vec![0, 1]],
                0.0,
            )
        };

        let mut args = base();
        args.0[3] = -1.0;
        let err = survfitaj(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15,
        )
        .unwrap_err();
        assert!(err.to_string().contains("sort1 must order observations"));

        let mut args = base();
        args.13[0][1] = 1;
        let err = survfitaj(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11, args.12, args.13, args.14, args.15,
        )
        .unwrap_err();
        assert!(err.to_string().contains("must be 0"));
    }
}
