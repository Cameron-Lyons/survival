//! Probability-in-state curves from a multi-state Cox model: the kernel of
//! R's `survfit.coxphms` (`R/survfit.coxphms.R`), its `multihaz`, and the
//! risk-set counts of `coxsurv1` and `coxsurv2` (`src/coxsurv1.c`,
//! `src/coxsurv2.c`).
//!
//! R recentres X at each newdata row, restacks the data and recounts every
//! risk set.  A newdata row only multiplies the risk scores of transition
//! `k` by `exp(-c[k])`, with `c[k] = (x_new - means) beta_k + offset_new`,
//! so the counts are made once per stratum, transition and time, with the
//! risk scores centred at `means`, and scaled per newdata row.

use ndarray::{Array2, Array3, ArrayView2};
use rayon::prelude::*;

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::expm::survexpm;
use crate::internal::sorting::ordered_subset;
use crate::internal::validation::validate_length;
use crate::regression::coxphms::MsDesign;
#[cfg(feature = "python")]
use {
    super::survfit_confint::ConfType,
    super::survfitaj::{SurvfitAJData, SurvfitAJOptions, SurvfitAJResult, survfitaj},
    crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec},
    crate::regression::coxphms::{ms_design, to_index},
    numpy::PyArray3,
    pyo3::prelude::*,
};

/// The unstacked data of a multi-state fit and its coefficients.
pub(crate) struct CoxmsCurveData<'a> {
    pub start: Option<&'a [f64]>,
    pub time: &'a [f64],
    /// 1-based state each row moves to, 0 when censored.
    pub endpoint: &'a [usize],
    /// 1-based current state of each row (survcheck2's).
    pub istate: &'a [usize],
    pub weights: &'a [f64],
    pub offset: &'a [f64],
    /// The curve (user stratum) of each row; `None` for a row in none.
    pub strata: &'a [Option<usize>],
    pub nstrata: usize,
    /// `n x nx`, NaN for a missing value.
    pub x: ArrayView2<'a, f64>,
    pub design: &'a MsDesign,
    /// Codes of each `strata()` term, -1 for a missing value.
    pub strata_term_codes: &'a [Vec<i32>],
    pub beta: &'a [f64],
    pub means: &'a [f64],
    /// `fit$share$scale`: the hazard multiplier of each transition.
    pub share_scale: Option<&'a [f64]>,
}

/// `pstate` (`m x ntime x nstate`) and `cumhaz` (`m x ntime x ntrans`),
/// newdata row outermost; the strata's times are stacked.
pub(crate) struct CoxmsCurves {
    pub pstate: Array3<f64>,
    pub cumhaz: Array3<f64>,
}

/// The counts `multihaz` reads for one transition within one stratum, per
/// output time: `n[2]` (the weighted risk-score sum of the risk set),
/// `n[4]` (the weighted events) and `n[8]` (the Efron mean risk set).
/// Risk scores are `exp(eta0)`, centred at the fit's means.
struct TransitionCounts {
    at_risk: Vec<f64>,
    events: Vec<f64>,
    efron: Vec<f64>,
}

/// The Efron mean risk set of coxsurv1/2: `n2` for one event, else the mean
/// of `n2 - k n5 / n3^2` over `k = 0 .. n3 - 1`, the sum started from
/// `carried`.  coxsurv2 starts it from 0; coxsurv1 does not reset `n[8]`
/// between output times and starts it from the value of the previous
/// (later) time, which R's right-censored curves inherit.
fn efron_mean(carried: f64, n2: f64, n3: usize, n5: f64) -> f64 {
    if n3 <= 1 {
        return n2;
    }
    let count = n3 as f64;
    let mean_weight = n5 / (count * count);
    let total = (0..n3).fold(carried, |total, k| total + (n2 - k as f64 * mean_weight));
    total / count
}

/// coxsurv1/coxsurv2 for one transition: walk the rows backwards from the
/// last output time, adding a row to the risk set once `stop >= t` (and,
/// for counting data, `start < t`) and removing it once `start >= t`.
/// `rows` are the data rows at risk for the transition, `risk` their
/// `exp(eta0)`, and an event is an endpoint of `to` at `stop == t`.
fn transition_counts(
    d: &CoxmsCurveData<'_>,
    rows: &[usize],
    risk: &[f64],
    to: usize,
    utime: &[f64],
) -> TransitionCounts {
    let ntime = utime.len();
    let mut counts = TransitionCounts {
        at_risk: vec![0.0; ntime],
        events: vec![0.0; ntime],
        efron: vec![0.0; ntime],
    };
    let positions: Vec<usize> = (0..rows.len()).collect();
    let stop_of: Vec<f64> = rows.iter().map(|&i| d.time[i]).collect();
    let sort2 = ordered_subset(&positions, &stop_of, false);
    let sort1 = d.start.map(|start| {
        let start_of: Vec<f64> = rows.iter().map(|&i| start[i]).collect();
        ordered_subset(&positions, &start_of, false)
    });
    let mut in_risk_set = vec![false; rows.len()];
    let (mut person2, mut person1) = (rows.len(), rows.len());
    let (mut n0, mut n2) = (0usize, 0.0);
    for t in (0..ntime).rev() {
        let dtime = utime[t];
        let (mut n3, mut n4, mut n5) = (0usize, 0.0, 0.0);
        while person2 > 0 {
            let p = sort2[person2 - 1];
            let i = rows[p];
            if d.time[i] < dtime {
                break;
            }
            let weighted = d.weights[i] * risk[p];
            if d.start.is_none_or(|start| start[i] < dtime) {
                in_risk_set[p] = true;
                n0 += 1;
                n2 += weighted;
            }
            if d.time[i] == dtime && d.endpoint[i] == to {
                n3 += 1;
                n4 += d.weights[i];
                n5 += weighted;
            }
            person2 -= 1;
        }
        if let (Some(start), Some(sort1)) = (d.start, &sort1) {
            while person1 > 0 {
                let p = sort1[person1 - 1];
                let i = rows[p];
                if start[i] < dtime {
                    break;
                }
                if in_risk_set[p] {
                    n0 -= 1;
                    n2 = if n0 == 0 {
                        0.0
                    } else {
                        n2 - d.weights[i] * risk[p]
                    };
                }
                person1 -= 1;
            }
        }
        counts.at_risk[t] = n2;
        counts.events[t] = n4;
        let carried = match (d.start, counts.efron.get(t + 1)) {
            (None, Some(&later)) => later,
            _ => 0.0,
        };
        counts.efron[t] = efron_mean(carried, n2, n3, n5);
    }
    counts
}

/// The curves of `survfit.coxphms` for every row of `newx` (`m x nx`, with
/// `newoffset`), on the time grid `utime` of each stratum (survfitAJ's),
/// starting from each stratum's `p0`.
///
/// Transition `k` uses the rows whose current state is its `from` state,
/// without a missing value in a covariate it uses or in a strata term its
/// block uses (R's `stacker(dropzero = FALSE)`), all counted within each
/// user stratum.  `stype` 1 updates `p` by `p (I + A)`, 2 by `p expm(A)`;
/// `ctype` 2 uses the Efron risk set.  Shared baselines pool the events and
/// the scaled risk sets of their transitions.
pub(crate) fn coxms_curves(
    d: &CoxmsCurveData<'_>,
    utime: &[Vec<f64>],
    p0: &[Vec<f64>],
    newx: ArrayView2<'_, f64>,
    newoffset: &[f64],
    stype: u8,
    ctype: u8,
) -> SurvivalResult<CoxmsCurves> {
    let design = d.design;
    let n = d.time.len();
    let (nx, ntrans) = (design.nx, design.ntrans());
    for (len, name) in [
        (d.endpoint.len(), "endpoint"),
        (d.istate.len(), "istate"),
        (d.weights.len(), "weights"),
        (d.offset.len(), "offset"),
        (d.strata.len(), "strata"),
        (d.x.nrows(), "x"),
    ] {
        validate_length(n, len, name)?;
    }
    if let Some(start) = d.start {
        validate_length(n, start.len(), "start")?;
    }
    for codes in d.strata_term_codes {
        validate_length(n, codes.len(), "strata terms")?;
    }
    validate_length(nx, d.x.ncols(), "x columns")?;
    validate_length(nx, d.means.len(), "means")?;
    validate_length(nx, newx.ncols(), "newx columns")?;
    validate_length(newx.nrows(), newoffset.len(), "newoffset")?;
    validate_length(d.nstrata, utime.len(), "utime")?;
    validate_length(d.nstrata, p0.len(), "p0")?;
    if d.beta.len() < design.ncoef() {
        return Err(SurvivalError::invalid_input(
            "beta needs a value per coefficient of cmap",
        ));
    }
    if let Some(scale) = d.share_scale {
        validate_length(ntrans, scale.len(), "share_scale")?;
    }
    if !(1..=2).contains(&stype) {
        return Err(SurvivalError::invalid_input("stype must be 1 or 2"));
    }
    if !(1..=2).contains(&ctype) {
        return Err(SurvivalError::invalid_input("ctype must be 1 or 2"));
    }
    let nstate = p0.first().map_or(0, Vec::len);
    if p0.iter().any(|row| row.len() != nstate)
        || design
            .from
            .iter()
            .chain(&design.to)
            .chain(d.istate)
            .any(|&state| state == 0 || state > nstate)
    {
        return Err(SurvivalError::invalid_input(
            "every state must be a column of p0",
        ));
    }
    if d.strata.iter().flatten().any(|&s| s >= d.nstrata) {
        return Err(SurvivalError::invalid_input("strata code out of range"));
    }

    // each transition's X coefficients, and its constant ph() term
    let coefficient = |c: usize, k: usize| {
        let index = design.cmap[(c, k)];
        (index > 0).then(|| d.beta[index as usize - 1])
    };
    let x_terms: Vec<Vec<(usize, f64)>> = (0..ntrans)
        .map(|k| {
            (0..nx)
                .filter_map(|c| coefficient(c, k).map(|b| (c, b)))
                .collect()
        })
        .collect();
    let ph_terms: Vec<f64> = (0..ntrans)
        .map(|k| {
            (nx..design.cmap.nrows())
                .filter_map(|c| coefficient(c, k))
                .sum()
        })
        .collect();
    // the strata terms of each transition's block: those of its first
    // transition (the stacker's rule)
    let block_terms: Vec<Vec<usize>> = (0..ntrans)
        .map(|k| {
            let first = (0..ntrans)
                .find(|&j| design.baseline[j] == design.baseline[k])
                .expect("k itself shares its baseline");
            (0..design.strata_use.nrows())
                .filter(|&t| design.strata_use[(t, first)])
                .collect()
        })
        .collect();

    // the counts per stratum and transition
    let mut rows_by: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); nstate + 1]; d.nstrata];
    for i in 0..n {
        if let Some(s) = d.strata[i] {
            rows_by[s][d.istate[i]].push(i);
        }
    }
    let counts: Vec<Vec<TransitionCounts>> = (0..d.nstrata)
        .into_par_iter()
        .map(|s| {
            (0..ntrans)
                .map(|k| {
                    let rows: Vec<usize> = rows_by[s][design.from[k]]
                        .iter()
                        .copied()
                        .filter(|&i| {
                            x_terms[k].iter().all(|&(c, _)| !d.x[(i, c)].is_nan())
                                && block_terms[k]
                                    .iter()
                                    .all(|&t| d.strata_term_codes[t][i] >= 0)
                        })
                        .collect();
                    let risk: Vec<f64> = rows
                        .iter()
                        .map(|&i| {
                            let eta: f64 = x_terms[k]
                                .iter()
                                .map(|&(c, b)| (d.x[(i, c)] - d.means[c]) * b)
                                .sum();
                            (eta + ph_terms[k] + d.offset[i]).exp()
                        })
                        .collect();
                    transition_counts(d, &rows, &risk, design.to[k], &utime[s])
                })
                .collect()
        })
        .collect();

    // shared baselines: R's `sharemat`, set when a baseline repeats
    let mut groups: Vec<(i32, Vec<usize>)> = Vec::new();
    for (k, &b) in design.baseline.iter().enumerate() {
        match groups.iter_mut().find(|(baseline, _)| *baseline == b) {
            Some((_, members)) => members.push(k),
            None => groups.push((b, vec![k])),
        }
    }
    let shared = groups.len() < ntrans;
    let scale = |k: usize| d.share_scale.map_or(1.0, |scale| scale[k]);

    let m = newx.nrows();
    let ntime: usize = utime.iter().map(Vec::len).sum();
    let mut pstate = Array3::<f64>::zeros((m, ntime, nstate));
    let mut cumhaz = Array3::<f64>::zeros((m, ntime, ntrans));
    let pstate_rows = pstate
        .as_slice_mut()
        .expect("a new array is contiguous")
        .par_chunks_mut((ntime * nstate).max(1));
    let cumhaz_rows = cumhaz
        .as_slice_mut()
        .expect("a new array is contiguous")
        .par_chunks_mut((ntime * ntrans).max(1));
    pstate_rows.zip(cumhaz_rows).enumerate().try_for_each(
        |(j, (pstate_j, cumhaz_j))| -> SurvivalResult<()> {
            // exp(-c[k]): the factor this row puts on transition k's risks
            let factor: Vec<f64> = (0..ntrans)
                .map(|k| {
                    let c: f64 = x_terms[k]
                        .iter()
                        .map(|&(col, b)| (newx[(j, col)] - d.means[col]) * b)
                        .sum();
                    (-(c + newoffset[j])).exp()
                })
                .collect();
            let mut hazard = vec![0.0; ntrans];
            let mut a = Array2::<f64>::zeros((nstate, nstate));
            let mut row = 0;
            for (s, times) in utime.iter().enumerate() {
                let mut p = p0[s].clone();
                let mut total = vec![0.0; ntrans];
                for t in 0..times.len() {
                    let risk_set = |k: usize| {
                        let counts = &counts[s][k];
                        let n = if ctype == 1 {
                            counts.at_risk[t]
                        } else {
                            counts.efron[t]
                        };
                        n * factor[k]
                    };
                    if shared {
                        for (_, members) in &groups {
                            let events: f64 = members.iter().map(|&k| counts[s][k].events[t]).sum();
                            let denominator: f64 =
                                members.iter().map(|&k| scale(k) * risk_set(k)).sum();
                            let pooled = if events == 0.0 {
                                0.0
                            } else {
                                events / denominator
                            };
                            for &k in members {
                                hazard[k] = pooled * scale(k);
                            }
                        }
                    } else {
                        for (k, h) in hazard.iter_mut().enumerate() {
                            let events = counts[s][k].events[t];
                            *h = if events == 0.0 {
                                0.0
                            } else {
                                events / risk_set(k)
                            };
                        }
                    }
                    for (k, &h) in hazard.iter().enumerate() {
                        total[k] += h;
                        cumhaz_j[row * ntrans + k] = total[k];
                    }
                    if hazard.iter().any(|&h| h != 0.0) {
                        a.fill(0.0);
                        for (k, &h) in hazard.iter().enumerate() {
                            a[(design.from[k] - 1, design.to[k] - 1)] = h;
                        }
                        for i in 0..nstate {
                            let departures: f64 = a.row(i).sum();
                            a[(i, i)] = -departures;
                        }
                        if stype == 2 {
                            a = survexpm(&a)?;
                        } else {
                            for i in 0..nstate {
                                a[(i, i)] += 1.0;
                            }
                        }
                        p = (0..nstate)
                            .map(|col| (0..nstate).map(|r| p[r] * a[(r, col)]).sum())
                            .collect();
                    }
                    pstate_j[row * nstate..(row + 1) * nstate].copy_from_slice(&p);
                    row += 1;
                }
            }
            Ok(())
        },
    )?;
    Ok(CoxmsCurves { pstate, cumhaz })
}

/// `survfit(<coxphms>, newdata)`: survfitAJ's time grid, counts and `p0`,
/// then the curves of every `newx` row.
///
/// The per-row vectors are the unstacked data of the fit at the rows the
/// curves use: `endpoint` (1-based state of `states` reached, 0 when
/// censored), `istate` (survcheck2's 1-based current state), `id` (subject
/// codes), `strata` (0-based user strata codes) and `strata_terms`.  The
/// design (`cmap`, `baseline`, `trans_from`, `trans_to`, `strata_use`) is as
/// for `coxphms_fit`.  `start_time`, `p0` and `time0` go to survfitAJ, which
/// runs without standard errors or timefix.
///
/// Returns survfitAJ's result, then `pstate` and `cumhaz` as
/// `(ntime, m, nstate)` and `(ntime, m, ntrans)` arrays.
#[cfg(feature = "python")]
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (time, endpoint, istate, x, cmap, cmap_nrow, baseline, trans_from, trans_to, id, states, beta, means, newx, stype, ctype, entry=None, weights=None, offset=None, strata=None, strata_terms=None, strata_use=None, share_scale=None, newoffset=None, start_time=None, p0=None, time0=false))]
pub fn coxphms_curves<'py>(
    py: Python<'py>,
    time: FloatVec,
    endpoint: IntVec,
    istate: IntVec,
    x: FloatMatrix,
    cmap: IntVec,
    cmap_nrow: usize,
    baseline: IntVec,
    trans_from: IntVec,
    trans_to: IntVec,
    id: IntVec,
    states: Vec<String>,
    beta: FloatVec,
    means: FloatVec,
    newx: FloatMatrix,
    stype: u8,
    ctype: u8,
    entry: Option<FloatVec>,
    weights: Option<FloatVec>,
    offset: Option<FloatVec>,
    strata: Option<IntVec>,
    strata_terms: Option<Vec<IntVec>>,
    strata_use: Option<IntVec>,
    share_scale: Option<FloatVec>,
    newoffset: Option<FloatVec>,
    start_time: Option<f64>,
    p0: Option<Vec<f64>>,
    time0: bool,
) -> PyResult<(
    SurvfitAJResult,
    Bound<'py, PyArray3<f64>>,
    Bound<'py, PyArray3<f64>>,
)> {
    let strata_terms: Vec<Vec<i32>> = strata_terms
        .unwrap_or_default()
        .into_iter()
        .map(IntVec::into_inner)
        .collect();
    let design = ms_design(
        x.ncol(),
        cmap,
        cmap_nrow,
        baseline,
        trans_from,
        trans_to,
        strata_use,
        strata_terms.len(),
    )?;
    let endpoint = to_index(endpoint, "endpoint")?;
    let istate = to_index(istate, "istate")?;
    let n = time.len();
    if let Some(&bad) = istate.iter().find(|&&s| s == 0 || s > states.len()) {
        return Err(SurvivalError::invalid_input(format!(
            "istate code {bad} does not name one of the states"
        ))
        .into());
    }
    let x = x.into_inner();
    let newx = newx.into_inner();
    let newoffset = newoffset.map_or_else(|| vec![0.0; newx.nrows()], |v| v.into_inner());
    let weights = weights.map_or_else(|| vec![1.0; n], |v| v.into_inner());
    let offset = offset.map_or_else(|| vec![0.0; n], |v| v.into_inner());
    let strata = strata.map(IntVec::into_inner);
    let data = SurvfitAJData::try_new(
        entry.as_ref().map(|entry| entry.to_vec()),
        time.to_vec(),
        endpoint.iter().map(|&state| state as i32).collect(),
        states.clone(),
        Some(weights.clone()),
        strata.clone(),
        Some(id.iter().map(|&code| i64::from(code)).collect()),
        Some(istate.iter().map(|&s| states[s - 1].clone()).collect()),
        Some(states.clone()),
        None,
    )?;
    let options = SurvfitAJOptions {
        se_fit: false,
        conf_type: ConfType::None,
        start_time,
        p0,
        time0,
        timefix: false,
        ..SurvfitAJOptions::default()
    };
    let beta = beta.into_inner();
    let means = means.into_inner();
    let share_scale = share_scale.map(|v| v.into_inner());
    let (engine, curves) = py.detach(|| -> SurvivalResult<_> {
        let engine = survfitaj(&data, &options)?;
        // each row's curve: the position of its code among the fitted strata
        let curve_of: Vec<Option<usize>> = match (&strata, &engine.strata_codes) {
            (Some(codes), Some(fitted)) => codes
                .iter()
                .map(|code| fitted.iter().position(|f| f == code))
                .collect(),
            _ => vec![Some(0); n],
        };
        let ranges = engine.curve_ranges();
        let utime: Vec<Vec<f64>> = ranges
            .iter()
            .map(|range| engine.time[range.clone()].to_vec())
            .collect();
        let curve_data = CoxmsCurveData {
            start: entry.as_deref(),
            time: &time,
            endpoint: &endpoint,
            istate: &istate,
            weights: &weights,
            offset: &offset,
            strata: &curve_of,
            nstrata: ranges.len(),
            x: x.view(),
            design: &design,
            strata_term_codes: &strata_terms,
            beta: &beta,
            means: &means,
            share_scale: share_scale.as_deref(),
        };
        let curves = coxms_curves(
            &curve_data,
            &utime,
            &engine.p0,
            newx.view(),
            &newoffset,
            stype,
            ctype,
        )?;
        Ok((engine, curves))
    })?;
    let as_numpy =
        |array: Array3<f64>| PyArray3::from_owned_array(py, array.permuted_axes([1, 0, 2]));
    Ok((engine, as_numpy(curves.pstate), as_numpy(curves.cumhaz)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-12 * e.abs().max(1.0),
                "{actual:?} != {expected:?}"
            );
        }
    }

    /// Competing risks a and b with tied events of a at time 2:
    /// `coxph(Surv(time, status) ~ x, id = id, init = c(0.5, -0.3), iter = 0,
    /// ties = "efron")` and its curves for x = 0 and 1 with the Efron risk
    /// sets (R 4.5.3 / survival 3.8-12).  At time 2 coxsurv1 starts the Efron
    /// sum from the value of time 3, and R's hazards carry that.
    #[test]
    fn efron_curves_match_r() {
        let time = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        let endpoint = [2, 2, 2, 3, 2, 0, 3, 2];
        let x = Array2::from_shape_vec((8, 1), vec![0., 1., 0., 1., 0., 1., 1., 0.]).unwrap();
        let design = MsDesign {
            cmap: Array2::from_shape_vec((1, 2), vec![1, 2]).unwrap(),
            nx: 1,
            baseline: vec![1, 2],
            strata_use: Array2::from_elem((0, 2), false),
            from: vec![1, 1],
            to: vec![2, 3],
        };
        let data = CoxmsCurveData {
            start: None,
            time: &time,
            endpoint: &endpoint,
            istate: &[1; 8],
            weights: &[1.0; 8],
            offset: &[0.0; 8],
            strata: &[Some(0); 8],
            nstrata: 1,
            x: x.view(),
            design: &design,
            strata_term_codes: &[],
            beta: &[0.5, -0.3],
            means: &[0.5],
            share_scale: None,
        };
        let newx = Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let curves = coxms_curves(
            &data,
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            &[vec![1.0, 0.0, 0.0]],
            newx.view(),
            &[0.0, 0.0],
            2,
            2,
        )
        .unwrap();
        let cumhaz = |j: usize, k: usize| curves.cumhaz.slice(ndarray::s![j, .., k]).to_vec();
        assert_close(
            &cumhaz(0, 0),
            &[
                0.0943851671995364,
                0.262275809891465,
                0.451046144290538,
                0.451046144290538,
                1.45104614429054,
            ],
        );
        assert_close(
            &cumhaz(1, 1),
            &[
                0.0,
                0.124230139262545,
                0.124230139262545,
                0.549787622450886,
                0.549787622450886,
            ],
        );
        assert_close(
            &curves.pstate.slice(ndarray::s![0, 4, ..]).to_vec(),
            &[0.111561216844069, 0.52344542122549, 0.364993361930441],
        );
        assert_close(
            &curves.pstate.slice(ndarray::s![1, 1, ..]).to_vec(),
            &[0.573125911355845, 0.339281558562375, 0.0875925300817798],
        );
    }

    #[test]
    fn efron_mean_is_the_mean_risk_set_of_the_tied_events() {
        assert_eq!(efron_mean(0.0, 10.0, 1, 3.0), 10.0);
        // (10 + (10 - 4 / 4)) / 2
        assert_eq!(efron_mean(0.0, 10.0, 2, 4.0), 9.5);
        assert_eq!(efron_mean(5.0, 10.0, 2, 4.0), 12.0);
    }
}
