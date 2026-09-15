//! Infinitesimal-jackknife residuals of a survival curve and the pseudo
//! values built from them: R's `residuals.survfit` (`R/residuals.survfit.R`
//! with `rsurvpart1` for single-endpoint curves, `R/rsurvpart2.R` and the C
//! kernel `src/survfitresid.c` for multi-state curves) and `pseudo`
//! (`R/pseudo.R`).
//!
//! The residual of observation `i` at time `t` is its influence on the
//! curve, `dS(t) / dw_i`; the pseudo value is `n * S(t) - (n - 1) *
//! S_{-i}(t)`, which the IJ approximates as `S(t) + n * dS(t) / dw_i`.

use super::survfit_summary::{RmeanOption, summary_survfit_times, survfit0, survmean};
use super::survfitaj::{
    AJPrepared, SurvfitAJData, SurvfitAJOptions, SurvfitAJResult, aj_prepare, survfitaj,
};
use super::survfitkm::{SurvType, SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::sorting::sorted_indices_by;
use crate::internal::validation::{validate_finite, validate_non_empty};
use ndarray::{Array2, Array3};
use pyo3::prelude::*;

/// The `type` argument of `residuals.survfit` / `pseudo`, after R's
/// aliases (`survival`, `chaz`, `rmst`, `rmts`, `sojourn`) are folded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualType {
    /// Probability in state: the survival curve.
    #[default]
    Pstate,
    /// The cumulative hazard.
    Cumhaz,
    /// Area under the curve: restricted mean survival / sojourn time.
    Auc,
}

impl ResidualType {
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "pstate" | "survival" => Ok(Self::Pstate),
            "cumhaz" | "chaz" => Ok(Self::Cumhaz),
            "sojourn" | "rmst" | "rmts" | "auc" => Ok(Self::Auc),
            other => Err(SurvivalError::invalid_input(format!(
                "type must be one of 'pstate', 'cumhaz', 'sojourn', 'survival', 'chaz', 'rmst', 'rmts', 'auc'; got {other:?}"
            ))),
        }
    }
}

/// Residuals (or pseudo values) of a survival curve: `values[r][j]` belongs
/// to row `r` at `times[j]`.  A row is an observation of the data, or a
/// subject when collapsed; `id` labels it (the observation index when no
/// id was given) and `curve` says which curve it contributed to.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitResid {
    #[pyo3(get)]
    pub id: Vec<i64>,
    #[pyo3(get)]
    pub curve: Vec<usize>,
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub values: Vec<Vec<f64>>,
}

/// `findInterval(t, dtime)`: the number of event times `<= t`.
fn find_interval(dtime: &[f64], t: f64) -> usize {
    dtime.partition_point(|&x| x <= t)
}

/// R's `approx(x, y, xout)` (linear, `yleft = 0`); `x` increasing.
fn approx_linear(x: &[f64], y: &[f64], xout: f64, rule2: bool) -> f64 {
    let n = x.len();
    if xout < x[0] {
        return 0.0;
    }
    if xout >= x[n - 1] {
        return if xout == x[n - 1] || rule2 {
            y[n - 1]
        } else {
            f64::NAN
        };
    }
    let k = x.partition_point(|&v| v <= xout); // x[k-1] <= xout < x[k]
    let (x0, x1, y0, y1) = (x[k - 1], x[k], y[k - 1], y[k]);
    y0 + (y1 - y0) * (xout - x0) / (x1 - x0)
}

/// Port of `rsurvpart1`: residuals of one curve for the rows `rows` of the
/// data, at the sorted `times`.  `curve` is the curve's row range in
/// `fit`.
#[allow(clippy::too_many_arguments)]
fn rsurvpart1(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    times: &[f64],
    kind: ResidualType,
    stype: SurvType,
    fit: &SurvfitKMResult,
    curve: std::ops::Range<usize>,
) -> Vec<Vec<f64>> {
    let ntime = times.len();
    // only the event times matter
    let events: Vec<usize> = curve.filter(|&i| fit.n_event[i] > 0.0).collect();
    let dtime: Vec<f64> = events.iter().map(|&i| fit.time[i]).collect();
    let nrisk: Vec<f64> = events.iter().map(|&i| fit.n_risk[i]).collect();
    let surv: Vec<f64> = events.iter().map(|&i| fit.surv[i]).collect();
    let mut hazard = Vec::with_capacity(events.len());
    let mut previous = 0.0;
    for &i in &events {
        hazard.push(fit.cumhaz[i] - previous);
        previous = fit.cumhaz[i];
    }
    let nevent = events.len();
    let n = rows.len();
    if nevent == 0 {
        return vec![vec![0.0; ntime]; n];
    }
    // tindex = largest event time <= reporting time, yindex the same for
    // each row's end time, sindex for its entry time
    let tindex: Vec<usize> = times.iter().map(|&t| find_interval(&dtime, t)).collect();
    let yindex: Vec<usize> = rows
        .iter()
        .map(|&r| find_interval(&dtime, stop[r]))
        .collect();
    let sindex: Option<Vec<usize>> = start.map(|start| {
        rows.iter()
            .map(|&r| find_interval(&dtime, start[r]))
            .collect()
    });
    // the dN term applies to all reporting times at or after a death
    let dmin = |row: usize, j: usize| -> usize {
        if status[rows[row]] == 0 {
            0
        } else {
            let a = yindex[row];
            if a == 0 || a > tindex[j] { 0 } else { a }
        }
    };
    let ymin = |row: usize, j: usize| yindex[row].min(tindex[j]);
    let smin = |row: usize, j: usize| sindex.as_ref().map(|s| s[row].min(tindex[j]));
    // c(0, cumsum(v)) and c(0, v) lookups, 1 + index as in R
    let cumsum0 = |v: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(v.len() + 1);
        out.push(0.0);
        let mut acc = 0.0;
        for value in v {
            acc += value;
            out.push(acc);
        }
        out
    };
    let with0 = |v: &[f64]| -> Vec<f64> {
        let mut out = Vec::with_capacity(v.len() + 1);
        out.push(0.0);
        out.extend_from_slice(v);
        out
    };
    let mut resid = vec![vec![0.0; ntime]; n];
    match kind {
        ResidualType::Cumhaz | ResidualType::Pstate
            if kind == ResidualType::Cumhaz || stype == SurvType::ExpCumhaz =>
        {
            // the hazard is the primary thing; for stype = 2 the survival
            // is exp(-cumhaz) with derivative -S(t) * d(cumhaz)
            let hsum = cumsum0(
                &hazard
                    .iter()
                    .zip(&nrisk)
                    .map(|(h, n)| h / n)
                    .collect::<Vec<_>>(),
            );
            let term1_table = with0(&nrisk.iter().map(|n| 1.0 / n).collect::<Vec<_>>());
            let surv_table = with0(&surv);
            for row in 0..n {
                for j in 0..ntime {
                    let term1 = term1_table[dmin(row, j)];
                    let term2 = hsum[ymin(row, j)];
                    let mut value = match smin(row, j) {
                        // events happen at the end of an interval, so no
                        // dN at the start
                        Some(s) => term1 + hsum[s] - term2,
                        None => term1 - term2,
                    };
                    if kind == ResidualType::Pstate {
                        value = -value * surv_table[tindex[j]];
                    }
                    resid[row][j] = value;
                }
            }
        }
        ResidualType::Pstate => {
            // avoid a 0/0 issue when S(t) = 0 and hazard = 1
            let temp: Vec<f64> = hazard
                .iter()
                .map(|&h| if h == 1.0 { 1.0 } else { 1.0 - h })
                .collect();
            let hsum = cumsum0(
                &hazard
                    .iter()
                    .zip(&nrisk)
                    .zip(&temp)
                    .map(|((h, n), t)| h / (n * t))
                    .collect::<Vec<_>>(),
            );
            let term1_table = with0(
                &temp
                    .iter()
                    .zip(&nrisk)
                    .map(|(t, n)| 1.0 / (t * n))
                    .collect::<Vec<_>>(),
            );
            let surv_table = with0(&surv);
            for row in 0..n {
                for j in 0..ntime {
                    let term1 = term1_table[dmin(row, j)];
                    let term2 = hsum[ymin(row, j)];
                    let stemp = surv_table[tindex[j]];
                    resid[row][j] = match smin(row, j) {
                        Some(s) => stemp * (term2 - (hsum[s] + term1)),
                        None => stemp * (term2 - term1),
                    };
                }
            }
        }
        ResidualType::Auc => {
            // see survfit:AUC in the methods document
            let t0 = if dtime[0] > 0.0 {
                0.0
            } else {
                2.0 * dtime[0] - 1.0
            };
            let mut aucd = Vec::with_capacity(nevent); // AUC from t0 to dtime
            let mut acc = 0.0;
            for k in 0..nevent {
                let previous = if k == 0 { t0 } else { dtime[k - 1] };
                let height = if k == 0 { 1.0 } else { surv[k - 1] };
                acc += (dtime[k] - previous) * height;
                aucd.push(acc);
            }
            let tmax = times.iter().copied().fold(f64::NAN, f64::max);
            let dmax = dtime[nevent - 1];
            let mut xs = Vec::with_capacity(nevent + 2);
            xs.push(t0);
            xs.extend_from_slice(&dtime);
            let mut ys = Vec::with_capacity(nevent + 2);
            ys.push(0.0);
            ys.extend_from_slice(&aucd);
            let extend = tmax > dmax;
            if extend {
                xs.push(tmax);
                ys.push(aucd[nevent - 1] + surv[nevent - 1] * (tmax - dmax));
            }
            let auctau: Vec<f64> = times
                .iter()
                .map(|&t| approx_linear(&xs, &ys, t, extend))
                .collect();
            let dd: Vec<f64> = nrisk
                .iter()
                .zip(&hazard)
                .map(|(n, h)| {
                    let value = match stype {
                        SurvType::ExpCumhaz => 1.0 / n,
                        SurvType::KaplanMeier => 1.0 / (n * (1.0 - h)),
                    };
                    if value.is_finite() { value } else { 0.0 } // past the last death
                })
                .collect();
            // each column of resid has a different weight vector: the AUC
            // from dtime[k] to tau
            for j in 0..ntime {
                let wt: Vec<f64> = aucd.iter().map(|a| auctau[j] - a).collect();
                let hsum = cumsum0(
                    &wt.iter()
                        .zip(&hazard)
                        .zip(&dd)
                        .map(|((w, h), d)| w * h * d)
                        .collect::<Vec<_>>(),
                );
                let term1_table =
                    with0(&wt.iter().zip(&dd).map(|(w, d)| w * d).collect::<Vec<_>>());
                for row in 0..n {
                    let term1 = term1_table[dmin(row, j)];
                    let term2 = hsum[ymin(row, j)];
                    resid[row][j] = match smin(row, j) {
                        Some(s) => term2 - (term1 + hsum[s]),
                        None => term2 - term1,
                    };
                }
            }
        }
        ResidualType::Cumhaz => unreachable!("handled by the first arm"),
    }
    resid
}

/// Sorted unique `times`, binned with `aeqSurv` when the fit was.
fn prepare_times(times: &[f64], timefix: bool) -> SurvivalResult<Vec<f64>> {
    validate_non_empty(times, "times")?;
    validate_finite(times, "times")?;
    let mut times = times.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup();
    if timefix {
        times = crate::data_prep::aeq_surv(&times, None, None)?.time;
        times.dedup();
    }
    Ok(times)
}

/// Port of `residuals.survfit` for single-endpoint curves (named after its
/// C kernel `survfitresid`).
///
/// The fit is recomputed from `data` and `options` (R re-reads the model
/// frame of the fit).  `collapse` sums the weighted residuals of the rows
/// of each `id` (only meaningful with an id that repeats); `weighted`
/// multiplies the rows by the case weights.
///
/// A `start_time` in the options drops observations from the curve but
/// not from the residuals: R hands the whole model frame to `rsurvpart1`,
/// where a row that ends before the first event time of the curve has no
/// event-time index and therefore a residual of 0 at every time.
pub fn survfitresid(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    times: &[f64],
    kind: ResidualType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitResid> {
    let fit = survfitkm(data, options)?;
    residuals_from_fit(data, options, &fit, times, kind, collapse, weighted)
}

fn residuals_from_fit(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    fit: &SurvfitKMResult,
    times: &[f64],
    kind: ResidualType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitResid> {
    let times = prepare_times(times, options.timefix)?;
    let n = data.time.len();
    // the rows of each curve, in data order
    let strata_levels: Vec<i32> = data.strata.as_ref().map_or_else(
        || vec![0],
        |strata| {
            let mut levels = strata.clone();
            levels.sort_unstable();
            levels.dedup();
            levels
        },
    );
    let curve_of: Vec<usize> = (0..n)
        .map(|i| match &data.strata {
            Some(strata) => strata_levels
                .binary_search(&strata[i])
                .expect("strata codes come from the data"),
            None => 0,
        })
        .collect();
    let ranges = fit.curve_ranges();
    let collapse = collapse
        && data.id.as_ref().is_some_and(|id| {
            let mut unique = id.clone();
            unique.sort_unstable();
            unique.dedup();
            unique.len() < id.len()
        });
    if collapse && !weighted {
        return Err(SurvivalError::invalid_input(
            "invalid combination of options: collapse=TRUE and weighted=FALSE",
        ));
    }
    let (start, stop) = if options.timefix {
        let fixed = crate::data_prep::aeq_surv(&data.time, data.start.as_deref(), None)?;
        (fixed.time2, fixed.time)
    } else {
        (data.start.clone(), data.time.clone())
    };
    let mut resid = vec![vec![0.0; times.len()]; n];
    for (curve, range) in ranges.iter().enumerate() {
        let rows: Vec<usize> = (0..n).filter(|&i| curve_of[i] == curve).collect();
        if rows.is_empty() {
            continue;
        }
        let values = rsurvpart1(
            &rows,
            start.as_deref(),
            &stop,
            &data.status,
            &times,
            kind,
            options.stype,
            fit,
            range.clone(),
        );
        for (row, value) in rows.into_iter().zip(values) {
            resid[row] = value;
        }
    }
    let casewt = |i: usize| data.weights.as_ref().map_or(1.0, |w| w[i]);
    if collapse {
        let id = data.id.as_ref().expect("collapse requires an id");
        if let Some(strata) = &data.strata {
            // the same id in several curves cannot be collapsed
            let mut seen: std::collections::HashMap<i64, i32> = std::collections::HashMap::new();
            for i in 0..n {
                if *seen.entry(id[i]).or_insert(strata[i]) != strata[i] {
                    return Err(SurvivalError::invalid_input(
                        "same id appears in multiple curves, cannot collapse",
                    ));
                }
            }
        }
        let mut order: Vec<i64> = Vec::new();
        let mut index: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut values: Vec<Vec<f64>> = Vec::new();
        let mut curve: Vec<usize> = Vec::new();
        for i in 0..n {
            let slot = *index.entry(id[i]).or_insert_with(|| {
                order.push(id[i]);
                values.push(vec![0.0; times.len()]);
                curve.push(curve_of[i]);
                values.len() - 1
            });
            let weight = casewt(i);
            for (target, value) in values[slot].iter_mut().zip(&resid[i]) {
                *target += weight * value;
            }
        }
        return Ok(SurvfitResid {
            id: order,
            curve,
            times,
            values,
        });
    }
    if weighted {
        for (i, row) in resid.iter_mut().enumerate() {
            let weight = casewt(i);
            for value in row {
                *value *= weight;
            }
        }
    }
    Ok(SurvfitResid {
        id: data.id.clone().unwrap_or_else(|| (0..n as i64).collect()),
        curve: curve_of,
        times,
        values: resid,
    })
}

/// Port of `pseudo` (`R/pseudo.R`) for single-endpoint curves: the
/// jackknife pseudo values `S(t) + n * residual` at `times`, with `n` the
/// number of observations (subjects, with an id) of the curve after any
/// `start_time`.  Rows are collapsed by id when an id repeats; rows that a
/// `start_time` removed from the curve keep the curve's estimate (their
/// residual is 0, see [`survfitresid`]).
pub fn pseudo(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    times: &[f64],
    kind: ResidualType,
) -> SurvivalResult<SurvfitResid> {
    let fit = survfitkm(data, options)?;
    let mut residuals = residuals_from_fit(data, options, &fit, times, kind, true, true)?;
    // summary(fit, rmean = t) refuses a truncation point before the first
    // time of the fit (survfitKM objects carry no start.time)
    let smallest = fit.time.iter().copied().fold(f64::INFINITY, f64::min);
    if kind == ResidualType::Auc && residuals.times.iter().any(|&t| t < smallest) {
        return Err(SurvivalError::invalid_input(
            "Truncation point for the mean time in state is < smallest survival",
        ));
    }
    let nn: Vec<f64> = fit
        .n_id
        .as_ref()
        .unwrap_or(&fit.n)
        .iter()
        .map(|&v| v as f64)
        .collect();
    let times = &residuals.times;
    let n_curves = fit.n_curves();
    // yhat[time][curve]
    let yhat: Vec<Vec<f64>> = match kind {
        ResidualType::Pstate | ResidualType::Cumhaz => {
            let summary = summary_survfit_times(&fit, times, true)?;
            let ranges = summary.curve_ranges();
            (0..times.len())
                .map(|j| {
                    ranges
                        .iter()
                        .map(|range| match kind {
                            ResidualType::Pstate => summary.surv[range.start + j],
                            _ => summary.cumhaz[range.start + j],
                        })
                        .collect()
                })
                .collect()
        }
        ResidualType::Auc => {
            let fit0 = survfit0(&fit);
            let mut yhat = Vec::with_capacity(times.len());
            for &t in times {
                let table = survmean(&fit0, 1.0, RmeanOption::At(t))?;
                yhat.push(table.rmean.expect("rmean requested"));
            }
            yhat
        }
    };
    debug_assert!(yhat.iter().all(|row| row.len() == n_curves));
    for (row, &curve) in residuals.values.iter_mut().zip(&residuals.curve) {
        for (j, value) in row.iter_mut().enumerate() {
            *value = yhat[j][curve] + nn[curve] * *value;
        }
    }
    Ok(residuals)
}

// ---------------------------------------------------------------------------
// Multi-state curves
// ---------------------------------------------------------------------------

/// Residuals (or pseudo values) of a multi-state curve:
/// `values[r][k][j]` belongs to row `r`, state (or transition, for the
/// cumulative hazard) `k`, at `times[j]`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitAJResid {
    #[pyo3(get)]
    pub id: Vec<i64>,
    #[pyo3(get)]
    pub curve: Vec<usize>,
    #[pyo3(get)]
    pub times: Vec<f64>,
    /// The states, or the `from:to` transitions for the cumulative hazard.
    #[pyo3(get)]
    pub columns: Vec<String>,
    #[pyo3(get)]
    pub values: Vec<Vec<Vec<f64>>>,
}

/// Everything `survfitresid.c` reads for one curve.
struct AJResidData<'a> {
    entry: Option<&'a [f64]>,
    etime: &'a [f64],
    /// 1-based state entered, 0 = censored.
    status: &'a [usize],
    sort1: &'a [usize],
    sort2: &'a [usize],
    cstate: &'a [usize],
    wt: &'a [f64],
    p0: &'a [f64],
    /// `nobs x nstate` initial influence.
    i0: &'a Array2<f64>,
    otime: &'a [f64],
    starttime: f64,
    doauc: bool,
}

/// Port of `survfitresid` (`src/survfitresid.c`): the influence of every
/// observation on the Aalen-Johansen estimate (`[obs, time, state]`) and,
/// when asked, on the area under it.
fn survfitresid_kernel(d: &AJResidData<'_>) -> (Array3<f64>, Option<Array3<f64>>) {
    let nobs = d.sort2.len();
    let nstate = d.p0.len();
    let nout = d.otime.len();
    let mut infp = Array3::<f64>::zeros((nobs, nout, nstate));
    let mut infa = d.doauc.then(|| Array3::<f64>::zeros((nobs, nout, nstate)));
    let mut ws = vec![0.0; nstate]; // weighted count at risk, by state
    let mut atrisk = vec![false; nobs];
    let mut pstate: Vec<f64> = d.p0.to_vec();
    let mut cmat = Array2::<f64>::zeros((nstate, nstate)); // H = I + C
    let mut tempvec = vec![0.0; nstate];
    let mut starttime = d.starttime;

    // output times before the start have zero influence
    let mut itime = 0;
    while itime < nout && d.otime[itime] < starttime {
        itime += 1;
    }
    if itime < nout {
        for j in 0..nstate {
            for i in 0..nobs {
                infp[[i, itime, j]] = d.i0[[i, j]];
            }
        }
    }
    // copy the current time's influence forward to the next output time
    let carry_forward = |infp: &mut Array3<f64>, infa: &mut Option<Array3<f64>>, itime: usize| {
        if itime + 1 < nout {
            for i in 0..nobs {
                for k in 0..nstate {
                    infp[[i, itime + 1, k]] = infp[[i, itime, k]];
                }
            }
            if let Some(infa) = infa {
                for i in 0..nobs {
                    for k in 0..nstate {
                        infa[[i, itime + 1, k]] = infa[[i, itime, k]];
                    }
                }
            }
        }
    };
    // AUC influence += (pstate influence) * (t - starttime)
    let add_auc = |infp: &Array3<f64>, infa: &mut Option<Array3<f64>>, itime: usize, width: f64| {
        if let Some(infa) = infa {
            for i in 0..nobs {
                for k in 0..nstate {
                    infa[[i, itime, k]] += infp[[i, itime, k]] * width;
                }
            }
        }
    };
    if d.entry.is_none() {
        // everyone starts out at risk
        for i in 0..nobs {
            atrisk[i] = true;
            ws[d.cstate[i]] += d.wt[i];
        }
    }
    let mut eptr = 0; // index to sort1, the entry times
    let mut i = 0;
    while i < nobs {
        let p2 = d.sort2[i];
        let ctime = d.etime[p2];
        // finish the output times before this event time
        while itime < nout && d.otime[itime] < ctime {
            add_auc(&infp, &mut infa, itime, d.otime[itime] - starttime);
            if d.doauc {
                starttime = d.otime[itime];
            }
            carry_forward(&mut infp, &mut infa, itime);
            itime += 1;
        }
        if itime == nout {
            break; // no need to go past the last output time
        }
        if let Some(entry) = d.entry {
            // add subjects whose entry time is < ctime into the counts
            while eptr < nobs {
                let p1 = d.sort1[eptr];
                if entry[p1] < ctime {
                    atrisk[p1] = true;
                    ws[d.cstate[p1]] += d.wt[p1];
                    eptr += 1;
                } else {
                    break;
                }
            }
        }
        cmat.fill(0.0);
        // count the transitions at this time point; a "move" to the same
        // state does not count
        let mut nevent = 0;
        let mut oldstate = 0;
        let mut newstate = 0;
        let mut psave = p2;
        for &q in &d.sort2[i..] {
            if d.etime[q] > ctime {
                break;
            }
            if d.status[q] != 0 && d.cstate[q] != d.status[q] - 1 {
                newstate = d.status[q] - 1;
                oldstate = d.cstate[q];
                psave = q;
                nevent += 1;
                cmat[[oldstate, newstate]] += d.wt[q] / ws[oldstate];
                cmat[[oldstate, oldstate]] -= d.wt[q] / ws[oldstate];
            }
        }
        if nevent > 0 && d.doauc {
            // the AUC influence uses the prior influence matrix
            add_auc(&infp, &mut infa, itime, ctime - starttime);
            starttime = ctime;
        }
        // Update the derivative: S(t) = S(t-)(I + C), so dS/dw_i = U(I + C)
        // + S(t-) dC/dw_i; the second term affects only those at risk.
        if nevent == 1 {
            // all but the oldstate row of C are 0
            let temp = -cmat[[oldstate, oldstate]];
            for j in 0..nobs {
                let value = infp[[j, itime, oldstate]];
                infp[[j, itime, newstate]] += temp * value;
                infp[[j, itime, oldstate]] -= temp * value;
            }
            let temp2 = pstate[oldstate] / ws[oldstate]; // S(t-) / weight
            infp[[psave, itime, newstate]] += temp2; // the obs which moved
            infp[[psave, itime, oldstate]] -= temp2;
            for &q in &d.sort2[i..] {
                if atrisk[q] && d.cstate[q] == oldstate {
                    infp[[q, itime, oldstate]] += temp * temp2;
                    infp[[q, itime, newstate]] -= temp * temp2;
                }
            }
        } else if nevent > 1 {
            // U = U + U C, a matrix multiplication
            for j in 0..nobs {
                for k in 0..nstate {
                    tempvec[k] = (0..nstate)
                        .map(|kk| infp[[j, itime, kk]] * cmat[[kk, k]])
                        .sum();
                }
                for k in 0..nstate {
                    infp[[j, itime, k]] += tempvec[k];
                }
            }
            // the dH term for everyone still at risk
            for &q in &d.sort2[i..] {
                if atrisk[q] {
                    let old = d.cstate[q];
                    let temp2 = pstate[old] / ws[old];
                    for k in 0..nstate {
                        infp[[q, itime, k]] -= cmat[[old, k]] * temp2;
                    }
                }
            }
            // C's `status > 1` skips a tied transition into the first
            // state; kept as in survfitresid.c
            for &q in &d.sort2[i..] {
                if d.etime[q] > ctime {
                    break;
                }
                let old = d.cstate[q];
                if d.status[q] > 1 && old != d.status[q] - 1 {
                    let temp2 = pstate[old] / ws[old];
                    let new = d.status[q] - 1;
                    infp[[q, itime, old]] -= temp2;
                    infp[[q, itime, new]] += temp2;
                }
            }
        }
        // update p
        for j in 0..nstate {
            tempvec[j] = (0..nstate).map(|k| pstate[k] * cmat[[k, j]]).sum();
        }
        for j in 0..nstate {
            pstate[j] += tempvec[j];
        }
        // take all the events and censors tied at ctime out of the risk set
        while i < nobs {
            let q = d.sort2[i];
            if d.etime[q] > ctime {
                break;
            }
            ws[d.cstate[q]] -= d.wt[q];
            atrisk[q] = false;
            i += 1;
        }
    }
    // reporting times after the last event
    while itime < nout {
        add_auc(&infp, &mut infa, itime, d.otime[itime] - starttime);
        if d.doauc {
            starttime = d.otime[itime];
        }
        carry_forward(&mut infp, &mut infa, itime);
        itime += 1;
    }
    (infp, infa)
}

/// Port of `rsurvpart2` for `type = "cumhaz"`: `[obs][transition][time]`
/// residuals of one curve, computed from the fitted hazard increments.
#[allow(clippy::too_many_arguments)]
fn rsurvpart2_cumhaz(
    rows: &[usize],
    entry: Option<&[f64]>,
    etime: &[f64],
    status: &[usize],
    istate: &[usize],
    times: &[f64],
    fit: &SurvfitAJResult,
    range: std::ops::Range<usize>,
) -> Vec<Vec<Vec<f64>>> {
    let n = rows.len();
    let ntime = times.len();
    let nhaz = fit.hazard_from.len();
    let events: Vec<usize> = range
        .filter(|&i| fit.n_event[i].iter().sum::<f64>() > 0.0)
        .collect();
    let dtime: Vec<f64> = events.iter().map(|&i| fit.time[i]).collect();
    let nevent = events.len();
    let mut out = vec![vec![vec![0.0; ntime]; nhaz]; n];
    if nevent == 0 {
        return out;
    }
    // hazard increments per transition and the at-risk counts per state
    let mut hazard = vec![vec![0.0; nhaz]; nevent];
    let mut previous = vec![0.0; nhaz];
    for (e, &i) in events.iter().enumerate() {
        for k in 0..nhaz {
            hazard[e][k] = fit.cumhaz[i][k] - previous[k];
            previous[k] = fit.cumhaz[i][k];
        }
    }
    let safe = |e: usize, state: usize| -> f64 {
        let value = fit.n_risk[events[e]][state];
        if value == 0.0 { 1.0 } else { value }
    };
    let tindex: Vec<usize> = times.iter().map(|&t| find_interval(&dtime, t)).collect();
    let yindex: Vec<usize> = rows
        .iter()
        .map(|&r| find_interval(&dtime, etime[r]))
        .collect();
    let sindex: Option<Vec<usize>> = entry.map(|entry| {
        rows.iter()
            .map(|&r| find_interval(&dtime, entry[r]))
            .collect()
    });
    for k in 0..nhaz {
        let from = fit.hazard_from[k];
        let to = fit.hazard_to[k];
        // cumsum(c(0, hazard / nrisk[from])) and c(0, 1 / nrisk[from])
        let mut hsum = vec![0.0; nevent + 1];
        let mut term1 = vec![0.0; nevent + 1];
        for e in 0..nevent {
            hsum[e + 1] = hsum[e] + hazard[e][k] / safe(e, from);
            term1[e + 1] = 1.0 / safe(e, from);
        }
        for (row, &r) in rows.iter().enumerate() {
            let at_risk = istate[r] == from;
            if !at_risk {
                continue;
            }
            let event = status[r] == to + 1;
            for j in 0..ntime {
                let ymin = yindex[row].min(tindex[j]);
                let dmin = if event && yindex[row] != 0 && yindex[row] <= tindex[j] {
                    yindex[row]
                } else {
                    0
                };
                let mut value = if event { term1[dmin] } else { 0.0 };
                value -= hsum[ymin];
                if let Some(sindex) = &sindex {
                    // events happen at the end of an interval, so no dN at
                    // the start
                    value += hsum[sindex[row].min(tindex[j])];
                }
                out[row][k][j] = value;
            }
        }
    }
    out
}

/// Port of `residuals.survfit` for multi-state curves (`rsurvpart2`).
/// `collapse` sums the weighted rows of each cluster (the id by default).
///
/// As for single-endpoint curves, a `start_time` does not remove rows
/// from the residuals: R passes the whole model frame to `survfitresid.c`,
/// which walks every observation from the smallest event time (the curve's
/// `t0` only zeroes the influence at reporting times before it and starts
/// the area under the curve).
pub fn survfitresid_aj(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
    times: &[f64],
    kind: ResidualType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitAJResid> {
    let fit = survfitaj(data, options)?;
    residuals_aj_from_fit(data, options, &fit, times, kind, collapse, weighted)
}

fn residuals_aj_from_fit(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
    fit: &SurvfitAJResult,
    times: &[f64],
    kind: ResidualType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitAJResid> {
    let times = prepare_times(times, options.timefix)?;
    let n = data.time.len();
    let AJPrepared {
        start,
        time,
        weights,
        id,
        check,
    } = aj_prepare(data, options.timefix)?;
    let nstate = fit.states.len();
    let strata_levels: Vec<i32> = data.strata.as_ref().map_or_else(
        || vec![0],
        |strata| {
            let mut levels = strata.clone();
            levels.sort_unstable();
            levels.dedup();
            levels
        },
    );
    let curve_of: Vec<usize> = (0..n)
        .map(|i| match &data.strata {
            Some(strata) => strata_levels
                .binary_search(&strata[i])
                .expect("strata codes come from the data"),
            None => 0,
        })
        .collect();
    let cluster: Vec<i64> = match (&data.cluster, &data.id) {
        (Some(cluster), _) => cluster.clone(),
        (None, Some(id)) => id.clone(),
        (None, None) => (0..n as i64).collect(),
    };
    let collapse = collapse && {
        let mut unique = cluster.clone();
        unique.sort_unstable();
        unique.dedup();
        unique.len() < cluster.len()
    };
    if collapse && !weighted {
        return Err(SurvivalError::invalid_input(
            "invalid combination of options: collapse=TRUE and weighted=FALSE",
        ));
    }
    let ncol = match kind {
        ResidualType::Cumhaz => fit.hazard_from.len(),
        _ => nstate,
    };
    let mut resid = vec![vec![vec![0.0; times.len()]; ncol]; n];
    let ranges = fit.curve_ranges();
    for (curve, range) in ranges.iter().enumerate() {
        let rows: Vec<usize> = (0..n).filter(|&i| curve_of[i] == curve).collect();
        if rows.is_empty() {
            continue;
        }
        let values: Vec<Vec<Vec<f64>>> = match kind {
            ResidualType::Cumhaz => rsurvpart2_cumhaz(
                &rows,
                start.as_deref(),
                &time,
                &check.stat2,
                &check.istate,
                &times,
                fit,
                range.clone(),
            ),
            ResidualType::Pstate | ResidualType::Auc => {
                let p0 = &fit.p0[curve];
                // initial leverage when p0 was estimated and the initial
                // states vary; unweighted, as rsurvpart2 has it
                let mut inf0 = Array2::<f64>::zeros((rows.len(), nstate));
                if options.p0.is_none() && p0.iter().any(|&p| p < 1.0) {
                    let at_zero: Vec<usize> = (0..rows.len())
                        .filter(|&k| match &start {
                            None => true,
                            Some(start) => start[rows[k]] < fit.t0 && time[rows[k]] >= fit.t0,
                        })
                        .collect();
                    let wtsum: f64 = at_zero.iter().map(|&k| weights[rows[k]]).sum();
                    for &k in &at_zero {
                        for j in 0..nstate {
                            let indicator = if check.istate[rows[k]] == j { 1.0 } else { 0.0 };
                            inf0[[k, j]] = (indicator - p0[j]) / wtsum;
                        }
                    }
                }
                let etime: Vec<f64> = rows.iter().map(|&r| time[r]).collect();
                let entry: Option<Vec<f64>> = start
                    .as_ref()
                    .map(|start| rows.iter().map(|&r| start[r]).collect());
                let status: Vec<usize> = rows.iter().map(|&r| check.stat2[r]).collect();
                let cstate: Vec<usize> = rows.iter().map(|&r| check.istate[r]).collect();
                let wt: Vec<f64> = rows.iter().map(|&r| weights[r]).collect();
                let sort1: Vec<usize> = entry.as_deref().map_or_else(Vec::new, sorted_indices_by);
                let sort2 = sorted_indices_by(&etime);
                let (infp, infa) = survfitresid_kernel(&AJResidData {
                    entry: entry.as_deref(),
                    etime: &etime,
                    status: &status,
                    sort1: &sort1,
                    sort2: &sort2,
                    cstate: &cstate,
                    wt: &wt,
                    p0,
                    i0: &inf0,
                    otime: &times,
                    starttime: fit.t0,
                    doauc: kind == ResidualType::Auc,
                });
                let source = match kind {
                    ResidualType::Auc => infa.expect("auc requested"),
                    _ => infp,
                };
                (0..rows.len())
                    .map(|k| {
                        (0..nstate)
                            .map(|j| (0..times.len()).map(|t| source[[k, t, j]]).collect())
                            .collect()
                    })
                    .collect()
            }
        };
        for (row, value) in rows.into_iter().zip(values) {
            resid[row] = value;
        }
    }
    let columns: Vec<String> = match kind {
        ResidualType::Cumhaz => fit
            .hazard_from
            .iter()
            .zip(&fit.hazard_to)
            .map(|(from, to)| format!("{}:{}", from + 1, to + 1))
            .collect(),
        _ => fit.states.clone(),
    };
    let casewt = |i: usize| weights[i];
    if collapse {
        if data.strata.is_some() {
            let mut seen: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
            for i in 0..n {
                if *seen.entry(id[i] as i64).or_insert(curve_of[i]) != curve_of[i] {
                    return Err(SurvivalError::invalid_input(
                        "same id appears in multiple curves, cannot collapse",
                    ));
                }
            }
        }
        let mut order: Vec<i64> = Vec::new();
        let mut index: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        let mut values: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut curve: Vec<usize> = Vec::new();
        for i in 0..n {
            let slot = *index.entry(cluster[i]).or_insert_with(|| {
                order.push(cluster[i]);
                values.push(vec![vec![0.0; times.len()]; ncol]);
                curve.push(curve_of[i]);
                values.len() - 1
            });
            let weight = casewt(i);
            for (target_col, source_col) in values[slot].iter_mut().zip(&resid[i]) {
                for (target, value) in target_col.iter_mut().zip(source_col) {
                    *target += weight * value;
                }
            }
        }
        return Ok(SurvfitAJResid {
            id: order,
            curve,
            times,
            columns,
            values,
        });
    }
    if weighted {
        for (i, by_col) in resid.iter_mut().enumerate() {
            let weight = casewt(i);
            for col in by_col {
                for value in col {
                    *value *= weight;
                }
            }
        }
    }
    Ok(SurvfitAJResid {
        id: data.id.clone().unwrap_or_else(|| (0..n as i64).collect()),
        curve: curve_of,
        times,
        columns,
        values: resid,
    })
}

/// `summary(fit, times = , extend = TRUE)$pstate` / `$cumhaz` of one curve
/// at `t`: the row of `survfit0(fit)` at the largest time `<= t`, its first
/// row (`p0` / 0, or the curve's own first row when it starts at `t0`)
/// before that.
fn aj_value_at(
    fit: &SurvfitAJResult,
    curve: usize,
    range: &std::ops::Range<usize>,
    t: f64,
    cumhaz: bool,
) -> Vec<f64> {
    let count = fit.time[range.clone()].partition_point(|&x| x <= t);
    if count == 0 {
        let starts_at_t0 = !range.is_empty() && fit.time[range.start] == fit.t0;
        match (cumhaz, starts_at_t0) {
            (true, true) => fit.cumhaz[range.start].clone(),
            (true, false) => vec![0.0; fit.hazard_from.len()],
            (false, true) => fit.pstate[range.start].clone(),
            (false, false) => fit.p0[curve].clone(),
        }
    } else if cumhaz {
        fit.cumhaz[range.start + count - 1].clone()
    } else {
        fit.pstate[range.start + count - 1].clone()
    }
}

/// `survmean2`'s mean time in state up to `maxtime`: the area under each
/// state's probability from `t0`.
fn aj_mean_time_in_state(
    fit: &SurvfitAJResult,
    curve: usize,
    range: &std::ops::Range<usize>,
    maxtime: f64,
) -> Vec<f64> {
    let nstate = fit.states.len();
    // the survfit0 rows: (t0, p0) unless the curve already starts at t0
    let mut tt = Vec::with_capacity(range.len() + 1);
    let mut rows: Vec<&Vec<f64>> = Vec::with_capacity(range.len() + 1);
    if range.is_empty() || fit.time[range.start] != fit.t0 {
        tt.push(fit.t0);
        rows.push(&fit.p0[curve]);
    }
    for i in range.clone() {
        tt.push(fit.time[i]);
        rows.push(&fit.pstate[i]);
    }
    let mut out = vec![0.0; nstate];
    for (k, &t) in tt.iter().enumerate() {
        if t >= maxtime {
            break;
        }
        let next = tt
            .get(k + 1)
            .copied()
            .filter(|&next| next < maxtime)
            .unwrap_or(maxtime);
        for j in 0..nstate {
            out[j] += (next - t) * rows[k][j];
        }
    }
    out
}

/// Port of `pseudo` for multi-state curves: `pstate(t) + n * residual`
/// with `n` the number of subjects of the curve.
pub fn pseudo_aj(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
    times: &[f64],
    kind: ResidualType,
) -> SurvivalResult<SurvfitAJResid> {
    let fit = survfitaj(data, options)?;
    let mut residuals = residuals_aj_from_fit(data, options, &fit, times, kind, true, true)?;
    let ranges = fit.curve_ranges();
    // summary(fit, rmean = t) checks the truncation point against the
    // start.time when the fit has one (survfitAJ keeps it), the smallest
    // time otherwise
    let smallest = options
        .start_time
        .unwrap_or_else(|| fit.time.iter().copied().fold(f64::INFINITY, f64::min));
    if kind == ResidualType::Auc && residuals.times.iter().any(|&t| t < smallest) {
        return Err(SurvivalError::invalid_input(
            "Truncation point for the mean time in state is < smallest survival",
        ));
    }
    // yhat[curve][time][column]
    let yhat: Vec<Vec<Vec<f64>>> = ranges
        .iter()
        .enumerate()
        .map(|(curve, range)| {
            residuals
                .times
                .iter()
                .map(|&t| match kind {
                    ResidualType::Pstate => aj_value_at(&fit, curve, range, t, false),
                    ResidualType::Cumhaz => aj_value_at(&fit, curve, range, t, true),
                    ResidualType::Auc => aj_mean_time_in_state(&fit, curve, range, t),
                })
                .collect()
        })
        .collect();
    for (by_col, &curve) in residuals.values.iter_mut().zip(&residuals.curve) {
        let nn = fit.n_id[curve] as f64;
        for (k, col) in by_col.iter_mut().enumerate() {
            for (j, value) in col.iter_mut().enumerate() {
                *value = yhat[curve][j][k] + nn * *value;
            }
        }
    }
    Ok(residuals)
}

#[allow(clippy::too_many_arguments)]
fn aj_inputs(
    time: Vec<f64>,
    state: Vec<i32>,
    states: Vec<String>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    timefix: bool,
) -> SurvivalResult<(SurvfitAJData, SurvfitAJOptions)> {
    let data = SurvfitAJData::try_new(
        start,
        time,
        state,
        states,
        weights,
        strata,
        id,
        istate,
        istate_levels,
        cluster,
    )?;
    let options = SurvfitAJOptions {
        p0,
        timefix,
        ..Default::default()
    };
    Ok((data, options))
}

/// Python binding of [`survfitresid_aj`].
#[pyfunction(name = "survfitresid_aj")]
#[pyo3(signature = (time, state, states, times, start=None, weights=None, strata=None, id=None, istate=None, istate_levels=None, cluster=None, p0=None, type_="pstate", collapse=false, weighted=None, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survfitresid_aj_py(
    time: Vec<f64>,
    state: Vec<i32>,
    states: Vec<String>,
    times: Vec<f64>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    type_: &str,
    collapse: bool,
    weighted: Option<bool>,
    timefix: bool,
) -> PyResult<SurvfitAJResid> {
    let (data, options) = aj_inputs(
        time,
        state,
        states,
        start,
        weights,
        strata,
        id,
        istate,
        istate_levels,
        cluster,
        p0,
        timefix,
    )?;
    Ok(survfitresid_aj(
        &data,
        &options,
        &times,
        ResidualType::parse(type_)?,
        collapse,
        weighted.unwrap_or(collapse),
    )?)
}

/// Python binding of [`pseudo_aj`].
#[pyfunction(name = "pseudo_aj")]
#[pyo3(signature = (time, state, states, times, start=None, weights=None, strata=None, id=None, istate=None, istate_levels=None, cluster=None, p0=None, type_="pstate", timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn pseudo_aj_py(
    time: Vec<f64>,
    state: Vec<i32>,
    states: Vec<String>,
    times: Vec<f64>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    type_: &str,
    timefix: bool,
) -> PyResult<SurvfitAJResid> {
    let (data, options) = aj_inputs(
        time,
        state,
        states,
        start,
        weights,
        strata,
        id,
        istate,
        istate_levels,
        cluster,
        p0,
        timefix,
    )?;
    Ok(pseudo_aj(
        &data,
        &options,
        &times,
        ResidualType::parse(type_)?,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn km_inputs(
    time: Vec<f64>,
    status: Vec<i32>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    stype: i32,
    ctype: i32,
    timefix: bool,
) -> SurvivalResult<(SurvfitKMData, SurvfitKMOptions)> {
    let data = SurvfitKMData::try_new(start, time, status, weights, strata, id, None)?;
    let options = SurvfitKMOptions {
        stype: SurvType::from_code(stype)?,
        ctype: super::survfitkm::HazardType::from_code(ctype)?,
        timefix,
        ..Default::default()
    };
    Ok((data, options))
}

/// Python binding of [`survfitresid`].
#[pyfunction(name = "survfitresid")]
#[pyo3(signature = (time, status, times, start=None, weights=None, strata=None, id=None, type_="pstate", stype=1, ctype=1, collapse=false, weighted=None, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survfitresid_py(
    time: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    type_: &str,
    stype: i32,
    ctype: i32,
    collapse: bool,
    weighted: Option<bool>,
    timefix: bool,
) -> PyResult<SurvfitResid> {
    let (data, options) = km_inputs(
        time, status, start, weights, strata, id, stype, ctype, timefix,
    )?;
    Ok(survfitresid(
        &data,
        &options,
        &times,
        ResidualType::parse(type_)?,
        collapse,
        weighted.unwrap_or(collapse),
    )?)
}

/// Python binding of [`pseudo`].
#[pyfunction(name = "pseudo")]
#[pyo3(signature = (time, status, times, start=None, weights=None, strata=None, id=None, type_="pstate", stype=1, ctype=1, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn pseudo_py(
    time: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    type_: &str,
    stype: i32,
    ctype: i32,
    timefix: bool,
) -> PyResult<SurvfitResid> {
    let (data, options) = km_inputs(
        time, status, start, weights, strata, id, stype, ctype, timefix,
    )?;
    Ok(pseudo(
        &data,
        &options,
        &times,
        ResidualType::parse(type_)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aml() -> SurvfitKMData {
        let time = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ];
        let status = vec![
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        ];
        SurvfitKMData::right_censored(time, status).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-10 * b.abs().max(1.0)
    }

    #[test]
    fn residuals_and_pseudo_values_match_r_for_aml() {
        // residuals(survfit(Surv(time, status) ~ 1, aml), times = c(12, 24, 48))
        let times = [12.0, 24.0, 48.0];
        let options = SurvfitKMOptions::default();
        let resid =
            survfitresid(&aml(), &options, &times, ResidualType::Pstate, false, false).unwrap();
        assert_eq!(resid.values.len(), 23);
        assert_eq!(resid.times, times);
        assert!(close(resid.values[0][0], -0.0321361058601134));
        assert!(close(resid.values[0][1], -0.023764515257899));
        assert!(close(resid.values[1][0], 0.0113421550094518));
        assert!(close(resid.values[2][1], 0.0103969754253308));
        let cumhaz =
            survfitresid(&aml(), &options, &times, ResidualType::Cumhaz, false, false).unwrap();
        assert!(close(cumhaz.values[0][0], 0.0415456301161012));
        assert!(close(cumhaz.values[1][0], -0.0141723685843537));
        assert!(close(cumhaz.values[2][1], -0.0176325761968104));
        let auc = survfitresid(&aml(), &options, &times, ResidualType::Auc, false, false).unwrap();
        assert!(close(auc.values[0][0], -0.0831758034026465));
        assert!(close(auc.values[0][2], -0.782878746961923));
        assert!(close(auc.values[2][2], 0.350661625708885));
        // pseudo(fit, times, type)
        let ps = pseudo(&aml(), &options, &times, ResidualType::Pstate).unwrap();
        assert!(ps.values[0][0].abs() < 1e-12);
        assert!(close(ps.values[2][1], 0.785714285714286));
        assert!(close(ps.values[2][2], 0.119047619047619));
        let ps = pseudo(&aml(), &options, &times, ResidualType::Cumhaz).unwrap();
        assert!(close(ps.values[0][0], 1.24593124415048));
        assert!(close(ps.values[2][2], 1.75547476518401));
        let ps = pseudo(&aml(), &options, &times, ResidualType::Auc).unwrap();
        for value in &ps.values[0] {
            assert!(close(*value, 9.0));
        }
        assert!(close(ps.values[2][1], 23.4285714285714));
        assert!(close(ps.values[2][2], 35.0714285714286));
    }

    #[test]
    fn stype_two_uses_the_hazard_derivative() {
        let options = SurvfitKMOptions {
            stype: SurvType::ExpCumhaz,
            ..Default::default()
        };
        let times = [12.0];
        let pstate =
            survfitresid(&aml(), &options, &times, ResidualType::Pstate, false, false).unwrap();
        let cumhaz =
            survfitresid(&aml(), &options, &times, ResidualType::Cumhaz, false, false).unwrap();
        let fit = survfitkm(&aml(), &options).unwrap();
        let surv12 = fit.surv[fit.time.partition_point(|&t| t <= 12.0) - 1];
        for (p, h) in pstate.values.iter().zip(&cumhaz.values) {
            assert!(close(p[0], -h[0] * surv12));
        }
    }

    #[test]
    fn collapse_sums_a_subject_rows_and_weights_apply() {
        let data = SurvfitKMData::try_new(
            Some(vec![0.0, 2.0, 0.0, 3.0, 0.0, 4.0]),
            vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
            vec![0, 1, 1, 0, 0, 1],
            Some(vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0]),
            None,
            Some(vec![1, 1, 2, 2, 3, 3]),
            None,
        )
        .unwrap();
        let options = SurvfitKMOptions::default();
        let plain =
            survfitresid(&data, &options, &[5.0], ResidualType::Pstate, false, false).unwrap();
        let collapsed =
            survfitresid(&data, &options, &[5.0], ResidualType::Pstate, true, true).unwrap();
        assert_eq!(collapsed.id, vec![1, 2, 3]);
        assert!(close(
            collapsed.values[0][0],
            plain.values[0][0] + plain.values[1][0]
        ));
        assert!(close(
            collapsed.values[1][0],
            2.0 * (plain.values[2][0] + plain.values[3][0])
        ));
        assert!(survfitresid(&data, &options, &[5.0], ResidualType::Pstate, true, false).is_err());
        // pseudo values are inflated by the number of subjects
        let ps = pseudo(&data, &options, &[5.0], ResidualType::Pstate).unwrap();
        assert_eq!(ps.values.len(), 3);
        let fit = survfitkm(&data, &options).unwrap();
        let s5 = fit.surv[fit.time.partition_point(|&t| t <= 5.0) - 1];
        assert!(close(ps.values[0][0], s5 + 3.0 * collapsed.values[0][0]));
    }

    #[test]
    fn strata_are_handled_curve_by_curve() {
        let mut data = aml();
        data.strata = Some(vec![1; 11].into_iter().chain(vec![2; 12]).collect());
        let options = SurvfitKMOptions::default();
        let both = pseudo(&data, &options, &[12.0, 24.0, 48.0], ResidualType::Pstate).unwrap();
        assert_eq!(both.curve[0], 0);
        assert_eq!(both.curve[11], 1);
        let mut second = aml();
        second.time = second.time[11..].to_vec();
        second.status = second.status[11..].to_vec();
        let alone = pseudo(&second, &options, &[12.0, 24.0, 48.0], ResidualType::Pstate).unwrap();
        for (row, expected) in both.values[11..].iter().zip(&alone.values) {
            for (a, b) in row.iter().zip(expected) {
                assert!(close(*a, *b));
            }
        }
    }

    /// The synthetic_ties_mstate fixture frame.
    fn ties_mstate() -> SurvfitAJData {
        let time = vec![
            1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0,
        ];
        let event = [
            "a", "b", "censor", "b", "a", "b", "censor", "b", "a", "b", "a", "censor", "a",
            "censor", "a", "b",
        ];
        let state: Vec<i32> = event
            .iter()
            .map(|e| match *e {
                "a" => 1,
                "b" => 2,
                _ => 0,
            })
            .collect();
        SurvfitAJData::try_new(
            None,
            time,
            state,
            vec!["a".to_string(), "b".to_string()],
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn multistate_residuals_and_pseudo_values_match_r() {
        // residuals(survfit(Surv(time, event) ~ 1), times = c(2, 5, 8), type = "pstate")
        let data = ties_mstate();
        let options = SurvfitAJOptions::default();
        let times = [2.0, 5.0, 8.0];
        let resid =
            survfitresid_aj(&data, &options, &times, ResidualType::Pstate, false, false).unwrap();
        assert_eq!(resid.columns, vec!["(s0)", "a", "b"]);
        assert_eq!(resid.values.len(), 16);
        let fit = survfitaj(&data, &options).unwrap();
        // the residuals reproduce the robust variance of pstate at time 2
        let se = fit.std_err.as_ref().unwrap();
        let row_at_2 = fit.time.partition_point(|&t| t <= 2.0) - 1;
        for j in 0..3 {
            let norm = resid
                .values
                .iter()
                .map(|by_col| by_col[j][0] * by_col[j][0])
                .sum::<f64>()
                .sqrt();
            assert!(
                close(norm, se[row_at_2][j]),
                "{norm} != {}",
                se[row_at_2][j]
            );
        }
        // pseudo values average to the estimate
        let ps = pseudo_aj(&data, &options, &times, ResidualType::Pstate).unwrap();
        for j in 0..3 {
            let mean = ps.values.iter().map(|by_col| by_col[j][0]).sum::<f64>() / 16.0;
            assert!(close(mean, fit.pstate[row_at_2][j]));
        }
        // cumulative hazard and sojourn residuals sum to zero over subjects
        for kind in [ResidualType::Cumhaz, ResidualType::Auc] {
            let resid = survfitresid_aj(&data, &options, &times, kind, false, false).unwrap();
            let ncol = resid.columns.len();
            for k in 0..ncol {
                for t in 0..3 {
                    let total: f64 = resid.values.iter().map(|by_col| by_col[k][t]).sum();
                    assert!(total.abs() < 1e-12, "{kind:?} column {k} time {t}: {total}");
                }
            }
        }
        let cumhaz =
            survfitresid_aj(&data, &options, &times, ResidualType::Cumhaz, false, false).unwrap();
        assert_eq!(cumhaz.columns, vec!["1:2", "1:3"]);
    }

    #[test]
    fn start_time_keeps_every_row_as_r_does() {
        // fit <- survfit(Surv(time, status) ~ 1, aml, start.time = 10)
        // residuals(fit, times = c(12, 24, 48)); pseudo(fit, times = ...)
        let options = SurvfitKMOptions {
            start_time: Some(10.0),
            ..Default::default()
        };
        let times = [12.0, 24.0, 48.0];
        let resid =
            survfitresid(&aml(), &options, &times, ResidualType::Pstate, false, false).unwrap();
        assert_eq!(resid.values.len(), 23);
        // the observation at time 9 is not part of the curve: residual 0
        assert_eq!(resid.values[0], vec![0.0, 0.0, 0.0]);
        assert!(close(resid.values[1][0], 0.00308641975308642));
        assert!(close(resid.values[1][1], -0.03880070546737213));
        assert!(close(resid.values[3][2], -0.006823717141177459));
        let ps = pseudo(&aml(), &options, &times, ResidualType::Pstate).unwrap();
        // ... and its pseudo value is the estimate, inflated by fit$n = 18
        assert!(close(ps.values[0][0], 0.944444444444444));
        assert!(close(ps.values[0][2], 0.1058201058201058));
        assert!(close(ps.values[1][0], 1.0));
        assert!(ps.values[1][1].abs() < 1e-12);
        assert!(close(ps.values[3][1], -0.112244897959184));
        let auc = pseudo(&aml(), &options, &times, ResidualType::Auc).unwrap();
        assert!(close(auc.values[0][0], 2.0)); // the area from t0 = 10 to 12
        assert!(close(auc.values[0][1], 12.2142857142857));
        assert!(close(auc.values[2][2], 25.0714285714286));
        let resid_auc =
            survfitresid(&aml(), &options, &times, ResidualType::Auc, false, false).unwrap();
        assert!(close(resid_auc.values[1][1], -0.511_904_761_904_762));
        assert!(close(resid_auc.values[2][2], 0.139329805996473));
        // summary(fit, rmean = 5) refuses a point before the first time
        assert!(
            pseudo(&aml(), &options, &[5.0, 24.0], ResidualType::Auc)
                .unwrap_err()
                .to_string()
                .contains("smallest survival")
        );
    }

    #[test]
    fn multistate_start_time_keeps_every_row_as_r_does() {
        // fit <- survfit(Surv(time, event) ~ 1, start.time = 2) on the
        // synthetic_ties_mstate frame; residuals(fit, times = c(2, 5, 8))
        let data = ties_mstate();
        let options = SurvfitAJOptions {
            start_time: Some(2.0),
            ..Default::default()
        };
        let times = [2.0, 5.0, 8.0];
        let resid =
            survfitresid_aj(&data, &options, &times, ResidualType::Pstate, false, false).unwrap();
        assert_eq!(resid.values.len(), 16);
        // values[row][state][time]: the rows before the start take part
        assert!(close(resid.values[0][0][1], -0.03312800480769231));
        assert!(close(resid.values[0][1][1], 0.04965444711538462));
        assert!(close(resid.values[1][2][1], 0.045_973_557_692_307_7));
        assert!(close(resid.values[2][1][1], -0.00262920673076923));
        let ps = pseudo_aj(&data, &options, &times, ResidualType::Pstate).unwrap();
        assert!(close(ps.values[0][0][1], 0.175105168269231));
        assert!(close(ps.values[0][1][1], 0.808_969_350_961_538_4));
        assert!(close(ps.values[2][2][1], 0.2034254807692308));
        let auc = pseudo_aj(&data, &options, &times, ResidualType::Auc).unwrap();
        assert!(close(auc.values[0][0][2], 0.350116436298077));
        assert!(close(auc.values[0][1][2], 5.621_788_611_778_847));
        assert!(close(auc.values[2][2][2], 1.1062199519230769));
        // the truncation point is checked against the start.time
        assert!(
            pseudo_aj(&data, &options, &[1.5, 5.0], ResidualType::Auc)
                .unwrap_err()
                .to_string()
                .contains("smallest survival")
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(ResidualType::parse("weird").is_err());
        assert_eq!(ResidualType::parse("RMST").unwrap(), ResidualType::Auc);
        let options = SurvfitKMOptions::default();
        assert!(survfitresid(&aml(), &options, &[], ResidualType::Pstate, false, false).is_err());
        assert!(
            survfitresid(
                &aml(),
                &options,
                &[f64::NAN],
                ResidualType::Pstate,
                false,
                false
            )
            .is_err()
        );
    }
}
