use super::survfit_aj_summary::{summary_survfit_aj, time_in_state};
use super::survfit_summary::{
    RmeanOption, summary_survfit_times, survfit0_aj, survfit0_with, survmean,
};
use super::survfitaj::{
    AJPrepared, SurvfitAJData, SurvfitAJOptions, SurvfitAJResult, aj_prepare, survfitaj,
};
use super::survfitkm::{
    SurvType, SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, rows_by_curve, strata_index,
    survfitkm,
};
use crate::data_prep::aeq_counting;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::sorting::sorted_indices_by;
use crate::internal::step::find_interval;
use crate::internal::validation::{validate_finite, validate_non_empty};
use ndarray::{Array2, Array3, s};
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PseudoResidualType {
    #[default]
    Pstate,
    Cumhaz,
    Auc,
}

impl PseudoResidualType {
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

#[doc = " Residuals (or pseudo values) of a survival curve: `values[r][j]` belongs"]
#[doc = " to row `r` at `times[j]`.  A row is an observation of the data, or a"]
#[doc = " subject when collapsed; `id` labels it (the observation index when no"]
#[doc = " id was given) and `curve` says which curve it contributed to."]
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
    let k = x.partition_point(|&v| v <= xout);
    let (x0, x1, y0, y1) = (x[k - 1], x[k], y[k - 1], y[k]);
    y0 + (y1 - y0) * (xout - x0) / (x1 - x0)
}

#[allow(clippy::too_many_arguments)]
fn rsurvpart1(
    rows: &[usize],
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    times: &[f64],
    kind: PseudoResidualType,
    stype: SurvType,
    fit: &SurvfitKMResult,
    curve: std::ops::Range<usize>,
) -> Vec<Vec<f64>> {
    let ntime = times.len();
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
    let tindex: Vec<usize> = times
        .iter()
        .map(|&t| find_interval(&dtime, t, false))
        .collect();
    let yindex: Vec<usize> = rows
        .iter()
        .map(|&r| find_interval(&dtime, stop[r], false))
        .collect();
    let sindex: Option<Vec<usize>> = start.map(|start| {
        rows.iter()
            .map(|&r| find_interval(&dtime, start[r], false))
            .collect()
    });
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
        PseudoResidualType::Cumhaz | PseudoResidualType::Pstate
            if kind == PseudoResidualType::Cumhaz || stype == SurvType::ExpCumhaz =>
        {
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
                        Some(s) => term1 + hsum[s] - term2,
                        None => term1 - term2,
                    };
                    if kind == PseudoResidualType::Pstate {
                        value = -value * surv_table[tindex[j]];
                    }
                    resid[row][j] = value;
                }
            }
        }
        PseudoResidualType::Pstate => {
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
        PseudoResidualType::Auc => {
            let t0 = if dtime[0] > 0.0 {
                0.0
            } else {
                2.0 * dtime[0] - 1.0
            };
            let mut aucd = Vec::with_capacity(nevent);
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
                    if value.is_finite() { value } else { 0.0 }
                })
                .collect();
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
        PseudoResidualType::Cumhaz => unreachable!("handled by the first arm"),
    }
    resid
}

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

#[allow(clippy::too_many_arguments)]
pub fn survfitresid(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    times: &[f64],
    kind: PseudoResidualType,
    call_stype: SurvType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitResid> {
    let fit = survfitkm(data, options)?;
    residuals_from_fit(
        data, options, &fit, times, kind, call_stype, collapse, weighted,
    )
}

#[allow(clippy::too_many_arguments)]
fn residuals_from_fit(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    fit: &SurvfitKMResult,
    times: &[f64],
    kind: PseudoResidualType,
    call_stype: SurvType,
    collapse: bool,
    weighted: bool,
) -> SurvivalResult<SurvfitResid> {
    let times = prepare_times(times, options.timefix)?;
    let n = data.time.len();
    let (strata_levels, curve_of) = strata_index(data.strata.as_deref(), n);
    let ranges = fit.curve_ranges();
    if ranges.len() != strata_levels.len() {
        return Err(SurvivalError::invalid_input(
            "start.time has removed all the observations from at least one curve",
        ));
    }
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
        aeq_counting(data.start.as_deref(), &data.time)?
    } else {
        (data.start.clone(), data.time.clone())
    };
    let mut resid = vec![vec![0.0; times.len()]; n];
    for (range, rows) in ranges
        .iter()
        .zip(rows_by_curve(&curve_of, strata_levels.len()))
    {
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
            call_stype,
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

pub fn pseudo(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
    times: &[f64],
    kind: PseudoResidualType,
    call_stype: SurvType,
    collapse: bool,
) -> SurvivalResult<SurvfitResid> {
    let fit = survfitkm(data, options)?;
    let mut residuals = residuals_from_fit(
        data, options, &fit, times, kind, call_stype, collapse, collapse,
    )?;
    let smallest = fit.time.iter().copied().fold(f64::INFINITY, f64::min);
    if kind == PseudoResidualType::Auc && residuals.times.iter().any(|&t| t < smallest) {
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
    let yhat: Vec<Vec<f64>> = match kind {
        PseudoResidualType::Pstate | PseudoResidualType::Cumhaz => {
            let summary = summary_survfit_times(&fit, times, true)?;
            let ranges = summary.curve_ranges();
            (0..times.len())
                .map(|j| {
                    ranges
                        .iter()
                        .map(|range| match kind {
                            PseudoResidualType::Pstate => summary.surv[range.start + j],
                            _ => summary.cumhaz[range.start + j],
                        })
                        .collect()
                })
                .collect()
        }
        PseudoResidualType::Auc => {
            let fit0 = survfit0_with(&fit, false);
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

#[doc = " Residuals (or pseudo values) of a multi-state curve:"]
#[doc = " `values[r][k][j]` belongs to row `r`, state (or transition, for the"]
#[doc = " cumulative hazard) `k`, at `times[j]`."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitAJResid {
    #[pyo3(get)]
    pub id: Vec<i64>,
    #[pyo3(get)]
    pub curve: Vec<usize>,
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[doc = " The states, or the `from:to` transitions for the cumulative hazard."]
    #[pyo3(get)]
    pub columns: Vec<String>,
    #[pyo3(get)]
    pub values: Vec<Vec<Vec<f64>>>,
}

struct AJResidData<'a> {
    entry: Option<&'a [f64]>,
    etime: &'a [f64],
    status: &'a [usize],
    sort1: &'a [usize],
    sort2: &'a [usize],
    cstate: &'a [usize],
    wt: &'a [f64],
    p0: &'a [f64],
    i0: &'a Array2<f64>,
    otime: &'a [f64],
    starttime: f64,
    doauc: bool,
}

fn survfitresid_kernel(d: &AJResidData<'_>) -> (Array3<f64>, Option<Array3<f64>>) {
    let nobs = d.sort2.len();
    let nstate = d.p0.len();
    let nout = d.otime.len();
    let mut infp = Array3::<f64>::zeros((nobs, nstate, nout));
    let mut infa = d.doauc.then(|| Array3::<f64>::zeros((nobs, nstate, nout)));
    let mut cur = vec![0.0; nstate * nobs];
    let mut cur_auc = d.doauc.then(|| vec![0.0; nstate * nobs]);
    let mut update = vec![0.0; nstate * nobs];
    let mut ws = vec![0.0; nstate];
    let mut atrisk = vec![false; nobs];
    let mut pstate: Vec<f64> = d.p0.to_vec();
    let mut cmat = Array2::<f64>::zeros((nstate, nstate));
    let mut tempvec = vec![0.0; nstate];
    let mut starttime = d.starttime;

    let mut itime = 0;
    while itime < nout && d.otime[itime] < starttime {
        itime += 1;
    }
    if itime < nout {
        for j in 0..nstate {
            for i in 0..nobs {
                cur[j * nobs + i] = d.i0[[i, j]];
            }
        }
    }
    let store = |infp: &mut Array3<f64>,
                 infa: &mut Option<Array3<f64>>,
                 cur: &[f64],
                 cur_auc: &Option<Vec<f64>>,
                 itime: usize| {
        for k in 0..nstate {
            for i in 0..nobs {
                infp[[i, k, itime]] = cur[k * nobs + i];
            }
        }
        if let (Some(infa), Some(cur_auc)) = (infa, cur_auc) {
            for k in 0..nstate {
                for i in 0..nobs {
                    infa[[i, k, itime]] = cur_auc[k * nobs + i];
                }
            }
        }
    };
    let add_auc = |cur: &[f64], cur_auc: &mut Option<Vec<f64>>, width: f64| {
        if let Some(cur_auc) = cur_auc {
            for (area, &value) in cur_auc.iter_mut().zip(cur) {
                *area += value * width;
            }
        }
    };
    if d.entry.is_none() {
        for i in 0..nobs {
            atrisk[i] = true;
            ws[d.cstate[i]] += d.wt[i];
        }
    }
    let mut eptr = 0;
    let mut i = 0;
    while i < nobs {
        let p2 = d.sort2[i];
        let ctime = d.etime[p2];
        while itime < nout && d.otime[itime] < ctime {
            add_auc(&cur, &mut cur_auc, d.otime[itime] - starttime);
            if d.doauc {
                starttime = d.otime[itime];
            }
            store(&mut infp, &mut infa, &cur, &cur_auc, itime);
            itime += 1;
        }
        if itime == nout {
            break;
        }
        if let Some(entry) = d.entry {
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
            add_auc(&cur, &mut cur_auc, ctime - starttime);
            starttime = ctime;
        }
        if nevent == 1 {
            let temp = -cmat[[oldstate, oldstate]];
            if oldstate != newstate {
                let (low, high) = cur.split_at_mut(oldstate.max(newstate) * nobs);
                let (old_row, new_row) = if oldstate < newstate {
                    (
                        &mut low[oldstate * nobs..(oldstate + 1) * nobs],
                        &mut high[..nobs],
                    )
                } else {
                    (
                        &mut high[..nobs],
                        &mut low[newstate * nobs..(newstate + 1) * nobs],
                    )
                };
                for (old, new) in old_row.iter_mut().zip(new_row.iter_mut()) {
                    let value = *old;
                    *new += temp * value;
                    *old -= temp * value;
                }
            }
            let temp2 = pstate[oldstate] / ws[oldstate];
            cur[newstate * nobs + psave] += temp2;
            cur[oldstate * nobs + psave] -= temp2;
            for &q in &d.sort2[i..] {
                if atrisk[q] && d.cstate[q] == oldstate {
                    cur[oldstate * nobs + q] += temp * temp2;
                    cur[newstate * nobs + q] -= temp * temp2;
                }
            }
        } else if nevent > 1 {
            update.fill(0.0);
            for k in 0..nstate {
                let target = &mut update[k * nobs..(k + 1) * nobs];
                for kk in 0..nstate {
                    let weight = cmat[[kk, k]];
                    for (sum, &value) in target.iter_mut().zip(&cur[kk * nobs..(kk + 1) * nobs]) {
                        *sum += value * weight;
                    }
                }
            }
            for (value, delta) in cur.iter_mut().zip(&update) {
                *value += delta;
            }
            for &q in &d.sort2[i..] {
                if atrisk[q] {
                    let old = d.cstate[q];
                    let temp2 = pstate[old] / ws[old];
                    for k in 0..nstate {
                        cur[k * nobs + q] -= cmat[[old, k]] * temp2;
                    }
                }
            }
            for &q in &d.sort2[i..] {
                if d.etime[q] > ctime {
                    break;
                }
                let old = d.cstate[q];
                if d.status[q] > 1 && old != d.status[q] - 1 {
                    let temp2 = pstate[old] / ws[old];
                    let new = d.status[q] - 1;
                    cur[old * nobs + q] -= temp2;
                    cur[new * nobs + q] += temp2;
                }
            }
        }
        for j in 0..nstate {
            tempvec[j] = (0..nstate).map(|k| pstate[k] * cmat[[k, j]]).sum();
        }
        for j in 0..nstate {
            pstate[j] += tempvec[j];
        }
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
    while itime < nout {
        add_auc(&cur, &mut cur_auc, d.otime[itime] - starttime);
        if d.doauc {
            starttime = d.otime[itime];
        }
        store(&mut infp, &mut infa, &cur, &cur_auc, itime);
        itime += 1;
    }
    (infp, infa)
}

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
    let tindex: Vec<usize> = times
        .iter()
        .map(|&t| find_interval(&dtime, t, false))
        .collect();
    let yindex: Vec<usize> = rows
        .iter()
        .map(|&r| find_interval(&dtime, etime[r], false))
        .collect();
    let sindex: Option<Vec<usize>> = entry.map(|entry| {
        rows.iter()
            .map(|&r| find_interval(&dtime, entry[r], false))
            .collect()
    });
    for k in 0..nhaz {
        let from = fit.hazard_from[k];
        let to = fit.hazard_to[k];
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
                    value += hsum[sindex[row].min(tindex[j])];
                }
                out[row][k][j] = value;
            }
        }
    }
    out
}

pub fn survfitresid_aj(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
    times: &[f64],
    kind: PseudoResidualType,
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
    kind: PseudoResidualType,
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
    let (strata_levels, curve_of) = strata_index(data.strata.as_deref(), n);
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
        PseudoResidualType::Cumhaz => fit.hazard_from.len(),
        _ => nstate,
    };
    let mut resid = vec![vec![vec![0.0; times.len()]; ncol]; n];
    let ranges = fit.curve_ranges();
    for (curve, (range, rows)) in ranges
        .iter()
        .zip(rows_by_curve(&curve_of, strata_levels.len()))
        .enumerate()
    {
        if rows.is_empty() {
            continue;
        }
        let values: Vec<Vec<Vec<f64>>> = match kind {
            PseudoResidualType::Cumhaz => rsurvpart2_cumhaz(
                &rows,
                start.as_deref(),
                &time,
                &check.stat2,
                &check.istate,
                &times,
                fit,
                range.clone(),
            ),
            PseudoResidualType::Pstate | PseudoResidualType::Auc => {
                let p0 = &fit.p0[curve];
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
                    doauc: kind == PseudoResidualType::Auc,
                });
                let source = match kind {
                    PseudoResidualType::Auc => infa.expect("auc requested"),
                    _ => infp,
                };
                (0..rows.len())
                    .map(|k| {
                        (0..nstate)
                            .map(|j| source.slice(s![k, j, ..]).to_vec())
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
        PseudoResidualType::Cumhaz => fit
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

pub fn pseudo_aj(
    data: &SurvfitAJData,
    options: &SurvfitAJOptions,
    times: &[f64],
    kind: PseudoResidualType,
    collapse: bool,
) -> SurvivalResult<SurvfitAJResid> {
    let fit = survfitaj(data, options)?;
    let mut residuals =
        residuals_aj_from_fit(data, options, &fit, times, kind, collapse, collapse)?;
    let smallest = options
        .start_time
        .unwrap_or_else(|| fit.time.iter().copied().fold(f64::INFINITY, f64::min));
    if kind == PseudoResidualType::Auc && residuals.times.iter().any(|&t| t < smallest) {
        return Err(SurvivalError::invalid_input(
            "Truncation point for the mean time in state is < smallest survival",
        ));
    }
    let yhat: Vec<Vec<Vec<f64>>> = match kind {
        PseudoResidualType::Pstate | PseudoResidualType::Cumhaz => {
            let summary = summary_survfit_aj(&fit, Some(&residuals.times), false, true)?;
            let values = match kind {
                PseudoResidualType::Cumhaz => &summary.cumhaz,
                _ => &summary.pstate,
            };
            summary
                .curve_ranges()
                .into_iter()
                .map(|range| values[range].to_vec())
                .collect()
        }
        PseudoResidualType::Auc => {
            let fit0 = survfit0_aj(&fit);
            let nstate = fit0.states.len();
            fit0.curve_ranges()
                .into_iter()
                .map(|range| {
                    residuals
                        .times
                        .iter()
                        .map(|&t| {
                            (0..nstate)
                                .map(|state| time_in_state(&fit0, range.clone(), t, state))
                                .collect()
                        })
                        .collect()
                })
                .collect()
        }
    };
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
    time: FloatVec,
    state: IntVec,
    states: Vec<String>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    start_time: Option<f64>,
    timefix: bool,
) -> SurvivalResult<(SurvfitAJData, SurvfitAJOptions)> {
    let data = SurvfitAJData::try_new(
        start.map(FloatVec::into_inner),
        time.into_inner(),
        state.into_inner(),
        states,
        weights.map(FloatVec::into_inner),
        strata.map(IntVec::into_inner),
        id,
        istate,
        istate_levels,
        cluster,
    )?;
    let options = SurvfitAJOptions {
        p0,
        start_time,
        timefix,
        ..Default::default()
    };
    Ok((data, options))
}

#[doc = " Python binding of [`survfitresid_aj`]."]
#[pyfunction(name = "survfitresid_aj")]
#[pyo3(signature = (time, state, states, times, start=None, weights=None, strata=None, id=None, istate=None, istate_levels=None, cluster=None, p0=None, type_="pstate", collapse=false, weighted=None, timefix=true, start_time=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn survfitresid_aj_py(
    py: Python<'_>,
    time: FloatVec,
    state: IntVec,
    states: Vec<String>,
    times: Vec<f64>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    type_: &str,
    collapse: bool,
    weighted: Option<bool>,
    timefix: bool,
    start_time: Option<f64>,
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
        start_time,
        timefix,
    )?;
    let residual_type = PseudoResidualType::parse(type_)?;
    let weighted = weighted.unwrap_or(collapse);
    Ok(py.detach(|| survfitresid_aj(&data, &options, &times, residual_type, collapse, weighted))?)
}

#[doc = " Python binding of [`pseudo_aj`]."]
#[pyfunction(name = "pseudo_aj")]
#[pyo3(signature = (time, state, states, times, start=None, weights=None, strata=None, id=None, istate=None, istate_levels=None, cluster=None, p0=None, type_="pstate", timefix=true, collapse=true, start_time=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_aj_py(
    py: Python<'_>,
    time: FloatVec,
    state: IntVec,
    states: Vec<String>,
    times: Vec<f64>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    istate: Option<Vec<String>>,
    istate_levels: Option<Vec<String>>,
    cluster: Option<Vec<i64>>,
    p0: Option<Vec<f64>>,
    type_: &str,
    timefix: bool,
    collapse: bool,
    start_time: Option<f64>,
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
        start_time,
        timefix,
    )?;
    let residual_type = PseudoResidualType::parse(type_)?;
    Ok(py.detach(|| pseudo_aj(&data, &options, &times, residual_type, collapse))?)
}

#[allow(clippy::too_many_arguments)]
fn km_inputs(
    time: FloatVec,
    status: IntVec,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    stype: i32,
    ctype: i32,
    start_time: Option<f64>,
    timefix: bool,
) -> SurvivalResult<(SurvfitKMData, SurvfitKMOptions)> {
    let data = SurvfitKMData::try_new(
        start.map(FloatVec::into_inner),
        time.into_inner(),
        status.into_inner(),
        weights.map(FloatVec::into_inner),
        strata.map(IntVec::into_inner),
        id,
        None,
    )?;
    let options = SurvfitKMOptions {
        stype: SurvType::from_code(stype)?,
        ctype: super::survfitkm::HazardType::from_code(ctype)?,
        start_time,
        timefix,
        ..Default::default()
    };
    Ok((data, options))
}

#[doc = " Python binding of [`survfitresid`]."]
#[pyfunction(name = "survfitresid")]
#[pyo3(signature = (time, status, times, start=None, weights=None, strata=None, id=None, type_="pstate", stype=1, ctype=1, collapse=false, weighted=None, timefix=true, start_time=None, call_stype=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn survfitresid_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    times: Vec<f64>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    type_: &str,
    stype: i32,
    ctype: i32,
    collapse: bool,
    weighted: Option<bool>,
    timefix: bool,
    start_time: Option<f64>,
    call_stype: Option<i32>,
) -> PyResult<SurvfitResid> {
    let (data, options) = km_inputs(
        time, status, start, weights, strata, id, stype, ctype, start_time, timefix,
    )?;
    let call_stype = call_stype.map_or(Ok(options.stype), SurvType::from_code)?;
    let residual_type = PseudoResidualType::parse(type_)?;
    let weighted = weighted.unwrap_or(collapse);
    Ok(py.detach(|| {
        survfitresid(
            &data,
            &options,
            &times,
            residual_type,
            call_stype,
            collapse,
            weighted,
        )
    })?)
}

#[doc = " Python binding of [`pseudo`]."]
#[pyfunction(name = "pseudo")]
#[pyo3(signature = (time, status, times, start=None, weights=None, strata=None, id=None, type_="pstate", stype=1, ctype=1, timefix=true, collapse=true, start_time=None, call_stype=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pseudo_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    times: Vec<f64>,
    start: Option<FloatVec>,
    weights: Option<FloatVec>,
    strata: Option<IntVec>,
    id: Option<Vec<i64>>,
    type_: &str,
    stype: i32,
    ctype: i32,
    timefix: bool,
    collapse: bool,
    start_time: Option<f64>,
    call_stype: Option<i32>,
) -> PyResult<SurvfitResid> {
    let (data, options) = km_inputs(
        time, status, start, weights, strata, id, stype, ctype, start_time, timefix,
    )?;
    let call_stype = call_stype.map_or(Ok(options.stype), SurvType::from_code)?;
    let residual_type = PseudoResidualType::parse(type_)?;
    Ok(py.detach(|| pseudo(&data, &options, &times, residual_type, call_stype, collapse))?)
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
        let times = [12.0, 24.0, 48.0];
        let options = SurvfitKMOptions::default();
        let resid = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            options.stype,
            false,
            false,
        )
        .unwrap();
        assert_eq!(resid.values.len(), 23);
        assert_eq!(resid.times, times);
        assert!(close(resid.values[0][0], -0.0321361058601134));
        assert!(close(resid.values[0][1], -0.023764515257899));
        assert!(close(resid.values[1][0], 0.0113421550094518));
        assert!(close(resid.values[2][1], 0.0103969754253308));
        let cumhaz = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Cumhaz,
            options.stype,
            false,
            false,
        )
        .unwrap();
        assert!(close(cumhaz.values[0][0], 0.0415456301161012));
        assert!(close(cumhaz.values[1][0], -0.0141723685843537));
        assert!(close(cumhaz.values[2][1], -0.0176325761968104));
        let auc = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Auc,
            options.stype,
            false,
            false,
        )
        .unwrap();
        assert!(close(auc.values[0][0], -0.0831758034026465));
        assert!(close(auc.values[0][2], -0.782878746961923));
        assert!(close(auc.values[2][2], 0.350661625708885));
        let ps = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            options.stype,
            true,
        )
        .unwrap();
        assert!(ps.values[0][0].abs() < 1e-12);
        assert!(close(ps.values[2][1], 0.785714285714286));
        assert!(close(ps.values[2][2], 0.119047619047619));
        let ps = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Cumhaz,
            options.stype,
            true,
        )
        .unwrap();
        assert!(close(ps.values[0][0], 1.24593124415048));
        assert!(close(ps.values[2][2], 1.75547476518401));
        let ps = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Auc,
            options.stype,
            true,
        )
        .unwrap();
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
        let pstate = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            options.stype,
            false,
            false,
        )
        .unwrap();
        let cumhaz = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Cumhaz,
            options.stype,
            false,
            false,
        )
        .unwrap();
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
        let plain = survfitresid(
            &data,
            &options,
            &[5.0],
            PseudoResidualType::Pstate,
            options.stype,
            false,
            false,
        )
        .unwrap();
        let collapsed = survfitresid(
            &data,
            &options,
            &[5.0],
            PseudoResidualType::Pstate,
            options.stype,
            true,
            true,
        )
        .unwrap();
        assert_eq!(collapsed.id, vec![1, 2, 3]);
        assert!(close(
            collapsed.values[0][0],
            plain.values[0][0] + plain.values[1][0]
        ));
        assert!(close(
            collapsed.values[1][0],
            2.0 * (plain.values[2][0] + plain.values[3][0])
        ));
        assert!(
            survfitresid(
                &data,
                &options,
                &[5.0],
                PseudoResidualType::Pstate,
                options.stype,
                true,
                false
            )
            .is_err()
        );
        let ps = pseudo(
            &data,
            &options,
            &[5.0],
            PseudoResidualType::Pstate,
            options.stype,
            true,
        )
        .unwrap();
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
        let both = pseudo(
            &data,
            &options,
            &[12.0, 24.0, 48.0],
            PseudoResidualType::Pstate,
            options.stype,
            true,
        )
        .unwrap();
        assert_eq!(both.curve[0], 0);
        assert_eq!(both.curve[11], 1);
        let mut second = aml();
        second.time = second.time[11..].to_vec();
        second.status = second.status[11..].to_vec();
        let alone = pseudo(
            &second,
            &options,
            &[12.0, 24.0, 48.0],
            PseudoResidualType::Pstate,
            options.stype,
            true,
        )
        .unwrap();
        for (row, expected) in both.values[11..].iter().zip(&alone.values) {
            for (a, b) in row.iter().zip(expected) {
                assert!(close(*a, *b));
            }
        }
    }

    fn interleaved_counting_groups() -> SurvfitKMData {
        SurvfitKMData::try_new(
            Some(vec![0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 5.0]),
            vec![2.0, 3.0, 6.0, 4.0, 7.0, 5.0, 8.0, 9.0],
            vec![0, 0, 1, 0, 1, 0, 0, 0],
            Some(vec![0.5, 1.5, 0.5, 2.0, 1.5, 0.75, 2.0, 0.75]),
            Some(vec![17, -4, 17, 17, -4, -4, 17, -4]),
            Some(vec![91, 42, 91, 15, 42, 77, 15, 77]),
            None,
        )
        .unwrap()
    }

    fn select_rows<T: Clone>(values: &[T], rows: &[usize]) -> Vec<T> {
        rows.iter().map(|&row| values[row].clone()).collect()
    }

    #[test]
    fn grouped_counting_residuals_and_pseudo_preserve_row_and_subject_order() {
        let data = interleaved_counting_groups();
        let groups = [(0, vec![1, 4, 5, 7]), (1, vec![0, 2, 3, 6])];
        let times = [10.0, 7.0, 7.0];
        for start_time in [None, Some(4.5)] {
            let options = SurvfitKMOptions {
                start_time,
                timefix: false,
                ..Default::default()
            };
            for kind in [
                PseudoResidualType::Pstate,
                PseudoResidualType::Cumhaz,
                PseudoResidualType::Auc,
            ] {
                for (collapse, weighted) in [(false, false), (false, true), (true, true)] {
                    let both = survfitresid(
                        &data,
                        &options,
                        &times,
                        kind,
                        options.stype,
                        collapse,
                        weighted,
                    )
                    .unwrap();
                    let expected_ids = if collapse {
                        vec![91, 42, 15, 77]
                    } else {
                        data.id.clone().unwrap()
                    };
                    let expected_curves = if collapse {
                        vec![1, 0, 1, 0]
                    } else {
                        vec![1, 0, 1, 1, 0, 0, 1, 0]
                    };
                    assert_eq!(both.id, expected_ids);
                    assert_eq!(both.curve, expected_curves);
                    assert_eq!(both.times, vec![7.0, 10.0]);
                    let ps =
                        pseudo(&data, &options, &times, kind, options.stype, collapse).unwrap();
                    assert_eq!(ps.id, both.id);
                    assert_eq!(ps.curve, both.curve);
                    for (curve, rows) in &groups {
                        let alone_data = SurvfitKMData::try_new(
                            Some(select_rows(data.start.as_ref().unwrap(), rows)),
                            select_rows(&data.time, rows),
                            select_rows(&data.status, rows),
                            Some(select_rows(data.weights.as_ref().unwrap(), rows)),
                            None,
                            Some(select_rows(data.id.as_ref().unwrap(), rows)),
                            None,
                        )
                        .unwrap();
                        let alone = survfitresid(
                            &alone_data,
                            &options,
                            &times,
                            kind,
                            options.stype,
                            collapse,
                            weighted,
                        )
                        .unwrap();
                        let alone_ps =
                            pseudo(&alone_data, &options, &times, kind, options.stype, collapse)
                                .unwrap();
                        for (k, &id) in alone.id.iter().enumerate() {
                            let row = if collapse {
                                both.id.iter().position(|&value| value == id).unwrap()
                            } else {
                                rows[k]
                            };
                            assert_eq!(both.curve[row], *curve);
                            for j in 0..both.times.len() {
                                assert!(close(both.values[row][j], alone.values[k][j]));
                                assert!(close(ps.values[row][j], alone_ps.values[k][j]));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn grouped_multistate_residuals_preserve_histories_and_cluster_order() {
        let km = interleaved_counting_groups();
        let data = SurvfitAJData::try_new(
            km.start,
            km.time,
            vec![0, 1, 1, 0, 2, 0, 2, 0],
            vec!["a".to_string(), "b".to_string()],
            km.weights,
            km.strata,
            km.id,
            None,
            None,
            Some(vec![501, 23, 501, 501, 23, 23, 501, 23]),
        )
        .unwrap();
        let groups = [(0, vec![1, 4, 5, 7]), (1, vec![0, 2, 3, 6])];
        let times = [10.0, 7.0, 7.0];
        for start_time in [None, Some(2.5)] {
            let options = SurvfitAJOptions {
                start_time,
                timefix: false,
                ..Default::default()
            };
            for kind in [
                PseudoResidualType::Pstate,
                PseudoResidualType::Cumhaz,
                PseudoResidualType::Auc,
            ] {
                for (collapse, weighted) in [(false, false), (false, true), (true, true)] {
                    let both =
                        survfitresid_aj(&data, &options, &times, kind, collapse, weighted).unwrap();
                    assert_eq!(
                        both.id,
                        if collapse {
                            vec![501, 23]
                        } else {
                            data.id.clone().unwrap()
                        }
                    );
                    assert_eq!(
                        both.curve,
                        if collapse {
                            vec![1, 0]
                        } else {
                            vec![1, 0, 1, 1, 0, 0, 1, 0]
                        }
                    );
                    assert_eq!(both.times, vec![7.0, 10.0]);
                    let ps = pseudo_aj(&data, &options, &times, kind, collapse).unwrap();
                    assert_eq!(ps.id, both.id);
                    assert_eq!(ps.curve, both.curve);
                    for (curve, rows) in &groups {
                        let alone_data = SurvfitAJData::try_new(
                            Some(select_rows(data.start.as_ref().unwrap(), rows)),
                            select_rows(&data.time, rows),
                            select_rows(&data.state, rows),
                            data.states.clone(),
                            Some(select_rows(data.weights.as_ref().unwrap(), rows)),
                            None,
                            Some(select_rows(data.id.as_ref().unwrap(), rows)),
                            None,
                            None,
                            Some(select_rows(data.cluster.as_ref().unwrap(), rows)),
                        )
                        .unwrap();
                        let alone = survfitresid_aj(
                            &alone_data,
                            &options,
                            &times,
                            kind,
                            collapse,
                            weighted,
                        )
                        .unwrap();
                        let alone_ps =
                            pseudo_aj(&alone_data, &options, &times, kind, collapse).unwrap();
                        for (k, &id) in alone.id.iter().enumerate() {
                            let row = if collapse {
                                both.id.iter().position(|&value| value == id).unwrap()
                            } else {
                                rows[k]
                            };
                            assert_eq!(both.curve[row], *curve);
                            for (col, name) in both.columns.iter().enumerate() {
                                let separate_col =
                                    alone.columns.iter().position(|value| value == name);
                                for j in 0..both.times.len() {
                                    let expected =
                                        separate_col.map_or(0.0, |c| alone.values[k][c][j]);
                                    let expected_ps =
                                        separate_col.map_or(0.0, |c| alone_ps.values[k][c][j]);
                                    assert!(close(both.values[row][col][j], expected));
                                    assert!(close(ps.values[row][col][j], expected_ps));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
        let data = ties_mstate();
        let options = SurvfitAJOptions::default();
        let times = [2.0, 5.0, 8.0];
        let resid = survfitresid_aj(
            &data,
            &options,
            &times,
            PseudoResidualType::Pstate,
            false,
            false,
        )
        .unwrap();
        assert_eq!(resid.columns, vec!["(s0)", "a", "b"]);
        assert_eq!(resid.values.len(), 16);
        let fit = survfitaj(&data, &options).unwrap();
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
        let ps = pseudo_aj(&data, &options, &times, PseudoResidualType::Pstate, true).unwrap();
        for j in 0..3 {
            let mean = ps.values.iter().map(|by_col| by_col[j][0]).sum::<f64>() / 16.0;
            assert!(close(mean, fit.pstate[row_at_2][j]));
        }
        for kind in [PseudoResidualType::Cumhaz, PseudoResidualType::Auc] {
            let resid = survfitresid_aj(&data, &options, &times, kind, false, false).unwrap();
            let ncol = resid.columns.len();
            for k in 0..ncol {
                for t in 0..3 {
                    let total: f64 = resid.values.iter().map(|by_col| by_col[k][t]).sum();
                    assert!(total.abs() < 1e-12, "{kind:?} column {k} time {t}: {total}");
                }
            }
        }
        let cumhaz = survfitresid_aj(
            &data,
            &options,
            &times,
            PseudoResidualType::Cumhaz,
            false,
            false,
        )
        .unwrap();
        assert_eq!(cumhaz.columns, vec!["1:2", "1:3"]);
    }

    #[test]
    fn start_time_keeps_every_row_as_r_does() {
        let options = SurvfitKMOptions {
            start_time: Some(10.0),
            ..Default::default()
        };
        let times = [12.0, 24.0, 48.0];
        let resid = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            options.stype,
            false,
            false,
        )
        .unwrap();
        assert_eq!(resid.values.len(), 23);
        assert_eq!(resid.values[0], vec![0.0, 0.0, 0.0]);
        assert!(close(resid.values[1][0], 0.00308641975308642));
        assert!(close(resid.values[1][1], -0.03880070546737213));
        assert!(close(resid.values[3][2], -0.006823717141177459));
        let ps = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            options.stype,
            true,
        )
        .unwrap();
        assert!(close(ps.values[0][0], 0.944444444444444));
        assert!(close(ps.values[0][2], 0.1058201058201058));
        assert!(close(ps.values[1][0], 1.0));
        assert!(ps.values[1][1].abs() < 1e-12);
        assert!(close(ps.values[3][1], -0.112244897959184));
        let auc = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Auc,
            options.stype,
            true,
        )
        .unwrap();
        assert!(close(auc.values[0][0], 2.0));
        assert!(close(auc.values[0][1], 12.2142857142857));
        assert!(close(auc.values[2][2], 25.0714285714286));
        let resid_auc = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Auc,
            options.stype,
            false,
            false,
        )
        .unwrap();
        assert!(close(resid_auc.values[1][1], -0.511_904_761_904_762));
        assert!(close(resid_auc.values[2][2], 0.139329805996473));
        assert!(
            pseudo(
                &aml(),
                &options,
                &[5.0, 24.0],
                PseudoResidualType::Auc,
                options.stype,
                true
            )
            .unwrap_err()
            .to_string()
            .contains("smallest survival")
        );
    }

    #[test]
    fn start_time_that_empties_a_stratum_is_refused() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 9.0];
        let status = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1];
        let strata = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let options = SurvfitKMOptions {
            start_time: Some(5.0),
            ..Default::default()
        };
        let refused = |result: SurvivalResult<SurvfitResid>| {
            result
                .unwrap_err()
                .to_string()
                .contains("start.time has removed all the observations from at least one curve")
        };
        for (rows, n) in [(12, vec![0, 4, 4]), (8, vec![0, 4])] {
            let data = SurvfitKMData::try_new(
                None,
                time[..rows].to_vec(),
                status[..rows].to_vec(),
                None,
                Some(strata[..rows].to_vec()),
                None,
                None,
            )
            .unwrap();
            assert_eq!(survfitkm(&data, &options).unwrap().n, n);
            for kind in [PseudoResidualType::Pstate, PseudoResidualType::Auc] {
                assert!(refused(survfitresid(
                    &data,
                    &options,
                    &[6.0, 7.0],
                    kind,
                    options.stype,
                    false,
                    false
                )));
                assert!(refused(pseudo(
                    &data,
                    &options,
                    &[6.0, 7.0],
                    kind,
                    options.stype,
                    true
                )));
            }
        }
    }

    #[test]
    fn multistate_start_time_keeps_every_row_as_r_does() {
        let data = ties_mstate();
        let options = SurvfitAJOptions {
            start_time: Some(2.0),
            ..Default::default()
        };
        let times = [2.0, 5.0, 8.0];
        let resid = survfitresid_aj(
            &data,
            &options,
            &times,
            PseudoResidualType::Pstate,
            false,
            false,
        )
        .unwrap();
        assert_eq!(resid.values.len(), 16);
        assert!(close(resid.values[0][0][1], -0.03312800480769231));
        assert!(close(resid.values[0][1][1], 0.04965444711538462));
        assert!(close(resid.values[1][2][1], 0.045_973_557_692_307_7));
        assert!(close(resid.values[2][1][1], -0.00262920673076923));
        let ps = pseudo_aj(&data, &options, &times, PseudoResidualType::Pstate, true).unwrap();
        assert!(close(ps.values[0][0][1], 0.175105168269231));
        assert!(close(ps.values[0][1][1], 0.808_969_350_961_538_4));
        assert!(close(ps.values[2][2][1], 0.2034254807692308));
        let auc = pseudo_aj(&data, &options, &times, PseudoResidualType::Auc, true).unwrap();
        assert!(close(auc.values[0][0][2], 0.350116436298077));
        assert!(close(auc.values[0][1][2], 5.621_788_611_778_847));
        assert!(close(auc.values[2][2][2], 1.1062199519230769));
        assert!(
            pseudo_aj(&data, &options, &[1.5, 5.0], PseudoResidualType::Auc, true)
                .unwrap_err()
                .to_string()
                .contains("smallest survival")
        );
    }

    #[test]
    fn call_stype_picks_the_formula_and_the_fit_gives_the_curve() {
        let options = SurvfitKMOptions {
            stype: SurvType::ExpCumhaz,
            ..Default::default()
        };
        let times = [10.0, 20.0];
        let kaplan_meier = SurvType::KaplanMeier;
        let resid = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            kaplan_meier,
            false,
            false,
        )
        .unwrap();
        assert!(close(resid.values[0][0], -0.03437862026111806));
        assert!(close(resid.values[0][1], -0.0285491513533135));
        assert!(close(resid.values[1][0], 0.00954961673919946));
        let auc = survfitresid(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Auc,
            kaplan_meier,
            false,
            false,
        )
        .unwrap();
        assert!(close(auc.values[0][0], -0.0153620986438052));
        assert!(close(auc.values[1][1], -0.151269864863623));
        let ps = pseudo(
            &aml(),
            &options,
            &times,
            PseudoResidualType::Pstate,
            kaplan_meier,
            true,
        )
        .unwrap();
        assert!(close(ps.values[1][0], 1.0103494510073));
        let fh2 = SurvfitKMOptions {
            stype: SurvType::ExpCumhaz,
            ctype: super::super::survfitkm::HazardType::FlemingHarrington,
            ..Default::default()
        };
        let cumhaz = survfitresid(
            &aml(),
            &fh2,
            &times,
            PseudoResidualType::Cumhaz,
            kaplan_meier,
            false,
            false,
        )
        .unwrap();
        assert!(close(cumhaz.values[1][0], -0.0112852529328128));
        assert!(close(cumhaz.values[1][1], 0.0409916491134088));
    }

    #[test]
    fn rejects_bad_arguments() {
        assert!(PseudoResidualType::parse("weird").is_err());
        assert_eq!(
            PseudoResidualType::parse("RMST").unwrap(),
            PseudoResidualType::Auc
        );
        let options = SurvfitKMOptions::default();
        assert!(
            survfitresid(
                &aml(),
                &options,
                &[],
                PseudoResidualType::Pstate,
                options.stype,
                false,
                false
            )
            .is_err()
        );
        assert!(
            survfitresid(
                &aml(),
                &options,
                &[f64::NAN],
                PseudoResidualType::Pstate,
                options.stype,
                false,
                false
            )
            .is_err()
        );
    }
}
