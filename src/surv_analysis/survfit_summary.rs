//! The pieces of `summary.survfit`, `print.survfit` and `quantile.survfit`
//! that compute something: the time-0 row (`survfit0`), the per-curve table
//! of `survmean` (records, events, restricted mean and its standard error,
//! median with confidence limits), the curve evaluated at requested times
//! (`summary(fit, times = )`) and the quantiles of the curve with their
//! confidence limits.  All of them read a [`SurvfitKMResult`].

use super::survfitaj::{SurvfitAJCounts, SurvfitAJResult};
use super::survfitkm::{SurvfitCounts, SurvfitInfluence, SurvfitKMResult};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;
use pyo3::prelude::*;

/// `all.equal`'s tolerance, used by `survmean` and `quantile.survfit`.
fn r_tolerance() -> f64 {
    f64::EPSILON.sqrt()
}

/// Port of `survfit0`: insert a row at the starting time `t0` with `surv =
/// 1`, `cumhaz = 0`, zero counts and zero standard errors into every curve
/// that does not already start there.  Influence matrices get a zero column.
pub fn survfit0(fit: &SurvfitKMResult) -> SurvfitKMResult {
    let t0 = fit.t0;
    let ranges = fit.curve_ranges();
    let inserts: Vec<bool> = ranges
        .iter()
        .map(|range| range.is_empty() || fit.time[range.start] != t0)
        .collect();
    if inserts.iter().all(|&insert| !insert) {
        return fit.clone();
    }
    let n_new: usize = fit.time.len() + inserts.iter().filter(|&&insert| insert).count();
    // build the stacked vectors with the extra rows
    let addto = |values: &[f64], zero: f64, first: bool| -> Vec<f64> {
        let mut out = Vec::with_capacity(n_new);
        for (range, &insert) in ranges.iter().zip(&inserts) {
            if insert {
                out.push(if first {
                    values.get(range.start).copied().unwrap_or(zero)
                } else {
                    zero
                });
            }
            out.extend_from_slice(&values[range.clone()]);
        }
        out
    };
    let add_option = |values: &Option<Vec<f64>>, zero: f64| -> Option<Vec<f64>> {
        values.as_ref().map(|values| addto(values, zero, false))
    };
    let add_influence = |list: &Option<Vec<SurvfitInfluence>>| -> Option<Vec<SurvfitInfluence>> {
        list.as_ref().map(|list| {
            list.iter()
                .zip(&inserts)
                .map(|(influence, &insert)| SurvfitInfluence {
                    cluster: influence.cluster.clone(),
                    values: influence
                        .values
                        .iter()
                        .map(|row| {
                            let mut new_row = Vec::with_capacity(row.len() + 1);
                            if insert {
                                new_row.push(0.0);
                            }
                            new_row.extend_from_slice(row);
                            new_row
                        })
                        .collect(),
                })
                .collect()
        })
    };
    SurvfitKMResult {
        n: fit.n.clone(),
        time: addto(&fit.time, t0, false),
        n_risk: addto(&fit.n_risk, 0.0, true),
        n_event: addto(&fit.n_event, 0.0, false),
        n_censor: addto(&fit.n_censor, 0.0, false),
        n_enter: add_option(&fit.n_enter, 0.0),
        counts: fit.counts.as_ref().map(|counts| SurvfitCounts {
            n_risk: addto(&counts.n_risk, 0.0, true),
            n_event: addto(&counts.n_event, 0.0, false),
            n_censor: addto(&counts.n_censor, 0.0, false),
            n_enter: add_option(&counts.n_enter, 0.0),
        }),
        surv: addto(&fit.surv, 1.0, false),
        std_err: add_option(&fit.std_err, 0.0),
        cumhaz: addto(&fit.cumhaz, 0.0, false),
        std_chaz: add_option(&fit.std_chaz, 0.0),
        lower: add_option(&fit.lower, 1.0),
        upper: add_option(&fit.upper, 1.0),
        strata: fit.strata.as_ref().map(|strata| {
            strata
                .iter()
                .zip(&inserts)
                .map(|(&count, &insert)| count + usize::from(insert))
                .collect()
        }),
        strata_codes: fit.strata_codes.clone(),
        n_id: fit.n_id.clone(),
        logse: fit.logse,
        conf_int: fit.conf_int,
        conf_type: fit.conf_type.clone(),
        conf_lower: fit.conf_lower.clone(),
        type_: fit.type_.clone(),
        t0,
        influence_surv: add_influence(&fit.influence_surv),
        influence_chaz: add_influence(&fit.influence_chaz),
    }
}

/// `survfit0` for a multi-state curve: the inserted row carries `p0`,
/// zero hazards and, as R has it for `survfitms` objects, zero standard
/// errors and confidence limits.  The influence matrices are left alone.
pub fn survfit0_aj(fit: &SurvfitAJResult) -> SurvfitAJResult {
    let t0 = fit.t0;
    let ranges = fit.curve_ranges();
    let inserts: Vec<bool> = ranges
        .iter()
        .map(|range| range.is_empty() || fit.time[range.start] != t0)
        .collect();
    if inserts.iter().all(|&insert| !insert) {
        return fit.clone();
    }
    let n_new = fit.time.len() + inserts.iter().filter(|&&insert| insert).count();
    let addrows = |values: &[Vec<f64>], row: &dyn Fn(usize) -> Vec<f64>| -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(n_new);
        for (curve, (range, &insert)) in ranges.iter().zip(&inserts).enumerate() {
            if insert {
                out.push(row(curve));
            }
            out.extend(values[range.clone()].iter().cloned());
        }
        out
    };
    let nstate = fit.states.len();
    let nhaz = fit.hazard_from.len();
    let zeros_state = |_: usize| vec![0.0; nstate];
    let zeros_haz = |_: usize| vec![0.0; nhaz];
    let first_risk = |curve: usize| {
        let range = &ranges[curve];
        if range.is_empty() {
            vec![0.0; nstate]
        } else {
            fit.n_risk[range.start].clone()
        }
    };
    let add_option = |values: &Option<Vec<Vec<f64>>>, row: &dyn Fn(usize) -> Vec<f64>| {
        values.as_ref().map(|values| addrows(values, row))
    };
    let mut time = Vec::with_capacity(n_new);
    for (range, &insert) in ranges.iter().zip(&inserts) {
        if insert {
            time.push(t0);
        }
        time.extend_from_slice(&fit.time[range.clone()]);
    }
    SurvfitAJResult {
        n: fit.n.clone(),
        time,
        n_risk: addrows(&fit.n_risk, &first_risk),
        n_event: addrows(&fit.n_event, &zeros_state),
        n_censor: addrows(&fit.n_censor, &zeros_state),
        n_enter: add_option(&fit.n_enter, &zeros_state),
        n_transition: addrows(&fit.n_transition, &zeros_haz),
        counts: fit.counts.as_ref().map(|counts| SurvfitAJCounts {
            n_risk: addrows(&counts.n_risk, &|curve| {
                let range = &ranges[curve];
                if range.is_empty() {
                    vec![0.0; nstate]
                } else {
                    counts.n_risk[range.start].clone()
                }
            }),
            n_transition: addrows(&counts.n_transition, &zeros_haz),
            n_censor: addrows(&counts.n_censor, &zeros_state),
            n_enter: add_option(&counts.n_enter, &zeros_state),
        }),
        pstate: addrows(&fit.pstate, &|curve| fit.p0[curve].clone()),
        cumhaz: addrows(&fit.cumhaz, &zeros_haz),
        std_err: add_option(&fit.std_err, &zeros_state),
        std_chaz: add_option(&fit.std_chaz, &zeros_haz),
        std_auc: add_option(&fit.std_auc, &zeros_state),
        se0: fit.se0.clone(),
        lower: add_option(&fit.lower, &zeros_state),
        upper: add_option(&fit.upper, &zeros_state),
        p0: fit.p0.clone(),
        strata: fit.strata.as_ref().map(|strata| {
            strata
                .iter()
                .zip(&inserts)
                .map(|(&count, &insert)| count + usize::from(insert))
                .collect()
        }),
        strata_codes: fit.strata_codes.clone(),
        n_id: fit.n_id.clone(),
        states: fit.states.clone(),
        transitions: fit.transitions.clone(),
        hazard_from: fit.hazard_from.clone(),
        hazard_to: fit.hazard_to.clone(),
        logse: fit.logse,
        conf_int: fit.conf_int,
        conf_type: fit.conf_type.clone(),
        type_: fit.type_.clone(),
        t0,
        start_time: fit.start_time,
        influence_pstate: fit.influence_pstate.clone(),
    }
}

/// The `rmean` argument of `summary.survfit` / `print.survfit`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RmeanOption {
    /// No restricted mean.
    None,
    /// Every curve is truncated at the largest time of all curves.
    Common,
    /// Each curve is truncated at its own last time.
    Individual,
    /// A user-supplied truncation time.
    At(f64),
}

impl RmeanOption {
    /// `"none"`, `"common"`, `"individual"` or a number.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "common" => Ok(Self::Common),
            "individual" => Ok(Self::Individual),
            other => other
                .parse::<f64>()
                .map(Self::At)
                .map_err(|_| SurvivalError::invalid_input("Invalid value for rmean option")),
        }
    }
}

/// The per-curve table of `survmean` (`summary(fit)$table`).  Every vector
/// has one entry per curve; `rmean`/`se_rmean` are absent for
/// `RmeanOption::None` and the median limits are absent when the fit has no
/// confidence limits.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvmeanTable {
    #[pyo3(get)]
    pub records: Vec<f64>,
    /// `n.max`, or `n.id` when subjects were identified.
    #[pyo3(get)]
    pub n_max: Vec<f64>,
    #[pyo3(get)]
    pub n_start: Vec<f64>,
    #[pyo3(get)]
    pub events: Vec<f64>,
    #[pyo3(get)]
    pub rmean: Option<Vec<f64>>,
    #[pyo3(get)]
    pub se_rmean: Option<Vec<f64>>,
    #[pyo3(get)]
    pub median: Vec<f64>,
    #[pyo3(get)]
    pub lower: Option<Vec<f64>>,
    #[pyo3(get)]
    pub upper: Option<Vec<f64>>,
    /// The truncation time of each curve's restricted mean (`NaN` without).
    #[pyo3(get)]
    pub end_time: Vec<f64>,
}

/// `minmin` of `survmean`: the x at which the decreasing y first drops
/// below .5, with the midpoint rule for a flat exactly at .5.
fn minmin(y: &[f64], x: &[f64]) -> f64 {
    let tolerance = r_tolerance();
    let keep: Vec<usize> = (0..y.len())
        .filter(|&i| !y[i].is_nan() && y[i] < 0.5 + tolerance)
        .collect();
    let Some(&first) = keep.first() else {
        return f64::NAN;
    };
    if (y[first] - 0.5).abs() < tolerance
        && let Some(&below) = keep.iter().find(|&&i| y[i] < y[first])
    {
        return (x[first] + x[below]) / 2.0;
    }
    x[first]
}

/// One row of `survmean`'s output (its inner `pfun`).
#[allow(clippy::too_many_arguments)]
fn survmean_row(
    nused: f64,
    time: &[f64],
    surv: &[f64],
    n_risk: &[f64],
    n_event: &[f64],
    limits: Option<(&[f64], &[f64])>,
    start_time: f64,
    end_time: f64,
    nid: Option<f64>,
) -> [f64; 9] {
    let (mean, varmean) = if end_time.is_nan() {
        (0.0, 0.0) // placeholders
    } else {
        let hh = |i: usize| -> f64 {
            if n_risk[i] - n_event[i] == 0.0 {
                0.0
            } else {
                n_event[i] / (n_risk[i] * (n_risk[i] - n_event[i]))
            }
        };
        let keep = time.iter().filter(|&&t| t <= end_time).count();
        let (temptime, tempsurv, hh): (Vec<f64>, Vec<f64>, Vec<f64>) = if keep == 0 {
            // the cutoff is before the first event
            (vec![end_time], vec![1.0], vec![0.0])
        } else {
            let mut temptime = time[..keep].to_vec();
            temptime.push(end_time);
            let mut tempsurv = surv[..keep].to_vec();
            tempsurv.push(surv[keep - 1]);
            let mut hh: Vec<f64> = (0..keep).map(hh).collect();
            hh.push(0.0);
            (temptime, tempsurv, hh)
        };
        let n = temptime.len();
        // width of rectangles, then their area
        let rectangles: Vec<f64> = (0..n)
            .map(|i| {
                let previous = if i == 0 { start_time } else { temptime[i - 1] };
                let height = if i == 0 { 1.0 } else { tempsurv[i - 1] };
                (temptime[i] - previous) * height
            })
            .collect();
        let mean: f64 = rectangles.iter().sum();
        // varmean = sum(cumsum(rev(rectangles[-1]))^2 * rev(hh)[-1]), see
        // Miller, Survival Analysis (1981) pp. 195-196
        let mut varmean = 0.0;
        let mut tail = 0.0;
        for i in (1..n).rev() {
            tail += rectangles[i];
            varmean += tail * tail * hh[i - 1];
        }
        (mean, varmean)
    };
    let maxn = nid.unwrap_or_else(|| n_risk.iter().copied().fold(f64::NAN, f64::max));
    let med = minmin(surv, time);
    let (lower, upper) = match limits {
        Some((lower, upper)) => (minmin(lower, time), minmin(upper, time)),
        None => (0.0, 0.0),
    };
    [
        nused,
        maxn,
        n_risk.first().copied().unwrap_or(f64::NAN),
        n_event.iter().sum(),
        mean,
        varmean.sqrt(),
        med,
        lower,
        upper,
    ]
}

/// Port of `survmean` (`R/print.survfit.R`), the work behind
/// `summary(fit)$table` and `print(fit)`.  `scale` divides the times; the
/// area under the curve starts at `fit.t0` (R's `x$t0`).
///
/// `summary.survfit` calls this on `survfit0(fit)`; `print.survfit` on the
/// fit itself; the table is the same either way.  Both check a numeric
/// `rmean` against the smallest time of the fit they were given, so does
/// this function.
pub fn survmean(
    fit: &SurvfitKMResult,
    scale: f64,
    rmean: RmeanOption,
) -> SurvivalResult<SurvmeanTable> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SurvivalError::invalid_input(
            "scale must be a positive number",
        ));
    }
    if let RmeanOption::At(value) = rmean {
        if !value.is_finite() {
            return Err(SurvivalError::invalid_input(
                "Invalid value for rmean option",
            ));
        }
        let smallest = fit.time.iter().copied().fold(f64::INFINITY, f64::min);
        if value < smallest {
            return Err(SurvivalError::invalid_input(
                "Truncation point for the mean time in state is < smallest survival",
            ));
        }
    }
    let start_time = fit.t0;
    let stime: Vec<f64> = fit.time.iter().map(|t| t / scale).collect();
    let ranges = fit.curve_ranges();
    let last_time: Vec<f64> = ranges
        .iter()
        .map(|range| stime[..range.end].last().copied().unwrap_or(f64::NAN))
        .collect();
    let end_time: Vec<f64> = match rmean {
        RmeanOption::None => vec![f64::NAN; ranges.len()],
        RmeanOption::At(value) => vec![value / scale; ranges.len()],
        RmeanOption::Common => {
            let common = last_time.iter().copied().fold(f64::NAN, f64::max);
            vec![common; ranges.len()]
        }
        RmeanOption::Individual => last_time,
    };
    let n_curves = ranges.len();
    let mut table = SurvmeanTable {
        records: Vec::with_capacity(n_curves),
        n_max: Vec::with_capacity(n_curves),
        n_start: Vec::with_capacity(n_curves),
        events: Vec::with_capacity(n_curves),
        rmean: (rmean != RmeanOption::None).then(|| Vec::with_capacity(n_curves)),
        se_rmean: (rmean != RmeanOption::None).then(|| Vec::with_capacity(n_curves)),
        median: Vec::with_capacity(n_curves),
        lower: fit.lower.as_ref().map(|_| Vec::with_capacity(n_curves)),
        upper: fit.upper.as_ref().map(|_| Vec::with_capacity(n_curves)),
        end_time: end_time.clone(),
    };
    for (curve, range) in ranges.iter().enumerate() {
        let limits = match (&fit.lower, &fit.upper) {
            (Some(lower), Some(upper)) => Some((&lower[range.clone()], &upper[range.clone()])),
            _ => None,
        };
        let row = survmean_row(
            fit.n[curve] as f64,
            &stime[range.clone()],
            &fit.surv[range.clone()],
            &fit.n_risk[range.clone()],
            &fit.n_event[range.clone()],
            limits,
            start_time,
            end_time[curve],
            fit.n_id.as_ref().map(|n_id| n_id[curve] as f64),
        );
        table.records.push(row[0]);
        table.n_max.push(row[1]);
        table.n_start.push(row[2]);
        table.events.push(row[3]);
        if let (Some(rmean), Some(se_rmean)) = (&mut table.rmean, &mut table.se_rmean) {
            rmean.push(row[4]);
            se_rmean.push(row[5]);
        }
        table.median.push(row[6]);
        if let (Some(lower), Some(upper)) = (&mut table.lower, &mut table.upper) {
            lower.push(row[7]);
            upper.push(row[8]);
        }
    }
    Ok(table)
}

/// `findInterval(t, times)`: the number of `times` that are `<= t`
/// (`left_open`: `< t`).  `times` must be sorted.
fn find_interval(times: &[f64], t: f64, left_open: bool) -> usize {
    if left_open {
        times.partition_point(|&x| x < t)
    } else {
        times.partition_point(|&x| x <= t)
    }
}

/// `summary.survfit` without a `times` argument: `censored = FALSE` keeps
/// only the rows with events, accumulating the censoring and entry counts
/// in between into the next kept row; `censored = TRUE` is the fit itself.
/// Either way `std_err` is put on the survival scale (`logse = false`).
pub fn summary_survfit(fit: &SurvfitKMResult, censored: bool) -> SurvfitKMResult {
    let mut out = fit.clone();
    if !censored {
        let ranges = fit.curve_ranges();
        let keep: Vec<usize> = (0..fit.time.len())
            .filter(|&i| fit.n_event[i] > 0.0)
            .collect();
        // the kept rows of each curve, as a range into `keep`
        let kept_ranges: Vec<std::ops::Range<usize>> = ranges
            .iter()
            .map(|range| {
                keep.partition_point(|&i| i < range.start)..keep.partition_point(|&i| i < range.end)
            })
            .collect();
        let pick = |values: &[f64]| -> Vec<f64> { keep.iter().map(|&i| values[i]).collect() };
        // sums between the kept rows: diff(c(0, c(0, cumsum(x))[indx + 1]))
        let delta = |values: &[f64]| -> Vec<f64> {
            let mut out = Vec::with_capacity(keep.len());
            for (range, kept) in ranges.iter().zip(&kept_ranges) {
                let mut next = range.start;
                for &i in &keep[kept.clone()] {
                    out.push(values[next..=i].iter().sum());
                    next = i + 1;
                }
            }
            out
        };
        out.time = pick(&fit.time);
        out.n_risk = pick(&fit.n_risk);
        out.n_event = pick(&fit.n_event);
        out.surv = pick(&fit.surv);
        out.cumhaz = pick(&fit.cumhaz);
        out.std_err = fit.std_err.as_deref().map(pick);
        out.std_chaz = fit.std_chaz.as_deref().map(pick);
        out.lower = fit.lower.as_deref().map(pick);
        out.upper = fit.upper.as_deref().map(pick);
        out.n_censor = delta(&fit.n_censor);
        out.n_enter = fit.n_enter.as_deref().map(delta);
        out.counts = fit.counts.as_ref().map(|counts| SurvfitCounts {
            n_risk: pick(&counts.n_risk),
            n_event: pick(&counts.n_event),
            n_censor: delta(&counts.n_censor),
            n_enter: counts.n_enter.as_deref().map(delta),
        });
        out.strata = fit
            .strata
            .as_ref()
            .map(|_| kept_ranges.iter().map(ExactSizeIterator::len).collect());
        out.influence_surv = None;
        out.influence_chaz = None;
    }
    out.std_err = out.std_err_surv_scale();
    out.logse = false;
    out
}

/// `summary(fit, times = )`: every curve evaluated at `times`, R's
/// `findrow` applied to `survfit0(fit)`.  With `extend = false` times past
/// a curve's last time are dropped from that curve (R errors when nothing
/// is left).  Counts between the requested times are summed when the
/// times increase (`dosum`); otherwise they are looked up.  `std_err` is
/// on the survival scale.
pub fn summary_survfit_times(
    fit: &SurvfitKMResult,
    times: &[f64],
    extend: bool,
) -> SurvivalResult<SurvfitKMResult> {
    if times.is_empty() {
        return Err(SurvivalError::invalid_input("no values in times vector"));
    }
    validate_finite(times, "times")?;
    let dosum = times.windows(2).all(|pair| pair[1] > pair[0]);
    let fit0 = survfit0(fit);
    let ranges = fit0.curve_ranges();
    let mut out = SurvfitKMResult {
        time: Vec::new(),
        n_risk: Vec::new(),
        n_event: Vec::new(),
        n_censor: Vec::new(),
        n_enter: fit0.n_enter.as_ref().map(|_| Vec::new()),
        counts: None,
        surv: Vec::new(),
        std_err: fit0.std_err.as_ref().map(|_| Vec::new()),
        cumhaz: Vec::new(),
        std_chaz: fit0.std_chaz.as_ref().map(|_| Vec::new()),
        lower: fit0.lower.as_ref().map(|_| Vec::new()),
        upper: fit0.upper.as_ref().map(|_| Vec::new()),
        strata: fit0.strata.as_ref().map(|_| Vec::new()),
        influence_surv: None,
        influence_chaz: None,
        logse: false,
        ..fit0.clone()
    };
    for range in &ranges {
        let curve_time = &fit0.time[range.clone()];
        let times: Vec<f64> = if extend {
            times.to_vec()
        } else {
            let maxtime = curve_time.iter().copied().fold(f64::NAN, f64::max);
            times.iter().copied().filter(|&t| t <= maxtime).collect()
        };
        if curve_time.is_empty() {
            return Err(SurvivalError::invalid_input(
                "no points selected for one or more curves, data error (?) or consider using the extend argument",
            ));
        }
        let index1: Vec<usize> = times
            .iter()
            .map(|&t| find_interval(curve_time, t, false))
            .collect();
        let index2: Vec<usize> = times
            .iter()
            .map(|&t| 1 + find_interval(curve_time, t, true))
            .collect();
        // x[pmax(1, index1)]
        let ssub = |values: &[f64]| -> Vec<f64> {
            index1
                .iter()
                .map(|&i| values[range.start + i.max(1) - 1])
                .collect()
        };
        // diff(c(0, c(0, cumsum(x))[index1 + 1]))
        let delta = |values: &[f64]| -> Vec<f64> {
            let mut cumulative = vec![0.0; range.len() + 1];
            for (k, i) in range.clone().enumerate() {
                cumulative[k + 1] = cumulative[k] + values[i];
            }
            let mut previous = 0.0;
            index1
                .iter()
                .map(|&i| {
                    let value = cumulative[i];
                    let out = value - previous;
                    previous = value;
                    out
                })
                .collect()
        };
        let counts =
            |values: &[f64]| -> Vec<f64> { if dosum { delta(values) } else { ssub(values) } };
        out.time.extend_from_slice(&times);
        for (target, source) in [(&mut out.surv, &fit0.surv), (&mut out.cumhaz, &fit0.cumhaz)] {
            target.extend(ssub(source));
        }
        for (target, source) in [
            (&mut out.std_err, &fit0.std_err),
            (&mut out.std_chaz, &fit0.std_chaz),
            (&mut out.lower, &fit0.lower),
            (&mut out.upper, &fit0.upper),
        ] {
            if let (Some(target), Some(source)) = (target, source) {
                target.extend(ssub(source));
            }
        }
        // every observation ends with a censor or event, so the number at
        // risk after the last observed time is 0
        out.n_risk.extend(index2.iter().map(|&i| {
            if i <= range.len() {
                fit0.n_risk[range.start + i - 1]
            } else {
                0.0
            }
        }));
        out.n_event.extend(counts(&fit0.n_event));
        out.n_censor.extend(counts(&fit0.n_censor));
        if let (Some(target), Some(source)) = (&mut out.n_enter, &fit0.n_enter) {
            target.extend(counts(source));
        }
        if let Some(strata) = &mut out.strata {
            strata.push(times.len());
        }
    }
    if fit0.logse
        && let Some(std_err) = &mut out.std_err
    {
        for (se, surv) in std_err.iter_mut().zip(&out.surv) {
            *se *= surv;
        }
    }
    Ok(out)
}

/// Quantiles of one or more survival curves with the quantiles of the
/// confidence bands (`quantile.survfit`).  Rows are curves, columns
/// `probs`; `NaN` where the curve never reaches the probability.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitQuantiles {
    #[pyo3(get)]
    pub probs: Vec<f64>,
    #[pyo3(get)]
    pub quantile: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub lower: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub upper: Option<Vec<Vec<f64>>>,
}

/// `approx(x, seq_along(x), v, method = "constant", f = 1)` on a strictly
/// increasing `x` with the `NA` points removed: the index of the first `x`
/// at or above `v`, `None` outside the range.
fn approx_constant_index(points: &[(f64, usize)], v: f64) -> Option<usize> {
    let first = points.first()?;
    if v < first.0 {
        return None;
    }
    let k = points.partition_point(|&(x, _)| x < v);
    points.get(k).map(|&(_, index)| index)
}

/// `findq` of `quantile.survfit`: where a horizontal line at `p` intersects
/// the distribution curve `y = 1 - S` at `x`, with the midpoint rule for a
/// flat exactly at `p` and the flat at the end of the curve.
fn findq(x: &[f64], y: &[f64], probs: &[f64], tol: f64) -> Vec<f64> {
    let ymax = y
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::NAN, f64::max);
    let pmin = probs.iter().copied().fold(f64::INFINITY, f64::min);
    if ymax.is_nan() || ymax < pmin {
        return vec![f64::NAN; probs.len()];
    }
    let xmax = x[x.len() - 1];
    // Remove duplicate y values, i.e. the censors, since dups cause issues
    // for approx (R's duplicated() treats NA as a repeat of NA).
    let mut xs = Vec::with_capacity(x.len());
    let mut ys = Vec::with_capacity(y.len());
    let mut seen = std::collections::HashSet::with_capacity(y.len());
    for (&xi, &yi) in x.iter().zip(y) {
        let key = if yi.is_nan() {
            f64::NAN.to_bits()
        } else {
            (yi + 0.0).to_bits()
        };
        if seen.insert(key) {
            xs.push(xi);
            ys.push(yi);
        }
    }
    let n = ys.len();
    let finite: Vec<usize> = (0..n).filter(|&i| !ys[i].is_nan()).collect();
    let plus: Vec<(f64, usize)> = finite.iter().map(|&i| (ys[i] + tol, i)).collect();
    let minus: Vec<(f64, usize)> = finite.iter().map(|&i| (ys[i] - tol, i)).collect();
    probs
        .iter()
        .map(|&p| {
            let indx1 = approx_constant_index(&plus, p);
            let indx2 = approx_constant_index(&minus, p);
            let mut quant = match (indx1, indx2) {
                (Some(i1), Some(i2)) => (xs[i1] + xs[i2]) / 2.0,
                _ => f64::NAN,
            };
            if p == 0.0 {
                quant = xs[0];
            }
            if !ys[n - 1].is_nan()
                && (p - ys[n - 1]).abs() < tol
                && let Some(i1) = indx1
            {
                // end of the curve
                quant = (xs[i1] + xmax) / 2.0;
            }
            quant
        })
        .collect()
}

/// Port of `quantile.survfit`.  `conf_int` adds the quantiles of the
/// confidence bands when the fit has them; `scale` divides the result and
/// `tolerance` (default `sqrt(.Machine$double.eps)`) decides what counts
/// as a flat exactly at a probability.
///
/// A probability of 0 reports R's `x$start.time` when the object has one
/// and 0 otherwise.  `survfitKM` records the starting time as `t0` and never
/// sets `start.time` (`survfit(..., start.time = 10)` still gives
/// `quantile(fit, probs = 0) == 0` in survival 3.8-11), so the origin of a
/// [`SurvfitKMResult`] is 0; [`quantile_survfit_from`] takes it as an
/// argument for curves that do carry one (`survfit.coxph` objects).
pub fn quantile_survfit(
    fit: &SurvfitKMResult,
    probs: &[f64],
    conf_int: bool,
    scale: f64,
    tolerance: Option<f64>,
) -> SurvivalResult<SurvfitQuantiles> {
    quantile_survfit_from(fit, probs, conf_int, 0.0, scale, tolerance)
}

/// [`quantile_survfit`] with an explicit origin: the time reported for a
/// probability of 0 (R's `x$start.time`, 0 when absent), which also heads
/// the distribution curve `findq` walks.
pub fn quantile_survfit_from(
    fit: &SurvfitKMResult,
    probs: &[f64],
    conf_int: bool,
    origin: f64,
    scale: f64,
    tolerance: Option<f64>,
) -> SurvivalResult<SurvfitQuantiles> {
    if probs.iter().any(|p| p.is_nan()) {
        return Err(SurvivalError::invalid_input("invalid probability"));
    }
    if probs.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(SurvivalError::invalid_input("Invalid probability"));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SurvivalError::invalid_input(
            "scale must be a positive number",
        ));
    }
    if !origin.is_finite() {
        return Err(SurvivalError::invalid_input("start time must be finite"));
    }
    let tol = tolerance.unwrap_or_else(r_tolerance);
    let conf_int = conf_int && fit.lower.is_some() && fit.upper.is_some();
    let xmin = origin;
    let doquant = |time: &[f64], surv: &[f64]| -> Vec<f64> {
        let mut x = Vec::with_capacity(time.len() + 1);
        x.push(xmin);
        x.extend_from_slice(time);
        let mut y = Vec::with_capacity(surv.len() + 1);
        y.push(0.0);
        y.extend(surv.iter().map(|s| 1.0 - s));
        findq(&x, &y, probs, tol)
            .into_iter()
            .map(|q| q / scale)
            .collect()
    };
    let mut out = SurvfitQuantiles {
        probs: probs.to_vec(),
        quantile: Vec::new(),
        lower: conf_int.then(Vec::new),
        upper: conf_int.then(Vec::new),
    };
    for range in fit.curve_ranges() {
        let time = &fit.time[range.clone()];
        out.quantile.push(doquant(time, &fit.surv[range.clone()]));
        if let (Some(lower), Some(upper)) = (&mut out.lower, &mut out.upper) {
            let band_lower = fit.lower.as_ref().expect("checked above");
            let band_upper = fit.upper.as_ref().expect("checked above");
            lower.push(doquant(time, &band_lower[range.clone()]));
            upper.push(doquant(time, &band_upper[range.clone()]));
        }
    }
    Ok(out)
}

/// Python binding of [`survfit0`].
#[pyfunction(name = "survfit0")]
pub fn survfit0_py(fit: &SurvfitKMResult) -> SurvfitKMResult {
    survfit0(fit)
}

/// Python binding of [`survfit0_aj`].
#[pyfunction(name = "survfit0_aj")]
pub fn survfit0_aj_py(fit: &SurvfitAJResult) -> SurvfitAJResult {
    survfit0_aj(fit)
}

/// Python binding of [`survmean`]; `rmean` is `"none"`, `"common"`,
/// `"individual"` or a number.
#[pyfunction(name = "survmean")]
#[pyo3(signature = (fit, scale=1.0, rmean="common"))]
pub fn survmean_py(fit: &SurvfitKMResult, scale: f64, rmean: &str) -> PyResult<SurvmeanTable> {
    Ok(survmean(fit, scale, RmeanOption::parse(rmean)?)?)
}

/// Python binding of [`summary_survfit`] and [`summary_survfit_times`].
#[pyfunction(name = "summary_survfit")]
#[pyo3(signature = (fit, times=None, censored=false, extend=false))]
pub fn summary_survfit_py(
    fit: &SurvfitKMResult,
    times: Option<Vec<f64>>,
    censored: bool,
    extend: bool,
) -> PyResult<SurvfitKMResult> {
    match times {
        Some(times) => Ok(summary_survfit_times(fit, &times, extend)?),
        None => Ok(summary_survfit(fit, censored)),
    }
}

/// Python binding of [`quantile_survfit`]; `probs` defaults to the
/// quartiles.
#[pyfunction(name = "quantile_survfit")]
#[pyo3(signature = (fit, probs=None, conf_int=true, scale=1.0, tolerance=None))]
pub fn quantile_survfit_py(
    fit: &SurvfitKMResult,
    probs: Option<Vec<f64>>,
    conf_int: bool,
    scale: f64,
    tolerance: Option<f64>,
) -> PyResult<SurvfitQuantiles> {
    let probs = probs.unwrap_or_else(|| vec![0.25, 0.5, 0.75]);
    Ok(quantile_survfit(fit, &probs, conf_int, scale, tolerance)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surv_analysis::survfitkm::{SurvfitKMData, SurvfitKMOptions, survfitkm};

    fn aml_maintained() -> SurvfitKMResult {
        let time = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        survfitkm(
            &SurvfitKMData::right_censored(time, status).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap()
    }

    fn aml_by_x() -> SurvfitKMResult {
        let time = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0, 5.0, 5.0, 8.0, 8.0,
            12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
        ];
        let status = vec![
            1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
        ];
        let strata = vec![1; 11].into_iter().chain(vec![2; 12]).collect();
        survfitkm(
            &SurvfitKMData::try_new(None, time, status, None, Some(strata), None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn survfit0_prepends_a_time_zero_row() {
        let fit = aml_by_x();
        let fit0 = survfit0(&fit);
        assert_eq!(fit0.strata, Some(vec![11, 11]));
        assert_eq!(fit0.time[0], 0.0);
        assert_eq!(fit0.surv[0], 1.0);
        assert_eq!(fit0.n_risk[0], 11.0);
        assert_eq!(fit0.n_event[0], 0.0);
        assert_eq!(fit0.lower.as_ref().unwrap()[0], 1.0);
        assert_eq!(fit0.std_err.as_ref().unwrap()[0], 0.0);
        assert_eq!(fit0.time[11], 0.0);
        assert_eq!(fit0.n_risk[11], 12.0);
        // idempotent
        let again = survfit0(&fit0);
        assert_eq!(again.time, fit0.time);
        assert_eq!(again.strata, fit0.strata);
    }

    #[test]
    fn survmean_matches_r_summary_table() {
        // summary(survfit(Surv(time, status) ~ x, aml))$table
        let table = survmean(&survfit0(&aml_by_x()), 1.0, RmeanOption::Common).unwrap();
        assert_eq!(table.records, vec![11.0, 12.0]);
        assert_eq!(table.n_max, vec![11.0, 12.0]);
        assert_eq!(table.n_start, vec![11.0, 12.0]);
        assert_eq!(table.events, vec![7.0, 11.0]);
        let rmean = table.rmean.as_ref().unwrap();
        assert!(close(rmean[0], 52.6454545454545, 1e-12));
        assert!(close(rmean[1], 22.7083333333333, 1e-12));
        let se = table.se_rmean.as_ref().unwrap();
        assert!(close(se[0], 19.8286027955626, 1e-10));
        assert!(close(se[1], 4.18094198103315, 1e-10));
        assert_eq!(table.median, vec![31.0, 23.0]);
        assert_eq!(table.lower.as_ref().unwrap(), &[18.0, 8.0]);
        assert!(table.upper.as_ref().unwrap().iter().all(|v| v.is_nan()));
        assert_eq!(table.end_time, vec![161.0, 161.0]);
        let none = survmean(&aml_by_x(), 1.0, RmeanOption::None).unwrap();
        assert!(none.rmean.is_none());
        assert!(none.end_time.iter().all(|v| v.is_nan()));
        let individual = survmean(&aml_by_x(), 1.0, RmeanOption::Individual).unwrap();
        assert_eq!(individual.end_time, vec![161.0, 45.0]);
        assert!(survmean(&aml_by_x(), 1.0, RmeanOption::At(1.0)).is_err());
    }

    #[test]
    fn survmean_with_a_truncation_time_matches_r() {
        // summary(fit, rmean = 24)$table["rmean"] for the maintained arm
        let table = survmean(&survfit0(&aml_maintained()), 1.0, RmeanOption::At(24.0)).unwrap();
        let rmean = table.rmean.as_ref().unwrap()[0];
        // 9 + 4 * 10/11 + 5 * 9/11 + 5 * 0.7159 + 1 * 0.6136
        assert!(close(
            rmean,
            9.0 + 40.0 / 11.0 + 45.0 / 11.0 + 5.0 * 0.715909090909091 + 0.613636363636364,
            1e-10
        ));
        assert_eq!(table.end_time, vec![24.0]);
    }

    #[test]
    fn summary_at_times_matches_r() {
        // summary(fit, times = c(0, 10, 20, 30, 50, 100, 200), extend = TRUE)
        let out = summary_survfit_times(
            &aml_by_x(),
            &[0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0],
            true,
        )
        .unwrap();
        assert_eq!(out.strata, Some(vec![7, 7]));
        assert_eq!(&out.time[..7], &[0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0]);
        assert_eq!(&out.n_risk[..7], &[11.0, 10.0, 7.0, 5.0, 1.0, 1.0, 0.0]);
        assert_eq!(&out.n_event[..7], &[0.0, 1.0, 2.0, 1.0, 3.0, 0.0, 0.0]);
        assert_eq!(&out.n_censor[..7], &[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
        assert!(close(out.surv[1], 0.909090909090909, 1e-12));
        assert!(close(out.surv[4], 0.184090909090909, 1e-12));
        assert!(!out.logse);
        let std_err = out.std_err.as_ref().unwrap();
        assert_eq!(std_err[0], 0.0);
        assert!(close(std_err[1], 0.0866784172041448, 1e-10));
        assert!(std_err[11].is_nan()); // S = 0 with an infinite se(log S)
        assert_eq!(&out.n_risk[7..], &[12.0, 8.0, 6.0, 4.0, 0.0, 0.0, 0.0]);
        // extend = FALSE drops the times past the end of each curve
        let short = summary_survfit_times(&aml_by_x(), &[10.0, 200.0], false).unwrap();
        assert_eq!(short.strata, Some(vec![1, 1]));
        assert!(summary_survfit_times(&aml_by_x(), &[], true).is_err());
    }

    #[test]
    fn summary_without_times_keeps_event_rows() {
        let fit = aml_maintained();
        let out = summary_survfit(&fit, false);
        assert_eq!(out.time, vec![9.0, 13.0, 18.0, 23.0, 31.0, 34.0, 48.0]);
        // censorings tied with an event stay on its row, later ones are
        // carried to the next event row and those after the last are dropped
        assert_eq!(out.n_censor, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let se = out.std_err.as_ref().unwrap();
        assert!(close(se[0], 0.0866784172041448, 1e-10));
        assert!(!out.logse);
        let all = summary_survfit(&fit, true);
        assert_eq!(all.time, fit.time);
    }

    #[test]
    fn quantiles_match_r() {
        // quantile(survfit(Surv(time, status) ~ x, aml))
        let q = quantile_survfit(&aml_by_x(), &[0.25, 0.5, 0.75], true, 1.0, None).unwrap();
        assert_eq!(
            q.quantile,
            vec![vec![18.0, 31.0, 48.0], vec![8.0, 23.0, 33.0]]
        );
        assert_eq!(
            q.lower.as_ref().unwrap(),
            &vec![vec![13.0, 18.0, 34.0], vec![5.0, 8.0, 27.0]]
        );
        let upper = q.upper.as_ref().unwrap();
        assert!(upper[0].iter().all(|v| v.is_nan()));
        assert_eq!(upper[1][0], 30.0);
        assert!(upper[1][1].is_nan());
        let plain = quantile_survfit(&aml_by_x(), &[0.0, 0.5], false, 2.0, None).unwrap();
        assert!(plain.lower.is_none());
        assert_eq!(plain.quantile[0], vec![0.0, 15.5]);
        assert!(quantile_survfit(&aml_by_x(), &[1.5], true, 1.0, None).is_err());
    }

    #[test]
    fn quantile_midpoint_rule_for_a_flat_at_the_probability() {
        // S = 0.5 on [2, 4): the median is the midpoint 3
        let fit = survfitkm(
            &SurvfitKMData::right_censored(vec![1.0, 2.0, 4.0, 5.0], vec![1, 1, 1, 1]).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();
        let q = quantile_survfit(&fit, &[0.5, 0.75, 0.9, 1.0], false, 1.0, None).unwrap();
        assert_eq!(q.quantile[0][0], 3.0);
        assert_eq!(q.quantile[0][1], 4.5);
        assert_eq!(q.quantile[0][2], 5.0);
        assert_eq!(q.quantile[0][3], 5.0);
        let table = survmean(&survfit0(&fit), 1.0, RmeanOption::None).unwrap();
        assert_eq!(table.median, vec![3.0]);
    }
}
