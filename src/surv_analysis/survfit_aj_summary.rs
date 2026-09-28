//! Multistate summary rows and restricted mean time in each state.
//! Mirrors `summary.survfitms` and `survmean2`, with one scan per curve.

use super::survfit_summary::{RmeanOption, survfit0_aj, survfit0_aj_rows};
use super::survfitaj::SurvfitAJResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;

/// State-major table rows (`n`, `nevent`, optionally `rmean`, `se(rmean)`)
/// and truncation times. Within each state the curves vary fastest.
pub type AJMeanTable = (Vec<Vec<f64>>, Vec<f64>, Vec<String>);

pub fn survmean_aj(
    fit: &SurvfitAJResult,
    scale: f64,
    rmean: RmeanOption,
) -> SurvivalResult<AJMeanTable> {
    let fit0 = survfit0_aj(fit);
    survmean_table(&fit0, 1, |i, _, state| fit0.pstate[i][state], scale, rmean)
}

/// `survmean2` of a multi-state Cox curve set (`survfit.coxphms`) whose
/// counts and time grid are those of `fit`: `pstate(i, j, s)` is the
/// probability of state `s` for newdata row `j` at row `i` of `fit`, and
/// `p0` the curves' starting distributions.  The rows are those of
/// `survfit0` (the `t0` rows take `p0`); the table's rows run over the
/// strata fastest, then the `ndata` newdata rows, then the states.
pub(crate) fn survmean_coxms(
    fit: &SurvfitAJResult,
    ndata: usize,
    pstate: impl Fn(usize, usize, usize) -> f64,
    p0: &[Vec<f64>],
    scale: f64,
    rmean: RmeanOption,
) -> SurvivalResult<AJMeanTable> {
    let rows = survfit0_aj_rows(fit);
    let fit0 = survfit0_aj(fit);
    survmean_table(
        &fit0,
        ndata,
        |i, j, state| match usize::try_from(rows[i]) {
            Ok(row) => pstate(row, j, state),
            Err(_) => p0[(-1 - rows[i]) as usize][state],
        },
        scale,
        rmean,
    )
}

/// `survmean2` on `fit0`, a `survfit0` fit with `ndata` probability curves
/// per curve of its counts, read by `pstate(row, data, state)`.
fn survmean_table(
    fit0: &SurvfitAJResult,
    ndata: usize,
    pstate: impl Fn(usize, usize, usize) -> f64,
    scale: f64,
    rmean: RmeanOption,
) -> SurvivalResult<AJMeanTable> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SurvivalError::invalid_input(
            "scale must be finite and positive",
        ));
    }
    if let RmeanOption::At(time) = rmean {
        validate_finite(&[time], "rmean")?;
    }
    let fit = fit0;
    let ranges = fit.curve_ranges();
    let max_time = fit.time.iter().copied().fold(fit.t0, f64::max);
    let ends: Vec<f64> = ranges
        .iter()
        .map(|range| match rmean {
            RmeanOption::None => f64::NAN,
            RmeanOption::Common => max_time,
            RmeanOption::Individual => fit.time[range.end - 1],
            RmeanOption::At(time) => time,
        })
        .collect();
    if ends
        .iter()
        .any(|&end| end < fit.start_time.unwrap_or(fit.t0))
    {
        return Err(SurvivalError::invalid_input(
            "truncation time precedes the start of the curve",
        ));
    }
    let mut columns = vec!["n".into(), "nevent".into()];
    let mean = !matches!(rmean, RmeanOption::None);
    if mean {
        columns.push("rmean".into());
        if fit.std_auc.is_some() {
            columns.push("se(rmean)".into());
        }
    }
    let mut rows = Vec::with_capacity(ranges.len() * ndata * fit.states.len());
    for state in 0..fit.states.len() {
        for data in 0..ndata {
            for (curve, range) in ranges.iter().enumerate() {
                let mut row = vec![
                    fit.n[curve] as f64,
                    range.clone().map(|i| fit.n_event[i][state]).sum(),
                ];
                if mean {
                    let end = ends[curve];
                    let area = area_to(&fit.time, range.clone(), end, |i| pstate(i, data, state));
                    row.push(area / scale);
                    if let Some(auc) = &fit.std_auc {
                        let times = &fit.time[range.clone()];
                        let hi = times.partition_point(|&time| time < end);
                        let se = if hi == 0 {
                            auc[range.start][state]
                        } else if hi == times.len() {
                            auc[range.end - 1][state]
                        } else {
                            let left = range.start + hi - 1;
                            let fraction =
                                (end - fit.time[left]) / (fit.time[left + 1] - fit.time[left]);
                            auc[left][state] + fraction * (auc[left + 1][state] - auc[left][state])
                        };
                        row.push(se / scale);
                    }
                }
                rows.push(row);
            }
        }
    }
    let mut ends = if mean {
        ends.into_iter().map(|end| end / scale).collect()
    } else {
        Vec::new()
    };
    if ends
        .first()
        .is_some_and(|first| ends.iter().all(|end| end == first))
    {
        ends.truncate(1);
    }
    Ok((rows, ends, columns))
}

/// `survmean2`'s restricted mean time in `state` up to `end` for the rows
/// `range` of one curve of a `survfit0_aj` fit: the area under the state's
/// probability curve from the curve's first time to `end`.
pub(crate) fn time_in_state(
    fit0: &SurvfitAJResult,
    range: std::ops::Range<usize>,
    end: f64,
    state: usize,
) -> f64 {
    area_to(&fit0.time, range, end, |i| fit0.pstate[i][state])
}

/// The area under the step function `value(row)` on the rows `range` of
/// `time`, from the first of them to `end`.
fn area_to(
    time: &[f64],
    range: std::ops::Range<usize>,
    end: f64,
    value: impl Fn(usize) -> f64,
) -> f64 {
    range
        .clone()
        .map(|i| {
            let next = if i + 1 < range.end {
                time[i + 1].min(end)
            } else {
                end
            };
            (next - time[i]).max(0.0) * value(i)
        })
        .sum()
}

/// The rows `summary.survfitms` reports, for a fit `source` that is the
/// `survfit0` of the curves when `times` are requested: `selection` are the
/// rows whose probabilities and hazards it reports (R's `ssub` rows or
/// `index1`), `risk_selection` those of `n.risk` (`None` past the end of a
/// curve), `intervals` the rows whose counts it sums into each reported row,
/// `output_times` the reported times and `sizes` the reported rows of each
/// curve.
struct SummaryIndex {
    selection: Vec<usize>,
    risk_selection: Vec<Option<usize>>,
    intervals: Vec<std::ops::Range<usize>>,
    output_times: Vec<f64>,
    sizes: Vec<usize>,
}

/// `times` sorted and made unique, after R's checks.
fn summary_times(times: Option<&[f64]>) -> SurvivalResult<Option<Vec<f64>>> {
    times
        .map(|times| {
            if times.is_empty() {
                return Err(SurvivalError::invalid_input("no values in times vector"));
            }
            validate_finite(times, "times")?;
            let mut values = times.to_vec();
            values.sort_by(f64::total_cmp);
            values.dedup();
            Ok(values)
        })
        .transpose()
}

/// See [`SummaryIndex`]; `requested` are the [`summary_times`].
fn summary_index(
    source: &SurvfitAJResult,
    requested: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SummaryIndex {
    let mut index = SummaryIndex {
        selection: Vec::new(),
        risk_selection: Vec::new(),
        intervals: Vec::new(),
        output_times: Vec::new(),
        sizes: Vec::new(),
    };
    for range in source.curve_ranges() {
        let before = index.selection.len();
        let mut previous = range.start;
        if let Some(times) = requested {
            let observed = &source.time[range.clone()];
            for &time in times {
                if observed.is_empty() || (!extend && time > observed[observed.len() - 1]) {
                    continue;
                }
                let right = range.start + observed.partition_point(|&value| value <= time);
                index
                    .selection
                    .push(right.saturating_sub(1).max(range.start));
                let risk = range.start + observed.partition_point(|&value| value < time);
                index
                    .risk_selection
                    .push((risk < range.end).then_some(risk));
                index.intervals.push(previous..right);
                previous = right;
                index.output_times.push(time);
            }
        } else {
            for i in range.clone() {
                if censored || source.n_event[i].iter().any(|&value| value > 0.0) {
                    index.selection.push(i);
                    index.risk_selection.push(Some(i));
                    index.intervals.push(previous..i + 1);
                    previous = i + 1;
                    index.output_times.push(source.time[i]);
                }
            }
        }
        index.sizes.push(index.selection.len() - before);
    }
    index
}

/// [`summary_index`]'s `selection` for the curves `fit`, in the encoding of
/// [`survfit0_aj_rows`]: a row of `fit`, or `-1 - s` for the row at `t0`
/// that `survfit0_aj` inserts into curve `s` (only when `times` are given).
pub(crate) fn summary_rows(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SurvivalResult<Vec<i64>> {
    let requested = summary_times(times)?;
    Ok(match &requested {
        Some(times) => {
            let rows = survfit0_aj_rows(fit);
            summary_index(&survfit0_aj(fit), Some(times), censored, extend)
                .selection
                .into_iter()
                .map(|i| rows[i])
                .collect()
        }
        None => summary_index(fit, None, censored, extend)
            .selection
            .into_iter()
            .map(|i| i as i64)
            .collect(),
    })
}

/// Select step-function values and accumulate counts between reporting times.
/// Risk counts use the next observation, probabilities use the preceding one.
pub fn summary_survfit_aj(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SurvivalResult<SurvfitAJResult> {
    let requested = summary_times(times)?;
    let source = if requested.is_some() {
        survfit0_aj(fit)
    } else {
        fit.clone()
    };
    let mut out = source.clone();
    let SummaryIndex {
        selection,
        risk_selection,
        intervals,
        output_times,
        sizes,
    } = summary_index(&source, requested.as_deref(), censored, extend);
    let pick = |values: &[Vec<f64>]| selection.iter().map(|&i| values[i].clone()).collect();
    let sum = |values: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let width = values.first().map_or(0, Vec::len);
        intervals
            .iter()
            .map(|range| {
                let mut totals = vec![0.0; width];
                for i in range.clone() {
                    for (total, value) in totals.iter_mut().zip(&values[i]) {
                        *total += value;
                    }
                }
                totals
            })
            .collect()
    };
    out.time = output_times;
    out.strata = source.strata.as_ref().map(|_| sizes);
    out.n_risk = risk_selection
        .iter()
        .map(|index| {
            index.map_or_else(
                || vec![0.0; source.states.len()],
                |i| source.n_risk[i].clone(),
            )
        })
        .collect();
    out.n_event = if requested.is_some() {
        sum(&source.n_event)
    } else {
        pick(&source.n_event)
    };
    out.n_censor = sum(&source.n_censor);
    out.n_enter = source.n_enter.as_ref().map(|values| sum(values));
    out.n_transition = sum(&source.n_transition);
    out.pstate = pick(&source.pstate);
    out.cumhaz = pick(&source.cumhaz);
    out.std_err = source.std_err.as_ref().map(|values| pick(values));
    out.std_chaz = source.std_chaz.as_ref().map(|values| pick(values));
    out.lower = source.lower.as_ref().map(|values| pick(values));
    out.upper = source.upper.as_ref().map(|values| pick(values));
    out.std_auc = source.std_auc.as_ref().map(|values| pick(values));
    // These raw matrices have different time dimensions; summary rows do not
    // represent a refitted model and must not retain stale influence arrays.
    out.influence_pstate = None;
    out.counts = None;
    Ok(out)
}
