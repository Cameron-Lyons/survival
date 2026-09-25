//! Multistate summary rows and restricted mean time in each state.
//! Mirrors `summary.survfitms` and `survmean2`, with one scan per curve.

use super::survfit_summary::{RmeanOption, survfit0_aj};
use super::survfitaj::SurvfitAJResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;
use pyo3::prelude::*;

/// State-major table rows (`n`, `nevent`, optionally `rmean`, `se(rmean)`)
/// and truncation times. Within each state the curves vary fastest.
pub type AJMeanTable = (Vec<Vec<f64>>, Vec<f64>, Vec<String>);

pub fn survmean_aj(
    fit: &SurvfitAJResult,
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
    let fit = survfit0_aj(fit);
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
    let mut rows = Vec::with_capacity(ranges.len() * fit.states.len());
    for state in 0..fit.states.len() {
        for (curve, range) in ranges.iter().enumerate() {
            let mut row = vec![
                fit.n[curve] as f64,
                range.clone().map(|i| fit.n_event[i][state]).sum(),
            ];
            if mean {
                let end = ends[curve];
                let area: f64 = range
                    .clone()
                    .map(|i| {
                        let next = if i + 1 < range.end {
                            fit.time[i + 1].min(end)
                        } else {
                            end
                        };
                        (next - fit.time[i]).max(0.0) * fit.pstate[i][state]
                    })
                    .sum();
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

/// Select step-function values and accumulate counts between reporting times.
/// Risk counts use the next observation, probabilities use the preceding one.
pub fn summary_survfit_aj(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SurvivalResult<SurvfitAJResult> {
    let requested = times
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
        .transpose()?;
    let source = if requested.is_some() {
        survfit0_aj(fit)
    } else {
        fit.clone()
    };
    let mut out = source.clone();
    let mut selection = Vec::new();
    let mut risk_selection = Vec::new();
    let mut intervals = Vec::new();
    let mut output_times = Vec::new();
    let mut sizes = Vec::new();
    for range in source.curve_ranges() {
        let before = selection.len();
        let mut previous = range.start;
        if let Some(times) = &requested {
            let observed = &source.time[range.clone()];
            for &time in times {
                if observed.is_empty() || (!extend && time > observed[observed.len() - 1]) {
                    continue;
                }
                let right = range.start + observed.partition_point(|&value| value <= time);
                selection.push(right.saturating_sub(1).max(range.start));
                let risk = range.start + observed.partition_point(|&value| value < time);
                risk_selection.push((risk < range.end).then_some(risk));
                intervals.push(previous..right);
                previous = right;
                output_times.push(time);
            }
        } else {
            for i in range.clone() {
                if censored || source.n_event[i].iter().any(|&value| value > 0.0) {
                    selection.push(i);
                    risk_selection.push(Some(i));
                    intervals.push(previous..i + 1);
                    previous = i + 1;
                    output_times.push(source.time[i]);
                }
            }
        }
        sizes.push(selection.len() - before);
    }
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

#[pymethods]
impl SurvfitAJResult {
    #[pyo3(signature=(times=None, censored=false, extend=false))]
    fn summary(
        &self,
        py: Python<'_>,
        times: Option<Vec<f64>>,
        censored: bool,
        extend: bool,
    ) -> PyResult<Self> {
        py.detach(|| summary_survfit_aj(self, times.as_deref(), censored, extend))
            .map_err(Into::into)
    }
    #[pyo3(signature=(scale=1.0, rmean="common"))]
    fn mean_table(&self, py: Python<'_>, scale: f64, rmean: &str) -> PyResult<AJMeanTable> {
        let option = RmeanOption::parse(rmean)?;
        py.detach(|| survmean_aj(self, scale, option))
            .map_err(Into::into)
    }
}
