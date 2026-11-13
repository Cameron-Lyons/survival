use super::survfit_summary::{RmeanOption, survfit0_aj, survfit0_aj_rows};
use super::survfitaj::SurvfitAJResult;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_finite;

pub type AJMeanTable = (Vec<Vec<f64>>, Vec<f64>, Vec<String>);

pub fn survmean_aj(
    fit: &SurvfitAJResult,
    scale: f64,
    rmean: RmeanOption,
) -> SurvivalResult<AJMeanTable> {
    let fit0 = survfit0_aj(fit);
    survmean_table(&fit0, 1, |i, _, state| fit0.pstate[i][state], scale, rmean)
}

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

pub(crate) fn time_in_state(
    fit0: &SurvfitAJResult,
    range: std::ops::Range<usize>,
    end: f64,
    state: usize,
) -> f64 {
    area_to(&fit0.time, range, end, |i| fit0.pstate[i][state])
}

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

#[derive(Clone, Copy)]
enum SummaryRow {
    Observed(usize),
    Initial(usize),
}

struct SummaryIndex {
    selection: Vec<SummaryRow>,
    risk_selection: Vec<Option<usize>>,
    intervals: Vec<std::ops::Range<usize>>,
    output_times: Vec<f64>,
    sizes: Vec<usize>,
}

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
    for (curve, range) in source.curve_ranges().into_iter().enumerate() {
        let before = index.selection.len();
        let mut previous = range.start;
        if let Some(times) = requested {
            let observed = &source.time[range.clone()];
            let initial = observed.first().is_none_or(|&time| time != source.t0);
            let last = observed.last().copied().unwrap_or(source.t0);
            for &time in times {
                if !extend && time > last {
                    continue;
                }
                let right = range.start + observed.partition_point(|&value| value <= time);
                index.selection.push(if right > range.start {
                    SummaryRow::Observed(right - 1)
                } else if initial {
                    SummaryRow::Initial(curve)
                } else {
                    SummaryRow::Observed(range.start)
                });
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
                    index.selection.push(SummaryRow::Observed(i));
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

pub(crate) fn summary_rows(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SurvivalResult<Vec<i64>> {
    let requested = summary_times(times)?;
    Ok(summary_index(fit, requested.as_deref(), censored, extend)
        .selection
        .into_iter()
        .map(|row| match row {
            SummaryRow::Observed(i) => i as i64,
            SummaryRow::Initial(curve) => -1 - curve as i64,
        })
        .collect())
}

pub fn summary_survfit_aj(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
) -> SurvivalResult<SurvfitAJResult> {
    summary_survfit_aj_with_counts(fit, times, censored, extend, true)
}

pub fn summary_survfit_aj_with_counts(
    fit: &SurvfitAJResult,
    times: Option<&[f64]>,
    censored: bool,
    extend: bool,
    dosum: bool,
) -> SurvivalResult<SurvfitAJResult> {
    let requested = summary_times(times)?;
    let SummaryIndex {
        selection,
        risk_selection,
        intervals,
        output_times,
        sizes,
    } = summary_index(fit, requested.as_deref(), censored, extend);
    // The initial row has zero counts, hazards and standard errors. Select it
    // virtually so a sparse summary never copies the full fitted time grid.
    let pick = |values: &[Vec<f64>], width: usize| -> Vec<Vec<f64>> {
        selection
            .iter()
            .map(|&row| match row {
                SummaryRow::Observed(i) => values[i].clone(),
                SummaryRow::Initial(_) => vec![0.0; width],
            })
            .collect()
    };
    let sum = |values: &[Vec<f64>], width: usize| -> Vec<Vec<f64>> {
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
    let counts = |values: &[Vec<f64>], width: usize| {
        if requested.is_some() && !dosum {
            pick(values, width)
        } else {
            sum(values, width)
        }
    };
    let ns = fit.states.len();
    let nh = fit.hazard_from.len();
    let pick_states = |values: &[Vec<f64>]| pick(values, ns);
    Ok(SurvfitAJResult {
        n: fit.n.clone(),
        time: output_times,
        n_risk: risk_selection
            .iter()
            .map(|index| index.map_or_else(|| vec![0.0; ns], |i| fit.n_risk[i].clone()))
            .collect(),
        n_event: if requested.is_some() {
            counts(&fit.n_event, ns)
        } else {
            pick(&fit.n_event, ns)
        },
        n_censor: counts(&fit.n_censor, ns),
        n_enter: fit.n_enter.as_ref().map(|values| counts(values, ns)),
        n_transition: counts(&fit.n_transition, nh),
        counts: None,
        pstate: selection
            .iter()
            .map(|&row| match row {
                SummaryRow::Observed(i) => fit.pstate[i].clone(),
                SummaryRow::Initial(curve) => fit.p0[curve].clone(),
            })
            .collect(),
        cumhaz: pick(&fit.cumhaz, nh),
        std_err: fit.std_err.as_deref().map(pick_states),
        std_chaz: fit.std_chaz.as_deref().map(|values| pick(values, nh)),
        std_auc: fit.std_auc.as_deref().map(pick_states),
        se0: fit.se0.clone(),
        lower: fit.lower.as_deref().map(pick_states),
        upper: fit.upper.as_deref().map(pick_states),
        p0: fit.p0.clone(),
        strata: fit.strata.as_ref().map(|_| sizes),
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
        t0: fit.t0,
        start_time: fit.start_time,
        influence_pstate: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surv_analysis::{SurvfitAJData, SurvfitAJOptions, survfitaj};

    fn counting_fit() -> SurvfitAJResult {
        let data = SurvfitAJData::try_new(
            Some(vec![0., 1., 0., 2., 0., 0., 3., 0.]),
            vec![1., 4., 2., 5., 3., 6., 6., 7.],
            vec![1, 2, 1, 0, 2, 0, 1, 0],
            vec!["a".into(), "b".into()],
            None,
            Some(vec![0, 0, 1, 1, 0, 1, 0, 1]),
            Some(vec![1, 1, 2, 2, 3, 4, 5, 6]),
            Some(
                [
                    "healthy", "a", "healthy", "a", "healthy", "healthy", "healthy", "healthy",
                ]
                .map(String::from)
                .to_vec(),
            ),
            None,
            None,
        )
        .unwrap();
        survfitaj(
            &data,
            &SurvfitAJOptions {
                entry: true,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn rows<const N: usize>(values: [[f64; 3]; N]) -> Vec<Vec<f64>> {
        values.map(Vec::from).to_vec()
    }

    #[test]
    fn requested_multistate_counts_sample_preceding_rows_when_dosum_is_false() {
        let fit = counting_fit();
        let out =
            summary_survfit_aj_with_counts(&fit, Some(&[0., 2., 6., 8., 2.]), false, true, false)
                .unwrap();
        assert_eq!(out.time, vec![0., 2., 6., 8., 0., 2., 6., 8.]);
        assert_eq!(out.strata, Some(vec![4, 4]));
        assert_eq!(
            out.n_risk,
            rows([
                [0., 0., 0.],
                [1., 1., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [3., 0., 0.],
                [2., 0., 0.],
                [0., 0., 0.],
            ])
        );
        assert_eq!(
            out.n_event,
            rows([
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ])
        );
        assert_eq!(
            out.n_censor,
            rows([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
            ])
        );
        assert_eq!(
            out.n_enter,
            Some(rows([
                [2., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [3., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ]))
        );
        assert_eq!(
            out.n_transition,
            rows([
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ])
        );
        let accumulated =
            summary_survfit_aj(&fit, Some(&[0., 2., 6., 8., 2.]), false, true).unwrap();
        assert_eq!(out.pstate, accumulated.pstate);
        assert_eq!(out.cumhaz, accumulated.cumhaz);
        assert_eq!(out.std_err, accumulated.std_err);
        assert_eq!(out.lower, accumulated.lower);
        assert_eq!(out.upper, accumulated.upper);
    }

    #[test]
    fn requested_multistate_counts_accumulate_by_default() {
        let fit = counting_fit();
        let out = summary_survfit_aj(&fit, Some(&[0., 2., 6., 8.]), false, true).unwrap();
        assert_eq!(
            out.n_event,
            rows([
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 1., 2.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ])
        );
        assert_eq!(
            out.n_censor,
            rows([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [1., 1., 0.],
                [1., 0., 0.],
            ])
        );
        assert_eq!(
            out.n_enter,
            Some(rows([
                [2., 0., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [3., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ]))
        );
        assert_eq!(
            out.n_transition,
            rows([
                [0., 0., 0.],
                [1., 0., 0.],
                [1., 1., 1.],
                [0., 0., 0.],
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.],
            ])
        );
    }

    #[test]
    fn dosum_does_not_change_multistate_summaries_without_requested_times() {
        let fit = counting_fit();
        for censored in [false, true] {
            let accumulated = summary_survfit_aj(&fit, None, censored, false).unwrap();
            let sampled =
                summary_survfit_aj_with_counts(&fit, None, censored, false, false).unwrap();
            assert_eq!(sampled.time, accumulated.time);
            assert_eq!(sampled.n_event, accumulated.n_event);
            assert_eq!(sampled.n_censor, accumulated.n_censor);
            assert_eq!(sampled.n_enter, accumulated.n_enter);
            assert_eq!(sampled.n_transition, accumulated.n_transition);
        }
    }

    #[test]
    fn requested_summaries_match_explicit_origins_without_changing_the_fit() {
        let data = SurvfitAJData::try_new(
            None,
            vec![1., 2., 3., 4., 5., 6., 7., 8.],
            vec![1, 0, 2, 0, 2, 1, 0, 0],
            vec!["a".into(), "b".into()],
            Some(vec![0.5, 1., 2., 1., 1.5, 1., 0.5, 1.]),
            Some(vec![-2, 7, -2, 7, -2, 7, -2, 7]),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        for se_fit in [false, true] {
            let fit = survfitaj(
                &data,
                &SurvfitAJOptions {
                    start_time: Some(0.5),
                    se_fit,
                    influence: se_fit,
                    ..Default::default()
                },
            )
            .unwrap();
            let selected = fit.select_curves(&[1, 0, 1]).unwrap();
            for source in [&fit, &selected, &fit.select_states(&[2, 0]).unwrap()] {
                let original = source.clone();
                let with_origin = survfit0_aj(source);
                for extend in [false, true] {
                    for dosum in [false, true] {
                        let times = [9., 0., 0.5, 1., 2.5, 7., 9.];
                        let summary = summary_survfit_aj_with_counts(
                            source,
                            Some(&times),
                            false,
                            extend,
                            dosum,
                        )
                        .unwrap();
                        let explicit = summary_survfit_aj_with_counts(
                            &with_origin,
                            Some(&times),
                            false,
                            extend,
                            dosum,
                        )
                        .unwrap();
                        assert_eq!(summary, explicit);
                        assert!(summary.influence_pstate.is_none());
                        assert!(summary.counts.is_none());
                        assert_eq!(*source, original);
                    }
                }
            }
        }
    }

    #[test]
    fn requested_summaries_of_empty_curves_keep_the_initial_state_and_dimensions() {
        let data = SurvfitAJData::try_new(
            None,
            vec![1., 2.],
            vec![0, 0],
            vec!["a".into()],
            None,
            Some(vec![0, 1]),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let fit = survfitaj(&data, &SurvfitAJOptions::default()).unwrap();
        let empty = summary_survfit_aj(&fit, None, false, false).unwrap();
        assert!(empty.time.is_empty());
        assert_eq!(empty.strata, Some(vec![0, 0]));
        for extend in [false, true] {
            for dosum in [false, true] {
                let times = [-1., 0., 1.];
                let summary =
                    summary_survfit_aj_with_counts(&empty, Some(&times), false, extend, dosum)
                        .unwrap();
                let explicit = summary_survfit_aj_with_counts(
                    &survfit0_aj(&empty),
                    Some(&times),
                    false,
                    extend,
                    dosum,
                )
                .unwrap();
                assert_eq!(summary, explicit);
                for state in &summary.pstate {
                    assert_eq!(state, &empty.p0[0]);
                }
                assert!(summary.n_risk.iter().all(|row| row == &[0., 0.]));
                assert!(summary.n_event.iter().all(|row| row == &[0., 0.]));
            }
        }
    }
}
