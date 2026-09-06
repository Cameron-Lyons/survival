use crate::constants::{CONCORDANCE_COUNT_SIZE, PARALLEL_THRESHOLD_LARGE};
use crate::data_prep::aeq_surv;
use crate::internal::fenwick::FenwickTree;
use crate::internal::statistical::{ConcordanceSummary, ConcordanceTimeWeight};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::BTreeMap;

type ConcordanceRankRows = Vec<(f64, f64, f64, f64)>;
type ConcordanceInfluenceOutput = (Vec<Vec<f64>>, Vec<f64>, f64);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ConcordanceTieCounts {
    tied_x: f64,
    tied_y: f64,
    tied_xy: f64,
}

fn validate_right_concordance_inputs(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if time.len() != status.len() || time.len() != risk_scores.len() {
        return Err(PyValueError::new_err(
            "time, status, and risk_scores must have the same length",
        ));
    }
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_no_nan(risk_scores, "risk_scores")?;
    validate_finite(risk_scores, "risk_scores")?;
    validate_binary_i32(status, "status")?;
    if let Some(values) = weights {
        if values.len() != time.len() {
            return Err(PyValueError::new_err(
                "weights must have the same length as time",
            ));
        }
        validate_no_nan(values, "weights")?;
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
    }
    Ok(())
}

fn validate_counting_concordance_inputs(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if start.len() != stop.len() || start.len() != status.len() || start.len() != risk_scores.len()
    {
        return Err(PyValueError::new_err(
            "start, stop, status, and risk_scores must have the same length",
        ));
    }
    validate_no_nan(start, "start")?;
    validate_finite(start, "start")?;
    validate_no_nan(stop, "stop")?;
    validate_finite(stop, "stop")?;
    validate_no_nan(risk_scores, "risk_scores")?;
    validate_finite(risk_scores, "risk_scores")?;
    validate_binary_i32(status, "status")?;
    if let Some(values) = weights {
        if values.len() != start.len() {
            return Err(PyValueError::new_err(
                "weights must have the same length as start",
            ));
        }
        validate_no_nan(values, "weights")?;
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
    }

    for (idx, (&entry, &exit)) in start.iter().zip(stop.iter()).enumerate() {
        if entry >= exit {
            return Err(PyValueError::new_err(format!(
                "start must be less than stop for observation {}",
                idx
            )));
        }
    }
    Ok(())
}

fn validate_strata_length(n: usize, strata: &[i32], response_name: &str) -> PyResult<()> {
    if strata.len() != n {
        return Err(PyValueError::new_err(format!(
            "strata must have the same length as {response_name}"
        )));
    }
    Ok(())
}

fn strata_groups(strata: &[i32]) -> Vec<Vec<usize>> {
    let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, &stratum) in strata.iter().enumerate() {
        groups.entry(stratum).or_default().push(idx);
    }
    groups.into_values().collect()
}

// Correct floating-point ties once, globally, before splitting strata or clipping
// exit times. Every sweep below then uses exact equality and open start bounds.
fn validate_concordance_horizons(ymin: Option<f64>, ymax: Option<f64>) -> PyResult<()> {
    for (name, bound) in [("ymin", ymin), ("ymax", ymax)] {
        if bound.is_some_and(f64::is_nan) {
            return Err(PyValueError::new_err(format!("{name} must not be NaN")));
        }
    }
    Ok(())
}

fn clip_concordance_times(time: &mut [f64], ymin: Option<f64>) {
    if let Some(lower) = ymin {
        for value in time {
            *value = value.max(lower);
        }
    }
}

fn prepare_right_concordance_times(
    time: Vec<f64>,
    timefix: bool,
    ymin: Option<f64>,
) -> PyResult<Vec<f64>> {
    let mut time = if timefix {
        aeq_surv(time, None)?.time
    } else {
        time
    };
    clip_concordance_times(&mut time, ymin);
    Ok(time)
}

fn prepare_counting_concordance_times(
    mut start: Vec<f64>,
    mut stop: Vec<f64>,
    timefix: Option<bool>,
    ymin: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if timefix.unwrap_or(true) {
        let n = start.len();
        start.extend_from_slice(&stop);
        start = aeq_surv(start, None)?.time;
        stop = start.split_off(n);
        if let Some(idx) = start
            .iter()
            .zip(&stop)
            .position(|(entry, exit)| entry >= exit)
        {
            return Err(PyValueError::new_err(format!(
                "aeqSurv exception: interval at observation {idx} has effective length 0"
            )));
        }
    }
    clip_concordance_times(&mut stop, ymin);
    Ok((start, stop))
}

fn retained_concordance_events(time: &[f64], status: &[i32], ymax: Option<f64>) -> f64 {
    time.iter()
        .zip(status)
        .filter(|&(time, event)| *event == 1 && ymax.is_none_or(|upper| *time <= upper))
        .count() as f64
}

fn bounded_concordance_multiplier(time: f64, multiplier: f64, ymax: Option<f64>) -> f64 {
    if ymax.is_some_and(|upper| time > upper) {
        0.0
    } else {
        multiplier
    }
}

fn parse_concordance_time_weight(timewt: &str) -> PyResult<ConcordanceTimeWeight> {
    match timewt {
        "n" => Ok(ConcordanceTimeWeight::N),
        "S" => Ok(ConcordanceTimeWeight::S),
        "S/G" => Ok(ConcordanceTimeWeight::SOverG),
        "n/G2" => Ok(ConcordanceTimeWeight::NOverG2),
        "I" => Ok(ConcordanceTimeWeight::I),
        _ => Err(PyValueError::new_err(
            "timewt must be one of 'n', 'S', 'S/G', 'n/G2', 'I'",
        )),
    }
}

fn parse_counting_concordance_time_weight(timewt: &str) -> PyResult<ConcordanceTimeWeight> {
    match parse_concordance_time_weight(timewt)? {
        ConcordanceTimeWeight::SOverG | ConcordanceTimeWeight::NOverG2 => {
            Err(PyValueError::new_err(
                "S/G and n/G2 timewt options are not supported for counting-process data",
            ))
        }
        value => Ok(value),
    }
}

fn concordance_time_weight_multiplier(
    time_weight: ConcordanceTimeWeight,
    total_weight: f64,
    survival: f64,
    censoring_survival: f64,
    nrisk: f64,
) -> f64 {
    if nrisk <= 0.0 {
        return 0.0;
    }
    match time_weight {
        ConcordanceTimeWeight::S => total_weight * survival / nrisk,
        ConcordanceTimeWeight::SOverG => {
            if censoring_survival > 0.0 {
                total_weight * survival / (censoring_survival * nrisk)
            } else {
                0.0
            }
        }
        ConcordanceTimeWeight::NOverG2 => {
            if censoring_survival > 0.0 {
                1.0 / (censoring_survival * censoring_survival)
            } else {
                0.0
            }
        }
        ConcordanceTimeWeight::I => 1.0 / nrisk,
        ConcordanceTimeWeight::N => 1.0,
    }
}

fn multiplier_at(multipliers: &[(f64, f64)], time: f64) -> f64 {
    multipliers
        .binary_search_by(|(candidate, _)| {
            if *candidate == time {
                std::cmp::Ordering::Equal
            } else {
                candidate.total_cmp(&time)
            }
        })
        .map_or(0.0, |idx| multipliers[idx].1)
}

#[inline]
fn concordance_case_weight(weights: Option<&[f64]>, index: usize) -> f64 {
    weights.map_or(1.0, |values| values[index])
}

fn right_concordance_time_weight_multipliers(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> Vec<(f64, f64)> {
    // R bases this fallback on original unweighted events in each stratum,
    // including events beyond ymax and events with zero case weight.
    let time_weight = if status.iter().filter(|&&event| event == 1).count() < 2 {
        ConcordanceTimeWeight::N
    } else {
        time_weight
    };
    if time_weight == ConcordanceTimeWeight::N {
        let mut values: Vec<f64> = time
            .iter()
            .zip(status.iter())
            .filter_map(|(&time, &event)| (event == 1).then_some(time))
            .collect();
        values.sort_by(f64::total_cmp);
        values.dedup();
        return values
            .into_iter()
            .map(|time| (time, bounded_concordance_multiplier(time, 1.0, ymax)))
            .collect();
    }

    let total_weight = weights.map_or(time.len() as f64, |values| values.iter().sum());
    let mut survival = 1.0;
    let mut censoring_survival = 1.0;
    let mut nrisk = total_weight;
    let mut multipliers = Vec::new();
    let mut time_order: Vec<usize> = (0..time.len()).collect();
    time_order.sort_by(|&left, &right| {
        time[left]
            .total_cmp(&time[right])
            .then_with(|| left.cmp(&right))
    });

    let mut group_start = 0;
    while group_start < time_order.len() {
        let event_time = time[time_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < time_order.len() && time[time_order[group_end]] == event_time {
            group_end += 1;
        }

        let mut death_weight = 0.0;
        let mut censor_weight = 0.0;
        let mut group_weight = 0.0;
        for &idx in &time_order[group_start..group_end] {
            let weight = concordance_case_weight(weights, idx);
            group_weight += weight;
            if status[idx] == 1 {
                death_weight += weight;
            } else {
                censor_weight += weight;
            }
        }

        if time_order[group_start..group_end]
            .iter()
            .any(|&idx| status[idx] == 1)
        {
            multipliers.push((
                event_time,
                bounded_concordance_multiplier(
                    event_time,
                    concordance_time_weight_multiplier(
                        time_weight,
                        total_weight,
                        survival,
                        censoring_survival,
                        nrisk,
                    ),
                    ymax,
                ),
            ));
            if nrisk > 0.0 {
                survival *= ((nrisk - death_weight) / nrisk).max(0.0);
            }
        }
        let censor_risk = nrisk - death_weight;
        if censor_weight > 0.0 && censor_risk > 0.0 {
            censoring_survival *= ((censor_risk - censor_weight) / censor_risk).max(0.0);
        }
        nrisk = (nrisk - group_weight).max(0.0);
        group_start = group_end;
    }
    multipliers
}

fn weighted_pairs(weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    let squared: f64 = weights.iter().map(|weight| weight * weight).sum();
    0.5 * (total * total - squared)
}

fn conditional_variance_increment(
    at_risk: &FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
    weight: f64,
) -> f64 {
    let rank = risk_levels.partition_point(|&risk| risk < risk_score);
    let lower = rank_prefix_weight_before(at_risk, rank);
    let lower_or_tied = rank_prefix_weight_before(at_risk, rank + 1);
    let tied = lower_or_tied - lower;
    let greater = at_risk.total() - lower_or_tied;
    weight
        * (greater * (weight + 2.0 * (lower + tied))
            + lower * (weight + 2.0 * (greater + tied))
            + (greater - lower) * (greater - lower))
}

fn add_concordance_risk_observation<const FULL_SUMMARY: bool>(
    at_risk: &mut FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
    weight: f64,
    z2: &mut f64,
) {
    if FULL_SUMMARY {
        *z2 += conditional_variance_increment(at_risk, risk_levels, risk_score, weight);
    }
    let rank = risk_levels.partition_point(|&risk| risk < risk_score);
    at_risk.update(rank, weight);
}

fn remove_concordance_risk_observation<const FULL_SUMMARY: bool>(
    at_risk: &mut FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
    weight: f64,
    z2: &mut f64,
) {
    let rank = risk_levels.partition_point(|&risk| risk < risk_score);
    at_risk.update(rank, -weight);
    if FULL_SUMMARY {
        *z2 -= conditional_variance_increment(at_risk, risk_levels, risk_score, weight);
    }
}

fn tied_event_risk_pairs(
    event_indices: &[usize],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
) -> f64 {
    let mut events: Vec<(f64, f64)> = event_indices
        .iter()
        .map(|&idx| (risk_scores[idx], concordance_case_weight(weights, idx)))
        .collect();
    events.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut tied_weight = 0.0;
    let mut window_weight = 0.0;
    let mut left = 0;
    for right in 0..events.len() {
        while left < right && events[right].0 != events[left].0 {
            window_weight -= events[left].1;
            left += 1;
        }
        tied_weight += events[right].1 * window_weight;
        window_weight += events[right].1;
    }
    tied_weight
}

fn right_concordance_summary_counts_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64) {
    right_concordance_sweep::<true>(time, status, risk_scores, weights, time_weight, ymax)
}

// Scalar indices use this same sweep with variance updates and outcome-tie
// bookkeeping removed at compile time.
fn right_concordance_sweep<const FULL_SUMMARY: bool>(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64) {
    let multipliers =
        right_concordance_time_weight_multipliers(time, status, weights, time_weight, ymax);
    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();

    let mut time_order: Vec<usize> = (0..time.len()).collect();
    time_order.sort_by(|&left, &right| {
        time[right]
            .total_cmp(&time[left])
            .then_with(|| left.cmp(&right))
    });
    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut summary = ConcordanceSummary::default();
    let mut counts = ConcordanceTieCounts::default();
    let mut z2 = 0.0;
    let mut conditional_variance_numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    let mut group_start = 0;
    while group_start < time_order.len() {
        let event_time = time[time_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < time_order.len() && time[time_order[group_end]] == event_time {
            group_end += 1;
        }

        // Same-time censors are still at risk for the group's events.
        for &idx in &time_order[group_start..group_end] {
            if status[idx] == 0 {
                add_concordance_risk_observation::<FULL_SUMMARY>(
                    &mut at_risk,
                    &risk_levels,
                    risk_scores[idx],
                    concordance_case_weight(weights, idx),
                    &mut z2,
                );
            }
        }
        let multiplier = multiplier_at(&multipliers, event_time);
        if multiplier > 0.0 {
            let comparable_weight = at_risk.total();
            for &event_idx in &time_order[group_start..group_end] {
                if status[event_idx] != 1 {
                    continue;
                }
                let lower_end = risk_levels.partition_point(|&risk| risk < risk_scores[event_idx]);
                let tied_end = risk_levels.partition_point(|&risk| risk <= risk_scores[event_idx]);
                let lower = rank_prefix_weight_before(&at_risk, lower_end);
                let lower_or_tied = rank_prefix_weight_before(&at_risk, tied_end);
                let event_weight = concordance_case_weight(weights, event_idx) * multiplier;
                let tied_weight = lower_or_tied - lower;
                if FULL_SUMMARY {
                    counts.tied_x += event_weight * tied_weight;
                }
                summary.concordant += event_weight * (lower + 0.5 * tied_weight);
                summary.comparable += event_weight * comparable_weight;
            }

            if FULL_SUMMARY && group_end - group_start > 1 {
                let event_indices: Vec<usize> = time_order[group_start..group_end]
                    .iter()
                    .copied()
                    .filter(|&idx| status[idx] == 1)
                    .collect();
                let event_weights: Vec<f64> = event_indices
                    .iter()
                    .map(|&idx| concordance_case_weight(weights, idx))
                    .collect();
                let tied_xy = tied_event_risk_pairs(&event_indices, risk_scores, weights);
                counts.tied_xy += tied_xy * multiplier;
                counts.tied_y += (weighted_pairs(&event_weights) - tied_xy).max(0.0) * multiplier;
            }
        }

        let mut death_weight = 0.0;
        for &idx in &time_order[group_start..group_end] {
            if status[idx] != 1 {
                continue;
            }
            let weight = concordance_case_weight(weights, idx);
            add_concordance_risk_observation::<FULL_SUMMARY>(
                &mut at_risk,
                &risk_levels,
                risk_scores[idx],
                weight,
                &mut z2,
            );
            if FULL_SUMMARY {
                death_weight += weight;
            }
        }
        let risk_weight = at_risk.total();
        if FULL_SUMMARY && death_weight > 0.0 && multiplier > 0.0 && risk_weight > 0.0 {
            conditional_variance_numerator += death_weight * multiplier * z2 / risk_weight;
            comparable_pair_weight += death_weight * (risk_weight - death_weight) * multiplier;
        }
        group_start = group_end;
    }
    (
        summary,
        counts,
        conditional_variance_numerator,
        comparable_pair_weight,
    )
}

fn counting_concordance_time_weight_multipliers(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> Vec<(f64, f64)> {
    let time_weight = if status.iter().filter(|&&event| event == 1).count() < 2 {
        ConcordanceTimeWeight::N
    } else {
        time_weight
    };
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    if time_weight == ConcordanceTimeWeight::N {
        let mut multipliers = Vec::new();
        let mut group_start = 0;
        while group_start < event_indices.len() {
            let event_time = stop[event_indices[group_start]];
            multipliers.push((
                event_time,
                bounded_concordance_multiplier(event_time, 1.0, ymax),
            ));
            group_start += 1;
            while group_start < event_indices.len()
                && stop[event_indices[group_start]] == event_time
            {
                group_start += 1;
            }
        }
        return multipliers;
    }

    let total_weight = weights.map_or(stop.len() as f64, |values| values.iter().sum());
    let mut survival = 1.0;
    let mut multipliers = Vec::new();
    let mut start_order: Vec<usize> = (0..start.len()).collect();
    start_order.sort_by(|&left, &right| {
        start[left]
            .total_cmp(&start[right])
            .then_with(|| left.cmp(&right))
    });
    let mut stop_order: Vec<usize> = (0..stop.len()).collect();
    stop_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut nrisk = 0.0;
    let mut start_pos = 0;
    let mut stop_pos = 0;
    let mut group_start = 0;
    while group_start < event_indices.len() {
        let event_time = stop[event_indices[group_start]];
        while start_pos < start_order.len() && start[start_order[start_pos]] < event_time {
            nrisk += concordance_case_weight(weights, start_order[start_pos]);
            start_pos += 1;
        }
        while stop_pos < stop_order.len() && stop[stop_order[stop_pos]] < event_time {
            nrisk -= concordance_case_weight(weights, stop_order[stop_pos]);
            stop_pos += 1;
        }

        let mut group_end = group_start + 1;
        let mut death_weight = concordance_case_weight(weights, event_indices[group_start]);
        while group_end < event_indices.len() && stop[event_indices[group_end]] == event_time {
            death_weight += concordance_case_weight(weights, event_indices[group_end]);
            group_end += 1;
        }
        multipliers.push((
            event_time,
            bounded_concordance_multiplier(
                event_time,
                concordance_time_weight_multiplier(time_weight, total_weight, survival, 1.0, nrisk),
                ymax,
            ),
        ));
        if nrisk > 0.0 {
            survival *= ((nrisk - death_weight) / nrisk).max(0.0);
        }
        group_start = group_end;
    }
    multipliers
}

fn counting_concordance_summary_counts_for_vectors(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64) {
    counting_concordance_sweep::<true>(start, stop, status, risk_scores, weights, time_weight, ymax)
}

fn counting_concordance_sweep<const FULL_SUMMARY: bool>(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64) {
    let multipliers = counting_concordance_time_weight_multipliers(
        start,
        stop,
        status,
        weights,
        time_weight,
        ymax,
    );
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();
    let mut start_order: Vec<usize> = (0..start.len()).collect();
    start_order.sort_by(|&left, &right| {
        start[left]
            .total_cmp(&start[right])
            .then_with(|| left.cmp(&right))
    });
    let mut stop_order: Vec<usize> = (0..stop.len()).collect();
    stop_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut active = vec![false; stop.len()];
    let mut z2 = 0.0;
    let mut numerator = 0.0;
    let mut summary = ConcordanceSummary::default();
    let mut ties = ConcordanceTieCounts::default();
    let mut comparable_pair_weight = 0.0;
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut event_group_start = 0usize;
    while event_group_start < event_indices.len() {
        let event_time = stop[event_indices[event_group_start]];
        while start_cursor < start_order.len() && start[start_order[start_cursor]] < event_time {
            let idx = start_order[start_cursor];
            if !active[idx] {
                add_concordance_risk_observation::<FULL_SUMMARY>(
                    &mut at_risk,
                    &risk_levels,
                    risk_scores[idx],
                    concordance_case_weight(weights, idx),
                    &mut z2,
                );
                active[idx] = true;
            }
            start_cursor += 1;
        }
        while stop_cursor < stop_order.len() && stop[stop_order[stop_cursor]] < event_time {
            let idx = stop_order[stop_cursor];
            if active[idx] {
                remove_concordance_risk_observation::<FULL_SUMMARY>(
                    &mut at_risk,
                    &risk_levels,
                    risk_scores[idx],
                    concordance_case_weight(weights, idx),
                    &mut z2,
                );
                active[idx] = false;
            }
            stop_cursor += 1;
        }

        let mut event_group_end = event_group_start + 1;
        let mut death_weight = if FULL_SUMMARY {
            concordance_case_weight(weights, event_indices[event_group_start])
        } else {
            0.0
        };
        while event_group_end < event_indices.len()
            && stop[event_indices[event_group_end]] == event_time
        {
            if FULL_SUMMARY {
                death_weight += concordance_case_weight(weights, event_indices[event_group_end]);
            }
            event_group_end += 1;
        }
        let risk_weight = at_risk.total();
        let multiplier = multiplier_at(&multipliers, event_time);
        if risk_weight > 0.0 && multiplier > 0.0 {
            let events = &event_indices[event_group_start..event_group_end];
            if FULL_SUMMARY {
                let event_weights: Vec<f64> = events
                    .iter()
                    .map(|&idx| concordance_case_weight(weights, idx))
                    .collect();
                let tied_xy = tied_event_risk_pairs(events, risk_scores, weights);
                ties.tied_xy += tied_xy * multiplier;
                ties.tied_y += (weighted_pairs(&event_weights) - tied_xy).max(0.0) * multiplier;
                numerator += death_weight * multiplier * z2 / risk_weight;
                comparable_pair_weight += death_weight * (risk_weight - death_weight) * multiplier;
            }

            // The conditional variance uses the full risk set. Pair counts use
            // the remaining censors and later exits after simultaneous deaths
            // leave. Querying that set directly avoids subtracting large event
            // pair totals from the concordant and predictor-tie counts.
            for &idx in events {
                remove_concordance_risk_observation::<FULL_SUMMARY>(
                    &mut at_risk,
                    &risk_levels,
                    risk_scores[idx],
                    concordance_case_weight(weights, idx),
                    &mut z2,
                );
                active[idx] = false;
            }
            let comparable_weight = at_risk.total();
            if comparable_weight > 0.0 {
                for &idx in events {
                    let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                    let lower = rank_prefix_weight_before(&at_risk, rank);
                    let tied_weight = (at_risk.prefix_sum(rank) - lower).max(0.0);
                    let event_weight = concordance_case_weight(weights, idx) * multiplier;
                    if FULL_SUMMARY {
                        ties.tied_x += event_weight * tied_weight;
                    }
                    summary.concordant += event_weight * (lower + 0.5 * tied_weight);
                    summary.comparable += event_weight * comparable_weight;
                }
            }
        }
        event_group_start = event_group_end;
    }
    (summary, ties, numerator, comparable_pair_weight)
}

fn rank_from_active_risk_set(
    at_risk: &FenwickTree,
    risk_levels: &[f64],
    event_risk: f64,
) -> Option<(f64, f64)> {
    let risk_weight = at_risk.total();
    if risk_weight <= 0.0 {
        return None;
    }

    let lower_end = risk_levels.partition_point(|&risk| risk < event_risk);
    let not_greater_end = risk_levels.partition_point(|&risk| risk <= event_risk);
    let lower = rank_prefix_weight_before(at_risk, lower_end);
    let not_greater = rank_prefix_weight_before(at_risk, not_greater_end);
    let greater = risk_weight - not_greater;
    Some(((lower - greater) / risk_weight, risk_weight))
}

#[inline]
fn rank_prefix_weight_before(at_risk: &FenwickTree, end: usize) -> f64 {
    if end == 0 {
        0.0
    } else {
        at_risk.prefix_sum(end - 1)
    }
}

fn right_concordance_rank_rows_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceRankRows {
    let multipliers =
        right_concordance_time_weight_multipliers(time, status, case_weights, time_weight, ymax);
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        time[left]
            .total_cmp(&time[right])
            .then_with(|| left.cmp(&right))
    });

    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();
    let mut time_order: Vec<usize> = (0..time.len()).collect();
    time_order.sort_by(|&left, &right| {
        time[left]
            .total_cmp(&time[right])
            .then_with(|| left.cmp(&right))
    });

    let mut at_risk = FenwickTree::new(risk_levels.len());
    for (idx, &risk_score) in risk_scores.iter().enumerate() {
        let rank = risk_levels.partition_point(|&risk| risk < risk_score);
        at_risk.update(rank, concordance_case_weight(case_weights, idx));
    }
    let mut time_cursor = 0usize;
    let mut rows = Vec::with_capacity(event_indices.len());
    let mut event_group_start = 0usize;
    while event_group_start < event_indices.len() {
        let event_time = time[event_indices[event_group_start]];
        while time_cursor < time_order.len() && time[time_order[time_cursor]] < event_time {
            let idx = time_order[time_cursor];
            let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
            at_risk.update(rank, -concordance_case_weight(case_weights, idx));
            time_cursor += 1;
        }

        let mut event_group_end = event_group_start + 1;
        while event_group_end < event_indices.len()
            && time[event_indices[event_group_end]] == event_time
        {
            event_group_end += 1;
        }

        for &event_idx in &event_indices[event_group_start..event_group_end] {
            let multiplier = multiplier_at(&multipliers, time[event_idx]);
            if multiplier <= 0.0 {
                continue;
            }
            if let Some((rank, risk_weight)) =
                rank_from_active_risk_set(&at_risk, &risk_levels, risk_scores[event_idx])
            {
                rows.push((
                    time[event_idx],
                    rank,
                    risk_weight * multiplier,
                    concordance_case_weight(case_weights, event_idx),
                ));
            }
        }

        event_group_start = event_group_end;
    }
    rows
}

fn counting_concordance_rank_rows_for_vectors(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceRankRows {
    let multipliers = counting_concordance_time_weight_multipliers(
        start,
        stop,
        status,
        case_weights,
        time_weight,
        ymax,
    );
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();
    let mut start_order: Vec<usize> = (0..start.len()).collect();
    start_order.sort_by(|&left, &right| {
        start[left]
            .total_cmp(&start[right])
            .then_with(|| left.cmp(&right))
    });
    let mut stop_order: Vec<usize> = (0..stop.len()).collect();
    stop_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut rows = Vec::with_capacity(event_indices.len());
    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut active = vec![false; stop.len()];
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut event_group_start = 0usize;
    while event_group_start < event_indices.len() {
        let event_time = stop[event_indices[event_group_start]];
        while start_cursor < start_order.len() && start[start_order[start_cursor]] < event_time {
            let idx = start_order[start_cursor];
            if !active[idx] {
                let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                at_risk.update(rank, concordance_case_weight(case_weights, idx));
                active[idx] = true;
            }
            start_cursor += 1;
        }
        while stop_cursor < stop_order.len() && stop[stop_order[stop_cursor]] < event_time {
            let idx = stop_order[stop_cursor];
            if active[idx] {
                let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                at_risk.update(rank, -concordance_case_weight(case_weights, idx));
                active[idx] = false;
            }
            stop_cursor += 1;
        }

        let mut event_group_end = event_group_start + 1;
        while event_group_end < event_indices.len()
            && stop[event_indices[event_group_end]] == event_time
        {
            event_group_end += 1;
        }

        let multiplier = multiplier_at(&multipliers, event_time);
        if multiplier > 0.0 {
            for &event_idx in &event_indices[event_group_start..event_group_end] {
                if let Some((rank, risk_weight)) =
                    rank_from_active_risk_set(&at_risk, &risk_levels, risk_scores[event_idx])
                {
                    rows.push((
                        event_time,
                        rank,
                        risk_weight * multiplier,
                        concordance_case_weight(case_weights, event_idx),
                    ));
                }
            }
        }

        event_group_start = event_group_end;
    }
    rows
}

fn sort_rank_rows_by_time(rows: &mut ConcordanceRankRows) {
    rows.sort_by(|left, right| left.0.total_cmp(&right.0));
}

fn stratified_right_concordance_rank_rows(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceRankRows {
    let mut rows = Vec::new();
    for indices in strata_groups(strata) {
        let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        rows.extend(right_concordance_rank_rows_for_vectors(
            &group_time,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        ));
    }
    sort_rank_rows_by_time(&mut rows);
    rows
}

#[allow(clippy::too_many_arguments)]
fn stratified_counting_concordance_rank_rows_for_strata(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceRankRows {
    let mut rows = Vec::new();
    for indices in strata_groups(strata) {
        let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
        let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        rows.extend(counting_concordance_rank_rows_for_vectors(
            &group_start,
            &group_stop,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        ));
    }
    sort_rank_rows_by_time(&mut rows);
    rows
}

/// Raw derivatives of the five pair counts with respect to each case weight.
/// Time weights are held fixed, matching survival::concordancefit.
fn influence_from_rows(
    influence_rows: Vec<[f64; 5]>,
    weights: Option<&[f64]>,
) -> ConcordanceInfluenceOutput {
    // Each pair occurs once in each endpoint's derivative. Weighting those
    // derivatives recovers twice the pooled counts, including across strata.
    let mut concordant = 0.0;
    let mut comparable = 0.0;
    for (idx, row) in influence_rows.iter().enumerate() {
        let weight = 0.5 * concordance_case_weight(weights, idx);
        concordant += weight * (row[0] + 0.5 * row[2]);
        comparable += weight * (row[0] + row[1] + row[2]);
    }
    let output_rows: Vec<Vec<f64>> = influence_rows.iter().map(|row| row.to_vec()).collect();
    if comparable <= 0.0 {
        return (output_rows, vec![f64::NAN; influence_rows.len()], f64::NAN);
    }

    let c_index = concordant / comparable;
    let dfbeta: Vec<f64> = influence_rows
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            let comparable_row = row[0] + row[1] + row[2];
            (row[0] + 0.5 * row[2] - comparable_row * c_index)
                * concordance_case_weight(weights, idx)
                / comparable
        })
        .collect();
    let variance = dfbeta.iter().map(|value| value * value).sum();
    (output_rows, dfbeta, variance)
}

/// Weights below, above, and at a predictor rank.
fn influence_rank_weights(tree: &FenwickTree, rank: usize) -> [f64; 3] {
    let lower = rank_prefix_weight_before(tree, rank);
    let through_rank = tree.prefix_sum(rank);
    [lower, tree.total() - through_rank, through_rank - lower]
}

fn finish_concordance_risk_influence(
    row: &mut [f64; 5],
    entry_counts: [f64; 3],
    event_counts: [f64; 3],
) {
    // A larger predictor on the event, rather than the risk comparator, is
    // concordant. Subtract events at/before entry for counting responses.
    row[0] += event_counts[1] - entry_counts[1];
    row[1] += event_counts[0] - entry_counts[0];
    row[2] += event_counts[2] - entry_counts[2];
}

/// Two Fenwick trees accumulate both endpoint derivatives in O(n log n).
/// Events leave the risk set before their time group's comparisons; censors
/// leave afterward. A counting row enters only after its open start boundary.
fn concordance_raw_influence_rows(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    multipliers: &[(f64, f64)],
) -> Vec<[f64; 5]> {
    let n = stop.len();
    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();
    let ranks: Vec<usize> = risk_scores
        .iter()
        .map(|&risk| risk_levels.partition_point(|&value| value < risk))
        .collect();
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&left, &right| stop[left].total_cmp(&stop[right]));
    let mut start_order: Vec<usize> = (0..n).collect();
    if let Some(start) = start {
        start_order.sort_by(|&left, &right| start[left].total_cmp(&start[right]));
    }

    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut earlier_events = FenwickTree::new(risk_levels.len());
    let mut entry_counts = vec![[0.0; 3]; n];
    let mut rows = vec![[0.0; 5]; n];
    if start.is_none() {
        for (idx, &rank) in ranks.iter().enumerate() {
            at_risk.update(rank, concordance_case_weight(weights, idx));
        }
    }
    let mut start_cursor = 0;
    let mut group_start = 0;
    while group_start < n {
        let event_time = stop[stop_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < n && stop[stop_order[group_end]] == event_time {
            group_end += 1;
        }
        if let Some(start) = start {
            while start_cursor < n && start[start_order[start_cursor]] < event_time {
                let idx = start_order[start_cursor];
                entry_counts[idx] = influence_rank_weights(&earlier_events, ranks[idx]);
                at_risk.update(ranks[idx], concordance_case_weight(weights, idx));
                start_cursor += 1;
            }
        }

        let mut events: Vec<usize> = stop_order[group_start..group_end]
            .iter()
            .copied()
            .filter(|&idx| status[idx] == 1)
            .collect();
        for &idx in &events {
            finish_concordance_risk_influence(
                &mut rows[idx],
                entry_counts[idx],
                influence_rank_weights(&earlier_events, ranks[idx]),
            );
            at_risk.update(ranks[idx], -concordance_case_weight(weights, idx));
        }

        let multiplier = multiplier_at(multipliers, event_time);
        for &idx in &events {
            let counts = influence_rank_weights(&at_risk, ranks[idx]);
            for column in 0..3 {
                rows[idx][column] += counts[column] * multiplier;
            }
        }
        // Simultaneous events are outcome ties, not risk comparators. Aggregate
        // their predictor groups so even one large event tie stays O(n log n).
        events.sort_unstable_by_key(|&idx| ranks[idx]);
        let death_weight: f64 = events
            .iter()
            .map(|&idx| concordance_case_weight(weights, idx))
            .sum();
        let mut tie_start = 0;
        while tie_start < events.len() {
            let mut tie_end = tie_start + 1;
            while tie_end < events.len() && ranks[events[tie_end]] == ranks[events[tie_start]] {
                tie_end += 1;
            }
            let tie_weight: f64 = events[tie_start..tie_end]
                .iter()
                .map(|&idx| concordance_case_weight(weights, idx))
                .sum();
            for &idx in &events[tie_start..tie_end] {
                rows[idx][3] += (death_weight - tie_weight) * multiplier;
                rows[idx][4] += (tie_weight - concordance_case_weight(weights, idx)) * multiplier;
            }
            tie_start = tie_end;
        }
        for &idx in &events {
            earlier_events.update(
                ranks[idx],
                concordance_case_weight(weights, idx) * multiplier,
            );
        }
        for &idx in &stop_order[group_start..group_end] {
            if status[idx] == 0 {
                finish_concordance_risk_influence(
                    &mut rows[idx],
                    entry_counts[idx],
                    influence_rank_weights(&earlier_events, ranks[idx]),
                );
                at_risk.update(ranks[idx], -concordance_case_weight(weights, idx));
            }
        }
        group_start = group_end;
    }
    rows
}

fn right_concordance_raw_influence(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> Vec<[f64; 5]> {
    let multipliers =
        right_concordance_time_weight_multipliers(time, status, weights, time_weight, ymax);
    concordance_raw_influence_rows(None, time, status, risk_scores, weights, &multipliers)
}

fn counting_concordance_raw_influence(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> Vec<[f64; 5]> {
    let multipliers = counting_concordance_time_weight_multipliers(
        start,
        stop,
        status,
        weights,
        time_weight,
        ymax,
    );
    concordance_raw_influence_rows(
        Some(start),
        stop,
        status,
        risk_scores,
        weights,
        &multipliers,
    )
}

fn right_concordance_influence_rows_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceInfluenceOutput {
    influence_from_rows(
        right_concordance_raw_influence(time, status, risk_scores, weights, time_weight, ymax),
        weights,
    )
}

fn counting_concordance_influence_rows_for_vectors(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceInfluenceOutput {
    influence_from_rows(
        counting_concordance_raw_influence(
            start,
            stop,
            status,
            risk_scores,
            weights,
            time_weight,
            ymax,
        ),
        weights,
    )
}

fn remap_stratified_influence(
    n: usize,
    groups: Vec<Vec<usize>>,
    weights: Option<&[f64]>,
    mut compute_group: impl FnMut(&[usize]) -> Vec<[f64; 5]>,
) -> ConcordanceInfluenceOutput {
    let mut rows = vec![[0.0; 5]; n];
    for indices in groups {
        let group_rows = compute_group(&indices);
        for (local_idx, &original_idx) in indices.iter().enumerate() {
            rows[original_idx] = group_rows[local_idx];
        }
    }
    influence_from_rows(rows, weights)
}

fn stratified_right_concordance_influence_rows(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceInfluenceOutput {
    remap_stratified_influence(time.len(), strata_groups(strata), weights, |indices| {
        let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        right_concordance_raw_influence(
            &group_time,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn stratified_counting_concordance_influence_rows_for_strata(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> ConcordanceInfluenceOutput {
    remap_stratified_influence(stop.len(), strata_groups(strata), weights, |indices| {
        let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
        let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        counting_concordance_raw_influence(
            &group_start,
            &group_stop,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        )
    })
}

fn stratified_right_concordance_summary_counts(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64, f64) {
    let mut total = ConcordanceSummary::default();
    let mut ties = ConcordanceTieCounts::default();
    let mut conditional_variance_numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    for indices in strata_groups(strata) {
        let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        let (
            summary,
            group_ties,
            group_conditional_variance_numerator,
            group_comparable_pair_weight,
        ) = right_concordance_summary_counts_for_vectors(
            &group_time,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        );
        total.concordant += summary.concordant;
        total.comparable += summary.comparable;
        ties.tied_x += group_ties.tied_x;
        ties.tied_y += group_ties.tied_y;
        ties.tied_xy += group_ties.tied_xy;
        conditional_variance_numerator += group_conditional_variance_numerator;
        comparable_pair_weight += group_comparable_pair_weight;
    }
    let n_event = retained_concordance_events(time, status, ymax);
    (
        total,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
    )
}

#[allow(clippy::too_many_arguments)]
fn stratified_counting_concordance_summary_counts(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    ymax: Option<f64>,
) -> (ConcordanceSummary, ConcordanceTieCounts, f64, f64, f64) {
    let mut total = ConcordanceSummary::default();
    let mut ties = ConcordanceTieCounts::default();
    let mut conditional_variance_numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    for indices in strata_groups(strata) {
        let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
        let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        let (
            summary,
            group_ties,
            group_conditional_variance_numerator,
            group_comparable_pair_weight,
        ) = counting_concordance_summary_counts_for_vectors(
            &group_start,
            &group_stop,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
            ymax,
        );
        total.concordant += summary.concordant;
        total.comparable += summary.comparable;
        ties.tied_x += group_ties.tied_x;
        ties.tied_y += group_ties.tied_y;
        ties.tied_xy += group_ties.tied_xy;
        conditional_variance_numerator += group_conditional_variance_numerator;
        comparable_pair_weight += group_comparable_pair_weight;
    }
    let n_event = retained_concordance_events(stop, status, ymax);
    (
        total,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
    )
}

fn validate_legacy_concordance_inputs(
    y: &[f64],
    x: &[i32],
    wt: &[f64],
    timewt: &[f64],
    sortstart: Option<&[usize]>,
    sortstop: &[usize],
) -> PyResult<()> {
    let n = y.len();
    validate_length(n, x.len(), "x")?;
    validate_length(n, wt.len(), "wt")?;
    validate_length(n, timewt.len(), "timewt")?;
    validate_length(n, sortstop.len(), "sortstop")?;

    validate_no_nan(y, "y")?;
    validate_finite(y, "y")?;
    validate_no_nan(wt, "wt")?;
    validate_finite(wt, "wt")?;
    validate_non_negative(wt, "wt")?;
    validate_no_nan(timewt, "timewt")?;
    validate_finite(timewt, "timewt")?;
    validate_non_negative(timewt, "timewt")?;

    if let Some((index, value)) = x.iter().enumerate().find(|(_, value)| **value < 0) {
        return Err(PyValueError::new_err(format!(
            "x contains negative value {value} at index {index}"
        )));
    }
    if let Some((index, value)) = x
        .iter()
        .enumerate()
        .find(|(_, value)| **value as usize >= n)
    {
        return Err(PyValueError::new_err(format!(
            "x value {value} at index {index} is outside observation count {n}"
        )));
    }
    if let Some((index, value)) = sortstop.iter().enumerate().find(|(_, value)| **value >= n) {
        return Err(PyValueError::new_err(format!(
            "sortstop value {value} at index {index} is outside observation count {n}"
        )));
    }
    if let Some(values) = sortstart {
        validate_length(n, values.len(), "sortstart")?;
        if let Some((index, value)) = values.iter().enumerate().find(|(_, value)| **value >= n) {
            return Err(PyValueError::new_err(format!(
                "sortstart value {value} at index {index} is outside observation count {n}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn concordance_index(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<f64> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    let (summary, _, _, _) = right_concordance_sweep::<false>(
        &time,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    );
    Ok(summary.concordant / summary.comparable)
}

fn conditional_variance_from_numerator(numerator: f64, comparable: f64) -> f64 {
    numerator / (4.0 * comparable * comparable)
}

fn build_concordance_summary_dict(
    summary: ConcordanceSummary,
    ties: ConcordanceTieCounts,
    n_event: Option<f64>,
    conditional_variance: f64,
) -> PyResult<Py<PyDict>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("concordance", summary.concordant / summary.comparable)?;
        dict.set_item("concordant", summary.concordant)?;
        dict.set_item("comparable", summary.comparable)?;
        dict.set_item("tied_x", ties.tied_x)?;
        dict.set_item("tied_y", ties.tied_y)?;
        dict.set_item("tied_xy", ties.tied_xy)?;
        dict.set_item("conditional_variance", conditional_variance)?;
        if let Some(value) = n_event {
            dict.set_item("n_event", value)?;
        }
        Ok(dict.into())
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn concordance_summary(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<Py<PyDict>> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    let (summary, ties, conditional_variance_numerator, comparable_pair_weight) =
        right_concordance_summary_counts_for_vectors(
            &time,
            &status,
            &risk_scores,
            weights.as_deref(),
            time_weight,
            ymax,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_concordance_summary_dict(
        summary,
        ties,
        Some(retained_concordance_events(&time, &status, ymax)),
        conditional_variance,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn stratified_concordance_summary(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<Py<PyDict>> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    validate_strata_length(time.len(), &strata, "time")?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    let (summary, ties, n_event, conditional_variance_numerator, comparable_pair_weight) =
        stratified_right_concordance_summary_counts(
            &time,
            &status,
            &risk_scores,
            &strata,
            weights.as_deref(),
            time_weight,
            ymax,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_concordance_summary_dict(summary, ties, Some(n_event), conditional_variance)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn concordance_rank_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceRankRows> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(right_concordance_rank_rows_for_vectors(
        &time,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn stratified_concordance_rank_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceRankRows> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    validate_strata_length(time.len(), &strata, "time")?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(stratified_right_concordance_rank_rows(
        &time,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn concordance_influence_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(right_concordance_influence_rows_for_vectors(
        &time,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=true, ymin=None, ymax=None))]
pub fn stratified_concordance_influence_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: bool,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let time = prepare_right_concordance_times(time, timefix, ymin)?;
    validate_strata_length(time.len(), &strata, "time")?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(stratified_right_concordance_influence_rows(
        &time,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn counting_concordance_index(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<f64> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    let (summary, _, _, _) = counting_concordance_sweep::<false>(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    );
    Ok(summary.concordant / summary.comparable)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn counting_concordance_summary(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<Py<PyDict>> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    let (summary, ties, conditional_variance_numerator, comparable_pair_weight) =
        counting_concordance_summary_counts_for_vectors(
            &start,
            &stop,
            &status,
            &risk_scores,
            weights.as_deref(),
            time_weight,
            ymax,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_concordance_summary_dict(
        summary,
        ties,
        Some(retained_concordance_events(&stop, &status, ymax)),
        conditional_variance,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn stratified_counting_concordance_summary(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<Py<PyDict>> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    validate_strata_length(start.len(), &strata, "start")?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    let (summary, ties, n_event, conditional_variance_numerator, comparable_pair_weight) =
        stratified_counting_concordance_summary_counts(
            &start,
            &stop,
            &status,
            &risk_scores,
            &strata,
            weights.as_deref(),
            time_weight,
            ymax,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_concordance_summary_dict(summary, ties, Some(n_event), conditional_variance)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn counting_concordance_rank_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceRankRows> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    Ok(counting_concordance_rank_rows_for_vectors(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn stratified_counting_concordance_rank_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceRankRows> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    validate_strata_length(start.len(), &strata, "start")?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    Ok(stratified_counting_concordance_rank_rows_for_strata(
        &start,
        &stop,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn counting_concordance_influence_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    Ok(counting_concordance_influence_rows_for_vectors(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None, ymin=None, ymax=None))]
pub fn stratified_counting_concordance_influence_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    ymin: Option<f64>,
    ymax: Option<f64>,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_counting_concordance_inputs(&start, &stop, &status, &risk_scores, weights.as_deref())?;
    validate_concordance_horizons(ymin, ymax)?;
    let (start, stop) = prepare_counting_concordance_times(start, stop, timefix, ymin)?;
    validate_strata_length(start.len(), &strata, "start")?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    Ok(stratified_counting_concordance_influence_rows_for_strata(
        &start,
        &stop,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        ymax,
    ))
}

#[pyfunction]
pub fn concordance(
    y: Vec<f64>,
    x: Vec<i32>,
    wt: Vec<f64>,
    timewt: Vec<f64>,
    sortstart: Option<Vec<usize>>,
    sortstop: Vec<usize>,
) -> PyResult<Py<PyDict>> {
    validate_legacy_concordance_inputs(&y, &x, &wt, &timewt, sortstart.as_deref(), &sortstop)?;
    let n = y.len();
    let ntree = x.iter().map(|&value| value as usize).max().unwrap_or(0) + usize::from(n > 0);
    let mut nwt = vec![0.0; ntree];
    let mut twt = vec![0.0; ntree];
    let mut count = vec![0.0; CONCORDANCE_COUNT_SIZE];
    let mut utime = 0;
    let i2 = 0;
    let mut i = 0;
    while i < n {
        let ii = sortstop[i];
        let current_time = y[ii];
        let should_skip = match sortstart.as_ref() {
            Some(ss) if i2 < n => y[ss[i2]] >= current_time,
            _ => false,
        };
        if should_skip || y[ii] == 0.0 {
            addin(&mut nwt, &mut twt, x[ii] as usize, wt[ii]);
            i += 1;
        } else {
            let mut ndeath = 0;
            let mut _dwt = 0.0;
            let mut _dwt2 = 0.0;
            let xsave = x[ii];
            let adjtimewt = timewt[utime];
            utime += 1;
            while i + ndeath < n && y[sortstop[i + ndeath]] == current_time {
                let jj = sortstop[i + ndeath];
                if x[jj] == xsave {
                    count[2] += 1.0;
                } else if i > PARALLEL_THRESHOLD_LARGE {
                    let (concordant, discordant): (f64, f64) = (0..i)
                        .into_par_iter()
                        .map(|k| {
                            let kk = sortstop[k];
                            if x[kk] != x[jj] {
                                if (x[kk] < x[jj] && y[kk] > current_time)
                                    || (x[kk] > x[jj] && y[kk] < current_time)
                                {
                                    (1.0, 0.0)
                                } else {
                                    (0.0, 1.0)
                                }
                            } else {
                                (0.0, 0.0)
                            }
                        })
                        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                    count[0] += concordant;
                    count[1] += discordant;
                } else {
                    for &kk in &sortstop[..i] {
                        if x[kk] != x[jj] {
                            if (x[kk] < x[jj] && y[kk] > current_time)
                                || (x[kk] > x[jj] && y[kk] < current_time)
                            {
                                count[0] += 1.0;
                            } else {
                                count[1] += 1.0;
                            }
                        }
                    }
                }
                _dwt += wt[jj];
                _dwt2 += wt[jj] * adjtimewt;
                ndeath += 1;
            }
            count[4] += (ndeath as f64) * (ndeath as f64 - 1.0) / 2.0;
            for &jj in &sortstop[i..i + ndeath] {
                addin(&mut nwt, &mut twt, x[jj] as usize, wt[jj]);
            }
            i += ndeath;
        }
    }
    count[3] -= count[4];
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("count", count)?;
        Ok(dict.into())
    })
}
#[inline]
fn addin(nwt: &mut [f64], twt: &mut [f64], x: usize, weight: f64) {
    nwt[x] += weight;
    let mut node_index = x;
    while node_index != 0 {
        let parent_index = (node_index - 1) / 2;
        twt[parent_index] += weight;
        node_index = parent_index;
    }
    twt[x] += weight;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[cfg(feature = "python")]
    fn summary_value(summary: &Py<PyDict>, key: &str) -> f64 {
        Python::attach(|py| {
            summary
                .bind(py)
                .get_item(key)
                .unwrap()
                .unwrap()
                .extract()
                .unwrap()
        })
    }

    #[test]
    #[cfg(feature = "python")]
    fn native_right_outputs_share_r_time_canonicalization() {
        initialize_python();
        for time in [
            vec![1.0, 1.0 + 0.75e-9, 1.0 + 1.5e-9],
            vec![1.0, 1.0 + 5e-9, 2.0],
            vec![1e9, 1e9 + 5.0, 1e9 + 30.0],
            vec![1.0, 1.0 + 1e-8, 1.0 + 2e-8],
        ] {
            for timefix in [false, true] {
                let status = vec![1, 1, 0];
                let risk = vec![2.0, 3.0, 1.0];
                let index = concordance_index(
                    time.clone(),
                    status.clone(),
                    risk.clone(),
                    None,
                    "n".into(),
                    timefix,
                    None,
                    None,
                )
                .unwrap();
                let summary = concordance_summary(
                    time.clone(),
                    status.clone(),
                    risk.clone(),
                    None,
                    "n".into(),
                    timefix,
                    None,
                    None,
                )
                .unwrap();
                let (influence, dfbeta, variance) = concordance_influence_rows(
                    time.clone(),
                    status.clone(),
                    risk.clone(),
                    None,
                    "n".into(),
                    timefix,
                    None,
                    None,
                )
                .unwrap();
                let ranks = concordance_rank_rows(
                    time.clone(),
                    status,
                    risk,
                    None,
                    "n".into(),
                    timefix,
                    None,
                    None,
                )
                .unwrap();
                let comparable = if timefix { 2.0 } else { 3.0 };
                assert_eq!(summary_value(&summary, "concordant"), 2.0);
                assert_eq!(summary_value(&summary, "comparable"), comparable);
                assert_eq!(summary_value(&summary, "tied_y"), f64::from(timefix));
                assert_eq!(index, 2.0 / comparable);
                assert_eq!(index, summary_value(&summary, "concordance"));
                let from_rows: f64 = influence
                    .iter()
                    .map(|row| row[..3].iter().sum::<f64>() / 2.0)
                    .sum();
                assert_eq!(from_rows, comparable);
                assert_eq!(ranks[0].0 == ranks[1].0, timefix);
                if timefix {
                    assert_eq!(dfbeta, vec![0.0; 3]);
                    assert_eq!(variance, 0.0);
                } else {
                    assert!((variance - 2.0 / 27.0).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "python")]
    fn native_counting_canonicalizes_before_clipping_lower_limit() {
        initialize_python();
        let start = vec![0.0, 2.0 - 5e-13, 0.0];
        let stop = vec![1.0, 3.0, 4.0];
        let status = vec![1, 1, 0];
        let risk = vec![-2.0, -1.0, -3.0];
        let summary = counting_concordance_summary(
            start.clone(),
            stop.clone(),
            status.clone(),
            risk.clone(),
            None,
            "n".into(),
            None,
            Some(2.0),
            None,
        )
        .unwrap();
        let index = counting_concordance_index(
            start.clone(),
            stop.clone(),
            status.clone(),
            risk.clone(),
            None,
            "n".into(),
            None,
            Some(2.0),
            None,
        )
        .unwrap();
        let (_, _, variance) = counting_concordance_influence_rows(
            start.clone(),
            stop.clone(),
            status.clone(),
            risk.clone(),
            None,
            "n".into(),
            None,
            Some(2.0),
            None,
        )
        .unwrap();
        let ranks = counting_concordance_rank_rows(
            start,
            stop,
            status,
            risk,
            None,
            "n".into(),
            None,
            Some(2.0),
            None,
        )
        .unwrap();
        assert_eq!(summary_value(&summary, "concordant"), 2.0);
        assert_eq!(summary_value(&summary, "comparable"), 3.0);
        assert_eq!(index, 2.0 / 3.0);
        assert!((variance - 2.0 / 27.0).abs() < 1e-14);
        assert_eq!(ranks, vec![(2.0, 0.0, 3.0, 1.0), (3.0, 0.5, 2.0, 1.0)]);
    }

    #[test]
    fn cutoff_preserves_original_unweighted_event_fallback() {
        for (status, weights, comparable) in [
            ([1, 1, 0], [2.0, 3.0, 5.0], 1.6),
            ([1, 0, 0], [2.0, 3.0, 5.0], 16.0),
            ([1, 1, 0], [2.0, 0.0, 5.0], 10.0 / 7.0),
        ] {
            let time = [1.0, 2.0, 3.0];
            let risk = [3.0, 2.0, 1.0];
            let right = right_concordance_summary_counts_for_vectors(
                &time,
                &status,
                &risk,
                Some(&weights),
                ConcordanceTimeWeight::I,
                Some(1.0),
            );
            let counting = counting_concordance_summary_counts_for_vectors(
                &[0.0; 3],
                &time,
                &status,
                &risk,
                Some(&weights),
                ConcordanceTimeWeight::I,
                Some(1.0),
            );
            for (summary, _, _, _) in [right, counting] {
                assert!((summary.comparable - comparable).abs() < 1e-14);
                assert_eq!(summary.concordant, summary.comparable);
            }
            assert_eq!(retained_concordance_events(&time, &status, Some(1.0)), 1.0);
        }
    }

    #[test]
    fn stratified_fallback_and_empty_comparisons_use_pooled_semantics() {
        let time = [1.0, 2.0, 1.0, 2.0];
        let status = [1, 0, 1, 0];
        let risk = [3.0, 2.0, 1.0, 2.0];
        let weights = [2.0, 1.0, 3.0, 2.0];
        let strata = [0, 0, 1, 1];
        let (summary, _, _, _, _) = stratified_right_concordance_summary_counts(
            &time,
            &status,
            &risk,
            &strata,
            Some(&weights),
            ConcordanceTimeWeight::I,
            None,
        );
        assert_eq!(summary.concordant, 2.0);
        assert_eq!(summary.comparable, 8.0);
        let (_, dfbeta, variance) = stratified_right_concordance_influence_rows(
            &time,
            &status,
            &risk,
            &strata,
            Some(&weights),
            ConcordanceTimeWeight::I,
            Some(0.0),
        );
        assert!(dfbeta.iter().all(|value| value.is_nan()));
        assert!(variance.is_nan());
        let (summary, _, n_event, numerator, comparable) =
            stratified_counting_concordance_summary_counts(
                &[-1.0; 4],
                &time,
                &status,
                &risk,
                &strata,
                Some(&weights),
                ConcordanceTimeWeight::I,
                Some(0.0),
            );
        assert_eq!(n_event, 0.0);
        assert!((summary.concordant / summary.comparable).is_nan());
        assert!(conditional_variance_from_numerator(numerator, comparable).is_nan());
        let (_, _, numerator, comparable) = right_concordance_summary_counts_for_vectors(
            &[1.0, 1.0],
            &[1, 1],
            &[1.0, 2.0],
            None,
            ConcordanceTimeWeight::N,
            None,
        );
        assert_eq!(
            conditional_variance_from_numerator(numerator, comparable),
            f64::INFINITY
        );
    }

    #[test]
    fn signed_zero_times_are_exact_ties() {
        let (summary, _, _, _) = right_concordance_summary_counts_for_vectors(
            &[-0.0, 0.0],
            &[1, 0],
            &[2.0, 1.0],
            None,
            ConcordanceTimeWeight::N,
            None,
        );
        assert_eq!(summary.concordant, 1.0);
        assert_eq!(summary.comparable, 1.0);
    }

    #[test]
    fn infinite_and_reversed_horizons_match_r_boundaries() {
        initialize_python();
        let status = [1, 1, 0];
        let risk = [-3.0, -1.0, -2.0];
        for (ymin, ymax, expected_comparable) in [
            (Some(f64::INFINITY), None, 2.0),
            (Some(f64::NEG_INFINITY), Some(f64::INFINITY), 3.0),
            (Some(f64::INFINITY), Some(2.0), 0.0),
            (Some(3.0), Some(2.0), 0.0),
            (None, Some(f64::NEG_INFINITY), 0.0),
        ] {
            validate_concordance_horizons(ymin, ymax).unwrap();
            let time = prepare_right_concordance_times(vec![1.0, 2.0, 3.0], true, ymin).unwrap();
            let (summary, _, _, _) = right_concordance_summary_counts_for_vectors(
                &time,
                &status,
                &risk,
                None,
                ConcordanceTimeWeight::N,
                ymax,
            );
            let (start, stop) =
                prepare_counting_concordance_times(vec![0.0; 3], vec![1.0, 2.0, 3.0], None, ymin)
                    .unwrap();
            let counting = counting_concordance_summary_counts_for_vectors(
                &start,
                &stop,
                &status,
                &risk,
                None,
                ConcordanceTimeWeight::N,
                ymax,
            )
            .0;
            assert_eq!(summary.comparable, expected_comparable);
            assert_eq!(counting.comparable, expected_comparable);
            assert_eq!(counting.concordant, summary.concordant);
        }
        assert!(validate_concordance_horizons(Some(f64::NAN), None).is_err());
        assert!(validate_concordance_horizons(None, Some(f64::NAN)).is_err());
    }

    #[test]
    fn validate_right_concordance_rejects_malformed_inputs() {
        initialize_python();

        let status_err = validate_right_concordance_inputs(&[1.0, 2.0], &[1, 2], &[0.4, 0.1], None)
            .expect_err("non-binary status should be rejected");
        assert!(
            status_err
                .to_string()
                .contains("status must contain only 0/1")
        );

        let time_err =
            validate_right_concordance_inputs(&[1.0, f64::INFINITY], &[1, 0], &[0.4, 0.1], None)
                .expect_err("non-finite time should be rejected");
        assert!(time_err.to_string().contains("time contains non-finite"));

        let risk_err =
            validate_right_concordance_inputs(&[1.0, 2.0], &[1, 0], &[0.4, f64::NAN], None)
                .expect_err("NaN risk score should be rejected");
        assert!(risk_err.to_string().contains("risk_scores contains NaN"));

        let weight_err = validate_right_concordance_inputs(
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            Some(&[1.0, -1.0]),
        )
        .expect_err("negative weights should be rejected");
        assert!(weight_err.to_string().contains("weights contains negative"));
    }

    #[test]
    fn validate_counting_concordance_rejects_malformed_inputs() {
        initialize_python();
        let interval_err = validate_counting_concordance_inputs(
            &[0.0, 2.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            None,
        )
        .expect_err("zero-width counting interval should be rejected");
        assert!(
            interval_err
                .to_string()
                .contains("start must be less than stop")
        );
        let weight_err = validate_counting_concordance_inputs(
            &[0.0, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            Some(&[1.0, f64::NAN]),
        )
        .expect_err("NaN weights should be rejected");
        assert!(weight_err.to_string().contains("weights contains NaN"));
        validate_counting_concordance_inputs(
            &[-2.0, -1.0],
            &[-1.0, 0.0],
            &[1, 0],
            &[0.4, 0.1],
            None,
        )
        .expect("finite negative times are valid survival times");
    }

    #[test]
    fn validate_counting_concordance_honors_exact_timefix() {
        initialize_python();
        let start = vec![1.0];
        let stop = vec![1.0 + 5e-10];
        for timefix in [None, Some(true)] {
            let error =
                prepare_counting_concordance_times(start.clone(), stop.clone(), timefix, None)
                    .expect_err("corrected interval has effective length zero");
            assert!(error.to_string().contains("effective length 0"));
        }
        let prepared =
            prepare_counting_concordance_times(start.clone(), stop.clone(), Some(false), None)
                .expect("exact timefix accepts every strictly positive interval");
        assert_eq!(prepared, (start, stop));
    }

    #[test]
    fn concordance_rank_rows_report_weighted_event_rows() {
        initialize_python();

        let rows = concordance_rank_rows(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 0],
            vec![0.9, 0.6, 0.4, 0.1],
            Some(vec![2.0, 1.0, 3.0, 1.0]),
            "n".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], (1.0, 5.0 / 7.0, 7.0, 2.0));
        assert_eq!(rows[1], (2.0, 4.0 / 5.0, 5.0, 1.0));
        assert_eq!(rows[2], (3.0, 0.25, 4.0, 3.0));
    }

    #[test]
    fn concordance_rank_rows_group_near_tied_event_times() {
        initialize_python();

        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.1, 0.5, 0.2];
        let weights = Some(vec![2.0, 1.0, 3.0, 1.0]);
        let exact = concordance_rank_rows(
            vec![1.0, 1.0, 2.0, 3.0],
            status.clone(),
            risk.clone(),
            weights.clone(),
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();
        let near = concordance_rank_rows(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            status,
            risk,
            weights,
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(near.len(), exact.len());
        for (actual, expected) in near.iter().zip(exact.iter()) {
            assert!((actual.1 - expected.1).abs() < 1e-12);
            assert!((actual.2 - expected.2).abs() < 1e-12);
            assert!((actual.3 - expected.3).abs() < 1e-12);
        }
    }

    #[test]
    fn concordance_rank_rows_unweighted_matches_unit_weights() {
        initialize_python();

        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.6, 0.4, 0.1];
        let unweighted = concordance_rank_rows(
            time.clone(),
            status.clone(),
            risk.clone(),
            None,
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();
        let unit_weighted = concordance_rank_rows(
            time,
            status,
            risk,
            Some(vec![1.0; 4]),
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(unweighted, unit_weighted);

        let start = vec![0.0, 0.0, 0.5, 1.5];
        let stop = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.7, 0.4, 0.1];
        let unweighted = counting_concordance_rank_rows(
            start.clone(),
            stop.clone(),
            status.clone(),
            risk.clone(),
            None,
            "S".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
        let unit_weighted = counting_concordance_rank_rows(
            start,
            stop,
            status,
            risk,
            Some(vec![1.0; 4]),
            "S".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(unweighted, unit_weighted);
    }

    #[test]
    fn concordance_rank_rows_sweep_removes_earlier_times() {
        initialize_python();

        let rows = concordance_rank_rows(
            vec![3.0, 1.0, 2.0, 1.0 + 5e-10, 0.5],
            vec![0, 1, 1, 1, 0],
            vec![0.2, 0.9, 0.5, 0.1, 0.8],
            Some(vec![1.0, 2.0, 3.0, 1.0, 4.0]),
            "n".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].0, 1.0);
        assert!((rows[0].1 - 5.0 / 7.0).abs() < 1e-12);
        assert_eq!(rows[0].2, 7.0);
        assert_eq!(rows[0].3, 2.0);
        assert_eq!(rows[1].0, 1.0);
        assert!((rows[1].1 + 6.0 / 7.0).abs() < 1e-12);
        assert_eq!(rows[1].2, 7.0);
        assert_eq!(rows[1].3, 1.0);
        assert_eq!(rows[2].0, 2.0);
        assert_eq!(rows[2].1, 0.25);
        assert_eq!(rows[2].2, 4.0);
        assert_eq!(rows[2].3, 3.0);
    }

    #[test]
    fn counting_concordance_rank_rows_use_delayed_entry_risk_sets() {
        initialize_python();

        let rows = counting_concordance_rank_rows(
            vec![0.0, 0.0, 0.5, 1.5],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 0],
            vec![0.9, 0.7, 0.4, 0.1],
            None,
            "n".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], (1.0, 2.0 / 3.0, 3.0, 1.0));
        assert_eq!(rows[1], (2.0, 2.0 / 3.0, 3.0, 1.0));
        assert_eq!(rows[2], (3.0, 1.0 / 2.0, 2.0, 1.0));
    }

    #[test]
    fn counting_concordance_time_weights_sweep_duplicate_event_times() {
        initialize_python();

        let start = vec![0.0, 0.0, 0.25, 0.0, 1.0];
        let stop = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 0, 1];
        let weights = vec![2.0, 1.0, 3.0, 0.5, 4.0];

        let multipliers = counting_concordance_time_weight_multipliers(
            &start,
            &stop,
            &status,
            Some(&weights),
            ConcordanceTimeWeight::S,
            None,
        );

        assert_eq!(multipliers.len(), 3);
        assert_eq!(multipliers[0].0, 1.0);
        assert_eq!(multipliers[1].0, 2.0);
        assert_eq!(multipliers[2].0, 3.0);
        assert!((multipliers[0].1 - 21.0 / 13.0).abs() < 1e-12);
        assert!((multipliers[1].1 - 49.0 / 65.0).abs() < 1e-12);
        assert!((multipliers[2].1 - 441.0 / 520.0).abs() < 1e-12);
    }

    #[test]
    fn counting_concordance_rank_rows_share_duplicate_event_time_weight() {
        initialize_python();

        let rows = counting_concordance_rank_rows(
            vec![0.0, 0.0, 0.25, 0.0, 1.0],
            vec![1.0, 1.0, 2.0, 2.0, 3.0],
            vec![1, 1, 1, 0, 1],
            vec![0.9, 0.2, 0.7, 0.1, 0.8],
            Some(vec![2.0, 1.0, 3.0, 0.5, 4.0]),
            "S".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].0, 1.0);
        assert!((rows[0].1 - 9.0 / 13.0).abs() < 1e-12);
        assert!((rows[0].2 - 10.5).abs() < 1e-12);
        assert_eq!(rows[0].3, 2.0);
        assert_eq!(rows[1].0, 1.0);
        assert!((rows[1].1 + 9.0 / 13.0).abs() < 1e-12);
        assert!((rows[1].2 - 10.5).abs() < 1e-12);
        assert_eq!(rows[1].3, 1.0);
        assert_eq!(rows[2].0, 2.0);
        assert!((rows[2].1 + 7.0 / 15.0).abs() < 1e-12);
        assert!((rows[2].2 - 147.0 / 26.0).abs() < 1e-12);
        assert_eq!(rows[2].3, 3.0);
        assert_eq!(rows[3].0, 3.0);
        assert_eq!(rows[3].1, 0.0);
        assert!((rows[3].2 - 441.0 / 130.0).abs() < 1e-12);
        assert_eq!(rows[3].3, 4.0);
    }

    #[test]
    fn fused_summaries_match_weighted_r_counts() {
        let start = [0.0, 0.0, 0.0, 1.0, 0.0, 2.0];
        let stop = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let status = [1, 0, 1, 1, 0, 1];
        let risk = [3.0, 3.0, 2.0, 1.0, 2.0, 0.0];
        let weights = [2.0, 1.0, 0.0, 1.5, 3.0, 0.5];

        // survival 3.8.11 concordancefit, reverse=TRUE, timefix=FALSE.
        for (time_weight, concordant, comparable, tied_x) in [
            (ConcordanceTimeWeight::N, 11.75, 17.25, 2.0),
            (ConcordanceTimeWeight::S, 11.9, 18.3, 2.0),
            (ConcordanceTimeWeight::SOverG, 12.08, 19.56, 2.0),
            (ConcordanceTimeWeight::NOverG2, 12.08, 19.56, 2.0),
            (ConcordanceTimeWeight::I, 1.525, 2.55, 0.25),
        ] {
            let (summary, ties, _, _) = right_concordance_summary_counts_for_vectors(
                &stop,
                &status,
                &risk,
                Some(&weights),
                time_weight,
                None,
            );
            let scalar = right_concordance_sweep::<false>(
                &stop,
                &status,
                &risk,
                Some(&weights),
                time_weight,
                None,
            )
            .0;
            assert_eq!(scalar, summary);
            assert!((summary.concordant - concordant).abs() < 1e-12);
            assert!((summary.comparable - comparable).abs() < 1e-12);
            assert!((ties.tied_x - tied_x).abs() < 1e-12);
        }
        for (time_weight, concordant, comparable, tied_x) in [
            (ConcordanceTimeWeight::N, 7.0, 12.5, 2.0),
            (ConcordanceTimeWeight::S, 28.0 / 3.0, 16.0, 8.0 / 3.0),
            (ConcordanceTimeWeight::I, 7.0 / 6.0, 7.0 / 3.0, 1.0 / 3.0),
        ] {
            let (summary, ties, _, _) = counting_concordance_summary_counts_for_vectors(
                &start,
                &stop,
                &status,
                &risk,
                Some(&weights),
                time_weight,
                None,
            );
            let scalar = counting_concordance_sweep::<false>(
                &start,
                &stop,
                &status,
                &risk,
                Some(&weights),
                time_weight,
                None,
            )
            .0;
            assert_eq!(scalar, summary);
            assert!((summary.concordant - concordant).abs() < 1e-12);
            assert!((summary.comparable - comparable).abs() < 1e-12);
            assert!((ties.tied_x - tied_x).abs() < 1e-12);
        }
    }

    #[test]
    fn fused_summaries_preserve_small_concordant_weight() {
        // Forming concordant=(signed+comparable)/2 would cancel the small
        // concordant contribution against the large discordant contribution.
        let start = [0.0; 3];
        let stop = [1.0; 3];
        let status = [1, 0, 0];
        let risk = [1.0, 0.0, 2.0];
        let weights = [1.0, 1e-8, 1e8];
        let right = right_concordance_summary_counts_for_vectors(
            &stop,
            &status,
            &risk,
            Some(&weights),
            ConcordanceTimeWeight::N,
            None,
        );
        let counting = counting_concordance_summary_counts_for_vectors(
            &start,
            &stop,
            &status,
            &risk,
            Some(&weights),
            ConcordanceTimeWeight::N,
            None,
        );
        for (summary, ties, _, _) in [right, counting] {
            assert!((summary.concordant - 1e-8).abs() < 1e-22);
            assert!((summary.comparable - 1e8).abs() < 1e-7);
            assert_eq!(ties, ConcordanceTieCounts::default());
        }
    }

    #[test]
    fn stratified_concordance_summary_counts_within_strata() {
        initialize_python();

        let (summary, ties, n_event, _, _) = stratified_right_concordance_summary_counts(
            &[1.0, 2.0, 1.0, 2.0],
            &[1, 0, 1, 0],
            &[0.9, 0.1, 0.2, 0.8],
            &[0, 0, 1, 1],
            None,
            ConcordanceTimeWeight::N,
            None,
        );

        assert_eq!(n_event, 2.0);
        assert_eq!(summary.concordant, 1.0);
        assert_eq!(summary.comparable, 2.0);
        assert_eq!(summary.c_index(), 0.5);
        assert_eq!(ties, ConcordanceTieCounts::default());
    }

    #[test]
    fn right_concordance_tie_counts_separate_risk_and_event_time_ties() {
        let (_, risk_tie, _, _) = right_concordance_summary_counts_for_vectors(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 0, 1],
            &[0.2, 0.4, 0.4, 1.0],
            None,
            ConcordanceTimeWeight::N,
            None,
        );
        assert_eq!(
            risk_tie,
            ConcordanceTieCounts {
                tied_x: 1.0,
                tied_y: 0.0,
                tied_xy: 0.0,
            }
        );

        let (_, event_time_tie, _, _) = right_concordance_summary_counts_for_vectors(
            &[1.0, 2.0, 2.0, 2.0, 4.0],
            &[1, 1, 1, 1, 1],
            &[0.2, 0.4, 0.4, 0.8, 1.0],
            None,
            ConcordanceTimeWeight::N,
            None,
        );
        assert_eq!(
            event_time_tie,
            ConcordanceTieCounts {
                tied_x: 0.0,
                tied_y: 2.0,
                tied_xy: 1.0,
            }
        );

        let (_, distinct_scores, _, _) = right_concordance_summary_counts_for_vectors(
            &[1.0, 2.0, 3.0],
            &[1, 0, 0],
            &[0.0, 0.5e-12, 1e-12],
            None,
            ConcordanceTimeWeight::N,
            None,
        );
        assert_eq!(distinct_scores.tied_x, 0.0);
    }

    #[test]
    fn right_concordance_tie_counts_apply_case_and_time_weights() {
        let (_, counts, _, _) = right_concordance_summary_counts_for_vectors(
            &[1.0, 2.0, 2.0, 3.0],
            &[1, 1, 1, 0],
            &[0.1, 0.5, 0.5, 0.5],
            Some(&[1.0, 2.0, 3.0, 4.0]),
            ConcordanceTimeWeight::I,
            None,
        );

        assert!((counts.tied_x - 20.0 / 9.0).abs() < 1e-12);
        assert!((counts.tied_xy - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(counts.tied_y, 0.0);
    }

    #[test]
    fn right_concordance_conditional_variance_matches_reference() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let status = [1, 1, 0, 1];
        let risk = [0.2, 0.4, 0.4, 1.0];
        let (_, _, numerator, comparable_pair_weight) =
            right_concordance_summary_counts_for_vectors(
                &time,
                &status,
                &risk,
                None,
                ConcordanceTimeWeight::N,
                None,
            );

        let variance = conditional_variance_from_numerator(numerator, comparable_pair_weight);
        assert!((variance - 0.065).abs() < 1e-12);
    }

    #[test]
    fn counting_concordance_conditional_variance_matches_reference() {
        let start = [0.0, 0.0, 0.25, 0.5, 0.0, 1.0, 1.0];
        let stop = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let status = [1, 1, 1, 1, 0, 1, 0];
        let risk = [0.9, 0.2, 0.7, 0.4, 0.1, 0.8, 0.3];
        let (_, _, numerator, comparable_pair_weight) =
            counting_concordance_summary_counts_for_vectors(
                &start,
                &stop,
                &status,
                &risk,
                None,
                ConcordanceTimeWeight::N,
                None,
            );

        let variance = conditional_variance_from_numerator(numerator, comparable_pair_weight);
        assert!((variance - 0.075).abs() < 1e-12, "variance={variance}");
    }

    #[test]
    fn stratified_counting_concordance_summary_counts_within_strata() {
        initialize_python();

        let (summary, _, n_event, _, _) = stratified_counting_concordance_summary_counts(
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 2.0, 1.0, 2.0],
            &[1, 0, 1, 0],
            &[0.9, 0.1, 0.2, 0.8],
            &[0, 0, 1, 1],
            None,
            ConcordanceTimeWeight::N,
            None,
        );

        assert_eq!(n_event, 2.0);
        assert_eq!(summary.concordant, 1.0);
        assert_eq!(summary.comparable, 2.0);
        assert_eq!(summary.c_index(), 0.5);
    }

    #[test]
    fn stratified_concordance_rank_rows_preserve_within_strata_ranks() {
        initialize_python();

        let rows = stratified_concordance_rank_rows(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 1, 0],
            vec![0.9, 0.1, 0.2, 0.8],
            vec![0, 0, 1, 1],
            None,
            "n".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows, vec![(1.0, 0.5, 2.0, 1.0), (1.0, -0.5, 2.0, 1.0)]);
    }

    #[test]
    fn stratified_counting_concordance_rank_rows_preserve_within_strata_ranks() {
        initialize_python();

        let rows = stratified_counting_concordance_rank_rows(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 1, 0],
            vec![0.9, 0.1, 0.2, 0.8],
            vec![0, 0, 1, 1],
            None,
            "n".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rows, vec![(1.0, 0.5, 2.0, 1.0), (1.0, -0.5, 2.0, 1.0)]);
    }

    #[test]
    fn stratified_concordance_influence_rows_remap_to_original_rows() {
        initialize_python();

        let (influence, dfbeta, variance) = stratified_concordance_influence_rows(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 1, 0],
            vec![0.9, 0.1, 0.2, 0.8],
            vec![0, 0, 1, 1],
            None,
            "n".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(influence[0], vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(dfbeta, vec![0.25, 0.25, -0.25, -0.25]);
        assert_eq!(variance, 0.25);
    }

    #[test]
    fn stratified_counting_concordance_influence_rows_remap_to_original_rows() {
        initialize_python();

        let (influence, dfbeta, variance) = stratified_counting_concordance_influence_rows(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 1, 0],
            vec![0.9, 0.1, 0.2, 0.8],
            vec![0, 0, 1, 1],
            None,
            "n".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(influence[0], vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(dfbeta, vec![0.25, 0.25, -0.25, -0.25]);
        assert_eq!(variance, 0.25);
    }

    #[test]
    fn concordance_influence_rows_report_dfbeta_and_variance() {
        initialize_python();

        let (influence, dfbeta, variance) = concordance_influence_rows(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 0],
            vec![0.9, 0.1, 0.4, 0.2],
            Some(vec![2.0, 1.0, 3.0, 1.0]),
            "n".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(influence[0], vec![5.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![2.0, 4.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![3.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![5.0, 1.0, 0.0, 0.0, 0.0]);
        assert!((dfbeta[0] - 40.0 / 289.0).abs() < 1e-12);
        assert!((dfbeta[1] + 44.0 / 289.0).abs() < 1e-12);
        assert!((variance - dfbeta.iter().map(|value| value * value).sum::<f64>()).abs() < 1e-12);
    }

    #[test]
    fn concordance_influence_rows_group_near_tied_event_times() {
        initialize_python();

        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.1, 0.5, 0.2];
        let weights = Some(vec![2.0, 1.0, 3.0, 1.0]);
        let exact = concordance_influence_rows(
            vec![1.0, 1.0, 2.0, 3.0],
            status.clone(),
            risk.clone(),
            weights.clone(),
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();
        let near = concordance_influence_rows(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            status,
            risk,
            weights,
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(near.0.len(), exact.0.len());
        for (actual_row, expected_row) in near.0.iter().zip(exact.0.iter()) {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
        for (actual, expected) in near.1.iter().zip(exact.1.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((near.2 - exact.2).abs() < 1e-12);
    }

    #[test]
    fn concordance_influence_rows_unweighted_matches_unit_weights() {
        initialize_python();

        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.1, 0.4, 0.2];
        let unweighted = concordance_influence_rows(
            time.clone(),
            status.clone(),
            risk.clone(),
            None,
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();
        let unit_weighted = concordance_influence_rows(
            time,
            status,
            risk,
            Some(vec![1.0; 4]),
            "S".to_string(),
            true,
            None,
            None,
        )
        .unwrap();

        assert_eq!(unweighted, unit_weighted);

        let start = vec![0.0, 0.0, 0.5, 1.5];
        let stop = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 1, 0];
        let risk = vec![0.9, 0.1, 0.4, 0.2];
        let unweighted = counting_concordance_influence_rows(
            start.clone(),
            stop.clone(),
            status.clone(),
            risk.clone(),
            None,
            "S".to_string(),
            None,
            None,
            None,
        )
        .unwrap();
        let unit_weighted = counting_concordance_influence_rows(
            start,
            stop,
            status,
            risk,
            Some(vec![1.0; 4]),
            "S".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(unweighted, unit_weighted);
    }

    #[test]
    fn counting_concordance_influence_rows_use_delayed_entry_risk_sets() {
        initialize_python();

        let (influence, dfbeta, variance) = counting_concordance_influence_rows(
            vec![0.0, 0.0, 0.5, 1.5],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 1, 0],
            vec![0.9, 0.1, 0.4, 0.2],
            None,
            "n".to_string(),
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(influence[0], vec![2.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![1.0, 2.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![2.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![1.0, 1.0, 0.0, 0.0, 0.0]);
        assert!((dfbeta[0] - 0.16).abs() < 1e-12);
        assert!((dfbeta[1] + 0.16).abs() < 1e-12);
        assert!((variance - dfbeta.iter().map(|value| value * value).sum::<f64>()).abs() < 1e-12);
    }
}
