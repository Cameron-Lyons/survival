use crate::constants::{CONCORDANCE_COUNT_SIZE, PARALLEL_THRESHOLD_LARGE, TIME_EPSILON, same_time};
use crate::internal::fenwick::FenwickTree;
use crate::internal::statistical::{
    ConcordanceSummary, ConcordanceTimeWeight, concordance_index_with_horizon,
    concordance_summary_with_horizon, concordance_summary_with_horizon_and_weights,
    concordance_summary_with_horizon_weights_and_time_weight, counting_process_concordance_index,
    counting_process_concordance_summary, counting_process_concordance_summary_with_weights,
    counting_process_concordance_summary_with_weights_and_time_weight,
};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::BTreeMap;

type ConcordanceRankRows = Vec<(f64, f64, f64, f64)>;
type IndexedConcordanceRankRows = Vec<(usize, f64, f64, f64, f64)>;
type ConcordanceRankRowsWithIndices = (ConcordanceRankRows, Vec<usize>);
type ConcordanceInfluenceOutput = (Vec<Vec<f64>>, Vec<f64>, f64);
type ConcordanceCountRows = Vec<[f64; 5]>;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ConcordanceTieCounts {
    tied_x: f64,
    tied_y: f64,
    tied_xy: f64,
}

fn concordance_count_row(summary: ConcordanceSummary, ties: ConcordanceTieCounts) -> [f64; 5] {
    let concordant = summary.concordant - 0.5 * ties.tied_x;
    let discordant = summary.comparable - concordant - ties.tied_x;
    [
        concordant,
        discordant.max(0.0),
        ties.tied_x,
        ties.tied_y,
        ties.tied_xy,
    ]
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
    validate_non_negative(time, "time")?;
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
    timefix: Option<bool>,
) -> PyResult<()> {
    if start.len() != stop.len() || start.len() != status.len() || start.len() != risk_scores.len()
    {
        return Err(PyValueError::new_err(
            "start, stop, status, and risk_scores must have the same length",
        ));
    }
    validate_no_nan(start, "start")?;
    validate_finite(start, "start")?;
    validate_non_negative(start, "start")?;
    validate_no_nan(stop, "stop")?;
    validate_finite(stop, "stop")?;
    validate_non_negative(stop, "stop")?;
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
        let invalid = match timefix {
            Some(false) => entry >= exit,
            _ => entry >= exit - TIME_EPSILON,
        };
        if invalid {
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

fn prepare_counting_concordance_times(
    start: &[f64],
    stop: &[f64],
    timefix: Option<bool>,
) -> (Vec<f64>, Vec<f64>) {
    let mut prepared_start = start.to_vec();
    let mut prepared_stop = stop.to_vec();
    if timefix != Some(true) {
        return (prepared_start, prepared_stop);
    }

    let mut points = Vec::with_capacity(start.len() + stop.len());
    for (idx, &value) in prepared_start.iter().enumerate() {
        points.push((value, 0usize, idx));
    }
    for (idx, &value) in prepared_stop.iter().enumerate() {
        points.push((value, 1usize, idx));
    }
    points.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });

    let mut cursor = 0;
    while cursor < points.len() {
        let base = points[cursor].0;
        let mut scan = cursor + 1;
        while scan < points.len() && points[scan].0 - base < TIME_EPSILON {
            let (_, vector_idx, row_idx) = points[scan];
            if vector_idx == 0 {
                prepared_start[row_idx] = base;
            } else {
                prepared_stop[row_idx] = base;
            }
            scan += 1;
        }
        cursor = scan;
    }

    (prepared_start, prepared_stop)
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

fn effective_concordance_time_weight(
    status: &[i32],
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceTimeWeight {
    if status.iter().sum::<i32>() < 2 {
        ConcordanceTimeWeight::N
    } else {
        time_weight
    }
}

fn multiplier_at(multipliers: &[(f64, f64)], time: f64) -> f64 {
    match multipliers.binary_search_by(|(candidate, _)| candidate.total_cmp(&time)) {
        Ok(idx) => multipliers[idx].1,
        Err(idx) => {
            if idx < multipliers.len() && same_time(multipliers[idx].0, time) {
                multipliers[idx].1
            } else if idx > 0 && same_time(multipliers[idx - 1].0, time) {
                multipliers[idx - 1].1
            } else {
                0.0
            }
        }
    }
}

#[inline]
fn concordance_time_precedes(left: f64, right: f64) -> bool {
    left < right && !same_time(left, right)
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
) -> Vec<(f64, f64)> {
    right_concordance_time_weight_multipliers_by(time, status, weights, time_weight, same_time)
}

fn right_concordance_exact_time_weight_multipliers(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> Vec<(f64, f64)> {
    right_concordance_time_weight_multipliers_by(
        time,
        status,
        weights,
        time_weight,
        |left, right| left == right,
    )
}

fn right_concordance_time_weight_multipliers_by(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    times_match: impl Fn(f64, f64) -> bool,
) -> Vec<(f64, f64)> {
    let time_weight = effective_concordance_time_weight(status, time_weight);
    if time_weight == ConcordanceTimeWeight::N {
        let mut values: Vec<f64> = time
            .iter()
            .zip(status.iter())
            .filter_map(|(&time, &event)| (event == 1).then_some(time))
            .collect();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|left, right| times_match(*left, *right));
        return values.into_iter().map(|time| (time, 1.0)).collect();
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
        while group_end < time_order.len() && times_match(time[time_order[group_end]], event_time) {
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

        if death_weight > 0.0 {
            multipliers.push((
                event_time,
                concordance_time_weight_multiplier(
                    time_weight,
                    total_weight,
                    survival,
                    censoring_survival,
                    nrisk,
                ),
            ));
            if nrisk > 0.0 {
                survival *= ((nrisk - death_weight) / nrisk).max(0.0);
            }
        }
        let censoring_risk = (nrisk - death_weight).max(0.0);
        if censor_weight > 0.0 && censoring_risk > 0.0 {
            censoring_survival *= ((censoring_risk - censor_weight) / censoring_risk).max(0.0);
        }
        nrisk = (nrisk - group_weight).max(0.0);
        group_start = group_end;
    }
    multipliers
}

fn exact_multiplier_at(multipliers: &[(f64, f64)], time: f64) -> f64 {
    multipliers
        .binary_search_by(|(candidate, _)| candidate.total_cmp(&time))
        .map_or(0.0, |idx| multipliers[idx].1)
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

fn add_conditional_variance_observation(
    at_risk: &mut FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
    weight: f64,
    z2: &mut f64,
) {
    *z2 += conditional_variance_increment(at_risk, risk_levels, risk_score, weight);
    let rank = risk_levels.partition_point(|&risk| risk < risk_score);
    at_risk.update(rank, weight);
}

fn remove_conditional_variance_observation(
    at_risk: &mut FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
    weight: f64,
    z2: &mut f64,
) {
    let rank = risk_levels.partition_point(|&risk| risk < risk_score);
    at_risk.update(rank, -weight);
    *z2 -= conditional_variance_increment(at_risk, risk_levels, risk_score, weight);
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

fn right_concordance_tie_counts_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> (ConcordanceTieCounts, f64, f64) {
    let multipliers =
        right_concordance_exact_time_weight_multipliers(time, status, weights, time_weight);
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

        for &idx in &time_order[group_start..group_end] {
            if status[idx] != 1 {
                add_conditional_variance_observation(
                    &mut at_risk,
                    &risk_levels,
                    risk_scores[idx],
                    concordance_case_weight(weights, idx),
                    &mut z2,
                );
            }
        }

        let multiplier = exact_multiplier_at(&multipliers, event_time);
        if multiplier > 0.0 {
            let event_indices: Vec<usize> = time_order[group_start..group_end]
                .iter()
                .copied()
                .filter(|&idx| status[idx] == 1)
                .collect();
            for &event_idx in &event_indices {
                let lower_end = risk_levels.partition_point(|&risk| risk < risk_scores[event_idx]);
                let tied_end = risk_levels.partition_point(|&risk| risk <= risk_scores[event_idx]);
                let lower = rank_prefix_weight_before(&at_risk, lower_end);
                let lower_or_tied = rank_prefix_weight_before(&at_risk, tied_end);
                counts.tied_x += concordance_case_weight(weights, event_idx)
                    * (lower_or_tied - lower)
                    * multiplier;
            }

            if event_indices.len() > 1 {
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
            add_conditional_variance_observation(
                &mut at_risk,
                &risk_levels,
                risk_scores[idx],
                weight,
                &mut z2,
            );
            if status[idx] == 1 {
                death_weight += weight;
            }
        }
        let risk_weight = at_risk.total();
        if death_weight > 0.0 && multiplier > 0.0 && risk_weight > 0.0 {
            conditional_variance_numerator += death_weight * multiplier * z2 / risk_weight;
            comparable_pair_weight += death_weight * (risk_weight - death_weight) * multiplier;
        }
        group_start = group_end;
    }
    (
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
) -> Vec<(f64, f64)> {
    let time_weight = effective_concordance_time_weight(status, time_weight);
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
            multipliers.push((event_time, 1.0));
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
            concordance_time_weight_multiplier(time_weight, total_weight, survival, 1.0, nrisk),
        ));
        if nrisk > 0.0 {
            survival *= ((nrisk - death_weight) / nrisk).max(0.0);
        }
        group_start = group_end;
    }
    multipliers
}

fn counting_concordance_conditional_variance_numerator(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> (ConcordanceTieCounts, f64, f64) {
    let multipliers =
        counting_concordance_time_weight_multipliers(start, stop, status, weights, time_weight);
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
    let mut counts = ConcordanceTieCounts::default();
    let mut numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut event_group_start = 0usize;
    while event_group_start < event_indices.len() {
        let event_time = stop[event_indices[event_group_start]];
        while start_cursor < start_order.len() && start[start_order[start_cursor]] < event_time {
            let idx = start_order[start_cursor];
            if !active[idx] {
                add_conditional_variance_observation(
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
                remove_conditional_variance_observation(
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
        let mut death_weight = concordance_case_weight(weights, event_indices[event_group_start]);
        while event_group_end < event_indices.len()
            && stop[event_indices[event_group_end]] == event_time
        {
            death_weight += concordance_case_weight(weights, event_indices[event_group_end]);
            event_group_end += 1;
        }
        let risk_weight = at_risk.total();
        let multiplier = multiplier_at(&multipliers, event_time);
        if risk_weight > 0.0 && multiplier > 0.0 {
            let event_group = &event_indices[event_group_start..event_group_end];
            let mut events_by_risk = event_group.to_vec();
            events_by_risk.sort_by(|&left, &right| {
                risk_scores[left]
                    .total_cmp(&risk_scores[right])
                    .then_with(|| left.cmp(&right))
            });
            let mut risk_group_start = 0;
            while risk_group_start < events_by_risk.len() {
                let mut risk_group_end = risk_group_start + 1;
                while risk_group_end < events_by_risk.len()
                    && risk_scores[events_by_risk[risk_group_end]]
                        == risk_scores[events_by_risk[risk_group_start]]
                {
                    risk_group_end += 1;
                }
                let event_tied_weight: f64 = events_by_risk[risk_group_start..risk_group_end]
                    .iter()
                    .map(|&idx| concordance_case_weight(weights, idx))
                    .sum();
                for &event_idx in &events_by_risk[risk_group_start..risk_group_end] {
                    let lower_end =
                        risk_levels.partition_point(|&risk| risk < risk_scores[event_idx]);
                    let tied_end =
                        risk_levels.partition_point(|&risk| risk <= risk_scores[event_idx]);
                    let active_tied = rank_prefix_weight_before(&at_risk, tied_end)
                        - rank_prefix_weight_before(&at_risk, lower_end);
                    counts.tied_x += concordance_case_weight(weights, event_idx)
                        * (active_tied - event_tied_weight).max(0.0)
                        * multiplier;
                }
                risk_group_start = risk_group_end;
            }

            if event_group.len() > 1 {
                let event_weights: Vec<f64> = event_group
                    .iter()
                    .map(|&idx| concordance_case_weight(weights, idx))
                    .collect();
                let tied_xy = tied_event_risk_pairs(event_group, risk_scores, weights);
                counts.tied_xy += tied_xy * multiplier;
                counts.tied_y += (weighted_pairs(&event_weights) - tied_xy).max(0.0) * multiplier;
            }
            numerator += death_weight * multiplier * z2 / risk_weight;
            comparable_pair_weight += death_weight * (risk_weight - death_weight) * multiplier;
        }
        event_group_start = event_group_end;
    }
    (counts, numerator, comparable_pair_weight)
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
    order_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> IndexedConcordanceRankRows {
    let multipliers =
        right_concordance_time_weight_multipliers(time, status, case_weights, time_weight);
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        time[left]
            .total_cmp(&time[right])
            .then_with(|| order_scores[right].total_cmp(&order_scores[left]))
            .then_with(|| right.cmp(&left))
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
        while time_cursor < time_order.len()
            && concordance_time_precedes(time[time_order[time_cursor]], event_time)
        {
            let idx = time_order[time_cursor];
            let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
            at_risk.update(rank, -concordance_case_weight(case_weights, idx));
            time_cursor += 1;
        }

        let mut event_group_end = event_group_start + 1;
        while event_group_end < event_indices.len()
            && same_time(time[event_indices[event_group_end]], event_time)
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
                    event_idx,
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
    order_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> IndexedConcordanceRankRows {
    let multipliers = counting_concordance_time_weight_multipliers(
        start,
        stop,
        status,
        case_weights,
        time_weight,
    );
    let mut event_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect();
    event_indices.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| order_scores[right].total_cmp(&order_scores[left]))
            .then_with(|| right.cmp(&left))
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
                        event_idx,
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

fn reassemble_stratified_rank_rows(
    n: usize,
    groups: Vec<Vec<usize>>,
    display_times: Option<&[f64]>,
    output_len: Option<usize>,
    mut compute_group: impl FnMut(&[usize]) -> IndexedConcordanceRankRows,
) -> PyResult<ConcordanceRankRows> {
    let output_len = output_len.unwrap_or(n);
    if output_len > n {
        return Err(PyValueError::new_err(
            "output_len cannot exceed the input length",
        ));
    }
    let mut result = vec![[0.0; 4]; output_len];
    for indices in groups {
        let mut group_rows = compute_group(&indices);
        let filtered_indices;
        let output_indices = if output_len == n {
            indices.as_slice()
        } else {
            group_rows.retain(|row| indices[row.0] < output_len);
            filtered_indices = indices
                .iter()
                .copied()
                .filter(|&index| index < output_len)
                .collect::<Vec<_>>();
            filtered_indices.as_slice()
        };
        if output_indices.is_empty() {
            continue;
        }
        if group_rows.is_empty() {
            return Err(PyValueError::new_err(
                "number of items to replace is not a multiple of replacement length",
            ));
        }
        let group_indices = &indices;
        let source: Vec<f64> = (0..4)
            .flat_map(|column| {
                group_rows.iter().map(move |row| match column {
                    0 => display_times.map_or(row.1, |values| values[group_indices[row.0]]),
                    1 => row.2,
                    2 => row.3,
                    _ => row.4,
                })
            })
            .collect();
        let target_len = output_indices.len() * 4;
        if !target_len.is_multiple_of(source.len()) {
            return Err(PyValueError::new_err(
                "number of items to replace is not a multiple of replacement length",
            ));
        }
        for column in 0..4 {
            for (local_idx, &original_idx) in output_indices.iter().enumerate() {
                result[original_idx][column] =
                    source[(column * output_indices.len() + local_idx) % source.len()];
            }
        }
    }
    Ok(result
        .into_iter()
        .map(|row| (row[0], row[1], row[2], row[3]))
        .collect())
}

fn split_indexed_rank_rows(rows: IndexedConcordanceRankRows) -> ConcordanceRankRowsWithIndices {
    let mut values = Vec::with_capacity(rows.len());
    let mut indices = Vec::with_capacity(rows.len());
    for (index, time, rank, time_weight, case_weight) in rows {
        indices.push(index);
        values.push((time, rank, time_weight, case_weight));
    }
    (values, indices)
}

#[allow(clippy::too_many_arguments)]
fn stratified_right_concordance_rank_rows(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    order_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    display_times: Option<&[f64]>,
    output_len: Option<usize>,
) -> PyResult<ConcordanceRankRows> {
    reassemble_stratified_rank_rows(
        time.len(),
        strata_groups(strata),
        display_times,
        output_len,
        |indices| {
            let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
            let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
            let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
            let group_order: Vec<f64> = indices.iter().map(|&idx| order_scores[idx]).collect();
            let group_weights: Option<Vec<f64>> =
                weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
            right_concordance_rank_rows_for_vectors(
                &group_time,
                &group_status,
                &group_risk,
                &group_order,
                group_weights.as_deref(),
                time_weight,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn stratified_counting_concordance_rank_rows_for_strata(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    order_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
    display_stop: Option<&[f64]>,
    output_len: Option<usize>,
) -> PyResult<ConcordanceRankRows> {
    reassemble_stratified_rank_rows(
        stop.len(),
        strata_groups(strata),
        display_stop,
        output_len,
        |indices| {
            let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
            let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
            let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
            let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
            let group_order: Vec<f64> = indices.iter().map(|&idx| order_scores[idx]).collect();
            let group_weights: Option<Vec<f64>> =
                weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
            counting_concordance_rank_rows_for_vectors(
                &group_start,
                &group_stop,
                &group_status,
                &group_risk,
                &group_order,
                group_weights.as_deref(),
                time_weight,
            )
        },
    )
}

fn add_influence_pair(
    influence_rows: &mut [[f64; 5]],
    left: usize,
    right: usize,
    column: usize,
    value: f64,
) {
    let share = 0.5 * value;
    influence_rows[left][column] += share;
    influence_rows[right][column] += share;
}

fn influence_from_rows(
    influence_rows: Vec<[f64; 5]>,
    concordant: f64,
    comparable: f64,
) -> ConcordanceInfluenceOutput {
    let output_rows: Vec<Vec<f64>> = influence_rows.iter().map(|row| row.to_vec()).collect();
    if comparable <= 0.0 {
        return (output_rows, vec![0.0; influence_rows.len()], 0.0);
    }

    let somer = (2.0 * concordant - comparable) / comparable;
    let dfbeta: Vec<f64> = influence_rows
        .iter()
        .map(|row| {
            let comparable_row = row[0] + row[1] + row[2];
            ((row[0] - row[1]) - comparable_row * somer) / (2.0 * comparable)
        })
        .collect();
    let variance = dfbeta.iter().map(|value| value * value).sum();
    (output_rows, dfbeta, variance)
}

fn right_concordance_influence_rows_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceInfluenceOutput {
    let multipliers =
        right_concordance_time_weight_multipliers(time, status, case_weights, time_weight);
    let mut influence_rows = vec![[0.0; 5]; time.len()];
    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for left in 0..time.len() {
        for right in left + 1..time.len() {
            if status[left] == 1 && status[right] == 1 && same_time(time[left], time[right]) {
                let multiplier = multiplier_at(&multipliers, time[left]);
                let pair_weight = concordance_case_weight(case_weights, left)
                    * concordance_case_weight(case_weights, right)
                    * multiplier;
                if pair_weight <= 0.0 {
                    continue;
                }
                let column = if risk_scores[left] == risk_scores[right] {
                    4
                } else {
                    3
                };
                add_influence_pair(&mut influence_rows, left, right, column, pair_weight);
                continue;
            }

            let (event_idx, risk_idx) = if status[left] == 1
                && (concordance_time_precedes(time[left], time[right])
                    || (status[right] != 1 && same_time(time[left], time[right])))
            {
                (left, right)
            } else if status[right] == 1
                && (concordance_time_precedes(time[right], time[left])
                    || (status[left] != 1 && same_time(time[right], time[left])))
            {
                (right, left)
            } else {
                continue;
            };
            let multiplier = multiplier_at(&multipliers, time[event_idx]);
            let pair_weight = concordance_case_weight(case_weights, event_idx)
                * concordance_case_weight(case_weights, risk_idx)
                * multiplier;
            if pair_weight <= 0.0 {
                continue;
            }
            comparable += pair_weight;
            let diff = risk_scores[event_idx] - risk_scores[risk_idx];
            let column = if diff > 0.0 {
                concordant += pair_weight;
                0
            } else if diff < 0.0 {
                1
            } else {
                concordant += 0.5 * pair_weight;
                2
            };
            add_influence_pair(
                &mut influence_rows,
                event_idx,
                risk_idx,
                column,
                pair_weight,
            );
        }
    }

    influence_from_rows(influence_rows, concordant, comparable)
}

fn counting_concordance_influence_rows_for_vectors(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    case_weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceInfluenceOutput {
    let multipliers = counting_concordance_time_weight_multipliers(
        start,
        stop,
        status,
        case_weights,
        time_weight,
    );
    let mut influence_rows = vec![[0.0; 5]; stop.len()];
    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for (event_idx, &event) in status.iter().enumerate() {
        if event != 1 {
            continue;
        }
        let event_time = stop[event_idx];
        let multiplier = multiplier_at(&multipliers, event_time);
        if multiplier <= 0.0 {
            continue;
        }
        for risk_idx in 0..stop.len() {
            if risk_idx == event_idx {
                continue;
            }
            let pair_weight = concordance_case_weight(case_weights, event_idx)
                * concordance_case_weight(case_weights, risk_idx)
                * multiplier;
            if pair_weight <= 0.0 {
                continue;
            }
            if status[risk_idx] == 1 && stop[risk_idx] == event_time {
                if event_idx < risk_idx {
                    let column = if risk_scores[event_idx] == risk_scores[risk_idx] {
                        4
                    } else {
                        3
                    };
                    add_influence_pair(
                        &mut influence_rows,
                        event_idx,
                        risk_idx,
                        column,
                        pair_weight,
                    );
                }
                continue;
            }
            if !(start[risk_idx] < event_time && event_time <= stop[risk_idx]) {
                continue;
            }

            comparable += pair_weight;
            let diff = risk_scores[event_idx] - risk_scores[risk_idx];
            let column = if diff > 0.0 {
                concordant += pair_weight;
                0
            } else if diff < 0.0 {
                1
            } else {
                concordant += 0.5 * pair_weight;
                2
            };
            add_influence_pair(
                &mut influence_rows,
                event_idx,
                risk_idx,
                column,
                pair_weight,
            );
        }
    }

    influence_from_rows(influence_rows, concordant, comparable)
}

fn remap_stratified_influence(
    n: usize,
    groups: Vec<Vec<usize>>,
    mut compute_group: impl FnMut(&[usize]) -> ConcordanceInfluenceOutput,
) -> ConcordanceInfluenceOutput {
    let mut influence_rows = vec![[0.0; 5]; n];

    for indices in groups {
        let (group_influence, _group_dfbeta, _group_variance) = compute_group(&indices);
        for (local_idx, &original_idx) in indices.iter().enumerate() {
            influence_rows[original_idx].copy_from_slice(&group_influence[local_idx]);
        }
    }

    let concordant = influence_rows.iter().map(|row| row[0] + 0.5 * row[2]).sum();
    let comparable = influence_rows
        .iter()
        .map(|row| row[0] + row[1] + row[2])
        .sum();
    influence_from_rows(influence_rows, concordant, comparable)
}

fn stratified_right_concordance_influence_rows(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceInfluenceOutput {
    remap_stratified_influence(time.len(), strata_groups(strata), |indices| {
        let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        right_concordance_influence_rows_for_vectors(
            &group_time,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
        )
    })
}

fn stratified_counting_concordance_influence_rows_for_strata(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceInfluenceOutput {
    remap_stratified_influence(stop.len(), strata_groups(strata), |indices| {
        let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
        let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        counting_concordance_influence_rows_for_vectors(
            &group_start,
            &group_stop,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
        )
    })
}

fn right_concordance_summary_for_vectors(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let time_weight = effective_concordance_time_weight(status, time_weight);
    if time_weight == ConcordanceTimeWeight::N {
        match weights {
            Some(values) => concordance_summary_with_horizon_and_weights(
                risk_scores,
                time,
                status,
                Some(values),
                None,
            ),
            None => concordance_summary_with_horizon(risk_scores, time, status, None),
        }
    } else {
        concordance_summary_with_horizon_weights_and_time_weight(
            risk_scores,
            time,
            status,
            weights,
            None,
            time_weight,
        )
    }
}

fn counting_concordance_summary_for_vectors(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let time_weight = effective_concordance_time_weight(status, time_weight);
    if time_weight == ConcordanceTimeWeight::N {
        match weights {
            Some(values) => counting_process_concordance_summary_with_weights(
                risk_scores,
                start,
                stop,
                status,
                Some(values),
            ),
            None => counting_process_concordance_summary(risk_scores, start, stop, status),
        }
    } else {
        counting_process_concordance_summary_with_weights_and_time_weight(
            risk_scores,
            start,
            stop,
            status,
            weights,
            time_weight,
        )
    }
}

fn stratified_right_concordance_summary_counts(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> (
    ConcordanceSummary,
    ConcordanceTieCounts,
    f64,
    f64,
    f64,
    ConcordanceCountRows,
) {
    let mut total = ConcordanceSummary::default();
    let mut ties = ConcordanceTieCounts::default();
    let mut conditional_variance_numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    let mut strata_counts = Vec::new();
    for indices in strata_groups(strata) {
        let group_time: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        let summary = right_concordance_summary_for_vectors(
            &group_time,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
        );
        total.concordant += summary.concordant;
        total.comparable += summary.comparable;
        let (group_ties, group_conditional_variance_numerator, group_comparable_pair_weight) =
            right_concordance_tie_counts_for_vectors(
                &group_time,
                &group_status,
                &group_risk,
                group_weights.as_deref(),
                time_weight,
            );
        ties.tied_x += group_ties.tied_x;
        ties.tied_y += group_ties.tied_y;
        ties.tied_xy += group_ties.tied_xy;
        strata_counts.push(concordance_count_row(summary, group_ties));
        conditional_variance_numerator += group_conditional_variance_numerator;
        comparable_pair_weight += group_comparable_pair_weight;
    }
    let n_event = status.iter().filter(|&&event| event == 1).count() as f64;
    (
        total,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
        strata_counts,
    )
}

fn stratified_counting_concordance_summary_counts(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    strata: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> (
    ConcordanceSummary,
    ConcordanceTieCounts,
    f64,
    f64,
    f64,
    ConcordanceCountRows,
) {
    let mut total = ConcordanceSummary::default();
    let mut ties = ConcordanceTieCounts::default();
    let mut conditional_variance_numerator = 0.0;
    let mut comparable_pair_weight = 0.0;
    let mut strata_counts = Vec::new();
    for indices in strata_groups(strata) {
        let group_start: Vec<f64> = indices.iter().map(|&idx| start[idx]).collect();
        let group_stop: Vec<f64> = indices.iter().map(|&idx| stop[idx]).collect();
        let group_status: Vec<i32> = indices.iter().map(|&idx| status[idx]).collect();
        let group_risk: Vec<f64> = indices.iter().map(|&idx| risk_scores[idx]).collect();
        let group_weights: Option<Vec<f64>> =
            weights.map(|values| indices.iter().map(|&idx| values[idx]).collect());
        let summary = counting_concordance_summary_for_vectors(
            &group_start,
            &group_stop,
            &group_status,
            &group_risk,
            group_weights.as_deref(),
            time_weight,
        );
        total.concordant += summary.concordant;
        total.comparable += summary.comparable;
        let (group_ties, group_conditional_variance_numerator, group_comparable_pair_weight) =
            counting_concordance_conditional_variance_numerator(
                &group_start,
                &group_stop,
                &group_status,
                &group_risk,
                group_weights.as_deref(),
                time_weight,
            );
        ties.tied_x += group_ties.tied_x;
        ties.tied_y += group_ties.tied_y;
        ties.tied_xy += group_ties.tied_xy;
        strata_counts.push(concordance_count_row(summary, group_ties));
        conditional_variance_numerator += group_conditional_variance_numerator;
        comparable_pair_weight += group_comparable_pair_weight;
    }
    let n_event = status.iter().filter(|&&event| event == 1).count() as f64;
    (
        total,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
        strata_counts,
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
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn concordance_index(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<f64> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    Ok(
        if time_weight == ConcordanceTimeWeight::N && weights.is_none() {
            concordance_index_with_horizon(&risk_scores, &time, &status, None)
        } else {
            concordance_summary_with_horizon_weights_and_time_weight(
                &risk_scores,
                &time,
                &status,
                weights.as_deref(),
                None,
                time_weight,
            )
            .c_index()
        },
    )
}

fn conditional_variance_from_numerator(numerator: f64, comparable: f64) -> f64 {
    numerator / (4.0 * comparable * comparable)
}

fn build_right_concordance_summary_dict(
    summary: ConcordanceSummary,
    ties: ConcordanceTieCounts,
    n_event: Option<f64>,
    conditional_variance: f64,
    strata_counts: Option<ConcordanceCountRows>,
) -> PyResult<Py<PyDict>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        dict.set_item("concordance", summary.c_index())?;
        dict.set_item("concordant", summary.concordant)?;
        dict.set_item("comparable", summary.comparable)?;
        dict.set_item("tied_x", ties.tied_x)?;
        dict.set_item("tied_y", ties.tied_y)?;
        dict.set_item("tied_xy", ties.tied_xy)?;
        dict.set_item("conditional_variance", conditional_variance)?;
        if let Some(value) = n_event {
            dict.set_item("n_event", value)?;
        }
        if let Some(value) = strata_counts {
            dict.set_item("strata_counts", value)?;
        }
        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn concordance_summary(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<Py<PyDict>> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_concordance_time_weight(&timewt)?;

    let summary = right_concordance_summary_for_vectors(
        &time,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
    );
    let (ties, conditional_variance_numerator, comparable_pair_weight) =
        right_concordance_tie_counts_for_vectors(
            &time,
            &status,
            &risk_scores,
            weights.as_deref(),
            time_weight,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_right_concordance_summary_dict(summary, ties, None, conditional_variance, None)
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string()))]
pub fn stratified_concordance_summary(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<Py<PyDict>> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_strata_length(time.len(), &strata, "time")?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    let (
        summary,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
        strata_counts,
    ) = stratified_right_concordance_summary_counts(
        &time,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
    );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_right_concordance_summary_dict(
        summary,
        ties,
        Some(n_event),
        conditional_variance,
        Some(strata_counts),
    )
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), order_scores=None))]
pub fn concordance_rank_rows_with_indices(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    order_scores: Option<Vec<f64>>,
) -> PyResult<ConcordanceRankRowsWithIndices> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    if order_scores
        .as_ref()
        .is_some_and(|values| values.len() != time.len())
    {
        return Err(PyValueError::new_err(
            "order_scores must have the same length as time",
        ));
    }
    let time_weight = parse_concordance_time_weight(&timewt)?;
    let order_scores = order_scores.as_deref().unwrap_or(&risk_scores);
    Ok(split_indexed_rank_rows(
        right_concordance_rank_rows_for_vectors(
            &time,
            &status,
            &risk_scores,
            order_scores,
            weights.as_deref(),
            time_weight,
        ),
    ))
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string(), order_scores=None))]
pub fn concordance_rank_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    order_scores: Option<Vec<f64>>,
) -> PyResult<ConcordanceRankRows> {
    Ok(concordance_rank_rows_with_indices(
        time,
        status,
        risk_scores,
        weights,
        timewt,
        order_scores,
    )?
    .0)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string(), order_scores=None, display_times=None, output_len=None))]
pub fn stratified_concordance_rank_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    order_scores: Option<Vec<f64>>,
    display_times: Option<Vec<f64>>,
    output_len: Option<usize>,
) -> PyResult<ConcordanceRankRows> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_strata_length(time.len(), &strata, "time")?;
    if order_scores
        .as_ref()
        .is_some_and(|values| values.len() != time.len())
    {
        return Err(PyValueError::new_err(
            "order_scores must have the same length as time",
        ));
    }
    if display_times
        .as_ref()
        .is_some_and(|values| values.len() != time.len())
    {
        return Err(PyValueError::new_err(
            "display_times must have the same length as time",
        ));
    }
    let time_weight = parse_concordance_time_weight(&timewt)?;
    let order_scores = order_scores.as_deref().unwrap_or(&risk_scores);
    stratified_right_concordance_rank_rows(
        &time,
        &status,
        &risk_scores,
        order_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        display_times.as_deref(),
        output_len,
    )
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, weights=None, timewt="n".to_string()))]
pub fn concordance_influence_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(right_concordance_influence_rows_for_vectors(
        &time,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
    ))
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_scores, strata, weights=None, timewt="n".to_string()))]
pub fn stratified_concordance_influence_rows(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
) -> PyResult<ConcordanceInfluenceOutput> {
    validate_right_concordance_inputs(&time, &status, &risk_scores, weights.as_deref())?;
    validate_strata_length(time.len(), &strata, "time")?;
    let time_weight = parse_concordance_time_weight(&timewt)?;
    Ok(stratified_right_concordance_influence_rows(
        &time,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
    ))
}

#[pyfunction]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None))]
pub fn counting_concordance_index(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
) -> PyResult<f64> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    Ok(
        if time_weight == ConcordanceTimeWeight::N && weights.is_none() {
            counting_process_concordance_index(&risk_scores, &start, &stop, &status)
        } else {
            counting_process_concordance_summary_with_weights_and_time_weight(
                &risk_scores,
                &start,
                &stop,
                &status,
                weights.as_deref(),
                time_weight,
            )
            .c_index()
        },
    )
}

#[pyfunction]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None))]
pub fn counting_concordance_summary(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
) -> PyResult<Py<PyDict>> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;

    let summary = counting_concordance_summary_for_vectors(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
    );
    let (ties, conditional_variance_numerator, comparable_pair_weight) =
        counting_concordance_conditional_variance_numerator(
            &start,
            &stop,
            &status,
            &risk_scores,
            weights.as_deref(),
            time_weight,
        );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_right_concordance_summary_dict(summary, ties, None, conditional_variance, None)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None))]
pub fn stratified_counting_concordance_summary(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
) -> PyResult<Py<PyDict>> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    validate_strata_length(start.len(), &strata, "start")?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    let (
        summary,
        ties,
        n_event,
        conditional_variance_numerator,
        comparable_pair_weight,
        strata_counts,
    ) = stratified_counting_concordance_summary_counts(
        &start,
        &stop,
        &status,
        &risk_scores,
        &strata,
        weights.as_deref(),
        time_weight,
    );
    let conditional_variance =
        conditional_variance_from_numerator(conditional_variance_numerator, comparable_pair_weight);
    build_right_concordance_summary_dict(
        summary,
        ties,
        Some(n_event),
        conditional_variance,
        Some(strata_counts),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, order_scores=None))]
pub fn counting_concordance_rank_rows_with_indices(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    order_scores: Option<Vec<f64>>,
) -> PyResult<ConcordanceRankRowsWithIndices> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    if order_scores
        .as_ref()
        .is_some_and(|values| values.len() != start.len())
    {
        return Err(PyValueError::new_err(
            "order_scores must have the same length as start",
        ));
    }
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    let order_scores = order_scores.as_deref().unwrap_or(&risk_scores);
    Ok(split_indexed_rank_rows(
        counting_concordance_rank_rows_for_vectors(
            &start,
            &stop,
            &status,
            &risk_scores,
            order_scores,
            weights.as_deref(),
            time_weight,
        ),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None, order_scores=None))]
pub fn counting_concordance_rank_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    order_scores: Option<Vec<f64>>,
) -> PyResult<ConcordanceRankRows> {
    Ok(counting_concordance_rank_rows_with_indices(
        start,
        stop,
        status,
        risk_scores,
        weights,
        timewt,
        timefix,
        order_scores,
    )?
    .0)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None, order_scores=None, display_stop=None, output_len=None))]
pub fn stratified_counting_concordance_rank_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
    order_scores: Option<Vec<f64>>,
    display_stop: Option<Vec<f64>>,
    output_len: Option<usize>,
) -> PyResult<ConcordanceRankRows> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    validate_strata_length(start.len(), &strata, "start")?;
    if order_scores
        .as_ref()
        .is_some_and(|values| values.len() != start.len())
    {
        return Err(PyValueError::new_err(
            "order_scores must have the same length as start",
        ));
    }
    if display_stop
        .as_ref()
        .is_some_and(|values| values.len() != start.len())
    {
        return Err(PyValueError::new_err(
            "display_stop must have the same length as start",
        ));
    }
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    let order_scores = order_scores.as_deref().unwrap_or(&risk_scores);
    stratified_counting_concordance_rank_rows_for_strata(
        &start,
        &stop,
        &status,
        &risk_scores,
        order_scores,
        &strata,
        weights.as_deref(),
        time_weight,
        display_stop.as_deref(),
        output_len,
    )
}

#[pyfunction]
#[pyo3(signature = (start, stop, status, risk_scores, weights=None, timewt="n".to_string(), timefix=None))]
pub fn counting_concordance_influence_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
) -> PyResult<ConcordanceInfluenceOutput> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
    let time_weight = parse_counting_concordance_time_weight(&timewt)?;
    Ok(counting_concordance_influence_rows_for_vectors(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        time_weight,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (start, stop, status, risk_scores, strata, weights=None, timewt="n".to_string(), timefix=None))]
pub fn stratified_counting_concordance_influence_rows(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    risk_scores: Vec<f64>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timewt: String,
    timefix: Option<bool>,
) -> PyResult<ConcordanceInfluenceOutput> {
    let (start, stop) = prepare_counting_concordance_times(&start, &stop, timefix);
    validate_counting_concordance_inputs(
        &start,
        &stop,
        &status,
        &risk_scores,
        weights.as_deref(),
        timefix,
    )?;
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
            None,
        )
        .expect_err("zero-width counting interval should be rejected");
        assert!(
            interval_err
                .to_string()
                .contains("start must be less than stop")
        );

        let start_err = validate_counting_concordance_inputs(
            &[-0.1, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            None,
            None,
        )
        .expect_err("negative start time should be rejected");
        assert!(start_err.to_string().contains("start contains negative"));

        let weight_err = validate_counting_concordance_inputs(
            &[0.0, 0.0],
            &[1.0, 2.0],
            &[1, 0],
            &[0.4, 0.1],
            Some(&[1.0, f64::NAN]),
            None,
        )
        .expect_err("NaN weights should be rejected");
        assert!(weight_err.to_string().contains("weights contains NaN"));
    }

    #[test]
    fn validate_counting_concordance_honors_exact_timefix() {
        initialize_python();

        let near_width_err = validate_counting_concordance_inputs(
            &[1.0],
            &[1.0 + TIME_EPSILON / 2.0],
            &[1],
            &[0.4],
            None,
            None,
        )
        .expect_err("legacy near-width counting interval should be rejected");
        assert!(
            near_width_err
                .to_string()
                .contains("start must be less than stop")
        );

        validate_counting_concordance_inputs(
            &[1.0],
            &[1.0 + TIME_EPSILON / 2.0],
            &[1],
            &[0.4],
            None,
            Some(false),
        )
        .expect("exact timefix should accept strictly positive near-width interval");
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
            None,
        )
        .unwrap();
        let near = concordance_rank_rows(
            vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0, 3.0],
            status,
            risk,
            weights,
            "S".to_string(),
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
            None,
        )
        .unwrap();
        let unit_weighted = concordance_rank_rows(
            time,
            status,
            risk,
            Some(vec![1.0; 4]),
            "S".to_string(),
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
        )
        .unwrap();

        assert_eq!(unweighted, unit_weighted);
    }

    #[test]
    fn right_concordance_censoring_weights_use_post_death_risk_set() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0];
        let status = [1, 1, 0, 1, 0];
        let weights = [1.0, 2.0, 3.0, 1.0, 2.0];

        for time_weight in [
            ConcordanceTimeWeight::SOverG,
            ConcordanceTimeWeight::NOverG2,
        ] {
            let multipliers = right_concordance_time_weight_multipliers(
                &time,
                &status,
                Some(&weights),
                time_weight,
            );
            assert_eq!(multipliers, vec![(1.0, 1.0), (2.0, 1.0), (3.0, 4.0)]);
        }
    }

    #[test]
    fn single_event_concordance_uses_unweighted_time_counts() {
        let time = [4.0, 7.0, 2.0, 5.0];
        let status = [0, 0, 1, 0];
        let risk = [0.5, 1.0, -0.5, 0.0];
        let unweighted = right_concordance_summary_for_vectors(
            &time,
            &status,
            &risk,
            None,
            ConcordanceTimeWeight::N,
        );
        let importance = right_concordance_summary_for_vectors(
            &time,
            &status,
            &risk,
            None,
            ConcordanceTimeWeight::I,
        );

        assert_eq!(importance, unweighted);
        assert_eq!(
            right_concordance_time_weight_multipliers(
                &time,
                &status,
                None,
                ConcordanceTimeWeight::I,
            ),
            vec![(2.0, 1.0)]
        );
    }

    #[test]
    fn concordance_rank_rows_report_original_event_indices() {
        initialize_python();

        let time = vec![1.0, 3.0, 2.0, 4.0, 4.0, 2.0];
        let status = vec![1, 0, 1, 1, 0, 1];
        let scores = vec![0.2, 0.9, 0.4, 0.7, 0.7, 0.1];
        let risk: Vec<f64> = scores.iter().map(|value| -value).collect();
        let (_, right_indices) = concordance_rank_rows_with_indices(
            time.clone(),
            status.clone(),
            risk.clone(),
            None,
            "n".to_string(),
            Some(scores.clone()),
        )
        .unwrap();
        let (_, counting_indices) = counting_concordance_rank_rows_with_indices(
            vec![0.0; time.len()],
            time,
            status,
            risk,
            None,
            "n".to_string(),
            None,
            Some(scores),
        )
        .unwrap();

        assert_eq!(right_indices, vec![0, 2, 5, 3]);
        assert_eq!(counting_indices, right_indices);
    }

    #[test]
    fn concordance_rank_rows_sweep_removes_earlier_times() {
        initialize_python();

        let rows = concordance_rank_rows(
            vec![3.0, 1.0, 2.0, 1.0 + TIME_EPSILON / 2.0, 0.5],
            vec![0, 1, 1, 1, 0],
            vec![0.2, 0.9, 0.5, 0.1, 0.8],
            Some(vec![1.0, 2.0, 3.0, 1.0, 4.0]),
            "n".to_string(),
            None,
        )
        .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].0, 1.0);
        assert!((rows[0].1 - 5.0 / 7.0).abs() < 1e-12);
        assert_eq!(rows[0].2, 7.0);
        assert_eq!(rows[0].3, 2.0);
        assert_eq!(rows[1].0, 1.0 + TIME_EPSILON / 2.0);
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
    fn stratified_concordance_summary_counts_within_strata() {
        initialize_python();

        let (summary, ties, n_event, _, _, strata_counts) =
            stratified_right_concordance_summary_counts(
                &[1.0, 2.0, 1.0, 2.0],
                &[1, 0, 1, 0],
                &[0.9, 0.1, 0.2, 0.8],
                &[0, 0, 1, 1],
                None,
                ConcordanceTimeWeight::N,
            );

        assert_eq!(n_event, 2.0);
        assert_eq!(summary.concordant, 1.0);
        assert_eq!(summary.comparable, 2.0);
        assert_eq!(summary.c_index(), 0.5);
        assert_eq!(ties, ConcordanceTieCounts::default());
        assert_eq!(
            strata_counts,
            vec![[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn right_concordance_tie_counts_separate_risk_and_event_time_ties() {
        let (risk_tie, _, _) = right_concordance_tie_counts_for_vectors(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 0, 1],
            &[0.2, 0.4, 0.4, 1.0],
            None,
            ConcordanceTimeWeight::N,
        );
        assert_eq!(
            risk_tie,
            ConcordanceTieCounts {
                tied_x: 1.0,
                tied_y: 0.0,
                tied_xy: 0.0,
            }
        );

        let (event_time_tie, _, _) = right_concordance_tie_counts_for_vectors(
            &[1.0, 2.0, 2.0, 2.0, 4.0],
            &[1, 1, 1, 1, 1],
            &[0.2, 0.4, 0.4, 0.8, 1.0],
            None,
            ConcordanceTimeWeight::N,
        );
        assert_eq!(
            event_time_tie,
            ConcordanceTieCounts {
                tied_x: 0.0,
                tied_y: 2.0,
                tied_xy: 1.0,
            }
        );

        let (distinct_scores, _, _) = right_concordance_tie_counts_for_vectors(
            &[1.0, 2.0, 3.0],
            &[1, 0, 0],
            &[0.0, 0.5e-12, 1e-12],
            None,
            ConcordanceTimeWeight::N,
        );
        assert_eq!(distinct_scores.tied_x, 0.0);
    }

    #[test]
    fn right_concordance_tie_counts_apply_case_and_time_weights() {
        let (counts, _, _) = right_concordance_tie_counts_for_vectors(
            &[1.0, 2.0, 2.0, 3.0],
            &[1, 1, 1, 0],
            &[0.1, 0.5, 0.5, 0.5],
            Some(&[1.0, 2.0, 3.0, 4.0]),
            ConcordanceTimeWeight::I,
        );

        assert!((counts.tied_x - 20.0 / 9.0).abs() < 1e-12);
        assert!((counts.tied_xy - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(counts.tied_y, 0.0);
    }

    #[test]
    fn right_concordance_counts_same_time_censored_pairs() {
        let (counts, numerator, comparable_pair_weight) = right_concordance_tie_counts_for_vectors(
            &[1.0, 1.0],
            &[1, 0],
            &[0.5, 0.5],
            Some(&[2.0, 3.0]),
            ConcordanceTimeWeight::N,
        );

        assert_eq!(counts.tied_x, 6.0);
        assert_eq!(counts.tied_y, 0.0);
        assert_eq!(counts.tied_xy, 0.0);
        assert_eq!(numerator, 0.0);
        assert_eq!(comparable_pair_weight, 6.0);
    }

    #[test]
    fn right_concordance_same_time_censor_influence_matches_pair_counts() {
        let (influence, dfbeta, variance) = right_concordance_influence_rows_for_vectors(
            &[1.0, 1.0],
            &[1, 0],
            &[0.5, 0.5],
            Some(&[2.0, 3.0]),
            ConcordanceTimeWeight::N,
        );

        assert_eq!(influence[0], vec![0.0, 0.0, 3.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![0.0, 0.0, 3.0, 0.0, 0.0]);
        assert_eq!(dfbeta, vec![0.0, 0.0]);
        assert_eq!(variance, 0.0);
    }

    #[test]
    fn right_concordance_same_time_censor_remains_in_rank_risk_set() {
        let rows = right_concordance_rank_rows_for_vectors(
            &[1.0, 1.0],
            &[1, 0],
            &[0.2, 0.8],
            &[0.2, 0.8],
            Some(&[2.0, 3.0]),
            ConcordanceTimeWeight::N,
        );

        assert_eq!(rows, vec![(0, 1.0, -0.6, 5.0, 2.0)]);
    }

    #[test]
    fn right_concordance_conditional_variance_matches_reference() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let status = [1, 1, 0, 1];
        let risk = [0.2, 0.4, 0.4, 1.0];
        let (_, numerator, comparable_pair_weight) = right_concordance_tie_counts_for_vectors(
            &time,
            &status,
            &risk,
            None,
            ConcordanceTimeWeight::N,
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
        let (_ties, numerator, comparable_pair_weight) =
            counting_concordance_conditional_variance_numerator(
                &start,
                &stop,
                &status,
                &risk,
                None,
                ConcordanceTimeWeight::N,
            );

        let variance = conditional_variance_from_numerator(numerator, comparable_pair_weight);
        assert!((variance - 0.075).abs() < 1e-12, "variance={variance}");
    }

    #[test]
    fn counting_concordance_same_time_pairs_match_reference() {
        let start = [0.0; 5];
        let stop = [1.0, 1.0, 2.0, 2.0, 3.0];
        let status = [1, 1, 1, 0, 1];
        let risk = [0.9, 0.2, 0.7, 0.1, 0.8];
        let weights = [2.0, 1.0, 3.0, 0.5, 4.0];
        let summary = counting_concordance_summary_for_vectors(
            &start,
            &stop,
            &status,
            &risk,
            Some(&weights),
            ConcordanceTimeWeight::S,
        );
        let (ties, numerator, pair_weight) = counting_concordance_conditional_variance_numerator(
            &start,
            &stop,
            &status,
            &risk,
            Some(&weights),
            ConcordanceTimeWeight::S,
        );

        assert_eq!(summary.concordant, 17.0);
        assert_eq!(summary.comparable, 36.0);
        assert!((summary.c_index() - 17.0 / 36.0).abs() < 1e-12);
        assert_eq!(
            ties,
            ConcordanceTieCounts {
                tied_x: 0.0,
                tied_y: 2.0,
                tied_xy: 0.0,
            }
        );
        assert!(
            (conditional_variance_from_numerator(numerator, pair_weight) - 0.0279348544973545)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn stratified_counting_concordance_summary_counts_within_strata() {
        initialize_python();

        let (summary, ties, n_event, _, _, strata_counts) =
            stratified_counting_concordance_summary_counts(
                &[0.0, 0.0, 0.0, 0.0],
                &[1.0, 2.0, 1.0, 2.0],
                &[1, 0, 1, 0],
                &[0.9, 0.1, 0.2, 0.8],
                &[0, 0, 1, 1],
                None,
                ConcordanceTimeWeight::N,
            );

        assert_eq!(n_event, 2.0);
        assert_eq!(summary.concordant, 1.0);
        assert_eq!(summary.comparable, 2.0);
        assert_eq!(summary.c_index(), 0.5);
        assert_eq!(ties, ConcordanceTieCounts::default());
        assert_eq!(
            strata_counts,
            vec![[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn weighted_stratified_count_rows_match_reference() {
        let (_, _, _, _, _, right_counts) = stratified_right_concordance_summary_counts(
            &[1.0, 3.0, 2.0, 4.0, 4.0, 2.0],
            &[1, 1, 1, 1, 1, 1],
            &[-0.2, -0.9, -0.4, -0.7, -0.7, -0.1],
            &[0, 0, 1, 1, 1, 0],
            Some(&[1.0, 2.0, 1.5, 0.5, 3.0, 2.5]),
            ConcordanceTimeWeight::N,
        );
        let (_, _, _, _, _, counting_counts) = stratified_counting_concordance_summary_counts(
            &[0.0, 0.0, 1.0, 2.5, 0.0, 0.0, 0.5, 2.0],
            &[2.0, 4.0, 3.0, 5.0, 1.0, 4.0, 3.0, 5.0],
            &[1, 0, 1, 1, 1, 1, 0, 1],
            &[-0.8, -0.2, -0.5, -0.1, -0.3, -0.9, -0.4, -0.6],
            &[0, 0, 0, 0, 1, 1, 1, 1],
            Some(&[1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.5, 2.0]),
            ConcordanceTimeWeight::N,
        );

        assert_eq!(
            right_counts,
            vec![[7.0, 2.5, 0.0, 0.0, 0.0], [5.25, 0.0, 0.0, 0.0, 1.5]]
        );
        assert_eq!(
            counting_counts,
            vec![[0.0, 7.25, 0.0, 0.0, 0.0], [10.5, 2.0, 0.0, 0.0, 0.0]]
        );
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
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                (1.0, 2.0, 1.0, 2.0),
                (0.5, 1.0, 0.5, 1.0),
                (1.0, 2.0, 1.0, 2.0),
                (-0.5, 1.0, -0.5, 1.0),
            ]
        );
    }

    #[test]
    fn stratified_concordance_rank_rows_recycle_display_times() {
        initialize_python();

        let rows = stratified_concordance_rank_rows(
            vec![2e-9, 4e-9, 6e-9, 2e-9, 0.0, 6e-9, 4e-9],
            vec![1, 0, 1, 0, 0, 1, 0],
            vec![0.5, 0.5, -1.0, 0.5, 0.0, 0.5, 1.0],
            vec![0, 0, 0, 1, 1, 1, 0],
            Some(vec![1.5, 1.5, 0.5, 1.5, 1.0, 1.5, 0.5]),
            "n/G2".to_string(),
            None,
            Some(vec![2.0, 6.0, 7.0, 2.0, 1.0, 7.0, 6.0]),
            None,
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                (2.0, 4.0, 2.0, 4.0),
                (7.0, 12.499999999999998, 7.0, 12.499999999999998),
                (0.0, 1.5, 0.0, 1.5),
                (7.0, 1.5, 1.5, 0.0),
                (0.0, 7.0, 1.5, 1.5),
                (1.5, 0.0, 7.0, 1.5),
                (0.0, 0.5, 0.0, 0.5),
            ]
        );
    }

    #[test]
    fn stratified_concordance_rank_rows_reject_empty_stratum_residuals() {
        initialize_python();

        let error = stratified_concordance_rank_rows(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 0, 0],
            vec![0.2, 0.6, 0.4, 0.8],
            vec![0, 0, 1, 1],
            None,
            "n".to_string(),
            None,
            None,
            None,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .ends_with("number of items to replace is not a multiple of replacement length")
        );
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
            None,
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                (1.0, 2.0, 1.0, 2.0),
                (0.5, 1.0, 0.5, 1.0),
                (1.0, 2.0, 1.0, 2.0),
                (-0.5, 1.0, -0.5, 1.0),
            ]
        );
    }

    #[test]
    fn stratified_counting_rank_rows_exclude_time_weight_padding() {
        initialize_python();

        let rows = stratified_counting_concordance_rank_rows(
            vec![2.0, 1.0, 1.0, 2.0, 0.0, 4.0, 0.0],
            vec![6.0, 3.0, 4.0, 4.0, 4.0, 7.0, 8.0],
            vec![0, 1, 1, 0, 0, 0, 1],
            vec![1.0, 0.5, -1.0, 1.0, 0.0, 0.0, 0.0],
            vec![0, 1, 0, 1, 1, 0, 0],
            Some(vec![1.0, 0.5, 2.0, 0.5, 0.5, 1.0, 0.0]),
            "I".to_string(),
            Some(false),
            Some(vec![-1.0, -0.5, 1.0, -1.0, 0.0, 0.0, 0.0]),
            Some(vec![6.0, 3.0, 4.0, 4.0, 4.0, 7.0, 8.0]),
            Some(6),
        )
        .unwrap();

        assert_eq!(
            rows,
            vec![
                (4.0, 2.0, 1.0, -1.0 / 3.0),
                (3.0, 0.5, 1.5, 0.0),
                (-1.0 / 3.0, 4.0, 2.0, 1.0),
                (0.0, 3.0, 0.5, 1.5),
                (1.5, 0.0, 3.0, 0.5),
                (1.0, -1.0 / 3.0, 4.0, 2.0),
            ]
        );
    }

    #[test]
    fn stratified_counting_rank_rows_reject_non_recyclable_residuals() {
        initialize_python();

        let error = stratified_counting_concordance_rank_rows(
            vec![0.0, 0.0, 1.0, 2.5, 0.0, 0.0, 0.5, 2.0],
            vec![2.0, 4.0, 3.0, 5.0, 1.0, 4.0, 3.0, 5.0],
            vec![1, 0, 1, 1, 1, 1, 0, 1],
            vec![-0.8, -0.2, -0.5, -0.1, -0.3, -0.9, -0.4, -0.6],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            Some(vec![1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.5, 2.0]),
            "n".to_string(),
            None,
            None,
            None,
            None,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .ends_with("number of items to replace is not a multiple of replacement length")
        );
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
        )
        .unwrap();

        assert_eq!(influence[0], vec![0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(dfbeta, vec![0.125, 0.125, -0.125, -0.125]);
        assert_eq!(variance, 0.0625);
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
        )
        .unwrap();

        assert_eq!(influence[0], vec![0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(dfbeta, vec![0.125, 0.125, -0.125, -0.125]);
        assert_eq!(variance, 0.0625);
    }

    #[test]
    fn stratified_concordance_influence_uses_global_ratio() {
        initialize_python();

        let (influence, dfbeta, variance) = stratified_concordance_influence_rows(
            vec![1.0, 3.0, 2.0, 4.0, 4.0, 2.0],
            vec![1, 1, 1, 1, 1, 1],
            vec![-0.2, -0.9, -0.4, -0.7, -0.7, -0.1],
            vec![0, 0, 1, 1, 1, 0],
            Some(vec![1.0, 2.0, 1.5, 0.5, 3.0, 2.5]),
            "n".to_string(),
        )
        .unwrap();

        assert_eq!(influence[0], vec![1.0, 1.25, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.375, 0.0, 0.0, 0.0, 0.75]);
        let expected = [
            -0.058891123240448,
            0.0402183280666475,
            0.03016374604998565,
            0.004309106578569375,
            0.02585463947141625,
            -0.04165469692617065,
        ];
        for (actual, expected) in dfbeta.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((variance - 0.008417674442479675).abs() < 1e-12);
    }

    #[test]
    fn stratified_counting_concordance_influence_uses_global_ratio() {
        initialize_python();

        let (influence, dfbeta, variance) = stratified_counting_concordance_influence_rows(
            vec![0.0, 0.0, 1.0, 2.5, 0.0, 0.0, 0.5, 2.0],
            vec![2.0, 4.0, 3.0, 5.0, 1.0, 4.0, 3.0, 5.0],
            vec![1, 0, 1, 1, 1, 1, 0, 1],
            vec![-0.8, -0.2, -0.5, -0.1, -0.3, -0.9, -0.4, -0.6],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            Some(vec![1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.5, 2.0]),
            "n".to_string(),
            None,
        )
        .unwrap();

        assert_eq!(influence[0], vec![0.0, 1.75, 0.0, 0.0, 0.0]);
        assert_eq!(influence[4], vec![5.25, 0.0, 0.0, 0.0, 0.0]);
        let expected = [
            -0.04710783528280725,
            -0.0672969075468675,
            -0.070661752924211,
            -0.0100945361320301,
            0.124499278961705,
            0.0086524595417401,
            0.088928056401218,
            -0.026918763018747,
        ];
        for (actual, expected) in dfbeta.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((variance - 0.03605075943231025).abs() < 1e-12);
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
        )
        .unwrap();

        assert_eq!(influence[0], vec![5.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![1.0, 2.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![4.5, 1.5, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![2.5, 0.5, 0.0, 0.0, 0.0]);
        assert!((dfbeta[0] - 20.0 / 289.0).abs() < 1e-12);
        assert!((dfbeta[1] + 22.0 / 289.0).abs() < 1e-12);
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
        )
        .unwrap();
        let near = concordance_influence_rows(
            vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0, 3.0],
            status,
            risk,
            weights,
            "S".to_string(),
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
        )
        .unwrap();
        let unit_weighted =
            concordance_influence_rows(time, status, risk, Some(vec![1.0; 4]), "S".to_string())
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
        )
        .unwrap();

        assert_eq!(influence[0], vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[1], vec![0.5, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(influence[2], vec![1.0, 0.5, 0.0, 0.0, 0.0]);
        assert_eq!(influence[3], vec![0.5, 0.5, 0.0, 0.0, 0.0]);
        assert!((dfbeta[0] - 0.08).abs() < 1e-12);
        assert!((dfbeta[1] + 0.08).abs() < 1e-12);
        assert!((variance - dfbeta.iter().map(|value| value * value).sum::<f64>()).abs() < 1e-12);
    }

    #[test]
    fn counting_concordance_same_time_influence_matches_reference() {
        initialize_python();

        let (influence, dfbeta, variance) = counting_concordance_influence_rows(
            vec![0.0; 5],
            vec![1.0, 1.0, 2.0, 2.0, 3.0],
            vec![1, 1, 1, 0, 1],
            vec![0.9, 0.2, 0.7, 0.1, 0.8],
            Some(vec![2.0, 1.0, 3.0, 0.5, 4.0]),
            "S".to_string(),
            None,
        )
        .unwrap();

        assert_eq!(
            influence,
            vec![
                vec![7.5, 0.0, 0.0, 1.0, 0.0],
                vec![0.25, 3.5, 0.0, 1.0, 0.0],
                vec![3.75, 7.5, 0.0, 0.0, 0.0],
                vec![1.5, 0.0, 0.0, 0.0, 0.0],
                vec![4.0, 8.0, 0.0, 0.0, 0.0],
            ]
        );
        let expected = [
            0.1099537037037035,
            -0.04224537037037035,
            -0.0434027777777778,
            0.02199074074074075,
            -0.0462962962962963,
        ];
        for (actual, expected) in dfbeta.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((variance - 0.018385229123799725).abs() < 1e-12);
    }
}
