use crate::constants::{
    DEFAULT_CONCORDANCE, DIVISION_FLOOR, ITERATIVE_MAX_ITER, LCG64_INCREMENT, LCG64_MULTIPLIER,
    TIED_PAIR_WEIGHT, TIME_EPSILON,
};
use crate::internal::fenwick::FenwickTree;
use std::f64::consts::SQRT_2;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ConcordanceSummary {
    pub(crate) concordant: f64,
    pub(crate) comparable: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConcordanceTimeWeight {
    N,
    S,
    SOverG,
    NOverG2,
    I,
}

impl ConcordanceSummary {
    #[inline]
    pub(crate) fn c_index(self) -> f64 {
        if self.comparable > 0.0 {
            self.concordant / self.comparable
        } else {
            DEFAULT_CONCORDANCE
        }
    }
}

#[inline]
pub(crate) fn sample_normal(rng: &mut fastrand::Rng) -> f64 {
    let u1: f64 = rng.f64().max(1e-10);
    let u2: f64 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline]
pub(crate) fn probit(p: f64) -> f64 {
    normal_inverse_cdf(p)
}

#[inline]
pub(crate) fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[inline]
pub(crate) fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

#[inline]
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

#[inline]
pub(crate) fn concordance_index_with_horizon(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    horizon: Option<f64>,
) -> f64 {
    concordance_summary_with_horizon(risk_scores, time, event, horizon).c_index()
}

pub(crate) fn concordance_summary_with_horizon(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    horizon: Option<f64>,
) -> ConcordanceSummary {
    concordance_summary_with_horizon_and_weights(risk_scores, time, event, None, horizon)
}

pub(crate) fn concordance_summary_with_horizon_and_weights(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    horizon: Option<f64>,
) -> ConcordanceSummary {
    concordance_summary_with_horizon_weights_and_time_weight(
        risk_scores,
        time,
        event,
        weights,
        horizon,
        ConcordanceTimeWeight::N,
    )
}

pub(crate) fn concordance_summary_with_horizon_weights_and_time_weight(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    horizon: Option<f64>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let n = risk_scores.len();
    if n < 2 || time.len() != n || event.len() != n {
        return ConcordanceSummary::default();
    }
    if weights.is_some_and(|values| values.len() != n) {
        return ConcordanceSummary::default();
    }

    if time.iter().any(|value| !value.is_finite())
        || risk_scores.iter().any(|value| !value.is_finite())
        || horizon.is_some_and(|value| !value.is_finite())
    {
        return concordance_summary_quadratic(
            risk_scores,
            time,
            event,
            weights,
            horizon,
            time_weight,
        );
    }

    concordance_summary_ranked(risk_scores, time, event, weights, horizon, time_weight)
}

pub(crate) fn counting_process_concordance_index(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
) -> f64 {
    counting_process_concordance_summary(risk_scores, start, stop, event).c_index()
}

pub(crate) fn counting_process_concordance_summary(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
) -> ConcordanceSummary {
    counting_process_concordance_summary_with_weights(risk_scores, start, stop, event, None)
}

pub(crate) fn counting_process_concordance_summary_with_weights(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
) -> ConcordanceSummary {
    counting_process_concordance_summary_with_weights_and_time_weight(
        risk_scores,
        start,
        stop,
        event,
        weights,
        ConcordanceTimeWeight::N,
    )
}

pub(crate) fn counting_process_concordance_summary_with_weights_and_time_weight(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let n = risk_scores.len();
    if n < 2 || start.len() != n || stop.len() != n || event.len() != n {
        return ConcordanceSummary::default();
    }
    if weights.is_some_and(|values| values.len() != n) {
        return ConcordanceSummary::default();
    }

    if start.iter().any(|value| !value.is_finite())
        || stop.iter().any(|value| !value.is_finite())
        || risk_scores.iter().any(|value| !value.is_finite())
    {
        return counting_process_concordance_summary_quadratic(
            risk_scores,
            start,
            stop,
            event,
            weights,
            time_weight,
        );
    }

    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();

    let mut start_order: Vec<usize> = (0..n).collect();
    start_order.sort_by(|&a, &b| start[a].total_cmp(&start[b]));
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]));

    let mut event_times: Vec<f64> = (0..n)
        .filter(|&idx| event[idx] == 1)
        .map(|idx| stop[idx])
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();

    let event_time_multipliers = if time_weight == ConcordanceTimeWeight::N {
        Vec::new()
    } else {
        counting_process_time_weight_multipliers(start, stop, event, weights, time_weight)
    };

    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut active = vec![false; n];
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for event_time in event_times {
        while start_cursor < n && start[start_order[start_cursor]] < event_time {
            let idx = start_order[start_cursor];
            if !active[idx] {
                let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                at_risk.update(rank, observation_weight(weights, idx));
                active[idx] = true;
            }
            start_cursor += 1;
        }

        while stop_cursor < n && stop[stop_order[stop_cursor]] < event_time {
            let idx = stop_order[stop_cursor];
            if active[idx] {
                let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                at_risk.update(rank, -observation_weight(weights, idx));
                active[idx] = false;
            }
            stop_cursor += 1;
        }

        let event_time_multiplier = if time_weight == ConcordanceTimeWeight::N {
            1.0
        } else {
            event_time_multiplier_at(&event_time_multipliers, event_time)
        };

        while stop_cursor < n && stop[stop_order[stop_cursor]] <= event_time {
            let idx = stop_order[stop_cursor];
            if active[idx] {
                let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
                at_risk.update(rank, -observation_weight(weights, idx));
                active[idx] = false;
            }
            stop_cursor += 1;
        }

        let at_risk_total = at_risk.total();
        if at_risk_total <= 0.0 || event_time_multiplier <= 0.0 {
            continue;
        }
        for idx in 0..n {
            if event[idx] == 1 && stop[idx] == event_time {
                let event_weight = observation_weight(weights, idx) * event_time_multiplier;
                comparable += event_weight * at_risk_total;
                concordant += event_weight
                    * concordance_contribution_for_rank(&at_risk, &risk_levels, risk_scores[idx]);
            }
        }
    }

    ConcordanceSummary {
        concordant,
        comparable,
    }
}

#[cfg(test)]
fn counting_process_concordance_quadratic(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
) -> f64 {
    counting_process_concordance_summary_quadratic(
        risk_scores,
        start,
        stop,
        event,
        None,
        ConcordanceTimeWeight::N,
    )
    .c_index()
}

fn counting_process_concordance_summary_quadratic(
    risk_scores: &[f64],
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let n = risk_scores.len();
    let mut concordant = 0.0;
    let mut comparable = 0.0;
    let event_time_multipliers = if time_weight == ConcordanceTimeWeight::N {
        Vec::new()
    } else {
        counting_process_time_weight_multipliers(start, stop, event, weights, time_weight)
    };

    for event_idx in 0..n {
        if event[event_idx] != 1 {
            continue;
        }
        let event_time = stop[event_idx];
        let event_time_multiplier = if time_weight == ConcordanceTimeWeight::N {
            1.0
        } else {
            event_time_multiplier_at(&event_time_multipliers, event_time)
        };
        if event_time_multiplier <= 0.0 {
            continue;
        }
        for risk_idx in 0..n {
            if risk_idx == event_idx {
                continue;
            }
            if start[risk_idx] < event_time && stop[risk_idx] > event_time {
                let pair_weight = observation_weight(weights, event_idx)
                    * observation_weight(weights, risk_idx)
                    * event_time_multiplier;
                comparable += pair_weight;
                let diff = risk_scores[event_idx] - risk_scores[risk_idx];
                if diff > 0.0 {
                    concordant += pair_weight;
                } else if diff.abs() < DIVISION_FLOOR {
                    concordant += TIED_PAIR_WEIGHT * pair_weight;
                }
            }
        }
    }

    ConcordanceSummary {
        concordant,
        comparable,
    }
}

#[cfg(test)]
fn concordance_index_quadratic(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    horizon: Option<f64>,
) -> f64 {
    concordance_summary_quadratic(
        risk_scores,
        time,
        event,
        None,
        horizon,
        ConcordanceTimeWeight::N,
    )
    .c_index()
}

fn concordance_summary_quadratic(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    horizon: Option<f64>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let n = risk_scores.len();
    let mut concordant = 0.0;
    let mut comparable = 0.0;
    let event_time_multipliers = if time_weight == ConcordanceTimeWeight::N {
        Vec::new()
    } else {
        right_censored_time_weight_multipliers(time, event, weights, time_weight)
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let i_comparable = event[i] == 1
                && time[i] < time[j]
                && match horizon {
                    Some(h) => time[i] <= h,
                    None => true,
                };
            let j_comparable = event[j] == 1
                && time[j] < time[i]
                && match horizon {
                    Some(h) => time[j] <= h,
                    None => true,
                };

            if i_comparable {
                let event_time_multiplier = if time_weight == ConcordanceTimeWeight::N {
                    1.0
                } else {
                    event_time_multiplier_at(&event_time_multipliers, time[i])
                };
                let pair_weight = observation_weight(weights, i)
                    * observation_weight(weights, j)
                    * event_time_multiplier;
                comparable += pair_weight;
                if risk_scores[i] > risk_scores[j] {
                    concordant += pair_weight;
                } else if (risk_scores[i] - risk_scores[j]).abs() < DIVISION_FLOOR {
                    concordant += TIED_PAIR_WEIGHT * pair_weight;
                }
            } else if j_comparable {
                let event_time_multiplier = if time_weight == ConcordanceTimeWeight::N {
                    1.0
                } else {
                    event_time_multiplier_at(&event_time_multipliers, time[j])
                };
                let pair_weight = observation_weight(weights, j)
                    * observation_weight(weights, i)
                    * event_time_multiplier;
                comparable += pair_weight;
                if risk_scores[j] > risk_scores[i] {
                    concordant += pair_weight;
                } else if (risk_scores[i] - risk_scores[j]).abs() < DIVISION_FLOOR {
                    concordant += TIED_PAIR_WEIGHT * pair_weight;
                }
            }
        }
    }

    ConcordanceSummary {
        concordant,
        comparable,
    }
}

#[cfg(test)]
fn concordance_index_ranked(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    horizon: Option<f64>,
) -> f64 {
    concordance_summary_ranked(
        risk_scores,
        time,
        event,
        None,
        horizon,
        ConcordanceTimeWeight::N,
    )
    .c_index()
}

fn concordance_summary_ranked(
    risk_scores: &[f64],
    time: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    horizon: Option<f64>,
    time_weight: ConcordanceTimeWeight,
) -> ConcordanceSummary {
    let n = risk_scores.len();
    let mut time_order: Vec<usize> = (0..n).collect();
    time_order.sort_by(|&a, &b| time[b].total_cmp(&time[a]));

    let mut risk_levels = risk_scores.to_vec();
    risk_levels.sort_by(f64::total_cmp);
    risk_levels.dedup();

    let mut at_risk = FenwickTree::new(risk_levels.len());
    let mut concordant = 0.0;
    let mut comparable = 0.0;
    let mut group_start = 0;
    let event_time_multipliers = if time_weight == ConcordanceTimeWeight::N {
        Vec::new()
    } else {
        right_censored_time_weight_multipliers(time, event, weights, time_weight)
    };

    while group_start < n {
        let current_time = time[time_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < n && time[time_order[group_end]] == current_time {
            group_end += 1;
        }

        let at_risk_total = at_risk.total();
        let event_time_multiplier = if time_weight == ConcordanceTimeWeight::N {
            1.0
        } else {
            event_time_multiplier_at(&event_time_multipliers, current_time)
        };
        if at_risk_total > 0.0 && event_time_multiplier > 0.0 {
            for &idx in &time_order[group_start..group_end] {
                if event[idx] != 1 || horizon.is_some_and(|h| time[idx] > h) {
                    continue;
                }

                let event_weight = observation_weight(weights, idx) * event_time_multiplier;
                comparable += event_weight * at_risk_total;
                concordant += event_weight
                    * concordance_contribution_for_rank(&at_risk, &risk_levels, risk_scores[idx]);
            }
        }

        for &idx in &time_order[group_start..group_end] {
            let rank = risk_levels.partition_point(|&risk| risk < risk_scores[idx]);
            at_risk.update(rank, observation_weight(weights, idx));
        }

        group_start = group_end;
    }

    ConcordanceSummary {
        concordant,
        comparable,
    }
}

#[inline]
fn observation_weight(weights: Option<&[f64]>, idx: usize) -> f64 {
    weights.map_or(1.0, |values| values[idx])
}

fn event_time_multiplier_at(event_time_multipliers: &[(f64, f64)], event_time: f64) -> f64 {
    event_time_multipliers
        .binary_search_by(|&(time, _)| time.total_cmp(&event_time))
        .map_or(0.0, |idx| event_time_multipliers[idx].1)
}

fn time_weight_multiplier_from_components(
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
        ConcordanceTimeWeight::N => 1.0,
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
    }
}

fn right_censored_time_weight_multipliers(
    time: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> Vec<(f64, f64)> {
    if time_weight == ConcordanceTimeWeight::N {
        return Vec::new();
    }

    let n = time.len();
    let total_weight: f64 = (0..n).map(|idx| observation_weight(weights, idx)).sum();
    let mut time_order: Vec<usize> = (0..n).collect();
    time_order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut nrisk = total_weight;
    let mut survival = 1.0;
    let mut censoring_survival = 1.0;
    let mut multipliers = Vec::new();
    let mut group_start = 0usize;

    while group_start < n {
        let current_time = time[time_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < n && time[time_order[group_end]] == current_time {
            group_end += 1;
        }

        let mut death_weight = 0.0;
        let mut censor_weight = 0.0;
        let mut group_weight = 0.0;
        for &idx in &time_order[group_start..group_end] {
            let weight = observation_weight(weights, idx);
            group_weight += weight;
            if event[idx] == 1 {
                death_weight += weight;
            } else {
                censor_weight += weight;
            }
        }

        if death_weight > 0.0 {
            multipliers.push((
                current_time,
                time_weight_multiplier_from_components(
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
        if censor_weight > 0.0 && nrisk > 0.0 {
            censoring_survival *= ((nrisk - censor_weight) / nrisk).max(0.0);
        }
        nrisk -= group_weight;
        group_start = group_end;
    }

    multipliers
}

fn counting_process_time_weight_multipliers(
    start: &[f64],
    stop: &[f64],
    event: &[i32],
    weights: Option<&[f64]>,
    time_weight: ConcordanceTimeWeight,
) -> Vec<(f64, f64)> {
    if time_weight == ConcordanceTimeWeight::N {
        return Vec::new();
    }

    let n = start.len();
    let total_weight: f64 = (0..n).map(|idx| observation_weight(weights, idx)).sum();
    let mut event_times: Vec<f64> = (0..n)
        .filter(|&idx| event[idx] == 1)
        .map(|idx| stop[idx])
        .collect();
    event_times.sort_by(f64::total_cmp);
    event_times.dedup();

    let mut start_order: Vec<usize> = (0..n).collect();
    start_order.sort_by(|&a, &b| start[a].total_cmp(&start[b]));
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]));

    let mut active_weight = 0.0;
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut survival = 1.0;
    let mut multipliers = Vec::with_capacity(event_times.len());

    for event_time in event_times {
        while start_cursor < n && start[start_order[start_cursor]] < event_time {
            active_weight += observation_weight(weights, start_order[start_cursor]);
            start_cursor += 1;
        }
        while stop_cursor < n && stop[stop_order[stop_cursor]] < event_time {
            active_weight -= observation_weight(weights, stop_order[stop_cursor]);
            stop_cursor += 1;
        }

        let death_weight = (0..n)
            .filter(|&idx| event[idx] == 1 && stop[idx] == event_time)
            .map(|idx| observation_weight(weights, idx))
            .sum::<f64>();

        multipliers.push((
            event_time,
            time_weight_multiplier_from_components(
                time_weight,
                total_weight,
                survival,
                1.0,
                active_weight,
            ),
        ));
        if active_weight > 0.0 {
            survival *= ((active_weight - death_weight) / active_weight).max(0.0);
        }
    }

    multipliers
}

#[inline]
fn concordance_contribution_for_rank(
    at_risk: &FenwickTree,
    risk_levels: &[f64],
    risk_score: f64,
) -> f64 {
    let less_end = risk_levels.partition_point(|&risk| risk < risk_score);
    let near_tie_end = risk_levels.partition_point(|&risk| risk - risk_score < DIVISION_FLOOR);

    let lower_risk_count = prefix_count_before(at_risk, less_end);
    let lower_or_near_tied_count = prefix_count_before(at_risk, near_tie_end);
    let near_tied_count = lower_or_near_tied_count - lower_risk_count;

    lower_risk_count + TIED_PAIR_WEIGHT * near_tied_count
}

#[inline]
fn prefix_count_before(at_risk: &FenwickTree, end: usize) -> f64 {
    if end == 0 {
        0.0
    } else {
        at_risk.prefix_sum(end - 1)
    }
}

#[inline]
pub(crate) fn lcg64_next(state: &mut u64) {
    *state = state
        .wrapping_mul(LCG64_MULTIPLIER)
        .wrapping_add(LCG64_INCREMENT);
}

#[inline]
#[cfg(feature = "ml")]
pub(crate) fn lcg64_shuffle_with_state(indices: &mut [usize], state: &mut u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        lcg64_next(state);
        let j = (*state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[inline]
pub(crate) fn lcg64_shuffle_per_index_seed(indices: &mut [usize], seed: u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        let mut state = seed.wrapping_add(i as u64);
        lcg64_next(&mut state);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

#[inline]
pub(crate) fn compute_censoring_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;
        let mut total_at_time = 0;

        while i < n && (time[indices[i]] - current_time).abs() < TIME_EPSILON {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);
        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

#[inline]
pub(crate) fn km_step_prob_at(t: f64, unique_times: &[f64], km_values: &[f64]) -> f64 {
    if unique_times.is_empty() {
        return 1.0;
    }
    if t < unique_times[0] {
        return 1.0;
    }

    let mut left = 0;
    let mut right = unique_times.len();
    while left < right {
        let mid = (left + right) / 2;
        if unique_times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 { 1.0 } else { km_values[left - 1] }
}

#[inline]
#[allow(clippy::excessive_precision)]
pub(crate) fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[inline]
pub(crate) fn gamma_cdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(a, x)
}

#[inline]
pub(crate) fn gamma_inverse_cdf(p: f64, a: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let mut x = if a > 1.0 {
        let d = 1.0 / (9.0 * a);
        let z = normal_inverse_cdf(p);
        a * (1.0 - d + z * d.sqrt()).powi(3).max(0.001)
    } else {
        (p * ln_gamma(a).exp() * a).powf(1.0 / a).max(0.001)
    };

    let eps = 1e-10;
    let max_iter = 50;
    for _ in 0..max_iter {
        let cdf = gamma_cdf(x, a);
        let pdf = gamma_pdf(x, a);
        if pdf < 1e-300 {
            break;
        }
        let delta = (cdf - p) / pdf;
        x -= delta;
        x = x.max(1e-10);
        if delta.abs() < eps * x {
            break;
        }
    }
    x
}

#[inline]
fn gamma_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

#[inline]
pub(crate) fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    let k = df as f64 / 2.0;
    let x_half = x / 2.0;
    1.0 - lower_incomplete_gamma(k, x_half)
}

#[inline]
pub(crate) fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0)
}

#[inline]
pub(crate) fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[inline]
pub(crate) fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_continued_fraction(a, x)
    }
}

#[inline]
pub(crate) fn gamma_series(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = ITERATIVE_MAX_ITER;
    let mut sum = 1.0 / a;
    let mut term = sum;
    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

#[inline]
pub(crate) fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = ITERATIVE_MAX_ITER;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_ranked_matches_quadratic(
        risk_scores: &[f64],
        time: &[f64],
        event: &[i32],
        horizon: Option<f64>,
    ) {
        let ranked = concordance_index_ranked(risk_scores, time, event, horizon);
        let quadratic = concordance_index_quadratic(risk_scores, time, event, horizon);

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    fn assert_counting_ranked_matches_quadratic(
        risk_scores: &[f64],
        start: &[f64],
        stop: &[f64],
        event: &[i32],
    ) {
        let ranked = counting_process_concordance_index(risk_scores, start, stop, event);
        let quadratic = counting_process_concordance_quadratic(risk_scores, start, stop, event);

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    fn assert_weighted_ranked_matches_quadratic(
        risk_scores: &[f64],
        time: &[f64],
        event: &[i32],
        weights: &[f64],
        horizon: Option<f64>,
    ) {
        let ranked = concordance_summary_ranked(
            risk_scores,
            time,
            event,
            Some(weights),
            horizon,
            ConcordanceTimeWeight::N,
        )
        .c_index();
        let quadratic = concordance_summary_quadratic(
            risk_scores,
            time,
            event,
            Some(weights),
            horizon,
            ConcordanceTimeWeight::N,
        )
        .c_index();

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "weighted ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    fn assert_time_weighted_ranked_matches_quadratic(
        risk_scores: &[f64],
        time: &[f64],
        event: &[i32],
        weights: &[f64],
        time_weight: ConcordanceTimeWeight,
    ) {
        let ranked =
            concordance_summary_ranked(risk_scores, time, event, Some(weights), None, time_weight)
                .c_index();
        let quadratic = concordance_summary_quadratic(
            risk_scores,
            time,
            event,
            Some(weights),
            None,
            time_weight,
        )
        .c_index();

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "time-weighted ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    fn assert_weighted_counting_ranked_matches_quadratic(
        risk_scores: &[f64],
        start: &[f64],
        stop: &[f64],
        event: &[i32],
        weights: &[f64],
    ) {
        let ranked = counting_process_concordance_summary_with_weights(
            risk_scores,
            start,
            stop,
            event,
            Some(weights),
        )
        .c_index();
        let quadratic = counting_process_concordance_summary_quadratic(
            risk_scores,
            start,
            stop,
            event,
            Some(weights),
            ConcordanceTimeWeight::N,
        )
        .c_index();

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "weighted counting ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    fn assert_time_weighted_counting_ranked_matches_quadratic(
        risk_scores: &[f64],
        start: &[f64],
        stop: &[f64],
        event: &[i32],
        weights: &[f64],
        time_weight: ConcordanceTimeWeight,
    ) {
        let ranked = counting_process_concordance_summary_with_weights_and_time_weight(
            risk_scores,
            start,
            stop,
            event,
            Some(weights),
            time_weight,
        )
        .c_index();
        let quadratic = counting_process_concordance_summary_quadratic(
            risk_scores,
            start,
            stop,
            event,
            Some(weights),
            time_weight,
        )
        .c_index();

        assert!(
            (ranked - quadratic).abs() < 1e-12,
            "time-weighted counting ranked {ranked} differed from quadratic {quadratic}"
        );
    }

    #[test]
    fn test_ranked_concordance_matches_quadratic_for_common_cases() {
        let time = [5.0, 2.0, 7.0, 3.0, 3.0, 9.0, 1.0];
        let event = [1, 1, 0, 1, 0, 1, 1];
        let risk = [0.2, 0.9, 0.1, 0.7, 0.7, 0.4, 1.1];

        assert_ranked_matches_quadratic(&risk, &time, &event, None);
        assert_ranked_matches_quadratic(&risk, &time, &event, Some(4.0));
    }

    #[test]
    fn test_ranked_concordance_matches_quadratic_for_generated_inputs() {
        for n in 2..40 {
            let time: Vec<f64> = (0..n).map(|i| ((i * 7 + n * 3) % 11) as f64).collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 4 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 6 == 0 {
                        0.25
                    } else {
                        ((i * 11 + n * 5) % 17) as f64 / 10.0
                    }
                })
                .collect();

            assert_ranked_matches_quadratic(&risk, &time, &event, None);
            assert_ranked_matches_quadratic(&risk, &time, &event, Some(5.0));
        }
    }

    #[test]
    fn test_ranked_concordance_preserves_tie_tolerance() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1, 1, 1, 1];
        let risk = [0.4, 0.4 + DIVISION_FLOOR / 2.0, 0.1, 0.8];

        assert_ranked_matches_quadratic(&risk, &time, &event, None);
    }

    #[test]
    fn test_ranked_concordance_handles_signed_zero_scores() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1, 1, 1, 1];
        let risk = [-0.0, 0.0, 0.5, -0.25];

        assert_ranked_matches_quadratic(&risk, &time, &event, None);
    }

    #[test]
    fn test_weighted_ranked_concordance_matches_quadratic() {
        for n in 2..40 {
            let time: Vec<f64> = (0..n).map(|i| ((i * 7 + n * 3) % 11) as f64).collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 4 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 6 == 0 {
                        0.25
                    } else {
                        ((i * 11 + n * 5) % 17) as f64 / 10.0
                    }
                })
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 9 == 0 {
                        0.0
                    } else {
                        0.5 + (i % 5) as f64
                    }
                })
                .collect();

            assert_weighted_ranked_matches_quadratic(&risk, &time, &event, &weights, None);
            assert_weighted_ranked_matches_quadratic(&risk, &time, &event, &weights, Some(5.0));
        }
    }

    #[test]
    fn test_time_weighted_ranked_concordance_matches_quadratic() {
        for n in 2..40 {
            let time: Vec<f64> = (0..n).map(|i| ((i * 5 + n * 2) % 13) as f64).collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 5 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        0.15
                    } else {
                        ((i * 17 + n * 3) % 23) as f64 / 10.0
                    }
                })
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 10 == 0 {
                        0.0
                    } else {
                        0.25 + (i % 7) as f64
                    }
                })
                .collect();

            for time_weight in [
                ConcordanceTimeWeight::S,
                ConcordanceTimeWeight::SOverG,
                ConcordanceTimeWeight::NOverG2,
                ConcordanceTimeWeight::I,
            ] {
                assert_time_weighted_ranked_matches_quadratic(
                    &risk,
                    &time,
                    &event,
                    &weights,
                    time_weight,
                );
            }
        }
    }

    #[test]
    fn test_concordance_falls_back_for_non_finite_values() {
        let time = [1.0, f64::NAN, 3.0];
        let event = [1, 1, 1];
        let risk = [0.4, 0.3, 0.2];

        let c_index = concordance_index_with_horizon(&risk, &time, &event, None);
        let quadratic = concordance_index_quadratic(&risk, &time, &event, None);

        assert!((c_index - quadratic).abs() < 1e-12);
    }

    #[test]
    fn test_counting_process_concordance_matches_quadratic_for_common_case() {
        let start = [0.0, 0.0, 1.5, 2.5, 0.0, 3.0];
        let stop = [2.0, 2.0, 4.0, 5.0, 5.0, 6.0];
        let event = [1, 1, 1, 0, 1, 0];
        let risk = [0.9, 0.2, 0.7, 0.1, 0.5, 0.4];

        assert_counting_ranked_matches_quadratic(&risk, &start, &stop, &event);
    }

    #[test]
    fn test_counting_process_concordance_matches_quadratic_for_generated_inputs() {
        for n in 2..40 {
            let start: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 0.5).collect();
            let stop: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(i, &value)| value + 0.5 + ((i * 7 + n) % 6) as f64 * 0.25)
                .collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 5 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        0.35
                    } else {
                        ((i * 13 + n * 3) % 19) as f64 / 10.0
                    }
                })
                .collect();

            assert_counting_ranked_matches_quadratic(&risk, &start, &stop, &event);
        }
    }

    #[test]
    fn test_weighted_counting_concordance_matches_quadratic() {
        for n in 2..40 {
            let start: Vec<f64> = (0..n).map(|i| (i % 5) as f64 * 0.5).collect();
            let stop: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(i, &value)| value + 0.5 + ((i * 7 + n) % 6) as f64 * 0.25)
                .collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 5 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        0.35
                    } else {
                        ((i * 13 + n * 3) % 19) as f64 / 10.0
                    }
                })
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 8 == 0 {
                        0.0
                    } else {
                        0.25 + (i % 6) as f64
                    }
                })
                .collect();

            assert_weighted_counting_ranked_matches_quadratic(
                &risk, &start, &stop, &event, &weights,
            );
        }
    }

    #[test]
    fn test_time_weighted_counting_concordance_matches_quadratic() {
        for n in 2..40 {
            let start: Vec<f64> = (0..n).map(|i| (i % 6) as f64 * 0.25).collect();
            let stop: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(i, &value)| value + 0.5 + ((i * 5 + n) % 7) as f64 * 0.2)
                .collect();
            let event: Vec<i32> = (0..n)
                .map(|i| if (i + n) % 6 == 0 { 0 } else { 1 })
                .collect();
            let risk: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 8 == 0 {
                        0.45
                    } else {
                        ((i * 19 + n * 2) % 29) as f64 / 10.0
                    }
                })
                .collect();
            let weights: Vec<f64> = (0..n)
                .map(|i| {
                    if i % 9 == 0 {
                        0.0
                    } else {
                        0.5 + (i % 5) as f64
                    }
                })
                .collect();

            for time_weight in [ConcordanceTimeWeight::S, ConcordanceTimeWeight::I] {
                assert_time_weighted_counting_ranked_matches_quadratic(
                    &risk,
                    &start,
                    &stop,
                    &event,
                    &weights,
                    time_weight,
                );
            }
        }
    }

    #[test]
    fn test_counting_process_concordance_falls_back_for_non_finite_values() {
        let start = [0.0, 0.0, 1.0];
        let stop = [2.0, f64::NAN, 3.0];
        let event = [1, 1, 1];
        let risk = [0.4, 0.3, 0.2];

        let c_index = counting_process_concordance_index(&risk, &start, &stop, &event);
        let quadratic = counting_process_concordance_quadratic(&risk, &start, &stop, &event);

        assert!((c_index - quadratic).abs() < 1e-12);
    }

    #[test]
    fn test_chi2_sf_basic() {
        assert!((chi2_sf(0.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(-1.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(1.0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ln_gamma() {
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_inverse_cdf() {
        let result = gamma_inverse_cdf(0.475, 5.0);
        assert!(
            result > 4.0 && result < 5.0,
            "Expected ~4.5, got {}",
            result
        );

        let result2 = gamma_inverse_cdf(0.525, 6.0);
        assert!(
            result2 > 5.0 && result2 < 7.0,
            "Expected ~6, got {}",
            result2
        );
    }
}
