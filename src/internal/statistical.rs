use crate::constants::{
    DEFAULT_CONCORDANCE, DIVISION_FLOOR, LCG64_INCREMENT, LCG64_MULTIPLIER, TIED_PAIR_WEIGHT,
    TIME_EPSILON, same_time,
};
use crate::internal::dist::{lgammafn, pchisq, pgamma, pnorm, pt, qgamma, qnorm, qt};
use crate::internal::fenwick::FenwickTree;

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
fn concordance_time_precedes(left: f64, right: f64) -> bool {
    left < right && !same_time(left, right)
}

#[inline]
fn concordance_at_or_before_horizon(time: f64, horizon: f64) -> bool {
    time <= horizon || same_time(time, horizon)
}

#[inline]
pub(crate) fn sample_normal(rng: &mut crate::internal::rng::Rng) -> f64 {
    let u1: f64 = rng.f64().max(1e-10);
    let u2: f64 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Standard normal quantile (R's `qnorm(p)`); `p` outside `(0, 1)` maps to
/// the corresponding infinity so callers may pass slightly out-of-range
/// probabilities without triggering NaN.
#[inline]
pub(crate) fn probit(p: f64) -> f64 {
    normal_inverse_cdf(p)
}

/// Error function, `erf(x) = 2 pnorm(x sqrt 2) - 1`.
#[inline]
pub(crate) fn erf(x: f64) -> f64 {
    crate::internal::dist::erf(x)
}

/// Complementary error function, `erfc(x) = 2 pnorm(x sqrt 2, lower = FALSE)`.
#[inline]
pub(crate) fn erfc(x: f64) -> f64 {
    crate::internal::dist::erfc(x)
}

/// Standard normal distribution function (R's `pnorm(x)`).
#[inline]
pub(crate) fn normal_cdf(x: f64) -> f64 {
    pnorm(x, true, false)
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

    let mut event_order: Vec<usize> = (0..n).filter(|&idx| event[idx] == 1).collect();
    event_order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]).then_with(|| a.cmp(&b)));

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
    let mut event_group_start = 0usize;

    while event_group_start < event_order.len() {
        let event_time = stop[event_order[event_group_start]];
        let mut event_group_end = event_group_start + 1;
        while event_group_end < event_order.len()
            && stop[event_order[event_group_end]] == event_time
        {
            event_group_end += 1;
        }

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
        if at_risk_total > 0.0 && event_time_multiplier > 0.0 {
            for &idx in &event_order[event_group_start..event_group_end] {
                let event_weight = observation_weight(weights, idx) * event_time_multiplier;
                comparable += event_weight * at_risk_total;
                concordant += event_weight
                    * concordance_contribution_for_rank(&at_risk, &risk_levels, risk_scores[idx]);
            }
        }

        event_group_start = event_group_end;
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
                && concordance_time_precedes(time[i], time[j])
                && match horizon {
                    Some(h) => concordance_at_or_before_horizon(time[i], h),
                    None => true,
                };
            let j_comparable = event[j] == 1
                && concordance_time_precedes(time[j], time[i])
                && match horizon {
                    Some(h) => concordance_at_or_before_horizon(time[j], h),
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
                } else if risk_scores[i] == risk_scores[j] {
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
                } else if risk_scores[i] == risk_scores[j] {
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
        while group_end < n && same_time(time[time_order[group_end]], current_time) {
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
                if event[idx] != 1
                    || horizon.is_some_and(|h| !concordance_at_or_before_horizon(time[idx], h))
                {
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
    match event_time_multipliers.binary_search_by(|&(time, _)| time.total_cmp(&event_time)) {
        Ok(idx) => event_time_multipliers[idx].1,
        Err(idx) => {
            if idx < event_time_multipliers.len()
                && same_time(event_time_multipliers[idx].0, event_time)
            {
                event_time_multipliers[idx].1
            } else if idx > 0 && same_time(event_time_multipliers[idx - 1].0, event_time) {
                event_time_multipliers[idx - 1].1
            } else {
                0.0
            }
        }
    }
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
        while group_end < n && same_time(time[time_order[group_end]], current_time) {
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
    let mut event_order: Vec<usize> = (0..n).filter(|&idx| event[idx] == 1).collect();
    event_order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]).then_with(|| a.cmp(&b)));

    let mut start_order: Vec<usize> = (0..n).collect();
    start_order.sort_by(|&a, &b| start[a].total_cmp(&start[b]));
    let mut stop_order: Vec<usize> = (0..n).collect();
    stop_order.sort_by(|&a, &b| stop[a].total_cmp(&stop[b]));

    let mut active_weight = 0.0;
    let mut start_cursor = 0usize;
    let mut stop_cursor = 0usize;
    let mut survival = 1.0;
    let mut multipliers = Vec::new();
    let mut event_group_start = 0usize;

    while event_group_start < event_order.len() {
        let event_time = stop[event_order[event_group_start]];
        let mut event_group_end = event_group_start + 1;
        while event_group_end < event_order.len()
            && stop[event_order[event_group_end]] == event_time
        {
            event_group_end += 1;
        }

        while start_cursor < n && start[start_order[start_cursor]] < event_time {
            active_weight += observation_weight(weights, start_order[start_cursor]);
            start_cursor += 1;
        }
        while stop_cursor < n && stop[stop_order[stop_cursor]] < event_time {
            active_weight -= observation_weight(weights, stop_order[stop_cursor]);
            stop_cursor += 1;
        }

        let death_weight = event_order[event_group_start..event_group_end]
            .iter()
            .map(|&idx| observation_weight(weights, idx))
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

        event_group_start = event_group_end;
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
    let tie_end = risk_levels.partition_point(|&risk| risk <= risk_score);

    let lower_risk_count = prefix_count_before(at_risk, less_end);
    let lower_or_tied_count = prefix_count_before(at_risk, tie_end);
    let tied_count = lower_or_tied_count - lower_risk_count;

    lower_risk_count + TIED_PAIR_WEIGHT * tied_count
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
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

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

/// Standard normal quantile (R's `qnorm(p)`); `p <= 0` gives `-Inf` and
/// `p >= 1` gives `+Inf` instead of R's NaN for out-of-range input.
#[inline]
pub(crate) fn normal_inverse_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    qnorm(p, true, false)
}

/// `qnorm(1 - alpha/2)`, the two-sided normal critical value, or `None`
/// when `alpha` is not in `(0, 1)`.
#[inline]
pub(crate) fn two_sided_normal_quantile(alpha: f64) -> Option<f64> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }

    let z = qnorm(alpha / 2.0, false, false);
    z.is_finite().then_some(z)
}

/// Gamma quantile with unit scale (R's `qgamma(p, a)`); `p <= 0` gives 0
/// and `p >= 1` gives `+Inf`.
#[inline]
pub(crate) fn gamma_inverse_cdf(p: f64, a: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    qgamma(p, a, 1.0, true, false)
}

/// Chi-squared survival function (R's `pchisq(x, df, lower.tail = FALSE)`),
/// evaluated directly in the upper tail; 1 for `x <= 0` or `df == 0`.
#[inline]
pub(crate) fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }
    pchisq(x, df as f64, false, false)
}

/// Chi-squared distribution function (R's `pchisq(x, df)`); zero for
/// `x <= 0` or a non-positive `df`.
#[inline]
pub(crate) fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    pchisq(x, df, true, false)
}

/// `log|gamma(x)|` (R's `lgamma(x)`).
#[inline]
pub(crate) fn ln_gamma(x: f64) -> f64 {
    lgammafn(x)
}

/// Student t density (R's `dt(x, df)`).
#[inline]
pub(crate) fn student_t_pdf(value: f64, df: f64) -> f64 {
    crate::internal::dist::dt(value, df, false)
}

/// Student t distribution function (R's `pt(x, df)`).
#[inline]
pub(crate) fn student_t_cdf(value: f64, df: f64) -> f64 {
    pt(value, df, true, false)
}

/// Student t quantile (R's `qt(p, df)`); NaN outside `[0, 1]`.
#[inline]
pub(crate) fn student_t_inverse_cdf(probability: f64, df: f64) -> f64 {
    qt(probability, df, true, false)
}

/// Regularized lower incomplete gamma function `P(a, x)` (R's
/// `pgamma(x, a)`); zero for `x < 0` or a non-positive shape.
#[inline]
pub(crate) fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    pgamma(x, a, 1.0, true, false)
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

    fn assert_concordance_summary_close(actual: ConcordanceSummary, expected: ConcordanceSummary) {
        assert!(
            (actual.concordant - expected.concordant).abs() < 1e-12,
            "concordant {} differed from {}",
            actual.concordant,
            expected.concordant
        );
        assert!(
            (actual.comparable - expected.comparable).abs() < 1e-12,
            "comparable {} differed from {}",
            actual.comparable,
            expected.comparable
        );
        assert!(
            (actual.c_index() - expected.c_index()).abs() < 1e-12,
            "c-index {} differed from {}",
            actual.c_index(),
            expected.c_index()
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
    fn test_ranked_concordance_distinguishes_near_equal_risk_scores() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1, 1, 1, 1];
        let risk = [0.4, 0.4 + DIVISION_FLOOR / 2.0, 0.1, 0.8];
        let tied_risk = [0.4, 0.4, 0.1, 0.8];

        assert_ranked_matches_quadratic(&risk, &time, &event, None);
        let distinct = concordance_summary_with_horizon(&risk, &time, &event, None);
        let tied = concordance_summary_with_horizon(&tied_risk, &time, &event, None);
        assert!((tied.concordant - distinct.concordant - 0.5).abs() < DIVISION_FLOOR);
    }

    #[test]
    fn test_right_censored_concordance_groups_near_tied_event_times() {
        let exact_time = [1.0, 1.0, 2.0, 3.0];
        let near_time = [1.0, 1.0 + TIME_EPSILON / 2.0, 2.0, 3.0];
        let event = [1, 1, 1, 0];
        let risk = [0.9, 0.1, 0.5, 0.2];

        for horizon in [None, Some(1.0)] {
            let exact = concordance_summary_with_horizon(&risk, &exact_time, &event, horizon);
            let near = concordance_summary_with_horizon(&risk, &near_time, &event, horizon);
            let near_quadratic = concordance_summary_quadratic(
                &risk,
                &near_time,
                &event,
                None,
                horizon,
                ConcordanceTimeWeight::N,
            );

            assert_concordance_summary_close(near, exact);
            assert_concordance_summary_close(near, near_quadratic);
        }
    }

    #[test]
    fn test_time_weighted_concordance_groups_near_tied_event_times() {
        let exact_time = [1.0, 1.0, 2.0, 3.0, 4.0];
        let near_time = [1.0, 1.0 + TIME_EPSILON / 2.0, 2.0, 3.0, 4.0];
        let event = [1, 1, 1, 0, 1];
        let risk = [0.9, 0.1, 0.5, 0.2, 0.4];
        let weights = [1.0, 2.0, 1.0, 1.5, 0.5];

        for time_weight in [
            ConcordanceTimeWeight::S,
            ConcordanceTimeWeight::SOverG,
            ConcordanceTimeWeight::NOverG2,
            ConcordanceTimeWeight::I,
        ] {
            let exact = concordance_summary_ranked(
                &risk,
                &exact_time,
                &event,
                Some(&weights),
                None,
                time_weight,
            );
            let near = concordance_summary_ranked(
                &risk,
                &near_time,
                &event,
                Some(&weights),
                None,
                time_weight,
            );
            let near_quadratic = concordance_summary_quadratic(
                &risk,
                &near_time,
                &event,
                Some(&weights),
                None,
                time_weight,
            );

            assert_concordance_summary_close(near, exact);
            assert_concordance_summary_close(near, near_quadratic);
        }
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
    fn test_counting_process_concordance_groups_duplicate_event_indices() {
        let start = [0.0, 0.0, 0.25, 0.5, 0.0, 1.0, 1.0];
        let stop = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let event = [1, 1, 1, 1, 0, 1, 0];
        let risk = [0.9, 0.2, 0.7, 0.4, 0.1, 0.8, 0.3];
        let weights = [1.0, 0.0, 2.0, 1.5, 0.5, 3.0, 1.0];

        assert_counting_ranked_matches_quadratic(&risk, &start, &stop, &event);
        assert_weighted_counting_ranked_matches_quadratic(&risk, &start, &stop, &event, &weights);
        for time_weight in [
            ConcordanceTimeWeight::S,
            ConcordanceTimeWeight::NOverG2,
            ConcordanceTimeWeight::I,
        ] {
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
    #[allow(clippy::excessive_precision)]
    fn test_chi2_sf_basic() {
        assert!((chi2_sf(0.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(-1.0, 1) - 1.0).abs() < 1e-10);
        assert!((chi2_sf(1.0, 0) - 1.0).abs() < 1e-10);
        // R: pchisq(3.84, 1, lower.tail = FALSE)
        assert!((chi2_sf(3.84, 1) - 0.050043521248705224).abs() < 1e-16);
        assert!((chi2_cdf(3.84, 1.0) - 0.94995647875129474).abs() < 1e-15);
        assert_eq!(chi2_cdf(0.0, 1.0), 0.0);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_ln_gamma() {
        assert!(ln_gamma(1.0).abs() < 1e-10);
        assert!(ln_gamma(2.0).abs() < 1e-10);
        // R: lgamma(0.5) = log(sqrt(pi))
        assert!((ln_gamma(0.5) - 0.57236494292470008).abs() < 1e-15);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn normal_helpers_match_r() {
        // R: pnorm(1.96), qnorm(0.975), pnorm(-10)
        assert!((normal_cdf(1.96) - 0.97500210485177963).abs() < 1e-15);
        assert!((normal_inverse_cdf(0.975) - 1.9599639845400536).abs() < 1e-15);
        assert!((normal_cdf(-10.0) / 7.6198530241605269e-24 - 1.0).abs() < 1e-14);
        assert_eq!(normal_inverse_cdf(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_inverse_cdf(1.0), f64::INFINITY);
        assert_eq!(normal_inverse_cdf(-0.1), f64::NEG_INFINITY);
        assert!(normal_inverse_cdf(f64::NAN).is_nan());
        assert!((probit(0.025) + 1.9599639845400536).abs() < 1e-15);
        assert!((two_sided_normal_quantile(0.05).unwrap() - 1.9599639845400536).abs() < 1e-15);
        assert_eq!(two_sided_normal_quantile(0.0), None);
        assert_eq!(two_sided_normal_quantile(1.0), None);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn erf_helpers_match_reference_values() {
        // R: 2 * pnorm(x * sqrt(2)) - 1 and 2 * pnorm(x * sqrt(2), lower = FALSE),
        // erf(3) against its true value 0.99997790950300141456...; erf(1e-8) is
        // compared with 2/sqrt(pi) * 1e-8, which R's own expression cannot
        // resolve.
        assert!((erf(0.5) - 0.52049987781304652).abs() < 1e-16);
        assert!((erf(-0.5) + 0.52049987781304652).abs() < 1e-16);
        assert!((erf(3.0) - 0.99997790950300141).abs() < 1.2e-16);
        assert!((erfc(3.0) / 2.2090496998585394e-05 - 1.0).abs() < 1e-15);
        assert!((erf(1e-8) / 1.1283791670955126e-08 - 1.0).abs() < 1e-15);
        assert!((erfc(-3.0) - (2.0 - 2.2090496998585394e-05)).abs() < 1e-15);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erfc(0.0), 1.0);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn student_t_helpers_match_reference_values_and_boundaries() {
        assert!((student_t_pdf(1.0, 5.0) - 0.21967979735098059).abs() < 1e-16);
        assert!((student_t_cdf(1.0, 5.0) - 0.81839126617543867).abs() < 1e-15);
        assert_eq!(student_t_pdf(f64::INFINITY, 5.0), 0.0);
        assert!(student_t_pdf(f64::NAN, 5.0).is_nan());
        assert_eq!(student_t_cdf(f64::NEG_INFINITY, 5.0), 0.0);
        assert_eq!(student_t_cdf(f64::INFINITY, 5.0), 1.0);
        assert_eq!(student_t_inverse_cdf(0.0, 5.0), f64::NEG_INFINITY);
        assert_eq!(student_t_inverse_cdf(1.0, 5.0), f64::INFINITY);
        assert!(student_t_inverse_cdf(1.5, 5.0).is_nan());
        assert!(student_t_inverse_cdf(f64::NAN, 5.0).is_nan());

        for probability in [0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
            let quantile = student_t_inverse_cdf(probability, 5.0);
            assert!((student_t_cdf(quantile, 5.0) - probability).abs() < 1e-15);
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn test_gamma_helpers() {
        // R: qgamma(0.475, 5), qgamma(0.525, 6), pgamma(4.5, 5)
        assert!((gamma_inverse_cdf(0.475, 5.0) - 4.5375048990088311).abs() < 1e-14);
        assert!((gamma_inverse_cdf(0.525, 6.0) - 5.8200445519969533).abs() < 1e-14);
        assert!((lower_incomplete_gamma(5.0, 4.5) - 0.46789642362528439).abs() < 1e-15);
        assert_eq!(gamma_inverse_cdf(0.0, 5.0), 0.0);
        assert_eq!(gamma_inverse_cdf(1.0, 5.0), f64::INFINITY);
        assert_eq!(lower_incomplete_gamma(5.0, 0.0), 0.0);
        assert_eq!(lower_incomplete_gamma(0.0, 1.0), 0.0);
        assert_eq!(lower_incomplete_gamma(5.0, -1.0), 0.0);
    }
}
