use crate::core::strata_order::stratum_groups;
use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::TieMethod;
use crate::regression::log_risk_sweep::has_common_entry;
use crate::regression::log_scale::LogScale;

mod common_entry;

pub(super) fn needs_log_space(score: &[f64], weights: &[f64]) -> bool {
    score.iter().zip(weights).any(|(&risk, &weight)| {
        !(1e-100..=1e100).contains(&risk)
            || !(1e-100..=1e100).contains(&weight)
            || !(1e-100..=1e100).contains(&(risk * weight))
    })
}

fn log_add(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let high = left.max(right);
    high + (left.min(right) - high).exp().ln_1p()
}

struct LogSumTree {
    base: usize,
    sums: Vec<LogScale>,
}

impl LogSumTree {
    fn new(values: &[LogScale]) -> Self {
        let base = values.len().max(1).next_power_of_two();
        let mut sums = vec![LogScale::EMPTY; 2 * base];
        sums[base..base + values.len()].copy_from_slice(values);
        for index in (1..base).rev() {
            sums[index] = sums[2 * index].mix(sums[2 * index + 1]).0;
        }
        Self { base, sums }
    }

    fn set(&mut self, row: usize, value: LogScale) {
        let mut index = self.base + row;
        self.sums[index] = value;
        while index > 1 {
            index /= 2;
            self.sums[index] = self.sums[2 * index].mix(self.sums[2 * index + 1]).0;
        }
    }

    fn total(&self) -> LogScale {
        self.sums[1]
    }

    fn range(&self, first: usize, end: usize) -> LogScale {
        let (mut left, mut right) = (self.base + first, self.base + end);
        let mut sum = LogScale::EMPTY;
        while left < right {
            if left % 2 == 1 {
                sum = sum.mix(self.sums[left]).0;
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                sum = sum.mix(self.sums[right]).0;
            }
            left /= 2;
            right /= 2;
        }
        sum
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn martingale_log(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> SurvivalResult<Vec<f64>> {
    let log_score: Vec<f64> = score.iter().map(|value| value.ln()).collect();
    martingale_log_predictors(start, stop, status, &log_score, weights, strata, method)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn martingale_log_predictors(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    log_score: &[f64],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> SurvivalResult<Vec<f64>> {
    let scales: Vec<_> = log_score
        .iter()
        .map(|&value| LogScale::from_log(value))
        .collect();
    martingale_log_scales(start, stop, status, &scales, weights, strata, method)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn martingale_log_scales(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    log_score: &[LogScale],
    weights: &[f64],
    strata: &[i32],
    method: TieMethod,
) -> SurvivalResult<Vec<f64>> {
    let mut residuals = vec![0.0; stop.len()];
    for (_, rows) in stratum_groups(strata) {
        if !rows.iter().any(|&row| status[row] == 1) {
            continue;
        }
        let log_risk: Vec<LogScale> = rows
            .iter()
            .map(|&row| log_score[row].shifted(weights[row].ln()))
            .collect();
        let mut by_stop: Vec<usize> = (0..rows.len()).collect();
        by_stop.sort_by(|&left, &right| stop[rows[left]].total_cmp(&stop[rows[right]]));
        if has_common_entry(&rows, stop, start) {
            common_entry::martingale(
                &rows,
                &by_stop,
                &log_risk,
                stop,
                status,
                log_score,
                weights,
                method,
                &mut residuals,
            )?;
            continue;
        }
        let by_start = start.map(|start| {
            let mut order: Vec<usize> = (0..rows.len()).collect();
            order.sort_by(|&left, &right| start[rows[left]].total_cmp(&start[rows[right]]));
            order
        });
        let initial = if start.is_none() {
            log_risk.clone()
        } else {
            vec![LogScale::EMPTY; rows.len()]
        };
        let mut risk = LogSumTree::new(&initial);
        let (mut entered, mut departed, mut first) = (0, 0, 0);
        let mut times = Vec::new();
        let mut hazards = Vec::new();
        let mut event_hazards = Vec::new();
        while first < rows.len() {
            let group_start = first;
            let time = stop[rows[by_stop[first]]];
            let mut end = first + 1;
            while end < rows.len() && stop[rows[by_stop[end]]] == time {
                end += 1;
            }
            let mut ndead = 0;
            let mut death_weight = f64::NEG_INFINITY;
            let mut death_risk = LogScale::EMPTY;
            for &position in &by_stop[first..end] {
                let row = rows[position];
                if status[row] == 1 {
                    ndead += 1;
                    death_weight = log_add(death_weight, weights[row].ln());
                    death_risk = death_risk.mix(log_risk[position]).0;
                }
            }
            first = end;
            if ndead == 0 {
                continue;
            }
            if let (Some(start), Some(order)) = (start, &by_start) {
                while entered < rows.len() && start[rows[order[entered]]] < time {
                    let position = order[entered];
                    risk.set(position, log_risk[position]);
                    entered += 1;
                }
            }
            while departed < rows.len() && stop[rows[by_stop[departed]]] < time {
                risk.set(by_stop[departed], LogScale::EMPTY);
                departed += 1;
            }
            let denominator = risk.total();
            if denominator.is_zero() && death_weight != f64::NEG_INFINITY {
                return Err(SurvivalError::invalid_input(
                    "positive-weight events require a positive risk-set sum",
                ));
            }
            let (mut hazard, mut event_hazard) = (LogScale::EMPTY, LogScale::EMPTY);
            if death_weight != f64::NEG_INFINITY {
                let steps = if method.is_efron() { ndead } else { 1 };
                let average = death_weight - (steps as f64).ln();
                let nondeath_risk = if steps > 1 {
                    for &position in &by_stop[group_start..end] {
                        if status[rows[position]] == 1 {
                            risk.set(position, LogScale::EMPTY);
                        }
                    }
                    let remaining = risk.total();
                    for &position in &by_stop[group_start..end] {
                        if status[rows[position]] == 1 {
                            risk.set(position, log_risk[position]);
                        }
                    }
                    remaining
                } else {
                    LogScale::EMPTY
                };
                for step in 0..steps {
                    let fraction = step as f64 / ndead as f64;
                    let adjusted = if step == 0 {
                        denominator
                    } else {
                        nondeath_risk.mix(death_risk.shifted((-fraction).ln_1p())).0
                    };
                    let increment = adjusted.reciprocal(average);
                    hazard = hazard.mix(increment).0;
                    event_hazard = event_hazard.mix(increment.shifted((-fraction).ln_1p())).0;
                }
            }
            times.push(time);
            hazards.push(hazard);
            event_hazards.push(event_hazard);
        }
        let integrals = LogSumTree::new(&hazards);
        for &row in &rows {
            let first = start.map_or(0, |start| times.partition_point(|&time| time <= start[row]));
            let end = times.partition_point(|&time| time <= stop[row]);
            let exposure = if status[row] == 1 {
                integrals
                    .range(first, end - 1)
                    .mix(event_hazards[end - 1])
                    .0
            } else {
                integrals.range(first, end)
            };
            let expected = if log_score[row].is_zero() || exposure.is_zero() {
                0.0
            } else {
                exposure.exp_at_scale(log_score[row])
            };
            if !expected.is_finite() {
                return Err(SurvivalError::computation(
                    "martingale expected count is not finite",
                ));
            }
            residuals[row] = f64::from(status[row]) - expected;
        }
    }
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_weighted_sets_keep_huge_origins_out_of_hazard_masses() {
        for shift in [1e16, -1e16, 1e308, -1e308] {
            for method in [TieMethod::Breslow, TieMethod::Efron] {
                let residuals = martingale_log_predictors(
                    Some(&[0., 0., 2., 2.]),
                    &[1., 1., 3., 3.],
                    &[1, 0, 1, 0],
                    &[0., 0., shift, shift],
                    &[1., 3., 1., 3.],
                    &[0; 4],
                    method,
                )
                .unwrap();
                for (actual, expected) in residuals.iter().zip([0.75, -0.25, 0.75, -0.25]) {
                    assert!((actual - expected).abs() < 1e-15);
                }
            }
        }
    }

    #[test]
    fn tied_huge_origins_preserve_efron_fractions_and_small_predictors() {
        for origin in [1e16, -1e16, 1e308, -1e308] {
            let residuals = martingale_log_predictors(
                None,
                &[1.; 3],
                &[1, 1, 0],
                &[origin; 3],
                &[1., 3., 2.],
                &[0; 3],
                TieMethod::Efron,
            )
            .unwrap();
            for (actual, expected) in residuals.iter().zip([5. / 12., 5. / 12., -5. / 6.]) {
                assert!((actual - expected).abs() < 2e-15);
            }
            let residuals = martingale_log_scales(
                None,
                &[1.; 2],
                &[1, 0],
                &[LogScale::new(origin, -1.), LogScale::new(origin, 0.5)],
                &[1., 3.],
                &[0; 2],
                TieMethod::Breslow,
            )
            .unwrap();
            let denominator = (-1_f64).exp() + 3. * 0.5_f64.exp();
            assert!((residuals[0] - (1. - (-1_f64).exp() / denominator)).abs() < 1e-15);
            assert!((residuals[1] + 0.5_f64.exp() / denominator).abs() < 1e-15);
        }
    }

    #[test]
    fn compensated_predictors_preserve_martingale_row_multiplicities() {
        let risk = LogScale::from_predictor_parts(1e308, 1e292);
        let residuals = martingale_log_scales(
            None,
            &[1.; 3],
            &[1, 0, 0],
            &[risk; 3],
            &[1.; 3],
            &[0; 3],
            TieMethod::Breslow,
        )
        .unwrap();
        for (actual, expected) in residuals.iter().zip([2. / 3., -1. / 3., -1. / 3.]) {
            assert!((actual - expected).abs() < 1e-15);
        }
    }
}
