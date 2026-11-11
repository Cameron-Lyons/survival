//! Nested risk sets need only reverse positive risk sums and forward positive
//! hazard sums. Keep deaths separate for the Efron event correction; never
//! recover a small interval by subtracting two large cumulative hazards.

use super::log_add;
use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::TieMethod;
use crate::regression::log_scale::LogScale;

struct TimeHazard {
    first: usize,
    end: usize,
    hazard: LogScale,
    event_hazard: LogScale,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn martingale(
    rows: &[usize],
    by_stop: &[usize],
    log_risk: &[LogScale],
    stop: &[f64],
    status: &[i32],
    log_score: &[LogScale],
    weights: &[f64],
    method: TieMethod,
    residuals: &mut [f64],
) -> SurvivalResult<()> {
    let mut risk = LogScale::EMPTY;
    let mut records = Vec::new();
    let mut end = by_stop.len();
    while end > 0 {
        let time = stop[rows[by_stop[end - 1]]];
        let mut first = end - 1;
        while first > 0 && stop[rows[by_stop[first - 1]]] == time {
            first -= 1;
        }
        let mut nondeath_risk = risk;
        let mut death_risk = LogScale::EMPTY;
        let mut death_weight = f64::NEG_INFINITY;
        let mut ndead = 0;
        for &position in &by_stop[first..end] {
            let row = rows[position];
            if status[row] == 1 {
                ndead += 1;
                death_weight = log_add(death_weight, weights[row].ln());
                death_risk = death_risk.mix(log_risk[position]).0;
            } else {
                nondeath_risk = nondeath_risk.mix(log_risk[position]).0;
            }
        }
        risk = nondeath_risk.mix(death_risk).0;
        if risk.is_zero() && death_weight != f64::NEG_INFINITY {
            return Err(SurvivalError::invalid_input(
                "positive-weight events require a positive risk-set sum",
            ));
        }
        let (mut hazard, mut event_hazard) = (LogScale::EMPTY, LogScale::EMPTY);
        if death_weight != f64::NEG_INFINITY {
            let steps = if method.is_efron() { ndead } else { 1 };
            let average = death_weight - (steps as f64).ln();
            for step in 0..steps {
                let fraction = step as f64 / ndead as f64;
                let adjusted = if step == 0 {
                    risk
                } else {
                    nondeath_risk.mix(death_risk.shifted((-fraction).ln_1p())).0
                };
                let increment = adjusted.reciprocal(average);
                hazard = hazard.mix(increment).0;
                event_hazard = event_hazard.mix(increment.shifted((-fraction).ln_1p())).0;
            }
        }
        records.push(TimeHazard {
            first,
            end,
            hazard,
            event_hazard,
        });
        end = first;
    }
    let mut cumulative = LogScale::EMPTY;
    for record in records.iter().rev() {
        for &position in &by_stop[record.first..record.end] {
            let row = rows[position];
            let increment = if status[row] == 1 {
                record.event_hazard
            } else {
                record.hazard
            };
            let exposure = cumulative.mix(increment).0;
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
        cumulative = cumulative.mix(record.hazard).0;
    }
    Ok(())
}
