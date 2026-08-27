use crate::internal::typed_inputs::{AndersenGillInput, CountingProcessData, Weights};
use pyo3::prelude::*;
use std::collections::BTreeMap;

pub(crate) struct AgmartData<'a> {
    pub(crate) start: &'a [f64],
    pub(crate) stop: &'a [f64],
    pub(crate) event: &'a [i32],
    pub(crate) score: &'a [f64],
    pub(crate) wt: &'a [f64],
    pub(crate) strata: &'a [i32],
}

fn used_rows(input: &AgmartData<'_>) -> Vec<bool> {
    let n = input.start.len();
    let mut event_times_by_stratum: BTreeMap<i32, Vec<f64>> = BTreeMap::new();
    for idx in 0..n {
        if input.event[idx] > 0 {
            event_times_by_stratum
                .entry(input.strata[idx])
                .or_default()
                .push(input.stop[idx]);
        }
    }
    for times in event_times_by_stratum.values_mut() {
        times.sort_by(f64::total_cmp);
        times.dedup();
    }

    (0..n)
        .map(|idx| {
            let Some(times) = event_times_by_stratum.get(&input.strata[idx]) else {
                return false;
            };
            let after_start = times.partition_point(|&time| time <= input.start[idx]);
            let through_stop = times.partition_point(|&time| time <= input.stop[idx]);
            after_start < through_stop
        })
        .collect()
}

/// Compute counting-process martingale residuals in two descending-time sweeps.
///
/// `strata` contains row-level stratum identifiers. Rows whose `(start, stop]`
/// interval contains no event time are excluded from both risk-set sums and
/// retain a zero residual.
pub(crate) fn compute_agmart_by_stratum(method: i32, input: AgmartData<'_>) -> Vec<f64> {
    let n = input.start.len();
    let mut resid = vec![0.0; n];
    if n == 0 {
        return resid;
    }

    debug_assert_eq!(input.stop.len(), n);
    debug_assert_eq!(input.event.len(), n);
    debug_assert_eq!(input.score.len(), n);
    debug_assert_eq!(input.wt.len(), n);
    debug_assert_eq!(input.strata.len(), n);

    let used = used_rows(&input);
    let mut start_order: Vec<usize> = (0..n).filter(|&idx| used[idx]).collect();
    let mut stop_order = start_order.clone();
    start_order.sort_by(|&lhs, &rhs| {
        input.strata[lhs]
            .cmp(&input.strata[rhs])
            .then_with(|| input.start[rhs].total_cmp(&input.start[lhs]))
            .then_with(|| lhs.cmp(&rhs))
    });
    stop_order.sort_by(|&lhs, &rhs| {
        input.strata[lhs]
            .cmp(&input.strata[rhs])
            .then_with(|| input.stop[rhs].total_cmp(&input.stop[lhs]))
            .then_with(|| lhs.cmp(&rhs))
    });
    let nused = stop_order.len();
    if nused == 0 {
        return resid;
    }

    let mut at_risk = vec![false; n];
    let mut start_pos = 0;
    let mut stop_pos = 0;
    let mut denominator = 0.0;
    let mut cumulative_hazard = 0.0;
    let mut stratum = input.strata[stop_order[0]];

    while stop_pos < nused {
        let mut scan = stop_pos;
        let death_time = loop {
            if scan >= nused {
                break None;
            }
            let row_idx = stop_order[scan];
            if input.strata[row_idx] != stratum {
                while start_pos < nused {
                    let start_idx = start_order[start_pos];
                    if input.strata[start_idx] != stratum {
                        break;
                    }
                    resid[start_idx] -= cumulative_hazard * input.score[start_idx];
                    start_pos += 1;
                }
                cumulative_hazard = 0.0;
                denominator = 0.0;
                stratum = input.strata[row_idx];
                stop_pos = start_pos;
                scan = stop_pos;
                continue;
            }
            if input.event[row_idx] > 0 {
                break Some(input.stop[row_idx]);
            }
            scan += 1;
        };
        let Some(death_time) = death_time else {
            break;
        };

        while start_pos < nused {
            let row_idx = start_order[start_pos];
            if input.strata[row_idx] != stratum || input.start[row_idx] < death_time {
                break;
            }
            if at_risk[row_idx] {
                denominator -= input.score[row_idx] * input.wt[row_idx];
                resid[row_idx] -= cumulative_hazard * input.score[row_idx];
            }
            start_pos += 1;
        }

        let tied_start = stop_pos;
        let mut deaths = 0usize;
        let mut death_risk_sum = 0.0;
        let mut death_weight_sum = 0.0;
        while stop_pos < nused {
            let row_idx = stop_order[stop_pos];
            if input.strata[row_idx] != stratum || input.stop[row_idx] < death_time {
                break;
            }
            if input.event[row_idx] == 1 {
                at_risk[row_idx] = true;
                resid[row_idx] = 1.0 + cumulative_hazard * input.score[row_idx];
                deaths += 1;
                let weighted_risk = input.score[row_idx] * input.wt[row_idx];
                denominator += weighted_risk;
                death_risk_sum += weighted_risk;
                death_weight_sum += input.wt[row_idx];
            } else if input.start[row_idx] < death_time {
                at_risk[row_idx] = true;
                denominator += input.score[row_idx] * input.wt[row_idx];
                resid[row_idx] = cumulative_hazard * input.score[row_idx];
            }
            stop_pos += 1;
        }

        debug_assert!(deaths > 0);
        let hazard = if method == 0 || deaths == 1 {
            death_weight_sum / denominator
        } else {
            let step_weight = death_weight_sum / deaths as f64;
            let mut hazard = 0.0;
            let mut death_hazard = 0.0;
            for step in 0..deaths {
                let fraction = step as f64 / deaths as f64;
                let adjusted_denominator = denominator - fraction * death_risk_sum;
                hazard += step_weight / adjusted_denominator;
                death_hazard += step_weight * (1.0 - fraction) / adjusted_denominator;
            }
            let tied_death_adjustment = hazard - death_hazard;
            for &row_idx in &stop_order[tied_start..stop_pos] {
                if input.event[row_idx] > 0 {
                    resid[row_idx] += tied_death_adjustment * input.score[row_idx];
                }
            }
            hazard
        };
        cumulative_hazard += hazard;
    }

    while start_pos < nused {
        let row_idx = start_order[start_pos];
        if at_risk[row_idx] {
            resid[row_idx] -= cumulative_hazard * input.score[row_idx];
        }
        start_pos += 1;
    }
    resid
}

fn boundary_markers_to_strata(boundaries: &[i32]) -> Vec<i32> {
    let mut stratum = 0;
    boundaries
        .iter()
        .map(|&boundary| {
            let current = stratum;
            if boundary == 1 {
                stratum += 1;
            }
            current
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn agmart(
    n: usize,
    method: i32,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    score: Vec<f64>,
    wt: Vec<f64>,
    strata: Vec<i32>,
) -> PyResult<Vec<f64>> {
    let input = AndersenGillInput::try_new(
        CountingProcessData::try_new(start, stop, event)?,
        score,
        Some(Weights::try_new(wt)?),
        Some(strata),
    )?;
    if input.counting.start.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "start length must equal n",
        ));
    }
    agmart_typed(&input, Some(method))
}

#[pyfunction(name = "agmart")]
#[pyo3(signature = (input, method=None))]
pub(crate) fn agmart_typed(input: &AndersenGillInput, method: Option<i32>) -> PyResult<Vec<f64>> {
    let weights = input.weights_or_unit_cow();
    let boundaries = input.strata_or_default_cow();
    let strata = boundary_markers_to_strata(boundaries.as_ref());
    let data = AgmartData {
        start: &input.counting.start,
        stop: &input.counting.stop,
        event: &input.counting.event,
        score: &input.score,
        wt: weights.as_ref(),
        strata: &strata,
    };
    Ok(compute_agmart_by_stratum(method.unwrap_or(0), data))
}
