use crate::internal::typed_inputs::{AndersenGillInput, CountingProcessData, Weights};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
pub(crate) struct AgmartData<'a> {
    pub(crate) start: &'a [f64],
    pub(crate) stop: &'a [f64],
    pub(crate) event: &'a [i32],
    pub(crate) score: &'a [f64],
    pub(crate) wt: &'a [f64],
    pub(crate) strata: &'a [i32],
}

fn cumulative_hazard_at(times: &[f64], cumulative: &[f64], time: f64) -> f64 {
    let position = times.partition_point(|&value| value <= time);
    if position == 0 {
        0.0
    } else {
        cumulative[position - 1]
    }
}

fn compute_stratum_agmart(method: i32, input: &AgmartData<'_>, rows: &[usize], resid: &mut [f64]) {
    if rows.is_empty() {
        return;
    }

    let mut start_order = rows.to_vec();
    start_order.sort_by(|&lhs, &rhs| {
        input.start[lhs]
            .total_cmp(&input.start[rhs])
            .then_with(|| lhs.cmp(&rhs))
    });
    let mut stop_order = rows.to_vec();
    stop_order.sort_by(|&lhs, &rhs| {
        input.stop[lhs]
            .total_cmp(&input.stop[rhs])
            .then_with(|| lhs.cmp(&rhs))
    });
    let mut event_order: Vec<usize> = rows
        .iter()
        .copied()
        .filter(|&idx| input.event[idx] != 0)
        .collect();
    event_order.sort_by(|&lhs, &rhs| {
        input.stop[lhs]
            .total_cmp(&input.stop[rhs])
            .then_with(|| lhs.cmp(&rhs))
    });
    if event_order.is_empty() {
        return;
    }

    let mut event_times = Vec::new();
    let mut cumulative_hazard = Vec::new();
    let mut hazard_increment = Vec::new();
    let mut event_hazard_increment = Vec::new();
    let mut active_risk = 0.0;
    let mut start_ptr = 0usize;
    let mut stop_ptr = 0usize;
    let mut event_ptr = 0usize;
    let mut cumulative = 0.0;

    while event_ptr < event_order.len() {
        let time = input.stop[event_order[event_ptr]];
        while start_ptr < start_order.len() && input.start[start_order[start_ptr]] < time {
            let idx = start_order[start_ptr];
            active_risk += input.score[idx] * input.wt[idx];
            start_ptr += 1;
        }
        while stop_ptr < stop_order.len() && input.stop[stop_order[stop_ptr]] < time {
            let idx = stop_order[stop_ptr];
            active_risk -= input.score[idx] * input.wt[idx];
            stop_ptr += 1;
        }

        let group_start = event_ptr;
        while event_ptr < event_order.len() && input.stop[event_order[event_ptr]] == time {
            event_ptr += 1;
        }
        let deaths = event_ptr - group_start;
        let mut death_weight = 0.0;
        let mut death_risk = 0.0;
        for &idx in &event_order[group_start..event_ptr] {
            death_weight += input.wt[idx];
            death_risk += input.score[idx] * input.wt[idx];
        }
        let step_weight = death_weight / deaths as f64;
        let mut hazard = 0.0;
        let mut event_hazard = 0.0;
        for step in 0..deaths {
            let fraction = if method == 0 {
                0.0
            } else {
                step as f64 / deaths as f64
            };
            let denominator = active_risk - fraction * death_risk;
            hazard += step_weight / denominator;
            event_hazard += step_weight * (1.0 - fraction) / denominator;
        }
        cumulative += hazard;
        event_times.push(time);
        cumulative_hazard.push(cumulative);
        hazard_increment.push(hazard);
        event_hazard_increment.push(event_hazard);
    }

    for &idx in rows {
        let mut integrated_hazard =
            cumulative_hazard_at(&event_times, &cumulative_hazard, input.stop[idx])
                - cumulative_hazard_at(&event_times, &cumulative_hazard, input.start[idx]);
        if input.event[idx] != 0 {
            let event_idx = event_times
                .binary_search_by(|value| value.total_cmp(&input.stop[idx]))
                .expect("an event row must have a matching event time");
            integrated_hazard += event_hazard_increment[event_idx] - hazard_increment[event_idx];
        }
        resid[idx] = input.event[idx] as f64 - input.score[idx] * integrated_hazard;
    }
}

pub(crate) fn compute_agmart(method: i32, input: AgmartData) -> Vec<f64> {
    let n = input.start.len();
    let mut resid = vec![0.0; n];
    let mut stratum_start = 0usize;
    for idx in 0..n {
        if input.strata[idx] == 1 || idx + 1 == n {
            let rows: Vec<usize> = (stratum_start..=idx).collect();
            compute_stratum_agmart(method, &input, &rows, &mut resid);
            stratum_start = idx + 1;
        }
    }
    resid
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
    let strata = input.strata_or_default_cow();
    let data = AgmartData {
        start: &input.counting.start,
        stop: &input.counting.stop,
        event: &input.counting.event,
        score: &input.score,
        wt: weights.as_ref(),
        strata: strata.as_ref(),
    };
    Ok(compute_agmart(method.unwrap_or(0), data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force_agmart(method: i32, input: &AgmartData<'_>) -> Vec<f64> {
        let mut residuals: Vec<f64> = input.event.iter().map(|&value| value as f64).collect();
        let mut stratum_start = 0usize;
        for stratum_end in 0..input.start.len() {
            if input.strata[stratum_end] != 1 && stratum_end + 1 != input.start.len() {
                continue;
            }
            let rows: Vec<usize> = (stratum_start..=stratum_end).collect();
            let mut event_times: Vec<f64> = rows
                .iter()
                .filter(|&&idx| input.event[idx] != 0)
                .map(|&idx| input.stop[idx])
                .collect();
            event_times.sort_by(f64::total_cmp);
            event_times.dedup();
            for time in event_times {
                let risk_rows: Vec<usize> = rows
                    .iter()
                    .copied()
                    .filter(|&idx| input.start[idx] < time && input.stop[idx] >= time)
                    .collect();
                let death_rows: Vec<usize> = rows
                    .iter()
                    .copied()
                    .filter(|&idx| input.stop[idx] == time && input.event[idx] != 0)
                    .collect();
                let denominator: f64 = risk_rows
                    .iter()
                    .map(|&idx| input.score[idx] * input.wt[idx])
                    .sum();
                let death_risk: f64 = death_rows
                    .iter()
                    .map(|&idx| input.score[idx] * input.wt[idx])
                    .sum();
                let death_weight: f64 = death_rows.iter().map(|&idx| input.wt[idx]).sum();
                let step_weight = death_weight / death_rows.len() as f64;
                let mut hazard = 0.0;
                let mut event_hazard = 0.0;
                for step in 0..death_rows.len() {
                    let fraction = if method == 0 {
                        0.0
                    } else {
                        step as f64 / death_rows.len() as f64
                    };
                    let adjusted = denominator - fraction * death_risk;
                    hazard += step_weight / adjusted;
                    event_hazard += step_weight * (1.0 - fraction) / adjusted;
                }
                for idx in risk_rows {
                    let increment = if input.stop[idx] == time && input.event[idx] != 0 {
                        event_hazard
                    } else {
                        hazard
                    };
                    residuals[idx] -= input.score[idx] * increment;
                }
            }
            stratum_start = stratum_end + 1;
        }
        residuals
    }

    #[test]
    fn cumulative_sweep_matches_direct_counting_process_risk_sets() {
        let input = AgmartData {
            start: &[1.0, 0.0, 0.5, 0.0, 2.0, 0.0, 1.5, 0.5, 2.5, 1.0],
            stop: &[3.0, 1.0, 3.0, 2.0, 4.0, 2.0, 4.0, 1.5, 5.0, 3.5],
            event: &[1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
            score: &[0.7, 1.2, 0.8, 1.5, 0.9, 0.6, 1.1, 1.3, 0.75, 1.4],
            wt: &[1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 1.2, 1.0, 1.7, 0.9],
            strata: &[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        };

        for method in [0, 1] {
            let actual = compute_agmart(method, AgmartData { ..input });
            let expected = brute_force_agmart(method, &input);
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}
