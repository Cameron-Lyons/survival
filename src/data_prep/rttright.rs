use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

use super::aeq_surv_module::aeq_surv;
use crate::constants::{DIVISION_FLOOR, same_time};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_negative,
};

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RttrightResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub order: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (time, status, weights=None, timefix=true, renorm=true))]
pub fn rttright(
    time: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<RttrightResult> {
    rttright_impl(time, status, weights, timefix, renorm)
}

fn rttright_impl(
    time: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<RttrightResult> {
    let n = time.len();

    if status.len() != n {
        return Err(PyValueError::new_err(
            "time and status must have same length",
        ));
    }

    let weights_ref = weights.as_deref();
    if let Some(init_weights) = weights_ref
        && init_weights.len() != n
    {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    validate_rttright_inputs(&time, &status, weights_ref)?;

    if n == 0 {
        return Ok(RttrightResult {
            weights: vec![],
            time: vec![],
            status: vec![],
            order: vec![],
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));

    let sorted_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
    let sorted_weights: Vec<f64> = indices
        .iter()
        .map(|&i| rttright_case_weight(weights_ref, i))
        .collect();
    let sorted_weights = normalize_case_weights(&sorted_weights, renorm)?;

    let km_weights = compute_km_weights(&sorted_time, &sorted_status, &sorted_weights, timefix);

    Ok(RttrightResult {
        weights: km_weights,
        time: sorted_time,
        status: sorted_status,
        order: indices,
    })
}

fn validate_rttright_inputs(time: &[f64], status: &[i32], weights: Option<&[f64]>) -> PyResult<()> {
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    if let Some(weights) = weights {
        validate_no_nan(weights, "weights")?;
        validate_finite(weights, "weights")?;
        validate_non_negative(weights, "weights")?;
    }

    validate_binary_i32(status, "status")?;

    Ok(())
}

#[inline]
fn rttright_case_weight(weights: Option<&[f64]>, index: usize) -> f64 {
    weights.map_or(1.0, |wts| wts[index])
}

fn normalize_case_weights(weights: &[f64], renorm: bool) -> PyResult<Vec<f64>> {
    if !renorm {
        return Ok(weights.to_vec());
    }

    let total = weights.iter().sum::<f64>();
    if total <= DIVISION_FLOOR {
        return Err(PyValueError::new_err(
            "weights must have positive sum when renorm is true",
        ));
    }

    Ok(weights.iter().map(|weight| weight / total).collect())
}

fn same_rttright_time(left: f64, right: f64, timefix: bool) -> bool {
    if timefix {
        same_time(left, right)
    } else {
        left == right
    }
}

fn compute_km_weights(
    time: &[f64],
    status: &[i32],
    init_weights: &[f64],
    timefix: bool,
) -> Vec<f64> {
    let n = time.len();
    if n == 0 {
        return vec![];
    }

    let mut weights = vec![0.0; n];
    let mut n_at_risk = init_weights.iter().sum::<f64>();
    let mut current_g = 1.0;

    let mut start = 0;
    while start < n {
        let block_time = time[start];
        let mut end = start + 1;
        while end < n && same_rttright_time(time[end], block_time, timefix) {
            end += 1;
        }

        let mut event_weight = 0.0;
        let mut censor_weight = 0.0;
        for row in start..end {
            if status[row] == 1 {
                event_weight += init_weights[row];
                weights[row] = if current_g > DIVISION_FLOOR {
                    init_weights[row] / current_g
                } else {
                    init_weights[row]
                };
            } else {
                censor_weight += init_weights[row];
            }
        }

        let risk_after_events = n_at_risk - event_weight;
        if risk_after_events > DIVISION_FLOOR && censor_weight > 0.0 {
            current_g *= 1.0 - censor_weight / risk_after_events;
        }
        n_at_risk = risk_after_events - censor_weight;
        start = end;
    }

    weights
}

const PARALLEL_TIME_MATRIX_WORK: usize = 8_192;

fn compute_time_matrix_group(
    time: &[f64],
    status: &[i32],
    query_times: &[f64],
    case_weights: &[f64],
    indices: &[usize],
    renorm: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let group_weights = indices
        .iter()
        .map(|&index| case_weights[index])
        .collect::<Vec<_>>();
    let group_weights = if renorm {
        let total = group_weights.iter().sum::<f64>();
        if total <= 0.0 {
            return Err(PyValueError::new_err(
                "weights must have positive sum when renorm is true",
            ));
        }
        group_weights
            .into_iter()
            .map(|weight| weight / total)
            .collect()
    } else {
        group_weights
    };

    let mut order = (0..indices.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        time[indices[left]]
            .total_cmp(&time[indices[right]])
            .then_with(|| left.cmp(&right))
    });

    let mut event_g = vec![1.0; indices.len()];
    let mut block_times = Vec::with_capacity(indices.len());
    let mut post_block_g = Vec::with_capacity(indices.len());
    let mut current_g = 1.0;
    let mut n_at_risk = order
        .iter()
        .map(|&local_index| group_weights[local_index])
        .sum::<f64>();

    let mut start = 0;
    while start < order.len() {
        let block_time = time[indices[order[start]]];
        let mut end = start + 1;
        while end < order.len() && time[indices[order[end]]] == block_time {
            end += 1;
        }

        let mut event_weight = 0.0;
        let mut censor_weight = 0.0;
        for &local_index in &order[start..end] {
            event_g[local_index] = current_g;
            if status[indices[local_index]] == 1 {
                event_weight += group_weights[local_index];
            } else {
                censor_weight += group_weights[local_index];
            }
        }

        let risk_after_events = n_at_risk - event_weight;
        if risk_after_events > 0.0 && censor_weight > 0.0 {
            current_g *= 1.0 - censor_weight / risk_after_events;
        }
        n_at_risk = risk_after_events - censor_weight;
        block_times.push(block_time);
        post_block_g.push(current_g);
        start = end;
    }

    let query_g = query_times
        .iter()
        .map(|query_time| {
            let block_index = block_times.partition_point(|block_time| block_time < query_time);
            if block_index == 0 {
                1.0
            } else {
                post_block_g[block_index - 1]
            }
        })
        .collect::<Vec<_>>();

    let make_row = |local_index: usize| {
        let row_index = indices[local_index];
        let row_time = time[row_index];
        let row_weight = group_weights[local_index];
        let mut row = vec![0.0; query_times.len()];
        if status[row_index] == 1 {
            for (column, &g_at_time) in query_g.iter().enumerate() {
                row[column] = row_weight / event_g[local_index].max(g_at_time);
            }
        } else {
            for (column, (&query_time, &g_at_time)) in
                query_times.iter().zip(query_g.iter()).enumerate()
            {
                if row_time >= query_time {
                    row[column] = row_weight / g_at_time;
                }
            }
        }
        row
    };

    let work = indices.len().saturating_mul(query_times.len());
    if work >= PARALLEL_TIME_MATRIX_WORK {
        Ok((0..indices.len()).into_par_iter().map(make_row).collect())
    } else {
        Ok((0..indices.len()).map(make_row).collect())
    }
}

/// Construct R-compatible redistribution weights at each requested time.
pub fn rttright_time_matrix(
    time: Vec<f64>,
    status: Vec<i32>,
    query_times: Vec<f64>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let n = time.len();
    if status.len() != n {
        return Err(PyValueError::new_err(
            "time and status must have same length",
        ));
    }
    if let Some(init_weights) = weights.as_deref()
        && init_weights.len() != n
    {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    if let Some(strata) = strata.as_deref()
        && strata.len() != n
    {
        return Err(PyValueError::new_err(
            "time, status, and strata must have same length",
        ));
    }
    validate_rttright_inputs(&time, &status, weights.as_deref())?;
    validate_finite(&query_times, "times")?;

    let time = if timefix {
        crate::data_prep::aeq_surv_module::aeq_surv(time, None)?.time
    } else {
        time
    };
    let case_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let mut strata_indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, stratum) in strata.into_iter().enumerate() {
        strata_indices.entry(stratum).or_default().push(index);
    }

    let mut matrix = vec![vec![0.0; query_times.len()]; n];
    for indices in strata_indices.values() {
        let rows = compute_time_matrix_group(
            &time,
            &status,
            &query_times,
            &case_weights,
            indices,
            renorm,
        )?;
        for (&row_index, row) in indices.iter().zip(rows) {
            matrix[row_index] = row;
        }
    }
    Ok(matrix)
}

struct RttrightCountingInputs<'a> {
    start: &'a [f64],
    stop: &'a [f64],
    status: &'a [i32],
    weights: &'a [f64],
    last: &'a [bool],
    strata: &'a [i32],
    times: Option<&'a [f64]>,
    delta: f64,
}

fn validate_rttright_counting_inputs(inputs: &RttrightCountingInputs<'_>) -> PyResult<()> {
    let n = inputs.stop.len();
    for (name, len) in [
        ("start", inputs.start.len()),
        ("status", inputs.status.len()),
        ("weights", inputs.weights.len()),
        ("last", inputs.last.len()),
        ("strata", inputs.strata.len()),
    ] {
        if len != n {
            return Err(PyValueError::new_err(format!(
                "{name} must have same length as stop"
            )));
        }
    }
    validate_finite(inputs.start, "start")?;
    validate_finite(inputs.stop, "stop")?;
    validate_binary_i32(inputs.status, "status")?;
    validate_finite(inputs.weights, "weights")?;
    validate_non_negative(inputs.weights, "weights")?;
    if let Some(times) = inputs.times {
        validate_finite(times, "times")?;
    }
    if !inputs.delta.is_finite() || inputs.delta <= 0.0 {
        return Err(PyValueError::new_err(
            "delta must be finite and greater than zero",
        ));
    }
    if let Some(index) = inputs
        .start
        .iter()
        .zip(inputs.stop)
        .position(|(&left, &right)| left >= right)
    {
        return Err(PyValueError::new_err(format!(
            "start must be less than stop at index {index}"
        )));
    }
    Ok(())
}

fn counting_censor_survival(
    start: &[f64],
    km_stop: &[f64],
    censor: &[bool],
    weights: &[f64],
    indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let mut censor_events = indices
        .iter()
        .filter_map(|&index| censor[index].then_some((km_stop[index], weights[index])))
        .collect::<Vec<_>>();
    censor_events.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    if censor_events.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut start_order = indices.to_vec();
    start_order.sort_unstable_by(|&left, &right| {
        start[left]
            .total_cmp(&start[right])
            .then_with(|| left.cmp(&right))
    });
    let mut stop_order = indices.to_vec();
    stop_order.sort_unstable_by(|&left, &right| {
        km_stop[left]
            .total_cmp(&km_stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut event_times = Vec::with_capacity(censor_events.len());
    let mut survival_values = Vec::with_capacity(censor_events.len());
    let mut current = 1.0;
    let mut active_weight = 0.0;
    let mut active_positive_rows = 0usize;
    let mut start_position = 0;
    let mut stop_position = 0;
    let mut event_position = 0;

    while event_position < censor_events.len() {
        let event_time = censor_events[event_position].0;
        let mut event_end = event_position + 1;
        let mut event_weight = censor_events[event_position].1;
        let mut event_positive_rows = usize::from(event_weight > 0.0);
        while event_end < censor_events.len() && censor_events[event_end].0 == event_time {
            event_weight += censor_events[event_end].1;
            event_positive_rows += usize::from(censor_events[event_end].1 > 0.0);
            event_end += 1;
        }

        while start_position < start_order.len() && start[start_order[start_position]] < event_time
        {
            let weight = weights[start_order[start_position]];
            active_weight += weight;
            active_positive_rows += usize::from(weight > 0.0);
            start_position += 1;
        }
        while stop_position < stop_order.len() && km_stop[stop_order[stop_position]] < event_time {
            let weight = weights[stop_order[stop_position]];
            active_weight -= weight;
            active_positive_rows -= usize::from(weight > 0.0);
            stop_position += 1;
        }

        if active_positive_rows == event_positive_rows {
            active_weight = event_weight;
        }
        if active_weight > 0.0 {
            current *= 1.0 - event_weight / active_weight;
        }
        event_times.push(event_time);
        survival_values.push(current);
        event_position = event_end;
    }

    (event_times, survival_values)
}

#[inline]
fn counting_survival_before(event_times: &[f64], survival_values: &[f64], time: f64) -> f64 {
    let event_index = event_times.partition_point(|&event_time| event_time < time);
    if event_index == 0 {
        1.0
    } else {
        survival_values[event_index - 1]
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_rttright_counting_group(
    start: &[f64],
    stop: &[f64],
    km_stop: &[f64],
    status: &[i32],
    weights: &[f64],
    last: &[bool],
    censor: &[bool],
    indices: &[usize],
    times: Option<&[f64]>,
) -> Vec<Vec<f64>> {
    let (survival_times, survival_values) =
        counting_censor_survival(start, km_stop, censor, weights, indices);

    let Some(times) = times else {
        return indices
            .iter()
            .map(|&index| {
                let value = if last[index] && status[index] > 0 {
                    let g =
                        counting_survival_before(&survival_times, &survival_values, stop[index]);
                    rttright_divide(weights[index], g)
                } else {
                    0.0
                };
                vec![value]
            })
            .collect();
    };

    let query_g = times
        .iter()
        .map(|&time| counting_survival_before(&survival_times, &survival_values, time))
        .collect::<Vec<_>>();
    let make_row = |&index: &usize| {
        let mut row = vec![0.0; times.len()];
        for (column, (&query_time, &g)) in times.iter().zip(&query_g).enumerate() {
            if start[index] < query_time && query_time <= stop[index] {
                row[column] = rttright_divide(weights[index], g);
            }
        }
        if last[index] && stop[index] > 0.0 {
            let stop_g = counting_survival_before(&survival_times, &survival_values, stop[index]);
            for (value, &g) in row.iter_mut().zip(&query_g) {
                *value = rttright_divide(weights[index], stop_g.max(g));
            }
        }
        row
    };
    let work = indices.len().saturating_mul(times.len());
    if work >= PARALLEL_TIME_MATRIX_WORK {
        indices.par_iter().map(make_row).collect()
    } else {
        indices.iter().map(make_row).collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rttright_counting_matrix(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weights: Vec<f64>,
    last: Vec<bool>,
    strata: Vec<i32>,
    delta: f64,
    times: Option<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_rttright_counting_inputs(&RttrightCountingInputs {
        start: &start,
        stop: &stop,
        status: &status,
        weights: &weights,
        last: &last,
        strata: &strata,
        times: times.as_deref(),
        delta,
    })?;

    let mut strata_indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, stratum) in strata.into_iter().enumerate() {
        strata_indices.entry(stratum).or_default().push(index);
    }
    let censor = (0..stop.len())
        .map(|index| last[index] && status[index] == 0)
        .collect::<Vec<_>>();
    let mut km_stop = stop.clone();
    for (index, &is_censor) in censor.iter().enumerate() {
        if is_censor {
            km_stop[index] += delta;
        }
    }
    let width = times.as_ref().map_or(1, Vec::len);
    let mut matrix = vec![vec![0.0; width]; stop.len()];
    for indices in strata_indices.values() {
        let rows = compute_rttright_counting_group(
            &start,
            &stop,
            &km_stop,
            &status,
            &weights,
            &last,
            &censor,
            indices,
            times.as_deref(),
        );
        for (&row_index, row) in indices.iter().zip(rows) {
            matrix[row_index] = row;
        }
    }
    Ok(matrix)
}

#[pyfunction]
#[pyo3(signature = (time, status, strata, weights=None, timefix=true, renorm=true))]
pub fn rttright_stratified(
    time: Vec<f64>,
    status: Vec<i32>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<RttrightResult> {
    let n = time.len();

    if status.len() != n || strata.len() != n {
        return Err(PyValueError::new_err(
            "time, status, and strata must have same length",
        ));
    }

    let weights_ref = weights.as_deref();
    if let Some(init_weights) = weights_ref
        && init_weights.len() != n
    {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    validate_rttright_inputs(&time, &status, weights_ref)?;

    let mut strata_indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, &s) in strata.iter().enumerate() {
        strata_indices.entry(s).or_default().push(i);
    }

    let mut final_weights = vec![0.0; n];
    let mut final_order = vec![0; n];

    let mut offset = 0;
    for indices in strata_indices.values() {
        let strata_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let strata_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
        let strata_weights =
            weights_ref.map(|wts| indices.iter().map(|&i| wts[i]).collect::<Vec<_>>());

        let result = rttright_impl(strata_time, strata_status, strata_weights, timefix, renorm)?;

        for (sorted_pos, &local_idx) in result.order.iter().enumerate() {
            let orig_idx = indices[local_idx];
            final_weights[orig_idx] = result.weights[sorted_pos];
            final_order[offset + sorted_pos] = orig_idx;
        }
        offset += indices.len();
    }

    Ok(RttrightResult {
        weights: final_weights,
        time,
        status,
        order: final_order,
    })
}

#[inline]
fn rttright_divide(numerator: f64, denominator: f64) -> f64 {
    if denominator != 0.0 {
        numerator / denominator
    } else if numerator == 0.0 {
        f64::NAN
    } else {
        f64::INFINITY
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    times,
    strata=None,
    weights=None,
    timefix=true,
    renorm=true
))]
#[allow(clippy::too_many_arguments)]
pub fn rttright_matrix(
    time: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let n = time.len();
    if status.len() != n {
        return Err(PyValueError::new_err(
            "time and status must have same length",
        ));
    }
    if let Some(values) = strata.as_ref()
        && values.len() != n
    {
        return Err(PyValueError::new_err(
            "strata must have same length as time",
        ));
    }
    if let Some(values) = weights.as_ref()
        && values.len() != n
    {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    validate_rttright_inputs(&time, &status, weights.as_deref())?;
    validate_finite(&times, "times")?;

    if n == 0 {
        return Ok(vec![]);
    }

    let time = if timefix {
        aeq_surv(time, None)?.time
    } else {
        time
    };
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let case_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let mut matrix = vec![vec![0.0; times.len()]; n];
    let mut strata_indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (index, &stratum) in strata.iter().enumerate() {
        strata_indices.entry(stratum).or_default().push(index);
    }

    for indices in strata_indices.values_mut() {
        let total_weight = indices
            .iter()
            .map(|&index| case_weights[index])
            .sum::<f64>();
        if renorm && total_weight <= 0.0 {
            return Err(PyValueError::new_err(
                "weights must have positive sum when renorm is true",
            ));
        }
        let scale = if renorm { total_weight } else { 1.0 };
        indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]).then_with(|| a.cmp(&b)));

        let normalized_weights = indices
            .iter()
            .map(|&index| case_weights[index] / scale)
            .collect::<Vec<_>>();
        let mut event_g = vec![1.0; indices.len()];
        let mut block_times = Vec::with_capacity(indices.len());
        let mut post_block_g = Vec::with_capacity(indices.len());
        let mut current_g = 1.0;
        let mut n_at_risk = normalized_weights.iter().sum::<f64>();

        let mut start = 0;
        while start < indices.len() {
            let block_time = time[indices[start]];
            let mut end = start + 1;
            while end < indices.len() && time[indices[end]] == block_time {
                end += 1;
            }

            let mut event_weight = 0.0;
            let mut censor_weight = 0.0;
            for sorted_position in start..end {
                event_g[sorted_position] = current_g;
                if status[indices[sorted_position]] == 1 {
                    event_weight += normalized_weights[sorted_position];
                } else {
                    censor_weight += normalized_weights[sorted_position];
                }
            }

            let risk_after_events = n_at_risk - event_weight;
            if risk_after_events > 0.0 && censor_weight > 0.0 {
                current_g *= 1.0 - censor_weight / risk_after_events;
            }
            n_at_risk = risk_after_events - censor_weight;
            block_times.push(block_time);
            post_block_g.push(current_g);
            start = end;
        }

        let query_g = times
            .iter()
            .map(|query_time| {
                let block_index = block_times.partition_point(|block_time| block_time < query_time);
                if block_index == 0 {
                    1.0
                } else {
                    post_block_g[block_index - 1]
                }
            })
            .collect::<Vec<_>>();

        for (sorted_position, &row_index) in indices.iter().enumerate() {
            let row_weight = normalized_weights[sorted_position];
            if status[row_index] == 1 {
                for (column_index, &g_at_time) in query_g.iter().enumerate() {
                    matrix[row_index][column_index] =
                        rttright_divide(row_weight, event_g[sorted_position].max(g_at_time));
                }
            } else {
                for (column_index, (&query_time, &g_at_time)) in
                    times.iter().zip(&query_g).enumerate()
                {
                    if time[row_index] >= query_time {
                        matrix[row_index][column_index] = rttright_divide(row_weight, g_at_time);
                    }
                }
            }
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::{index_permutations, initialize_python};

    fn assert_close_slice(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected.iter()) {
            assert!((left - right).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    fn assert_nested_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_close_slice(actual_row, expected_row);
        }
    }

    fn naive_counting_censor_survival(
        start: &[f64],
        km_stop: &[f64],
        censor: &[bool],
        weights: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut event_times = (0..start.len())
            .filter_map(|index| censor[index].then_some(km_stop[index]))
            .collect::<Vec<_>>();
        event_times.sort_unstable_by(f64::total_cmp);
        event_times.dedup_by(|left, right| *left == *right);

        let mut survival_values = Vec::with_capacity(event_times.len());
        let mut current = 1.0;
        for &event_time in &event_times {
            let risk = (0..start.len())
                .filter(|&index| start[index] < event_time && event_time <= km_stop[index])
                .map(|index| weights[index])
                .sum::<f64>();
            let events = (0..start.len())
                .filter(|&index| censor[index] && km_stop[index] == event_time)
                .map(|index| weights[index])
                .sum::<f64>();
            if risk > 0.0 {
                current *= 1.0 - events / risk;
            }
            survival_values.push(current);
        }
        (event_times, survival_values)
    }

    fn next_random(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    #[test]
    fn test_rttright_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let result = rttright(time, status, None, true, true).unwrap();

        assert!(result.weights[0] > 0.0);
        assert!(result.weights[2] > 0.0);
        assert!(result.weights[4] > 0.0);

        assert_eq!(result.weights[1], 0.0);
        assert_eq!(result.weights[3], 0.0);
    }

    #[test]
    fn test_rttright_all_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];

        let result = rttright(time, status, None, true, true).unwrap();

        for w in &result.weights {
            assert!((*w - (1.0 / 3.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rttright_matches_r_normalized_right_censoring_weights() {
        let result = rttright(vec![1.0, 2.0, 3.0], vec![0, 1, 1], None, true, true).unwrap();

        assert_close_slice(&result.weights, &[0.0, 0.5, 0.5]);

        let raw = rttright(vec![1.0, 2.0, 3.0], vec![0, 1, 1], None, true, false).unwrap();

        assert_close_slice(&raw.weights, &[0.0, 1.5, 1.5]);

        let weighted = rttright(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 1],
            Some(vec![2.0, 1.0, 3.0]),
            true,
            true,
        )
        .unwrap();
        let weighted_raw = rttright(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 1],
            Some(vec![2.0, 1.0, 3.0]),
            true,
            false,
        )
        .unwrap();

        assert_close_slice(&weighted.weights, &[0.0, 0.25, 0.75]);
        assert_close_slice(&weighted_raw.weights, &[0.0, 1.5, 4.5]);
    }

    #[test]
    fn test_rttright_tied_blocks_are_atomic() {
        let result =
            rttright(vec![1.0, 2.0, 2.0, 3.0], vec![0, 1, 1, 1], None, true, true).unwrap();

        assert_close_slice(&result.weights, &[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn test_rttright_unweighted_matches_unit_weights() {
        let time = vec![1.0, 2.0, 2.0, 3.0];
        let status = vec![0, 1, 1, 1];
        let weights = vec![1.0; time.len()];

        let unweighted = rttright(time.clone(), status.clone(), None, true, true).unwrap();
        let weighted = rttright(time, status, Some(weights), true, true).unwrap();

        assert_eq!(unweighted.time, weighted.time);
        assert_eq!(unweighted.status, weighted.status);
        assert_eq!(unweighted.order, weighted.order);
        assert_close_slice(&unweighted.weights, &weighted.weights);
    }

    #[test]
    fn test_rttright_timefix_controls_near_tie_grouping() {
        let fixed = rttright(
            vec![1.0, 1.0 + 5e-10, 2.0],
            vec![0, 1, 1],
            None,
            true,
            false,
        )
        .unwrap();
        let exact = rttright(
            vec![1.0, 1.0 + 5e-10, 2.0],
            vec![0, 1, 1],
            None,
            false,
            false,
        )
        .unwrap();

        assert_close_slice(&fixed.weights, &[0.0, 1.0, 2.0]);
        assert_close_slice(&exact.weights, &[0.0, 1.5, 1.5]);
    }

    #[test]
    fn test_rttright_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = rttright(time, status, None, true, true).unwrap();
        assert!(result.weights.is_empty());
    }

    #[test]
    fn test_rttright_stratified_aligns_weights_to_original_rows() {
        let result = rttright_stratified(
            vec![3.0, 1.0, 2.0, 1.5],
            vec![1, 0, 1, 1],
            vec![0, 0, 1, 1],
            None,
            true,
            true,
        )
        .unwrap();

        assert_close_slice(&result.weights, &[1.0, 0.0, 0.5, 0.5]);

        let mut order = result.order.clone();
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(result.order, vec![1, 0, 3, 2]);
    }

    #[test]
    fn test_rttright_stratified_unweighted_matches_unit_weights() {
        let time = vec![3.0, 1.0, 2.0, 1.5];
        let status = vec![1, 0, 1, 1];
        let strata = vec![0, 0, 1, 1];
        let weights = vec![1.0; time.len()];

        let unweighted = rttright_stratified(
            time.clone(),
            status.clone(),
            strata.clone(),
            None,
            true,
            true,
        )
        .unwrap();
        let weighted =
            rttright_stratified(time, status, strata, Some(weights), true, true).unwrap();

        assert_eq!(unweighted.time, weighted.time);
        assert_eq!(unweighted.status, weighted.status);
        assert_eq!(unweighted.order, weighted.order);
        assert_close_slice(&unweighted.weights, &weighted.weights);
    }

    #[test]
    fn test_rttright_stratified_validates_weights_length() {
        initialize_python();

        let err = rttright_stratified(
            vec![1.0, 2.0],
            vec![1, 0],
            vec![0, 0],
            Some(vec![1.0]),
            true,
            true,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("weights must have same length as time")
        );
    }

    #[test]
    fn test_rttright_time_matrix_matches_r_layout() {
        let result = rttright_time_matrix(
            vec![3.0, 1.0, 2.0],
            vec![1, 0, 1],
            vec![1.0, 2.0, 3.0],
            None,
            None,
            true,
            true,
        )
        .unwrap();

        assert_close_slice(&result[0], &[1.0 / 3.0, 0.5, 0.5]);
        assert_close_slice(&result[1], &[1.0 / 3.0, 0.0, 0.0]);
        assert_close_slice(&result[2], &[1.0 / 3.0, 0.5, 0.5]);
    }

    #[test]
    fn test_rttright_time_matrix_handles_strata_weights_and_ties() {
        let result = rttright_time_matrix(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            Some(vec![2.0, 2.0, 1.0, 3.0]),
            Some(vec![0, 0, 1, 1]),
            true,
            true,
        )
        .unwrap();

        assert_close_slice(&result[0], &[0.5, 0.0, 0.0, 0.0]);
        assert_close_slice(&result[1], &[0.5, 1.0, 1.0, 1.0]);
        assert_close_slice(&result[2], &[0.25, 0.25, 0.25, 0.0]);
        assert_close_slice(&result[3], &[0.75, 0.75, 0.75, 1.0]);

        let fixed = rttright_time_matrix(
            vec![1.0, 1.0 + 5e-10, 2.0],
            vec![0, 1, 1],
            vec![1.0, 2.0],
            None,
            None,
            true,
            false,
        )
        .unwrap();
        let exact = rttright_time_matrix(
            vec![1.0, 1.0 + 5e-10, 2.0],
            vec![0, 1, 1],
            vec![1.0, 2.0],
            None,
            None,
            false,
            false,
        )
        .unwrap();

        assert_close_slice(&fixed[1], &[1.0, 1.0]);
        assert_close_slice(&exact[1], &[1.0, 1.5]);

        let tiny = rttright_time_matrix(
            vec![1.0, 2.0],
            vec![1, 1],
            vec![1.0],
            Some(vec![1e-12, 1e-12]),
            None,
            true,
            true,
        )
        .unwrap();
        assert_close_slice(&tiny[0], &[0.5]);
        assert_close_slice(&tiny[1], &[0.5]);
    }

    #[test]
    fn test_rttright_counting_matrix_matches_r_layout() {
        let vector = rttright_counting_matrix(
            vec![0.0, 0.0, 0.0],
            vec![3.0, 2.0, 4.0],
            vec![1, 0, 1],
            vec![1.0 / 3.0; 3],
            vec![true; 3],
            vec![0; 3],
            0.25,
            None,
        )
        .unwrap();
        assert_nested_close(&vector, &[vec![0.5], vec![0.0], vec![0.5]]);

        let matrix = rttright_counting_matrix(
            vec![0.0, 0.0, 0.0],
            vec![3.0, 2.0, 4.0],
            vec![1, 0, 1],
            vec![1.0 / 3.0; 3],
            vec![true; 3],
            vec![0; 3],
            0.25,
            Some(vec![1.0, 2.0, 3.0, 4.0]),
        )
        .unwrap();
        assert_nested_close(
            &matrix,
            &[
                vec![1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5],
                vec![1.0 / 3.0; 4],
                vec![1.0 / 3.0, 1.0 / 3.0, 0.5, 0.5],
            ],
        );
    }

    #[test]
    fn test_rttright_counting_matrix_preserves_strata() {
        let result = rttright_counting_matrix(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![3.0, 2.0, 4.0, 2.0],
            vec![1, 0, 1, 1],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![true; 4],
            vec![0, 0, 1, 1],
            0.25,
            None,
        )
        .unwrap();

        assert_nested_close(&result, &[vec![1.0], vec![0.0], vec![0.5], vec![0.5]]);
    }

    #[test]
    fn test_rttright_counting_matrix_validates_inputs() {
        initialize_python();
        let length_error = rttright_counting_matrix(
            vec![0.0],
            vec![1.0, 2.0],
            vec![1, 0],
            vec![1.0, 1.0],
            vec![true, true],
            vec![0, 0],
            0.5,
            None,
        )
        .unwrap_err();
        assert!(
            length_error
                .to_string()
                .contains("start must have same length as stop")
        );

        let interval_error = rttright_counting_matrix(
            vec![1.0],
            vec![1.0],
            vec![1],
            vec![1.0],
            vec![true],
            vec![0],
            0.5,
            None,
        )
        .unwrap_err();
        assert!(
            interval_error
                .to_string()
                .contains("start must be less than stop")
        );

        let delta_error = rttright_counting_matrix(
            vec![0.0],
            vec![1.0],
            vec![1],
            vec![1.0],
            vec![true],
            vec![0],
            0.0,
            None,
        )
        .unwrap_err();
        assert!(delta_error.to_string().contains("delta must be finite"));
    }

    #[test]
    fn test_counting_censor_survival_sweep_matches_naive_risk_sets() {
        let mut seed = 20_260_820_u64;
        for _ in 0..500 {
            let n = (next_random(&mut seed) % 40 + 1) as usize;
            let mut start = Vec::with_capacity(n);
            let mut km_stop = Vec::with_capacity(n);
            let mut censor = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);
            for _ in 0..n {
                let left = (next_random(&mut seed) % 20) as f64 / 3.0;
                let right = left + (next_random(&mut seed) % 20 + 1) as f64 / 4.0;
                start.push(left);
                km_stop.push(right);
                censor.push(next_random(&mut seed).is_multiple_of(4));
                weights.push((next_random(&mut seed) % 17) as f64 / 11.0);
            }
            let indices = (0..n).collect::<Vec<_>>();
            let expected = naive_counting_censor_survival(&start, &km_stop, &censor, &weights);
            let actual = counting_censor_survival(&start, &km_stop, &censor, &weights, &indices);

            assert_eq!(actual.0, expected.0);
            assert_close_slice(&actual.1, &expected.1);
        }
    }

    #[test]
    fn test_rttright_matrix_matches_right_censored_reference() {
        let result = rttright_matrix(
            vec![3.0, 1.0, 2.0],
            vec![1, 0, 1],
            vec![1.0, 2.0, 3.0],
            None,
            None,
            true,
            true,
        )
        .unwrap();

        assert_nested_close(
            &result,
            &[
                vec![1.0 / 3.0, 0.5, 0.5],
                vec![1.0 / 3.0, 0.0, 0.0],
                vec![1.0 / 3.0, 0.5, 0.5],
            ],
        );
    }

    #[test]
    fn test_rttright_matrix_preserves_strata_and_weights() {
        let result = rttright_matrix(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            Some(vec![0, 0, 1, 1]),
            Some(vec![2.0, 1.0, 3.0, 1.0]),
            true,
            true,
        )
        .unwrap();

        assert_nested_close(
            &result,
            &[
                vec![2.0 / 3.0, 0.0, 0.0, 0.0],
                vec![1.0 / 3.0, 1.0, 1.0, 1.0],
                vec![0.75, 0.75, 0.75, 0.0],
                vec![0.25, 0.25, 0.25, 1.0],
            ],
        );
    }

    #[test]
    fn test_rttright_matrix_timefix_controls_near_ties() {
        let time = vec![1.0, 1.0 + 5e-10, 2.0];
        let status = vec![0, 1, 1];
        let times = time.clone();
        let fixed = rttright_matrix(
            time.clone(),
            status.clone(),
            times.clone(),
            None,
            None,
            true,
            false,
        )
        .unwrap();
        let exact = rttright_matrix(time, status, times, None, None, false, false).unwrap();

        assert_nested_close(
            &fixed,
            &[
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 2.0, 2.0],
            ],
        );
        assert_nested_close(
            &exact,
            &[
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.5, 1.5],
                vec![1.0, 1.5, 1.5],
            ],
        );
    }

    #[test]
    fn test_rttright_matrix_rejects_malformed_inputs() {
        initialize_python();

        let bad_strata = rttright_matrix(
            vec![1.0],
            vec![1],
            vec![1.0],
            Some(vec![]),
            None,
            true,
            true,
        )
        .unwrap_err();
        assert!(
            bad_strata
                .to_string()
                .contains("strata must have same length as time")
        );

        let bad_times = rttright_matrix(
            vec![1.0],
            vec![1],
            vec![f64::INFINITY],
            None,
            None,
            true,
            true,
        )
        .unwrap_err();
        assert!(bad_times.to_string().contains("times contains non-finite"));
    }

    #[test]
    fn test_rttright_rejects_malformed_inputs() {
        initialize_python();

        let err = rttright(vec![f64::NAN], vec![1], None, true, true).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = rttright(vec![1.0], vec![2], None, true, true).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        let err = rttright(vec![1.0], vec![1], Some(vec![-1.0]), true, true).unwrap_err();
        assert!(err.to_string().contains("weights contains negative value"));

        let err = rttright_stratified(
            vec![1.0],
            vec![1],
            vec![0],
            Some(vec![f64::INFINITY]),
            true,
            true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("weights contains non-finite"));

        let err = rttright(vec![1.0], vec![1], Some(vec![0.0]), true, true).unwrap_err();
        assert!(
            err.to_string()
                .contains("weights must have positive sum when renorm is true")
        );
    }

    #[test]
    fn test_rttright_is_invariant_to_input_order_with_unique_times() {
        let base_time = [1.0, 2.0, 3.0, 4.0];
        let base_status = [1, 0, 1, 1];
        let base_weights = [1.0, 2.0, 1.5, 0.5];

        let baseline = rttright(
            base_time.to_vec(),
            base_status.to_vec(),
            Some(base_weights.to_vec()),
            true,
            true,
        )
        .unwrap();

        for permutation in index_permutations(base_time.len()) {
            let time: Vec<f64> = permutation.iter().map(|&i| base_time[i]).collect();
            let status: Vec<i32> = permutation.iter().map(|&i| base_status[i]).collect();
            let weights: Vec<f64> = permutation.iter().map(|&i| base_weights[i]).collect();

            let result = rttright(time, status, Some(weights), true, true).unwrap();

            assert_eq!(result.time, baseline.time);
            assert_eq!(result.status, baseline.status);
            assert_eq!(result.weights, baseline.weights);

            let mut order = result.order.clone();
            order.sort_unstable();
            assert_eq!(order, vec![0, 1, 2, 3]);
        }
    }
}
