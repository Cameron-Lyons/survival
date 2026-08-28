use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RttrightCountingResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub matrix: Vec<Vec<f64>>,
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

const COUNTING_TIME_EPSILON: f64 = 1e-9;

fn timefix_counting_vectors(mut start: Vec<f64>, mut stop: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    let mut points = Vec::with_capacity(start.len() + stop.len());
    points.extend(
        start
            .iter()
            .copied()
            .enumerate()
            .map(|(row, value)| (value, 0usize, row)),
    );
    points.extend(
        stop.iter()
            .copied()
            .enumerate()
            .map(|(row, value)| (value, 1usize, row)),
    );
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
        while scan < points.len() && points[scan].0 - base < COUNTING_TIME_EPSILON {
            let (_, vector, row) = points[scan];
            if vector == 0 {
                start[row] = base;
            } else {
                stop[row] = base;
            }
            scan += 1;
        }
        cursor = scan;
    }
    (start, stop)
}

fn validate_counting_histories(id: &[i64], start: &[f64], stop: &[f64]) -> PyResult<Vec<bool>> {
    let mut subject_rows: HashMap<i64, Vec<usize>> = HashMap::new();
    for (row, &subject) in id.iter().enumerate() {
        subject_rows.entry(subject).or_default().push(row);
    }
    if subject_rows.is_empty() {
        return Err(PyNotImplementedError::new_err(
            "function not defined for delayed entry or multistate data",
        ));
    }

    let mut common_start = None;
    let mut last = vec![false; id.len()];
    for rows in subject_rows.values_mut() {
        let subject_start = rows
            .iter()
            .map(|&row| start[row])
            .min_by(f64::total_cmp)
            .expect("subject rows are non-empty");
        if common_start.is_none() {
            common_start = Some(subject_start);
        } else if common_start != Some(subject_start) {
            return Err(PyNotImplementedError::new_err(
                "function not defined for delayed entry or multistate data",
            ));
        }

        let last_row = rows
            .iter()
            .copied()
            .max_by(|&left, &right| {
                stop[left]
                    .total_cmp(&stop[right])
                    .then_with(|| left.cmp(&right))
            })
            .expect("subject rows are non-empty");
        last[last_row] = true;

        rows.sort_by(|&left, &right| {
            stop[left]
                .total_cmp(&stop[right])
                .then_with(|| start[left].total_cmp(&start[right]))
        });
        let mut previous_stop = None;
        for &row in rows.iter() {
            if stop[row] < start[row]
                || previous_stop.is_some_and(|previous| {
                    start[row] < previous - 1e-10 || start[row] > previous + 1e-10
                })
            {
                return Err(PyValueError::new_err(
                    "one or more flags are >0 in survcheck",
                ));
            }
            previous_stop = Some(stop[row]);
        }
    }
    Ok(last)
}

fn normalize_counting_weights(
    id: &[i64],
    strata_indices: &BTreeMap<i32, Vec<usize>>,
    weights: &[f64],
    renorm: bool,
) -> PyResult<Vec<f64>> {
    let mut subject_weights = HashMap::new();
    for (&subject, &weight) in id.iter().zip(weights) {
        if subject_weights
            .insert(subject, weight)
            .is_some_and(|previous| previous != weight)
        {
            return Err(PyValueError::new_err(
                "there are subjects with multiple weights",
            ));
        }
    }
    if !renorm {
        return Ok(weights.to_vec());
    }

    let mut normalized = weights.to_vec();
    for indices in strata_indices.values() {
        let mut seen = HashSet::new();
        let denominator = indices
            .iter()
            .filter_map(|&row| seen.insert(id[row]).then_some(weights[row]))
            .sum::<f64>();
        if denominator <= 0.0 {
            return Err(PyValueError::new_err(
                "weights must have positive sum when renorm is true",
            ));
        }
        for &row in indices {
            normalized[row] /= denominator;
        }
    }
    Ok(normalized)
}

fn counting_delta(start: &[f64], stop: &[f64], query_times: Option<&[f64]>) -> PyResult<f64> {
    let mut values =
        Vec::with_capacity(start.len() + stop.len() + query_times.map_or(0, <[f64]>::len));
    values.extend_from_slice(start);
    values.extend_from_slice(stop);
    if let Some(times) = query_times {
        values.extend_from_slice(times);
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| *left == *right);
    values
        .windows(2)
        .filter_map(|pair| (pair[1] > pair[0]).then_some(pair[1] - pair[0]))
        .min_by(f64::total_cmp)
        .map(|difference| difference / 2.0)
        .ok_or_else(|| {
            PyNotImplementedError::new_err(
                "function not defined for delayed entry or multistate data",
            )
        })
}

fn counting_km(
    start: &[f64],
    stop: &[f64],
    weights: &[f64],
    censor: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let mut start_order = (0..start.len()).collect::<Vec<_>>();
    start_order.sort_by(|&left, &right| {
        start[left]
            .total_cmp(&start[right])
            .then_with(|| left.cmp(&right))
    });
    let mut stop_order = (0..stop.len()).collect::<Vec<_>>();
    stop_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });
    let mut censor_order = (0..stop.len())
        .filter(|&row| censor[row])
        .collect::<Vec<_>>();
    censor_order.sort_by(|&left, &right| {
        stop[left]
            .total_cmp(&stop[right])
            .then_with(|| left.cmp(&right))
    });

    let mut survival_times = Vec::with_capacity(censor_order.len());
    let mut survival_values = Vec::with_capacity(censor_order.len());
    let mut current = 1.0;
    let mut risk = 0.0;
    let mut start_cursor = 0;
    let mut stop_cursor = 0;
    let mut censor_cursor = 0;
    while censor_cursor < censor_order.len() {
        let event_time = stop[censor_order[censor_cursor]];
        while start_cursor < start_order.len() && start[start_order[start_cursor]] < event_time {
            risk += weights[start_order[start_cursor]];
            start_cursor += 1;
        }
        while stop_cursor < stop_order.len() && stop[stop_order[stop_cursor]] < event_time {
            risk -= weights[stop_order[stop_cursor]];
            stop_cursor += 1;
        }

        let mut event_weight = 0.0;
        while censor_cursor < censor_order.len() && stop[censor_order[censor_cursor]] == event_time
        {
            event_weight += weights[censor_order[censor_cursor]];
            censor_cursor += 1;
        }
        if risk > 0.0 {
            current *= 1.0 - event_weight / risk;
        }
        survival_times.push(event_time);
        survival_values.push(current);
    }
    (survival_times, survival_values)
}

#[inline]
fn counting_survival_at(times: &[f64], values: &[f64], time: f64) -> f64 {
    let index = times.partition_point(|event_time| *event_time < time);
    if index == 0 { 1.0 } else { values[index - 1] }
}

enum CountingGroupResult {
    Weights(Vec<f64>),
    Matrix(Vec<Vec<f64>>),
}

#[allow(clippy::too_many_arguments)]
fn compute_counting_group(
    start: &[f64],
    stop: &[f64],
    status: &[i32],
    weights: &[f64],
    last: &[bool],
    indices: &[usize],
    query_times: Option<&[f64]>,
    delta: f64,
) -> CountingGroupResult {
    let group_start = indices.iter().map(|&row| start[row]).collect::<Vec<_>>();
    let mut group_stop = indices.iter().map(|&row| stop[row]).collect::<Vec<_>>();
    let group_weights = indices.iter().map(|&row| weights[row]).collect::<Vec<_>>();
    let censor = indices
        .iter()
        .map(|&row| last[row] && status[row] == 0)
        .collect::<Vec<_>>();
    for (row, &is_censor) in censor.iter().enumerate() {
        if is_censor {
            group_stop[row] += delta;
        }
    }
    let (survival_times, survival_values) =
        counting_km(&group_start, &group_stop, &group_weights, &censor);

    let Some(query_times) = query_times else {
        let result = indices
            .iter()
            .enumerate()
            .map(|(local_row, &row)| {
                if last[row] && status[row] > 0 {
                    let survival =
                        counting_survival_at(&survival_times, &survival_values, stop[row]);
                    rttright_divide(group_weights[local_row], survival)
                } else {
                    0.0
                }
            })
            .collect();
        return CountingGroupResult::Weights(result);
    };

    let query_survival = query_times
        .iter()
        .map(|&time| counting_survival_at(&survival_times, &survival_values, time))
        .collect::<Vec<_>>();
    let make_row = |local_row: usize| {
        let row = indices[local_row];
        let row_weight = group_weights[local_row];
        let mut values = vec![0.0; query_times.len()];
        for (column, (&query_time, &survival)) in
            query_times.iter().zip(&query_survival).enumerate()
        {
            if start[row] < query_time && query_time <= stop[row] {
                values[column] = rttright_divide(row_weight, survival);
            }
        }
        if last[row] && stop[row] > 0.0 {
            let stop_survival = counting_survival_at(&survival_times, &survival_values, stop[row]);
            for (column, &survival) in query_survival.iter().enumerate() {
                values[column] = rttright_divide(row_weight, stop_survival.max(survival));
            }
        }
        values
    };
    let work = indices.len().saturating_mul(query_times.len());
    let matrix = if work >= PARALLEL_TIME_MATRIX_WORK {
        (0..indices.len()).into_par_iter().map(make_row).collect()
    } else {
        (0..indices.len()).map(make_row).collect()
    };
    CountingGroupResult::Matrix(matrix)
}

/// Redistribute counting-process censoring mass for complete subject histories.
#[allow(clippy::too_many_arguments)]
pub fn rttright_counting(
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    id: Vec<i64>,
    query_times: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<RttrightCountingResult> {
    let n = start.len();
    if stop.len() != n || status.len() != n || id.len() != n {
        return Err(PyValueError::new_err(
            "start, stop, status, and id must have the same length",
        ));
    }
    if weights.as_ref().is_some_and(|values| values.len() != n) {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    if strata.as_ref().is_some_and(|values| values.len() != n) {
        return Err(PyValueError::new_err(
            "rttright strata must have the same length as the Surv response",
        ));
    }
    validate_finite(&start, "start")?;
    validate_finite(&stop, "stop")?;
    validate_binary_i32(&status, "status")?;
    if let Some(values) = weights.as_deref() {
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
    }
    if let Some(times) = query_times.as_deref() {
        validate_finite(times, "times")?;
    }

    let (start, stop) = if timefix {
        timefix_counting_vectors(start, stop)
    } else {
        (start, stop)
    };
    let last = validate_counting_histories(&id, &start, &stop)?;
    if last
        .iter()
        .zip(&status)
        .filter(|(is_last, event)| **is_last && **event > 0)
        .count()
        <= 1
    {
        return Err(PyNotImplementedError::new_err(
            "function not defined for delayed entry or multistate data",
        ));
    }

    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let mut strata_indices: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (row, stratum) in strata.into_iter().enumerate() {
        strata_indices.entry(stratum).or_default().push(row);
    }
    let case_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let case_weights = normalize_counting_weights(&id, &strata_indices, &case_weights, renorm)?;
    let delta = counting_delta(&start, &stop, query_times.as_deref())?;

    let mut result_weights = if query_times.is_none() {
        vec![0.0; n]
    } else {
        vec![]
    };
    let mut result_matrix = query_times
        .as_ref()
        .map_or_else(Vec::new, |times| vec![vec![0.0; times.len()]; n]);
    for indices in strata_indices.values() {
        match compute_counting_group(
            &start,
            &stop,
            &status,
            &case_weights,
            &last,
            indices,
            query_times.as_deref(),
            delta,
        ) {
            CountingGroupResult::Weights(weights) => {
                for (&row, weight) in indices.iter().zip(weights) {
                    result_weights[row] = weight;
                }
            }
            CountingGroupResult::Matrix(matrix) => {
                for (&row, values) in indices.iter().zip(matrix) {
                    result_matrix[row] = values;
                }
            }
        }
    }
    Ok(RttrightCountingResult {
        weights: result_weights,
        matrix: result_matrix,
    })
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
    fn test_rttright_counting_matches_vector_and_matrix_references() {
        let start = vec![0.0, 1.0, 0.0, 2.0];
        let stop = vec![1.0, 3.0, 2.0, 4.0];
        let status = vec![0, 1, 0, 1];
        let id = vec![1, 1, 2, 2];
        let vector = rttright_counting(
            start.clone(),
            stop.clone(),
            status.clone(),
            id.clone(),
            None,
            None,
            None,
            true,
            true,
        )
        .unwrap();
        assert_close_slice(&vector.weights, &[0.0, 0.5, 0.0, 0.5]);
        assert!(vector.matrix.is_empty());

        let interleaved = rttright_counting(
            vec![2.0, 0.0, 0.0, 1.0],
            vec![4.0, 1.0, 2.0, 3.0],
            vec![1, 0, 0, 1],
            vec![2, 1, 2, 1],
            None,
            None,
            None,
            true,
            true,
        )
        .unwrap();
        assert_close_slice(&interleaved.weights, &[0.5, 0.0, 0.0, 0.5]);

        let matrix = rttright_counting(
            start,
            stop,
            status,
            id,
            Some(vec![1.0, 2.0, 3.0, 4.0]),
            None,
            None,
            true,
            true,
        )
        .unwrap();
        assert!(matrix.weights.is_empty());
        assert_nested_close(
            &matrix.matrix,
            &[
                vec![0.5, 0.0, 0.0, 0.0],
                vec![0.5, 0.5, 0.5, 0.5],
                vec![0.5, 0.5, 0.0, 0.0],
                vec![0.5, 0.5, 0.5, 0.5],
            ],
        );
    }

    #[test]
    fn test_rttright_counting_preserves_weighted_strata() {
        let result = rttright_counting(
            vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5],
            vec![1.0, 3.0, 2.0, 4.0, 1.5, 2.5],
            vec![0, 1, 0, 1, 0, 0],
            vec![1, 1, 2, 2, 3, 3],
            Some(vec![1.0, 2.0, 3.0, 4.0]),
            Some(vec![2.0, 2.0, 1.0, 1.0, 3.0, 3.0]),
            Some(vec![0, 0, 1, 1, 0, 0]),
            true,
            true,
        )
        .unwrap();
        assert_nested_close(
            &result.matrix,
            &[
                vec![0.4, 0.0, 0.0, 0.0],
                vec![0.4, 0.4, 1.0, 1.0],
                vec![1.0, 1.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![0.6, 0.0, 0.0, 0.0],
                vec![0.6, 0.6, 0.6, 0.6],
            ],
        );
    }

    #[test]
    fn test_rttright_counting_validates_histories_and_subject_weights() {
        initialize_python();

        let delayed = rttright_counting(
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            vec![1, 1],
            vec![1, 2],
            None,
            None,
            None,
            true,
            true,
        )
        .unwrap_err();
        assert!(delayed.to_string().contains("delayed entry"));

        let multiple_weights = rttright_counting(
            vec![0.0, 1.0, 0.0],
            vec![1.0, 2.0, 2.0],
            vec![0, 1, 1],
            vec![1, 1, 2],
            None,
            Some(vec![1.0, 2.0, 1.0]),
            None,
            true,
            true,
        )
        .unwrap_err();
        assert!(multiple_weights.to_string().contains("multiple weights"));

        let fixed = rttright_counting(
            vec![0.0, 1.0 + 5e-10, 0.0],
            vec![1.0, 2.0, 2.0],
            vec![0, 1, 1],
            vec![1, 1, 2],
            None,
            None,
            None,
            true,
            true,
        )
        .unwrap();
        assert_close_slice(&fixed.weights, &[0.0, 0.5, 0.5]);
        let exact = rttright_counting(
            vec![0.0, 1.0 + 5e-10, 0.0],
            vec![1.0, 2.0, 2.0],
            vec![0, 1, 1],
            vec![1, 1, 2],
            None,
            None,
            None,
            false,
            true,
        )
        .unwrap_err();
        assert!(exact.to_string().contains("survcheck"));
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
