use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

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
