use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

use crate::constants::{DIVISION_FLOOR, same_time};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_no_nan, validate_non_negative,
};

/// Result of redistribute-to-the-right weight calculation
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RttrightResult {
    /// Redistributed weights for each observation
    #[pyo3(get)]
    pub weights: Vec<f64>,
    /// Original time values (sorted)
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Original status values (sorted)
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// Sort order indices
    #[pyo3(get)]
    pub order: Vec<usize>,
}

/// Compute redistribute-to-the-right weights for censored data.
///
/// This implements the IPCW (Inverse Probability of Censoring Weighting)
/// approach where the weight of each censored observation is redistributed
/// to observations with longer survival times.
///
/// The Kaplan-Meier estimator can be derived from this redistribution.
///
/// # Arguments
/// * `time` - Survival/censoring times
/// * `status` - Event indicator (1=event, 0=censored)
/// * `weights` - Optional initial weights (default: 1.0 for all)
/// * `timefix` - Coalesce nearly-equal times like R's `aeqSurv`
/// * `renorm` - Normalize weights to sum to one before redistribution
///
/// # Returns
/// * `RttrightResult` containing redistributed weights
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

    let init_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    if init_weights.len() != n {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    validate_rttright_inputs(&time, &status, &init_weights)?;

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
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| init_weights[i]).collect();
    let sorted_weights = normalize_case_weights(&sorted_weights, renorm)?;

    let km_weights = compute_km_weights(&sorted_time, &sorted_status, &sorted_weights, timefix);

    Ok(RttrightResult {
        weights: km_weights,
        time: sorted_time,
        status: sorted_status,
        order: indices,
    })
}

fn validate_rttright_inputs(time: &[f64], status: &[i32], weights: &[f64]) -> PyResult<()> {
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_no_nan(weights, "weights")?;
    validate_finite(weights, "weights")?;
    validate_non_negative(weights, "weights")?;

    validate_binary_i32(status, "status")?;

    Ok(())
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

/// Compute IPCW weights using the Kaplan-Meier censoring distribution.
///
/// Within a time block, observed events receive the censoring survival just
/// before the block; censoring at that same time updates the censoring curve
/// only after those same-time events have left the risk set. This matches the
/// shifted-censoring construction used by R's `rttright`.
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

/// Compute IPCW weights with stratification
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

    let init_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    if init_weights.len() != n {
        return Err(PyValueError::new_err(
            "weights must have same length as time",
        ));
    }
    validate_rttright_inputs(&time, &status, &init_weights)?;

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
        let strata_weights: Vec<f64> = indices.iter().map(|&i| init_weights[i]).collect();

        let result = rttright_impl(
            strata_time,
            strata_status,
            Some(strata_weights),
            timefix,
            renorm,
        )?;

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
    use crate::tests::common::initialize_python;
    use itertools::Itertools;

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

        for permutation in (0..base_time.len()).permutations(base_time.len()) {
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
