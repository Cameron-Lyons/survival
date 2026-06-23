use crate::constants::{TIME_EPSILON, same_time};
use crate::internal::logrank::logrank_statistic_from_flat_covariance;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvDiffResult {
    #[pyo3(get)]
    pub observed: Vec<f64>,
    #[pyo3(get)]
    pub expected: Vec<f64>,
    #[pyo3(get)]
    pub variance: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub chi_squared: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: usize,
}

#[pyfunction]
#[pyo3(signature = (time, status, group, strata=None, rho=None, timefix=false))]
pub fn compute_logrank_components(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
    timefix: bool,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    if n > i32::MAX as usize {
        return Err(value_error("time length exceeds i32 calculation capacity"));
    }
    validate_length(n, status.len(), "status")?;
    validate_length(n, group.len(), "group")?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    validate_binary_i32(&status, "status")?;
    validate_group_codes(&group, n)?;
    let strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    validate_length(n, strata_vec.len(), "strata")?;
    validate_strata_markers(&strata_vec)?;
    let rho_val = rho.unwrap_or(0.0);
    if !rho_val.is_finite() {
        return Err(value_error("rho must be finite"));
    }
    let max_group = group.iter().max().copied().unwrap_or(0);
    let ngroup = if max_group > 0 { max_group as usize } else { 1 };
    let nstrat = strata_ranges(&strata_vec).len().max(1);
    let prepared = prepare_right_logrank_inputs(&time, &status, &group, &strata_vec, timefix);
    let mut obs = vec![0.0; ngroup * nstrat];
    let mut exp = vec![0.0; ngroup * nstrat];
    let mut var = vec![0.0; ngroup * ngroup];
    let mut risk = vec![0.0; ngroup];
    let mut kaplan = vec![0.0; n];
    let params = SurvDiffParams {
        nn: n as i32,
        nngroup: ngroup as i32,
        _nstrat: nstrat as i32,
        rho: rho_val,
    };
    let input = SurvDiffInput {
        time: &prepared.time,
        status: &prepared.status,
        group: &prepared.group,
        strata: &prepared.strata,
    };
    let mut output = SurvDiffOutput {
        obs: &mut obs,
        exp: &mut exp,
        var: &mut var,
        risk: &mut risk,
        kaplan: &mut kaplan,
    };
    compute_survdiff(params, input, &mut output);
    let mut observed_by_group = vec![0.0; ngroup];
    let mut expected_by_group = vec![0.0; ngroup];
    for stratum_idx in 0..nstrat {
        let offset = stratum_idx * ngroup;
        for group_idx in 0..ngroup {
            observed_by_group[group_idx] += obs[offset + group_idx];
            expected_by_group[group_idx] += exp[offset + group_idx];
        }
    }

    Ok(survdiff_result_from_flat_components(
        observed_by_group,
        expected_by_group,
        var,
        ngroup,
    ))
}

struct PreparedLogrankInput {
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Vec<i32>,
}

fn prepare_right_logrank_inputs(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    strata: &[i32],
    timefix: bool,
) -> PreparedLogrankInput {
    let mut prepared = PreparedLogrankInput {
        time: Vec::with_capacity(time.len()),
        status: Vec::with_capacity(status.len()),
        group: Vec::with_capacity(group.len()),
        strata: Vec::with_capacity(strata.len()),
    };

    for (start, end) in strata_ranges(strata) {
        let mut indices: Vec<usize> = (start..end).collect();
        indices.sort_by(|&left, &right| {
            time[left]
                .total_cmp(&time[right])
                .then_with(|| left.cmp(&right))
        });

        let range_start = prepared.time.len();
        for idx in indices {
            prepared.time.push(time[idx]);
            prepared.status.push(status[idx]);
            prepared.group.push(group[idx]);
            prepared.strata.push(0);
        }
        if let Some(marker) = prepared.strata.last_mut() {
            *marker = 1;
        }
        if timefix {
            coalesce_near_times(&mut prepared.time[range_start..]);
        }
    }

    prepared
}

fn coalesce_near_times(times: &mut [f64]) {
    let mut cursor = 0;
    while cursor < times.len() {
        let base = times[cursor];
        let mut scan = cursor + 1;
        while scan < times.len() && times[scan] - base < TIME_EPSILON {
            times[scan] = base;
            scan += 1;
        }
        cursor = scan;
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, group, entry_times, strata=None, rho=None, timefix=true))]
pub fn compute_counting_logrank_components(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    entry_times: Vec<f64>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
    timefix: bool,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    if n > i32::MAX as usize {
        return Err(value_error("time length exceeds i32 calculation capacity"));
    }
    validate_length(n, status.len(), "status")?;
    validate_length(n, group.len(), "group")?;
    validate_length(n, entry_times.len(), "entry_times")?;
    validate_no_nan(&time, "time")?;
    validate_finite(&time, "time")?;
    validate_non_negative(&time, "time")?;
    validate_no_nan(&entry_times, "entry_times")?;
    validate_finite(&entry_times, "entry_times")?;
    validate_non_negative(&entry_times, "entry_times")?;
    validate_binary_i32(&status, "status")?;
    validate_group_codes(&group, n)?;
    validate_counting_intervals(&entry_times, &time, timefix)?;
    let strata_vec = strata.unwrap_or_else(|| vec![0; n]);
    validate_length(n, strata_vec.len(), "strata")?;
    validate_strata_markers(&strata_vec)?;
    let rho_val = rho.unwrap_or(0.0);
    if !rho_val.is_finite() {
        return Err(value_error("rho must be finite"));
    }

    let max_group = group.iter().max().copied().unwrap_or(0);
    let ngroup = if max_group > 0 { max_group as usize } else { 1 };
    let mut observed = vec![0.0; ngroup];
    let mut expected = vec![0.0; ngroup];
    let mut variance = vec![0.0; ngroup * ngroup];

    for (start, end) in strata_ranges(&strata_vec) {
        accumulate_counting_logrank_stratum(
            start,
            end,
            CountingLogrankInput {
                time: &time,
                status: &status,
                group: &group,
                entry_times: &entry_times,
                rho: rho_val,
                timefix,
            },
            &mut observed,
            &mut expected,
            &mut variance,
        );
    }

    Ok(survdiff_result_from_flat_components(
        observed, expected, variance, ngroup,
    ))
}

fn validate_group_codes(values: &[i32], n: usize) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 1 {
            return Err(value_error(format!(
                "group must be >= 1; got {value} at index {idx}"
            )));
        }
        if value as usize > n {
            return Err(value_error(format!(
                "group values must be between 1 and the number of observations ({n}); got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_strata_markers(values: &[i32]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "strata must be >= 0; got {value} at index {idx}"
            )));
        }
        if value > 1 {
            return Err(value_error(format!(
                "strata values must be 0 or 1; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_strata_codes(values: &[i32]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "strata must be >= 0; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn strata_code_order(strata: &[i32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..strata.len()).collect();
    order.sort_by(|&left, &right| {
        strata[left]
            .cmp(&strata[right])
            .then_with(|| left.cmp(&right))
    });
    order
}

fn marker_for_order(strata: &[i32], order: &[usize]) -> Vec<i32> {
    let mut markers = vec![0; order.len()];
    for (position, &idx) in order.iter().enumerate() {
        if position + 1 == order.len() || strata[order[position + 1]] != strata[idx] {
            markers[position] = 1;
        }
    }
    markers
}

#[pyfunction]
#[pyo3(signature = (time, status, group, strata, rho=None, timefix=false))]
pub fn stratified_logrank_components(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Vec<i32>,
    rho: Option<f64>,
    timefix: bool,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    validate_length(n, strata.len(), "strata")?;
    validate_strata_codes(&strata)?;

    let order = strata_code_order(&strata);
    let markers = marker_for_order(&strata, &order);
    let sorted_time = order.iter().map(|&idx| time[idx]).collect();
    let sorted_status = order.iter().map(|&idx| status[idx]).collect();
    let sorted_group = order.iter().map(|&idx| group[idx]).collect();

    compute_logrank_components(
        sorted_time,
        sorted_status,
        sorted_group,
        Some(markers),
        rho,
        timefix,
    )
}

#[pyfunction]
#[pyo3(signature = (time, status, group, entry_times, strata, rho=None, timefix=true))]
pub fn stratified_counting_logrank_components(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    entry_times: Vec<f64>,
    strata: Vec<i32>,
    rho: Option<f64>,
    timefix: bool,
) -> PyResult<SurvDiffResult> {
    let n = time.len();
    validate_length(n, entry_times.len(), "entry_times")?;
    validate_length(n, strata.len(), "strata")?;
    validate_strata_codes(&strata)?;

    let order = strata_code_order(&strata);
    let markers = marker_for_order(&strata, &order);
    let sorted_time = order.iter().map(|&idx| time[idx]).collect();
    let sorted_status = order.iter().map(|&idx| status[idx]).collect();
    let sorted_group = order.iter().map(|&idx| group[idx]).collect();
    let sorted_entry_times = order.iter().map(|&idx| entry_times[idx]).collect();

    compute_counting_logrank_components(
        sorted_time,
        sorted_status,
        sorted_group,
        sorted_entry_times,
        Some(markers),
        rho,
        timefix,
    )
}

fn validate_counting_intervals(entry_times: &[f64], time: &[f64], timefix: bool) -> PyResult<()> {
    for (idx, (&entry_time, &exit_time)) in entry_times.iter().zip(time.iter()).enumerate() {
        let invalid = if timefix {
            entry_time >= exit_time - TIME_EPSILON
        } else {
            entry_time >= exit_time
        };
        if invalid {
            return Err(value_error(format!(
                "entry_times must be less than time for observation {idx}"
            )));
        }
    }
    Ok(())
}

fn survdiff_result_from_flat_components(
    observed: Vec<f64>,
    expected: Vec<f64>,
    variance: Vec<f64>,
    n_groups: usize,
) -> SurvDiffResult {
    let (chi_sq, df) =
        logrank_statistic_from_flat_covariance(&observed, &expected, &variance, n_groups);
    let mut variance_matrix = Vec::new();
    for group_idx in 0..n_groups {
        let start = group_idx * n_groups;
        let end = start + n_groups;
        variance_matrix.push(variance[start..end].to_vec());
    }
    SurvDiffResult {
        observed,
        expected,
        variance: variance_matrix,
        chi_squared: chi_sq,
        degrees_of_freedom: df,
    }
}

fn strata_ranges(strata: &[i32]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (idx, &marker) in strata.iter().enumerate() {
        if marker == 1 {
            ranges.push((start, idx + 1));
            start = idx + 1;
        }
    }
    if start < strata.len() {
        ranges.push((start, strata.len()));
    }
    ranges
}

struct CountingLogrankInput<'a> {
    time: &'a [f64],
    status: &'a [i32],
    group: &'a [i32],
    entry_times: &'a [f64],
    rho: f64,
    timefix: bool,
}

fn accumulate_counting_logrank_stratum(
    start: usize,
    end: usize,
    input: CountingLogrankInput<'_>,
    observed: &mut [f64],
    expected: &mut [f64],
    variance: &mut [f64],
) {
    if start >= end {
        return;
    }

    let n_groups = observed.len();
    let mut exit_indices: Vec<usize> = (start..end).collect();
    exit_indices.sort_by(|&left, &right| input.time[left].total_cmp(&input.time[right]));
    let mut entry_indices: Vec<usize> = (start..end).collect();
    entry_indices
        .sort_by(|&left, &right| input.entry_times[left].total_cmp(&input.entry_times[right]));

    let mut at_risk = vec![0.0; n_groups];
    let mut entry_cursor = 0;
    let mut km_survival = 1.0_f64;
    let mut cursor = 0;
    while cursor < exit_indices.len() {
        let current_time = input.time[exit_indices[cursor]];
        while entry_cursor < entry_indices.len()
            && entry_precedes_event(
                input.entry_times[entry_indices[entry_cursor]],
                current_time,
                input.timefix,
            )
        {
            let idx = entry_indices[entry_cursor];
            let group_idx = (input.group[idx] - 1) as usize;
            at_risk[group_idx] += 1.0;
            entry_cursor += 1;
        }

        let mut events_by_group = vec![0.0; n_groups];
        let mut removed_by_group = vec![0.0; n_groups];
        let mut total_events = 0.0;
        while cursor < exit_indices.len()
            && same_logrank_time(
                input.time[exit_indices[cursor]],
                current_time,
                input.timefix,
            )
        {
            let idx = exit_indices[cursor];
            let group_idx = (input.group[idx] - 1) as usize;
            removed_by_group[group_idx] += 1.0;
            if input.status[idx] == 1 {
                events_by_group[group_idx] += 1.0;
                total_events += 1.0;
            }
            cursor += 1;
        }

        if total_events > 0.0 {
            let total_at_risk: f64 = at_risk.iter().sum();
            if total_at_risk > 0.0 {
                let weight = km_survival.powf(input.rho);
                for group_idx in 0..n_groups {
                    observed[group_idx] += weight * events_by_group[group_idx];
                    expected[group_idx] +=
                        weight * total_events * at_risk[group_idx] / total_at_risk;
                }

                if total_at_risk > 1.0 {
                    let var_factor =
                        weight * weight * total_events * (total_at_risk - total_events)
                            / (total_at_risk * (total_at_risk - 1.0));
                    for row in 0..n_groups {
                        let row_start = row * n_groups;
                        for col in 0..n_groups {
                            let diagonal = if row == col { 1.0 } else { 0.0 };
                            variance[row_start + col] += var_factor
                                * at_risk[row]
                                * (diagonal - at_risk[col] / total_at_risk);
                        }
                    }
                }

                km_survival *= 1.0 - total_events / total_at_risk;
            }
        }

        for group_idx in 0..n_groups {
            at_risk[group_idx] -= removed_by_group[group_idx];
        }
    }
}

fn entry_precedes_event(entry_time: f64, event_time: f64, timefix: bool) -> bool {
    if timefix {
        entry_time < event_time - TIME_EPSILON
    } else {
        entry_time < event_time
    }
}

fn same_logrank_time(left: f64, right: f64, timefix: bool) -> bool {
    if timefix {
        same_time(left, right)
    } else {
        left == right
    }
}

/// Backward-compatible alias retained for the public crate and Python APIs.
#[pyfunction]
#[pyo3(signature = (time, status, group, strata=None, rho=None, timefix=None))]
pub fn survdiff2(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    strata: Option<Vec<i32>>,
    rho: Option<f64>,
    timefix: Option<bool>,
) -> PyResult<SurvDiffResult> {
    compute_logrank_components(time, status, group, strata, rho, timefix.unwrap_or(false))
}

pub(crate) struct SurvDiffInput<'a> {
    pub(crate) time: &'a [f64],
    pub(crate) status: &'a [i32],
    pub(crate) group: &'a [i32],
    pub(crate) strata: &'a [i32],
}

pub(crate) struct SurvDiffOutput<'a> {
    pub(crate) obs: &'a mut [f64],
    pub(crate) exp: &'a mut [f64],
    pub(crate) var: &'a mut [f64],
    pub(crate) risk: &'a mut [f64],
    pub(crate) kaplan: &'a mut [f64],
}

pub(crate) struct SurvDiffParams {
    pub(crate) nn: i32,
    pub(crate) nngroup: i32,
    pub(crate) _nstrat: i32,
    pub(crate) rho: f64,
}

pub(crate) fn compute_survdiff(
    params: SurvDiffParams,
    input: SurvDiffInput,
    output: &mut SurvDiffOutput,
) {
    let ntotal = params.nn as usize;
    let ngroup = params.nngroup as usize;
    let mut istart = 0;
    let mut koff = 0;
    for v in output.var.iter_mut() {
        *v = 0.0;
    }
    for o in output.obs.iter_mut() {
        *o = 0.0;
    }
    for e in output.exp.iter_mut() {
        *e = 0.0;
    }
    while istart < ntotal {
        let mut n = istart;
        while n < ntotal && input.strata[n] != 1 {
            n += 1;
        }
        if n < ntotal {
            n += 1;
        }
        if params.rho != 0.0 {
            let mut km = 1.0;
            let mut i = istart;
            while i < n {
                let current_time = input.time[i];
                let mut deaths = 0;
                let mut j = i;
                while j < n && input.time[j] == current_time {
                    output.kaplan[j] = km;
                    deaths += input.status[j] as usize;
                    j += 1;
                }
                let nrisk = (n - i) as f64;
                if nrisk > 0.0 && deaths > 0 {
                    km *= (nrisk - deaths as f64) / nrisk;
                }
                i = j;
            }
        }
        for r in output.risk.iter_mut().take(ngroup) {
            *r = 0.0;
        }
        let mut i = n.saturating_sub(1);
        loop {
            if i < istart || (istart == 0 && n == 0) {
                break;
            }
            let current_time = input.time[i];
            let mut deaths = 0;
            let mut j = i;
            let wt = if params.rho == 0.0 {
                1.0
            } else {
                output.kaplan[i].powf(params.rho)
            };
            loop {
                let k = (input.group[j] - 1) as usize;
                output.risk[k] += 1.0;
                deaths += input.status[j] as usize;
                if j == istart {
                    break;
                }
                if input.time[j - 1] != current_time {
                    break;
                }
                j -= 1;
            }
            let nrisk = (n - j) as f64;
            if deaths > 0 {
                for (k, risk_val) in output.risk.iter().take(ngroup).enumerate() {
                    let exp_index = koff + k;
                    output.exp[exp_index] += wt * (deaths as f64) * risk_val / nrisk;
                }
                for ti in j..=i {
                    if input.status[ti] == 1 {
                        let obs_index = koff + (input.group[ti] - 1) as usize;
                        output.obs[obs_index] += wt;
                    }
                }
                if nrisk > 1.0 {
                    let wt_sq = wt * wt;
                    let factor =
                        wt_sq * (deaths as f64) * (nrisk - deaths as f64) / (nrisk * (nrisk - 1.0));
                    for (j_group, &rj) in output.risk.iter().take(ngroup).enumerate() {
                        let var_start = j_group * ngroup;
                        let tmp = factor * rj;
                        for (k_group, &rk) in output.risk.iter().take(ngroup).enumerate() {
                            output.var[var_start + k_group] += tmp
                                * (if j_group == k_group {
                                    1.0 - rk / nrisk
                                } else {
                                    -rk / nrisk
                                });
                        }
                    }
                }
            }
            if j == istart {
                break;
            }
            i = j - 1;
        }
        istart = n;
        koff += ngroup;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_logrank_components_all_censored_has_zero_degrees_of_freedom() {
        let result = compute_logrank_components(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 0, 0],
            vec![1, 1, 2, 2],
            None,
            None,
            false,
        )
        .expect("all-censored logrank components should not underflow df");

        assert_eq!(result.chi_squared, 0.0);
        assert_eq!(result.degrees_of_freedom, 0);
    }

    #[test]
    fn compute_logrank_components_rejects_non_binary_status() {
        let err = compute_logrank_components(vec![1.0], vec![2], vec![1], None, None, false)
            .expect_err("non-binary status should fail");

        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn compute_logrank_components_rejects_sparse_huge_group_codes() {
        let err =
            compute_logrank_components(vec![1.0, 2.0], vec![1, 0], vec![1, 3], None, None, false)
                .expect_err("group code beyond n should fail");

        assert!(err.to_string().contains("group values must be between 1"));
    }

    #[test]
    fn compute_logrank_components_rejects_non_marker_strata() {
        let err =
            compute_logrank_components(vec![1.0], vec![1], vec![1], Some(vec![2]), None, false)
                .expect_err("strata markers should be 0 or 1");

        assert!(err.to_string().contains("strata values must be 0 or 1"));
    }

    #[test]
    fn compute_logrank_components_aggregates_strata() {
        let stratified = compute_logrank_components(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 0, 1],
            vec![1, 2, 1, 2],
            Some(vec![0, 1, 0, 1]),
            None,
            false,
        )
        .expect("stratified components should compute");
        let first_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![1, 0], vec![1, 2], None, None, false)
                .expect("first stratum should compute");
        let second_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![0, 1], vec![1, 2], None, None, false)
                .expect("second stratum should compute");

        assert_eq!(stratified.observed.len(), 2);
        assert_eq!(stratified.expected.len(), 2);
        assert!(
            (stratified.observed[0] - first_stratum.observed[0] - second_stratum.observed[0]).abs()
                < 1e-12
        );
        assert!(
            (stratified.observed[1] - first_stratum.observed[1] - second_stratum.observed[1]).abs()
                < 1e-12
        );
        assert!(
            (stratified.expected[0] - first_stratum.expected[0] - second_stratum.expected[0]).abs()
                < 1e-12
        );
        assert!(
            (stratified.expected[1] - first_stratum.expected[1] - second_stratum.expected[1]).abs()
                < 1e-12
        );
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (stratified.variance[row][col]
                        - first_stratum.variance[row][col]
                        - second_stratum.variance[row][col])
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn compute_logrank_components_counts_trailing_unmarked_stratum() {
        let stratified = compute_logrank_components(
            vec![1.0, 2.0, 1.0, 2.0],
            vec![1, 0, 0, 1],
            vec![1, 2, 1, 2],
            Some(vec![0, 1, 0, 0]),
            None,
            false,
        )
        .expect("stratified components should compute with trailing unmarked stratum");
        let first_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![1, 0], vec![1, 2], None, None, false)
                .expect("first stratum should compute");
        let second_stratum =
            compute_logrank_components(vec![1.0, 2.0], vec![0, 1], vec![1, 2], None, None, false)
                .expect("second stratum should compute");

        for group_idx in 0..2 {
            assert!(
                (stratified.observed[group_idx]
                    - first_stratum.observed[group_idx]
                    - second_stratum.observed[group_idx])
                    .abs()
                    < 1e-12
            );
            assert!(
                (stratified.expected[group_idx]
                    - first_stratum.expected[group_idx]
                    - second_stratum.expected[group_idx])
                    .abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn compute_logrank_components_honors_exact_timefix() {
        let fixed = compute_logrank_components(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            vec![1, 1, 0, 1],
            vec![1, 2, 1, 2],
            None,
            None,
            true,
        )
        .expect("time-fixed right-censored components should compute");
        let exact = compute_logrank_components(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            vec![1, 1, 0, 1],
            vec![1, 2, 1, 2],
            None,
            None,
            false,
        )
        .expect("exact right-censored components should compute");

        assert_eq!(exact.observed, vec![1.0, 2.0]);
        assert!((exact.expected[0] - 5.0 / 6.0).abs() < 1e-12);
        assert!((exact.expected[1] - 13.0 / 6.0).abs() < 1e-12);
        assert!((exact.variance[0][0] - 17.0 / 36.0).abs() < 1e-12);
        assert!((exact.chi_squared - 1.0 / 17.0).abs() < 1e-12);
        assert_eq!(fixed.chi_squared, 0.0);
    }

    #[test]
    fn compute_logrank_components_sorts_within_strata() {
        let sorted = compute_logrank_components(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            vec![1, 1, 0, 1],
            vec![1, 2, 1, 2],
            None,
            None,
            false,
        )
        .expect("sorted components should compute");
        let unsorted = compute_logrank_components(
            vec![3.0, 1.0 + 5e-10, 2.0, 1.0],
            vec![1, 1, 0, 1],
            vec![2, 2, 1, 1],
            None,
            None,
            false,
        )
        .expect("unsorted components should compute");

        assert_eq!(unsorted.observed, sorted.observed);
        assert_eq!(unsorted.expected, sorted.expected);
        assert_eq!(unsorted.variance, sorted.variance);
        assert_eq!(unsorted.chi_squared, sorted.chi_squared);
    }

    #[test]
    fn compute_counting_logrank_components_honors_exact_timefix() {
        let fixed = compute_counting_logrank_components(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            vec![1, 1, 0, 1],
            vec![1, 2, 1, 2],
            vec![0.0, 0.0, 0.0, 0.0],
            None,
            None,
            true,
        )
        .expect("time-fixed counting components should compute");
        let exact = compute_counting_logrank_components(
            vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
            vec![1, 1, 0, 1],
            vec![1, 2, 1, 2],
            vec![0.0, 0.0, 0.0, 0.0],
            None,
            None,
            false,
        )
        .expect("exact counting components should compute");

        assert_eq!(exact.observed, vec![1.0, 2.0]);
        assert!((exact.expected[0] - 5.0 / 6.0).abs() < 1e-12);
        assert!((exact.expected[1] - 13.0 / 6.0).abs() < 1e-12);
        assert!((exact.variance[0][0] - 17.0 / 36.0).abs() < 1e-12);
        assert!((exact.chi_squared - 1.0 / 17.0).abs() < 1e-12);
        assert_eq!(fixed.chi_squared, 0.0);
    }

    #[test]
    fn compute_counting_logrank_components_aggregates_strata_covariance() {
        let stratified = compute_counting_logrank_components(
            vec![1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            vec![1, 1, 0, 0, 1, 1],
            vec![1, 2, 3, 1, 2, 3],
            vec![0.0; 6],
            Some(vec![0, 0, 1, 0, 0, 1]),
            None,
            true,
        )
        .expect("stratified counting components should compute");
        let first = compute_counting_logrank_components(
            vec![1.0, 2.0, 3.0],
            vec![1, 1, 0],
            vec![1, 2, 3],
            vec![0.0; 3],
            None,
            None,
            true,
        )
        .expect("first counting stratum should compute");
        let second = compute_counting_logrank_components(
            vec![1.5, 2.5, 3.5],
            vec![0, 1, 1],
            vec![1, 2, 3],
            vec![0.0; 3],
            None,
            None,
            true,
        )
        .expect("second counting stratum should compute");

        for group_idx in 0..3 {
            assert!(
                (stratified.observed[group_idx]
                    - first.observed[group_idx]
                    - second.observed[group_idx])
                    .abs()
                    < 1e-12
            );
            assert!(
                (stratified.expected[group_idx]
                    - first.expected[group_idx]
                    - second.expected[group_idx])
                    .abs()
                    < 1e-12
            );
            for col_idx in 0..3 {
                assert!(
                    (stratified.variance[group_idx][col_idx]
                        - first.variance[group_idx][col_idx]
                        - second.variance[group_idx][col_idx])
                        .abs()
                        < 1e-12
                );
            }
        }
        assert_eq!(stratified.degrees_of_freedom, 2);
    }

    #[test]
    fn stratified_logrank_components_accepts_unsorted_strata_codes() {
        let raw = stratified_logrank_components(
            vec![2.0, 1.0, 3.0, 1.5, 2.5, 3.5],
            vec![0, 1, 1, 1, 1, 0],
            vec![2, 1, 2, 1, 2, 1],
            vec![1, 0, 0, 1, 0, 1],
            Some(0.5),
            true,
        )
        .expect("raw-strata right-censored components should compute");
        let marker = compute_logrank_components(
            vec![1.0, 3.0, 2.5, 2.0, 1.5, 3.5],
            vec![1, 1, 1, 0, 1, 0],
            vec![1, 2, 2, 2, 1, 1],
            Some(vec![0, 0, 1, 0, 0, 1]),
            Some(0.5),
            true,
        )
        .expect("marker-strata right-censored components should compute");

        assert_eq!(raw.observed, marker.observed);
        assert_eq!(raw.expected, marker.expected);
        assert_eq!(raw.variance, marker.variance);
        assert_eq!(raw.chi_squared, marker.chi_squared);
    }

    #[test]
    fn stratified_counting_logrank_components_accepts_unsorted_strata_codes() {
        let raw = stratified_counting_logrank_components(
            vec![2.0, 2.5, 4.0, 4.5, 3.0, 5.0],
            vec![1, 1, 0, 0, 1, 1],
            vec![1, 1, 2, 2, 1, 2],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 2.0],
            vec![0, 1, 0, 1, 0, 0],
            Some(0.5),
            true,
        )
        .expect("raw-strata counting components should compute");
        let marker = compute_counting_logrank_components(
            vec![2.0, 4.0, 3.0, 5.0, 2.5, 4.5],
            vec![1, 0, 1, 1, 1, 0],
            vec![1, 2, 1, 2, 1, 2],
            vec![0.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            Some(vec![0, 0, 0, 1, 0, 1]),
            Some(0.5),
            true,
        )
        .expect("marker-strata counting components should compute");

        assert_eq!(raw.observed, marker.observed);
        assert_eq!(raw.expected, marker.expected);
        assert_eq!(raw.variance, marker.variance);
        assert_eq!(raw.chi_squared, marker.chi_squared);
    }

    #[test]
    fn compute_logrank_components_uses_full_multigroup_covariance_statistic() {
        let result = compute_logrank_components(
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![1, 1, 1, 0, 1, 0, 1, 1, 0],
            vec![1, 1, 2, 1, 3, 2, 3, 2, 3],
            None,
            None,
            false,
        )
        .expect("three-group survdiff components should compute");

        assert_eq!(result.degrees_of_freedom, 2);
        assert_eq!(result.observed, vec![2.0, 2.0, 2.0]);
        assert_eq!(result.expected, vec![1.0, 2.25, 2.75]);
        assert!((result.variance[0][0] - 0.6825396825396826).abs() < 1e-12);
        assert!((result.variance[0][1] + 0.3273809523809524).abs() < 1e-12);
        assert!((result.chi_squared - 1.5105257668985863).abs() < 1e-12);
    }
}
