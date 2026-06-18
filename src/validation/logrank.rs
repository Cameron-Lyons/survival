use crate::constants::TIME_EPSILON;
use crate::internal::numpy_utils::{extract_optional_vec_f64, extract_vec_f64, extract_vec_i32};
use crate::internal::statistical::chi2_sf;
use crate::internal::validation::{
    validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::prelude::*;
use std::collections::HashMap;
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct LogRankResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub observed: Vec<f64>,
    #[pyo3(get)]
    pub expected: Vec<f64>,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub weight_type: String,
}
#[pymethods]
impl LogRankResult {
    #[new]
    fn new(
        statistic: f64,
        p_value: f64,
        df: usize,
        observed: Vec<f64>,
        expected: Vec<f64>,
        variance: f64,
        weight_type: String,
    ) -> Self {
        Self {
            statistic,
            p_value,
            df,
            observed,
            expected,
            variance,
            weight_type,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum WeightType {
    LogRank,
    Wilcoxon,
    TaroneWare,
    PetoPeto,
    FlemingHarrington { p: f64, q: f64 },
}
pub fn weighted_logrank_test(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    weight_type: WeightType,
) -> LogRankResult {
    weighted_logrank_test_with_entry_times(time, status, group, None, weight_type)
}

pub fn weighted_logrank_test_with_entry_times(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    entry_times: Option<&[f64]>,
    weight_type: WeightType,
) -> LogRankResult {
    let n = time.len();
    if n == 0 {
        return LogRankResult {
            statistic: 0.0,
            p_value: 1.0,
            df: 1,
            observed: vec![],
            expected: vec![],
            variance: 0.0,
            weight_type: "LogRank".to_string(),
        };
    }
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    let n_groups = unique_groups.len();
    if n_groups < 2 {
        return LogRankResult {
            statistic: 0.0,
            p_value: 1.0,
            df: 0,
            observed: vec![0.0; n_groups],
            expected: vec![0.0; n_groups],
            variance: 0.0,
            weight_type: weight_name(&weight_type),
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
    let group_to_index: HashMap<i32, usize> = unique_groups
        .iter()
        .enumerate()
        .map(|(idx, &g)| (g, idx))
        .collect();
    let mut at_risk: Vec<f64> = vec![0.0; n_groups];
    let entry_indices = entry_times.map(|entry| {
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| entry[a].total_cmp(&entry[b]));
        indices
    });
    let mut entry_cursor = 0;
    if entry_times.is_none() {
        for &grp in group {
            if let Some(&g) = group_to_index.get(&grp) {
                at_risk[g] += 1.0;
            }
        }
    }
    let mut observed = vec![0.0; n_groups];
    let mut expected = vec![0.0; n_groups];
    let mut variance_sum = 0.0;
    let mut km_survival = 1.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        if let (Some(entry), Some(sorted_entries)) = (entry_times, entry_indices.as_ref()) {
            while entry_cursor < n
                && entry[sorted_entries[entry_cursor]] < current_time - TIME_EPSILON
            {
                let idx = sorted_entries[entry_cursor];
                if let Some(&g) = group_to_index.get(&group[idx]) {
                    at_risk[g] += 1.0;
                }
                entry_cursor += 1;
            }
        }
        let mut events_by_group = vec![0.0; n_groups];
        let mut total_events = 0.0;
        let mut removed = vec![0.0; n_groups];
        while i < n && same_time(time[indices[i]], current_time) {
            let idx = indices[i];
            let g = group_to_index.get(&group[idx]).copied().unwrap_or(0);
            removed[g] += 1.0;
            if status[idx] == 1 {
                events_by_group[g] += 1.0;
                total_events += 1.0;
            }
            i += 1;
        }
        if total_events > 0.0 {
            let total_at_risk: f64 = at_risk.iter().sum();
            if total_at_risk > 0.0 {
                let weight = match weight_type {
                    WeightType::LogRank => 1.0,
                    WeightType::Wilcoxon => total_at_risk,
                    WeightType::TaroneWare => total_at_risk.sqrt(),
                    WeightType::PetoPeto => km_survival,
                    WeightType::FlemingHarrington { p, q } => {
                        km_survival.powf(p) * (1.0 - km_survival).powf(q)
                    }
                };
                for g in 0..n_groups {
                    observed[g] += weight * events_by_group[g];
                    let exp_g = total_events * at_risk[g] / total_at_risk;
                    expected[g] += weight * exp_g;
                }
                if total_at_risk > 1.0 {
                    let var_factor = total_events * (total_at_risk - total_events)
                        / (total_at_risk * total_at_risk * (total_at_risk - 1.0));
                    for &n_g in at_risk.iter().take(n_groups - 1) {
                        let n_not_g = total_at_risk - n_g;
                        variance_sum += weight * weight * var_factor * n_g * n_not_g;
                    }
                }
                km_survival *= 1.0 - total_events / total_at_risk;
            }
        }
        for g in 0..n_groups {
            at_risk[g] -= removed[g];
        }
    }
    let statistic = if variance_sum > 0.0 {
        let diff = observed[0] - expected[0];
        diff * diff / variance_sum
    } else {
        0.0
    };
    let p_value = chi2_sf(statistic, n_groups - 1);
    LogRankResult {
        statistic,
        p_value,
        df: n_groups - 1,
        observed,
        expected,
        variance: variance_sum,
        weight_type: weight_name(&weight_type),
    }
}

fn validate_logrank_inputs(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    entry_times: Option<&[f64]>,
) -> PyResult<()> {
    validate_length(time.len(), status.len(), "status")?;
    validate_length(time.len(), group.len(), "group")?;
    validate_binary_status(status)?;
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    if let Some(entry) = entry_times {
        validate_length(time.len(), entry.len(), "entry_times")?;
        validate_no_nan(entry, "entry_times")?;
        validate_finite(entry, "entry_times")?;
        validate_non_negative(entry, "entry_times")?;
        for (idx, (&entry_time, &exit_time)) in entry.iter().zip(time.iter()).enumerate() {
            if entry_time >= exit_time - TIME_EPSILON {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "entry_times must be less than time for observation {}",
                    idx
                )));
            }
        }
    }
    Ok(())
}

fn validate_binary_status(status: &[i32]) -> PyResult<()> {
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "status must contain only 0/1 values; got {} at index {}",
                value, idx
            )));
        }
    }
    Ok(())
}

fn validate_fleming_harrington_parameters(rho: f64, gamma: f64) -> PyResult<()> {
    if !rho.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rho must be finite",
        ));
    }
    if !gamma.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gamma must be finite",
        ));
    }
    Ok(())
}

fn validate_logrank_trend_inputs(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    scores: Option<&[f64]>,
) -> PyResult<()> {
    validate_length(time.len(), status.len(), "status")?;
    validate_length(time.len(), group.len(), "group")?;
    validate_binary_status(status)?;
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    if let Some(values) = scores {
        let mut unique_groups: Vec<i32> = group.to_vec();
        unique_groups.sort();
        unique_groups.dedup();
        validate_length(unique_groups.len(), values.len(), "scores")?;
        validate_no_nan(values, "scores")?;
        validate_finite(values, "scores")?;
    }
    Ok(())
}

fn same_time(left: f64, right: f64) -> bool {
    (left - right).abs() < TIME_EPSILON
}

fn weight_name(weight_type: &WeightType) -> String {
    match weight_type {
        WeightType::LogRank => "LogRank".to_string(),
        WeightType::Wilcoxon => "Wilcoxon".to_string(),
        WeightType::TaroneWare => "TaroneWare".to_string(),
        WeightType::PetoPeto => "PetoPeto".to_string(),
        WeightType::FlemingHarrington { p, q } => format!("FlemingHarrington(p={}, q={})", p, q),
    }
}
/// Perform log-rank test comparing survival between groups.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times. Accepts numpy, pandas, polars, or lists.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// group : array-like
///     Group membership indicator (integer-coded).
/// weight_type : str, optional
///     Weight function: "logrank" (default), "wilcoxon", "tarone-ware", or "peto-peto".
///
/// Returns
/// -------
/// LogRankResult
///     Object with: statistic (test statistic), p_value, observed/expected counts per group.
///
/// Examples
/// --------
/// >>> result = survival.logrank_test(time, status, group)
/// >>> print(f"p-value: {result.p_value:.4f}")
#[pyfunction]
#[pyo3(signature = (time, status, group, weight_type=None, entry_times=None))]
pub fn logrank_test(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    weight_type: Option<&str>,
    entry_times: Option<&Bound<'_, PyAny>>,
) -> PyResult<LogRankResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let group = extract_vec_i32(group)?;
    let entry_times = extract_optional_vec_f64(entry_times)?;
    validate_logrank_inputs(&time, &status, &group, entry_times.as_deref())?;
    let wt = match weight_type {
        Some("wilcoxon") | Some("Wilcoxon") => WeightType::Wilcoxon,
        Some("tarone-ware") | Some("TaroneWare") => WeightType::TaroneWare,
        Some("peto-peto") | Some("PetoPeto") | Some("peto") => WeightType::PetoPeto,
        _ => WeightType::LogRank,
    };
    Ok(weighted_logrank_test_with_entry_times(
        &time,
        &status,
        &group,
        entry_times.as_deref(),
        wt,
    ))
}
#[pyfunction]
#[pyo3(signature = (time, status, group, rho=0.0, gamma=0.0, entry_times=None))]
pub fn fleming_harrington_test(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    rho: f64,
    gamma: f64,
    entry_times: Option<&Bound<'_, PyAny>>,
) -> PyResult<LogRankResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let group = extract_vec_i32(group)?;
    let entry_times = extract_optional_vec_f64(entry_times)?;
    validate_logrank_inputs(&time, &status, &group, entry_times.as_deref())?;
    validate_fleming_harrington_parameters(rho, gamma)?;
    Ok(weighted_logrank_test_with_entry_times(
        &time,
        &status,
        &group,
        entry_times.as_deref(),
        WeightType::FlemingHarrington { p: rho, q: gamma },
    ))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TrendTestResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub trend_direction: String,
}
#[pymethods]
impl TrendTestResult {
    #[new]
    fn new(statistic: f64, p_value: f64, trend_direction: String) -> Self {
        Self {
            statistic,
            p_value,
            trend_direction,
        }
    }
}
pub(crate) fn logrank_trend_test(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    scores: Option<&[f64]>,
) -> TrendTestResult {
    let n = time.len();
    if n == 0 {
        return TrendTestResult {
            statistic: 0.0,
            p_value: 1.0,
            trend_direction: "none".to_string(),
        };
    }
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    let n_groups = unique_groups.len();
    let default_scores: Vec<f64> = (0..n_groups).map(|i| i as f64).collect();
    let scores = scores.unwrap_or(&default_scores);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
    let group_to_index: HashMap<i32, usize> = unique_groups
        .iter()
        .enumerate()
        .map(|(idx, &g)| (g, idx))
        .collect();
    let mut at_risk: Vec<f64> = vec![0.0; n_groups];
    for &grp in group {
        if let Some(&g) = group_to_index.get(&grp) {
            at_risk[g] += 1.0;
        }
    }
    let mut u_stat = 0.0;
    let mut var_stat = 0.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events_by_group = vec![0.0; n_groups];
        let mut total_events = 0.0;
        let mut removed = vec![0.0; n_groups];
        while i < n && same_time(time[indices[i]], current_time) {
            let idx = indices[i];
            let g = group_to_index.get(&group[idx]).copied().unwrap_or(0);
            removed[g] += 1.0;
            if status[idx] == 1 {
                events_by_group[g] += 1.0;
                total_events += 1.0;
            }
            i += 1;
        }
        if total_events > 0.0 {
            let total_at_risk: f64 = at_risk.iter().sum();
            if total_at_risk > 1.0 {
                let mut score_mean = 0.0;
                let mut score_var = 0.0;
                for g in 0..n_groups {
                    score_mean += scores[g] * at_risk[g] / total_at_risk;
                }
                for g in 0..n_groups {
                    score_var += at_risk[g] * (scores[g] - score_mean).powi(2) / total_at_risk;
                }
                for g in 0..n_groups {
                    let exp_g = total_events * at_risk[g] / total_at_risk;
                    u_stat += scores[g] * (events_by_group[g] - exp_g);
                }
                let var_factor = total_events * (total_at_risk - total_events)
                    / (total_at_risk * (total_at_risk - 1.0));
                var_stat += var_factor * score_var * total_at_risk;
            }
        }
        for g in 0..n_groups {
            at_risk[g] -= removed[g];
        }
    }
    let statistic = if var_stat > 0.0 {
        u_stat * u_stat / var_stat
    } else {
        0.0
    };
    let p_value = chi2_sf(statistic, 1);
    let trend_direction = if u_stat > 0.0 {
        "increasing".to_string()
    } else if u_stat < 0.0 {
        "decreasing".to_string()
    } else {
        "none".to_string()
    };
    TrendTestResult {
        statistic,
        p_value,
        trend_direction,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, group, scores=None))]
pub fn logrank_trend(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    scores: Option<Vec<f64>>,
) -> PyResult<TrendTestResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let group = extract_vec_i32(group)?;
    let scores_ref = scores.as_deref();
    validate_logrank_trend_inputs(&time, &status, &group, scores_ref)?;
    Ok(logrank_trend_test(&time, &status, &group, scores_ref))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delayed_entry_logrank_updates_risk_sets() {
        let entry_times = vec![0.0, 0.0, 1.0, 2.0];
        let time = vec![2.0, 4.0, 3.0, 5.0];
        let status = vec![1, 0, 1, 1];
        let group = vec![1, 0, 1, 0];

        let result = weighted_logrank_test_with_entry_times(
            &time,
            &status,
            &group,
            Some(&entry_times),
            WeightType::LogRank,
        );

        assert_eq!(result.observed, vec![1.0, 2.0]);
        assert_eq!(result.expected, vec![2.0, 1.0]);
        assert!((result.variance - 4.0 / 9.0).abs() < 1e-10);
        assert!((result.statistic - 2.25).abs() < 1e-10);
    }

    #[test]
    fn logrank_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 1];
        let group = vec![0, 1, 0, 1];

        let exact = weighted_logrank_test(&exact_time, &status, &group, WeightType::LogRank);
        let near = weighted_logrank_test(&near_time, &status, &group, WeightType::LogRank);

        assert_eq!(near.observed, exact.observed);
        assert_eq!(near.expected, exact.expected);
        assert!((near.variance - exact.variance).abs() < 1e-12);
        assert!((near.statistic - exact.statistic).abs() < 1e-12);
        assert!((near.p_value - exact.p_value).abs() < 1e-12);
    }
}
