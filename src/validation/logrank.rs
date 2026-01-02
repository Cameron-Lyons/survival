use crate::utilities::numpy_utils::{extract_vec_f64, extract_vec_i32};
use crate::utilities::statistical::chi2_sf;
use pyo3::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
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
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut at_risk: Vec<f64> = vec![0.0; n_groups];
    for &grp in group {
        let g = unique_groups.iter().position(|&x| x == grp).unwrap_or(0);
        at_risk[g] += 1.0;
    }
    let mut observed = vec![0.0; n_groups];
    let mut expected = vec![0.0; n_groups];
    let mut variance_sum = 0.0;
    let mut km_survival = 1.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events_by_group = vec![0.0; n_groups];
        let mut total_events = 0.0;
        let mut removed = vec![0.0; n_groups];
        while i < n && time[indices[i]] == current_time {
            let idx = indices[i];
            let g = unique_groups
                .iter()
                .position(|&x| x == group[idx])
                .unwrap_or(0);
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
///     Object with: chi_square (test statistic), p_value, observed/expected counts per group.
///
/// Examples
/// --------
/// >>> result = survival.logrank_test(time, status, group)
/// >>> print(f"p-value: {result.p_value:.4f}")
#[pyfunction]
#[pyo3(signature = (time, status, group, weight_type=None))]
pub fn logrank_test(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    weight_type: Option<&str>,
) -> PyResult<LogRankResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let group = extract_vec_i32(group)?;
    let wt = match weight_type {
        Some("wilcoxon") | Some("Wilcoxon") => WeightType::Wilcoxon,
        Some("tarone-ware") | Some("TaroneWare") => WeightType::TaroneWare,
        Some("peto-peto") | Some("PetoPeto") | Some("peto") => WeightType::PetoPeto,
        _ => WeightType::LogRank,
    };
    Ok(weighted_logrank_test(&time, &status, &group, wt))
}
#[pyfunction]
#[pyo3(signature = (time, status, group, p, q))]
pub fn fleming_harrington_test(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    group: &Bound<'_, PyAny>,
    p: f64,
    q: f64,
) -> PyResult<LogRankResult> {
    let time = extract_vec_f64(time)?;
    let status = extract_vec_i32(status)?;
    let group = extract_vec_i32(group)?;
    Ok(weighted_logrank_test(
        &time,
        &status,
        &group,
        WeightType::FlemingHarrington { p, q },
    ))
}
#[derive(Debug, Clone)]
#[pyclass]
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
pub fn logrank_trend_test(
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
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut at_risk: Vec<f64> = vec![0.0; n_groups];
    for &grp in group {
        let g = unique_groups.iter().position(|&x| x == grp).unwrap_or(0);
        at_risk[g] += 1.0;
    }
    let mut u_stat = 0.0;
    let mut var_stat = 0.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events_by_group = vec![0.0; n_groups];
        let mut total_events = 0.0;
        let mut removed = vec![0.0; n_groups];
        while i < n && time[indices[i]] == current_time {
            let idx = indices[i];
            let g = unique_groups
                .iter()
                .position(|&x| x == group[idx])
                .unwrap_or(0);
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
    Ok(logrank_trend_test(&time, &status, &group, scores_ref))
}
