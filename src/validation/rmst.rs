use crate::constants::z_score_for_confidence;
use crate::utilities::statistical::{lower_incomplete_gamma, normal_cdf as norm_cdf};
use pyo3::prelude::*;
use std::fmt;

fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0)
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct RMSTResult {
    pub rmst: f64,
    pub variance: f64,
    pub se: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub tau: f64,
}

impl fmt::Display for RMSTResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RMSTResult(rmst={:.4}, se={:.4}, ci=[{:.4}, {:.4}], tau={:.2})",
            self.rmst, self.se, self.ci_lower, self.ci_upper, self.tau
        )
    }
}

#[pymethods]
impl RMSTResult {
    #[new]
    fn new(rmst: f64, variance: f64, se: f64, ci_lower: f64, ci_upper: f64, tau: f64) -> Self {
        Self {
            rmst,
            variance,
            se,
            ci_lower,
            ci_upper,
            tau,
        }
    }
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct RMSTComparisonResult {
    #[pyo3(get)]
    pub rmst_diff: f64,
    #[pyo3(get)]
    pub rmst_ratio: f64,
    #[pyo3(get)]
    pub diff_se: f64,
    #[pyo3(get)]
    pub diff_ci_lower: f64,
    #[pyo3(get)]
    pub diff_ci_upper: f64,
    #[pyo3(get)]
    pub ratio_ci_lower: f64,
    #[pyo3(get)]
    pub ratio_ci_upper: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub rmst_group1: RMSTResult,
    #[pyo3(get)]
    pub rmst_group2: RMSTResult,
}
#[pymethods]
impl RMSTComparisonResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        rmst_diff: f64,
        rmst_ratio: f64,
        diff_se: f64,
        diff_ci_lower: f64,
        diff_ci_upper: f64,
        ratio_ci_lower: f64,
        ratio_ci_upper: f64,
        p_value: f64,
        rmst_group1: RMSTResult,
        rmst_group2: RMSTResult,
    ) -> Self {
        Self {
            rmst_diff,
            rmst_ratio,
            diff_se,
            diff_ci_lower,
            diff_ci_upper,
            ratio_ci_lower,
            ratio_ci_upper,
            p_value,
            rmst_group1,
            rmst_group2,
        }
    }
}
pub fn compute_rmst(time: &[f64], status: &[i32], tau: f64, confidence_level: f64) -> RMSTResult {
    let n = time.len();
    if n == 0 {
        return RMSTResult {
            rmst: 0.0,
            variance: 0.0,
            se: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            tau,
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut unique_times: Vec<f64> = Vec::new();
    let mut n_events: Vec<f64> = Vec::new();
    let mut n_risk: Vec<f64> = Vec::new();
    let mut total_at_risk = n as f64;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        if current_time > tau {
            break;
        }
        let mut events = 0.0;
        let mut removed = 0.0;
        while i < n && time[indices[i]] == current_time {
            removed += 1.0;
            if status[indices[i]] == 1 {
                events += 1.0;
            }
            i += 1;
        }
        if events > 0.0 {
            unique_times.push(current_time);
            n_events.push(events);
            n_risk.push(total_at_risk);
        }
        total_at_risk -= removed;
    }
    let m = unique_times.len();
    if m == 0 {
        return RMSTResult {
            rmst: tau,
            variance: 0.0,
            se: 0.0,
            ci_lower: tau,
            ci_upper: tau,
            tau,
        };
    }
    let mut survival = Vec::with_capacity(m);
    let mut surv = 1.0;
    for j in 0..m {
        surv *= 1.0 - n_events[j] / n_risk[j];
        survival.push(surv);
    }
    let mut rmst = 0.0;
    let mut prev_time = 0.0;
    for j in 0..m {
        let prev_surv = if j == 0 { 1.0 } else { survival[j - 1] };
        rmst += prev_surv * (unique_times[j] - prev_time);
        prev_time = unique_times[j];
    }
    let last_surv = survival[m - 1];
    rmst += last_surv * (tau - prev_time);
    let mut variance = 0.0;
    let mut cum_area_after: Vec<f64> = vec![0.0; m];
    for j in (0..m).rev() {
        let area_to_tau = if j == m - 1 {
            survival[j] * (tau - unique_times[j])
        } else {
            survival[j] * (unique_times[j + 1] - unique_times[j]) + cum_area_after[j + 1]
        };
        cum_area_after[j] = area_to_tau;
    }
    for j in 0..m {
        let d = n_events[j];
        let y = n_risk[j];
        if y > d && y > 0.0 {
            let area = cum_area_after[j];
            variance += d * area * area / (y * (y - d));
        }
    }
    let se = variance.sqrt();
    let z = z_score_for_confidence(confidence_level);
    let ci_lower = (rmst - z * se).max(0.0);
    let ci_upper = rmst + z * se;
    RMSTResult {
        rmst,
        variance,
        se,
        ci_lower,
        ci_upper,
        tau,
    }
}
pub fn compare_rmst(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    tau: f64,
    confidence_level: f64,
) -> RMSTComparisonResult {
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    if unique_groups.len() < 2 {
        let result = compute_rmst(time, status, tau, confidence_level);
        return RMSTComparisonResult {
            rmst_diff: 0.0,
            rmst_ratio: 1.0,
            diff_se: 0.0,
            diff_ci_lower: 0.0,
            diff_ci_upper: 0.0,
            ratio_ci_lower: 1.0,
            ratio_ci_upper: 1.0,
            p_value: 1.0,
            rmst_group1: result.clone(),
            rmst_group2: result,
        };
    }
    let g1 = unique_groups[0];
    let g2 = unique_groups[1];
    let mut time1 = Vec::new();
    let mut status1 = Vec::new();
    let mut time2 = Vec::new();
    let mut status2 = Vec::new();
    for i in 0..time.len() {
        if group[i] == g1 {
            time1.push(time[i]);
            status1.push(status[i]);
        } else if group[i] == g2 {
            time2.push(time[i]);
            status2.push(status[i]);
        }
    }
    let (rmst1, rmst2) = rayon::join(
        || compute_rmst(&time1, &status1, tau, confidence_level),
        || compute_rmst(&time2, &status2, tau, confidence_level),
    );
    let diff = rmst1.rmst - rmst2.rmst;
    let diff_var = rmst1.variance + rmst2.variance;
    let diff_se = diff_var.sqrt();
    let z = z_score_for_confidence(confidence_level);
    let diff_ci_lower = diff - z * diff_se;
    let diff_ci_upper = diff + z * diff_se;
    let ratio = if rmst2.rmst > 0.0 {
        rmst1.rmst / rmst2.rmst
    } else {
        f64::INFINITY
    };
    let (ratio_ci_lower, ratio_ci_upper) = if rmst1.rmst > 0.0 && rmst2.rmst > 0.0 {
        let log_ratio = ratio.ln();
        let log_ratio_var =
            rmst1.variance / (rmst1.rmst * rmst1.rmst) + rmst2.variance / (rmst2.rmst * rmst2.rmst);
        let log_ratio_se = log_ratio_var.sqrt();
        (
            (log_ratio - z * log_ratio_se).exp(),
            (log_ratio + z * log_ratio_se).exp(),
        )
    } else {
        (0.0, f64::INFINITY)
    };
    let z_stat = if diff_se > 0.0 { diff / diff_se } else { 0.0 };
    let p_value = 2.0 * (1.0 - norm_cdf(z_stat.abs()));
    RMSTComparisonResult {
        rmst_diff: diff,
        rmst_ratio: ratio,
        diff_se,
        diff_ci_lower,
        diff_ci_upper,
        ratio_ci_lower,
        ratio_ci_upper,
        p_value,
        rmst_group1: rmst1,
        rmst_group2: rmst2,
    }
}
/// Compute Restricted Mean Survival Time (RMST).
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// tau : float
///     Time horizon for restriction.
/// confidence_level : float, optional
///     Confidence level (default 0.95).
///
/// Returns
/// -------
/// RMSTResult
///     Object with: rmst (estimate), std_err, conf_lower, conf_upper.
#[pyfunction]
#[pyo3(signature = (time, status, tau, confidence_level=None))]
pub fn rmst(
    time: Vec<f64>,
    status: Vec<i32>,
    tau: f64,
    confidence_level: Option<f64>,
) -> PyResult<RMSTResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_rmst(&time, &status, tau, conf))
}

/// Compare RMST between two groups.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// group : array-like
///     Group indicator (0 or 1).
/// tau : float
///     Time horizon for restriction.
/// confidence_level : float, optional
///     Confidence level (default 0.95).
///
/// Returns
/// -------
/// RMSTComparisonResult
///     Object with: difference, std_err, conf_lower, conf_upper, p_value, rmst_group1, rmst_group2.
#[pyfunction]
#[pyo3(signature = (time, status, group, tau, confidence_level=None))]
pub fn rmst_comparison(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    tau: f64,
    confidence_level: Option<f64>,
) -> PyResult<RMSTComparisonResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compare_rmst(&time, &status, &group, tau, conf))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct MedianSurvivalResult {
    #[pyo3(get)]
    pub median: Option<f64>,
    #[pyo3(get)]
    pub ci_lower: Option<f64>,
    #[pyo3(get)]
    pub ci_upper: Option<f64>,
    #[pyo3(get)]
    pub quantile: f64,
}
#[pymethods]
impl MedianSurvivalResult {
    #[new]
    fn new(
        median: Option<f64>,
        ci_lower: Option<f64>,
        ci_upper: Option<f64>,
        quantile: f64,
    ) -> Self {
        Self {
            median,
            ci_lower,
            ci_upper,
            quantile,
        }
    }
}
pub fn compute_survival_quantile(
    time: &[f64],
    status: &[i32],
    quantile: f64,
    confidence_level: f64,
) -> MedianSurvivalResult {
    let n = time.len();
    if n == 0 {
        return MedianSurvivalResult {
            median: None,
            ci_lower: None,
            ci_upper: None,
            quantile,
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut unique_times: Vec<f64> = Vec::new();
    let mut survival: Vec<f64> = Vec::new();
    let mut ci_lower_vec: Vec<f64> = Vec::new();
    let mut ci_upper_vec: Vec<f64> = Vec::new();
    let mut total_at_risk = n as f64;
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    let z = z_score_for_confidence(confidence_level);
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events = 0.0;
        let mut removed = 0.0;
        while i < n && time[indices[i]] == current_time {
            removed += 1.0;
            if status[indices[i]] == 1 {
                events += 1.0;
            }
            i += 1;
        }
        if events > 0.0 && total_at_risk > 0.0 {
            surv *= 1.0 - events / total_at_risk;
            if total_at_risk > events {
                var_sum += events / (total_at_risk * (total_at_risk - events));
            }
            let se = surv * var_sum.sqrt();
            let lower = (surv - z * se).clamp(0.0, 1.0);
            let upper = (surv + z * se).clamp(0.0, 1.0);
            unique_times.push(current_time);
            survival.push(surv);
            ci_lower_vec.push(lower);
            ci_upper_vec.push(upper);
        }
        total_at_risk -= removed;
    }
    let target = 1.0 - quantile;
    let median = survival
        .iter()
        .position(|&s| s <= target)
        .map(|idx| unique_times[idx]);
    let ci_lower = ci_upper_vec
        .iter()
        .position(|&s| s <= target)
        .map(|idx| unique_times[idx]);
    let ci_upper = ci_lower_vec
        .iter()
        .position(|&s| s <= target)
        .map(|idx| unique_times[idx]);
    MedianSurvivalResult {
        median,
        ci_lower,
        ci_upper,
        quantile,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, quantile=None, confidence_level=None))]
pub fn survival_quantile(
    time: Vec<f64>,
    status: Vec<i32>,
    quantile: Option<f64>,
    confidence_level: Option<f64>,
) -> PyResult<MedianSurvivalResult> {
    let q = quantile.unwrap_or(0.5);
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_survival_quantile(&time, &status, q, conf))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct CumulativeIncidenceResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub cif: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub variance: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub event_types: Vec<i32>,
    #[pyo3(get)]
    pub n_risk: Vec<usize>,
}
#[pymethods]
impl CumulativeIncidenceResult {
    #[new]
    fn new(
        time: Vec<f64>,
        cif: Vec<Vec<f64>>,
        variance: Vec<Vec<f64>>,
        event_types: Vec<i32>,
        n_risk: Vec<usize>,
    ) -> Self {
        Self {
            time,
            cif,
            variance,
            event_types,
            n_risk,
        }
    }
}
pub fn compute_cumulative_incidence(time: &[f64], status: &[i32]) -> CumulativeIncidenceResult {
    let n = time.len();
    if n == 0 {
        return CumulativeIncidenceResult {
            time: vec![],
            cif: vec![],
            variance: vec![],
            event_types: vec![],
            n_risk: vec![],
        };
    }
    let mut event_types: Vec<i32> = status.iter().filter(|&&s| s > 0).copied().collect();
    event_types.sort();
    event_types.dedup();
    if event_types.is_empty() {
        return CumulativeIncidenceResult {
            time: vec![],
            cif: vec![],
            variance: vec![],
            event_types: vec![],
            n_risk: vec![],
        };
    }
    let n_event_types = event_types.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut unique_times: Vec<f64> = Vec::new();
    let mut n_risk_vec: Vec<usize> = Vec::new();
    let mut events_by_type: Vec<Vec<f64>> = vec![Vec::new(); n_event_types];
    let mut total_at_risk = n;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut event_counts = vec![0.0; n_event_types];
        let mut removed = 0usize;
        while i < n && time[indices[i]] == current_time {
            let s = status[indices[i]];
            removed += 1;
            if let Some(idx) = event_types.iter().position(|&e| e == s) {
                event_counts[idx] += 1.0;
            }
            i += 1;
        }
        let has_events = event_counts.iter().any(|&c| c > 0.0);
        if has_events {
            unique_times.push(current_time);
            n_risk_vec.push(total_at_risk);
            for (k, count) in event_counts.into_iter().enumerate() {
                events_by_type[k].push(count);
            }
        }
        total_at_risk -= removed;
    }
    let m = unique_times.len();
    let mut cif: Vec<Vec<f64>> = vec![Vec::with_capacity(m); n_event_types];
    let mut variance: Vec<Vec<f64>> = vec![Vec::with_capacity(m); n_event_types];
    let mut km_survival = 1.0;
    let mut cum_cif = vec![0.0; n_event_types];
    for j in 0..m {
        let y = n_risk_vec[j] as f64;
        let total_events: f64 = events_by_type.iter().map(|ev| ev[j]).sum();
        for k in 0..n_event_types {
            let d_k = events_by_type[k][j];
            if y > 0.0 {
                cum_cif[k] += km_survival * d_k / y;
            }
            cif[k].push(cum_cif[k]);
            variance[k].push(0.0);
        }
        if y > 0.0 {
            km_survival *= 1.0 - total_events / y;
        }
    }
    CumulativeIncidenceResult {
        time: unique_times,
        cif,
        variance,
        event_types,
        n_risk: n_risk_vec,
    }
}
#[pyfunction]
pub fn cumulative_incidence(
    time: Vec<f64>,
    status: Vec<i32>,
) -> PyResult<CumulativeIncidenceResult> {
    Ok(compute_cumulative_incidence(&time, &status))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct NNTResult {
    #[pyo3(get)]
    pub nnt: f64,
    #[pyo3(get)]
    pub nnt_ci_lower: f64,
    #[pyo3(get)]
    pub nnt_ci_upper: f64,
    #[pyo3(get)]
    pub absolute_risk_reduction: f64,
    #[pyo3(get)]
    pub arr_ci_lower: f64,
    #[pyo3(get)]
    pub arr_ci_upper: f64,
    #[pyo3(get)]
    pub time_horizon: f64,
}
#[pymethods]
impl NNTResult {
    #[new]
    fn new(
        nnt: f64,
        nnt_ci_lower: f64,
        nnt_ci_upper: f64,
        absolute_risk_reduction: f64,
        arr_ci_lower: f64,
        arr_ci_upper: f64,
        time_horizon: f64,
    ) -> Self {
        Self {
            nnt,
            nnt_ci_lower,
            nnt_ci_upper,
            absolute_risk_reduction,
            arr_ci_lower,
            arr_ci_upper,
            time_horizon,
        }
    }
}
pub fn compute_nnt(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    time_horizon: f64,
    confidence_level: f64,
) -> NNTResult {
    let surv1 = compute_survival_at_time(time, status, group, 0, time_horizon);
    let surv2 = compute_survival_at_time(time, status, group, 1, time_horizon);
    let risk1 = 1.0 - surv1.0;
    let risk2 = 1.0 - surv2.0;
    let arr = risk2 - risk1;
    let z = z_score_for_confidence(confidence_level);
    let arr_se = (surv1.1 + surv2.1).sqrt();
    let arr_ci_lower = arr - z * arr_se;
    let arr_ci_upper = arr + z * arr_se;
    let nnt = if arr.abs() > 1e-10 {
        1.0 / arr
    } else {
        f64::INFINITY
    };
    let (nnt_ci_lower, nnt_ci_upper) = if arr_ci_lower > 0.0 && arr_ci_upper > 0.0 {
        (1.0 / arr_ci_upper, 1.0 / arr_ci_lower)
    } else if arr_ci_lower < 0.0 && arr_ci_upper < 0.0 {
        (1.0 / arr_ci_lower, 1.0 / arr_ci_upper)
    } else {
        (f64::NEG_INFINITY, f64::INFINITY)
    };
    NNTResult {
        nnt,
        nnt_ci_lower,
        nnt_ci_upper,
        absolute_risk_reduction: arr,
        arr_ci_lower,
        arr_ci_upper,
        time_horizon,
    }
}
fn compute_survival_at_time(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    target_group: i32,
    t: f64,
) -> (f64, f64) {
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    if unique_groups.len() <= target_group as usize {
        return (1.0, 0.0);
    }
    let g = unique_groups[target_group as usize];
    let mut filtered_time = Vec::new();
    let mut filtered_status = Vec::new();
    for i in 0..time.len() {
        if group[i] == g {
            filtered_time.push(time[i]);
            filtered_status.push(status[i]);
        }
    }
    if filtered_time.is_empty() {
        return (1.0, 0.0);
    }
    let n = filtered_time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        filtered_time[a]
            .partial_cmp(&filtered_time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    let mut total_at_risk = n as f64;
    let mut i = 0;
    while i < n {
        let current_time = filtered_time[indices[i]];
        if current_time > t {
            break;
        }
        let mut events = 0.0;
        let mut removed = 0.0;
        while i < n && filtered_time[indices[i]] == current_time {
            removed += 1.0;
            if filtered_status[indices[i]] == 1 {
                events += 1.0;
            }
            i += 1;
        }
        if events > 0.0 && total_at_risk > 0.0 {
            surv *= 1.0 - events / total_at_risk;
            if total_at_risk > events {
                var_sum += events / (total_at_risk * (total_at_risk - events));
            }
        }
        total_at_risk -= removed;
    }
    let variance = surv * surv * var_sum;
    (surv, variance)
}
#[pyfunction]
#[pyo3(signature = (time, status, group, time_horizon, confidence_level=None))]
pub fn number_needed_to_treat(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    time_horizon: f64,
    confidence_level: Option<f64>,
) -> PyResult<NNTResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_nnt(&time, &status, &group, time_horizon, conf))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ChangepointInfo {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub hazard_before: f64,
    #[pyo3(get)]
    pub hazard_after: f64,
    #[pyo3(get)]
    pub likelihood_ratio: f64,
    #[pyo3(get)]
    pub p_value: f64,
}

#[pymethods]
impl ChangepointInfo {
    #[new]
    fn new(
        time: f64,
        hazard_before: f64,
        hazard_after: f64,
        likelihood_ratio: f64,
        p_value: f64,
    ) -> Self {
        Self {
            time,
            hazard_before,
            hazard_after,
            likelihood_ratio,
            p_value,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RMSTOptimalThresholdResult {
    #[pyo3(get)]
    pub optimal_tau: f64,
    #[pyo3(get)]
    pub max_followup: f64,
    #[pyo3(get)]
    pub changepoints: Vec<ChangepointInfo>,
    #[pyo3(get)]
    pub n_changepoints: usize,
    #[pyo3(get)]
    pub rmst_at_optimal: RMSTResult,
}

#[pymethods]
impl RMSTOptimalThresholdResult {
    #[new]
    fn new(
        optimal_tau: f64,
        max_followup: f64,
        changepoints: Vec<ChangepointInfo>,
        n_changepoints: usize,
        rmst_at_optimal: RMSTResult,
    ) -> Self {
        Self {
            optimal_tau,
            max_followup,
            changepoints,
            n_changepoints,
            rmst_at_optimal,
        }
    }
}

fn compute_piecewise_exp_likelihood(
    event_times: &[f64],
    censor_times: &[f64],
    changepoints: &[f64],
) -> f64 {
    if changepoints.is_empty() {
        let total_exposure: f64 = event_times.iter().chain(censor_times.iter()).sum();
        let n_events = event_times.len() as f64;
        if total_exposure <= 0.0 || n_events == 0.0 {
            return 0.0;
        }
        let lambda = n_events / total_exposure;
        return n_events * lambda.ln() - lambda * total_exposure;
    }
    let mut boundaries: Vec<f64> = vec![0.0];
    boundaries.extend(changepoints.iter().copied());
    boundaries.push(f64::INFINITY);
    let mut log_lik = 0.0;
    for i in 0..(boundaries.len() - 1) {
        let t_start = boundaries[i];
        let t_end = boundaries[i + 1];
        let mut n_events_interval = 0.0;
        let mut exposure_interval = 0.0;
        for &t in event_times {
            if t > t_start && t <= t_end {
                n_events_interval += 1.0;
            }
            let contribution = (t.min(t_end) - t_start).max(0.0);
            exposure_interval += contribution;
        }
        for &t in censor_times {
            let contribution = (t.min(t_end) - t_start).max(0.0);
            exposure_interval += contribution;
        }
        if exposure_interval > 0.0 && n_events_interval > 0.0 {
            let lambda = n_events_interval / exposure_interval;
            log_lik += n_events_interval * lambda.ln() - lambda * exposure_interval;
        }
    }
    log_lik
}

fn compute_hazard_in_interval(
    event_times: &[f64],
    censor_times: &[f64],
    t_start: f64,
    t_end: f64,
) -> f64 {
    let mut n_events = 0.0;
    let mut exposure = 0.0;
    for &t in event_times {
        if t > t_start && t <= t_end {
            n_events += 1.0;
        }
        let contribution = (t.min(t_end) - t_start).max(0.0);
        exposure += contribution;
    }
    for &t in censor_times {
        let contribution = (t.min(t_end) - t_start).max(0.0);
        exposure += contribution;
    }
    if exposure > 0.0 {
        n_events / exposure
    } else {
        0.0
    }
}

pub fn compute_rmst_optimal_threshold(
    time: &[f64],
    status: &[i32],
    alpha: f64,
    min_events_per_interval: usize,
    confidence_level: f64,
) -> RMSTOptimalThresholdResult {
    let n = time.len();
    if n == 0 {
        let empty_rmst = RMSTResult {
            rmst: 0.0,
            variance: 0.0,
            se: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            tau: 0.0,
        };
        return RMSTOptimalThresholdResult {
            optimal_tau: 0.0,
            max_followup: 0.0,
            changepoints: vec![],
            n_changepoints: 0,
            rmst_at_optimal: empty_rmst,
        };
    }
    let mut event_times: Vec<f64> = Vec::new();
    let mut censor_times: Vec<f64> = Vec::new();
    for i in 0..n {
        if status[i] == 1 {
            event_times.push(time[i]);
        } else {
            censor_times.push(time[i]);
        }
    }
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let max_followup = time.iter().fold(0.0_f64, |a, &b| a.max(b));
    if event_times.is_empty() {
        let rmst_result = compute_rmst(time, status, max_followup, confidence_level);
        return RMSTOptimalThresholdResult {
            optimal_tau: max_followup,
            max_followup,
            changepoints: vec![],
            n_changepoints: 0,
            rmst_at_optimal: rmst_result,
        };
    }
    let mut unique_event_times: Vec<f64> = event_times.clone();
    unique_event_times.dedup();
    let min_events = min_events_per_interval.max(2);
    let mut candidate_changepoints: Vec<f64> = Vec::new();
    let mut cumulative_events = 0usize;
    for &t in &unique_event_times {
        let events_at_t = event_times.iter().filter(|&&et| et == t).count();
        cumulative_events += events_at_t;
        let events_after = event_times.len() - cumulative_events;
        if cumulative_events >= min_events && events_after >= min_events {
            candidate_changepoints.push(t);
        }
    }
    if candidate_changepoints.is_empty() {
        let rmst_result = compute_rmst(time, status, max_followup, confidence_level);
        return RMSTOptimalThresholdResult {
            optimal_tau: max_followup,
            max_followup,
            changepoints: vec![],
            n_changepoints: 0,
            rmst_at_optimal: rmst_result,
        };
    }
    let null_likelihood = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
    let mut significant_changepoints: Vec<(f64, f64, f64)> = Vec::new();
    for &cp in &candidate_changepoints {
        let alt_likelihood = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[cp]);
        let lr_stat = 2.0 * (alt_likelihood - null_likelihood);
        if lr_stat > 0.0 {
            let p_value = 1.0 - chi_squared_cdf(lr_stat, 1.0);
            if p_value < alpha {
                significant_changepoints.push((cp, lr_stat, p_value));
            }
        }
    }
    let mut selected_changepoints: Vec<f64> = significant_changepoints
        .iter()
        .map(|&(cp, _, _)| cp)
        .collect();
    selected_changepoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    loop {
        if selected_changepoints.len() <= 1 {
            break;
        }
        let current_likelihood =
            compute_piecewise_exp_likelihood(&event_times, &censor_times, &selected_changepoints);
        let mut min_lr_drop = f64::INFINITY;
        let mut worst_idx = 0;
        for i in 0..selected_changepoints.len() {
            let mut reduced: Vec<f64> = selected_changepoints.clone();
            reduced.remove(i);
            let reduced_likelihood =
                compute_piecewise_exp_likelihood(&event_times, &censor_times, &reduced);
            let lr_drop = 2.0 * (current_likelihood - reduced_likelihood);
            if lr_drop < min_lr_drop {
                min_lr_drop = lr_drop;
                worst_idx = i;
            }
        }
        let p_value_drop = 1.0 - chi_squared_cdf(min_lr_drop, 1.0);
        if p_value_drop >= alpha {
            selected_changepoints.remove(worst_idx);
        } else {
            break;
        }
    }
    let mut changepoint_info: Vec<ChangepointInfo> = Vec::new();
    let mut boundaries: Vec<f64> = vec![0.0];
    boundaries.extend(selected_changepoints.iter().copied());
    boundaries.push(f64::INFINITY);
    for (i, &cp) in selected_changepoints.iter().enumerate() {
        let t_start_before = boundaries[i];
        let t_end_before = cp;
        let t_start_after = cp;
        let t_end_after = boundaries[i + 2];
        let hazard_before =
            compute_hazard_in_interval(&event_times, &censor_times, t_start_before, t_end_before);
        let hazard_after =
            compute_hazard_in_interval(&event_times, &censor_times, t_start_after, t_end_after);
        let (lr_stat, p_val) = significant_changepoints
            .iter()
            .find(|&&(c, _, _)| (c - cp).abs() < 1e-10)
            .map(|&(_, lr, p)| (lr, p))
            .unwrap_or((0.0, 1.0));
        changepoint_info.push(ChangepointInfo {
            time: cp,
            hazard_before,
            hazard_after,
            likelihood_ratio: lr_stat,
            p_value: p_val,
        });
    }
    let optimal_tau = if selected_changepoints.is_empty() {
        max_followup
    } else {
        *selected_changepoints.last().unwrap()
    };
    let rmst_at_optimal = compute_rmst(time, status, optimal_tau, confidence_level);
    RMSTOptimalThresholdResult {
        optimal_tau,
        max_followup,
        changepoints: changepoint_info,
        n_changepoints: selected_changepoints.len(),
        rmst_at_optimal,
    }
}

/// Compute optimal RMST threshold using reduced piecewise exponential model.
///
/// Uses the RPEXE approach (Han et al. 2025) to identify statistically
/// significant changepoints in the hazard function, then returns the largest
/// changepoint as the optimal tau for RMST computation.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// alpha : float, optional
///     Significance level for changepoint detection (default 0.05).
/// min_events_per_interval : int, optional
///     Minimum events required in each interval (default 5).
/// confidence_level : float, optional
///     Confidence level for RMST computation (default 0.95).
///
/// Returns
/// -------
/// RMSTOptimalThresholdResult
///     Object with: optimal_tau, max_followup, changepoints (list of
///     ChangepointInfo), n_changepoints, rmst_at_optimal.
#[pyfunction]
#[pyo3(signature = (time, status, alpha=None, min_events_per_interval=None, confidence_level=None))]
pub fn rmst_optimal_threshold(
    time: Vec<f64>,
    status: Vec<i32>,
    alpha: Option<f64>,
    min_events_per_interval: Option<usize>,
    confidence_level: Option<f64>,
) -> PyResult<RMSTOptimalThresholdResult> {
    let alpha = alpha.unwrap_or(0.05);
    let min_events = min_events_per_interval.unwrap_or(5);
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_rmst_optimal_threshold(
        &time, &status, alpha, min_events, conf,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rmst_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let tau = 5.0;

        let result = compute_rmst(&time, &status, tau, 0.95);

        assert!(result.rmst > 0.0);
        assert!(result.rmst <= tau);
        assert!(result.se >= 0.0);
        assert!(result.ci_lower <= result.rmst);
        assert!(result.ci_upper >= result.rmst);
        assert_eq!(result.tau, tau);
    }

    #[test]
    fn test_compute_rmst_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_rmst(&time, &status, 5.0, 0.95);

        assert_eq!(result.rmst, 0.0);
        assert_eq!(result.variance, 0.0);
    }

    #[test]
    fn test_compute_rmst_no_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = compute_rmst(&time, &status, 5.0, 0.95);

        assert_eq!(result.rmst, 5.0);
    }

    #[test]
    fn test_compare_rmst_two_groups() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 0];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);

        assert!(result.rmst_group1.rmst > 0.0);
        assert!(result.rmst_group2.rmst > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_compare_rmst_single_group() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 0];
        let group = vec![0, 0, 0];

        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);

        assert_eq!(result.rmst_diff, 0.0);
        assert_eq!(result.rmst_ratio, 1.0);
    }

    #[test]
    fn test_survival_quantile_median() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        assert!(result.median.is_some());
        assert_eq!(result.quantile, 0.5);
    }

    #[test]
    fn test_survival_quantile_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        assert!(result.median.is_none());
    }

    #[test]
    fn test_cumulative_incidence_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 2, 1, 0, 2];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(!result.time.is_empty());
        assert_eq!(result.event_types.len(), 2);
        assert_eq!(result.cif.len(), 2);
    }

    #[test]
    fn test_cumulative_incidence_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(result.time.is_empty());
        assert!(result.cif.is_empty());
    }

    #[test]
    fn test_cumulative_incidence_no_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(result.event_types.is_empty());
    }

    #[test]
    fn test_compute_nnt_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 1, 1, 0];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = compute_nnt(&time, &status, &group, 5.0, 0.95);

        assert!(result.nnt.is_finite() || result.nnt.is_infinite());
        assert_eq!(result.time_horizon, 5.0);
    }

    #[test]
    fn test_rmst_result_new() {
        let result = RMSTResult::new(3.5, 0.25, 0.5, 2.5, 4.5, 5.0);

        assert_eq!(result.rmst, 3.5);
        assert_eq!(result.variance, 0.25);
        assert_eq!(result.se, 0.5);
        assert_eq!(result.ci_lower, 2.5);
        assert_eq!(result.ci_upper, 4.5);
        assert_eq!(result.tau, 5.0);
    }

    #[test]
    fn test_chi_squared_cdf() {
        let cdf_0 = chi_squared_cdf(0.0, 1.0);
        assert!(cdf_0.abs() < 0.01);
        let cdf_small = chi_squared_cdf(0.5, 1.0);
        assert!(cdf_small > 0.0 && cdf_small < 1.0);
        let cdf_large = chi_squared_cdf(10.0, 1.0);
        assert!(cdf_large > 0.99);
    }

    #[test]
    fn test_piecewise_exp_likelihood_no_changepoints() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let censor_times = vec![2.5, 4.5];
        let lik = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
        assert!(lik.is_finite());
    }

    #[test]
    fn test_piecewise_exp_likelihood_with_changepoint() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let censor_times = vec![2.5, 7.5];
        let lik_null = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
        let lik_alt = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[5.0]);
        assert!(lik_null.is_finite());
        assert!(lik_alt.is_finite());
    }

    #[test]
    fn test_hazard_in_interval() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let censor_times = vec![2.5, 4.5];
        let hazard = compute_hazard_in_interval(&event_times, &censor_times, 0.0, 3.0);
        assert!(hazard >= 0.0);
    }

    #[test]
    fn test_rmst_optimal_threshold_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 0.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_rmst_optimal_threshold_no_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0, 0, 0, 0, 0];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 5.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_rmst_optimal_threshold_basic() {
        let mut time: Vec<f64> = Vec::new();
        let mut status: Vec<i32> = Vec::new();
        for i in 1..=20 {
            time.push(i as f64);
            status.push(1);
        }
        for i in 1..=10 {
            time.push(i as f64 + 0.5);
            status.push(0);
        }
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 3, 0.95);
        assert!(result.optimal_tau > 0.0);
        assert!(result.optimal_tau <= result.max_followup);
        assert!(result.rmst_at_optimal.rmst > 0.0);
    }

    #[test]
    fn test_rmst_optimal_threshold_few_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 5.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_changepoint_info_new() {
        let info = ChangepointInfo::new(5.0, 0.1, 0.2, 4.5, 0.03);
        assert_eq!(info.time, 5.0);
        assert_eq!(info.hazard_before, 0.1);
        assert_eq!(info.hazard_after, 0.2);
        assert_eq!(info.likelihood_ratio, 4.5);
        assert_eq!(info.p_value, 0.03);
    }

    #[test]
    fn test_rmst_optimal_threshold_result_new() {
        let rmst = RMSTResult::new(3.5, 0.25, 0.5, 2.5, 4.5, 5.0);
        let changepoints = vec![ChangepointInfo::new(3.0, 0.1, 0.2, 4.5, 0.03)];
        let result = RMSTOptimalThresholdResult::new(3.0, 5.0, changepoints, 1, rmst);
        assert_eq!(result.optimal_tau, 3.0);
        assert_eq!(result.max_followup, 5.0);
        assert_eq!(result.n_changepoints, 1);
        assert_eq!(result.changepoints.len(), 1);
    }
}
