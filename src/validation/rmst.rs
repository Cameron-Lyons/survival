use crate::utilities::statistical::normal_cdf as norm_cdf;
use pyo3::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
pub struct RMSTResult {
    #[pyo3(get)]
    pub rmst: f64,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub tau: f64,
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
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
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
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
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
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
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
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
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
