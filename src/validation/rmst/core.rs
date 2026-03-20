
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
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
#[pyclass(from_py_object)]
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
    if n > PARALLEL_THRESHOLD_XLARGE {
        indices.par_sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
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
pub(crate) fn compare_rmst(
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
    let conf = confidence_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
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
    let conf = confidence_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    Ok(compare_rmst(&time, &status, &group, tau, conf))
}
