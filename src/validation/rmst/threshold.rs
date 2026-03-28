
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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

pub(crate) fn compute_rmst_optimal_threshold(
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
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
            let p_value = 1.0 - chi2_cdf(lr_stat, 1.0);
            if p_value < alpha {
                significant_changepoints.push((cp, lr_stat, p_value));
            }
        }
    }
    let mut selected_changepoints: Vec<f64> = significant_changepoints
        .iter()
        .map(|&(cp, _, _)| cp)
        .collect();
    selected_changepoints.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let p_value_drop = 1.0 - chi2_cdf(min_lr_drop, 1.0);
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
        selected_changepoints[selected_changepoints.len() - 1]
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
    let conf = confidence_level.unwrap_or(DEFAULT_CONFIDENCE_LEVEL);
    Ok(compute_rmst_optimal_threshold(
        &time, &status, alpha, min_events, conf,
    ))
}
