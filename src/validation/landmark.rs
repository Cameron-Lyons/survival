use crate::constants::PARALLEL_THRESHOLD_SMALL;
use crate::utilities::statistical::normal_cdf as norm_cdf;
use pyo3::prelude::*;
use rayon::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
pub struct LandmarkResult {
    #[pyo3(get)]
    pub landmark_time: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_excluded: usize,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub original_indices: Vec<usize>,
}
#[pymethods]
impl LandmarkResult {
    #[new]
    fn new(
        landmark_time: f64,
        n_at_risk: usize,
        n_excluded: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        original_indices: Vec<usize>,
    ) -> Self {
        Self {
            landmark_time,
            n_at_risk,
            n_excluded,
            time,
            status,
            original_indices,
        }
    }
}
pub fn compute_landmark(time: &[f64], status: &[i32], landmark_time: f64) -> LandmarkResult {
    let n = time.len();
    let mut new_time = Vec::new();
    let mut new_status = Vec::new();
    let mut original_indices = Vec::new();
    let mut n_excluded = 0usize;
    for i in 0..n {
        if time[i] > landmark_time {
            new_time.push(time[i] - landmark_time);
            new_status.push(status[i]);
            original_indices.push(i);
        } else {
            n_excluded += 1;
        }
    }
    let n_at_risk = new_time.len();
    LandmarkResult {
        landmark_time,
        n_at_risk,
        n_excluded,
        time: new_time,
        status: new_status,
        original_indices,
    }
}
#[pyfunction]
pub fn landmark_analysis(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_time: f64,
) -> PyResult<LandmarkResult> {
    Ok(compute_landmark(&time, &status, landmark_time))
}
pub fn compute_landmarks_parallel(
    time: &[f64],
    status: &[i32],
    landmark_times: &[f64],
) -> Vec<LandmarkResult> {
    landmark_times
        .par_iter()
        .map(|&lt| compute_landmark(time, status, lt))
        .collect()
}
#[pyfunction]
pub fn landmark_analysis_batch(
    time: Vec<f64>,
    status: Vec<i32>,
    landmark_times: Vec<f64>,
) -> PyResult<Vec<LandmarkResult>> {
    Ok(compute_landmarks_parallel(&time, &status, &landmark_times))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct ConditionalSurvivalResult {
    #[pyo3(get)]
    pub given_time: f64,
    #[pyo3(get)]
    pub target_time: f64,
    #[pyo3(get)]
    pub conditional_survival: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
}
#[pymethods]
impl ConditionalSurvivalResult {
    #[new]
    fn new(
        given_time: f64,
        target_time: f64,
        conditional_survival: f64,
        ci_lower: f64,
        ci_upper: f64,
        n_at_risk: usize,
    ) -> Self {
        Self {
            given_time,
            target_time,
            conditional_survival,
            ci_lower,
            ci_upper,
            n_at_risk,
        }
    }
}
pub fn compute_conditional_survival(
    time: &[f64],
    status: &[i32],
    given_time: f64,
    target_time: f64,
    confidence_level: f64,
) -> ConditionalSurvivalResult {
    let n = time.len();
    if n == 0 || target_time <= given_time {
        return ConditionalSurvivalResult {
            given_time,
            target_time,
            conditional_survival: 1.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            n_at_risk: 0,
        };
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut surv_given = 1.0;
    let mut surv_target = 1.0;
    let mut var_given = 0.0;
    let mut var_target = 0.0;
    let mut total_at_risk = n as f64;
    let mut n_at_given = 0usize;
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
            let hazard = events / total_at_risk;
            if current_time <= given_time {
                surv_given *= 1.0 - hazard;
                if total_at_risk > events {
                    var_given += events / (total_at_risk * (total_at_risk - events));
                }
            }
            if current_time <= target_time {
                surv_target *= 1.0 - hazard;
                if total_at_risk > events {
                    var_target += events / (total_at_risk * (total_at_risk - events));
                }
            }
        }
        if current_time <= given_time {
            n_at_given = (total_at_risk - removed) as usize;
        }
        total_at_risk -= removed;
    }
    let conditional = if surv_given > 0.0 {
        surv_target / surv_given
    } else {
        0.0
    };
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
    let var_conditional = if surv_given > 0.0 {
        conditional * conditional * (var_target - var_given).abs()
    } else {
        0.0
    };
    let se = var_conditional.sqrt();
    let ci_lower = (conditional - z * se).clamp(0.0, 1.0);
    let ci_upper = (conditional + z * se).clamp(0.0, 1.0);
    ConditionalSurvivalResult {
        given_time,
        target_time,
        conditional_survival: conditional,
        ci_lower,
        ci_upper,
        n_at_risk: n_at_given,
    }
}
#[pyfunction]
#[pyo3(signature = (time, status, given_time, target_time, confidence_level=None))]
pub fn conditional_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    given_time: f64,
    target_time: f64,
    confidence_level: Option<f64>,
) -> PyResult<ConditionalSurvivalResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_conditional_survival(
        &time,
        &status,
        given_time,
        target_time,
        conf,
    ))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct HazardRatioResult {
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub se_log_hr: f64,
    #[pyo3(get)]
    pub z_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
}
#[pymethods]
impl HazardRatioResult {
    #[new]
    fn new(
        hazard_ratio: f64,
        ci_lower: f64,
        ci_upper: f64,
        se_log_hr: f64,
        z_statistic: f64,
        p_value: f64,
    ) -> Self {
        Self {
            hazard_ratio,
            ci_lower,
            ci_upper,
            se_log_hr,
            z_statistic,
            p_value,
        }
    }
}
pub fn compute_hazard_ratio(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    confidence_level: f64,
) -> HazardRatioResult {
    let n = time.len();
    if n == 0 {
        return HazardRatioResult {
            hazard_ratio: 1.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            se_log_hr: 0.0,
            z_statistic: 0.0,
            p_value: 1.0,
        };
    }
    let mut unique_groups: Vec<i32> = group.to_vec();
    unique_groups.sort();
    unique_groups.dedup();
    if unique_groups.len() < 2 {
        return HazardRatioResult {
            hazard_ratio: 1.0,
            ci_lower: 1.0,
            ci_upper: 1.0,
            se_log_hr: 0.0,
            z_statistic: 0.0,
            p_value: 1.0,
        };
    }
    let g1 = unique_groups[0];
    let g2 = unique_groups[1];
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut n1_at_risk = 0.0;
    let mut n2_at_risk = 0.0;
    for &grp in group {
        if grp == g1 {
            n1_at_risk += 1.0;
        } else if grp == g2 {
            n2_at_risk += 1.0;
        }
    }
    let mut sum_o_e: f64 = 0.0;
    let mut sum_var: f64 = 0.0;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut d1 = 0.0;
        let mut d2 = 0.0;
        let mut r1 = 0.0;
        let mut r2 = 0.0;
        while i < n && time[indices[i]] == current_time {
            let idx = indices[i];
            if group[idx] == g1 {
                r1 += 1.0;
                if status[idx] == 1 {
                    d1 += 1.0;
                }
            } else if group[idx] == g2 {
                r2 += 1.0;
                if status[idx] == 1 {
                    d2 += 1.0;
                }
            }
            i += 1;
        }
        let d = d1 + d2;
        let y = n1_at_risk + n2_at_risk;
        if d > 0.0 && y > 1.0 {
            let e1 = d * n1_at_risk / y;
            sum_o_e += d1 - e1;
            let v = d * n1_at_risk * n2_at_risk * (y - d) / (y * y * (y - 1.0));
            sum_var += v;
        }
        n1_at_risk -= r1;
        n2_at_risk -= r2;
    }
    let log_hr: f64 = if sum_var > 0.0 {
        sum_o_e / sum_var
    } else {
        0.0
    };
    let hazard_ratio = log_hr.exp();
    let se_log_hr: f64 = if sum_var > 0.0 {
        1.0 / sum_var.sqrt()
    } else {
        0.0
    };
    let z: f64 = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
    let ci_lower = (log_hr - z * se_log_hr).exp();
    let ci_upper = (log_hr + z * se_log_hr).exp();
    let z_statistic: f64 = if se_log_hr > 0.0 {
        log_hr / se_log_hr
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - norm_cdf(z_statistic.abs()));
    HazardRatioResult {
        hazard_ratio,
        ci_lower,
        ci_upper,
        se_log_hr,
        z_statistic,
        p_value,
    }
}
/// Compute hazard ratio between two groups.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// group : array-like
///     Group indicator (0 or 1).
/// confidence_level : float, optional
///     Confidence level (default 0.95).
///
/// Returns
/// -------
/// HazardRatioResult
///     Object with: hazard_ratio, std_err, conf_lower, conf_upper, p_value.
#[pyfunction]
#[pyo3(signature = (time, status, group, confidence_level=None))]
pub fn hazard_ratio(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    confidence_level: Option<f64>,
) -> PyResult<HazardRatioResult> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_hazard_ratio(&time, &status, &group, conf))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvivalAtTimeResult {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub survival: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_events: usize,
}
#[pymethods]
impl SurvivalAtTimeResult {
    #[new]
    fn new(
        time: f64,
        survival: f64,
        ci_lower: f64,
        ci_upper: f64,
        n_at_risk: usize,
        n_events: usize,
    ) -> Self {
        Self {
            time,
            survival,
            ci_lower,
            ci_upper,
            n_at_risk,
            n_events,
        }
    }
}
pub fn compute_survival_at_times(
    time: &[f64],
    status: &[i32],
    eval_times: &[f64],
    confidence_level: f64,
) -> Vec<SurvivalAtTimeResult> {
    let n = time.len();
    if n == 0 {
        return eval_times
            .iter()
            .map(|&t| SurvivalAtTimeResult {
                time: t,
                survival: 1.0,
                ci_lower: 1.0,
                ci_upper: 1.0,
                n_at_risk: 0,
                n_events: 0,
            })
            .collect();
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut event_times: Vec<f64> = Vec::new();
    let mut survival_vals: Vec<f64> = Vec::new();
    let mut var_vals: Vec<f64> = Vec::new();
    let mut n_risk_vals: Vec<usize> = Vec::new();
    let mut cum_events: Vec<usize> = Vec::new();
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    let mut total_at_risk = n as f64;
    let mut total_events = 0usize;
    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut events = 0.0;
        let mut removed = 0.0;
        while i < n && time[indices[i]] == current_time {
            removed += 1.0;
            if status[indices[i]] == 1 {
                events += 1.0;
                total_events += 1;
            }
            i += 1;
        }
        if events > 0.0 && total_at_risk > 0.0 {
            surv *= 1.0 - events / total_at_risk;
            if total_at_risk > events {
                var_sum += events / (total_at_risk * (total_at_risk - events));
            }
            event_times.push(current_time);
            survival_vals.push(surv);
            var_vals.push(surv * surv * var_sum);
            n_risk_vals.push(total_at_risk as usize);
            cum_events.push(total_events);
        }
        total_at_risk -= removed;
    }
    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };
    let results: Vec<SurvivalAtTimeResult> = if eval_times.len() > PARALLEL_THRESHOLD_SMALL {
        eval_times
            .par_iter()
            .map(|&t| {
                let (survival, var, n_risk, n_ev) = if event_times.is_empty() || t < event_times[0]
                {
                    (1.0, 0.0, n, 0)
                } else {
                    let idx = event_times.partition_point(|&et| et <= t);
                    if idx == 0 {
                        (1.0, 0.0, n, 0)
                    } else {
                        (
                            survival_vals[idx - 1],
                            var_vals[idx - 1],
                            n_risk_vals[idx - 1],
                            cum_events[idx - 1],
                        )
                    }
                };
                let se = var.sqrt();
                let ci_lower = (survival - z * se).clamp(0.0, 1.0);
                let ci_upper = (survival + z * se).clamp(0.0, 1.0);
                SurvivalAtTimeResult {
                    time: t,
                    survival,
                    ci_lower,
                    ci_upper,
                    n_at_risk: n_risk,
                    n_events: n_ev,
                }
            })
            .collect()
    } else {
        let mut results = Vec::with_capacity(eval_times.len());
        for &t in eval_times {
            let (survival, var, n_risk, n_ev) = if event_times.is_empty() || t < event_times[0] {
                (1.0, 0.0, n, 0)
            } else {
                let idx = event_times.partition_point(|&et| et <= t);
                if idx == 0 {
                    (1.0, 0.0, n, 0)
                } else {
                    (
                        survival_vals[idx - 1],
                        var_vals[idx - 1],
                        n_risk_vals[idx - 1],
                        cum_events[idx - 1],
                    )
                }
            };
            let se = var.sqrt();
            let ci_lower = (survival - z * se).clamp(0.0, 1.0);
            let ci_upper = (survival + z * se).clamp(0.0, 1.0);
            results.push(SurvivalAtTimeResult {
                time: t,
                survival,
                ci_lower,
                ci_upper,
                n_at_risk: n_risk,
                n_events: n_ev,
            });
        }
        results
    };
    results
}
#[pyfunction]
#[pyo3(signature = (time, status, eval_times, confidence_level=None))]
pub fn survival_at_times(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Vec<f64>,
    confidence_level: Option<f64>,
) -> PyResult<Vec<SurvivalAtTimeResult>> {
    let conf = confidence_level.unwrap_or(0.95);
    Ok(compute_survival_at_times(&time, &status, &eval_times, conf))
}
#[derive(Debug, Clone)]
#[pyclass]
pub struct LifeTableResult {
    #[pyo3(get)]
    pub interval_start: Vec<f64>,
    #[pyo3(get)]
    pub interval_end: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_deaths: Vec<f64>,
    #[pyo3(get)]
    pub n_censored: Vec<f64>,
    #[pyo3(get)]
    pub n_effective: Vec<f64>,
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub se_survival: Vec<f64>,
}
#[pymethods]
impl LifeTableResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        interval_start: Vec<f64>,
        interval_end: Vec<f64>,
        n_at_risk: Vec<f64>,
        n_deaths: Vec<f64>,
        n_censored: Vec<f64>,
        n_effective: Vec<f64>,
        hazard: Vec<f64>,
        survival: Vec<f64>,
        se_survival: Vec<f64>,
    ) -> Self {
        Self {
            interval_start,
            interval_end,
            n_at_risk,
            n_deaths,
            n_censored,
            n_effective,
            hazard,
            survival,
            se_survival,
        }
    }
}
pub fn compute_life_table(time: &[f64], status: &[i32], breaks: &[f64]) -> LifeTableResult {
    let n = time.len();
    let n_intervals = breaks.len().saturating_sub(1);
    if n == 0 || n_intervals == 0 {
        return LifeTableResult {
            interval_start: vec![],
            interval_end: vec![],
            n_at_risk: vec![],
            n_deaths: vec![],
            n_censored: vec![],
            n_effective: vec![],
            hazard: vec![],
            survival: vec![],
            se_survival: vec![],
        };
    }
    let mut interval_start = Vec::with_capacity(n_intervals);
    let mut interval_end = Vec::with_capacity(n_intervals);
    let mut n_deaths = vec![0.0; n_intervals];
    let mut n_censored = vec![0.0; n_intervals];
    for i in 0..n_intervals {
        interval_start.push(breaks[i]);
        interval_end.push(breaks[i + 1]);
    }
    for i in 0..n {
        let t = time[i];
        for j in 0..n_intervals {
            if t >= breaks[j] && t < breaks[j + 1] {
                if status[i] == 1 {
                    n_deaths[j] += 1.0;
                } else {
                    n_censored[j] += 1.0;
                }
                break;
            }
        }
    }
    let mut n_at_risk = Vec::with_capacity(n_intervals);
    let mut remaining = n as f64;
    for j in 0..n_intervals {
        n_at_risk.push(remaining);
        remaining -= n_deaths[j] + n_censored[j];
    }
    let n_effective: Vec<f64> = (0..n_intervals)
        .map(|j| n_at_risk[j] - n_censored[j] / 2.0)
        .collect();
    let hazard: Vec<f64> = (0..n_intervals)
        .map(|j| {
            if n_effective[j] > 0.0 {
                n_deaths[j] / n_effective[j]
            } else {
                0.0
            }
        })
        .collect();
    let mut survival = Vec::with_capacity(n_intervals);
    let mut se_survival = Vec::with_capacity(n_intervals);
    let mut surv = 1.0;
    let mut var_sum = 0.0;
    for j in 0..n_intervals {
        surv *= 1.0 - hazard[j];
        survival.push(surv);
        if n_effective[j] > 0.0 && n_effective[j] > n_deaths[j] {
            var_sum += n_deaths[j] / (n_effective[j] * (n_effective[j] - n_deaths[j]));
        }
        se_survival.push(surv * var_sum.sqrt());
    }
    LifeTableResult {
        interval_start,
        interval_end,
        n_at_risk,
        n_deaths,
        n_censored,
        n_effective,
        hazard,
        survival,
        se_survival,
    }
}
#[pyfunction]
pub fn life_table(time: Vec<f64>, status: Vec<i32>, breaks: Vec<f64>) -> PyResult<LifeTableResult> {
    Ok(compute_life_table(&time, &status, &breaks))
}
