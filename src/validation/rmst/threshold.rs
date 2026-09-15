//! Data-driven choice of the truncation time of a restricted mean: a
//! piecewise-exponential changepoint search on the hazard (no R
//! counterpart).  Times are binned with `aeqSurv` first, as `survfit`
//! does, and the restricted mean at the chosen horizon is
//! `summary(survfit(Surv(time, status) ~ 1), rmean = tau)$table`.

use super::kaplan_meier;
use crate::data_prep::aeq_times;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pchisq;
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::surv_analysis::{RmeanOption, survmean};
use pyo3::prelude::*;

/// A hazard changepoint retained by the search.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct ChangepointInfo {
    pub time: f64,
    pub hazard_before: f64,
    pub hazard_after: f64,
    pub likelihood_ratio: f64,
    pub p_value: f64,
}

/// The chosen horizon and the restricted mean survival time up to it.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct RMSTOptimalThresholdResult {
    pub optimal_tau: f64,
    pub max_followup: f64,
    pub changepoints: Vec<ChangepointInfo>,
    pub rmean: f64,
    pub se_rmean: f64,
}

/// Restricted mean up to `tau` and its standard error from the
/// Kaplan-Meier curve.
fn restricted_mean(time: &[f64], status: &[i32], tau: f64) -> SurvivalResult<(f64, f64)> {
    let km = kaplan_meier(time, status, None, 0.95)?;
    let table = survmean(&km, 1.0, RmeanOption::At(tau))?;
    Ok((
        table.rmean.as_ref().map_or(f64::NAN, |v| v[0]),
        table.se_rmean.as_ref().map_or(f64::NAN, |v| v[0]),
    ))
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

/// Search for hazard changepoints and report the restricted mean up to
/// the last one retained (or the maximum follow-up when none is).
pub fn rmst_optimal_threshold(
    time: &[f64],
    status: &[i32],
    alpha: f64,
    min_events_per_interval: usize,
) -> SurvivalResult<RMSTOptimalThresholdResult> {
    let n = time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, status.len(), "status")?;
    validate_finite(time, "time")?;
    validate_binary_i32(status, "status")?;
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(SurvivalError::invalid_input(
            "alpha must be greater than 0 and less than 1",
        ));
    }
    if min_events_per_interval < 2 {
        return Err(SurvivalError::invalid_input(
            "min_events_per_interval must be at least 2",
        ));
    }
    let time = aeq_times(time);
    let time = time.as_slice();
    let mut event_times: Vec<f64> = Vec::new();
    let mut censor_times: Vec<f64> = Vec::new();
    for i in 0..n {
        if status[i] == 1 {
            event_times.push(time[i]);
        } else {
            censor_times.push(time[i]);
        }
    }
    event_times.sort_by(f64::total_cmp);
    let max_followup = time.iter().fold(0.0_f64, |a, &b| a.max(b));
    if event_times.is_empty() {
        let (rmean, se_rmean) = restricted_mean(time, status, max_followup)?;
        return Ok(RMSTOptimalThresholdResult {
            optimal_tau: max_followup,
            max_followup,
            changepoints: vec![],
            rmean,
            se_rmean,
        });
    }
    let mut unique_event_times: Vec<f64> = event_times.clone();
    unique_event_times.dedup();
    let min_events = min_events_per_interval;
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
        let (rmean, se_rmean) = restricted_mean(time, status, max_followup)?;
        return Ok(RMSTOptimalThresholdResult {
            optimal_tau: max_followup,
            max_followup,
            changepoints: vec![],
            rmean,
            se_rmean,
        });
    }
    let null_likelihood = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
    let mut significant_changepoints: Vec<(f64, f64, f64)> = Vec::new();
    for &cp in &candidate_changepoints {
        let alt_likelihood = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[cp]);
        let lr_stat = 2.0 * (alt_likelihood - null_likelihood);
        if lr_stat > 0.0 {
            let p_value = pchisq(lr_stat, 1.0, false, false);
            if p_value < alpha {
                significant_changepoints.push((cp, lr_stat, p_value));
            }
        }
    }
    let mut selected_changepoints: Vec<f64> = significant_changepoints
        .iter()
        .map(|&(cp, _, _)| cp)
        .collect();
    selected_changepoints.sort_by(f64::total_cmp);
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
        let p_value_drop = pchisq(min_lr_drop, 1.0, false, false);
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
            .find(|&&(c, _, _)| c == cp)
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
    let (rmean, se_rmean) = restricted_mean(time, status, optimal_tau)?;
    Ok(RMSTOptimalThresholdResult {
        optimal_tau,
        max_followup,
        changepoints: changepoint_info,
        rmean,
        se_rmean,
    })
}

#[pyfunction(name = "rmst_optimal_threshold")]
#[pyo3(signature = (time, status, alpha=0.05, min_events_per_interval=5))]
pub fn rmst_optimal_threshold_py(
    time: Vec<f64>,
    status: Vec<i32>,
    alpha: f64,
    min_events_per_interval: usize,
) -> PyResult<RMSTOptimalThresholdResult> {
    Ok(rmst_optimal_threshold(
        &time,
        &status,
        alpha,
        min_events_per_interval,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piecewise_exponential_likelihood_prefers_a_real_changepoint() {
        let event_times = [1.0, 1.5, 2.0, 2.5, 3.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let censor_times = [20.0, 21.0];
        let null = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
        let split = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[3.0]);
        assert!(split > null);
        let hazard = compute_hazard_in_interval(&event_times, &censor_times, 0.0, 3.0);
        assert!(hazard > 0.0);
    }

    #[test]
    fn threshold_search_returns_a_restricted_mean() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1];
        let result = rmst_optimal_threshold(&time, &status, 0.05, 2).unwrap();
        assert_eq!(result.max_followup, 10.0);
        assert!(result.optimal_tau > 0.0 && result.optimal_tau <= 10.0);
        assert!(result.rmean > 0.0 && result.rmean <= result.optimal_tau);
        assert!(result.se_rmean >= 0.0);
        assert!(rmst_optimal_threshold(&[], &[], 0.05, 2).is_err());
        assert!(rmst_optimal_threshold(&time, &status, 0.05, 1).is_err());
    }

    #[test]
    fn no_events_use_the_maximum_follow_up() {
        let result = rmst_optimal_threshold(&[1.0, 2.0, 3.0], &[0, 0, 0], 0.05, 2).unwrap();
        assert_eq!(result.optimal_tau, 3.0);
        assert!((result.rmean - 3.0).abs() < 1e-12);
        assert!(result.changepoints.is_empty());
    }
}
