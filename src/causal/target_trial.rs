#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct TargetTrialResult {
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub hr_ci_lower: f64,
    #[pyo3(get)]
    pub hr_ci_upper: f64,
    #[pyo3(get)]
    pub risk_difference: f64,
    #[pyo3(get)]
    pub rd_ci_lower: f64,
    #[pyo3(get)]
    pub rd_ci_upper: f64,
    #[pyo3(get)]
    pub survival_treated: Vec<f64>,
    #[pyo3(get)]
    pub survival_control: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub n_eligible: usize,
    #[pyo3(get)]
    pub n_treated: usize,
    #[pyo3(get)]
    pub n_control: usize,
    #[pyo3(get)]
    pub n_clones: usize,
    #[pyo3(get)]
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TrialEmulationConfig {
    #[pyo3(get, set)]
    pub grace_period: f64,
    #[pyo3(get, set)]
    pub max_followup: f64,
    #[pyo3(get, set)]
    pub clone_censor_weighting: bool,
    #[pyo3(get, set)]
    pub stabilized_weights: bool,
    #[pyo3(get, set)]
    pub trim_weights: f64,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
}

#[pymethods]
impl TrialEmulationConfig {
    #[new]
    #[pyo3(signature = (grace_period=0.0, max_followup=None, clone_censor_weighting=true, stabilized_weights=true, trim_weights=0.01, n_bootstrap=200))]
    pub fn new(
        grace_period: f64,
        max_followup: Option<f64>,
        clone_censor_weighting: bool,
        stabilized_weights: bool,
        trim_weights: f64,
        n_bootstrap: usize,
    ) -> Self {
        TrialEmulationConfig {
            grace_period,
            max_followup: max_followup.unwrap_or(f64::INFINITY),
            clone_censor_weighting,
            stabilized_weights,
            trim_weights,
            n_bootstrap,
        }
    }
}

#[derive(Debug, Clone)]
struct ClonedObservation {
    original_id: usize,
    clone_type: i32,
    baseline_time: f64,
    followup_time: f64,
    event: i32,
    censored_by_protocol: bool,
    covariates: Vec<f64>,
    weight: f64,
}

fn create_clones(
    subject_id: usize,
    time: &[f64],
    status: &[i32],
    treatment_time: Option<f64>,
    x: &[f64],
    n_vars: usize,
    config: &TrialEmulationConfig,
) -> Vec<ClonedObservation> {
    let mut clones = Vec::new();

    let covariates: Vec<f64> = (0..n_vars).map(|j| x[subject_id * n_vars + j]).collect();
    let obs_time = time[subject_id];
    let obs_status = status[subject_id];

    let treated_clone = ClonedObservation {
        original_id: subject_id,
        clone_type: 1,
        baseline_time: 0.0,
        followup_time: match treatment_time {
            Some(t_time) if t_time <= config.grace_period => {
                (obs_time - t_time).min(config.max_followup)
            }
            Some(t_time) => t_time.min(config.max_followup),
            None => obs_time.min(config.max_followup),
        },
        event: match treatment_time {
            Some(t_time) if t_time <= config.grace_period => obs_status,
            Some(_) => 0,
            None => 0,
        },
        censored_by_protocol: treatment_time
            .map(|t| t > config.grace_period)
            .unwrap_or(true),
        covariates: covariates.clone(),
        weight: 1.0,
    };
    clones.push(treated_clone);

    let control_clone = ClonedObservation {
        original_id: subject_id,
        clone_type: 0,
        baseline_time: 0.0,
        followup_time: match treatment_time {
            Some(t_time) => t_time.min(obs_time).min(config.max_followup),
            None => obs_time.min(config.max_followup),
        },
        event: match treatment_time {
            Some(t_time) if t_time >= obs_time => obs_status,
            Some(_) => 0,
            None => obs_status,
        },
        censored_by_protocol: treatment_time.map(|t| t < obs_time).unwrap_or(false),
        covariates,
        weight: 1.0,
    };
    clones.push(control_clone);

    clones
}

fn compute_censoring_weights(
    clones: &mut [ClonedObservation],
    x_censoring: &[f64],
    n_vars_censoring: usize,
    config: &TrialEmulationConfig,
) {
    if !config.clone_censor_weighting {
        return;
    }

    let censored_times: Vec<f64> = clones
        .iter()
        .filter(|c| c.censored_by_protocol)
        .map(|c| c.followup_time)
        .collect();

    if censored_times.is_empty() {
        return;
    }

    let mut unique_times = censored_times.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    for clone in clones.iter_mut() {
        if clone.censored_by_protocol {
            clone.weight = 1.0 / config.trim_weights.max(0.1);
        }
    }

    if config.stabilized_weights {
        let total_weight: f64 = clones.iter().map(|c| c.weight).sum();
        let n = clones.len() as f64;
        for clone in clones.iter_mut() {
            clone.weight *= n / total_weight;
        }
    }
}

fn weighted_kaplan_meier(
    clones: &[ClonedObservation],
    treatment_arm: i32,
    time_points: &[f64],
) -> Vec<f64> {
    let arm_clones: Vec<&ClonedObservation> = clones
        .iter()
        .filter(|c| c.clone_type == treatment_arm)
        .collect();

    if arm_clones.is_empty() {
        return vec![1.0; time_points.len()];
    }

    let mut events: std::collections::BTreeMap<i64, (f64, f64)> = std::collections::BTreeMap::new();

    for clone in &arm_clones {
        let key = (clone.followup_time * 1e6) as i64;
        let entry = events.entry(key).or_insert((0.0, 0.0));
        entry.1 += clone.weight;
        if clone.event == 1 {
            entry.0 += clone.weight;
        }
    }

    let total_weight: f64 = arm_clones.iter().map(|c| c.weight).sum();
    let mut at_risk = total_weight;
    let mut cum_surv = 1.0;

    let mut survival = Vec::with_capacity(time_points.len());
    let event_times: Vec<i64> = events.keys().copied().collect();
    let mut event_idx = 0;

    for &t in time_points {
        let t_key = (t * 1e6) as i64;

        while event_idx < event_times.len() && event_times[event_idx] <= t_key {
            let key = event_times[event_idx];
            if let Some(&(d, n)) = events.get(&key) {
                if at_risk > 0.0 {
                    cum_surv *= 1.0 - d / at_risk;
                }
                at_risk -= n;
            }
            event_idx += 1;
        }

        survival.push(cum_surv);
    }

    survival
}

#[pyfunction]
#[pyo3(signature = (time, status, treatment_time, x_baseline, x_censoring, n_obs, n_vars_baseline, n_vars_censoring, config))]
pub fn target_trial_emulation(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment_time: Vec<Option<f64>>,
    x_baseline: Vec<f64>,
    x_censoring: Vec<f64>,
    n_obs: usize,
    n_vars_baseline: usize,
    n_vars_censoring: usize,
    config: &TrialEmulationConfig,
) -> PyResult<TargetTrialResult> {
    if time.len() != n_obs || status.len() != n_obs || treatment_time.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }

    let mut all_clones: Vec<ClonedObservation> = (0..n_obs)
        .into_par_iter()
        .flat_map(|i| {
            create_clones(
                i,
                &time,
                &status,
                treatment_time[i],
                &x_baseline,
                n_vars_baseline,
                config,
            )
        })
        .collect();

    compute_censoring_weights(&mut all_clones, &x_censoring, n_vars_censoring, config);

    let n_clones = all_clones.len();
    let n_treated: usize = all_clones.iter().filter(|c| c.clone_type == 1).count();
    let n_control = n_clones - n_treated;

    let max_time = all_clones
        .iter()
        .map(|c| c.followup_time)
        .fold(0.0, f64::max);
    let time_points: Vec<f64> = (0..100)
        .map(|i| max_time * (i as f64 + 1.0) / 100.0)
        .collect();

    let survival_treated = weighted_kaplan_meier(&all_clones, 1, &time_points);
    let survival_control = weighted_kaplan_meier(&all_clones, 0, &time_points);

    let tau = max_time;
    let rmst_treated = compute_rmst(&time_points, &survival_treated, tau);
    let rmst_control = compute_rmst(&time_points, &survival_control, tau);
    let risk_difference = rmst_treated - rmst_control;

    let hazard_ratio = estimate_hazard_ratio(&all_clones, n_vars_baseline);

    let (hr_se, rd_se) = bootstrap_standard_errors(
        &time,
        &status,
        &treatment_time,
        &x_baseline,
        &x_censoring,
        n_obs,
        n_vars_baseline,
        n_vars_censoring,
        config,
    );

    let z = 1.96;
    let hr_ci_lower = (hazard_ratio.ln() - z * hr_se).exp();
    let hr_ci_upper = (hazard_ratio.ln() + z * hr_se).exp();
    let rd_ci_lower = risk_difference - z * rd_se;
    let rd_ci_upper = risk_difference + z * rd_se;

    let weights: Vec<f64> = all_clones.iter().map(|c| c.weight).collect();

    Ok(TargetTrialResult {
        hazard_ratio,
        hr_ci_lower,
        hr_ci_upper,
        risk_difference,
        rd_ci_lower,
        rd_ci_upper,
        survival_treated,
        survival_control,
        time_points,
        n_eligible: n_obs,
        n_treated,
        n_control,
        n_clones,
        weights,
    })
}

fn compute_rmst(time_points: &[f64], survival: &[f64], tau: f64) -> f64 {
    if time_points.is_empty() {
        return 0.0;
    }

    let mut rmst = 0.0;
    let mut prev_time = 0.0;
    let mut prev_surv = 1.0;

    for (&t, &s) in time_points.iter().zip(survival.iter()) {
        if t > tau {
            rmst += prev_surv * (tau - prev_time);
            break;
        }
        rmst += prev_surv * (t - prev_time);
        prev_time = t;
        prev_surv = s;
    }

    rmst
}

fn estimate_hazard_ratio(clones: &[ClonedObservation], n_vars: usize) -> f64 {
    let mut beta = 0.0;

    for _ in 0..50 {
        let mut gradient = 0.0;
        let mut hessian = 0.0;

        let mut sorted_clones: Vec<&ClonedObservation> = clones.iter().collect();
        sorted_clones.sort_by(|a, b| {
            b.followup_time
                .partial_cmp(&a.followup_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_sum = 0.0;
        let mut weighted_trt = 0.0;

        for clone in &sorted_clones {
            let exp_eta = (beta * clone.clone_type as f64).exp() * clone.weight;
            risk_sum += exp_eta;
            weighted_trt += exp_eta * clone.clone_type as f64;

            if clone.event == 1 && risk_sum > 0.0 {
                let trt_bar = weighted_trt / risk_sum;
                gradient += clone.weight * (clone.clone_type as f64 - trt_bar);
                hessian += clone.weight * trt_bar * (1.0 - trt_bar);
            }
        }

        if hessian.abs() > 1e-10 {
            beta += gradient / hessian;
        }
    }

    beta.exp()
}

fn bootstrap_standard_errors(
    time: &[f64],
    status: &[i32],
    treatment_time: &[Option<f64>],
    x_baseline: &[f64],
    x_censoring: &[f64],
    n_obs: usize,
    n_vars_baseline: usize,
    n_vars_censoring: usize,
    config: &TrialEmulationConfig,
) -> (f64, f64) {
    let results: Vec<(f64, f64)> = (0..config.n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = fastrand::Rng::with_seed(b as u64 + 54321);
            let indices: Vec<usize> = (0..n_obs).map(|_| rng.usize(0..n_obs)).collect();

            let mut all_clones: Vec<ClonedObservation> = Vec::new();
            for &i in &indices {
                let boot_x: Vec<f64> = (0..n_vars_baseline)
                    .map(|j| x_baseline[i * n_vars_baseline + j])
                    .collect();
                let clones = create_clones(
                    0,
                    &[time[i]],
                    &[status[i]],
                    treatment_time[i],
                    &boot_x,
                    n_vars_baseline,
                    config,
                );
                for mut c in clones {
                    c.original_id = i;
                    all_clones.push(c);
                }
            }

            if all_clones.is_empty() {
                return None;
            }

            let hr = estimate_hazard_ratio(&all_clones, n_vars_baseline);

            let max_time = all_clones
                .iter()
                .map(|c| c.followup_time)
                .fold(0.0, f64::max);
            let time_points: Vec<f64> = (0..50)
                .map(|i| max_time * (i as f64 + 1.0) / 50.0)
                .collect();

            let surv_t = weighted_kaplan_meier(&all_clones, 1, &time_points);
            let surv_c = weighted_kaplan_meier(&all_clones, 0, &time_points);

            let rmst_t = compute_rmst(&time_points, &surv_t, max_time);
            let rmst_c = compute_rmst(&time_points, &surv_c, max_time);
            let rd = rmst_t - rmst_c;

            Some((hr.ln(), rd))
        })
        .collect();

    if results.len() < 2 {
        return (0.5, 0.5);
    }

    let mean_log_hr = results.iter().map(|(h, _)| h).sum::<f64>() / results.len() as f64;
    let mean_rd = results.iter().map(|(_, r)| r).sum::<f64>() / results.len() as f64;

    let var_log_hr = results
        .iter()
        .map(|(h, _)| (h - mean_log_hr).powi(2))
        .sum::<f64>()
        / (results.len() - 1) as f64;
    let var_rd = results
        .iter()
        .map(|(_, r)| (r - mean_rd).powi(2))
        .sum::<f64>()
        / (results.len() - 1) as f64;

    (var_log_hr.sqrt(), var_rd.sqrt())
}

#[pyfunction]
#[pyo3(signature = (enrollment_times, treatment_times, event_times, event_status, x_baseline, n_obs, n_vars, trial_starts))]
pub fn sequential_trial_emulation(
    enrollment_times: Vec<f64>,
    treatment_times: Vec<Option<f64>>,
    event_times: Vec<f64>,
    event_status: Vec<i32>,
    x_baseline: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    trial_starts: Vec<f64>,
) -> PyResult<Vec<TargetTrialResult>> {
    let mut results = Vec::new();

    for &trial_start in &trial_starts {
        let eligible: Vec<usize> = (0..n_obs)
            .filter(|&i| {
                enrollment_times[i] <= trial_start
                    && event_times[i] > trial_start
                    && treatment_times[i].map(|t| t >= trial_start).unwrap_or(true)
            })
            .collect();

        if eligible.len() < 10 {
            continue;
        }

        let time: Vec<f64> = eligible
            .iter()
            .map(|&i| event_times[i] - trial_start)
            .collect();
        let status: Vec<i32> = eligible.iter().map(|&i| event_status[i]).collect();
        let treatment_time: Vec<Option<f64>> = eligible
            .iter()
            .map(|&i| treatment_times[i].map(|t| t - trial_start))
            .collect();
        let x: Vec<f64> = {
            let mut result = Vec::with_capacity(eligible.len() * n_vars);
            for &i in &eligible {
                for j in 0..n_vars {
                    result.push(x_baseline[i * n_vars + j]);
                }
            }
            result
        };

        let config = TrialEmulationConfig::new(0.0, None, true, true, 0.01, 50);

        if let Ok(result) = target_trial_emulation(
            time,
            status,
            treatment_time,
            x.clone(),
            x,
            eligible.len(),
            n_vars,
            n_vars,
            &config,
        ) {
            results.push(result);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TrialEmulationConfig::new(0.0, Some(365.0), true, true, 0.01, 100);
        assert_eq!(config.grace_period, 0.0);
        assert_eq!(config.max_followup, 365.0);
    }

    #[test]
    fn test_target_trial_basic() {
        let time = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let status = vec![1, 0, 1, 0, 1, 0];
        let treatment_time = vec![Some(5.0), None, Some(10.0), None, Some(2.0), None];
        let x_baseline = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];

        let config = TrialEmulationConfig::new(7.0, Some(100.0), false, true, 0.01, 10);

        let result = target_trial_emulation(
            time,
            status,
            treatment_time,
            x_baseline.clone(),
            x_baseline,
            6,
            1,
            1,
            &config,
        )
        .unwrap();

        assert!(result.n_clones > 0);
        assert!(result.hazard_ratio > 0.0);
    }
}
