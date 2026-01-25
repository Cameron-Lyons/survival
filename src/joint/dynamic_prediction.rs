#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use crate::utilities::statistical::sample_normal;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DynamicPredictionResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival_mean: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_risk: Vec<f64>,
    #[pyo3(get)]
    pub conditional_survival: Vec<f64>,
    #[pyo3(get)]
    pub auc: f64,
    #[pyo3(get)]
    pub brier_score: f64,
}

#[pyfunction]
#[pyo3(signature = (
    beta_long,
    gamma_surv,
    alpha,
    random_effects,
    baseline_hazard,
    baseline_times,
    y_history,
    times_history,
    x_long_fixed,
    n_history,
    n_long_vars,
    x_surv,
    n_surv_vars,
    landmark_time,
    prediction_times,
    n_monte_carlo=500
))]
pub fn dynamic_prediction(
    beta_long: Vec<f64>,
    gamma_surv: Vec<f64>,
    alpha: f64,
    random_effects: Vec<f64>,
    baseline_hazard: Vec<f64>,
    baseline_times: Vec<f64>,
    y_history: Vec<f64>,
    times_history: Vec<f64>,
    x_long_fixed: Vec<f64>,
    n_history: usize,
    n_long_vars: usize,
    x_surv: Vec<f64>,
    n_surv_vars: usize,
    landmark_time: f64,
    prediction_times: Vec<f64>,
    n_monte_carlo: usize,
) -> PyResult<DynamicPredictionResult> {
    let b0 = random_effects.first().copied().unwrap_or(0.0);
    let b1 = random_effects.get(1).copied().unwrap_or(0.0);

    let prediction_times_filtered: Vec<f64> = prediction_times
        .into_iter()
        .filter(|&t| t > landmark_time)
        .collect();

    let n_times = prediction_times_filtered.len();

    let survival_samples: Vec<Vec<f64>> = (0..n_monte_carlo)
        .into_par_iter()
        .map(|mc_idx| {
            let mut rng = fastrand::Rng::with_seed(mc_idx as u64);

            let b0_sample = b0 + 0.1 * sample_normal(&mut rng);
            let b1_sample = b1 + 0.05 * sample_normal(&mut rng);

            prediction_times_filtered
                .iter()
                .map(|&t| {
                    let mut eta = 0.0;
                    for (k, &xk) in x_surv.iter().enumerate() {
                        if k < gamma_surv.len() {
                            eta += gamma_surv[k] * xk;
                        }
                    }

                    let mut m_t = b0_sample + b1_sample * t;
                    let x_avg: Vec<f64> = (0..n_long_vars)
                        .map(|j| {
                            (0..n_history)
                                .map(|i| x_long_fixed[i * n_long_vars + j])
                                .sum::<f64>()
                                / n_history.max(1) as f64
                        })
                        .collect();

                    for (j, &xj) in x_avg.iter().enumerate() {
                        if j < beta_long.len() {
                            m_t += beta_long[j] * xj;
                        }
                    }

                    eta += alpha * m_t;

                    let mut cum_hazard = 0.0;
                    for (t_idx, &bt) in baseline_times.iter().enumerate() {
                        if bt > landmark_time && bt <= t && t_idx < baseline_hazard.len() {
                            cum_hazard += baseline_hazard[t_idx] * eta.exp();
                        }
                    }

                    (-cum_hazard).exp()
                })
                .collect()
        })
        .collect();

    let survival_mean: Vec<f64> = (0..n_times)
        .map(|t| survival_samples.iter().map(|s| s[t]).sum::<f64>() / n_monte_carlo as f64)
        .collect();

    let survival_lower: Vec<f64> = (0..n_times)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_monte_carlo as f64 * 0.025) as usize]
        })
        .collect();

    let survival_upper: Vec<f64> = (0..n_times)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_monte_carlo as f64 * 0.975) as usize]
        })
        .collect();

    let cumulative_risk: Vec<f64> = survival_mean.iter().map(|&s| 1.0 - s).collect();

    let s_landmark = if !survival_mean.is_empty() {
        survival_mean[0]
    } else {
        1.0
    };
    let conditional_survival: Vec<f64> = survival_mean
        .iter()
        .map(|&s| if s_landmark > 0.0 { s / s_landmark } else { s })
        .collect();

    Ok(DynamicPredictionResult {
        time_points: prediction_times_filtered,
        survival_mean,
        survival_lower,
        survival_upper,
        cumulative_risk,
        conditional_survival,
        auc: 0.0,
        brier_score: 0.0,
    })
}

#[pyfunction]
#[pyo3(signature = (
    beta_long,
    gamma_surv,
    alpha,
    baseline_hazard,
    baseline_times,
    y_observed,
    times_observed,
    x_long_fixed,
    n_obs,
    n_long_vars,
    x_surv,
    n_surv_vars,
    event_time,
    event_status,
    horizon
))]
pub fn dynamic_auc(
    beta_long: Vec<f64>,
    gamma_surv: Vec<f64>,
    alpha: f64,
    baseline_hazard: Vec<f64>,
    baseline_times: Vec<f64>,
    y_observed: Vec<f64>,
    times_observed: Vec<f64>,
    x_long_fixed: Vec<f64>,
    n_obs: usize,
    n_long_vars: usize,
    x_surv: Vec<f64>,
    n_surv_vars: usize,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    horizon: f64,
) -> PyResult<f64> {
    let n_subjects = event_time.len();

    let risk_scores: Vec<f64> = (0..n_subjects)
        .map(|i| {
            let mut eta = 0.0;
            for (k, &xk) in x_surv[i * n_surv_vars..(i + 1) * n_surv_vars]
                .iter()
                .enumerate()
            {
                if k < gamma_surv.len() {
                    eta += gamma_surv[k] * xk;
                }
            }

            let subj_times: Vec<f64> = times_observed
                .iter()
                .copied()
                .filter(|t| *t <= horizon)
                .collect();

            let t_pred = subj_times.last().copied().unwrap_or(horizon);
            let mut m_t = 0.0;

            for (j, &bj) in beta_long.iter().enumerate() {
                if j < n_long_vars && i * n_long_vars + j < x_long_fixed.len() {
                    m_t += bj * x_long_fixed[i * n_long_vars + j];
                }
            }

            eta += alpha * m_t;
            eta
        })
        .collect();

    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for i in 0..n_subjects {
        for j in (i + 1)..n_subjects {
            if event_status[i] == 1 && event_time[i] <= horizon && event_time[j] > event_time[i] {
                comparable += 1.0;
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1.0;
                } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            } else if event_status[j] == 1
                && event_time[j] <= horizon
                && event_time[i] > event_time[j]
            {
                comparable += 1.0;
                if risk_scores[j] > risk_scores[i] {
                    concordant += 1.0;
                } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }
    }

    let auc = if comparable > 0.0 {
        concordant / comparable
    } else {
        0.5
    };

    Ok(auc)
}

#[pyfunction]
#[pyo3(signature = (
    survival_predictions,
    event_time,
    event_status,
    prediction_times
))]
pub fn dynamic_brier_score(
    survival_predictions: Vec<Vec<f64>>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    prediction_times: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let n_subjects = event_time.len();
    let n_times = prediction_times.len();

    let brier_scores: Vec<f64> = (0..n_times)
        .map(|t_idx| {
            let t = prediction_times[t_idx];
            let mut score_sum = 0.0;
            let mut weight_sum = 0.0;

            for i in 0..n_subjects {
                let pred = if t_idx < survival_predictions[i].len() {
                    survival_predictions[i][t_idx]
                } else {
                    0.5
                };

                let outcome = if event_time[i] <= t && event_status[i] == 1 {
                    0.0
                } else if event_time[i] > t {
                    1.0
                } else {
                    continue;
                };

                score_sum += (pred - outcome).powi(2);
                weight_sum += 1.0;
            }

            if weight_sum > 0.0 {
                score_sum / weight_sum
            } else {
                0.0
            }
        })
        .collect();

    Ok(brier_scores)
}

#[pyfunction]
pub fn landmarking_analysis(
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    covariates: Vec<f64>,
    n_subjects: usize,
    n_vars: usize,
    landmark_times: Vec<f64>,
    horizon: f64,
) -> PyResult<Vec<(f64, Vec<f64>, f64)>> {
    let mut results = Vec::new();

    for &lm in &landmark_times {
        let eligible: Vec<usize> = (0..n_subjects).filter(|&i| event_time[i] > lm).collect();

        if eligible.len() < 10 {
            continue;
        }

        let lm_time: Vec<f64> = eligible
            .iter()
            .map(|&i| (event_time[i] - lm).min(horizon - lm))
            .collect();

        let lm_status: Vec<i32> = eligible
            .iter()
            .map(|&i| {
                if event_time[i] <= horizon && event_status[i] == 1 {
                    1
                } else {
                    0
                }
            })
            .collect();

        let lm_x: Vec<f64> = {
            let mut result = Vec::with_capacity(eligible.len() * n_vars);
            for &i in &eligible {
                for j in 0..n_vars {
                    result.push(covariates[i * n_vars + j]);
                }
            }
            result
        };

        let n_lm = eligible.len();

        let mut beta = vec![0.0; n_vars];

        for _ in 0..50 {
            let mut gradient = vec![0.0; n_vars];
            let mut hessian_diag = vec![0.0; n_vars];

            let mut indices: Vec<usize> = (0..n_lm).collect();
            indices.sort_by(|&a, &b| {
                lm_time[b]
                    .partial_cmp(&lm_time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let eta: Vec<f64> = (0..n_lm)
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..n_vars {
                        e += lm_x[i * n_vars + j] * beta[j];
                    }
                    e.clamp(-700.0, 700.0)
                })
                .collect();

            let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

            let mut risk_sum = 0.0;
            let mut weighted_x = vec![0.0; n_vars];
            let mut weighted_x_sq = vec![0.0; n_vars];

            for &i in &indices {
                risk_sum += exp_eta[i];
                for j in 0..n_vars {
                    weighted_x[j] += exp_eta[i] * lm_x[i * n_vars + j];
                    weighted_x_sq[j] += exp_eta[i] * lm_x[i * n_vars + j] * lm_x[i * n_vars + j];
                }

                if lm_status[i] == 1 && risk_sum > 0.0 {
                    for j in 0..n_vars {
                        let x_bar = weighted_x[j] / risk_sum;
                        let x_sq_bar = weighted_x_sq[j] / risk_sum;
                        gradient[j] += lm_x[i * n_vars + j] - x_bar;
                        hessian_diag[j] += x_sq_bar - x_bar * x_bar;
                    }
                }
            }

            for j in 0..n_vars {
                if hessian_diag[j].abs() > 1e-10 {
                    beta[j] += gradient[j] / hessian_diag[j];
                }
            }
        }

        let concordance = compute_concordance(&lm_time, &lm_status, &lm_x, n_lm, n_vars, &beta);

        results.push((lm, beta, concordance));
    }

    Ok(results)
}

fn compute_concordance(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
) -> f64 {
    let risk_scores: Vec<f64> = (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            eta
        })
        .collect();

    let mut concordant = 0.0;
    let mut comparable = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            if status[i] == 1 && time[i] < time[j] {
                comparable += 1.0;
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1.0;
                } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            } else if status[j] == 1 && time[j] < time[i] {
                comparable += 1.0;
                if risk_scores[j] > risk_scores[i] {
                    concordant += 1.0;
                } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }
    }

    if comparable > 0.0 {
        concordant / comparable
    } else {
        0.5
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeVaryingAUCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub auc_values: Vec<f64>,
    #[pyo3(get)]
    pub auc_lower: Vec<f64>,
    #[pyo3(get)]
    pub auc_upper: Vec<f64>,
    #[pyo3(get)]
    pub integrated_auc: f64,
    #[pyo3(get)]
    pub n_cases: Vec<usize>,
    #[pyo3(get)]
    pub n_controls: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times, prediction_window, method="cumulative/dynamic"))]
pub fn time_varying_auc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
    prediction_window: f64,
    method: &str,
) -> PyResult<TimeVaryingAUCResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let mut auc_values = Vec::new();
    let mut auc_lower = Vec::new();
    let mut auc_upper = Vec::new();
    let mut n_cases_vec = Vec::new();
    let mut n_controls_vec = Vec::new();

    for &t in &eval_times {
        let (cases, controls): (Vec<usize>, Vec<usize>) = match method {
            "incident/dynamic" => {
                let cases: Vec<usize> = (0..n)
                    .filter(|&i| {
                        event_time[i] > t
                            && event_time[i] <= t + prediction_window
                            && event_status[i] == 1
                    })
                    .collect();
                let controls: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] > t + prediction_window)
                    .collect();
                (cases, controls)
            }
            _ => {
                let cases: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] <= t + prediction_window && event_status[i] == 1)
                    .collect();
                let controls: Vec<usize> = (0..n)
                    .filter(|&i| event_time[i] > t + prediction_window)
                    .collect();
                (cases, controls)
            }
        };

        n_cases_vec.push(cases.len());
        n_controls_vec.push(controls.len());

        if cases.is_empty() || controls.is_empty() {
            auc_values.push(0.5);
            auc_lower.push(0.0);
            auc_upper.push(1.0);
            continue;
        }

        let mut concordant = 0.0;
        let total = cases.len() as f64 * controls.len() as f64;

        for &case_idx in &cases {
            for &control_idx in &controls {
                if risk_scores[case_idx] > risk_scores[control_idx] {
                    concordant += 1.0;
                } else if (risk_scores[case_idx] - risk_scores[control_idx]).abs() < 1e-10 {
                    concordant += 0.5;
                }
            }
        }

        let auc = concordant / total;
        auc_values.push(auc);

        let se = (auc * (1.0 - auc) / cases.len().min(controls.len()) as f64).sqrt();
        auc_lower.push((auc - 1.96 * se).max(0.0));
        auc_upper.push((auc + 1.96 * se).min(1.0));
    }

    let integrated_auc = if eval_times.len() > 1 {
        let mut integral = 0.0;
        let mut total_weight = 0.0;
        for i in 1..eval_times.len() {
            let dt = eval_times[i] - eval_times[i - 1];
            let weight = (n_cases_vec[i] + n_cases_vec[i - 1]) as f64 / 2.0;
            integral += (auc_values[i] + auc_values[i - 1]) / 2.0 * dt * weight;
            total_weight += dt * weight;
        }
        if total_weight > 0.0 {
            integral / total_weight
        } else {
            auc_values.iter().sum::<f64>() / auc_values.len() as f64
        }
    } else if !auc_values.is_empty() {
        auc_values[0]
    } else {
        0.5
    };

    Ok(TimeVaryingAUCResult {
        times: eval_times,
        auc_values,
        auc_lower,
        auc_upper,
        integrated_auc,
        n_cases: n_cases_vec,
        n_controls: n_controls_vec,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DynamicCIndexResult {
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub lower: f64,
    #[pyo3(get)]
    pub upper: f64,
    #[pyo3(get)]
    pub n_concordant: usize,
    #[pyo3(get)]
    pub n_discordant: usize,
    #[pyo3(get)]
    pub n_tied: usize,
    #[pyo3(get)]
    pub n_pairs: usize,
    #[pyo3(get)]
    pub time_dependent_c: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, landmark_time, horizon, eval_times=None))]
pub fn dynamic_c_index(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    landmark_time: f64,
    horizon: f64,
    eval_times: Option<Vec<f64>>,
) -> PyResult<DynamicCIndexResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let eligible: Vec<usize> = (0..n).filter(|&i| event_time[i] > landmark_time).collect();

    let mut concordant = 0usize;
    let mut discordant = 0usize;
    let mut tied = 0usize;
    let mut total_pairs = 0usize;

    for (idx_i, &i) in eligible.iter().enumerate() {
        for &j in eligible.iter().skip(idx_i + 1) {
            let ti = event_time[i].min(horizon);
            let tj = event_time[j].min(horizon);
            let si = event_status[i];
            let sj = event_status[j];

            let i_event = si == 1 && event_time[i] <= horizon;
            let j_event = sj == 1 && event_time[j] <= horizon;

            if i_event && ti < tj {
                total_pairs += 1;
                if risk_scores[i] > risk_scores[j] {
                    concordant += 1;
                } else if risk_scores[i] < risk_scores[j] {
                    discordant += 1;
                } else {
                    tied += 1;
                }
            } else if j_event && tj < ti {
                total_pairs += 1;
                if risk_scores[j] > risk_scores[i] {
                    concordant += 1;
                } else if risk_scores[j] < risk_scores[i] {
                    discordant += 1;
                } else {
                    tied += 1;
                }
            }
        }
    }

    let c_index = if total_pairs > 0 {
        (concordant as f64 + 0.5 * tied as f64) / total_pairs as f64
    } else {
        0.5
    };

    let se = if total_pairs > 10 {
        (c_index * (1.0 - c_index) / total_pairs as f64).sqrt()
    } else {
        0.0
    };

    let lower = (c_index - 1.96 * se).max(0.0);
    let upper = (c_index + 1.96 * se).min(1.0);

    let default_times: Vec<f64> = {
        let min_t = event_time
            .iter()
            .filter(|&&t| t > landmark_time)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_t = horizon;
        (0..10)
            .map(|i| min_t + (max_t - min_t) * i as f64 / 9.0)
            .collect()
    };
    let times = eval_times.unwrap_or(default_times);

    let time_dependent_c: Vec<f64> = times
        .iter()
        .map(|&t| {
            let mut conc = 0.0;
            let mut pairs = 0.0;

            for (idx_i, &i) in eligible.iter().enumerate() {
                for &j in eligible.iter().skip(idx_i + 1) {
                    let ti = event_time[i];
                    let tj = event_time[j];
                    let si = event_status[i];

                    if si == 1 && ti <= t && tj > ti {
                        pairs += 1.0;
                        if risk_scores[i] > risk_scores[j] {
                            conc += 1.0;
                        } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                            conc += 0.5;
                        }
                    }

                    let sj = event_status[j];
                    if sj == 1 && tj <= t && ti > tj {
                        pairs += 1.0;
                        if risk_scores[j] > risk_scores[i] {
                            conc += 1.0;
                        } else if (risk_scores[i] - risk_scores[j]).abs() < 1e-10 {
                            conc += 0.5;
                        }
                    }
                }
            }

            if pairs > 0.0 { conc / pairs } else { 0.5 }
        })
        .collect();

    Ok(DynamicCIndexResult {
        c_index,
        se,
        lower,
        upper,
        n_concordant: concordant,
        n_discordant: discordant,
        n_tied: tied,
        n_pairs: total_pairs,
        time_dependent_c,
        eval_times: times,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct IPCWAUCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub auc_values: Vec<f64>,
    #[pyo3(get)]
    pub auc_se: Vec<f64>,
    #[pyo3(get)]
    pub integrated_auc: f64,
    #[pyo3(get)]
    pub ipcw_weights: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times))]
pub fn ipcw_auc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
) -> PyResult<IPCWAUCResult> {
    let n = risk_scores.len();
    if event_time.len() != n || event_status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| event_time[a].partial_cmp(&event_time[b]).unwrap());

    let mut km_surv = vec![1.0; n];
    let mut at_risk = n as f64;
    let mut prev_surv = 1.0;

    for (idx, &i) in sorted_indices.iter().enumerate() {
        if event_status[i] == 0 {
            let d = 1.0;
            prev_surv *= 1.0 - d / at_risk;
        }
        km_surv[i] = prev_surv.max(0.01);
        at_risk -= 1.0;
    }

    let ipcw_weights: Vec<f64> = (0..n).map(|i| 1.0 / km_surv[i]).collect();

    let max_weight = ipcw_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let ipcw_weights: Vec<f64> = ipcw_weights.iter().map(|&w| w / max_weight).collect();

    let mut auc_values = Vec::new();
    let mut auc_se = Vec::new();

    for &t in &eval_times {
        let cases: Vec<usize> = (0..n)
            .filter(|&i| event_time[i] <= t && event_status[i] == 1)
            .collect();

        let controls: Vec<usize> = (0..n).filter(|&i| event_time[i] > t).collect();

        if cases.is_empty() || controls.is_empty() {
            auc_values.push(0.5);
            auc_se.push(0.0);
            continue;
        }

        let mut weighted_concordant = 0.0;
        let mut total_weight = 0.0;

        for &case_idx in &cases {
            for &control_idx in &controls {
                let weight = ipcw_weights[case_idx] * ipcw_weights[control_idx];
                total_weight += weight;

                if risk_scores[case_idx] > risk_scores[control_idx] {
                    weighted_concordant += weight;
                } else if (risk_scores[case_idx] - risk_scores[control_idx]).abs() < 1e-10 {
                    weighted_concordant += 0.5 * weight;
                }
            }
        }

        let auc = if total_weight > 0.0 {
            weighted_concordant / total_weight
        } else {
            0.5
        };

        auc_values.push(auc);

        let se = (auc * (1.0 - auc) / (cases.len().min(controls.len()) as f64).max(1.0)).sqrt();
        auc_se.push(se);
    }

    let integrated_auc = if !auc_values.is_empty() {
        auc_values.iter().sum::<f64>() / auc_values.len() as f64
    } else {
        0.5
    };

    Ok(IPCWAUCResult {
        times: eval_times,
        auc_values,
        auc_se,
        integrated_auc,
        ipcw_weights,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SuperLandmarkResult {
    #[pyo3(get)]
    pub landmark_times: Vec<f64>,
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_errors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub c_indices: Vec<f64>,
    #[pyo3(get)]
    pub brier_scores: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
    #[pyo3(get)]
    pub pooled_coef: Vec<f64>,
    #[pyo3(get)]
    pub pooled_se: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (event_time, event_status, covariates, n_vars, landmark_times, horizon, max_iter=50))]
pub fn super_landmark_model(
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    covariates: Vec<f64>,
    n_vars: usize,
    landmark_times: Vec<f64>,
    horizon: f64,
    max_iter: usize,
) -> PyResult<SuperLandmarkResult> {
    let n_subjects = event_time.len();
    let n_landmarks = landmark_times.len();

    let mut all_coefs = Vec::new();
    let mut all_ses = Vec::new();
    let mut c_indices = Vec::new();
    let mut brier_scores = Vec::new();
    let mut n_at_risk_vec = Vec::new();
    let mut n_events_vec = Vec::new();

    for &lm in &landmark_times {
        let eligible: Vec<usize> = (0..n_subjects).filter(|&i| event_time[i] > lm).collect();
        let n_eligible = eligible.len();
        n_at_risk_vec.push(n_eligible);

        if n_eligible < 20 {
            all_coefs.push(vec![0.0; n_vars]);
            all_ses.push(vec![f64::INFINITY; n_vars]);
            c_indices.push(0.5);
            brier_scores.push(0.25);
            n_events_vec.push(0);
            continue;
        }

        let lm_time: Vec<f64> = eligible
            .iter()
            .map(|&i| (event_time[i] - lm).min(horizon - lm))
            .collect();

        let lm_status: Vec<i32> = eligible
            .iter()
            .map(|&i| {
                if event_time[i] <= horizon && event_status[i] == 1 {
                    1
                } else {
                    0
                }
            })
            .collect();

        let n_events_lm = lm_status.iter().filter(|&&s| s == 1).count();
        n_events_vec.push(n_events_lm);

        let lm_x: Vec<f64> = {
            let mut result = Vec::with_capacity(n_eligible * n_vars);
            for &i in &eligible {
                for j in 0..n_vars {
                    result.push(covariates[i * n_vars + j]);
                }
            }
            result
        };

        let mut beta = vec![0.0; n_vars];
        let mut info_diag = vec![0.0; n_vars];

        for _ in 0..max_iter {
            let mut gradient = vec![0.0; n_vars];
            let mut hessian_diag = vec![0.0; n_vars];

            let mut indices: Vec<usize> = (0..n_eligible).collect();
            indices.sort_by(|&a, &b| {
                lm_time[b]
                    .partial_cmp(&lm_time[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let eta: Vec<f64> = (0..n_eligible)
                .map(|i| {
                    let mut e = 0.0;
                    for j in 0..n_vars {
                        e += lm_x[i * n_vars + j] * beta[j];
                    }
                    e.clamp(-700.0, 700.0)
                })
                .collect();

            let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

            let mut risk_sum = 0.0;
            let mut weighted_x = vec![0.0; n_vars];
            let mut weighted_x_sq = vec![0.0; n_vars];

            for &i in &indices {
                risk_sum += exp_eta[i];
                for j in 0..n_vars {
                    weighted_x[j] += exp_eta[i] * lm_x[i * n_vars + j];
                    weighted_x_sq[j] += exp_eta[i] * lm_x[i * n_vars + j] * lm_x[i * n_vars + j];
                }

                if lm_status[i] == 1 && risk_sum > 0.0 {
                    for j in 0..n_vars {
                        let x_bar = weighted_x[j] / risk_sum;
                        let x_sq_bar = weighted_x_sq[j] / risk_sum;
                        gradient[j] += lm_x[i * n_vars + j] - x_bar;
                        hessian_diag[j] += x_sq_bar - x_bar * x_bar;
                    }
                }
            }

            for j in 0..n_vars {
                if hessian_diag[j].abs() > 1e-10 {
                    beta[j] += gradient[j] / hessian_diag[j];
                    beta[j] = beta[j].clamp(-10.0, 10.0);
                }
                info_diag[j] = hessian_diag[j];
            }
        }

        let std_errs: Vec<f64> = info_diag
            .iter()
            .map(|&info| {
                if info > 1e-10 {
                    (1.0 / info).sqrt()
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        let c_idx = compute_concordance(&lm_time, &lm_status, &lm_x, n_eligible, n_vars, &beta);

        let risk_scores: Vec<f64> = (0..n_eligible)
            .map(|i| {
                let mut eta = 0.0;
                for j in 0..n_vars {
                    eta += lm_x[i * n_vars + j] * beta[j];
                }
                eta
            })
            .collect();

        let pred_surv: Vec<f64> = risk_scores
            .iter()
            .map(|&r| (-r.exp() * 0.1).exp())
            .collect();

        let mut brier = 0.0;
        let mut brier_n = 0.0;
        for i in 0..n_eligible {
            let outcome = if lm_status[i] == 1 { 0.0 } else { 1.0 };
            brier += (pred_surv[i] - outcome).powi(2);
            brier_n += 1.0;
        }
        let brier_score = if brier_n > 0.0 { brier / brier_n } else { 0.25 };

        all_coefs.push(beta);
        all_ses.push(std_errs);
        c_indices.push(c_idx);
        brier_scores.push(brier_score);
    }

    let mut pooled_coef = vec![0.0; n_vars];
    let mut pooled_weights = vec![0.0; n_vars];

    for lm_idx in 0..n_landmarks {
        for j in 0..n_vars {
            let se = all_ses[lm_idx][j];
            if se.is_finite() && se > 1e-10 {
                let weight = 1.0 / (se * se);
                pooled_coef[j] += weight * all_coefs[lm_idx][j];
                pooled_weights[j] += weight;
            }
        }
    }

    for j in 0..n_vars {
        if pooled_weights[j] > 0.0 {
            pooled_coef[j] /= pooled_weights[j];
        }
    }

    let pooled_se: Vec<f64> = pooled_weights
        .iter()
        .map(|&w| {
            if w > 0.0 {
                (1.0 / w).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    Ok(SuperLandmarkResult {
        landmark_times,
        coefficients: all_coefs,
        std_errors: all_ses,
        c_indices,
        brier_scores,
        n_at_risk: n_at_risk_vec,
        n_events: n_events_vec,
        pooled_coef,
        pooled_se,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeDependentROCResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub sensitivity: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub specificity: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub thresholds: Vec<f64>,
    #[pyo3(get)]
    pub auc: Vec<f64>,
    #[pyo3(get)]
    pub optimal_threshold: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (risk_scores, event_time, event_status, eval_times, n_thresholds=100))]
pub fn time_dependent_roc(
    risk_scores: Vec<f64>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    eval_times: Vec<f64>,
    n_thresholds: usize,
) -> PyResult<TimeDependentROCResult> {
    let n = risk_scores.len();

    let min_risk = risk_scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_risk = risk_scores
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let thresholds: Vec<f64> = (0..n_thresholds)
        .map(|i| min_risk + (max_risk - min_risk) * i as f64 / (n_thresholds - 1) as f64)
        .collect();

    let mut all_sensitivity = Vec::new();
    let mut all_specificity = Vec::new();
    let mut all_auc = Vec::new();
    let mut optimal_thresholds = Vec::new();

    for &t in &eval_times {
        let cases: Vec<usize> = (0..n)
            .filter(|&i| event_time[i] <= t && event_status[i] == 1)
            .collect();

        let controls: Vec<usize> = (0..n).filter(|&i| event_time[i] > t).collect();

        let n_cases = cases.len();
        let n_controls = controls.len();

        if n_cases == 0 || n_controls == 0 {
            all_sensitivity.push(vec![0.0; n_thresholds]);
            all_specificity.push(vec![1.0; n_thresholds]);
            all_auc.push(0.5);
            optimal_thresholds.push(thresholds[n_thresholds / 2]);
            continue;
        }

        let mut sens = Vec::new();
        let mut spec = Vec::new();
        let mut max_youden = f64::NEG_INFINITY;
        let mut opt_thresh = thresholds[0];

        for &thresh in &thresholds {
            let tp = cases.iter().filter(|&&i| risk_scores[i] >= thresh).count();
            let tn = controls
                .iter()
                .filter(|&&i| risk_scores[i] < thresh)
                .count();

            let sensitivity_val = tp as f64 / n_cases as f64;
            let specificity_val = tn as f64 / n_controls as f64;

            sens.push(sensitivity_val);
            spec.push(specificity_val);

            let youden = sensitivity_val + specificity_val - 1.0;
            if youden > max_youden {
                max_youden = youden;
                opt_thresh = thresh;
            }
        }

        let mut auc = 0.0;
        for i in 1..n_thresholds {
            let dx = (1.0 - spec[i - 1]) - (1.0 - spec[i]);
            let dy = (sens[i - 1] + sens[i]) / 2.0;
            auc += dx * dy;
        }

        all_sensitivity.push(sens);
        all_specificity.push(spec);
        all_auc.push(auc.abs());
        optimal_thresholds.push(opt_thresh);
    }

    Ok(TimeDependentROCResult {
        times: eval_times,
        sensitivity: all_sensitivity,
        specificity: all_specificity,
        thresholds,
        auc: all_auc,
        optimal_threshold: optimal_thresholds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_prediction_basic() {
        let result = dynamic_prediction(
            vec![0.5, 0.3],
            vec![0.2],
            0.1,
            vec![0.0, 0.0],
            vec![0.01, 0.02, 0.03],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 0.5, 0.3],
            vec![1.0, 0.5, 1.0, 0.3, 1.0, 0.7],
            3,
            2,
            vec![0.5],
            1,
            2.0,
            vec![3.0, 4.0, 5.0],
            100,
        )
        .unwrap();

        assert!(!result.survival_mean.is_empty());
    }

    #[test]
    fn test_time_varying_auc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![1.5, 2.5, 3.5];

        let result = time_varying_auc(
            risk_scores,
            event_time,
            event_status,
            eval_times,
            1.0,
            "cumulative/dynamic",
        )
        .unwrap();

        assert_eq!(result.times.len(), 3);
        assert_eq!(result.auc_values.len(), 3);
    }

    #[test]
    fn test_dynamic_c_index() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];

        let result =
            dynamic_c_index(risk_scores, event_time, event_status, 0.0, 6.0, None).unwrap();

        assert!(result.c_index >= 0.0 && result.c_index <= 1.0);
        assert!(result.n_pairs > 0);
    }

    #[test]
    fn test_ipcw_auc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![2.0, 3.0, 4.0];

        let result = ipcw_auc(risk_scores, event_time, event_status, eval_times).unwrap();

        assert_eq!(result.times.len(), 3);
        assert!(result.integrated_auc >= 0.0 && result.integrated_auc <= 1.0);
    }

    #[test]
    fn test_super_landmark() {
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event_status = vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 0.5, 0.4, 0.6, 0.3, 0.8, 0.2, 0.7, 0.5,
            0.3, 0.6, 0.4,
        ];
        let landmark_times = vec![0.0, 2.0, 4.0];

        let result = super_landmark_model(
            event_time,
            event_status,
            covariates,
            2,
            landmark_times,
            12.0,
            30,
        )
        .unwrap();

        assert_eq!(result.landmark_times.len(), 3);
        assert_eq!(result.coefficients.len(), 3);
    }

    #[test]
    fn test_time_dependent_roc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![2.0, 3.5];

        let result =
            time_dependent_roc(risk_scores, event_time, event_status, eval_times, 20).unwrap();

        assert_eq!(result.times.len(), 2);
        assert_eq!(result.sensitivity.len(), 2);
        assert_eq!(result.thresholds.len(), 20);
    }
}
