#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::len_zero,
    clippy::collapsible_if
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
                        if bt > landmark_time && bt <= t {
                            if t_idx < baseline_hazard.len() {
                                cum_hazard += baseline_hazard[t_idx] * eta.exp();
                            }
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

        assert!(result.survival_mean.len() > 0);
    }
}
