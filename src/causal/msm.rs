#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::len_zero,
    clippy::manual_range_contains
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct MSMResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub effective_n: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
}

fn fit_propensity_model(treatment: &[i32], x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut beta = vec![0.0; p];

    for _ in 0..100 {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            let prob = 1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp());
            let residual = treatment[i] as f64 - prob;

            for j in 0..p {
                gradient[j] += x[i * p + j] * residual;
                hessian_diag[j] += x[i * p + j] * x[i * p + j] * prob * (1.0 - prob);
            }
        }

        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                beta[j] += gradient[j] / hessian_diag[j];
            }
        }
    }

    beta
}

fn compute_propensity_scores(x: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp())
        })
        .collect()
}

fn compute_iptw_weights(
    treatment: &[i32],
    propensity: &[f64],
    stabilized: bool,
    trim: f64,
) -> Vec<f64> {
    let n = treatment.len();

    let trimmed_ps: Vec<f64> = propensity
        .iter()
        .map(|&p| p.clamp(trim, 1.0 - trim))
        .collect();

    let mut weights: Vec<f64> = (0..n)
        .map(|i| {
            if treatment[i] == 1 {
                1.0 / trimmed_ps[i]
            } else {
                1.0 / (1.0 - trimmed_ps[i])
            }
        })
        .collect();

    if stabilized {
        let prop_treated = treatment.iter().map(|&t| t as f64).sum::<f64>() / n as f64;
        for i in 0..n {
            if treatment[i] == 1 {
                weights[i] *= prop_treated;
            } else {
                weights[i] *= 1.0 - prop_treated;
            }
        }
    }

    weights
}

fn weighted_cox_fit(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    weights: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut beta = vec![0.0; p];
    let mut loglik = f64::NEG_INFINITY;

    for _ in 0..100 {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..p {
                    e += x[i * p + j] * beta[j];
                }
                e.clamp(-700.0, 700.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; p];
        let mut weighted_x_sq = vec![0.0; p];
        let mut ll = 0.0;

        for &i in &indices {
            let w = weights[i] * exp_eta[i];
            risk_sum += w;

            for j in 0..p {
                weighted_x[j] += w * x[i * p + j];
                weighted_x_sq[j] += w * x[i * p + j] * x[i * p + j];
            }

            if status[i] == 1 && risk_sum > 0.0 {
                ll += weights[i] * (eta[i] - risk_sum.ln());

                for j in 0..p {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;

                    gradient[j] += weights[i] * (x[i * p + j] - x_bar);
                    hessian_diag[j] += weights[i] * (x_sq_bar - x_bar * x_bar);
                }
            }
        }

        loglik = ll;

        let mut max_change: f64 = 0.0;
        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        if max_change < 1e-6 {
            break;
        }
    }

    let std_errors: Vec<f64> = {
        let mut se = vec![0.0; p];
        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..p {
                    e += x[i * p + j] * beta[j];
                }
                e.clamp(-700.0, 700.0)
            })
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; p];
        let mut weighted_x_sq = vec![0.0; p];
        let mut info = vec![0.0; p];

        for &i in &indices {
            let w = weights[i] * exp_eta[i];
            risk_sum += w;

            for j in 0..p {
                weighted_x[j] += w * x[i * p + j];
                weighted_x_sq[j] += w * x[i * p + j] * x[i * p + j];
            }

            if status[i] == 1 && risk_sum > 0.0 {
                for j in 0..p {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    info[j] += weights[i] * (x_sq_bar - x_bar * x_bar);
                }
            }
        }

        for j in 0..p {
            se[j] = if info[j] > 1e-10 {
                (1.0 / info[j]).sqrt()
            } else {
                f64::INFINITY
            };
        }
        se
    };

    (beta, std_errors, loglik)
}

#[pyfunction]
#[pyo3(signature = (time, status, treatment, x_outcome, x_propensity, n_obs, n_outcome_vars, n_propensity_vars, stabilized=true, trim=None))]
pub fn marginal_structural_model(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment: Vec<i32>,
    x_outcome: Vec<f64>,
    x_propensity: Vec<f64>,
    n_obs: usize,
    n_outcome_vars: usize,
    n_propensity_vars: usize,
    stabilized: bool,
    trim: Option<f64>,
) -> PyResult<MSMResult> {
    if time.len() != n_obs || status.len() != n_obs || treatment.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }

    let trim_val = trim.unwrap_or(0.01);

    let ps_beta = fit_propensity_model(&treatment, &x_propensity, n_obs, n_propensity_vars);
    let propensity = compute_propensity_scores(&x_propensity, &ps_beta, n_obs, n_propensity_vars);

    let weights = compute_iptw_weights(&treatment, &propensity, stabilized, trim_val);

    let effective_n =
        weights.iter().sum::<f64>().powi(2) / weights.iter().map(|&w| w.powi(2)).sum::<f64>();

    let p = n_outcome_vars + 1;
    let x_full: Vec<f64> = (0..n_obs)
        .flat_map(|i| {
            let mut row = vec![treatment[i] as f64];
            row.extend((0..n_outcome_vars).map(|j| x_outcome[i * n_outcome_vars + j]));
            row
        })
        .collect();

    let (coefficients, std_errors, log_likelihood) =
        weighted_cox_fit(&time, &status, &x_full, &weights, n_obs, p);

    let hazard_ratios: Vec<f64> = coefficients.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let hr_ci_lower: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();

    let hr_ci_upper: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    Ok(MSMResult {
        coefficients,
        std_errors,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        weights,
        effective_n,
        log_likelihood,
    })
}

#[pyfunction]
#[pyo3(signature = (treatment_history, x_time_varying, n_obs, n_times, n_vars, stabilized=true, trim=None))]
pub fn compute_longitudinal_iptw(
    treatment_history: Vec<i32>,
    x_time_varying: Vec<f64>,
    n_obs: usize,
    n_times: usize,
    n_vars: usize,
    stabilized: bool,
    trim: Option<f64>,
) -> PyResult<Vec<f64>> {
    if treatment_history.len() != n_obs * n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "treatment_history length mismatch",
        ));
    }
    if x_time_varying.len() != n_obs * n_times * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_time_varying dimensions mismatch",
        ));
    }

    let trim_val = trim.unwrap_or(0.01);

    let mut cumulative_weights = vec![1.0; n_obs];

    for t in 0..n_times {
        let treatment_t: Vec<i32> = (0..n_obs)
            .map(|i| treatment_history[i * n_times + t])
            .collect();

        let x_t: Vec<f64> = {
            let mut result = Vec::with_capacity(n_obs * n_vars);
            for i in 0..n_obs {
                for j in 0..n_vars {
                    result.push(x_time_varying[i * n_times * n_vars + t * n_vars + j]);
                }
            }
            result
        };

        let ps_beta = fit_propensity_model(&treatment_t, &x_t, n_obs, n_vars);
        let propensity = compute_propensity_scores(&x_t, &ps_beta, n_obs, n_vars);

        for i in 0..n_obs {
            let ps = propensity[i].clamp(trim_val, 1.0 - trim_val);
            let weight = if treatment_t[i] == 1 {
                1.0 / ps
            } else {
                1.0 / (1.0 - ps)
            };
            cumulative_weights[i] *= weight;
        }
    }

    if stabilized {
        let mean_weight = cumulative_weights.iter().sum::<f64>() / n_obs as f64;
        for w in &mut cumulative_weights {
            *w /= mean_weight;
        }
    }

    Ok(cumulative_weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_propensity_model() {
        let treatment = vec![1, 0, 1, 0, 1, 0];
        let x = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        let beta = fit_propensity_model(&treatment, &x, 6, 1);
        assert_eq!(beta.len(), 1);
    }

    #[test]
    fn test_iptw_weights() {
        let treatment = vec![1, 0, 1, 0];
        let propensity = vec![0.6, 0.4, 0.7, 0.3];
        let weights = compute_iptw_weights(&treatment, &propensity, true, 0.01);
        assert_eq!(weights.len(), 4);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_msm_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];
        let treatment = vec![1, 0, 1, 0, 1, 0];
        let x_outcome = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];
        let x_propensity = vec![1.0, 0.5, 1.0, 0.3, 1.0, 0.7, 1.0, 0.2, 1.0, 0.8, 1.0, 0.4];

        let result = marginal_structural_model(
            time,
            status,
            treatment,
            x_outcome,
            x_propensity,
            6,
            1,
            2,
            true,
            Some(0.05),
        )
        .unwrap();

        assert!(result.coefficients.len() > 0);
        assert!(result.hazard_ratios.len() > 0);
    }
}
