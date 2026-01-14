#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_contains,
    clippy::if_same_then_else,
    clippy::manual_range_contains
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct IPCWResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub censoring_probs: Vec<f64>,
    #[pyo3(get)]
    pub treatment_effect: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_effective: f64,
}

fn fit_logistic_model(x: &[f64], y: &[i32], n: usize, p: usize, max_iter: usize) -> Vec<f64> {
    let mut beta = vec![0.0; p];

    for _ in 0..max_iter {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            let prob = 1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp());
            let residual = y[i] as f64 - prob;

            for j in 0..p {
                gradient[j] += x[i * p + j] * residual;
                hessian_diag[j] += x[i * p + j] * x[i * p + j] * prob * (1.0 - prob);
            }
        }

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

    beta
}

fn predict_probs(x: &[f64], beta: &[f64], n: usize, p: usize) -> Vec<f64> {
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

#[pyfunction]
#[pyo3(signature = (time, status, x_censoring, n_obs, n_vars, stabilized=true, trim=None))]
pub fn compute_ipcw_weights(
    time: Vec<f64>,
    status: Vec<i32>,
    x_censoring: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    stabilized: bool,
    trim: Option<f64>,
) -> PyResult<IPCWResult> {
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have length n_obs",
        ));
    }
    if x_censoring.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_censoring dimensions mismatch",
        ));
    }

    let censored: Vec<i32> = status.iter().map(|&s| if s == 0 { 1 } else { 0 }).collect();

    let mut unique_times: Vec<f64> = time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let mut censoring_probs = vec![1.0; n_obs];

    for (t_idx, &t) in unique_times.iter().enumerate() {
        let at_risk: Vec<usize> = (0..n_obs).filter(|&i| time[i] >= t).collect();

        if at_risk.is_empty() {
            continue;
        }

        let x_risk: Vec<f64> = {
            let mut result = Vec::with_capacity(at_risk.len() * n_vars);
            for &i in &at_risk {
                for j in 0..n_vars {
                    result.push(x_censoring[i * n_vars + j]);
                }
            }
            result
        };

        let y_risk: Vec<i32> = at_risk
            .iter()
            .map(|&i| {
                if (time[i] - t).abs() < 1e-10 && censored[i] == 1 {
                    1
                } else {
                    0
                }
            })
            .collect();

        let has_events = y_risk.iter().any(|&y| y == 1);
        if !has_events {
            continue;
        }

        let beta = fit_logistic_model(&x_risk, &y_risk, at_risk.len(), n_vars, 50);
        let censor_probs_t = predict_probs(&x_risk, &beta, at_risk.len(), n_vars);

        for (idx, &i) in at_risk.iter().enumerate() {
            if time[i] >= t {
                censoring_probs[i] *= 1.0 - censor_probs_t[idx];
            }
        }
    }

    let trim_threshold = trim.unwrap_or(0.01);
    for prob in &mut censoring_probs {
        *prob = prob.max(trim_threshold);
    }

    let mut weights: Vec<f64> = censoring_probs.iter().map(|&p| 1.0 / p).collect();

    if stabilized {
        let km_survival = compute_km_censoring(&time, &status, n_obs);
        for i in 0..n_obs {
            weights[i] *= km_survival[i];
        }
    }

    let n_effective = weights.iter().map(|&w| w.powi(2)).sum::<f64>().recip()
        * weights.iter().sum::<f64>().powi(2);

    Ok(IPCWResult {
        weights,
        censoring_probs,
        treatment_effect: 0.0,
        std_error: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
        n_effective,
    })
}

fn compute_km_censoring(time: &[f64], status: &[i32], n: usize) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut km_surv = vec![1.0; n];
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;

        let start_i = i;
        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        for j in start_i..i {
            km_surv[indices[j]] = cum_surv;
        }

        at_risk -= i - start_i;
    }

    km_surv
}

#[pyfunction]
#[pyo3(signature = (time, status, treatment, outcome, x_confounders, n_obs, n_vars, tau=None))]
pub fn ipcw_treatment_effect(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment: Vec<i32>,
    outcome: Vec<f64>,
    x_confounders: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    tau: Option<f64>,
) -> PyResult<IPCWResult> {
    let ipcw = compute_ipcw_weights(
        time.clone(),
        status.clone(),
        x_confounders.clone(),
        n_obs,
        n_vars,
        true,
        Some(0.01),
    )?;

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;
    let mut n_treated = 0.0;
    let mut n_control = 0.0;

    for i in 0..n_obs {
        let contrib = if time[i] <= tau_val && status[i] == 1 {
            outcome[i] * ipcw.weights[i]
        } else if time[i] > tau_val {
            outcome[i] * ipcw.weights[i]
        } else {
            continue;
        };

        if treatment[i] == 1 {
            sum_treated += contrib;
            n_treated += ipcw.weights[i];
        } else {
            sum_control += contrib;
            n_control += ipcw.weights[i];
        }
    }

    let mean_treated = if n_treated > 0.0 {
        sum_treated / n_treated
    } else {
        0.0
    };
    let mean_control = if n_control > 0.0 {
        sum_control / n_control
    } else {
        0.0
    };
    let treatment_effect = mean_treated - mean_control;

    let mut var_sum = 0.0;
    for i in 0..n_obs {
        if time[i] <= tau_val || status[i] == 1 {
            let resid = if treatment[i] == 1 {
                outcome[i] - mean_treated
            } else {
                outcome[i] - mean_control
            };
            var_sum += ipcw.weights[i] * ipcw.weights[i] * resid * resid;
        }
    }

    let std_error = (var_sum / (n_treated + n_control).powi(2)).sqrt();
    let z = 1.96;
    let ci_lower = treatment_effect - z * std_error;
    let ci_upper = treatment_effect + z * std_error;

    Ok(IPCWResult {
        weights: ipcw.weights,
        censoring_probs: ipcw.censoring_probs,
        treatment_effect,
        std_error,
        ci_lower,
        ci_upper,
        n_effective: ipcw.n_effective,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, x_censoring, n_obs, n_vars, time_points))]
pub fn ipcw_kaplan_meier(
    time: Vec<f64>,
    status: Vec<i32>,
    x_censoring: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time_points: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let ipcw = compute_ipcw_weights(
        time.clone(),
        status.clone(),
        x_censoring,
        n_obs,
        n_vars,
        true,
        Some(0.01),
    )?;

    let mut survival = Vec::with_capacity(time_points.len());
    let mut variance = Vec::with_capacity(time_points.len());

    for &t in &time_points {
        let mut numer = 0.0;
        let mut denom = 0.0;
        let mut var_sum = 0.0;

        for i in 0..n_obs {
            let at_risk = if time[i] >= t { 1.0 } else { 0.0 };
            let w = ipcw.weights[i];

            denom += w;

            if time[i] <= t && status[i] == 1 {
                numer += w;
                var_sum += w * w;
            }
        }

        let surv = if denom > 0.0 {
            1.0 - numer / denom
        } else {
            1.0
        };
        let var = if denom > 0.0 {
            var_sum / (denom * denom)
        } else {
            0.0
        };

        survival.push(surv);
        variance.push(var);
    }

    let ci_width: Vec<f64> = variance.iter().map(|&v| 1.96 * v.sqrt()).collect();

    Ok((time_points, survival, ci_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipcw_weights() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5];

        let result = compute_ipcw_weights(time, status, x, 5, 1, true, Some(0.01)).unwrap();
        assert_eq!(result.weights.len(), 5);
        assert!(result.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_km_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 0];
        let km = compute_km_censoring(&time, &status, 4);
        assert_eq!(km.len(), 4);
        assert!(km.iter().all(|&s| s <= 1.0 && s >= 0.0));
    }
}
