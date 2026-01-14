#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::len_zero
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct GComputationResult {
    #[pyo3(get)]
    pub ate: f64,
    #[pyo3(get)]
    pub ate_se: f64,
    #[pyo3(get)]
    pub ate_ci_lower: f64,
    #[pyo3(get)]
    pub ate_ci_upper: f64,
    #[pyo3(get)]
    pub potential_outcome_treated: f64,
    #[pyo3(get)]
    pub potential_outcome_control: f64,
    #[pyo3(get)]
    pub survival_treated: Vec<f64>,
    #[pyo3(get)]
    pub survival_control: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub rmst_treated: f64,
    #[pyo3(get)]
    pub rmst_control: f64,
    #[pyo3(get)]
    pub rmst_difference: f64,
}

#[derive(Debug, Clone)]
struct OutcomeModel {
    beta: Vec<f64>,
    scale: f64,
    shape: f64,
}

fn fit_outcome_model(time: &[f64], status: &[i32], x: &[f64], n: usize, p: usize) -> OutcomeModel {
    let mut beta = vec![0.0; p];

    let mean_time = time.iter().sum::<f64>() / n as f64;
    let scale = mean_time.max(0.01);

    let log_times: Vec<f64> = time.iter().filter(|&&t| t > 0.0).map(|t| t.ln()).collect();
    let shape = if log_times.len() > 1 {
        let mean_log = log_times.iter().sum::<f64>() / log_times.len() as f64;
        let var_log = log_times
            .iter()
            .map(|&l| (l - mean_log).powi(2))
            .sum::<f64>()
            / log_times.len() as f64;
        (std::f64::consts::PI / (6.0_f64.sqrt() * var_log.sqrt().max(0.1))).max(0.1)
    } else {
        1.0
    };

    for _ in 0..100 {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            let exp_eta = eta.clamp(-700.0, 700.0).exp();

            let log_t = time[i].max(1e-10).ln();
            let z = (log_t - scale.ln() - eta) * shape;

            let residual = if status[i] == 1 {
                shape * (1.0 - (-z).exp().exp())
            } else {
                -shape * (-z).exp()
            };

            for j in 0..p {
                gradient[j] += x[i * p + j] * residual;
                hessian_diag[j] += x[i * p + j] * x[i * p + j] * shape * shape;
            }
        }

        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                beta[j] += gradient[j] / (hessian_diag[j] + 0.01);
            }
        }
    }

    OutcomeModel { beta, scale, shape }
}

fn predict_survival(model: &OutcomeModel, x_row: &[f64], time_points: &[f64]) -> Vec<f64> {
    let mut eta = 0.0;
    for (j, &x_j) in x_row.iter().enumerate() {
        if j < model.beta.len() {
            eta += x_j * model.beta[j];
        }
    }

    time_points
        .iter()
        .map(|&t| {
            if t <= 0.0 {
                return 1.0;
            }
            let z = t / (model.scale * eta.exp());
            (-z.powf(model.shape)).exp()
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (time, status, treatment, x_confounders, n_obs, n_vars, tau=None, n_bootstrap=None))]
pub fn g_computation(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment: Vec<i32>,
    x_confounders: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    tau: Option<f64>,
    n_bootstrap: Option<usize>,
) -> PyResult<GComputationResult> {
    if time.len() != n_obs || status.len() != n_obs || treatment.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }
    if x_confounders.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_confounders dimensions mismatch",
        ));
    }

    let p = n_vars + 1;
    let x_full: Vec<f64> = (0..n_obs)
        .flat_map(|i| {
            let mut row = vec![treatment[i] as f64];
            row.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));
            row
        })
        .collect();

    let model = fit_outcome_model(&time, &status, &x_full, n_obs, p);

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let time_points: Vec<f64> = {
        let mut tp: Vec<f64> = time.clone();
        tp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        tp.dedup();
        tp.into_iter().filter(|&t| t <= tau_val).collect()
    };

    let survival_results: Vec<(Vec<f64>, Vec<f64>)> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut x_treated = vec![1.0];
            x_treated.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));

            let mut x_control = vec![0.0];
            x_control.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));

            let surv_treated = predict_survival(&model, &x_treated, &time_points);
            let surv_control = predict_survival(&model, &x_control, &time_points);

            (surv_treated, surv_control)
        })
        .collect();

    let n_times = time_points.len();
    let mut survival_treated = vec![0.0; n_times];
    let mut survival_control = vec![0.0; n_times];

    for (surv_t, surv_c) in &survival_results {
        for (t, (&st, &sc)) in surv_t.iter().zip(surv_c.iter()).enumerate() {
            survival_treated[t] += st;
            survival_control[t] += sc;
        }
    }

    for t in 0..n_times {
        survival_treated[t] /= n_obs as f64;
        survival_control[t] /= n_obs as f64;
    }

    let rmst_treated = compute_rmst(&time_points, &survival_treated, tau_val);
    let rmst_control = compute_rmst(&time_points, &survival_control, tau_val);
    let rmst_difference = rmst_treated - rmst_control;

    let potential_outcome_treated = rmst_treated;
    let potential_outcome_control = rmst_control;
    let ate = rmst_difference;

    let n_boot = n_bootstrap.unwrap_or(200);
    let ate_se = bootstrap_se(
        &time,
        &status,
        &treatment,
        &x_confounders,
        n_obs,
        n_vars,
        tau_val,
        n_boot,
    );

    let z = 1.96;
    let ate_ci_lower = ate - z * ate_se;
    let ate_ci_upper = ate + z * ate_se;

    Ok(GComputationResult {
        ate,
        ate_se,
        ate_ci_lower,
        ate_ci_upper,
        potential_outcome_treated,
        potential_outcome_control,
        survival_treated,
        survival_control,
        time_points,
        rmst_treated,
        rmst_control,
        rmst_difference,
    })
}

fn compute_rmst(time_points: &[f64], survival: &[f64], tau: f64) -> f64 {
    if time_points.is_empty() || survival.is_empty() {
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

    if time_points.last().map(|&t| t <= tau).unwrap_or(false) {
        rmst += prev_surv * (tau - prev_time);
    }

    rmst
}

fn bootstrap_se(
    time: &[f64],
    status: &[i32],
    treatment: &[i32],
    x_confounders: &[f64],
    n_obs: usize,
    n_vars: usize,
    tau: f64,
    n_bootstrap: usize,
) -> f64 {
    let ates: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = fastrand::Rng::with_seed(b as u64 + 12345);
            let indices: Vec<usize> = (0..n_obs).map(|_| rng.usize(0..n_obs)).collect();

            let boot_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let boot_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
            let boot_treatment: Vec<i32> = indices.iter().map(|&i| treatment[i]).collect();
            let boot_x: Vec<f64> = indices
                .iter()
                .flat_map(|&i| (0..n_vars).map(move |j| x_confounders[i * n_vars + j]))
                .collect();

            let p = n_vars + 1;
            let x_full: Vec<f64> = (0..n_obs)
                .flat_map(|i| {
                    let mut row = vec![boot_treatment[i] as f64];
                    row.extend((0..n_vars).map(|j| boot_x[i * n_vars + j]));
                    row
                })
                .collect();

            let model = fit_outcome_model(&boot_time, &boot_status, &x_full, n_obs, p);

            let time_points: Vec<f64> = {
                let mut tp: Vec<f64> = boot_time.clone();
                tp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                tp.dedup();
                tp.into_iter().filter(|&t| t <= tau).collect()
            };

            let mut rmst_t_sum = 0.0;
            let mut rmst_c_sum = 0.0;

            for i in 0..n_obs {
                let mut x_treated = vec![1.0];
                x_treated.extend((0..n_vars).map(|j| boot_x[i * n_vars + j]));
                let surv_t = predict_survival(&model, &x_treated, &time_points);
                rmst_t_sum += compute_rmst(&time_points, &surv_t, tau);

                let mut x_control = vec![0.0];
                x_control.extend((0..n_vars).map(|j| boot_x[i * n_vars + j]));
                let surv_c = predict_survival(&model, &x_control, &time_points);
                rmst_c_sum += compute_rmst(&time_points, &surv_c, tau);
            }

            let rmst_treated = rmst_t_sum / n_obs as f64;
            let rmst_control = rmst_c_sum / n_obs as f64;
            Some(rmst_treated - rmst_control)
        })
        .collect();

    if ates.len() < 2 {
        return 0.0;
    }

    let mean = ates.iter().sum::<f64>() / ates.len() as f64;
    let var = ates.iter().map(|&a| (a - mean).powi(2)).sum::<f64>() / (ates.len() - 1) as f64;
    var.sqrt()
}

#[pyfunction]
#[pyo3(signature = (time, status, treatment, x_confounders, n_obs, n_vars, time_points))]
pub fn g_computation_survival_curves(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment: Vec<i32>,
    x_confounders: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time_points: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let p = n_vars + 1;
    let x_full: Vec<f64> = (0..n_obs)
        .flat_map(|i| {
            let mut row = vec![treatment[i] as f64];
            row.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));
            row
        })
        .collect();

    let model = fit_outcome_model(&time, &status, &x_full, n_obs, p);

    let n_times = time_points.len();
    let mut survival_treated = vec![0.0; n_times];
    let mut survival_control = vec![0.0; n_times];

    for i in 0..n_obs {
        let mut x_treated = vec![1.0];
        x_treated.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));
        let surv_t = predict_survival(&model, &x_treated, &time_points);

        let mut x_control = vec![0.0];
        x_control.extend((0..n_vars).map(|j| x_confounders[i * n_vars + j]));
        let surv_c = predict_survival(&model, &x_control, &time_points);

        for t in 0..n_times {
            survival_treated[t] += surv_t[t];
            survival_control[t] += surv_c[t];
        }
    }

    for t in 0..n_times {
        survival_treated[t] /= n_obs as f64;
        survival_control[t] /= n_obs as f64;
    }

    Ok((time_points, survival_treated, survival_control))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmst() {
        let time_points = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let survival = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let rmst = compute_rmst(&time_points, &survival, 5.0);
        assert!(rmst > 0.0);
    }

    #[test]
    fn test_g_computation_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];
        let treatment = vec![1, 0, 1, 0, 1, 0];
        let x = vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4];

        let result = g_computation(time, status, treatment, x, 6, 1, Some(5.0), Some(10)).unwrap();

        assert!(result.survival_treated.len() > 0);
        assert!(result.survival_control.len() > 0);
    }
}
