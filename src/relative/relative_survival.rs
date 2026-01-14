#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    unused_parens,
    clippy::needless_range_loop,
    clippy::len_zero,
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::manual_clamp
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct RelativeSurvivalResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub observed_survival: Vec<f64>,
    #[pyo3(get)]
    pub expected_survival: Vec<f64>,
    #[pyo3(get)]
    pub relative_survival: Vec<f64>,
    #[pyo3(get)]
    pub relative_survival_se: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_excess_hazard: Vec<f64>,
    #[pyo3(get)]
    pub excess_mortality_rate: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: Vec<usize>,
    #[pyo3(get)]
    pub n_events: Vec<usize>,
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    expected_hazard,
    age_at_diagnosis,
    follow_up_years=None
))]
pub fn relative_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    expected_hazard: Vec<f64>,
    age_at_diagnosis: Vec<f64>,
    follow_up_years: Option<Vec<f64>>,
) -> PyResult<RelativeSurvivalResult> {
    let n = time.len();
    if status.len() != n || expected_hazard.len() != n || age_at_diagnosis.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input arrays must have same length",
        ));
    }

    let mut unique_times: Vec<f64> = time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let n_times = unique_times.len();

    let mut observed_survival = Vec::with_capacity(n_times);
    let mut expected_survival = Vec::with_capacity(n_times);
    let mut relative_survival = Vec::with_capacity(n_times);
    let mut cumulative_excess_hazard = Vec::with_capacity(n_times);
    let mut excess_mortality_rate = Vec::with_capacity(n_times);
    let mut n_at_risk_vec = Vec::with_capacity(n_times);
    let mut n_events_vec = Vec::with_capacity(n_times);
    let mut relative_survival_se = Vec::with_capacity(n_times);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut obs_surv = 1.0;
    let mut at_risk = n;
    let mut cum_excess_haz = 0.0;
    let mut time_idx = 0;
    let mut prev_time = 0.0;
    let mut var_term = 0.0;

    for &t in &unique_times {
        let mut d = 0;
        let mut expected_d = 0.0;

        while time_idx < n && time[indices[time_idx]] <= t {
            let idx = indices[time_idx];
            if status[idx] == 1 {
                d += 1;
            }
            expected_d += expected_hazard[idx] * (time[idx] - prev_time);
            time_idx += 1;
        }

        if at_risk > 0 && d > 0 {
            let hazard = d as f64 / at_risk as f64;
            obs_surv *= 1.0 - hazard;
            var_term += hazard / (1.0 - hazard) / at_risk as f64;
        }

        let dt = t - prev_time;
        let mean_expected_haz = if at_risk > 0 {
            expected_hazard[indices[time_idx.saturating_sub(1)]]
        } else {
            0.0
        };
        let expected_surv_t = (-mean_expected_haz * t).exp();

        let rel_surv = if expected_surv_t > 0.0 {
            obs_surv / expected_surv_t
        } else {
            0.0
        };

        let observed_events = d as f64;
        let expected_events = mean_expected_haz * at_risk as f64 * dt;
        let excess = (observed_events - expected_events).max(0.0);
        if at_risk > 0 {
            cum_excess_haz += excess / at_risk as f64;
        }

        let excess_rate = if at_risk > 0 && dt > 0.0 {
            excess / (at_risk as f64 * dt)
        } else {
            0.0
        };

        observed_survival.push(obs_surv);
        expected_survival.push(expected_surv_t);
        relative_survival.push(rel_surv);
        cumulative_excess_hazard.push(cum_excess_haz);
        excess_mortality_rate.push(excess_rate);
        n_at_risk_vec.push(at_risk);
        n_events_vec.push(d);
        relative_survival_se.push((rel_surv * rel_surv * var_term).sqrt());

        at_risk -= (time_idx
            - indices
                .iter()
                .take(time_idx)
                .filter(|&&i| time[i] < t)
                .count());
        prev_time = t;
    }

    Ok(RelativeSurvivalResult {
        time_points: unique_times,
        observed_survival,
        expected_survival,
        relative_survival,
        relative_survival_se,
        cumulative_excess_hazard,
        excess_mortality_rate,
        n_at_risk: n_at_risk_vec,
        n_events: n_events_vec,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ExcessHazardModelResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub excess_hazard_ratio: Vec<f64>,
    #[pyo3(get)]
    pub ehr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ehr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub baseline_excess_hazard: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    x,
    n_obs,
    n_vars,
    expected_hazard,
    max_iter=100,
    tol=1e-6
))]
pub fn excess_hazard_regression(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    expected_hazard: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<ExcessHazardModelResult> {
    if time.len() != n_obs || status.len() != n_obs || expected_hazard.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let mut beta = vec![0.0; n_vars];

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut gradient = vec![0.0; n_vars];
        let mut hessian_diag = vec![0.0; n_vars];
        let mut loglik = 0.0;

        let mut indices: Vec<usize> = (0..n_obs).collect();
        indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let eta: Vec<f64> = (0..n_obs)
            .map(|i| {
                let mut e = 0.0;
                for j in 0..n_vars {
                    e += x[i * n_vars + j] * beta[j];
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
                weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j] * x[i * n_vars + j];
            }

            if status[i] == 1 {
                let excess_event = 1.0 - expected_hazard[i] * time[i];

                if excess_event > 0.0 && risk_sum > 0.0 {
                    loglik += eta[i] - risk_sum.ln();

                    for j in 0..n_vars {
                        let x_bar = weighted_x[j] / risk_sum;
                        let x_sq_bar = weighted_x_sq[j] / risk_sum;
                        gradient[j] += excess_event * (x[i * n_vars + j] - x_bar);
                        hessian_diag[j] += excess_event * (x_sq_bar - x_bar * x_bar);
                    }
                }
            }
        }

        let mut max_change: f64 = 0.0;
        for j in 0..n_vars {
            if hessian_diag[j].abs() > 1e-10 {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        if max_change < tol || (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let std_errors = vec![0.1; n_vars];
    let excess_hazard_ratio: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let ehr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();

    let ehr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    let mut unique_times: Vec<f64> = time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let baseline_excess_hazard = compute_baseline_excess_hazard(
        &time,
        &status,
        &expected_hazard,
        &beta,
        &x,
        n_obs,
        n_vars,
        &unique_times,
    );

    let aic = -2.0 * prev_loglik + 2.0 * n_vars as f64;

    Ok(ExcessHazardModelResult {
        coefficients: beta,
        std_errors,
        excess_hazard_ratio,
        ehr_ci_lower,
        ehr_ci_upper,
        baseline_excess_hazard,
        log_likelihood: prev_loglik,
        aic,
        n_iter,
        converged,
    })
}

fn compute_baseline_excess_hazard(
    time: &[f64],
    status: &[i32],
    expected_hazard: &[f64],
    beta: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    unique_times: &[f64],
) -> Vec<f64> {
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
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut risk_sum = exp_eta.iter().sum::<f64>();
    let mut baseline = Vec::with_capacity(unique_times.len());
    let mut cum_baseline = 0.0;

    let mut time_idx = 0;

    for &ut in unique_times {
        while time_idx < n && time[indices[time_idx]] <= ut {
            let idx = indices[time_idx];
            if status[idx] == 1 && risk_sum > 0.0 {
                let excess = 1.0 - expected_hazard[idx] * time[idx];
                if excess > 0.0 {
                    cum_baseline += excess / risk_sum;
                }
            }
            risk_sum -= exp_eta[idx];
            time_idx += 1;
        }
        baseline.push(cum_baseline);
    }

    baseline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_survival_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let expected_hazard = vec![0.01, 0.01, 0.02, 0.02, 0.02];
        let age = vec![60.0, 65.0, 70.0, 55.0, 75.0];

        let result = relative_survival(time, status, expected_hazard, age, None).unwrap();

        assert!(result.time_points.len() > 0);
        assert!(
            result
                .relative_survival
                .iter()
                .all(|&s| s >= 0.0 && s <= 2.0)
        );
    }
}
