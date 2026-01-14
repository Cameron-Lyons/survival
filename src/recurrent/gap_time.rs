#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct GapTimeResult {
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
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub baseline_times: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (
    subject_id,
    start_time,
    stop_time,
    event_status,
    x,
    n_obs,
    n_vars,
    max_iter=100,
    tol=1e-6
))]
pub fn gap_time_model(
    subject_id: Vec<usize>,
    start_time: Vec<f64>,
    stop_time: Vec<f64>,
    event_status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<GapTimeResult> {
    if subject_id.len() != n_obs
        || start_time.len() != n_obs
        || stop_time.len() != n_obs
        || event_status.len() != n_obs
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let gap_times: Vec<f64> = stop_time
        .iter()
        .zip(start_time.iter())
        .map(|(&stop, &start)| (stop - start).max(0.001))
        .collect();

    let n_events = event_status.iter().filter(|&&s| s == 1).count();
    let n_subjects = subject_id.iter().copied().max().map(|x| x + 1).unwrap_or(0);

    let mut beta = vec![0.0; n_vars];

    let mut prev_loglik = f64::NEG_INFINITY;
    for iter in 0..max_iter {
        let mut gradient = vec![0.0; n_vars];
        let mut hessian_diag = vec![0.0; n_vars];

        let mut indices: Vec<usize> = (0..n_obs).collect();
        indices.sort_by(|&a, &b| {
            gap_times[b]
                .partial_cmp(&gap_times[a])
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
        let mut loglik = 0.0;

        for &i in &indices {
            risk_sum += exp_eta[i];
            for j in 0..n_vars {
                weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j] * x[i * n_vars + j];
            }

            if event_status[i] == 1 && risk_sum > 0.0 {
                loglik += eta[i] - risk_sum.ln();

                for j in 0..n_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    gradient[j] += x[i * n_vars + j] - x_bar;
                    hessian_diag[j] += x_sq_bar - x_bar * x_bar;
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
            break;
        }
        prev_loglik = loglik;
    }

    let std_errors: Vec<f64> = {
        let mut se = vec![0.0; n_vars];
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

        let mut indices: Vec<usize> = (0..n_obs).collect();
        indices.sort_by(|&a, &b| {
            gap_times[b]
                .partial_cmp(&gap_times[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; n_vars];
        let mut weighted_x_sq = vec![0.0; n_vars];
        let mut info = vec![0.0; n_vars];

        for &i in &indices {
            risk_sum += exp_eta[i];
            for j in 0..n_vars {
                weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j] * x[i * n_vars + j];
            }

            if event_status[i] == 1 && risk_sum > 0.0 {
                for j in 0..n_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    info[j] += x_sq_bar - x_bar * x_bar;
                }
            }
        }

        for j in 0..n_vars {
            se[j] = if info[j] > 1e-10 {
                (1.0 / info[j]).sqrt()
            } else {
                f64::INFINITY
            };
        }
        se
    };

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let hr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();

    let hr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    let mut unique_times: Vec<f64> = gap_times.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let baseline_hazard = compute_baseline_hazard(
        &gap_times,
        &event_status,
        &beta,
        &x,
        n_obs,
        n_vars,
        &unique_times,
    );

    let aic = -2.0 * prev_loglik + 2.0 * n_vars as f64;
    let bic = -2.0 * prev_loglik + (n_vars as f64) * (n_obs as f64).ln();

    Ok(GapTimeResult {
        coefficients: beta,
        std_errors,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_events,
        n_subjects,
        baseline_hazard,
        baseline_times: unique_times,
    })
}

fn compute_baseline_hazard(
    time: &[f64],
    status: &[i32],
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
    let mut baseline_hazard = Vec::with_capacity(unique_times.len());
    let mut cum_haz = 0.0;

    let mut time_idx = 0;

    for &ut in unique_times {
        while time_idx < n && time[indices[time_idx]] <= ut {
            let idx = indices[time_idx];
            if status[idx] == 1 && risk_sum > 0.0 {
                cum_haz += 1.0 / risk_sum;
            }
            risk_sum -= exp_eta[idx];
            time_idx += 1;
        }
        baseline_hazard.push(cum_haz);
    }

    baseline_hazard
}

#[pyfunction]
#[pyo3(signature = (
    subject_id,
    event_time,
    event_status,
    x,
    n_obs,
    n_vars,
    stratify_by_event_number=false
))]
pub fn pwp_gap_time(
    subject_id: Vec<usize>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    stratify_by_event_number: bool,
) -> PyResult<GapTimeResult> {
    let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
    sorted_indices.sort_by(|&a, &b| {
        subject_id[a].cmp(&subject_id[b]).then_with(|| {
            event_time[a]
                .partial_cmp(&event_time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let mut gap_starts = vec![0.0; n_obs];
    let mut prev_time: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

    for &i in &sorted_indices {
        let subj = subject_id[i];
        gap_starts[i] = *prev_time.get(&subj).unwrap_or(&0.0);
        if event_status[i] == 1 {
            prev_time.insert(subj, event_time[i]);
        }
    }

    gap_time_model(
        subject_id,
        gap_starts,
        event_time,
        event_status,
        x,
        n_obs,
        n_vars,
        100,
        1e-6,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_time_basic() {
        let subject_id = vec![0, 0, 1, 1, 2];
        let start_time = vec![0.0, 5.0, 0.0, 3.0, 0.0];
        let stop_time = vec![5.0, 10.0, 3.0, 8.0, 7.0];
        let event_status = vec![1, 1, 1, 0, 1];
        let x = vec![1.0, 0.5, 1.0, 0.3, 0.0];

        let result = gap_time_model(
            subject_id,
            start_time,
            stop_time,
            event_status,
            x,
            5,
            1,
            50,
            1e-4,
        )
        .unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(result.n_events, 4);
    }
}
