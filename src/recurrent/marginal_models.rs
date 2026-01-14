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

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum MarginalMethod {
    WeiLinWeissfeld,
    AndersenGill,
}

#[pymethods]
impl MarginalMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "wlw" | "wei_lin_weissfeld" | "weissfeld" => Ok(MarginalMethod::WeiLinWeissfeld),
            "ag" | "andersen_gill" => Ok(MarginalMethod::AndersenGill),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown marginal method",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MarginalModelResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub robust_se: Vec<f64>,
    #[pyo3(get)]
    pub naive_se: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub score_test: f64,
    #[pyo3(get)]
    pub wald_test: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
    #[pyo3(get)]
    pub mean_events_per_subject: f64,
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
    method,
    max_iter=100,
    tol=1e-6
))]
pub fn marginal_recurrent_model(
    subject_id: Vec<usize>,
    start_time: Vec<f64>,
    stop_time: Vec<f64>,
    event_status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    method: &MarginalMethod,
    max_iter: usize,
    tol: f64,
) -> PyResult<MarginalModelResult> {
    if subject_id.len() != n_obs
        || start_time.len() != n_obs
        || stop_time.len() != n_obs
        || event_status.len() != n_obs
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have length n_obs",
        ));
    }

    let n_events = event_status.iter().filter(|&&s| s == 1).count();
    let n_subjects = subject_id.iter().copied().max().map(|x| x + 1).unwrap_or(0);
    let mean_events = n_events as f64 / n_subjects.max(1) as f64;

    let mut beta = vec![0.0; n_vars];

    let time = match method {
        MarginalMethod::WeiLinWeissfeld => stop_time.clone(),
        MarginalMethod::AndersenGill => stop_time.clone(),
    };

    let mut prev_loglik = f64::NEG_INFINITY;
    for iter in 0..max_iter {
        let mut gradient = vec![0.0; n_vars];
        let mut hessian_diag = vec![0.0; n_vars];

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
        let mut loglik = 0.0;

        for &i in &indices {
            match method {
                MarginalMethod::AndersenGill => {
                    risk_sum += exp_eta[i];
                    for j in 0..n_vars {
                        weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                        weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j] * x[i * n_vars + j];
                    }
                }
                MarginalMethod::WeiLinWeissfeld => {
                    if time[i] >= start_time[i] {
                        risk_sum += exp_eta[i];
                        for j in 0..n_vars {
                            weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                            weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j] * x[i * n_vars + j];
                        }
                    }
                }
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

    let (naive_se, info_matrix) = compute_naive_se(&time, &event_status, &beta, &x, n_obs, n_vars);

    let robust_se = compute_robust_se(
        &subject_id,
        &time,
        &event_status,
        &beta,
        &x,
        n_obs,
        n_vars,
        &info_matrix,
    );

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let hr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(robust_se.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();

    let hr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(robust_se.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    let score_test: f64 = beta
        .iter()
        .zip(robust_se.iter())
        .map(|(&b, &se)| if se > 0.0 { (b / se).powi(2) } else { 0.0 })
        .sum();

    let wald_test = score_test;

    Ok(MarginalModelResult {
        coefficients: beta,
        robust_se,
        naive_se,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        log_likelihood: prev_loglik,
        score_test,
        wald_test,
        n_events,
        n_subjects,
        mean_events_per_subject: mean_events,
    })
}

fn compute_naive_se(
    time: &[f64],
    status: &[i32],
    beta: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
) -> (Vec<f64>, Vec<f64>) {
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
        risk_sum += exp_eta[i];
        for j in 0..p {
            weighted_x[j] += exp_eta[i] * x[i * p + j];
            weighted_x_sq[j] += exp_eta[i] * x[i * p + j] * x[i * p + j];
        }

        if status[i] == 1 && risk_sum > 0.0 {
            for j in 0..p {
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;
                info[j] += x_sq_bar - x_bar * x_bar;
            }
        }
    }

    let se: Vec<f64> = info
        .iter()
        .map(|&i| {
            if i > 1e-10 {
                (1.0 / i).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    (se, info)
}

fn compute_robust_se(
    subject_id: &[usize],
    time: &[f64],
    status: &[i32],
    beta: &[f64],
    x: &[f64],
    n: usize,
    p: usize,
    info_matrix: &[f64],
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
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_subjects = subject_id.iter().copied().max().map(|x| x + 1).unwrap_or(0);

    let mut subject_scores: Vec<Vec<f64>> = vec![vec![0.0; p]; n_subjects];

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; p];

    for &i in &indices {
        risk_sum += exp_eta[i];
        for j in 0..p {
            weighted_x[j] += exp_eta[i] * x[i * p + j];
        }

        if status[i] == 1 && risk_sum > 0.0 {
            let subj = subject_id[i];
            for j in 0..p {
                let x_bar = weighted_x[j] / risk_sum;
                subject_scores[subj][j] += x[i * p + j] - x_bar;
            }
        }
    }

    let mut meat = vec![0.0; p];
    for subj in 0..n_subjects {
        for j in 0..p {
            meat[j] += subject_scores[subj][j].powi(2);
        }
    }

    let robust_se: Vec<f64> = (0..p)
        .map(|j| {
            if info_matrix[j] > 1e-10 && meat[j] > 0.0 {
                (meat[j] / (info_matrix[j] * info_matrix[j])).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    robust_se
}

#[pyfunction]
pub fn andersen_gill(
    subject_id: Vec<usize>,
    start_time: Vec<f64>,
    stop_time: Vec<f64>,
    event_status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
) -> PyResult<MarginalModelResult> {
    marginal_recurrent_model(
        subject_id,
        start_time,
        stop_time,
        event_status,
        x,
        n_obs,
        n_vars,
        &MarginalMethod::AndersenGill,
        100,
        1e-6,
    )
}

#[pyfunction]
pub fn wei_lin_weissfeld(
    subject_id: Vec<usize>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
) -> PyResult<MarginalModelResult> {
    let start_time = vec![0.0; n_obs];
    marginal_recurrent_model(
        subject_id,
        start_time,
        event_time,
        event_status,
        x,
        n_obs,
        n_vars,
        &MarginalMethod::WeiLinWeissfeld,
        100,
        1e-6,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_andersen_gill_basic() {
        let subject_id = vec![0, 0, 1, 1, 2];
        let start_time = vec![0.0, 5.0, 0.0, 3.0, 0.0];
        let stop_time = vec![5.0, 10.0, 3.0, 8.0, 7.0];
        let event_status = vec![1, 1, 1, 0, 1];
        let x = vec![1.0, 0.5, 1.0, 0.3, 0.0];

        let result =
            andersen_gill(subject_id, start_time, stop_time, event_status, x, 5, 1).unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert_eq!(result.n_events, 4);
        assert!(result.robust_se[0].is_finite());
    }
}
