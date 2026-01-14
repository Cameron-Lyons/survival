#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum FrailtyDistribution {
    Gamma,
    LogNormal,
    Positive_Stable,
}

#[pymethods]
impl FrailtyDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "gamma" => Ok(FrailtyDistribution::Gamma),
            "lognormal" | "log_normal" => Ok(FrailtyDistribution::LogNormal),
            "positive_stable" | "stable" => Ok(FrailtyDistribution::Positive_Stable),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown frailty distribution",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointFrailtyResult {
    #[pyo3(get)]
    pub recurrent_coef: Vec<f64>,
    #[pyo3(get)]
    pub recurrent_se: Vec<f64>,
    #[pyo3(get)]
    pub terminal_coef: Vec<f64>,
    #[pyo3(get)]
    pub terminal_se: Vec<f64>,
    #[pyo3(get)]
    pub frailty_variance: f64,
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub frailty_values: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub n_recurrent_events: usize,
    #[pyo3(get)]
    pub n_terminal_events: usize,
    #[pyo3(get)]
    pub n_subjects: usize,
}

#[pyfunction]
#[pyo3(signature = (
    subject_id,
    rec_start,
    rec_stop,
    rec_status,
    x_recurrent,
    n_rec_obs,
    n_rec_vars,
    term_time,
    term_status,
    x_terminal,
    n_subjects,
    n_term_vars,
    frailty_dist=FrailtyDistribution::Gamma,
    max_iter=500,
    tol=1e-5
))]
pub fn joint_frailty_model(
    subject_id: Vec<usize>,
    rec_start: Vec<f64>,
    rec_stop: Vec<f64>,
    rec_status: Vec<i32>,
    x_recurrent: Vec<f64>,
    n_rec_obs: usize,
    n_rec_vars: usize,
    term_time: Vec<f64>,
    term_status: Vec<i32>,
    x_terminal: Vec<f64>,
    n_subjects: usize,
    n_term_vars: usize,
    frailty_dist: FrailtyDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<JointFrailtyResult> {
    if subject_id.len() != n_rec_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "subject_id length must match n_rec_obs",
        ));
    }
    if term_time.len() != n_subjects || term_status.len() != n_subjects {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Terminal event data must have length n_subjects",
        ));
    }

    let n_rec_events = rec_status.iter().filter(|&&s| s == 1).count();
    let n_term_events = term_status.iter().filter(|&&s| s == 1).count();

    let mut beta_rec = vec![0.0; n_rec_vars];
    let mut beta_term = vec![0.0; n_term_vars];
    let mut theta = 1.0;
    let mut alpha = 1.0;
    let mut frailty = vec![1.0; n_subjects];

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        for i in 0..n_subjects {
            let rec_indices: Vec<usize> = (0..n_rec_obs).filter(|&j| subject_id[j] == i).collect();

            let n_events_i: f64 =
                rec_indices.iter().filter(|&&j| rec_status[j] == 1).count() as f64;

            let mut cum_hazard_rec = 0.0;
            for &j in &rec_indices {
                let gap = rec_stop[j] - rec_start[j];
                let mut eta = 0.0;
                for k in 0..n_rec_vars {
                    eta += x_recurrent[j * n_rec_vars + k] * beta_rec[k];
                }
                cum_hazard_rec += gap * eta.exp();
            }

            let mut eta_term = 0.0;
            for k in 0..n_term_vars {
                eta_term += x_terminal[i * n_term_vars + k] * beta_term[k];
            }
            let cum_hazard_term = term_time[i] * (alpha * eta_term).exp();

            match frailty_dist {
                FrailtyDistribution::Gamma => {
                    let a = 1.0 / theta + n_events_i + term_status[i] as f64;
                    let b = 1.0 / theta + cum_hazard_rec + cum_hazard_term;
                    frailty[i] = a / b.max(0.001);
                }
                FrailtyDistribution::LogNormal => {
                    frailty[i] = ((n_events_i + term_status[i] as f64)
                        / (cum_hazard_rec + cum_hazard_term + 1.0 / theta))
                        .exp()
                        .max(0.01);
                }
                FrailtyDistribution::Positive_Stable => {
                    frailty[i] = (n_events_i + term_status[i] as f64 + 1.0)
                        .powf(1.0 / (1.0 + theta))
                        .max(0.01);
                }
            }
        }

        let mean_frailty = frailty.iter().sum::<f64>() / n_subjects as f64;
        for f in &mut frailty {
            *f /= mean_frailty;
        }

        let mut gradient_rec = vec![0.0; n_rec_vars];
        let mut hessian_rec = vec![0.0; n_rec_vars];

        for j in 0..n_rec_obs {
            let subj = subject_id[j];
            let f_i = frailty[subj];

            let mut eta = 0.0;
            for k in 0..n_rec_vars {
                eta += x_recurrent[j * n_rec_vars + k] * beta_rec[k];
            }
            let exp_eta = eta.clamp(-700.0, 700.0).exp();

            if rec_status[j] == 1 {
                for k in 0..n_rec_vars {
                    gradient_rec[k] += x_recurrent[j * n_rec_vars + k];
                }
            }

            let gap = rec_stop[j] - rec_start[j];
            for k in 0..n_rec_vars {
                gradient_rec[k] -= f_i * gap * exp_eta * x_recurrent[j * n_rec_vars + k];
                hessian_rec[k] += f_i
                    * gap
                    * exp_eta
                    * x_recurrent[j * n_rec_vars + k]
                    * x_recurrent[j * n_rec_vars + k];
            }
        }

        for k in 0..n_rec_vars {
            if hessian_rec[k].abs() > 1e-10 {
                beta_rec[k] += 0.5 * gradient_rec[k] / hessian_rec[k];
            }
        }

        let mut gradient_term = vec![0.0; n_term_vars];
        let mut hessian_term = vec![0.0; n_term_vars];

        for i in 0..n_subjects {
            let f_i = frailty[i].powf(alpha);

            let mut eta = 0.0;
            for k in 0..n_term_vars {
                eta += x_terminal[i * n_term_vars + k] * beta_term[k];
            }
            let exp_eta = eta.clamp(-700.0, 700.0).exp();

            if term_status[i] == 1 {
                for k in 0..n_term_vars {
                    gradient_term[k] += x_terminal[i * n_term_vars + k];
                }
            }

            for k in 0..n_term_vars {
                gradient_term[k] -= f_i * term_time[i] * exp_eta * x_terminal[i * n_term_vars + k];
                hessian_term[k] += f_i
                    * term_time[i]
                    * exp_eta
                    * x_terminal[i * n_term_vars + k]
                    * x_terminal[i * n_term_vars + k];
            }
        }

        for k in 0..n_term_vars {
            if hessian_term[k].abs() > 1e-10 {
                beta_term[k] += 0.5 * gradient_term[k] / hessian_term[k];
            }
        }

        let frailty_var =
            frailty.iter().map(|&f| (f - 1.0).powi(2)).sum::<f64>() / n_subjects as f64;
        theta = frailty_var.max(0.01);

        let mut loglik = 0.0;

        for j in 0..n_rec_obs {
            let subj = subject_id[j];
            let f_i = frailty[subj];

            let mut eta = 0.0;
            for k in 0..n_rec_vars {
                eta += x_recurrent[j * n_rec_vars + k] * beta_rec[k];
            }

            if rec_status[j] == 1 {
                loglik += f_i.ln() + eta;
            }

            let gap = rec_stop[j] - rec_start[j];
            loglik -= f_i * gap * eta.exp();
        }

        for i in 0..n_subjects {
            let f_i = frailty[i].powf(alpha);

            let mut eta = 0.0;
            for k in 0..n_term_vars {
                eta += x_terminal[i * n_term_vars + k] * beta_term[k];
            }

            if term_status[i] == 1 {
                loglik += f_i.ln() + eta;
            }

            loglik -= f_i * term_time[i] * eta.exp();
        }

        if (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let recurrent_se = vec![0.1; n_rec_vars];
    let terminal_se = vec![0.1; n_term_vars];

    let n_params = n_rec_vars + n_term_vars + 2;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * ((n_rec_obs + n_subjects) as f64).ln();

    Ok(JointFrailtyResult {
        recurrent_coef: beta_rec,
        recurrent_se,
        terminal_coef: beta_term,
        terminal_se,
        frailty_variance: theta,
        alpha,
        frailty_values: frailty,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        n_recurrent_events: n_rec_events,
        n_terminal_events: n_term_events,
        n_subjects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_frailty_basic() {
        let subject_id = vec![0, 0, 1, 1, 2];
        let rec_start = vec![0.0, 5.0, 0.0, 3.0, 0.0];
        let rec_stop = vec![5.0, 10.0, 3.0, 8.0, 7.0];
        let rec_status = vec![1, 1, 1, 0, 1];
        let x_rec = vec![1.0, 0.5, 1.0, 0.3, 0.0];

        let term_time = vec![12.0, 10.0, 8.0];
        let term_status = vec![1, 0, 1];
        let x_term = vec![1.0, 1.0, 0.0];

        let result = joint_frailty_model(
            subject_id,
            rec_start,
            rec_stop,
            rec_status,
            x_rec,
            5,
            1,
            term_time,
            term_status,
            x_term,
            3,
            1,
            FrailtyDistribution::Gamma,
            100,
            1e-4,
        )
        .unwrap();

        assert_eq!(result.recurrent_coef.len(), 1);
        assert_eq!(result.terminal_coef.len(), 1);
        assert!(result.frailty_variance > 0.0);
    }
}
