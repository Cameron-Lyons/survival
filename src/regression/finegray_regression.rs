use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

use crate::constants::{
    IPCW_SURVIVAL_FLOOR, PARALLEL_THRESHOLD_LARGE, Z_SCORE_90, Z_SCORE_95, Z_SCORE_99,
    clamped_normal_ci_bounds, exp_clamped, normal_ci_bounds_95, same_time,
};
use crate::internal::matrix::invert_matrix;
use crate::internal::statistical::{compute_censoring_km, km_step_prob_at, normal_cdf};

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct FineGrayResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_scores: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub variance_matrix: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub log_likelihood_null: f64,
    pub n_events: usize,
    pub n_competing: usize,
    pub n_censored: usize,
    pub n_observations: usize,
    pub event_type: i32,
    pub convergence: bool,
    pub iterations: usize,
}

impl fmt::Display for FineGrayResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FineGrayResult(coef={:?}, n_events={}, converged={})",
            self.coefficients, self.n_events, self.convergence
        )
    }
}

#[pymethods]
impl FineGrayResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        z_scores: Vec<f64>,
        p_values: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        variance_matrix: Vec<Vec<f64>>,
        log_likelihood: f64,
        log_likelihood_null: f64,
        n_events: usize,
        n_competing: usize,
        n_censored: usize,
        n_observations: usize,
        event_type: i32,
        convergence: bool,
        iterations: usize,
    ) -> Self {
        Self {
            coefficients,
            std_errors,
            z_scores,
            p_values,
            ci_lower,
            ci_upper,
            variance_matrix,
            log_likelihood,
            log_likelihood_null,
            n_events,
            n_competing,
            n_censored,
            n_observations,
            event_type,
            convergence,
            iterations,
        }
    }

    fn hazard_ratio(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|&coefficient| exp_clamped(coefficient))
            .collect()
    }

    fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Fine-Gray Subdistribution Hazard Model\n");
        s.push_str("======================================\n\n");
        s.push_str(&format!(
            "N={}, Events={}, Competing={}, Censored={}\n",
            self.n_observations, self.n_events, self.n_competing, self.n_censored
        ));
        s.push_str(&format!("Event type: {}\n\n", self.event_type));
        s.push_str("Coefficients:\n");
        s.push_str("  coef      exp(coef)  se(coef)   z        p\n");
        for i in 0..self.coefficients.len() {
            s.push_str(&format!(
                "  {:.4}    {:.4}     {:.4}     {:.3}    {:.4}\n",
                self.coefficients[i],
                exp_clamped(self.coefficients[i]),
                self.std_errors[i],
                self.z_scores[i],
                self.p_values[i]
            ));
        }
        s.push_str(&format!(
            "\nLog-likelihood: {:.4} (null: {:.4})\n",
            self.log_likelihood, self.log_likelihood_null
        ));
        s.push_str(&format!("Converged: {}\n", self.convergence));
        s
    }
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct CompetingRisksCIF {
    pub times: Vec<f64>,
    pub cif: Vec<f64>,
    pub variance: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_events: Vec<usize>,
    pub event_type: i32,
}

impl fmt::Display for CompetingRisksCIF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompetingRisksCIF(event_type={}, n_times={})",
            self.event_type,
            self.times.len()
        )
    }
}

#[pymethods]
impl CompetingRisksCIF {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        times: Vec<f64>,
        cif: Vec<f64>,
        variance: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        n_risk: Vec<usize>,
        n_events: Vec<usize>,
        event_type: i32,
    ) -> Self {
        Self {
            times,
            cif,
            variance,
            ci_lower,
            ci_upper,
            n_risk,
            n_events,
            event_type,
        }
    }
}

pub(crate) fn finegray_regression_core(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> FineGrayResult {
    let n = time.len();
    let p = if n > 0 && !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };

    if n == 0 || p == 0 {
        return FineGrayResult {
            coefficients: vec![],
            std_errors: vec![],
            z_scores: vec![],
            p_values: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            variance_matrix: vec![],
            log_likelihood: 0.0,
            log_likelihood_null: 0.0,
            n_events: 0,
            n_competing: 0,
            n_censored: 0,
            n_observations: 0,
            event_type,
            convergence: false,
            iterations: 0,
        };
    }

    let n_events = status.iter().filter(|&&s| s == event_type).count();
    let n_competing = status
        .iter()
        .filter(|&&s| s != 0 && s != event_type)
        .count();
    let n_censored = status.iter().filter(|&&s| s == 0).count();

    let (km_times, km_values) = compute_censoring_km(time, status);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut iterations = 0;

    let event_indices: Vec<usize> = indices
        .iter()
        .filter(|&&i| status[i] == event_type)
        .copied()
        .collect();

    let log_likelihood_null = compute_log_likelihood(
        &event_indices,
        &vec![0.0; p],
        time,
        status,
        covariates,
        event_type,
        &km_times,
        &km_values,
    );

    for iter in 0..max_iter {
        iterations = iter + 1;

        let (gradient, hessian, _ll) = compute_gradient_hessian(
            &event_indices,
            &beta,
            time,
            status,
            covariates,
            event_type,
            &km_times,
            &km_values,
        );

        let neg_hessian: Vec<Vec<f64>> = hessian
            .iter()
            .map(|row| row.iter().map(|&x| -x).collect())
            .collect();

        let hess_inv = match invert_matrix(&neg_hessian) {
            Some(inv) => inv,
            None => break,
        };

        let mut delta = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                delta[i] += hess_inv[i][j] * gradient[j];
            }
        }

        let max_delta = delta.iter().map(|&d| d.abs()).fold(0.0, f64::max);

        for i in 0..p {
            beta[i] += delta[i];
        }

        if max_delta < eps {
            converged = true;
            break;
        }
    }

    let log_likelihood = compute_log_likelihood(
        &event_indices,
        &beta,
        time,
        status,
        covariates,
        event_type,
        &km_times,
        &km_values,
    );

    let (_, hessian, _) = compute_gradient_hessian(
        &event_indices,
        &beta,
        time,
        status,
        covariates,
        event_type,
        &km_times,
        &km_values,
    );

    let neg_hessian: Vec<Vec<f64>> = hessian
        .iter()
        .map(|row| row.iter().map(|&x| -x).collect())
        .collect();

    let variance_matrix = invert_matrix(&neg_hessian).unwrap_or_else(|| vec![vec![0.0; p]; p]);

    let std_errors: Vec<f64> = (0..p)
        .map(|i| variance_matrix[i][i].max(0.0).sqrt())
        .collect();

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR {
                b / se
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let (ci_lower, ci_upper) = normal_ci_bounds_95(&beta, &std_errors);

    FineGrayResult {
        coefficients: beta,
        std_errors,
        z_scores,
        p_values,
        ci_lower,
        ci_upper,
        variance_matrix,
        log_likelihood,
        log_likelihood_null,
        n_events,
        n_competing,
        n_censored,
        n_observations: n,
        event_type,
        convergence: converged,
        iterations,
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_log_likelihood(
    event_indices: &[usize],
    beta: &[f64],
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    km_times: &[f64],
    km_values: &[f64],
) -> f64 {
    let n = time.len();
    let p = beta.len();

    let mut ll = 0.0;

    for &i in event_indices {
        let t_i = time[i];

        let mut eta_i = 0.0;
        for k in 0..p {
            eta_i += beta[k] * covariates[i][k];
        }

        let mut sum_exp_eta = 0.0;
        for j in 0..n {
            let in_risk_set = if status[j] == 0 || status[j] == event_type {
                time[j] >= t_i
            } else {
                true
            };

            if in_risk_set {
                let mut eta_j = 0.0;
                for k in 0..p {
                    eta_j += beta[k] * covariates[j][k];
                }

                let weight = if status[j] != 0 && status[j] != event_type && time[j] < t_i {
                    let g_ti = km_step_prob_at(t_i, km_times, km_values).max(IPCW_SURVIVAL_FLOOR);
                    let g_tj =
                        km_step_prob_at(time[j], km_times, km_values).max(IPCW_SURVIVAL_FLOOR);
                    g_ti / g_tj
                } else {
                    1.0
                };

                sum_exp_eta += weight * exp_clamped(eta_j);
            }
        }

        ll += eta_i - sum_exp_eta.ln();
    }

    ll
}

#[allow(clippy::too_many_arguments)]
fn compute_gradient_hessian(
    event_indices: &[usize],
    beta: &[f64],
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    km_times: &[f64],
    km_values: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n = time.len();
    let p = beta.len();

    let mut gradient = vec![0.0; p];
    let mut hessian = vec![vec![0.0; p]; p];
    let mut ll = 0.0;

    let compute_event_contribution = |i: usize| -> (Vec<f64>, Vec<Vec<f64>>, f64) {
        let t_i = time[i];
        let mut local_grad = vec![0.0; p];
        let mut local_hess = vec![vec![0.0; p]; p];

        let mut eta_i = 0.0;
        for k in 0..p {
            eta_i += beta[k] * covariates[i][k];
        }

        let mut s0 = 0.0;
        let mut s1 = vec![0.0; p];
        let mut s2 = vec![vec![0.0; p]; p];

        for j in 0..n {
            let in_risk_set = if status[j] == 0 || status[j] == event_type {
                time[j] >= t_i
            } else {
                true
            };

            if in_risk_set {
                let mut eta_j = 0.0;
                for k in 0..p {
                    eta_j += beta[k] * covariates[j][k];
                }

                let weight = if status[j] != 0 && status[j] != event_type && time[j] < t_i {
                    let g_ti = km_step_prob_at(t_i, km_times, km_values).max(IPCW_SURVIVAL_FLOOR);
                    let g_tj =
                        km_step_prob_at(time[j], km_times, km_values).max(IPCW_SURVIVAL_FLOOR);
                    g_ti / g_tj
                } else {
                    1.0
                };

                let exp_eta = exp_clamped(eta_j);
                let w_exp = weight * exp_eta;

                s0 += w_exp;

                for k in 0..p {
                    s1[k] += w_exp * covariates[j][k];
                }

                for k in 0..p {
                    for l in 0..p {
                        s2[k][l] += w_exp * covariates[j][k] * covariates[j][l];
                    }
                }
            }
        }

        let local_ll = eta_i - s0.ln();

        for k in 0..p {
            local_grad[k] = covariates[i][k] - s1[k] / s0;
        }

        for k in 0..p {
            for l in 0..p {
                local_hess[k][l] = -(s2[k][l] / s0 - (s1[k] / s0) * (s1[l] / s0));
            }
        }

        (local_grad, local_hess, local_ll)
    };

    if event_indices.len() > PARALLEL_THRESHOLD_LARGE {
        let results: Vec<_> = event_indices
            .par_iter()
            .map(|&i| compute_event_contribution(i))
            .collect();

        for (local_grad, local_hess, local_ll) in results {
            ll += local_ll;
            for k in 0..p {
                gradient[k] += local_grad[k];
            }
            for k in 0..p {
                for l in 0..p {
                    hessian[k][l] += local_hess[k][l];
                }
            }
        }
    } else {
        for &i in event_indices {
            let (local_grad, local_hess, local_ll) = compute_event_contribution(i);
            ll += local_ll;
            for k in 0..p {
                gradient[k] += local_grad[k];
            }
            for k in 0..p {
                for l in 0..p {
                    hessian[k][l] += local_hess[k][l];
                }
            }
        }
    }

    (gradient, hessian, ll)
}

pub(crate) fn competing_risks_cif_core(
    time: &[f64],
    status: &[i32],
    event_type: i32,
    confidence_level: f64,
) -> CompetingRisksCIF {
    let n = time.len();

    if n == 0 {
        return CompetingRisksCIF {
            times: vec![],
            cif: vec![],
            variance: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_risk: vec![],
            n_events: vec![],
            event_type,
        };
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut cif_values = Vec::new();
    let mut variance_values = Vec::new();
    let mut n_risk_values = Vec::new();
    let mut n_events_values = Vec::new();

    let mut km_surv = 1.0;
    let mut cum_inc = 0.0;
    let mut variance = 0.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut n_event_type = 0;
        let mut n_other_events = 0;
        let mut total_at_time = 0;

        while i < n && same_time(time[indices[i]], current_time) {
            let s = status[indices[i]];
            if s == event_type {
                n_event_type += 1;
            } else if s != 0 {
                n_other_events += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if n_event_type > 0 && at_risk > 0 {
            let hazard = n_event_type as f64 / at_risk as f64;
            cum_inc += km_surv * hazard;

            let term1 = if at_risk > n_event_type {
                hazard / (at_risk - n_event_type) as f64
            } else {
                0.0
            };
            variance += km_surv * km_surv * hazard * (1.0 - hazard) / at_risk as f64 + term1;
        }

        let total_events = n_event_type + n_other_events;
        if total_events > 0 && at_risk > 0 {
            km_surv *= 1.0 - total_events as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        cif_values.push(cum_inc);
        variance_values.push(variance);
        n_risk_values.push(at_risk);
        n_events_values.push(n_event_type);

        at_risk -= total_at_time;
    }

    let z = match confidence_level {
        x if (x - 0.90).abs() < 0.01 => Z_SCORE_90,
        x if (x - 0.95).abs() < 0.01 => Z_SCORE_95,
        x if (x - 0.99).abs() < 0.01 => Z_SCORE_99,
        _ => Z_SCORE_95,
    };

    let cif_se: Vec<f64> = variance_values.iter().map(|&v| v.sqrt()).collect();
    let (ci_lower, ci_upper) = clamped_normal_ci_bounds(&cif_values, &cif_se, z, 0.0, 1.0);

    CompetingRisksCIF {
        times: unique_times,
        cif: cif_values,
        variance: variance_values,
        ci_lower,
        ci_upper,
        n_risk: n_risk_values,
        n_events: n_events_values,
        event_type,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, event_type, max_iter=25, eps=1e-9))]
pub fn finegray_regression(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> PyResult<FineGrayResult> {
    validate_finegray_regression_input(&time, &status, &covariates, event_type, max_iter, eps)?;

    Ok(finegray_regression_core(
        &time,
        &status,
        &covariates,
        event_type,
        max_iter,
        eps,
    ))
}

#[pyfunction]
#[pyo3(signature = (time, status, event_type, confidence_level=0.95))]
pub fn competing_risks_cif(
    time: Vec<f64>,
    status: Vec<i32>,
    event_type: i32,
    confidence_level: f64,
) -> PyResult<CompetingRisksCIF> {
    validate_competing_risks_input(&time, &status, event_type, confidence_level)?;

    Ok(competing_risks_cif_core(
        &time,
        &status,
        event_type,
        confidence_level,
    ))
}

fn validate_survival_outcome(time: &[f64], status: &[i32], event_type: i32) -> PyResult<()> {
    if time.is_empty() {
        return Err(value_error("time must not be empty"));
    }
    if time.len() != status.len() {
        return Err(value_error("time and status must have the same length"));
    }
    if event_type <= 0 {
        return Err(value_error("event_type must be positive"));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "time contains non-finite value at index {}",
                idx
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "time contains negative value {} at index {}",
                value, idx
            )));
        }
    }
    for (idx, &value) in status.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "status contains negative value {} at index {}",
                value, idx
            )));
        }
    }
    Ok(())
}

fn validate_finegray_regression_input(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> PyResult<()> {
    validate_survival_outcome(time, status, event_type)?;
    if time.len() != covariates.len() {
        return Err(value_error("time and covariates must have the same length"));
    }
    if covariates.is_empty() || covariates[0].is_empty() {
        return Err(value_error("covariates must not be empty"));
    }
    let p = covariates[0].len();
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != p {
            return Err(value_error(format!(
                "all covariate rows must have the same length (row {} has {} instead of {})",
                row_idx,
                row.len(),
                p
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "covariates contains non-finite value at row {}, column {}",
                    row_idx, col_idx
                )));
            }
        }
    }
    if max_iter == 0 {
        return Err(value_error("max_iter must be positive"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(value_error("eps must be a positive finite value"));
    }
    Ok(())
}

fn validate_competing_risks_input(
    time: &[f64],
    status: &[i32],
    event_type: i32,
    confidence_level: f64,
) -> PyResult<()> {
    validate_survival_outcome(time, status, event_type)?;
    if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
        return Err(value_error(
            "confidence_level must be a finite value between 0 and 1",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finegray_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 2, 1, 0, 2, 1, 0, 1, 2, 1];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let result = finegray_regression_core(&time, &status, &covariates, 1, 25, 1e-9);

        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_errors.len(), 2);
        assert!(result.n_events > 0);
        assert!(result.n_competing > 0);
    }

    #[test]
    fn test_finegray_no_competing() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.2]).collect();

        let result = finegray_regression_core(&time, &status, &covariates, 1, 25, 1e-9);

        assert_eq!(result.n_competing, 0);
        assert!(result.n_events > 0);
    }

    #[test]
    fn test_competing_risks_cif_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 2, 1, 0, 2, 1, 0, 1];

        let result = competing_risks_cif_core(&time, &status, 1, 0.95);

        assert!(!result.times.is_empty());
        assert_eq!(result.times.len(), result.cif.len());
        for &c in &result.cif {
            assert!((0.0..=1.0).contains(&c));
        }
        for i in 1..result.cif.len() {
            assert!(result.cif[i] >= result.cif[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_competing_risks_cif_multiple_types() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 2, 1, 2, 0];

        let cif1 = competing_risks_cif_core(&time, &status, 1, 0.95);
        let cif2 = competing_risks_cif_core(&time, &status, 2, 0.95);

        assert!(cif1.cif.last().unwrap_or(&0.0) > &0.0);
        assert!(cif2.cif.last().unwrap_or(&0.0) > &0.0);

        let total_cif = cif1.cif.last().unwrap_or(&0.0) + cif2.cif.last().unwrap_or(&0.0);
        assert!(total_cif <= 1.0 + 1e-10);
    }

    #[test]
    fn test_competing_risks_cif_empty() {
        let result = competing_risks_cif_core(&[], &[], 1, 0.95);
        assert!(result.times.is_empty());
        assert!(result.cif.is_empty());
    }

    #[test]
    fn test_censoring_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let (km_times, km_values) = compute_censoring_km(&time, &status);

        assert!(!km_times.is_empty());
        assert_eq!(km_times.len(), km_values.len());

        for &v in &km_values {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_finegray_public_api_rejects_malformed_inputs() {
        assert!(
            finegray_regression(vec![], vec![], vec![], 1, 25, 1e-9)
                .unwrap_err()
                .to_string()
                .contains("time must not be empty")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![f64::NAN]], 1, 25, 1e-9,)
                .unwrap_err()
                .to_string()
                .contains("covariates contains non-finite")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![0.0]], 1, 0, 1e-9)
                .unwrap_err()
                .to_string()
                .contains("max_iter must be positive")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![0.0]], 1, 25, f64::INFINITY)
                .unwrap_err()
                .to_string()
                .contains("eps must be")
        );
    }

    #[test]
    fn test_competing_risks_cif_public_api_rejects_malformed_inputs() {
        assert!(
            competing_risks_cif(vec![1.0], vec![], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("same length")
        );
        assert!(
            competing_risks_cif(vec![f64::INFINITY], vec![1], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("time contains non-finite")
        );
        assert!(
            competing_risks_cif(vec![1.0], vec![-1], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("status contains negative")
        );
        assert!(
            competing_risks_cif(vec![1.0], vec![1], 1, 1.0)
                .unwrap_err()
                .to_string()
                .contains("confidence_level")
        );
    }

    #[test]
    fn test_hazard_ratios_are_clamped_for_large_coefficients() {
        let result = FineGrayResult::new(
            vec![1_000.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![999.0],
            vec![1001.0],
            vec![vec![1.0]],
            0.0,
            0.0,
            1,
            0,
            0,
            1,
            1,
            true,
            1,
        );

        assert!(result.hazard_ratio()[0].is_finite());
        assert!(result.summary().contains("Fine-Gray"));
    }
}
