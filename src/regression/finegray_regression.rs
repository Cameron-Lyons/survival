use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

use crate::constants::PARALLEL_THRESHOLD_LARGE;
use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
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
        self.coefficients.iter().map(|&c| c.exp()).collect()
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
                self.coefficients[i].exp(),
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
#[pyclass(str, get_all)]
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

fn compute_censoring_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;
        let mut total_at_time = 0;

        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);

        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

fn get_censoring_weight(t: f64, km_times: &[f64], km_values: &[f64]) -> f64 {
    if km_times.is_empty() {
        return 1.0;
    }

    if t < km_times[0] {
        return 1.0;
    }

    let mut left = 0;
    let mut right = km_times.len();

    while left < right {
        let mid = (left + right) / 2;
        if km_times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    let g = if left == 0 { 1.0 } else { km_values[left - 1] };
    g.max(0.01)
}

fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 {
        return None;
    }
    for row in mat {
        if row.len() != n {
            return None;
        }
    }

    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend(vec![0.0; n]);
            new_row[n + i] = 1.0;
            new_row
        })
        .collect();

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-12 {
            return None;
        }

        let pivot = aug[i][i];
        for val in aug[i].iter_mut().take(2 * n) {
            *val /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                let aug_i_clone: Vec<f64> = aug[i].iter().take(2 * n).copied().collect();
                for (j, aug_i_val) in aug_i_clone.iter().enumerate() {
                    aug[k][j] -= factor * aug_i_val;
                }
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

pub fn finegray_regression_core(
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
        .map(|(&b, &se)| if se > 1e-10 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let z_crit = 1.96;
    let ci_lower: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| b - z_crit * se)
        .collect();
    let ci_upper: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| b + z_crit * se)
        .collect();

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
                    let g_ti = get_censoring_weight(t_i, km_times, km_values);
                    let g_tj = get_censoring_weight(time[j], km_times, km_values);
                    g_ti / g_tj
                } else {
                    1.0
                };

                sum_exp_eta += weight * eta_j.exp();
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
                    let g_ti = get_censoring_weight(t_i, km_times, km_values);
                    let g_tj = get_censoring_weight(time[j], km_times, km_values);
                    g_ti / g_tj
                } else {
                    1.0
                };

                let exp_eta = eta_j.exp();
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

pub fn competing_risks_cif_core(
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

        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
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
        x if (x - 0.90).abs() < 0.01 => 1.645,
        x if (x - 0.95).abs() < 0.01 => 1.96,
        x if (x - 0.99).abs() < 0.01 => 2.576,
        _ => 1.96,
    };

    let ci_lower: Vec<f64> = cif_values
        .iter()
        .zip(variance_values.iter())
        .map(|(&c, &v)| (c - z * v.sqrt()).max(0.0))
        .collect();

    let ci_upper: Vec<f64> = cif_values
        .iter()
        .zip(variance_values.iter())
        .map(|(&c, &v)| (c + z * v.sqrt()).min(1.0))
        .collect();

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
    let n = time.len();

    if n != status.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have the same length",
        ));
    }

    if n != covariates.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and covariates must have the same length",
        ));
    }

    if covariates.is_empty() || covariates[0].is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must not be empty",
        ));
    }

    let p = covariates[0].len();
    for (i, row) in covariates.iter().enumerate() {
        if row.len() != p {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "all covariate rows must have the same length (row {} has {} instead of {})",
                i,
                row.len(),
                p
            )));
        }
    }

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
    if time.len() != status.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have the same length",
        ));
    }

    Ok(competing_risks_cif_core(
        &time,
        &status,
        event_type,
        confidence_level,
    ))
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
}
