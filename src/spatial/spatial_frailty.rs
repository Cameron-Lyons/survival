#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_swap,
    clippy::manual_range_contains
)]

use crate::utilities::statistical::normal_cdf;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SpatialFrailtyResult {
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
    pub spatial_frailties: Vec<f64>,
    #[pyo3(get)]
    pub frailty_variance: f64,
    #[pyo3(get)]
    pub spatial_correlation: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub dic: f64,
    #[pyo3(get)]
    pub n_regions: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum SpatialCorrelationStructure {
    CAR,
    SAR,
    Exponential,
    Matern,
}

#[pymethods]
impl SpatialCorrelationStructure {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "car" | "conditional_autoregressive" => Ok(SpatialCorrelationStructure::CAR),
            "sar" | "simultaneous_autoregressive" => Ok(SpatialCorrelationStructure::SAR),
            "exponential" | "exp" => Ok(SpatialCorrelationStructure::Exponential),
            "matern" => Ok(SpatialCorrelationStructure::Matern),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown spatial correlation structure",
            )),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    x,
    n_obs,
    n_vars,
    region_id,
    adjacency_matrix,
    n_regions,
    correlation_structure=SpatialCorrelationStructure::CAR,
    max_iter=100,
    tol=1e-6
))]
pub fn spatial_frailty_model(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    region_id: Vec<usize>,
    adjacency_matrix: Vec<f64>,
    n_regions: usize,
    correlation_structure: SpatialCorrelationStructure,
    max_iter: usize,
    tol: f64,
) -> PyResult<SpatialFrailtyResult> {
    if time.len() != n_obs || status.len() != n_obs || region_id.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and region_id must have length n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x must have length n_obs * n_vars",
        ));
    }
    if adjacency_matrix.len() != n_regions * n_regions {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "adjacency_matrix must have length n_regions * n_regions",
        ));
    }

    let precision_matrix = match correlation_structure {
        SpatialCorrelationStructure::CAR => {
            compute_car_precision(&adjacency_matrix, n_regions, 0.9)
        }
        SpatialCorrelationStructure::SAR => {
            compute_sar_precision(&adjacency_matrix, n_regions, 0.5)
        }
        SpatialCorrelationStructure::Exponential | SpatialCorrelationStructure::Matern => {
            compute_distance_precision(&adjacency_matrix, n_regions, 1.0, correlation_structure)
        }
    };

    let mut beta = vec![0.0; n_vars];
    let mut frailties = vec![0.0; n_regions];
    let mut frailty_variance = 1.0;
    let mut spatial_rho = 0.5;

    let mut converged = false;
    let mut log_lik = f64::NEG_INFINITY;

    for iter in 0..max_iter {
        let (new_beta, new_frailties, new_var, new_rho, new_loglik) = em_step(
            &time,
            &status,
            &x,
            &region_id,
            n_obs,
            n_vars,
            n_regions,
            &beta,
            &frailties,
            frailty_variance,
            spatial_rho,
            &adjacency_matrix,
            &precision_matrix,
            correlation_structure,
        );

        let max_change = beta
            .iter()
            .zip(new_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        beta = new_beta;
        frailties = new_frailties;
        frailty_variance = new_var;
        spatial_rho = new_rho;

        if (new_loglik - log_lik).abs() < tol && max_change < tol {
            converged = true;
            log_lik = new_loglik;
            break;
        }
        log_lik = new_loglik;
    }

    let se = compute_standard_errors(
        &time, &status, &x, &region_id, &beta, &frailties, n_obs, n_vars,
    );

    let hazard_ratios: Vec<f64> = beta.iter().map(|&b| b.exp()).collect();

    let z = 1.96;
    let hr_ci_lower: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(&b, &se)| (b - z * se).exp())
        .collect();

    let hr_ci_upper: Vec<f64> = beta
        .iter()
        .zip(se.iter())
        .map(|(&b, &se)| (b + z * se).exp())
        .collect();

    let p_d = 2.0 * (log_lik - compute_saturated_loglik(&time, &status));
    let dic = -2.0 * log_lik + 2.0 * p_d;

    Ok(SpatialFrailtyResult {
        coefficients: beta,
        std_errors: se,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        spatial_frailties: frailties,
        frailty_variance,
        spatial_correlation: spatial_rho,
        log_likelihood: log_lik,
        dic,
        n_regions,
        converged,
    })
}

fn compute_car_precision(adjacency: &[f64], n: usize, rho: f64) -> Vec<f64> {
    let mut precision = vec![0.0; n * n];

    for i in 0..n {
        let n_neighbors: f64 = (0..n).map(|j| adjacency[i * n + j]).sum();

        precision[i * n + i] = n_neighbors;

        for j in 0..n {
            if i != j && adjacency[i * n + j] > 0.0 {
                precision[i * n + j] = -rho * adjacency[i * n + j];
            }
        }
    }

    precision
}

fn compute_sar_precision(adjacency: &[f64], n: usize, rho: f64) -> Vec<f64> {
    let mut precision = vec![0.0; n * n];

    let row_sums: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| adjacency[i * n + j]).sum::<f64>().max(1.0))
        .collect();

    for i in 0..n {
        for j in 0..n {
            let w_ij = adjacency[i * n + j] / row_sums[i];
            let delta = if i == j { 1.0 } else { 0.0 };

            let i_rho_w = delta - rho * w_ij;

            for k in 0..n {
                let w_kj = adjacency[k * n + j] / row_sums[k];
                let i_rho_w_t = if k == i { 1.0 } else { 0.0 } - rho * w_kj;
                precision[i * n + j] += i_rho_w * i_rho_w_t;
            }
        }
    }

    precision
}

fn compute_distance_precision(
    distances: &[f64],
    n: usize,
    range_param: f64,
    structure: SpatialCorrelationStructure,
) -> Vec<f64> {
    let mut covariance = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..n {
            let d = distances[i * n + j];
            let cov = match structure {
                SpatialCorrelationStructure::Exponential => (-d / range_param).exp(),
                SpatialCorrelationStructure::Matern => {
                    let nu: f64 = 1.5;
                    let scaled_d = (2.0 * nu).sqrt() * d / range_param;
                    if scaled_d < 1e-10 {
                        1.0
                    } else {
                        let term = (1.0 + scaled_d) * (-scaled_d).exp();
                        term.max(0.0)
                    }
                }
                _ => {
                    if i == j {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            covariance[i * n + j] = cov;
        }
    }

    invert_matrix(&covariance, n)
}

fn invert_matrix(a: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * 2 * n];

    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * 2 * n + i].abs() > aug[max_row * 2 * n + i].abs() {
                max_row = k;
            }
        }

        for j in 0..(2 * n) {
            let temp = aug[i * 2 * n + j];
            aug[i * 2 * n + j] = aug[max_row * 2 * n + j];
            aug[max_row * 2 * n + j] = temp;
        }

        let pivot = aug[i * 2 * n + i];
        if pivot.abs() < 1e-10 {
            continue;
        }

        for j in 0..(2 * n) {
            aug[i * 2 * n + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..(2 * n) {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    inv
}

fn em_step(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    region_id: &[usize],
    n_obs: usize,
    n_vars: usize,
    n_regions: usize,
    beta: &[f64],
    frailties: &[f64],
    frailty_var: f64,
    spatial_rho: f64,
    adjacency: &[f64],
    precision: &[f64],
    structure: SpatialCorrelationStructure,
) -> (Vec<f64>, Vec<f64>, f64, f64, f64) {
    let mut new_beta = beta.to_vec();
    let mut new_frailties = frailties.to_vec();

    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..5 {
        let eta: Vec<f64> = (0..n_obs)
            .map(|i| {
                let mut e = new_frailties[region_id[i]];
                for j in 0..n_vars {
                    e += x[i * n_vars + j] * new_beta[j];
                }
                e.clamp(-500.0, 500.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient = vec![0.0; n_vars];
        let mut hessian_diag = vec![0.0; n_vars];

        let mut risk_sum = 0.0;
        let mut weighted_x = vec![0.0; n_vars];
        let mut weighted_x_sq = vec![0.0; n_vars];

        for &i in &indices {
            risk_sum += exp_eta[i];
            for j in 0..n_vars {
                weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
                weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j].powi(2);
            }

            if status[i] == 1 && risk_sum > 0.0 {
                for j in 0..n_vars {
                    let x_bar = weighted_x[j] / risk_sum;
                    let x_sq_bar = weighted_x_sq[j] / risk_sum;
                    gradient[j] += x[i * n_vars + j] - x_bar;
                    hessian_diag[j] += x_sq_bar - x_bar.powi(2);
                }
            }
        }

        for j in 0..n_vars {
            if hessian_diag[j].abs() > 1e-10 {
                new_beta[j] += 0.5 * gradient[j] / hessian_diag[j];
            }
        }
    }

    for r in 0..n_regions {
        let region_obs: Vec<usize> = (0..n_obs).filter(|&i| region_id[i] == r).collect();

        if region_obs.is_empty() {
            continue;
        }

        let mut frailty_grad = 0.0;
        let mut frailty_hess = 0.0;

        for &i in &region_obs {
            let mut lin_pred = new_frailties[r];
            for j in 0..n_vars {
                lin_pred += x[i * n_vars + j] * new_beta[j];
            }
            let exp_lp = lin_pred.clamp(-500.0, 500.0).exp();

            if status[i] == 1 {
                frailty_grad += 1.0;
            }
            frailty_grad -= exp_lp * time[i];
            frailty_hess += exp_lp * time[i];
        }

        let prior_precision = precision[r * n_regions + r] / frailty_var;
        frailty_grad -= prior_precision * new_frailties[r];
        frailty_hess += prior_precision;

        if frailty_hess.abs() > 1e-10 {
            new_frailties[r] += 0.5 * frailty_grad / frailty_hess;
        }
    }

    let mean_frailty: f64 = new_frailties.iter().sum::<f64>() / n_regions as f64;
    for f in new_frailties.iter_mut() {
        *f -= mean_frailty;
    }

    let new_frailty_var = new_frailties.iter().map(|&f| f.powi(2)).sum::<f64>() / n_regions as f64;
    let new_frailty_var = new_frailty_var.max(0.01);

    let new_spatial_rho = spatial_rho;

    let log_lik = compute_log_likelihood(
        time,
        status,
        x,
        region_id,
        &new_beta,
        &new_frailties,
        n_obs,
        n_vars,
    );

    (
        new_beta,
        new_frailties,
        new_frailty_var,
        new_spatial_rho,
        log_lik,
    )
}

fn compute_log_likelihood(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    region_id: &[usize],
    beta: &[f64],
    frailties: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> f64 {
    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut e = frailties[region_id[i]];
            for j in 0..n_vars {
                e += x[i * n_vars + j] * beta[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut log_lik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += exp_eta[i];
        if status[i] == 1 && risk_sum > 0.0 {
            log_lik += eta[i] - risk_sum.ln();
        }
    }

    log_lik
}

fn compute_saturated_loglik(time: &[f64], status: &[i32]) -> f64 {
    0.0
}

fn compute_standard_errors(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    region_id: &[usize],
    beta: &[f64],
    frailties: &[f64],
    n_obs: usize,
    n_vars: usize,
) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..n_obs).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut e = frailties[region_id[i]];
            for j in 0..n_vars {
                e += x[i * n_vars + j] * beta[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut info_diag = vec![0.0; n_vars];
    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; n_vars];
    let mut weighted_x_sq = vec![0.0; n_vars];

    for &i in &indices {
        risk_sum += exp_eta[i];
        for j in 0..n_vars {
            weighted_x[j] += exp_eta[i] * x[i * n_vars + j];
            weighted_x_sq[j] += exp_eta[i] * x[i * n_vars + j].powi(2);
        }

        if status[i] == 1 && risk_sum > 0.0 {
            for j in 0..n_vars {
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;
                info_diag[j] += x_sq_bar - x_bar.powi(2);
            }
        }
    }

    info_diag
        .iter()
        .map(|&info| {
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

#[pyfunction]
pub fn compute_spatial_smoothed_rates(
    observed_events: Vec<f64>,
    expected_events: Vec<f64>,
    adjacency_matrix: Vec<f64>,
    n_regions: usize,
    smoothing_param: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if observed_events.len() != n_regions || expected_events.len() != n_regions {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "observed_events and expected_events must have length n_regions",
        ));
    }

    let raw_rates: Vec<f64> = observed_events
        .iter()
        .zip(expected_events.iter())
        .map(|(&o, &e)| if e > 0.0 { o / e } else { 0.0 })
        .collect();

    let mut smoothed_rates = vec![0.0; n_regions];

    for i in 0..n_regions {
        let mut weighted_sum = raw_rates[i] * (1.0 - smoothing_param);
        let mut weight_sum = 1.0 - smoothing_param;

        for j in 0..n_regions {
            if adjacency_matrix[i * n_regions + j] > 0.0 {
                weighted_sum +=
                    smoothing_param * adjacency_matrix[i * n_regions + j] * raw_rates[j];
                weight_sum += smoothing_param * adjacency_matrix[i * n_regions + j];
            }
        }

        smoothed_rates[i] = weighted_sum / weight_sum;
    }

    Ok((raw_rates, smoothed_rates))
}

#[pyfunction]
pub fn moran_i_test(
    values: Vec<f64>,
    adjacency_matrix: Vec<f64>,
    n_regions: usize,
) -> PyResult<(f64, f64, f64)> {
    if values.len() != n_regions {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "values must have length n_regions",
        ));
    }

    let mean_val: f64 = values.iter().sum::<f64>() / n_regions as f64;
    let deviations: Vec<f64> = values.iter().map(|&v| v - mean_val).collect();

    let mut w_sum = 0.0;
    let mut numerator = 0.0;

    for i in 0..n_regions {
        for j in 0..n_regions {
            let w = adjacency_matrix[i * n_regions + j];
            w_sum += w;
            numerator += w * deviations[i] * deviations[j];
        }
    }

    let denominator: f64 = deviations.iter().map(|&d| d.powi(2)).sum();

    let moran_i = if denominator > 0.0 && w_sum > 0.0 {
        (n_regions as f64 / w_sum) * (numerator / denominator)
    } else {
        0.0
    };

    let expected_i = -1.0 / (n_regions as f64 - 1.0);

    let var_i = (n_regions as f64).powi(2) / (w_sum.powi(2) * (n_regions as f64 - 1.0));
    let z_score = (moran_i - expected_i) / var_i.sqrt();

    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    Ok((moran_i, z_score, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_frailty_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0];
        let x = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let region_id = vec![0, 0, 1, 1, 2, 2];
        let adjacency = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let result = spatial_frailty_model(
            time,
            status,
            x,
            6,
            1,
            region_id,
            adjacency,
            3,
            SpatialCorrelationStructure::CAR,
            50,
            1e-4,
        )
        .unwrap();

        assert_eq!(result.n_regions, 3);
        assert_eq!(result.spatial_frailties.len(), 3);
    }

    #[test]
    fn test_moran_i() {
        let values = vec![1.0, 2.0, 1.5, 2.5];
        let adjacency = vec![
            0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,
        ];

        let (moran_i, z_score, p_value) = moran_i_test(values, adjacency, 4).unwrap();

        assert!(moran_i >= -1.0 && moran_i <= 1.0);
        assert!(p_value >= 0.0 && p_value <= 1.0);
    }
}
