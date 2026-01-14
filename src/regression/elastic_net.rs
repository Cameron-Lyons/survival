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
pub enum PenaltyType {
    Lasso,
    Ridge,
    ElasticNet,
}

#[pymethods]
impl PenaltyType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "lasso" | "l1" => Ok(PenaltyType::Lasso),
            "ridge" | "l2" => Ok(PenaltyType::Ridge),
            "elastic_net" | "elasticnet" | "enet" => Ok(PenaltyType::ElasticNet),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown penalty type. Use 'lasso', 'ridge', or 'elastic_net'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ElasticNetConfig {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub l1_ratio: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub standardize: bool,
    #[pyo3(get, set)]
    pub warm_start: bool,
}

#[pymethods]
impl ElasticNetConfig {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-7, standardize=true, warm_start=false))]
    pub fn new(
        alpha: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
        standardize: bool,
        warm_start: bool,
    ) -> PyResult<Self> {
        if alpha < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "l1_ratio must be between 0 and 1",
            ));
        }
        Ok(ElasticNetConfig {
            alpha,
            l1_ratio,
            max_iter,
            tol,
            standardize,
            warm_start,
        })
    }

    #[staticmethod]
    pub fn lasso(alpha: f64) -> PyResult<Self> {
        Self::new(alpha, 1.0, 1000, 1e-7, true, false)
    }

    #[staticmethod]
    pub fn ridge(alpha: f64) -> PyResult<Self> {
        Self::new(alpha, 0.0, 1000, 1e-7, true, false)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ElasticNetCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub nonzero_indices: Vec<usize>,
    #[pyo3(get)]
    pub lambda_used: f64,
    #[pyo3(get)]
    pub l1_ratio: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub deviance: f64,
    #[pyo3(get)]
    pub df: f64,
    #[pyo3(get)]
    pub scale_factors: Option<Vec<f64>>,
    #[pyo3(get)]
    pub intercept: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ElasticNetCoxPath {
    #[pyo3(get)]
    pub lambdas: Vec<f64>,
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub deviances: Vec<f64>,
    #[pyo3(get)]
    pub df: Vec<f64>,
    #[pyo3(get)]
    pub n_iters: Vec<usize>,
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

fn standardize_matrix(x: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut means = vec![0.0; p];
    let mut sds = vec![1.0; p];
    let mut x_std = x.to_vec();

    for j in 0..p {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..n {
            let val = x[i * p + j];
            sum += val;
            sum_sq += val * val;
        }
        means[j] = sum / n as f64;
        let var = sum_sq / n as f64 - means[j] * means[j];
        sds[j] = var.sqrt().max(1e-10);

        for i in 0..n {
            x_std[i * p + j] = (x[i * p + j] - means[j]) / sds[j];
        }
    }

    (x_std, means, sds)
}

fn compute_cox_gradient_hessian(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    beta: &[f64],
    offset: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut gradient = vec![0.0; p];
    let mut hessian_diag = vec![0.0; p];

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = offset[i];
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

    for &i in &indices {
        let w = weights[i] * exp_eta[i];
        risk_sum += w;

        for j in 0..p {
            let xij = x[i * p + j];
            weighted_x[j] += w * xij;
            weighted_x_sq[j] += w * xij * xij;
        }

        if status[i] == 1 && risk_sum > 0.0 {
            for j in 0..p {
                let xij = x[i * p + j];
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;

                gradient[j] += weights[i] * (xij - x_bar);
                hessian_diag[j] += weights[i] * (x_sq_bar - x_bar * x_bar);
            }
        }
    }

    (gradient, hessian_diag)
}

fn compute_cox_deviance(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    beta: &[f64],
    offset: &[f64],
) -> f64 {
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = offset[i];
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

    let mut loglik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += weights[i] * exp_eta[i];

        if status[i] == 1 && risk_sum > 0.0 {
            loglik += weights[i] * (eta[i] - risk_sum.ln());
        }
    }

    -2.0 * loglik
}

#[allow(clippy::too_many_arguments)]
fn coordinate_descent_cox(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    offset: &[f64],
    lambda: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    beta_init: Option<&[f64]>,
) -> (Vec<f64>, usize, bool) {
    let mut beta = beta_init
        .map(|b| b.to_vec())
        .unwrap_or_else(|| vec![0.0; p]);

    let l1_penalty = lambda * l1_ratio;
    let l2_penalty = lambda * (1.0 - l1_ratio);

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let beta_old = beta.clone();

        let (gradient, hessian_diag) =
            compute_cox_gradient_hessian(x, n, p, time, status, weights, &beta, offset);

        for j in 0..p {
            let h_jj = hessian_diag[j] + l2_penalty;
            if h_jj.abs() < 1e-10 {
                continue;
            }

            let z = gradient[j] + hessian_diag[j] * beta[j];
            beta[j] = soft_threshold(z, l1_penalty) / h_jj;
        }

        let max_change: f64 = beta
            .iter()
            .zip(beta_old.iter())
            .map(|(&b, &b_old)| (b - b_old).abs())
            .fold(0.0, f64::max);

        if max_change < tol {
            converged = true;
            break;
        }
    }

    (beta, n_iter, converged)
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config, weights=None, offset=None))]
pub fn elastic_net_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: &ElasticNetConfig,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
) -> PyResult<ElasticNetCoxResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }
    if time.len() != n_obs || status.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have length n_obs",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);
    let off = offset.unwrap_or_else(|| vec![0.0; n_obs]);

    let (x_std, _means, sds) = if config.standardize {
        standardize_matrix(&x, n_obs, n_vars)
    } else {
        (x.clone(), vec![0.0; n_vars], vec![1.0; n_vars])
    };

    let (beta_std, n_iter, converged) = coordinate_descent_cox(
        &x_std,
        n_obs,
        n_vars,
        &time,
        &status,
        &wt,
        &off,
        config.alpha,
        config.l1_ratio,
        config.max_iter,
        config.tol,
        None,
    );

    let coefficients: Vec<f64> = if config.standardize {
        beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect()
    } else {
        beta_std
    };

    let nonzero_indices: Vec<usize> = coefficients
        .iter()
        .enumerate()
        .filter(|(_, c)| c.abs() > 1e-10)
        .map(|(i, _)| i)
        .collect();

    let df = nonzero_indices.len() as f64;
    let deviance =
        compute_cox_deviance(&x, n_obs, n_vars, &time, &status, &wt, &coefficients, &off);

    Ok(ElasticNetCoxResult {
        coefficients,
        nonzero_indices,
        lambda_used: config.alpha,
        l1_ratio: config.l1_ratio,
        n_iter,
        converged,
        deviance,
        df,
        scale_factors: if config.standardize { Some(sds) } else { None },
        intercept: 0.0,
    })
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, l1_ratio=0.5, n_lambda=100, lambda_min_ratio=None, weights=None, max_iter=1000, tol=1e-7))]
pub fn elastic_net_cox_path(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    l1_ratio: f64,
    n_lambda: usize,
    lambda_min_ratio: Option<f64>,
    weights: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<ElasticNetCoxPath> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);
    let off = vec![0.0; n_obs];

    let (x_std, _means, sds) = standardize_matrix(&x, n_obs, n_vars);

    let beta_zero = vec![0.0; n_vars];
    let (gradient, _) =
        compute_cox_gradient_hessian(&x_std, n_obs, n_vars, &time, &status, &wt, &beta_zero, &off);

    let lambda_max =
        gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / (n_obs as f64 * l1_ratio.max(0.001));

    let min_ratio = lambda_min_ratio.unwrap_or(if n_obs < n_vars { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    let lambdas: Vec<f64> = (0..n_lambda)
        .map(|i| {
            let frac = i as f64 / (n_lambda - 1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect();

    let mut all_coefficients = Vec::with_capacity(n_lambda);
    let mut all_deviances = Vec::with_capacity(n_lambda);
    let mut all_df = Vec::with_capacity(n_lambda);
    let mut all_n_iters = Vec::with_capacity(n_lambda);

    let mut beta_warm = vec![0.0; n_vars];

    for &lambda in &lambdas {
        let (beta_std, n_iter, _converged) = coordinate_descent_cox(
            &x_std,
            n_obs,
            n_vars,
            &time,
            &status,
            &wt,
            &off,
            lambda,
            l1_ratio,
            max_iter,
            tol,
            Some(&beta_warm),
        );

        beta_warm = beta_std.clone();

        let coefficients: Vec<f64> = beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect();

        let df = coefficients.iter().filter(|&&c| c.abs() > 1e-10).count() as f64;
        let deviance =
            compute_cox_deviance(&x, n_obs, n_vars, &time, &status, &wt, &coefficients, &off);

        all_coefficients.push(coefficients);
        all_deviances.push(deviance);
        all_df.push(df);
        all_n_iters.push(n_iter);
    }

    Ok(ElasticNetCoxPath {
        lambdas,
        coefficients: all_coefficients,
        deviances: all_deviances,
        df: all_df,
        n_iters: all_n_iters,
    })
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, l1_ratio=0.5, n_lambda=100, n_folds=10, weights=None))]
pub fn elastic_net_cox_cv(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    l1_ratio: f64,
    n_lambda: usize,
    n_folds: usize,
    weights: Option<Vec<f64>>,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let path = elastic_net_cox_path(
        x.clone(),
        n_obs,
        n_vars,
        time.clone(),
        status.clone(),
        l1_ratio,
        n_lambda,
        None,
        weights.clone(),
        1000,
        1e-7,
    )?;

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);

    let fold_assign: Vec<usize> = (0..n_obs).map(|i| i % n_folds).collect();

    let cv_deviances: Vec<Vec<f64>> = path
        .lambdas
        .par_iter()
        .map(|&lambda| {
            let mut fold_devs = Vec::with_capacity(n_folds);
            let x_local = &x;

            for fold in 0..n_folds {
                let train_idx: Vec<usize> =
                    (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();
                let test_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] == fold).collect();

                if train_idx.is_empty() || test_idx.is_empty() {
                    continue;
                }

                let train_x: Vec<f64> = {
                    let mut result = Vec::with_capacity(train_idx.len() * n_vars);
                    for &i in &train_idx {
                        for j in 0..n_vars {
                            result.push(x_local[i * n_vars + j]);
                        }
                    }
                    result
                };
                let train_time: Vec<f64> = train_idx.iter().map(|&i| time[i]).collect();
                let train_status: Vec<i32> = train_idx.iter().map(|&i| status[i]).collect();
                let train_wt: Vec<f64> = train_idx.iter().map(|&i| wt[i]).collect();

                let config =
                    ElasticNetConfig::new(lambda, l1_ratio, 1000, 1e-7, true, false).unwrap();
                if let Ok(result) = elastic_net_cox(
                    train_x,
                    train_idx.len(),
                    n_vars,
                    train_time,
                    train_status,
                    &config,
                    Some(train_wt),
                    None,
                ) {
                    let test_x: Vec<f64> = {
                        let mut result = Vec::with_capacity(test_idx.len() * n_vars);
                        for &i in &test_idx {
                            for j in 0..n_vars {
                                result.push(x_local[i * n_vars + j]);
                            }
                        }
                        result
                    };
                    let test_time: Vec<f64> = test_idx.iter().map(|&i| time[i]).collect();
                    let test_status: Vec<i32> = test_idx.iter().map(|&i| status[i]).collect();
                    let test_wt: Vec<f64> = test_idx.iter().map(|&i| wt[i]).collect();
                    let test_off = vec![0.0; test_idx.len()];

                    let dev = compute_cox_deviance(
                        &test_x,
                        test_idx.len(),
                        n_vars,
                        &test_time,
                        &test_status,
                        &test_wt,
                        &result.coefficients,
                        &test_off,
                    );
                    fold_devs.push(dev);
                }
            }

            fold_devs
        })
        .collect();

    let mean_deviances: Vec<f64> = cv_deviances
        .iter()
        .map(|devs| {
            if devs.is_empty() {
                f64::INFINITY
            } else {
                devs.iter().sum::<f64>() / devs.len() as f64
            }
        })
        .collect();

    let se_deviances: Vec<f64> = cv_deviances
        .iter()
        .zip(mean_deviances.iter())
        .map(|(devs, &mean)| {
            if devs.len() < 2 {
                f64::INFINITY
            } else {
                let var =
                    devs.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / (devs.len() - 1) as f64;
                (var / devs.len() as f64).sqrt()
            }
        })
        .collect();

    let (min_idx, &min_dev) = mean_deviances
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &f64::INFINITY));

    let lambda_min = path.lambdas[min_idx];

    let threshold = min_dev + se_deviances[min_idx];
    let lambda_1se = mean_deviances
        .iter()
        .enumerate()
        .filter(|(_, d)| **d <= threshold)
        .map(|(i, _)| path.lambdas[i])
        .next()
        .unwrap_or(lambda_min);

    Ok((lambda_min, lambda_1se, mean_deviances, se_deviances))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-10);
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-10);
        assert!((soft_threshold(1.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_net_config() {
        let config = ElasticNetConfig::lasso(0.1).unwrap();
        assert_eq!(config.l1_ratio, 1.0);

        let config = ElasticNetConfig::ridge(0.1).unwrap();
        assert_eq!(config.l1_ratio, 0.0);
    }

    #[test]
    fn test_elastic_net_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let config = ElasticNetConfig::new(0.1, 0.5, 100, 1e-5, true, false).unwrap();

        let result = elastic_net_cox(x, 4, 2, time, status, &config, None, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
    }
}
