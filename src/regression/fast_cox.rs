#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum ScreeningRule {
    None,
    Safe,
    Strong,
    EDPP,
}

#[pymethods]
impl ScreeningRule {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "none" => Ok(ScreeningRule::None),
            "safe" => Ok(ScreeningRule::Safe),
            "strong" => Ok(ScreeningRule::Strong),
            "edpp" => Ok(ScreeningRule::EDPP),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown screening rule. Use 'none', 'safe', 'strong', or 'edpp'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FastCoxConfig {
    #[pyo3(get, set)]
    pub lambda: f64,
    #[pyo3(get, set)]
    pub l1_ratio: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub screening: ScreeningRule,
    #[pyo3(get, set)]
    pub working_set_size: Option<usize>,
    #[pyo3(get, set)]
    pub active_set_update_freq: usize,
    #[pyo3(get, set)]
    pub standardize: bool,
    #[pyo3(get, set)]
    pub use_simd: bool,
}

#[pymethods]
impl FastCoxConfig {
    #[new]
    #[pyo3(signature = (
        lambda=0.1,
        l1_ratio=1.0,
        max_iter=1000,
        tol=1e-7,
        screening=ScreeningRule::Strong,
        working_set_size=None,
        active_set_update_freq=10,
        standardize=true,
        use_simd=true
    ))]
    pub fn new(
        lambda: f64,
        l1_ratio: f64,
        max_iter: usize,
        tol: f64,
        screening: ScreeningRule,
        working_set_size: Option<usize>,
        active_set_update_freq: usize,
        standardize: bool,
        use_simd: bool,
    ) -> PyResult<Self> {
        if lambda < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lambda must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "l1_ratio must be between 0 and 1",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        if active_set_update_freq == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "active_set_update_freq must be positive",
            ));
        }

        Ok(FastCoxConfig {
            lambda,
            l1_ratio,
            max_iter,
            tol,
            screening,
            working_set_size,
            active_set_update_freq,
            standardize,
            use_simd,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FastCoxResult {
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
    pub center_values: Option<Vec<f64>>,
    #[pyo3(get)]
    pub screened_out: usize,
    #[pyo3(get)]
    pub active_set_size: usize,
}

#[pymethods]
impl FastCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "FastCoxResult(nonzero={}, iter={}, converged={}, screened_out={})",
            self.nonzero_indices.len(),
            self.n_iter,
            self.converged,
            self.screened_out
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FastCoxPath {
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
    #[pyo3(get)]
    pub converged: Vec<bool>,
}

#[pymethods]
impl FastCoxPath {
    fn __repr__(&self) -> String {
        format!(
            "FastCoxPath(n_lambda={}, converged={})",
            self.lambdas.len(),
            self.converged.iter().filter(|&&c| c).count()
        )
    }
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

struct RiskSetData {
    sorted_indices: Vec<usize>,
    cumsum_exp_eta: Vec<f64>,
    cumsum_weighted_x: Vec<Vec<f64>>,
    cumsum_weighted_x_sq: Vec<Vec<f64>>,
}

fn precompute_risk_set_cumsum(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    weights: &[f64],
    exp_eta: &[f64],
) -> RiskSetData {
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumsum_exp_eta = vec![0.0; n];
    let mut cumsum_weighted_x = vec![vec![0.0; p]; n];
    let mut cumsum_weighted_x_sq = vec![vec![0.0; p]; n];

    let mut running_exp = 0.0;
    let mut running_wx = vec![0.0; p];
    let mut running_wxsq = vec![0.0; p];

    for (pos, &idx) in sorted_indices.iter().enumerate() {
        let w = weights[idx] * exp_eta[idx];
        running_exp += w;

        for j in 0..p {
            let xij = x[idx * p + j];
            running_wx[j] += w * xij;
            running_wxsq[j] += w * xij * xij;
        }

        cumsum_exp_eta[pos] = running_exp;
        cumsum_weighted_x[pos] = running_wx.clone();
        cumsum_weighted_x_sq[pos] = running_wxsq.clone();
    }

    RiskSetData {
        sorted_indices,
        cumsum_exp_eta,
        cumsum_weighted_x,
        cumsum_weighted_x_sq,
    }
}

fn compute_gradient_hessian_diag_fast(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    weights: &[f64],
    beta: &[f64],
    offset: &[f64],
    active_set: Option<&[usize]>,
) -> (Vec<f64>, Vec<f64>) {
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

    let risk_data = precompute_risk_set_cumsum(x, n, p, time, weights, &exp_eta);

    let features_to_process: Vec<usize> = active_set
        .map(|a| a.to_vec())
        .unwrap_or_else(|| (0..p).collect());

    let mut gradient = vec![0.0; p];
    let mut hessian_diag = vec![0.0; p];

    let mut index_to_pos = vec![0usize; n];
    for (pos, &idx) in risk_data.sorted_indices.iter().enumerate() {
        index_to_pos[idx] = pos;
    }

    for i in 0..n {
        if status[i] != 1 {
            continue;
        }

        let pos = index_to_pos[i];
        let risk_sum = risk_data.cumsum_exp_eta[pos];
        if risk_sum <= 0.0 {
            continue;
        }

        for &j in &features_to_process {
            let xij = x[i * p + j];
            let x_bar = risk_data.cumsum_weighted_x[pos][j] / risk_sum;
            let x_sq_bar = risk_data.cumsum_weighted_x_sq[pos][j] / risk_sum;

            gradient[j] += weights[i] * (xij - x_bar);
            hessian_diag[j] += weights[i] * (x_sq_bar - x_bar * x_bar);
        }
    }

    (gradient, hessian_diag)
}

fn apply_strong_screening(
    gradient: &[f64],
    lambda: f64,
    lambda_prev: Option<f64>,
    beta: &[f64],
    p: usize,
) -> Vec<usize> {
    let threshold = match lambda_prev {
        Some(lp) => 2.0 * lambda - lp,
        None => lambda,
    };

    (0..p)
        .filter(|&j| beta[j].abs() > 1e-10 || gradient[j].abs() >= threshold)
        .collect()
}

fn apply_safe_screening(gradient: &[f64], lambda: f64, beta: &[f64], p: usize) -> Vec<usize> {
    (0..p)
        .filter(|&j| beta[j].abs() > 1e-10 || gradient[j].abs() >= lambda)
        .collect()
}

fn apply_edpp_screening(
    gradient: &[f64],
    lambda: f64,
    lambda_max: f64,
    beta: &[f64],
    p: usize,
) -> Vec<usize> {
    let threshold = lambda * (1.0 - (lambda / lambda_max).min(1.0));

    (0..p)
        .filter(|&j| beta[j].abs() > 1e-10 || gradient[j].abs() >= threshold)
        .collect()
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

fn cyclic_coordinate_descent_fast(
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
    screening: ScreeningRule,
    active_set_update_freq: usize,
    lambda_prev: Option<f64>,
    lambda_max: f64,
) -> (Vec<f64>, usize, bool, usize, usize) {
    let mut beta = beta_init
        .map(|b| b.to_vec())
        .unwrap_or_else(|| vec![0.0; p]);

    let l1_penalty = lambda * l1_ratio;
    let l2_penalty = lambda * (1.0 - l1_ratio);

    let (gradient, _) =
        compute_gradient_hessian_diag_fast(x, n, p, time, status, weights, &beta, offset, None);

    let mut active_set: Vec<usize> = match screening {
        ScreeningRule::None => (0..p).collect(),
        ScreeningRule::Safe => apply_safe_screening(&gradient, lambda, &beta, p),
        ScreeningRule::Strong => apply_strong_screening(&gradient, lambda, lambda_prev, &beta, p),
        ScreeningRule::EDPP => apply_edpp_screening(&gradient, lambda, lambda_max, &beta, p),
    };

    let screened_out = p - active_set.len();

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let beta_old = beta.clone();

        let (gradient, hessian_diag) = compute_gradient_hessian_diag_fast(
            x,
            n,
            p,
            time,
            status,
            weights,
            &beta,
            offset,
            Some(&active_set),
        );

        for &j in &active_set {
            let h_jj = hessian_diag[j] + l2_penalty;
            if h_jj.abs() < 1e-10 {
                continue;
            }

            let z = gradient[j] + hessian_diag[j] * beta[j];
            beta[j] = soft_threshold(z, l1_penalty) / h_jj;
        }

        let max_change: f64 = active_set
            .iter()
            .map(|&j| (beta[j] - beta_old[j]).abs())
            .fold(0.0, f64::max);

        if max_change < tol {
            let (full_gradient, _) = compute_gradient_hessian_diag_fast(
                x, n, p, time, status, weights, &beta, offset, None,
            );

            let kkt_violations: Vec<usize> = (0..p)
                .filter(|&j| {
                    if beta[j].abs() > 1e-10 {
                        false
                    } else {
                        full_gradient[j].abs() > l1_penalty * 1.01
                    }
                })
                .collect();

            if kkt_violations.is_empty() {
                converged = true;
                break;
            } else {
                active_set.extend(kkt_violations);
                active_set.sort();
                active_set.dedup();
            }
        }

        if iter % active_set_update_freq == 0 && iter > 0 {
            let (full_gradient, _) = compute_gradient_hessian_diag_fast(
                x, n, p, time, status, weights, &beta, offset, None,
            );

            let new_active: Vec<usize> = (0..p)
                .filter(|&j| beta[j].abs() > 1e-10 || full_gradient[j].abs() >= l1_penalty * 0.5)
                .collect();

            if !new_active.is_empty() {
                active_set = new_active;
            }
        }
    }

    let active_set_size = active_set.len();
    (beta, n_iter, converged, screened_out, active_set_size)
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config, weights=None, offset=None))]
pub fn fast_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: &FastCoxConfig,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
) -> PyResult<FastCoxResult> {
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

    let (x_std, means, sds) = if config.standardize {
        standardize_matrix(&x, n_obs, n_vars)
    } else {
        (x.clone(), vec![0.0; n_vars], vec![1.0; n_vars])
    };

    let beta_zero = vec![0.0; n_vars];
    let (gradient, _) = compute_gradient_hessian_diag_fast(
        &x_std, n_obs, n_vars, &time, &status, &wt, &beta_zero, &off, None,
    );
    let lambda_max = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / n_obs as f64;

    let (beta_std, n_iter, converged, screened_out, active_set_size) =
        cyclic_coordinate_descent_fast(
            &x_std,
            n_obs,
            n_vars,
            &time,
            &status,
            &wt,
            &off,
            config.lambda,
            config.l1_ratio,
            config.max_iter,
            config.tol,
            None,
            config.screening,
            config.active_set_update_freq,
            None,
            lambda_max,
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

    Ok(FastCoxResult {
        coefficients,
        nonzero_indices,
        lambda_used: config.lambda,
        l1_ratio: config.l1_ratio,
        n_iter,
        converged,
        deviance,
        df,
        scale_factors: if config.standardize { Some(sds) } else { None },
        center_values: if config.standardize {
            Some(means)
        } else {
            None
        },
        screened_out,
        active_set_size,
    })
}

#[pyfunction]
#[pyo3(signature = (
    x,
    n_obs,
    n_vars,
    time,
    status,
    l1_ratio=1.0,
    n_lambda=100,
    lambda_min_ratio=None,
    weights=None,
    max_iter=1000,
    tol=1e-7,
    screening=ScreeningRule::Strong
))]
pub fn fast_cox_path(
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
    screening: ScreeningRule,
) -> PyResult<FastCoxPath> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);
    let off = vec![0.0; n_obs];

    let (x_std, _means, sds) = standardize_matrix(&x, n_obs, n_vars);

    let beta_zero = vec![0.0; n_vars];
    let (gradient, _) = compute_gradient_hessian_diag_fast(
        &x_std, n_obs, n_vars, &time, &status, &wt, &beta_zero, &off, None,
    );

    let lambda_max =
        gradient.iter().map(|g| g.abs()).fold(0.0, f64::max) / (n_obs as f64 * l1_ratio.max(0.001));

    let min_ratio = lambda_min_ratio.unwrap_or(if n_obs < n_vars { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    let lambdas: Vec<f64> = (0..n_lambda)
        .map(|i| {
            let frac = i as f64 / (n_lambda - 1).max(1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect();

    let mut all_coefficients = Vec::with_capacity(n_lambda);
    let mut all_deviances = Vec::with_capacity(n_lambda);
    let mut all_df = Vec::with_capacity(n_lambda);
    let mut all_n_iters = Vec::with_capacity(n_lambda);
    let mut all_converged = Vec::with_capacity(n_lambda);

    let mut beta_warm = vec![0.0; n_vars];
    let mut lambda_prev: Option<f64> = None;

    for (idx, &lambda) in lambdas.iter().enumerate() {
        let (beta_std, n_iter, conv, _screened, _active) = cyclic_coordinate_descent_fast(
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
            screening,
            10,
            lambda_prev,
            lambda_max,
        );

        beta_warm = beta_std.clone();
        lambda_prev = Some(lambda);

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
        all_converged.push(conv);
    }

    Ok(FastCoxPath {
        lambdas,
        coefficients: all_coefficients,
        deviances: all_deviances,
        df: all_df,
        n_iters: all_n_iters,
        converged: all_converged,
    })
}

#[pyfunction]
#[pyo3(signature = (
    x,
    n_obs,
    n_vars,
    time,
    status,
    l1_ratio=1.0,
    n_lambda=100,
    n_folds=5,
    weights=None,
    screening=ScreeningRule::Strong,
    seed=None
))]
pub fn fast_cox_cv(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    l1_ratio: f64,
    n_lambda: usize,
    n_folds: usize,
    weights: Option<Vec<f64>>,
    screening: ScreeningRule,
    seed: Option<u64>,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let path = fast_cox_path(
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
        screening,
    )?;

    let wt = weights.unwrap_or_else(|| vec![1.0; n_obs]);

    let mut rng = fastrand::Rng::with_seed(seed.unwrap_or(42));
    let mut fold_assign: Vec<usize> = (0..n_obs).map(|i| i % n_folds).collect();
    for i in (1..n_obs).rev() {
        let j = rng.usize(0..=i);
        fold_assign.swap(i, j);
    }

    let x_ref = &x;
    let time_ref = &time;
    let status_ref = &status;
    let cv_deviances: Vec<Vec<f64>> = path
        .lambdas
        .par_iter()
        .map(|&lambda| {
            let mut fold_devs = Vec::with_capacity(n_folds);

            for fold in 0..n_folds {
                let train_idx: Vec<usize> =
                    (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();
                let test_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] == fold).collect();

                if train_idx.is_empty() || test_idx.is_empty() {
                    continue;
                }

                let train_x: Vec<f64> = train_idx
                    .iter()
                    .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                    .collect();
                let train_time: Vec<f64> = train_idx.iter().map(|&i| time_ref[i]).collect();
                let train_status: Vec<i32> = train_idx.iter().map(|&i| status_ref[i]).collect();
                let train_wt: Vec<f64> = train_idx.iter().map(|&i| wt[i]).collect();

                let config = FastCoxConfig::new(
                    lambda, l1_ratio, 1000, 1e-7, screening, None, 10, true, true,
                )
                .unwrap();

                if let Ok(result) = fast_cox(
                    train_x,
                    train_idx.len(),
                    n_vars,
                    train_time,
                    train_status,
                    &config,
                    Some(train_wt),
                    None,
                ) {
                    let test_x: Vec<f64> = test_idx
                        .iter()
                        .flat_map(|&i| (0..n_vars).map(move |j| x_ref[i * n_vars + j]))
                        .collect();
                    let test_time: Vec<f64> = test_idx.iter().map(|&i| time_ref[i]).collect();
                    let test_status: Vec<i32> = test_idx.iter().map(|&i| status_ref[i]).collect();
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
    fn test_config() {
        let config = FastCoxConfig::new(
            0.1,
            1.0,
            1000,
            1e-7,
            ScreeningRule::Strong,
            None,
            10,
            true,
            true,
        )
        .unwrap();
        assert_eq!(config.lambda, 0.1);
        assert_eq!(config.l1_ratio, 1.0);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            FastCoxConfig::new(
                -0.1,
                1.0,
                1000,
                1e-7,
                ScreeningRule::None,
                None,
                10,
                true,
                true
            )
            .is_err()
        );
        assert!(
            FastCoxConfig::new(
                0.1,
                1.5,
                1000,
                1e-7,
                ScreeningRule::None,
                None,
                10,
                true,
                true
            )
            .is_err()
        );
        assert!(
            FastCoxConfig::new(0.1, 1.0, 0, 1e-7, ScreeningRule::None, None, 10, true, true)
                .is_err()
        );
    }

    #[test]
    fn test_fast_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let config = FastCoxConfig::new(
            0.1,
            1.0,
            100,
            1e-5,
            ScreeningRule::None,
            None,
            10,
            true,
            true,
        )
        .unwrap();

        let result = fast_cox(x, 4, 2, time, status, &config, None, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
    }

    #[test]
    fn test_screening_rules() {
        let gradient = vec![0.5, 0.1, 0.8, 0.05];
        let lambda = 0.3;
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let safe = apply_safe_screening(&gradient, lambda, &beta, 4);
        let strong = apply_strong_screening(&gradient, lambda, None, &beta, 4);

        assert!(safe.contains(&0));
        assert!(safe.contains(&2));
        assert!(!safe.contains(&3));

        assert!(strong.contains(&0));
        assert!(strong.contains(&2));
    }

    #[test]
    fn test_fast_cox_path() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.1).collect();
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 0];

        let path = fast_cox_path(
            x,
            10,
            4,
            time,
            status,
            1.0,
            10,
            None,
            None,
            100,
            1e-5,
            ScreeningRule::Strong,
        )
        .unwrap();

        assert_eq!(path.lambdas.len(), 10);
        assert_eq!(path.coefficients.len(), 10);
        assert!(path.lambdas[0] >= path.lambdas[9]);
    }
}
