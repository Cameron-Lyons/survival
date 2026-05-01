use crate::internal::typed_inputs::{CovariateMatrix, CoxRegressionInput, SurvivalData, Weights};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ElasticNetPathConfig {
    #[pyo3(get, set)]
    pub l1_ratio: f64,
    #[pyo3(get, set)]
    pub n_lambda: usize,
    #[pyo3(get, set)]
    pub lambda_min_ratio: Option<f64>,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl ElasticNetPathConfig {
    #[new]
    #[pyo3(signature = (
        l1_ratio=0.5,
        n_lambda=100,
        lambda_min_ratio=None,
        max_iter=1000,
        tol=1e-7
    ))]
    pub fn new(
        l1_ratio: f64,
        n_lambda: usize,
        lambda_min_ratio: Option<f64>,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "l1_ratio must be between 0 and 1",
            ));
        }
        if n_lambda < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_lambda must be at least 2",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        if let Some(lambda_min_ratio) = lambda_min_ratio
            && !(0.0..1.0).contains(&lambda_min_ratio)
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lambda_min_ratio must be greater than 0 and less than 1",
            ));
        }

        Ok(Self {
            l1_ratio,
            n_lambda,
            lambda_min_ratio,
            max_iter,
            tol,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ElasticNetCVConfig {
    #[pyo3(get, set)]
    pub l1_ratio: f64,
    #[pyo3(get, set)]
    pub n_lambda: usize,
    #[pyo3(get, set)]
    pub n_folds: usize,
}

#[pymethods]
impl ElasticNetCVConfig {
    #[new]
    #[pyo3(signature = (l1_ratio=0.5, n_lambda=100, n_folds=10))]
    pub fn new(l1_ratio: f64, n_lambda: usize, n_folds: usize) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "l1_ratio must be between 0 and 1",
            ));
        }
        if n_lambda < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_lambda must be at least 2",
            ));
        }
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }

        Ok(Self {
            l1_ratio,
            n_lambda,
            n_folds,
        })
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
        sds[j] = var.sqrt().max(crate::constants::DIVISION_FLOOR);

        for i in 0..n {
            x_std[i * p + j] = (x[i * p + j] - means[j]) / sds[j];
        }
    }

    (x_std, means, sds)
}

struct ElasticNetData<'a> {
    x: &'a [f64],
    n: usize,
    p: usize,
    time: &'a [f64],
    status: &'a [i32],
    weights: &'a [f64],
    offset: &'a [f64],
}

struct ElasticNetDescentConfig<'a> {
    lambda: f64,
    l1_ratio: f64,
    max_iter: usize,
    tol: f64,
    beta_init: Option<&'a [f64]>,
}

fn compute_cox_gradient_hessian(data: &ElasticNetData, beta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut gradient = vec![0.0; data.p];
    let mut hessian_diag = vec![0.0; data.p];

    let eta: Vec<f64> = (0..data.n)
        .map(|i| {
            let mut e = data.offset[i];
            for j in 0..data.p {
                e += data.x[i * data.p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut indices: Vec<usize> = (0..data.n).collect();
    indices.sort_by(|&a, &b| {
        data.time[b]
            .partial_cmp(&data.time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; data.p];
    let mut weighted_x_sq = vec![0.0; data.p];

    for &i in &indices {
        let w = data.weights[i] * exp_eta[i];
        risk_sum += w;

        for j in 0..data.p {
            let xij = data.x[i * data.p + j];
            weighted_x[j] += w * xij;
            weighted_x_sq[j] += w * xij * xij;
        }

        if data.status[i] == 1 && risk_sum > 0.0 {
            for j in 0..data.p {
                let xij = data.x[i * data.p + j];
                let x_bar = weighted_x[j] / risk_sum;
                let x_sq_bar = weighted_x_sq[j] / risk_sum;

                gradient[j] += data.weights[i] * (xij - x_bar);
                hessian_diag[j] += data.weights[i] * (x_sq_bar - x_bar * x_bar);
            }
        }
    }

    (gradient, hessian_diag)
}

fn compute_cox_deviance(data: &ElasticNetData, beta: &[f64]) -> f64 {
    let eta: Vec<f64> = (0..data.n)
        .map(|i| {
            let mut e = data.offset[i];
            for j in 0..data.p {
                e += data.x[i * data.p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut indices: Vec<usize> = (0..data.n).collect();
    indices.sort_by(|&a, &b| {
        data.time[b]
            .partial_cmp(&data.time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut loglik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += data.weights[i] * exp_eta[i];

        if data.status[i] == 1 && risk_sum > 0.0 {
            loglik += data.weights[i] * (eta[i] - risk_sum.ln());
        }
    }

    -2.0 * loglik
}

fn coordinate_descent_cox(
    data: &ElasticNetData,
    config: ElasticNetDescentConfig,
) -> (Vec<f64>, usize, bool) {
    let mut beta = config
        .beta_init
        .map(|b| b.to_vec())
        .unwrap_or_else(|| vec![0.0; data.p]);

    let l1_penalty = config.lambda * config.l1_ratio;
    let l2_penalty = config.lambda * (1.0 - config.l1_ratio);

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;
        let beta_old = beta.clone();

        let (gradient, hessian_diag) = compute_cox_gradient_hessian(data, &beta);

        for j in 0..data.p {
            let h_jj = hessian_diag[j] + l2_penalty;
            if h_jj.abs() < crate::constants::DIVISION_FLOOR {
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

        if max_change < config.tol {
            converged = true;
            break;
        }
    }

    (beta, n_iter, converged)
}

#[pyfunction]
#[pyo3(signature = (input, config))]
pub fn elastic_net_cox(
    input: &CoxRegressionInput,
    config: &ElasticNetConfig,
) -> PyResult<ElasticNetCoxResult> {
    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit();
    let off = input.offset_or_zero();

    let (x_std, _means, sds) = if config.standardize {
        standardize_matrix(x, n_obs, n_vars)
    } else {
        (x.to_vec(), vec![0.0; n_vars], vec![1.0; n_vars])
    };

    let fit_data = ElasticNetData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: &wt,
        offset: &off,
    };
    let (beta_std, n_iter, converged) = coordinate_descent_cox(
        &fit_data,
        ElasticNetDescentConfig {
            lambda: config.alpha,
            l1_ratio: config.l1_ratio,
            max_iter: config.max_iter,
            tol: config.tol,
            beta_init: None,
        },
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
        .filter(|(_, c)| c.abs() > crate::constants::DIVISION_FLOOR)
        .map(|(i, _)| i)
        .collect();

    let df = nonzero_indices.len() as f64;
    let deviance_data = ElasticNetData {
        x,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: &wt,
        offset: &off,
    };
    let deviance = compute_cox_deviance(&deviance_data, &coefficients);

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
#[pyo3(signature = (input, config=None))]
pub fn elastic_net_cox_path(
    input: &CoxRegressionInput,
    config: Option<&ElasticNetPathConfig>,
) -> PyResult<ElasticNetCoxPath> {
    let default_config;
    let config = match config {
        Some(config) => config,
        None => {
            default_config = ElasticNetPathConfig {
                l1_ratio: 0.5,
                n_lambda: 100,
                lambda_min_ratio: None,
                max_iter: 1000,
                tol: 1e-7,
            };
            &default_config
        }
    };

    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit();
    let off = input.offset_or_zero();

    let (x_std, _means, sds) = standardize_matrix(x, n_obs, n_vars);

    let beta_zero = vec![0.0; n_vars];
    let fit_data = ElasticNetData {
        x: &x_std,
        n: n_obs,
        p: n_vars,
        time,
        status,
        weights: &wt,
        offset: &off,
    };
    let (gradient, _) = compute_cox_gradient_hessian(&fit_data, &beta_zero);

    let lambda_max = gradient.iter().map(|g| g.abs()).fold(0.0, f64::max)
        / (n_obs as f64 * config.l1_ratio.max(0.001));

    let min_ratio = config
        .lambda_min_ratio
        .unwrap_or(if n_obs < n_vars { 0.01 } else { 0.0001 });
    let lambda_min = lambda_max * min_ratio;

    let lambdas: Vec<f64> = (0..config.n_lambda)
        .map(|i| {
            let frac = i as f64 / (config.n_lambda - 1) as f64;
            lambda_max * (lambda_min / lambda_max).powf(frac)
        })
        .collect();

    let mut all_coefficients = Vec::with_capacity(config.n_lambda);
    let mut all_deviances = Vec::with_capacity(config.n_lambda);
    let mut all_df = Vec::with_capacity(config.n_lambda);
    let mut all_n_iters = Vec::with_capacity(config.n_lambda);

    let mut beta_warm = vec![0.0; n_vars];

    for &lambda in &lambdas {
        let (beta_std, n_iter, _converged) = coordinate_descent_cox(
            &fit_data,
            ElasticNetDescentConfig {
                lambda,
                l1_ratio: config.l1_ratio,
                max_iter: config.max_iter,
                tol: config.tol,
                beta_init: Some(&beta_warm),
            },
        );

        beta_warm = beta_std.clone();

        let coefficients: Vec<f64> = beta_std
            .iter()
            .zip(sds.iter())
            .map(|(&b, &s)| if s > 0.0 { b / s } else { b })
            .collect();

        let df = coefficients
            .iter()
            .filter(|&&c| c.abs() > crate::constants::DIVISION_FLOOR)
            .count() as f64;
        let deviance_data = ElasticNetData {
            x,
            n: n_obs,
            p: n_vars,
            time,
            status,
            weights: &wt,
            offset: &off,
        };
        let deviance = compute_cox_deviance(&deviance_data, &coefficients);

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
#[pyo3(signature = (input, config=None))]
pub fn elastic_net_cox_cv(
    input: &CoxRegressionInput,
    config: Option<&ElasticNetCVConfig>,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    let default_config;
    let config = match config {
        Some(config) => config,
        None => {
            default_config = ElasticNetCVConfig {
                l1_ratio: 0.5,
                n_lambda: 100,
                n_folds: 10,
            };
            &default_config
        }
    };

    let path_config = ElasticNetPathConfig {
        l1_ratio: config.l1_ratio,
        n_lambda: config.n_lambda,
        lambda_min_ratio: None,
        max_iter: 1000,
        tol: 1e-7,
    };
    let path = elastic_net_cox_path(input, Some(&path_config))?;

    let x = &input.covariates.values;
    let n_obs = input.covariates.n_obs;
    let n_vars = input.covariates.n_vars;
    let time = &input.survival.time;
    let status = &input.survival.status;
    let wt = input.weights_or_unit();
    let offset = input.offset_or_zero();

    let fold_assign: Vec<usize> = (0..n_obs).map(|i| i % config.n_folds).collect();
    let fold_indices: Vec<(Vec<usize>, Vec<usize>)> = (0..config.n_folds)
        .map(|fold| {
            let train_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n_obs).filter(|&i| fold_assign[i] == fold).collect();
            (train_idx, test_idx)
        })
        .collect();

    let x_ref = x;
    let time_ref = time;
    let status_ref = status;
    let wt_ref = &wt;
    let cv_deviances: Vec<Vec<f64>> = path
        .lambdas
        .par_iter()
        .map(|&lambda| {
            let fold_devs_by_fold: Vec<Option<f64>> = fold_indices
                .par_iter()
                .map(|(train_idx, test_idx)| {
                    if train_idx.is_empty() || test_idx.is_empty() {
                        return None;
                    }

                    let train_x: Vec<f64> = {
                        let mut result = Vec::with_capacity(train_idx.len() * n_vars);
                        for &i in train_idx {
                            for j in 0..n_vars {
                                result.push(x_ref[i * n_vars + j]);
                            }
                        }
                        result
                    };
                    let train_time: Vec<f64> = train_idx.iter().map(|&i| time_ref[i]).collect();
                    let train_status: Vec<i32> = train_idx.iter().map(|&i| status_ref[i]).collect();
                    let train_wt: Vec<f64> = train_idx.iter().map(|&i| wt_ref[i]).collect();
                    let train_offset: Vec<f64> = train_idx.iter().map(|&i| offset[i]).collect();

                    let Ok(config) =
                        ElasticNetConfig::new(lambda, config.l1_ratio, 1000, 1e-7, true, false)
                    else {
                        return None;
                    };
                    let Ok(train_input) = CoxRegressionInput::try_new(
                        match CovariateMatrix::try_new(train_x, train_idx.len(), n_vars) {
                            Ok(covariates) => covariates,
                            Err(_) => return None,
                        },
                        match SurvivalData::try_new(train_time, train_status) {
                            Ok(survival) => survival,
                            Err(_) => return None,
                        },
                        match Weights::try_new(train_wt) {
                            Ok(weights) => Some(weights),
                            Err(_) => return None,
                        },
                        Some(train_offset),
                    ) else {
                        return None;
                    };

                    if let Ok(result) = elastic_net_cox(&train_input, &config) {
                        let test_x: Vec<f64> = {
                            let mut result = Vec::with_capacity(test_idx.len() * n_vars);
                            for &i in test_idx {
                                for j in 0..n_vars {
                                    result.push(x_ref[i * n_vars + j]);
                                }
                            }
                            result
                        };
                        let test_time: Vec<f64> = test_idx.iter().map(|&i| time_ref[i]).collect();
                        let test_status: Vec<i32> =
                            test_idx.iter().map(|&i| status_ref[i]).collect();
                        let test_wt: Vec<f64> = test_idx.iter().map(|&i| wt_ref[i]).collect();
                        let test_off: Vec<f64> = test_idx.iter().map(|&i| offset[i]).collect();

                        let test_data = ElasticNetData {
                            x: &test_x,
                            n: test_idx.len(),
                            p: n_vars,
                            time: &test_time,
                            status: &test_status,
                            weights: &test_wt,
                            offset: &test_off,
                        };
                        let dev = compute_cox_deviance(&test_data, &result.coefficients);
                        Some(dev)
                    } else {
                        None
                    }
                })
                .collect();

            fold_devs_by_fold.into_iter().flatten().collect()
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
        use crate::internal::typed_inputs::{CovariateMatrix, CoxRegressionInput, SurvivalData};

        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let config = ElasticNetConfig::new(0.1, 0.5, 100, 1e-5, true, false).unwrap();

        let input = CoxRegressionInput::try_new(
            CovariateMatrix::try_new(x, 4, 2).unwrap(),
            SurvivalData::try_new(time, status).unwrap(),
            None,
            None,
        )
        .unwrap();

        let result = elastic_net_cox(&input, &config).unwrap();
        assert_eq!(result.coefficients.len(), 2);
    }
}
