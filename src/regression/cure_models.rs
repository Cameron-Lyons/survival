#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use crate::utilities::statistical::{erf, normal_cdf, probit};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum CureDistribution {
    Weibull,
    LogNormal,
    LogLogistic,
    Exponential,
    Gamma,
}

#[pymethods]
impl CureDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "weibull" => Ok(CureDistribution::Weibull),
            "lognormal" | "log_normal" => Ok(CureDistribution::LogNormal),
            "loglogistic" | "log_logistic" => Ok(CureDistribution::LogLogistic),
            "exponential" | "exp" => Ok(CureDistribution::Exponential),
            "gamma" => Ok(CureDistribution::Gamma),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown distribution",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum LinkFunction {
    Logit,
    Probit,
    CLogLog,
    Identity,
}

#[pymethods]
impl LinkFunction {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "logit" => Ok(LinkFunction::Logit),
            "probit" => Ok(LinkFunction::Probit),
            "cloglog" | "c_log_log" => Ok(LinkFunction::CLogLog),
            "identity" => Ok(LinkFunction::Identity),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown link function",
            )),
        }
    }

    fn link(&self, p: f64) -> f64 {
        let p_clamped = p.clamp(1e-10, 1.0 - 1e-10);
        match self {
            LinkFunction::Logit => (p_clamped / (1.0 - p_clamped)).ln(),
            LinkFunction::Probit => probit(p_clamped),
            LinkFunction::CLogLog => (-(-p_clamped).ln_1p()).ln(),
            LinkFunction::Identity => p_clamped,
        }
    }

    fn inv_link(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Logit => 1.0 / (1.0 + (-eta).exp()),
            LinkFunction::Probit => normal_cdf(eta),
            LinkFunction::CLogLog => 1.0 - (-eta.exp()).exp(),
            LinkFunction::Identity => eta.clamp(0.0, 1.0),
        }
    }

    fn deriv(&self, eta: f64) -> f64 {
        match self {
            LinkFunction::Logit => {
                let p = 1.0 / (1.0 + (-eta).exp());
                p * (1.0 - p)
            }
            LinkFunction::Probit => normal_pdf(eta),
            LinkFunction::CLogLog => {
                let exp_eta = eta.exp();
                exp_eta * (-exp_eta).exp()
            }
            LinkFunction::Identity => 1.0,
        }
    }
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn weibull_surv(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-(t / scale).powf(shape)).exp()
}

fn weibull_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = t / scale;
    (shape / scale) * z.powf(shape - 1.0) * (-z.powf(shape)).exp()
}

fn lognormal_surv(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 - normal_cdf((t.ln() - mu) / sigma)
}

fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    normal_pdf(z) / (t * sigma)
}

fn loglogistic_surv(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + (t / scale).powf(shape))
}

fn loglogistic_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    (shape / scale) * (t / scale).powf(shape - 1.0) / (1.0 + z).powi(2)
}

fn exponential_surv(t: f64, rate: f64) -> f64 {
    if t <= 0.0 {
        return 1.0;
    }
    (-rate * t).exp()
}

fn exponential_pdf(t: f64, rate: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    rate * (-rate * t).exp()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MixtureCureConfig {
    #[pyo3(get, set)]
    pub distribution: CureDistribution,
    #[pyo3(get, set)]
    pub link: LinkFunction,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub em_max_iter: usize,
}

#[pymethods]
impl MixtureCureConfig {
    #[new]
    #[pyo3(signature = (distribution=CureDistribution::Weibull, link=LinkFunction::Logit, max_iter=100, tol=1e-6, em_max_iter=500))]
    pub fn new(
        distribution: CureDistribution,
        link: LinkFunction,
        max_iter: usize,
        tol: f64,
        em_max_iter: usize,
    ) -> Self {
        MixtureCureConfig {
            distribution,
            link,
            max_iter,
            tol,
            em_max_iter,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MixtureCureResult {
    #[pyo3(get)]
    pub cure_coef: Vec<f64>,
    #[pyo3(get)]
    pub survival_coef: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub cure_fraction: f64,
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
    pub cure_prob: Vec<f64>,
}

fn compute_surv_density(t: f64, scale: f64, shape: f64, dist: &CureDistribution) -> (f64, f64) {
    match dist {
        CureDistribution::Weibull => (weibull_surv(t, scale, shape), weibull_pdf(t, scale, shape)),
        CureDistribution::LogNormal => (
            lognormal_surv(t, scale, shape),
            lognormal_pdf(t, scale, shape),
        ),
        CureDistribution::LogLogistic => (
            loglogistic_surv(t, scale, shape),
            loglogistic_pdf(t, scale, shape),
        ),
        CureDistribution::Exponential => (exponential_surv(t, scale), exponential_pdf(t, scale)),
        CureDistribution::Gamma => (weibull_surv(t, scale, shape), weibull_pdf(t, scale, shape)),
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, x_cure, x_surv, config))]
pub fn mixture_cure_model(
    time: Vec<f64>,
    status: Vec<i32>,
    x_cure: Vec<f64>,
    x_surv: Vec<f64>,
    config: &MixtureCureConfig,
) -> PyResult<MixtureCureResult> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let p_cure = if x_cure.is_empty() {
        1
    } else {
        x_cure.len() / n
    };
    let p_surv = if x_surv.is_empty() {
        1
    } else {
        x_surv.len() / n
    };

    let x_cure_mat = if x_cure.is_empty() {
        vec![1.0; n]
    } else {
        x_cure.clone()
    };

    let x_surv_mat = if x_surv.is_empty() {
        vec![1.0; n]
    } else {
        x_surv.clone()
    };

    let mut beta_cure = vec![0.0; p_cure];
    let mut beta_surv = vec![0.0; p_surv];
    let mut scale = time.iter().copied().sum::<f64>() / n as f64;
    let mut shape = 1.0;

    let mut w = vec![0.5; n];

    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..config.em_max_iter {
        n_iter = iter + 1;

        let pi: Vec<f64> = (0..n)
            .map(|i| {
                let mut eta = 0.0;
                for j in 0..p_cure {
                    eta += x_cure_mat[i * p_cure + j] * beta_cure[j];
                }
                config.link.inv_link(eta)
            })
            .collect();

        for i in 0..n {
            let (s_t, f_t) = compute_surv_density(time[i], scale, shape, &config.distribution);
            if status[i] == 1 {
                let denom = pi[i] * f_t;
                w[i] = if denom > 1e-10 { 1.0 } else { 0.5 };
            } else {
                let numer = pi[i] * s_t;
                let denom = (1.0 - pi[i]) + pi[i] * s_t;
                w[i] = if denom > 1e-10 { numer / denom } else { 0.5 };
            }
        }

        for _ in 0..config.max_iter {
            let mut gradient = vec![0.0; p_cure];
            let mut hessian_diag = vec![0.0; p_cure];

            for i in 0..n {
                let mut eta = 0.0;
                for j in 0..p_cure {
                    eta += x_cure_mat[i * p_cure + j] * beta_cure[j];
                }
                let pi_i = config.link.inv_link(eta);
                let deriv = config.link.deriv(eta);

                for j in 0..p_cure {
                    let x_ij = x_cure_mat[i * p_cure + j];
                    gradient[j] += (w[i] - pi_i) * deriv * x_ij;
                    hessian_diag[j] += deriv * deriv * x_ij * x_ij;
                }
            }

            for j in 0..p_cure {
                if hessian_diag[j].abs() > 1e-10 {
                    beta_cure[j] += gradient[j] / (hessian_diag[j] + 1e-6);
                }
            }
        }

        let susceptible_times: Vec<f64> = (0..n)
            .filter(|&i| w[i] > 0.5 || status[i] == 1)
            .map(|i| time[i])
            .collect();

        if !susceptible_times.is_empty() {
            let mean_time = susceptible_times.iter().sum::<f64>() / susceptible_times.len() as f64;
            scale = mean_time.max(0.01);

            let log_times: Vec<f64> = susceptible_times
                .iter()
                .filter(|&&t| t > 0.0)
                .map(|t| t.ln())
                .collect();
            if log_times.len() > 1 {
                let mean_log = log_times.iter().sum::<f64>() / log_times.len() as f64;
                let var_log = log_times
                    .iter()
                    .map(|&l| (l - mean_log).powi(2))
                    .sum::<f64>()
                    / log_times.len() as f64;
                shape =
                    (std::f64::consts::PI / (6.0_f64.sqrt() * var_log.sqrt().max(0.1))).max(0.1);
            }
        }

        let mut loglik = 0.0;
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p_cure {
                eta += x_cure_mat[i * p_cure + j] * beta_cure[j];
            }
            let pi_i = config.link.inv_link(eta);
            let (s_t, f_t) = compute_surv_density(time[i], scale, shape, &config.distribution);

            if status[i] == 1 {
                let contrib = pi_i * f_t;
                loglik += contrib.max(1e-300).ln();
            } else {
                let contrib = (1.0 - pi_i) + pi_i * s_t;
                loglik += contrib.max(1e-300).ln();
            }
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let cure_fraction = (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p_cure {
                eta += x_cure_mat[i * p_cure + j] * beta_cure[j];
            }
            1.0 - config.link.inv_link(eta)
        })
        .sum::<f64>()
        / n as f64;

    let cure_prob: Vec<f64> = (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p_cure {
                eta += x_cure_mat[i * p_cure + j] * beta_cure[j];
            }
            1.0 - config.link.inv_link(eta)
        })
        .collect();

    let n_params = p_cure + p_surv + 2;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(MixtureCureResult {
        cure_coef: beta_cure,
        survival_coef: beta_surv,
        scale,
        shape,
        cure_fraction,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        cure_prob,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PromotionTimeCureResult {
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub cure_fraction: f64,
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
}

#[pyfunction]
#[pyo3(signature = (time, status, x, distribution=CureDistribution::Weibull, max_iter=500, tol=1e-6))]
pub fn promotion_time_cure_model(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    distribution: CureDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<PromotionTimeCureResult> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let p = if x.is_empty() { 1 } else { x.len() / n };
    let x_mat = if x.is_empty() {
        vec![1.0; n]
    } else {
        x.clone()
    };

    let mut theta = 1.0;
    let mut beta = vec![0.0; p];
    let mut scale = time.iter().sum::<f64>() / n as f64;
    let mut shape = 1.0;

    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut theta_numer = 0.0;
        let mut theta_denom = 0.0;

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i * p + j] * beta[j];
            }
            let exp_eta = eta.exp();

            let (s_0, f_0) = compute_surv_density(time[i], scale, shape, &distribution);
            let f_t = -theta * exp_eta * s_0.ln();

            if status[i] == 1 {
                let hazard = theta * exp_eta * (-s_0.ln().max(1e-300));
                let survival = (theta * exp_eta * (s_0.ln())).exp();
                let contrib = hazard * survival;
                loglik += contrib.max(1e-300).ln();

                theta_numer += 1.0;
                theta_denom += exp_eta * (-s_0.ln().max(1e-300));
            } else {
                let survival = (theta * exp_eta * s_0.ln()).exp();
                loglik += survival.max(1e-300).ln();
                theta_denom += exp_eta * (-s_0.ln().max(1e-300));
            }
        }

        if theta_denom > 1e-10 {
            theta = (theta_numer / theta_denom).max(0.01);
        }

        let susceptible_times: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 1)
            .map(|i| time[i])
            .collect();

        if !susceptible_times.is_empty() {
            scale = susceptible_times.iter().sum::<f64>() / susceptible_times.len() as f64;
            scale = scale.max(0.01);
        }

        if (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let cure_fraction = (-theta).exp();

    let n_params = p + 3;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(PromotionTimeCureResult {
        theta,
        coef: beta,
        scale,
        shape,
        cure_fraction,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
    })
}

#[pyfunction]
pub fn predict_cure_probability(
    result: &MixtureCureResult,
    x_new: Vec<f64>,
    n_new: usize,
    link: &LinkFunction,
) -> PyResult<Vec<f64>> {
    let p = result.cure_coef.len();
    if x_new.len() != n_new * p {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_new dimensions don't match model",
        ));
    }

    let probs: Vec<f64> = (0..n_new)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_new[i * p + j] * result.cure_coef[j];
            }
            1.0 - link.inv_link(eta)
        })
        .collect();

    Ok(probs)
}

#[pyfunction]
pub fn predict_survival_cure(
    result: &MixtureCureResult,
    time_points: Vec<f64>,
    x_cure: Vec<f64>,
    x_surv: Vec<f64>,
    n_subjects: usize,
    distribution: &CureDistribution,
    link: &LinkFunction,
) -> PyResult<Vec<Vec<f64>>> {
    let p_cure = result.cure_coef.len();

    let survival: Vec<Vec<f64>> = (0..n_subjects)
        .into_par_iter()
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p_cure {
                eta += x_cure[i * p_cure + j] * result.cure_coef[j];
            }
            let pi = link.inv_link(eta);

            time_points
                .iter()
                .map(|&t| {
                    let (s_t, _) =
                        compute_surv_density(t, result.scale, result.shape, distribution);
                    (1.0 - pi) + pi * s_t
                })
                .collect()
        })
        .collect();

    Ok(survival)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BoundedCumulativeHazardConfig {
    #[pyo3(get, set)]
    pub distribution: CureDistribution,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
}

#[pymethods]
impl BoundedCumulativeHazardConfig {
    #[new]
    #[pyo3(signature = (distribution=CureDistribution::Weibull, max_iter=500, tol=1e-6, alpha=1.0))]
    pub fn new(distribution: CureDistribution, max_iter: usize, tol: f64, alpha: f64) -> Self {
        BoundedCumulativeHazardConfig {
            distribution,
            max_iter,
            tol,
            alpha,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BoundedCumulativeHazardResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub cure_fraction: f64,
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
    pub cumulative_hazard_bound: f64,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
}

fn baseline_cumulative_hazard(t: f64, scale: f64, shape: f64, dist: &CureDistribution) -> f64 {
    let s_t = match dist {
        CureDistribution::Weibull => weibull_surv(t, scale, shape),
        CureDistribution::LogNormal => lognormal_surv(t, scale, shape),
        CureDistribution::LogLogistic => loglogistic_surv(t, scale, shape),
        CureDistribution::Exponential => exponential_surv(t, scale),
        CureDistribution::Gamma => weibull_surv(t, scale, shape),
    };
    -s_t.max(1e-300).ln()
}

fn baseline_hazard(t: f64, scale: f64, shape: f64, dist: &CureDistribution) -> f64 {
    let (s_t, f_t) = compute_surv_density(t, scale, shape, dist);
    if s_t > 1e-10 { f_t / s_t } else { 0.0 }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, config))]
pub fn bounded_cumulative_hazard_model(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<f64>,
    config: &BoundedCumulativeHazardConfig,
) -> PyResult<BoundedCumulativeHazardResult> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let mut beta = vec![0.0; p];
    let mut scale = time.iter().sum::<f64>() / n as f64;
    let mut shape = 1.0;
    let mut alpha = config.alpha;

    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i * p + j] * beta[j];
            }
            let exp_eta = eta.exp();

            let h_0_t = baseline_cumulative_hazard(time[i], scale, shape, &config.distribution);
            let lambda_0_t = baseline_hazard(time[i], scale, shape, &config.distribution);

            let s_pop = (-alpha * exp_eta * h_0_t).exp();
            let hazard_pop = alpha * exp_eta * lambda_0_t;

            if status[i] == 1 {
                loglik += hazard_pop.max(1e-300).ln() + s_pop.max(1e-300).ln();

                for j in 0..p {
                    let x_ij = x_mat[i * p + j];
                    gradient[j] += x_ij * (1.0 - alpha * exp_eta * h_0_t);
                    hessian_diag[j] += x_ij * x_ij * alpha * exp_eta * h_0_t;
                }
            } else {
                loglik += s_pop.max(1e-300).ln();

                for j in 0..p {
                    let x_ij = x_mat[i * p + j];
                    gradient[j] += -x_ij * alpha * exp_eta * h_0_t;
                    hessian_diag[j] += x_ij * x_ij * alpha * exp_eta * h_0_t;
                }
            }
        }

        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                beta[j] += 0.5 * gradient[j] / (hessian_diag[j] + 1e-6);
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }

        let event_times: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 1)
            .map(|i| time[i])
            .collect();

        if !event_times.is_empty() {
            let mean_t = event_times.iter().sum::<f64>() / event_times.len() as f64;
            scale = 0.9 * scale + 0.1 * mean_t.max(0.01);

            let log_times: Vec<f64> = event_times
                .iter()
                .filter(|&&t| t > 0.0)
                .map(|t| t.ln())
                .collect();
            if log_times.len() > 1 {
                let mean_log = log_times.iter().sum::<f64>() / log_times.len() as f64;
                let var_log: f64 = log_times
                    .iter()
                    .map(|&l| (l - mean_log).powi(2))
                    .sum::<f64>()
                    / log_times.len() as f64;
                let new_shape =
                    (std::f64::consts::PI / (6.0_f64.sqrt() * var_log.sqrt().max(0.1))).max(0.1);
                shape = 0.9 * shape + 0.1 * new_shape;
            }
        }

        let max_h0: f64 = time
            .iter()
            .map(|&t| baseline_cumulative_hazard(t, scale, shape, &config.distribution))
            .fold(f64::NEG_INFINITY, f64::max);

        if max_h0 > 0.0 {
            let d = status.iter().filter(|&&s| s == 1).count() as f64;
            let sum_exp_eta: f64 = (0..n)
                .map(|i| {
                    let mut eta = 0.0;
                    for j in 0..p {
                        eta += x_mat[i * p + j] * beta[j];
                    }
                    eta.exp()
                })
                .sum();
            let new_alpha = d / (sum_exp_eta * max_h0);
            alpha = 0.9 * alpha + 0.1 * new_alpha.clamp(0.01, 10.0);
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let cure_fraction = (-alpha).exp();
    let cumulative_hazard_bound = alpha;

    let std_errors: Vec<f64> = (0..p)
        .map(|j| {
            let mut info = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for k in 0..p {
                    eta += x_mat[i * p + k] * beta[k];
                }
                let exp_eta = eta.exp();
                let h_0_t = baseline_cumulative_hazard(time[i], scale, shape, &config.distribution);
                let x_ij = x_mat[i * p + j];
                info += x_ij * x_ij * alpha * exp_eta * h_0_t;
            }
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let n_params = p + 3;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(BoundedCumulativeHazardResult {
        coef: beta,
        scale,
        shape,
        alpha,
        cure_fraction,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        cumulative_hazard_bound,
        std_errors,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum NonMixtureType {
    GeometricGeneralized,
    NegativeBinomial,
    Poisson,
    Destructive,
}

#[pymethods]
impl NonMixtureType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "geometric" | "geometric_generalized" => Ok(NonMixtureType::GeometricGeneralized),
            "negative_binomial" | "nb" => Ok(NonMixtureType::NegativeBinomial),
            "poisson" => Ok(NonMixtureType::Poisson),
            "destructive" => Ok(NonMixtureType::Destructive),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown non-mixture type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NonMixtureCureConfig {
    #[pyo3(get, set)]
    pub model_type: NonMixtureType,
    #[pyo3(get, set)]
    pub distribution: CureDistribution,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub dispersion: f64,
}

#[pymethods]
impl NonMixtureCureConfig {
    #[new]
    #[pyo3(signature = (model_type=NonMixtureType::GeometricGeneralized, distribution=CureDistribution::Weibull, max_iter=500, tol=1e-6, dispersion=1.0))]
    pub fn new(
        model_type: NonMixtureType,
        distribution: CureDistribution,
        max_iter: usize,
        tol: f64,
        dispersion: f64,
    ) -> Self {
        NonMixtureCureConfig {
            model_type,
            distribution,
            max_iter,
            tol,
            dispersion,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct NonMixtureCureResult {
    #[pyo3(get)]
    pub coef: Vec<f64>,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub dispersion: f64,
    #[pyo3(get)]
    pub cure_fraction: f64,
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
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub survival_probs: Vec<f64>,
}

fn non_mixture_survival(
    t: f64,
    theta: f64,
    scale: f64,
    shape: f64,
    dist: &CureDistribution,
    model_type: &NonMixtureType,
    dispersion: f64,
) -> f64 {
    let s_0 = match dist {
        CureDistribution::Weibull => weibull_surv(t, scale, shape),
        CureDistribution::LogNormal => lognormal_surv(t, scale, shape),
        CureDistribution::LogLogistic => loglogistic_surv(t, scale, shape),
        CureDistribution::Exponential => exponential_surv(t, scale),
        CureDistribution::Gamma => weibull_surv(t, scale, shape),
    };

    match model_type {
        NonMixtureType::GeometricGeneralized => {
            let f_t = 1.0 - s_0;
            (1.0 + theta * f_t).powf(-1.0 / theta.max(1e-10))
        }
        NonMixtureType::NegativeBinomial => {
            let f_t = 1.0 - s_0;
            let r = 1.0 / dispersion;
            (1.0 + dispersion * theta * f_t).powf(-r)
        }
        NonMixtureType::Poisson => {
            let f_t = 1.0 - s_0;
            (-theta * f_t).exp()
        }
        NonMixtureType::Destructive => {
            let h_0 = -s_0.max(1e-300).ln();
            (theta * (s_0 - 1.0)).exp()
        }
    }
}

fn non_mixture_pdf(
    t: f64,
    theta: f64,
    scale: f64,
    shape: f64,
    dist: &CureDistribution,
    model_type: &NonMixtureType,
    dispersion: f64,
) -> f64 {
    let (s_0, f_0) = compute_surv_density(t, scale, shape, dist);
    let f_t = 1.0 - s_0;

    match model_type {
        NonMixtureType::GeometricGeneralized => {
            let base = 1.0 + theta * f_t;
            let s_pop = base.powf(-1.0 / theta.max(1e-10));
            let h_pop = f_0 / (base * s_0.max(1e-10));
            h_pop * s_pop
        }
        NonMixtureType::NegativeBinomial => {
            let r = 1.0 / dispersion;
            let base = 1.0 + dispersion * theta * f_t;
            let s_pop = base.powf(-r);
            let h_pop = (r * dispersion * theta * f_0) / (base * s_0.max(1e-10));
            h_pop * s_pop
        }
        NonMixtureType::Poisson => {
            let s_pop = (-theta * f_t).exp();
            let h_pop = theta * f_0 / s_0.max(1e-10);
            h_pop * s_pop
        }
        NonMixtureType::Destructive => {
            let s_pop = (theta * (s_0 - 1.0)).exp();
            let h_pop = theta * f_0;
            h_pop * s_pop
        }
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, config))]
pub fn non_mixture_cure_model(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<f64>,
    config: &NonMixtureCureConfig,
) -> PyResult<NonMixtureCureResult> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let p = if covariates.is_empty() {
        1
    } else {
        covariates.len() / n
    };
    let x_mat = if covariates.is_empty() {
        vec![1.0; n]
    } else {
        covariates.clone()
    };

    let mut beta = vec![0.0; p];
    let mut theta = 1.0;
    let mut scale = time.iter().sum::<f64>() / n as f64;
    let mut shape = 1.0;
    let mut dispersion = config.dispersion;

    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];
        let mut theta_grad = 0.0;
        let mut theta_hess = 0.0;

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i * p + j] * beta[j];
            }
            let exp_eta = eta.exp();
            let theta_i = theta * exp_eta;

            let s_pop = non_mixture_survival(
                time[i],
                theta_i,
                scale,
                shape,
                &config.distribution,
                &config.model_type,
                dispersion,
            );
            let f_pop = non_mixture_pdf(
                time[i],
                theta_i,
                scale,
                shape,
                &config.distribution,
                &config.model_type,
                dispersion,
            );

            if status[i] == 1 {
                loglik += f_pop.max(1e-300).ln();

                let eps = 1e-6;
                let s_pop_p = non_mixture_survival(
                    time[i],
                    theta_i * (1.0 + eps),
                    scale,
                    shape,
                    &config.distribution,
                    &config.model_type,
                    dispersion,
                );
                let f_pop_p = non_mixture_pdf(
                    time[i],
                    theta_i * (1.0 + eps),
                    scale,
                    shape,
                    &config.distribution,
                    &config.model_type,
                    dispersion,
                );
                let d_log_f = (f_pop_p.max(1e-300).ln() - f_pop.max(1e-300).ln()) / (theta_i * eps);

                for j in 0..p {
                    let x_ij = x_mat[i * p + j];
                    gradient[j] += x_ij * theta_i * d_log_f;
                    hessian_diag[j] += x_ij * x_ij * theta_i.powi(2) * d_log_f.abs();
                }
                theta_grad += exp_eta * d_log_f;
                theta_hess += exp_eta.powi(2) * d_log_f.abs();
            } else {
                loglik += s_pop.max(1e-300).ln();

                let eps = 1e-6;
                let s_pop_p = non_mixture_survival(
                    time[i],
                    theta_i * (1.0 + eps),
                    scale,
                    shape,
                    &config.distribution,
                    &config.model_type,
                    dispersion,
                );
                let d_log_s = (s_pop_p.max(1e-300).ln() - s_pop.max(1e-300).ln()) / (theta_i * eps);

                for j in 0..p {
                    let x_ij = x_mat[i * p + j];
                    gradient[j] += x_ij * theta_i * d_log_s;
                    hessian_diag[j] += x_ij * x_ij * theta_i.powi(2) * d_log_s.abs();
                }
                theta_grad += exp_eta * d_log_s;
                theta_hess += exp_eta.powi(2) * d_log_s.abs();
            }
        }

        for j in 0..p {
            if hessian_diag[j].abs() > 1e-10 {
                beta[j] += 0.3 * gradient[j] / (hessian_diag[j] + 1e-6);
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }

        if theta_hess.abs() > 1e-10 {
            theta += 0.3 * theta_grad / (theta_hess + 1e-6);
            theta = theta.clamp(0.01, 100.0);
        }

        let event_times: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 1)
            .map(|i| time[i])
            .collect();

        if !event_times.is_empty() {
            let mean_t = event_times.iter().sum::<f64>() / event_times.len() as f64;
            scale = 0.95 * scale + 0.05 * mean_t.max(0.01);

            let log_times: Vec<f64> = event_times
                .iter()
                .filter(|&&t| t > 0.0)
                .map(|t| t.ln())
                .collect();
            if log_times.len() > 1 {
                let mean_log = log_times.iter().sum::<f64>() / log_times.len() as f64;
                let var_log: f64 = log_times
                    .iter()
                    .map(|&l| (l - mean_log).powi(2))
                    .sum::<f64>()
                    / log_times.len() as f64;
                let new_shape =
                    (std::f64::consts::PI / (6.0_f64.sqrt() * var_log.sqrt().max(0.1))).max(0.1);
                shape = 0.95 * shape + 0.05 * new_shape;
            }
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let cure_fraction = match config.model_type {
        NonMixtureType::GeometricGeneralized => (1.0 + theta).powf(-1.0 / theta.max(1e-10)),
        NonMixtureType::NegativeBinomial => {
            let r = 1.0 / dispersion;
            (1.0 + dispersion * theta).powf(-r)
        }
        NonMixtureType::Poisson => (-theta).exp(),
        NonMixtureType::Destructive => (-theta).exp(),
    };

    let survival_probs: Vec<f64> = (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i * p + j] * beta[j];
            }
            let theta_i = theta * eta.exp();
            non_mixture_survival(
                time[i],
                theta_i,
                scale,
                shape,
                &config.distribution,
                &config.model_type,
                dispersion,
            )
        })
        .collect();

    let std_errors: Vec<f64> = (0..p)
        .map(|j| {
            let mut info = 0.0;
            for i in 0..n {
                let mut eta = 0.0;
                for k in 0..p {
                    eta += x_mat[i * p + k] * beta[k];
                }
                let x_ij = x_mat[i * p + j];
                info += x_ij * x_ij * eta.exp().powi(2);
            }
            if info > 1e-10 {
                (1.0 / info).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let n_params = p + 4;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n as f64).ln();

    Ok(NonMixtureCureResult {
        coef: beta,
        theta,
        scale,
        shape,
        dispersion,
        cure_fraction,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        std_errors,
        survival_probs,
    })
}

#[pyfunction]
pub fn predict_bounded_cumulative_hazard(
    result: &BoundedCumulativeHazardResult,
    time_points: Vec<f64>,
    covariates: Vec<f64>,
    n_subjects: usize,
    distribution: &CureDistribution,
) -> PyResult<Vec<Vec<f64>>> {
    let p = result.coef.len();

    let survival: Vec<Vec<f64>> = (0..n_subjects)
        .into_par_iter()
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += covariates[i * p + j] * result.coef[j];
            }
            let exp_eta = eta.exp();

            time_points
                .iter()
                .map(|&t| {
                    let h_0_t =
                        baseline_cumulative_hazard(t, result.scale, result.shape, distribution);
                    (-result.alpha * exp_eta * h_0_t).exp()
                })
                .collect()
        })
        .collect();

    Ok(survival)
}

#[pyfunction]
pub fn predict_non_mixture_survival(
    result: &NonMixtureCureResult,
    time_points: Vec<f64>,
    covariates: Vec<f64>,
    n_subjects: usize,
    model_type: &NonMixtureType,
    distribution: &CureDistribution,
) -> PyResult<Vec<Vec<f64>>> {
    let p = result.coef.len();

    let survival: Vec<Vec<f64>> = (0..n_subjects)
        .into_par_iter()
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += covariates[i * p + j] * result.coef[j];
            }
            let theta_i = result.theta * eta.exp();

            time_points
                .iter()
                .map(|&t| {
                    non_mixture_survival(
                        t,
                        theta_i,
                        result.scale,
                        result.shape,
                        distribution,
                        model_type,
                        result.dispersion,
                    )
                })
                .collect()
        })
        .collect();

    Ok(survival)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CureModelComparisonResult {
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub log_likelihoods: Vec<f64>,
    #[pyo3(get)]
    pub aic_values: Vec<f64>,
    #[pyo3(get)]
    pub bic_values: Vec<f64>,
    #[pyo3(get)]
    pub cure_fractions: Vec<f64>,
    #[pyo3(get)]
    pub best_model_aic: String,
    #[pyo3(get)]
    pub best_model_bic: String,
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, distributions=None))]
pub fn compare_cure_models(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<f64>,
    distributions: Option<Vec<String>>,
) -> PyResult<CureModelComparisonResult> {
    let dists = distributions.unwrap_or_else(|| {
        vec![
            "weibull".to_string(),
            "lognormal".to_string(),
            "loglogistic".to_string(),
        ]
    });

    let mut model_names = Vec::new();
    let mut log_likelihoods = Vec::new();
    let mut aic_values = Vec::new();
    let mut bic_values = Vec::new();
    let mut cure_fractions = Vec::new();

    for dist_name in &dists {
        let dist = match dist_name.to_lowercase().as_str() {
            "weibull" => CureDistribution::Weibull,
            "lognormal" | "log_normal" => CureDistribution::LogNormal,
            "loglogistic" | "log_logistic" => CureDistribution::LogLogistic,
            "exponential" | "exp" => CureDistribution::Exponential,
            _ => CureDistribution::Weibull,
        };

        let mixture_config = MixtureCureConfig::new(dist, LinkFunction::Logit, 50, 1e-5, 200);
        if let Ok(result) = mixture_cure_model(
            time.clone(),
            status.clone(),
            covariates.clone(),
            vec![],
            &mixture_config,
        ) {
            model_names.push(format!("Mixture-{}", dist_name));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }

        let bch_config = BoundedCumulativeHazardConfig::new(dist, 200, 1e-5, 1.0);
        if let Ok(result) = bounded_cumulative_hazard_model(
            time.clone(),
            status.clone(),
            covariates.clone(),
            &bch_config,
        ) {
            model_names.push(format!("BCH-{}", dist_name));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }

        let nm_config =
            NonMixtureCureConfig::new(NonMixtureType::GeometricGeneralized, dist, 200, 1e-5, 1.0);
        if let Ok(result) =
            non_mixture_cure_model(time.clone(), status.clone(), covariates.clone(), &nm_config)
        {
            model_names.push(format!("NonMixture-{}", dist_name));
            log_likelihoods.push(result.log_likelihood);
            aic_values.push(result.aic);
            bic_values.push(result.bic);
            cure_fractions.push(result.cure_fraction);
        }
    }

    let best_model_aic = if !aic_values.is_empty() {
        let min_idx = aic_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        model_names[min_idx].clone()
    } else {
        "None".to_string()
    };

    let best_model_bic = if !bic_values.is_empty() {
        let min_idx = bic_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        model_names[min_idx].clone()
    } else {
        "None".to_string()
    };

    Ok(CureModelComparisonResult {
        model_names,
        log_likelihoods,
        aic_values,
        bic_values,
        cure_fractions,
        best_model_aic,
        best_model_bic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_functions() {
        let logit = LinkFunction::Logit;
        assert!((logit.inv_link(0.0) - 0.5).abs() < 1e-6);
        assert!((logit.inv_link(logit.link(0.7)) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_weibull_surv() {
        assert!((weibull_surv(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!(weibull_surv(10.0, 1.0, 1.0) < 0.001);
    }

    #[test]
    fn test_mixture_cure_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = MixtureCureConfig::new(
            CureDistribution::Weibull,
            LinkFunction::Logit,
            50,
            1e-4,
            100,
        );

        let result = mixture_cure_model(time, status, vec![], vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
    }

    #[test]
    fn test_bounded_cumulative_hazard() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = BoundedCumulativeHazardConfig::new(CureDistribution::Weibull, 100, 1e-4, 1.0);

        let result = bounded_cumulative_hazard_model(time, status, vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
        assert!(result.alpha > 0.0);
    }

    #[test]
    fn test_non_mixture_cure() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = NonMixtureCureConfig::new(
            NonMixtureType::GeometricGeneralized,
            CureDistribution::Weibull,
            100,
            1e-4,
            1.0,
        );

        let result = non_mixture_cure_model(time, status, vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
        assert!(result.theta > 0.0);
    }

    #[test]
    fn test_non_mixture_types() {
        let time = vec![1.0, 2.0, 3.0, 5.0, 8.0, 12.0];
        let status = vec![1, 1, 0, 0, 0, 0];

        for model_type in [
            NonMixtureType::GeometricGeneralized,
            NonMixtureType::NegativeBinomial,
            NonMixtureType::Poisson,
            NonMixtureType::Destructive,
        ] {
            let config =
                NonMixtureCureConfig::new(model_type, CureDistribution::Weibull, 100, 1e-4, 1.0);
            let result = non_mixture_cure_model(time.clone(), status.clone(), vec![], &config);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_compare_cure_models() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];

        let result = compare_cure_models(time, status, vec![], None).unwrap();
        assert!(!result.model_names.is_empty());
        assert_eq!(result.model_names.len(), result.aic_values.len());
    }
}
