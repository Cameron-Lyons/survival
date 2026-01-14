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
}
