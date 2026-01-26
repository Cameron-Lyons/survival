#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum CopulaType {
    Clayton,
    Frank,
    Gumbel,
    Gaussian,
    Independent,
}

#[pymethods]
impl CopulaType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "clayton" => Ok(CopulaType::Clayton),
            "frank" => Ok(CopulaType::Frank),
            "gumbel" => Ok(CopulaType::Gumbel),
            "gaussian" | "normal" => Ok(CopulaType::Gaussian),
            "independent" | "indep" => Ok(CopulaType::Independent),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown copula type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CopulaCensoringConfig {
    #[pyo3(get, set)]
    pub copula_type: CopulaType,
    #[pyo3(get, set)]
    pub theta: Option<f64>,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub n_grid: usize,
}

#[pymethods]
impl CopulaCensoringConfig {
    #[new]
    #[pyo3(signature = (copula_type=CopulaType::Clayton, theta=None, max_iter=100, tol=1e-6, n_grid=100))]
    pub fn new(
        copula_type: CopulaType,
        theta: Option<f64>,
        max_iter: usize,
        tol: f64,
        n_grid: usize,
    ) -> Self {
        CopulaCensoringConfig {
            copula_type,
            theta,
            max_iter,
            tol,
            n_grid,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CopulaCensoringResult {
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub theta_se: f64,
    #[pyo3(get)]
    pub kendall_tau: f64,
    #[pyo3(get)]
    pub marginal_survival_t: Vec<f64>,
    #[pyo3(get)]
    pub marginal_survival_c: Vec<f64>,
    #[pyo3(get)]
    pub joint_survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

fn copula_density(u: f64, v: f64, theta: f64, copula_type: &CopulaType) -> f64 {
    let u = u.clamp(1e-10, 1.0 - 1e-10);
    let v = v.clamp(1e-10, 1.0 - 1e-10);

    match copula_type {
        CopulaType::Clayton => {
            if theta <= 0.0 {
                return 1.0;
            }
            let a = u.powf(-theta) + v.powf(-theta) - 1.0;
            if a <= 0.0 {
                return 1e-10;
            }
            (1.0 + theta) * (u * v).powf(-theta - 1.0) * a.powf(-2.0 - 1.0 / theta)
        }
        CopulaType::Frank => {
            if theta.abs() < 1e-10 {
                return 1.0;
            }
            let a = (-theta).exp() - 1.0;
            let b = (-theta * u).exp() - 1.0;
            let c = (-theta * v).exp() - 1.0;
            let d = (-theta).exp() - 1.0;
            -theta * a * (-theta * (u + v)).exp() / (d + b * c / a).powi(2)
        }
        CopulaType::Gumbel => {
            if theta <= 1.0 {
                return 1.0;
            }
            let lu = (-u.ln()).powf(theta);
            let lv = (-v.ln()).powf(theta);
            let s = lu + lv;
            let c = (-s.powf(1.0 / theta)).exp();
            let term1 = s.powf(2.0 / theta - 2.0);
            let term2 = (lu * lv).powf(1.0 - 1.0 / theta);
            let term3 = (theta - 1.0 + s.powf(1.0 / theta)) / (u * v);
            c * term1 * term2 * term3
        }
        CopulaType::Gaussian => {
            let rho = theta.clamp(-0.999, 0.999);
            let x = probit(u);
            let y = probit(v);
            let det = 1.0 - rho * rho;
            let exp_term = -(x * x + y * y - 2.0 * rho * x * y) / (2.0 * det);
            (1.0 / (2.0 * std::f64::consts::PI * det.sqrt())) * exp_term.exp()
                / (normal_pdf(x) * normal_pdf(y)).max(1e-10)
        }
        CopulaType::Independent => 1.0,
    }
}

fn copula_cdf(u: f64, v: f64, theta: f64, copula_type: &CopulaType) -> f64 {
    let u = u.clamp(1e-10, 1.0 - 1e-10);
    let v = v.clamp(1e-10, 1.0 - 1e-10);

    match copula_type {
        CopulaType::Clayton => {
            if theta <= 0.0 {
                return u * v;
            }
            let a = u.powf(-theta) + v.powf(-theta) - 1.0;
            if a <= 0.0 {
                return 0.0;
            }
            a.powf(-1.0 / theta)
        }
        CopulaType::Frank => {
            if theta.abs() < 1e-10 {
                return u * v;
            }
            let a = ((-theta * u).exp() - 1.0) * ((-theta * v).exp() - 1.0);
            let b = (-theta).exp() - 1.0;
            -theta.recip() * (1.0 + a / b).ln()
        }
        CopulaType::Gumbel => {
            if theta <= 1.0 {
                return u * v;
            }
            let lu = (-u.ln()).powf(theta);
            let lv = (-v.ln()).powf(theta);
            (-(lu + lv).powf(1.0 / theta)).exp()
        }
        CopulaType::Gaussian => {
            let rho = theta.clamp(-0.999, 0.999);
            let x = probit(u);
            let y = probit(v);
            bivariate_normal_cdf(x, y, rho)
        }
        CopulaType::Independent => u * v,
    }
}

fn probit(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    #[allow(clippy::excessive_precision)]
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < 1e-10 {
        return normal_cdf(x) * normal_cdf(y);
    }

    let a = -x;
    let b = -y;
    let r = rho;

    let mut sum = 0.0;
    let weights = [
        0.24914704581340277,
        0.2491470458134028,
        0.2334925365383548,
        0.2334925365383548,
        0.2031674267230659,
        0.2031674267230659,
    ];
    let points = [
        -0.2011940939974345,
        0.2011940939974345,
        -0.3941513470775634,
        0.3941513470775634,
        -0.5709721726085388,
        0.5709721726085388,
    ];

    for i in 0..6 {
        let t = points[i];
        let z = (1.0 + t) / 2.0;
        let rt = (1.0 - r * r * z * z).sqrt();
        let term1 = normal_cdf((b - r * a * z) / rt);
        let term2 = normal_cdf((a - r * b * z) / rt);
        sum += weights[i] * (term1 * normal_pdf(a * z) + term2 * normal_pdf(b * z));
    }

    let result = normal_cdf(a) * normal_cdf(b) + r * sum / 4.0;
    result.clamp(0.0, 1.0)
}

fn kendall_tau_from_theta(theta: f64, copula_type: &CopulaType) -> f64 {
    match copula_type {
        CopulaType::Clayton => theta / (theta + 2.0),
        CopulaType::Frank => {
            if theta.abs() < 1e-10 {
                0.0
            } else {
                1.0 - 4.0 / theta * (1.0 - debye_1(theta))
            }
        }
        CopulaType::Gumbel => 1.0 - 1.0 / theta,
        CopulaType::Gaussian => 2.0 / std::f64::consts::PI * theta.asin(),
        CopulaType::Independent => 0.0,
    }
}

fn debye_1(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    let n = 100;
    let mut sum = 0.0;
    for i in 1..=n {
        let t = i as f64 / n as f64 * x;
        sum += t / (t.exp() - 1.0) * (x / n as f64);
    }
    sum / x
}

#[pyfunction]
#[pyo3(signature = (time, event, censoring_indicator, covariates, config))]
pub fn copula_censoring_model(
    time: Vec<f64>,
    event: Vec<i32>,
    censoring_indicator: Vec<i32>,
    covariates: Vec<f64>,
    config: &CopulaCensoringConfig,
) -> PyResult<CopulaCensoringResult> {
    let n = time.len();
    if event.len() != n || censoring_indicator.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eval_times: Vec<f64> = (0..config.n_grid)
        .map(|i| (i as f64 + 1.0) / config.n_grid as f64 * max_time)
        .collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

    let marginal_survival_t = estimate_km(&time, &event, &eval_times);
    let marginal_survival_c = estimate_km(&time, &censoring_indicator, &eval_times);

    let mut theta = config.theta.unwrap_or(1.0);
    let mut converged = false;
    let mut n_iter = 0;
    let mut prev_loglik = f64::NEG_INFINITY;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient = 0.0;
        let mut hessian = 0.0;

        for i in 0..n {
            let t_i = time[i];

            let idx_t = eval_times
                .iter()
                .position(|&t| t >= t_i)
                .unwrap_or(config.n_grid - 1);
            let idx_c = idx_t;

            let s_t = marginal_survival_t[idx_t].clamp(1e-10, 1.0 - 1e-10);
            let s_c = marginal_survival_c[idx_c].clamp(1e-10, 1.0 - 1e-10);

            let f_t = if idx_t > 0 {
                (marginal_survival_t[idx_t - 1] - s_t).max(1e-10)
            } else {
                (1.0 - s_t).max(1e-10)
            };

            let f_c = if idx_c > 0 {
                (marginal_survival_c[idx_c - 1] - s_c).max(1e-10)
            } else {
                (1.0 - s_c).max(1e-10)
            };

            let u = 1.0 - s_t;
            let v = 1.0 - s_c;

            if event[i] == 1 && censoring_indicator[i] == 0 {
                let c_density = copula_density(u, v, theta, &config.copula_type);
                loglik += (f_t * c_density).max(1e-300).ln();
            } else if event[i] == 0 && censoring_indicator[i] == 1 {
                let c_density = copula_density(u, v, theta, &config.copula_type);
                loglik += (f_c * c_density).max(1e-300).ln();
            } else {
                let c_cdf = copula_cdf(u, v, theta, &config.copula_type);
                let joint_surv = 1.0 - u - v + c_cdf;
                loglik += joint_surv.max(1e-300).ln();
            }

            let eps = 0.01;
            let c_plus = copula_density(u, v, theta + eps, &config.copula_type);
            let c_minus = copula_density(u, v, theta - eps, &config.copula_type);
            let c_curr = copula_density(u, v, theta, &config.copula_type);

            if c_curr > 1e-10 {
                gradient += (c_plus - c_minus) / (2.0 * eps * c_curr);
                hessian += ((c_plus - 2.0 * c_curr + c_minus) / (eps * eps)) / c_curr
                    - ((c_plus - c_minus) / (2.0 * eps * c_curr)).powi(2);
            }
        }

        if hessian.abs() > 1e-10 {
            let update = gradient / (-hessian).max(1e-10);
            theta += 0.5 * update;
            theta = match config.copula_type {
                CopulaType::Clayton => theta.clamp(0.01, 50.0),
                CopulaType::Frank => theta.clamp(-50.0, 50.0),
                CopulaType::Gumbel => theta.clamp(1.01, 50.0),
                CopulaType::Gaussian => theta.clamp(-0.99, 0.99),
                CopulaType::Independent => 0.0,
            };
        }

        if (loglik - prev_loglik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let kendall_tau = kendall_tau_from_theta(theta, &config.copula_type);

    let joint_survival: Vec<Vec<f64>> = eval_times
        .iter()
        .enumerate()
        .map(|(i, _)| {
            eval_times
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    let u = 1.0 - marginal_survival_t[i];
                    let v = 1.0 - marginal_survival_c[j];
                    let c_cdf = copula_cdf(u, v, theta, &config.copula_type);
                    (1.0 - u - v + c_cdf).clamp(0.0, 1.0)
                })
                .collect()
        })
        .collect();

    let theta_se = 0.1;
    let n_params = 1;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;

    Ok(CopulaCensoringResult {
        theta,
        theta_se,
        kendall_tau,
        marginal_survival_t,
        marginal_survival_c,
        joint_survival,
        eval_times,
        log_likelihood: prev_loglik,
        aic,
        n_iter,
        converged,
    })
}

fn estimate_km(time: &[f64], event: &[i32], eval_times: &[f64]) -> Vec<f64> {
    let n = time.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

    let mut survival = 1.0;
    let mut at_risk = n as f64;
    let mut km_times = Vec::new();
    let mut km_values = Vec::new();

    km_times.push(0.0);
    km_values.push(1.0);

    for &i in &sorted_indices {
        if event[i] == 1 {
            survival *= 1.0 - 1.0 / at_risk;
            km_times.push(time[i]);
            km_values.push(survival);
        }
        at_risk -= 1.0;
    }

    eval_times
        .iter()
        .map(|&t| {
            let idx = km_times.iter().rposition(|&kt| kt <= t).unwrap_or(0);
            km_values[idx]
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SensitivityBoundsConfig {
    #[pyo3(get, set)]
    pub gamma_range: Vec<f64>,
    #[pyo3(get, set)]
    pub n_grid: usize,
    #[pyo3(get, set)]
    pub method: String,
}

#[pymethods]
impl SensitivityBoundsConfig {
    #[new]
    #[pyo3(signature = (gamma_range=None, n_grid=100, method="rosenbaum"))]
    pub fn new(gamma_range: Option<Vec<f64>>, n_grid: usize, method: &str) -> Self {
        SensitivityBoundsConfig {
            gamma_range: gamma_range.unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
            n_grid,
            method: method.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SensitivityBoundsResult {
    #[pyo3(get)]
    pub gamma_values: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub rmst_lower: Vec<f64>,
    #[pyo3(get)]
    pub rmst_upper: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_lower: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_upper: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub point_estimate: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, covariates, tau, config))]
pub fn sensitivity_bounds_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<i32>,
    covariates: Vec<f64>,
    tau: f64,
    config: &SensitivityBoundsConfig,
) -> PyResult<SensitivityBoundsResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let max_time = tau.min(time.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let eval_times: Vec<f64> = (0..config.n_grid)
        .map(|i| (i as f64 + 1.0) / config.n_grid as f64 * max_time)
        .collect();

    let treated_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] == 1).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] == 0).collect();

    let treated_time: Vec<f64> = treated_idx.iter().map(|&i| time[i]).collect();
    let treated_event: Vec<i32> = treated_idx.iter().map(|&i| event[i]).collect();
    let control_time: Vec<f64> = control_idx.iter().map(|&i| time[i]).collect();
    let control_event: Vec<i32> = control_idx.iter().map(|&i| event[i]).collect();

    let km_treated = estimate_km(&treated_time, &treated_event, &eval_times);
    let km_control = estimate_km(&control_time, &control_event, &eval_times);

    let point_estimate =
        compute_rmst(&km_treated, &eval_times, tau) - compute_rmst(&km_control, &eval_times, tau);

    let mut survival_lower = Vec::new();
    let mut survival_upper = Vec::new();
    let mut rmst_lower = Vec::new();
    let mut rmst_upper = Vec::new();
    let mut hazard_ratio_lower = Vec::new();
    let mut hazard_ratio_upper = Vec::new();

    for &gamma in &config.gamma_range {
        let adjustment = (gamma - 1.0) / (gamma + 1.0);

        let surv_lower: Vec<f64> = km_treated
            .iter()
            .zip(km_control.iter())
            .map(|(&st, &sc)| {
                let diff = st - sc;
                (diff - adjustment * (1.0 - st.min(sc))).clamp(-1.0, 1.0)
            })
            .collect();

        let surv_upper: Vec<f64> = km_treated
            .iter()
            .zip(km_control.iter())
            .map(|(&st, &sc)| {
                let diff = st - sc;
                (diff + adjustment * st.min(sc)).clamp(-1.0, 1.0)
            })
            .collect();

        survival_lower.push(surv_lower.clone());
        survival_upper.push(surv_upper.clone());

        let rmst_l = point_estimate - adjustment * tau * 0.5;
        let rmst_u = point_estimate + adjustment * tau * 0.5;
        rmst_lower.push(rmst_l);
        rmst_upper.push(rmst_u);

        let hr_point = estimate_hazard_ratio(&time, &event, &treatment);
        let hr_l = hr_point * (1.0 / gamma);
        let hr_u = hr_point * gamma;
        hazard_ratio_lower.push(hr_l);
        hazard_ratio_upper.push(hr_u);
    }

    Ok(SensitivityBoundsResult {
        gamma_values: config.gamma_range.clone(),
        survival_lower,
        survival_upper,
        rmst_lower,
        rmst_upper,
        hazard_ratio_lower,
        hazard_ratio_upper,
        eval_times,
        point_estimate,
    })
}

fn compute_rmst(survival: &[f64], times: &[f64], tau: f64) -> f64 {
    let mut rmst = 0.0;
    let mut prev_time = 0.0;

    for (i, &t) in times.iter().enumerate() {
        if t > tau {
            break;
        }
        let dt = t - prev_time;
        let s = if i > 0 { survival[i - 1] } else { 1.0 };
        rmst += s * dt;
        prev_time = t;
    }

    rmst
}

fn estimate_hazard_ratio(time: &[f64], event: &[i32], treatment: &[i32]) -> f64 {
    let n = time.len();

    let mut beta = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

    for _ in 0..50 {
        let mut gradient = 0.0;
        let mut hessian = 0.0;
        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_tt = 0.0;

        for &i in &sorted_indices {
            let t_i = treatment[i] as f64;
            let exp_bt = (beta * t_i).clamp(-700.0, 700.0).exp();

            risk_sum += exp_bt;
            weighted_t += exp_bt * t_i;
            weighted_tt += exp_bt * t_i * t_i;

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                gradient += t_i - t_bar;
                hessian += tt_bar - t_bar * t_bar;
            }
        }

        if hessian.abs() > 1e-10 {
            beta += gradient / hessian;
            beta = beta.clamp(-10.0, 10.0);
        }
    }

    beta.exp()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MNARSurvivalConfig {
    #[pyo3(get, set)]
    pub delta_range: Vec<f64>,
    #[pyo3(get, set)]
    pub pattern: String,
}

#[pymethods]
impl MNARSurvivalConfig {
    #[new]
    #[pyo3(signature = (delta_range=None, pattern="tilt"))]
    pub fn new(delta_range: Option<Vec<f64>>, pattern: &str) -> Self {
        MNARSurvivalConfig {
            delta_range: delta_range.unwrap_or_else(|| vec![-1.0, -0.5, 0.0, 0.5, 1.0]),
            pattern: pattern.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct MNARSurvivalResult {
    #[pyo3(get)]
    pub delta_values: Vec<f64>,
    #[pyo3(get)]
    pub adjusted_survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub adjusted_rmst: Vec<f64>,
    #[pyo3(get)]
    pub adjusted_median: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub reference_survival: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config))]
pub fn mnar_sensitivity_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    config: &MNARSurvivalConfig,
) -> PyResult<MNARSurvivalResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_grid = 100;
    let eval_times: Vec<f64> = (0..n_grid)
        .map(|i| (i as f64 + 1.0) / n_grid as f64 * max_time)
        .collect();

    let reference_survival = estimate_km(&time, &event, &eval_times);

    let mut adjusted_survival = Vec::new();
    let mut adjusted_rmst = Vec::new();
    let mut adjusted_median = Vec::new();

    for &delta in &config.delta_range {
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if event[i] == 0 {
                    (delta * time[i] / max_time).exp()
                } else {
                    1.0
                }
            })
            .collect();

        let adj_surv = estimate_weighted_km(&time, &event, &weights, &eval_times);
        adjusted_survival.push(adj_surv.clone());

        let rmst = compute_rmst(&adj_surv, &eval_times, max_time);
        adjusted_rmst.push(rmst);

        let median = adj_surv
            .iter()
            .zip(eval_times.iter())
            .find(|(s, _)| **s <= 0.5)
            .map(|(_, t)| *t)
            .unwrap_or(max_time);
        adjusted_median.push(median);
    }

    Ok(MNARSurvivalResult {
        delta_values: config.delta_range.clone(),
        adjusted_survival,
        adjusted_rmst,
        adjusted_median,
        eval_times,
        reference_survival,
    })
}

fn estimate_weighted_km(
    time: &[f64],
    event: &[i32],
    weights: &[f64],
    eval_times: &[f64],
) -> Vec<f64> {
    let n = time.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

    let mut survival = 1.0;
    let total_weight: f64 = weights.iter().sum();
    let mut at_risk_weight = total_weight;
    let mut km_times = Vec::new();
    let mut km_values = Vec::new();

    km_times.push(0.0);
    km_values.push(1.0);

    for &i in &sorted_indices {
        if event[i] == 1 {
            survival *= 1.0 - weights[i] / at_risk_weight;
            km_times.push(time[i]);
            km_values.push(survival);
        }
        at_risk_weight -= weights[i];
    }

    eval_times
        .iter()
        .map(|&t| {
            let idx = km_times.iter().rposition(|&kt| kt <= t).unwrap_or(0);
            km_values[idx]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copula_density() {
        let d = copula_density(0.5, 0.5, 2.0, &CopulaType::Clayton);
        assert!(d > 0.0);

        let d_indep = copula_density(0.5, 0.5, 0.0, &CopulaType::Independent);
        assert!((d_indep - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kendall_tau() {
        let tau = kendall_tau_from_theta(2.0, &CopulaType::Clayton);
        assert!((tau - 0.5).abs() < 1e-6);

        let tau_gumbel = kendall_tau_from_theta(2.0, &CopulaType::Gumbel);
        assert!((tau_gumbel - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sensitivity_bounds() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let config = SensitivityBoundsConfig::new(Some(vec![1.0, 2.0]), 50, "rosenbaum");
        let result =
            sensitivity_bounds_survival(time, event, treatment, vec![], 10.0, &config).unwrap();

        assert_eq!(result.gamma_values.len(), 2);
        assert_eq!(result.rmst_lower.len(), 2);
    }

    #[test]
    fn test_mnar_sensitivity() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];

        let config = MNARSurvivalConfig::new(Some(vec![-0.5, 0.0, 0.5]), "tilt");
        let result = mnar_sensitivity_survival(time, event, vec![], &config).unwrap();

        assert_eq!(result.delta_values.len(), 3);
        assert_eq!(result.adjusted_survival.len(), 3);
    }
}
