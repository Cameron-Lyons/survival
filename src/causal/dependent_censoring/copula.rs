

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]

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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    let u = u.clamp(
        crate::constants::DIVISION_FLOOR,
        1.0 - crate::constants::DIVISION_FLOOR,
    );
    let v = v.clamp(
        crate::constants::DIVISION_FLOOR,
        1.0 - crate::constants::DIVISION_FLOOR,
    );

    match copula_type {
        CopulaType::Clayton => {
            if theta <= 0.0 {
                return 1.0;
            }
            let a = u.powf(-theta) + v.powf(-theta) - 1.0;
            if a <= 0.0 {
                return crate::constants::DIVISION_FLOOR;
            }
            (1.0 + theta) * (u * v).powf(-theta - 1.0) * a.powf(-2.0 - 1.0 / theta)
        }
        CopulaType::Frank => {
            if theta.abs() < crate::constants::DIVISION_FLOOR {
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
                / (normal_pdf(x) * normal_pdf(y)).max(crate::constants::DIVISION_FLOOR)
        }
        CopulaType::Independent => 1.0,
    }
}

fn copula_cdf(u: f64, v: f64, theta: f64, copula_type: &CopulaType) -> f64 {
    let u = u.clamp(
        crate::constants::DIVISION_FLOOR,
        1.0 - crate::constants::DIVISION_FLOOR,
    );
    let v = v.clamp(
        crate::constants::DIVISION_FLOOR,
        1.0 - crate::constants::DIVISION_FLOOR,
    );

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
            if theta.abs() < crate::constants::DIVISION_FLOOR {
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
    crate::internal::statistical::normal_inverse_cdf(p.clamp(
        crate::constants::DIVISION_FLOOR,
        1.0 - crate::constants::DIVISION_FLOOR,
    ))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < crate::constants::DIVISION_FLOOR {
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
            if theta.abs() < crate::constants::DIVISION_FLOOR {
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
    if x.abs() < crate::constants::DIVISION_FLOOR {
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
    if !covariates.is_empty() && (n == 0 || !covariates.len().is_multiple_of(n)) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must be empty or have length n_obs * n_covariates",
        ));
    }

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eval_times: Vec<f64> = (0..config.n_grid)
        .map(|i| (i as f64 + 1.0) / config.n_grid as f64 * max_time)
        .collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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

            let s_t = marginal_survival_t[idx_t].clamp(
                crate::constants::DIVISION_FLOOR,
                1.0 - crate::constants::DIVISION_FLOOR,
            );
            let s_c = marginal_survival_c[idx_c].clamp(
                crate::constants::DIVISION_FLOOR,
                1.0 - crate::constants::DIVISION_FLOOR,
            );

            let f_t = if idx_t > 0 {
                (marginal_survival_t[idx_t - 1] - s_t).max(crate::constants::DIVISION_FLOOR)
            } else {
                (1.0 - s_t).max(crate::constants::DIVISION_FLOOR)
            };

            let f_c = if idx_c > 0 {
                (marginal_survival_c[idx_c - 1] - s_c).max(crate::constants::DIVISION_FLOOR)
            } else {
                (1.0 - s_c).max(crate::constants::DIVISION_FLOOR)
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

            if c_curr > crate::constants::DIVISION_FLOOR {
                gradient += (c_plus - c_minus) / (2.0 * eps * c_curr);
                hessian += ((c_plus - 2.0 * c_curr + c_minus) / (eps * eps)) / c_curr
                    - ((c_plus - c_minus) / (2.0 * eps * c_curr)).powi(2);
            }
        }

        if hessian.abs() > crate::constants::DIVISION_FLOOR {
            let update = gradient / (-hessian).max(crate::constants::DIVISION_FLOOR);
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
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
