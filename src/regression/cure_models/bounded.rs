
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    if s_t > crate::constants::DIVISION_FLOOR {
        f_t / s_t
    } else {
        0.0
    }
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
            if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
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
            if info > crate::constants::DIVISION_FLOOR {
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
