
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
            (1.0 + theta * f_t).powf(-1.0 / theta.max(crate::constants::DIVISION_FLOOR))
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
        NonMixtureType::Destructive => (theta * (s_0 - 1.0)).exp(),
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
            let s_pop = base.powf(-1.0 / theta.max(crate::constants::DIVISION_FLOOR));
            let h_pop = f_0 / (base * s_0.max(crate::constants::DIVISION_FLOOR));
            h_pop * s_pop
        }
        NonMixtureType::NegativeBinomial => {
            let r = 1.0 / dispersion;
            let base = 1.0 + dispersion * theta * f_t;
            let s_pop = base.powf(-r);
            let h_pop =
                (r * dispersion * theta * f_0) / (base * s_0.max(crate::constants::DIVISION_FLOOR));
            h_pop * s_pop
        }
        NonMixtureType::Poisson => {
            let s_pop = (-theta * f_t).exp();
            let h_pop = theta * f_0 / s_0.max(crate::constants::DIVISION_FLOOR);
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
    let dispersion = config.dispersion;

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
            if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
                beta[j] += 0.3 * gradient[j] / (hessian_diag[j] + 1e-6);
                beta[j] = beta[j].clamp(-10.0, 10.0);
            }
        }

        if theta_hess.abs() > crate::constants::DIVISION_FLOOR {
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
        NonMixtureType::GeometricGeneralized => {
            (1.0 + theta).powf(-1.0 / theta.max(crate::constants::DIVISION_FLOOR))
        }
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
            if info > crate::constants::DIVISION_FLOOR {
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
