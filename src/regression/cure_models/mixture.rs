
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

    let mut beta_cure = vec![0.0; p_cure];
    let beta_surv = vec![0.0; p_surv];
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
                w[i] = if denom > crate::constants::DIVISION_FLOOR {
                    1.0
                } else {
                    0.5
                };
            } else {
                let numer = pi[i] * s_t;
                let denom = (1.0 - pi[i]) + pi[i] * s_t;
                w[i] = if denom > crate::constants::DIVISION_FLOOR {
                    numer / denom
                } else {
                    0.5
                };
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
                if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
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
#[pyclass(from_py_object)]
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
    let beta = vec![0.0; p];
    let mut scale = time.iter().sum::<f64>() / n as f64;
    let shape = 1.0;

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

            let (s_0, _f_0) = compute_surv_density(time[i], scale, shape, &distribution);

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

        if theta_denom > crate::constants::DIVISION_FLOOR {
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
