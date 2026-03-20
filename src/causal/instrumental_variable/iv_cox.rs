

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]

pub struct IVCoxConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub two_stage: bool,
    #[pyo3(get, set)]
    pub robust_variance: bool,
}

#[pymethods]
impl IVCoxConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, two_stage=true, robust_variance=true))]
    pub fn new(max_iter: usize, tol: f64, two_stage: bool, robust_variance: bool) -> Self {
        IVCoxConfig {
            max_iter,
            tol,
            two_stage,
            robust_variance,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IVCoxResult {
    #[pyo3(get)]
    pub treatment_coef: f64,
    #[pyo3(get)]
    pub treatment_se: f64,
    #[pyo3(get)]
    pub treatment_z: f64,
    #[pyo3(get)]
    pub treatment_pvalue: f64,
    #[pyo3(get)]
    pub covariate_coef: Vec<f64>,
    #[pyo3(get)]
    pub covariate_se: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub first_stage_f: f64,
    #[pyo3(get)]
    pub first_stage_r2: f64,
    #[pyo3(get)]
    pub weak_instrument_test: f64,
    #[pyo3(get)]
    pub sargan_test: f64,
    #[pyo3(get)]
    pub sargan_pvalue: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, instruments, covariates, config))]
pub fn iv_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<f64>,
    instruments: Vec<f64>,
    covariates: Vec<f64>,
    config: &IVCoxConfig,
) -> PyResult<IVCoxResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_instruments = if instruments.is_empty() {
        0
    } else {
        instruments.len() / n
    };

    let p_cov = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    if n_instruments == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one instrument is required",
        ));
    }

    let treatment_mean = treatment.iter().sum::<f64>() / n as f64;
    let treatment_var: f64 = treatment
        .iter()
        .map(|&t| (t - treatment_mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let mut first_stage_coef = vec![0.0; n_instruments + p_cov + 1];

    for _ in 0..50 {
        let mut xtx = vec![vec![0.0; n_instruments + p_cov + 1]; n_instruments + p_cov + 1];
        let mut xty = vec![0.0; n_instruments + p_cov + 1];

        for i in 0..n {
            let mut x_row = vec![1.0];
            for k in 0..n_instruments {
                x_row.push(instruments[i * n_instruments + k]);
            }
            for k in 0..p_cov {
                x_row.push(covariates[i * p_cov + k]);
            }

            for j1 in 0..x_row.len() {
                xty[j1] += x_row[j1] * treatment[i];
                for j2 in 0..x_row.len() {
                    xtx[j1][j2] += x_row[j1] * x_row[j2];
                }
            }
        }

        for j in 0..first_stage_coef.len() {
            if xtx[j][j].abs() > crate::constants::DIVISION_FLOOR {
                first_stage_coef[j] = xty[j] / xtx[j][j];
            }
        }
    }

    let fitted_treatment: Vec<f64> = (0..n)
        .map(|i| {
            let mut pred = first_stage_coef[0];
            for k in 0..n_instruments {
                pred += first_stage_coef[1 + k] * instruments[i * n_instruments + k];
            }
            for k in 0..p_cov {
                pred += first_stage_coef[1 + n_instruments + k] * covariates[i * p_cov + k];
            }
            pred
        })
        .collect();

    let fitted_mean = fitted_treatment.iter().sum::<f64>() / n as f64;
    let fitted_var: f64 = fitted_treatment
        .iter()
        .map(|&f| (f - fitted_mean).powi(2))
        .sum::<f64>()
        / n as f64;

    let residual_var: f64 = treatment
        .iter()
        .zip(fitted_treatment.iter())
        .map(|(&t, &f)| (t - f).powi(2))
        .sum::<f64>()
        / n as f64;

    let first_stage_r2 = if treatment_var > crate::constants::DIVISION_FLOOR {
        (1.0 - residual_var / treatment_var).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let first_stage_f = if residual_var > crate::constants::DIVISION_FLOOR {
        (fitted_var / residual_var) * (n - n_instruments - p_cov - 1) as f64 / n_instruments as f64
    } else {
        0.0
    };

    let treatment_for_cox = if config.two_stage {
        fitted_treatment.clone()
    } else {
        treatment.clone()
    };

    let mut beta_treatment = 0.0;
    let mut beta_cov = vec![0.0; p_cov];
    let mut converged = false;
    let mut n_iter = 0;
    let mut final_hessian_treatment: f64 = 0.0;
    let mut final_hessian_cov: Vec<f64> = vec![0.0; p_cov];

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let eta: Vec<f64> = (0..n)
            .map(|i| {
                let mut e = beta_treatment * treatment_for_cox[i];
                for k in 0..p_cov {
                    e += beta_cov[k] * covariates[i * p_cov + k];
                }
                e.clamp(-700.0, 700.0)
            })
            .collect();

        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient_treatment = 0.0;
        let mut gradient_cov = vec![0.0; p_cov];
        let mut hessian_treatment: f64 = 0.0;
        let mut hessian_cov = vec![0.0; p_cov];

        let mut risk_sum = 0.0;
        let mut weighted_treatment = 0.0;
        let mut weighted_treatment_sq = 0.0;
        let mut weighted_cov = vec![0.0; p_cov];
        let mut weighted_cov_sq = vec![0.0; p_cov];

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_treatment += exp_eta[i] * treatment_for_cox[i];
            weighted_treatment_sq += exp_eta[i] * treatment_for_cox[i] * treatment_for_cox[i];
            for k in 0..p_cov {
                weighted_cov[k] += exp_eta[i] * covariates[i * p_cov + k];
                weighted_cov_sq[k] +=
                    exp_eta[i] * covariates[i * p_cov + k] * covariates[i * p_cov + k];
            }

            if event[i] == 1 && risk_sum > 0.0 {
                let treatment_bar = weighted_treatment / risk_sum;
                let treatment_sq_bar = weighted_treatment_sq / risk_sum;

                gradient_treatment += treatment_for_cox[i] - treatment_bar;
                hessian_treatment += treatment_sq_bar - treatment_bar * treatment_bar;

                for k in 0..p_cov {
                    let cov_bar = weighted_cov[k] / risk_sum;
                    let cov_sq_bar = weighted_cov_sq[k] / risk_sum;
                    gradient_cov[k] += covariates[i * p_cov + k] - cov_bar;
                    hessian_cov[k] += cov_sq_bar - cov_bar * cov_bar;
                }
            }
        }

        let old_beta = beta_treatment;

        if hessian_treatment.abs() > crate::constants::DIVISION_FLOOR {
            beta_treatment += gradient_treatment / hessian_treatment;
            beta_treatment = beta_treatment.clamp(-10.0, 10.0);
        }

        for k in 0..p_cov {
            if hessian_cov[k].abs() > crate::constants::DIVISION_FLOOR {
                beta_cov[k] += gradient_cov[k] / hessian_cov[k];
                beta_cov[k] = beta_cov[k].clamp(-10.0, 10.0);
            }
        }

        final_hessian_treatment = hessian_treatment;
        final_hessian_cov = hessian_cov;

        if (beta_treatment - old_beta).abs() < config.tol {
            converged = true;
            break;
        }
    }

    let treatment_se = if final_hessian_treatment > crate::constants::DIVISION_FLOOR {
        (1.0 / final_hessian_treatment).sqrt()
    } else {
        f64::INFINITY
    };

    let treatment_z = if treatment_se > crate::constants::DIVISION_FLOOR && treatment_se.is_finite()
    {
        beta_treatment / treatment_se
    } else {
        0.0
    };

    let treatment_pvalue = 2.0 * (1.0 - normal_cdf(treatment_z.abs()));

    let covariate_se: Vec<f64> = final_hessian_cov
        .iter()
        .map(|&h| {
            if h > crate::constants::DIVISION_FLOOR {
                (1.0 / h).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    let mut log_likelihood = 0.0;
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = beta_treatment * treatment_for_cox[i];
            for k in 0..p_cov {
                e += beta_cov[k] * covariates[i * p_cov + k];
            }
            e
        })
        .collect();

    let mut risk_sum = 0.0;
    for &i in &sorted_indices {
        risk_sum += eta[i].exp();
        if event[i] == 1 && risk_sum > 0.0 {
            log_likelihood += eta[i] - risk_sum.ln();
        }
    }

    let weak_instrument_test = first_stage_f;

    let sargan_test = if n_instruments > 1 {
        let residuals: Vec<f64> = treatment
            .iter()
            .zip(fitted_treatment.iter())
            .map(|(&t, &f)| t - f)
            .collect();

        let mut r_sum_sq = 0.0;
        for i in 0..n {
            for k in 0..n_instruments {
                r_sum_sq += (residuals[i] * instruments[i * n_instruments + k]).powi(2);
            }
        }

        let sigma_sq = residual_var;
        if sigma_sq > crate::constants::DIVISION_FLOOR {
            r_sum_sq / sigma_sq
        } else {
            0.0
        }
    } else {
        0.0
    };

    let sargan_pvalue = if n_instruments > 1 {
        1.0 - chi2_cdf(sargan_test, (n_instruments - 1) as f64)
    } else {
        1.0
    };

    Ok(IVCoxResult {
        treatment_coef: beta_treatment,
        treatment_se,
        treatment_z,
        treatment_pvalue,
        covariate_coef: beta_cov,
        covariate_se,
        log_likelihood,
        first_stage_f,
        first_stage_r2,
        weak_instrument_test,
        sargan_test,
        sargan_pvalue,
        n_iter,
        converged,
    })
}
