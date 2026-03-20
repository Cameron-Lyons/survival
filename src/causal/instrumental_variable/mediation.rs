
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MediationSurvivalConfig {
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl MediationSurvivalConfig {
    #[new]
    #[pyo3(signature = (max_iter=100, tol=1e-6, n_bootstrap=500, seed=None))]
    pub fn new(max_iter: usize, tol: f64, n_bootstrap: usize, seed: Option<u64>) -> Self {
        MediationSurvivalConfig {
            max_iter,
            tol,
            n_bootstrap,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MediationSurvivalResult {
    #[pyo3(get)]
    pub total_effect: f64,
    #[pyo3(get)]
    pub direct_effect: f64,
    #[pyo3(get)]
    pub indirect_effect: f64,
    #[pyo3(get)]
    pub proportion_mediated: f64,
    #[pyo3(get)]
    pub total_se: f64,
    #[pyo3(get)]
    pub direct_se: f64,
    #[pyo3(get)]
    pub indirect_se: f64,
    #[pyo3(get)]
    pub total_pvalue: f64,
    #[pyo3(get)]
    pub direct_pvalue: f64,
    #[pyo3(get)]
    pub indirect_pvalue: f64,
    #[pyo3(get)]
    pub treatment_to_mediator: f64,
    #[pyo3(get)]
    pub mediator_to_outcome: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, mediator, covariates, config))]
pub fn mediation_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<f64>,
    mediator: Vec<f64>,
    covariates: Vec<f64>,
    config: &MediationSurvivalConfig,
) -> PyResult<MediationSurvivalResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n || mediator.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let p_cov = if covariates.is_empty() {
        0
    } else {
        covariates.len() / n
    };

    let (alpha, _alpha_se) = fit_mediator_model(&treatment, &mediator, &covariates, n, p_cov);

    let (beta_total, _beta_total_se) = fit_outcome_model(
        &time,
        &event,
        &treatment,
        None,
        &covariates,
        n,
        p_cov,
        config.max_iter,
    );

    let (beta_direct, gamma, _) = fit_outcome_model_with_mediator(
        &time,
        &event,
        &treatment,
        &mediator,
        &covariates,
        n,
        p_cov,
        config.max_iter,
    );

    let total_effect = beta_total;
    let direct_effect = beta_direct;
    let indirect_effect = alpha * gamma;

    let proportion_mediated = if total_effect.abs() > crate::constants::DIVISION_FLOOR {
        (indirect_effect / total_effect).clamp(-1.0, 2.0)
    } else {
        0.0
    };

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let mut total_effects = Vec::new();
    let mut direct_effects = Vec::new();
    let mut indirect_effects = Vec::new();

    for _ in 0..config.n_bootstrap {
        let boot_indices: Vec<usize> = (0..n).map(|_| rng.usize(0..n)).collect();

        let boot_time: Vec<f64> = boot_indices.iter().map(|&i| time[i]).collect();
        let boot_event: Vec<i32> = boot_indices.iter().map(|&i| event[i]).collect();
        let boot_treatment: Vec<f64> = boot_indices.iter().map(|&i| treatment[i]).collect();
        let boot_mediator: Vec<f64> = boot_indices.iter().map(|&i| mediator[i]).collect();
        let cov_ref = &covariates;
        let boot_cov: Vec<f64> = if p_cov > 0 {
            boot_indices
                .iter()
                .flat_map(|&i| (0..p_cov).map(move |j| cov_ref[i * p_cov + j]))
                .collect()
        } else {
            vec![]
        };

        let (boot_alpha, _) =
            fit_mediator_model(&boot_treatment, &boot_mediator, &boot_cov, n, p_cov);
        let (boot_total, _) = fit_outcome_model(
            &boot_time,
            &boot_event,
            &boot_treatment,
            None,
            &boot_cov,
            n,
            p_cov,
            50,
        );
        let (boot_direct, boot_gamma, _) = fit_outcome_model_with_mediator(
            &boot_time,
            &boot_event,
            &boot_treatment,
            &boot_mediator,
            &boot_cov,
            n,
            p_cov,
            50,
        );

        total_effects.push(boot_total);
        direct_effects.push(boot_direct);
        indirect_effects.push(boot_alpha * boot_gamma);
    }

    let total_se = std_dev(&total_effects);
    let direct_se = std_dev(&direct_effects);
    let indirect_se = std_dev(&indirect_effects);

    let total_z = if total_se > crate::constants::DIVISION_FLOOR {
        total_effect / total_se
    } else {
        0.0
    };
    let direct_z = if direct_se > crate::constants::DIVISION_FLOOR {
        direct_effect / direct_se
    } else {
        0.0
    };
    let indirect_z = if indirect_se > crate::constants::DIVISION_FLOOR {
        indirect_effect / indirect_se
    } else {
        0.0
    };

    let total_pvalue = 2.0 * (1.0 - normal_cdf(total_z.abs()));
    let direct_pvalue = 2.0 * (1.0 - normal_cdf(direct_z.abs()));
    let indirect_pvalue = 2.0 * (1.0 - normal_cdf(indirect_z.abs()));

    Ok(MediationSurvivalResult {
        total_effect,
        direct_effect,
        indirect_effect,
        proportion_mediated,
        total_se,
        direct_se,
        indirect_se,
        total_pvalue,
        direct_pvalue,
        indirect_pvalue,
        treatment_to_mediator: alpha,
        mediator_to_outcome: gamma,
    })
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

fn fit_mediator_model(
    treatment: &[f64],
    mediator: &[f64],
    _covariates: &[f64],
    n: usize,
    _p_cov: usize,
) -> (f64, f64) {
    let mut alpha = 0.0;

    let treatment_mean = treatment.iter().sum::<f64>() / n as f64;
    let mediator_mean = mediator.iter().sum::<f64>() / n as f64;

    let mut cov_tm = 0.0;
    let mut var_t = 0.0;

    for i in 0..n {
        let t_centered = treatment[i] - treatment_mean;
        let m_centered = mediator[i] - mediator_mean;
        cov_tm += t_centered * m_centered;
        var_t += t_centered * t_centered;
    }

    if var_t > crate::constants::DIVISION_FLOOR {
        alpha = cov_tm / var_t;
    }

    let residuals: Vec<f64> = (0..n)
        .map(|i| mediator[i] - mediator_mean - alpha * (treatment[i] - treatment_mean))
        .collect();

    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / n as f64;
    let alpha_se = if var_t > crate::constants::DIVISION_FLOOR {
        (mse / var_t).sqrt()
    } else {
        f64::INFINITY
    };

    (alpha, alpha_se)
}

#[allow(clippy::too_many_arguments)]
fn fit_outcome_model(
    time: &[f64],
    event: &[i32],
    treatment: &[f64],
    _mediator: Option<&[f64]>,
    _covariates: &[f64],
    n: usize,
    _p_cov: usize,
    max_iter: usize,
) -> (f64, f64) {
    let mut beta = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n)
            .map(|i| (beta * treatment[i]).clamp(-700.0, 700.0))
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut gradient = 0.0;
        let mut hessian = 0.0;
        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_tt = 0.0;

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_t += exp_eta[i] * treatment[i];
            weighted_tt += exp_eta[i] * treatment[i] * treatment[i];

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                gradient += treatment[i] - t_bar;
                hessian += tt_bar - t_bar * t_bar;
            }
        }

        if hessian.abs() > crate::constants::DIVISION_FLOOR {
            beta += gradient / hessian;
            beta = beta.clamp(-10.0, 10.0);
        }
    }

    let se = 0.1;
    (beta, se)
}

#[allow(clippy::too_many_arguments)]
fn fit_outcome_model_with_mediator(
    time: &[f64],
    event: &[i32],
    treatment: &[f64],
    mediator: &[f64],
    _covariates: &[f64],
    n: usize,
    _p_cov: usize,
    max_iter: usize,
) -> (f64, f64, f64) {
    let mut beta = 0.0;
    let mut gamma = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..max_iter {
        let eta: Vec<f64> = (0..n)
            .map(|i| (beta * treatment[i] + gamma * mediator[i]).clamp(-700.0, 700.0))
            .collect();
        let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

        let mut grad_beta = 0.0;
        let mut grad_gamma = 0.0;
        let mut hess_beta = 0.0;
        let mut hess_gamma = 0.0;

        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_m = 0.0;
        let mut weighted_tt = 0.0;
        let mut weighted_mm = 0.0;

        for &i in &sorted_indices {
            risk_sum += exp_eta[i];
            weighted_t += exp_eta[i] * treatment[i];
            weighted_m += exp_eta[i] * mediator[i];
            weighted_tt += exp_eta[i] * treatment[i] * treatment[i];
            weighted_mm += exp_eta[i] * mediator[i] * mediator[i];

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let m_bar = weighted_m / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                let mm_bar = weighted_mm / risk_sum;

                grad_beta += treatment[i] - t_bar;
                grad_gamma += mediator[i] - m_bar;
                hess_beta += tt_bar - t_bar * t_bar;
                hess_gamma += mm_bar - m_bar * m_bar;
            }
        }

        if hess_beta.abs() > crate::constants::DIVISION_FLOOR {
            beta += grad_beta / hess_beta;
            beta = beta.clamp(-10.0, 10.0);
        }

        if hess_gamma.abs() > crate::constants::DIVISION_FLOOR {
            gamma += grad_gamma / hess_gamma;
            gamma = gamma.clamp(-10.0, 10.0);
        }
    }

    let se = 0.1;
    (beta, gamma, se)
}
