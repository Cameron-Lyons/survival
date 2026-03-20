
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SpikeSlabConfig {
    #[pyo3(get, set)]
    pub spike_var: f64,
    #[pyo3(get, set)]
    pub slab_var: f64,
    #[pyo3(get, set)]
    pub prior_inclusion: f64,
    #[pyo3(get, set)]
    pub n_iter: usize,
    #[pyo3(get, set)]
    pub burnin: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl SpikeSlabConfig {
    #[new]
    #[pyo3(signature = (spike_var=0.001, slab_var=10.0, prior_inclusion=0.5, n_iter=2000, burnin=1000, seed=None))]
    pub fn new(
        spike_var: f64,
        slab_var: f64,
        prior_inclusion: f64,
        n_iter: usize,
        burnin: usize,
        seed: Option<u64>,
    ) -> Self {
        SpikeSlabConfig {
            spike_var,
            slab_var,
            prior_inclusion,
            n_iter,
            burnin,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SpikeSlabResult {
    #[pyo3(get)]
    pub posterior_inclusion_prob: Vec<f64>,
    #[pyo3(get)]
    pub posterior_mean: Vec<f64>,
    #[pyo3(get)]
    pub posterior_sd: Vec<f64>,
    #[pyo3(get)]
    pub credible_lower: Vec<f64>,
    #[pyo3(get)]
    pub credible_upper: Vec<f64>,
    #[pyo3(get)]
    pub selected_variables: Vec<usize>,
    #[pyo3(get)]
    pub n_selected: usize,
    #[pyo3(get)]
    pub log_marginal_likelihood: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, config))]
pub fn spike_slab_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    config: &SpikeSlabConfig,
) -> PyResult<SpikeSlabResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut rng = match config.seed {
        Some(s) => fastrand::Rng::with_seed(s),
        None => fastrand::Rng::new(),
    };

    let mut gamma = vec![false; n_covariates];
    let mut beta = vec![0.0; n_covariates];

    let mut inclusion_counts = vec![0usize; n_covariates];
    let mut beta_samples: Vec<Vec<f64>> = Vec::new();
    let mut log_marg_lik_sum = 0.0;
    let mut n_marg_samples = 0;

    for iter in 0..config.n_iter {
        for j in 0..n_covariates {
            let (grad, hess) =
                compute_gradient_hessian_single(&time, &event, &covariates, n_covariates, &beta, j);

            let log_lik_spike =
                -0.5 * beta[j].powi(2) / config.spike_var - 0.5 * config.spike_var.ln();
            let log_lik_slab =
                -0.5 * beta[j].powi(2) / config.slab_var - 0.5 * config.slab_var.ln();

            let log_prior_ratio = (config.prior_inclusion / (1.0 - config.prior_inclusion)).ln();
            let log_bf = log_lik_slab - log_lik_spike;

            let prob_gamma_1 = 1.0 / (1.0 + (-log_prior_ratio - log_bf).exp());

            gamma[j] = rng.f64() < prob_gamma_1;

            let prior_var = if gamma[j] {
                config.slab_var
            } else {
                config.spike_var
            };
            let posterior_var = 1.0 / (hess.abs() + 1.0 / prior_var);
            let posterior_mean = posterior_var * (hess * beta[j] + grad);

            beta[j] = posterior_mean + sample_normal(&mut rng) * posterior_var.sqrt();
        }

        if iter >= config.burnin {
            for j in 0..n_covariates {
                if gamma[j] {
                    inclusion_counts[j] += 1;
                }
            }
            beta_samples.push(beta.clone());

            let log_lik =
                compute_cox_loglik(&time, &event, &beta, &gamma, &covariates, n_covariates);
            log_marg_lik_sum += log_lik;
            n_marg_samples += 1;
        }
    }

    let n_post = (config.n_iter - config.burnin) as f64;
    let posterior_inclusion_prob: Vec<f64> = inclusion_counts
        .iter()
        .map(|&c| c as f64 / n_post)
        .collect();

    let posterior_mean: Vec<f64> = (0..n_covariates)
        .map(|j| beta_samples.iter().map(|b| b[j]).sum::<f64>() / n_post)
        .collect();

    let posterior_sd: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mean = posterior_mean[j];
            let var: f64 = beta_samples
                .iter()
                .map(|b| (b[j] - mean).powi(2))
                .sum::<f64>()
                / n_post;
            var.sqrt()
        })
        .collect();

    let credible_lower: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mut vals: Vec<f64> = beta_samples.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.025 * n_post) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let credible_upper: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mut vals: Vec<f64> = beta_samples.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.975 * n_post) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let selected_variables: Vec<usize> = (0..n_covariates)
        .filter(|&j| posterior_inclusion_prob[j] > 0.5)
        .collect();

    let log_marginal_likelihood = if n_marg_samples > 0 {
        log_marg_lik_sum / n_marg_samples as f64
    } else {
        0.0
    };

    Ok(SpikeSlabResult {
        posterior_inclusion_prob,
        posterior_mean,
        posterior_sd,
        credible_lower,
        credible_upper,
        selected_variables: selected_variables.clone(),
        n_selected: selected_variables.len(),
        log_marginal_likelihood,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct HorseshoeConfig {
    #[pyo3(get, set)]
    pub tau_global: f64,
    #[pyo3(get, set)]
    pub n_iter: usize,
    #[pyo3(get, set)]
    pub burnin: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl HorseshoeConfig {
    #[new]
    #[pyo3(signature = (tau_global=1.0, n_iter=2000, burnin=1000, seed=None))]
    pub fn new(tau_global: f64, n_iter: usize, burnin: usize, seed: Option<u64>) -> Self {
        HorseshoeConfig {
            tau_global,
            n_iter,
            burnin,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct HorseshoeResult {
    #[pyo3(get)]
    pub posterior_mean: Vec<f64>,
    #[pyo3(get)]
    pub posterior_sd: Vec<f64>,
    #[pyo3(get)]
    pub credible_lower: Vec<f64>,
    #[pyo3(get)]
    pub credible_upper: Vec<f64>,
    #[pyo3(get)]
    pub shrinkage_factors: Vec<f64>,
    #[pyo3(get)]
    pub local_scales: Vec<f64>,
    #[pyo3(get)]
    pub global_scale: f64,
    #[pyo3(get)]
    pub effective_df: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, config))]
pub fn horseshoe_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    config: &HorseshoeConfig,
) -> PyResult<HorseshoeResult> {
    let n = time.len();
    if event.len() != n || covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input dimensions mismatch",
        ));
    }

    let mut rng = match config.seed {
        Some(s) => fastrand::Rng::with_seed(s),
        None => fastrand::Rng::new(),
    };

    let mut beta: Vec<f64> = vec![0.0; n_covariates];
    let mut lambda: Vec<f64> = vec![1.0; n_covariates];
    let mut nu: Vec<f64> = vec![1.0; n_covariates];
    let mut tau: f64 = config.tau_global;
    let mut xi: f64 = 1.0;

    let mut beta_samples: Vec<Vec<f64>> = Vec::new();
    let mut tau_samples: Vec<f64> = Vec::new();
    let mut lambda_samples: Vec<Vec<f64>> = Vec::new();

    for iter in 0..config.n_iter {
        for j in 0..n_covariates {
            let (grad, hess) =
                compute_gradient_hessian_single(&time, &event, &covariates, n_covariates, &beta, j);

            let prior_var: f64 = tau.powi(2) * lambda[j].powi(2);
            let posterior_var = 1.0 / (hess.abs() + 1.0 / prior_var);
            let posterior_mean = posterior_var * (hess * beta[j] + grad);

            beta[j] = posterior_mean + sample_normal(&mut rng) * posterior_var.sqrt();
        }

        for j in 0..n_covariates {
            let rate = 1.0 / nu[j] + beta[j].powi(2) / (2.0 * tau.powi(2));
            let lambda_sq_inv = sample_gamma(&mut rng, 1.0, 1.0 / rate);
            lambda[j] = (1.0 / lambda_sq_inv).sqrt().max(1e-10);

            let rate_nu = 1.0 + 1.0 / lambda[j].powi(2);
            nu[j] = 1.0 / sample_gamma(&mut rng, 1.0, 1.0 / rate_nu).max(1e-10);
        }

        let sum_b2_l2: f64 = (0..n_covariates)
            .map(|j| beta[j].powi(2) / lambda[j].powi(2))
            .sum();
        let rate_tau = 1.0 / xi + sum_b2_l2 / 2.0;
        let tau_sq_inv = sample_gamma(&mut rng, (n_covariates as f64 + 1.0) / 2.0, 1.0 / rate_tau);
        tau = (1.0 / tau_sq_inv).sqrt().max(1e-10);

        let rate_xi = 1.0 + 1.0 / tau.powi(2);
        xi = 1.0 / sample_gamma(&mut rng, 1.0, 1.0 / rate_xi).max(1e-10);

        if iter >= config.burnin {
            beta_samples.push(beta.clone());
            tau_samples.push(tau);
            lambda_samples.push(lambda.clone());
        }
    }

    let n_post = beta_samples.len() as f64;

    let posterior_mean: Vec<f64> = (0..n_covariates)
        .map(|j| beta_samples.iter().map(|b| b[j]).sum::<f64>() / n_post)
        .collect();

    let posterior_sd: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mean = posterior_mean[j];
            let var: f64 = beta_samples
                .iter()
                .map(|b| (b[j] - mean).powi(2))
                .sum::<f64>()
                / n_post;
            var.sqrt()
        })
        .collect();

    let credible_lower: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mut vals: Vec<f64> = beta_samples.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.025 * n_post) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let credible_upper: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mut vals: Vec<f64> = beta_samples.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.975 * n_post) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let global_scale = tau_samples.iter().sum::<f64>() / tau_samples.len() as f64;

    let local_scales: Vec<f64> = (0..n_covariates)
        .map(|j| lambda_samples.iter().map(|l| l[j]).sum::<f64>() / n_post)
        .collect();

    let shrinkage_factors: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let kappa: f64 = lambda_samples
                .iter()
                .zip(tau_samples.iter())
                .map(|(l, &t)| {
                    let s = t.powi(2) * l[j].powi(2);
                    s / (1.0 + s)
                })
                .sum::<f64>()
                / n_post;
            kappa
        })
        .collect();

    let effective_df: f64 = shrinkage_factors.iter().sum();

    Ok(HorseshoeResult {
        posterior_mean,
        posterior_sd,
        credible_lower,
        credible_upper,
        shrinkage_factors,
        local_scales,
        global_scale,
        effective_df,
    })
}

