
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BayesianModelAveragingConfig {
    #[pyo3(get, set)]
    pub n_iter: usize,
    #[pyo3(get, set)]
    pub burnin: usize,
    #[pyo3(get, set)]
    pub prior_inclusion_prob: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl BayesianModelAveragingConfig {
    #[new]
    #[pyo3(signature = (n_iter=2000, burnin=1000, prior_inclusion_prob=0.5, seed=None))]
    pub fn new(n_iter: usize, burnin: usize, prior_inclusion_prob: f64, seed: Option<u64>) -> Self {
        BayesianModelAveragingConfig {
            n_iter,
            burnin,
            prior_inclusion_prob,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BayesianModelAveragingResult {
    #[pyo3(get)]
    pub posterior_inclusion_prob: Vec<f64>,
    #[pyo3(get)]
    pub posterior_mean_coef: Vec<f64>,
    #[pyo3(get)]
    pub posterior_sd_coef: Vec<f64>,
    #[pyo3(get)]
    pub model_posterior_probs: Vec<f64>,
    #[pyo3(get)]
    pub best_model_indices: Vec<usize>,
    #[pyo3(get)]
    pub bayes_factor_vs_null: Vec<f64>,
    #[pyo3(get)]
    pub n_models_visited: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, config))]
pub fn bayesian_model_averaging_cox(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    config: &BayesianModelAveragingConfig,
) -> PyResult<BayesianModelAveragingResult> {
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

    let mut included = vec![true; n_covariates];
    let mut beta = vec![0.0; n_covariates];

    let mut inclusion_counts = vec![0usize; n_covariates];
    let mut beta_samples: Vec<Vec<f64>> = Vec::new();
    let mut model_counts: std::collections::HashMap<Vec<bool>, usize> =
        std::collections::HashMap::new();

    for iter in 0..config.n_iter {
        for j in 0..n_covariates {
            let current_loglik =
                compute_cox_loglik(&time, &event, &beta, &included, &covariates, n_covariates);

            let mut proposed_included = included.clone();
            proposed_included[j] = !proposed_included[j];

            let proposed_loglik = compute_cox_loglik(
                &time,
                &event,
                &beta,
                &proposed_included,
                &covariates,
                n_covariates,
            );

            let prior_ratio = if proposed_included[j] {
                config.prior_inclusion_prob / (1.0 - config.prior_inclusion_prob)
            } else {
                (1.0 - config.prior_inclusion_prob) / config.prior_inclusion_prob
            };

            let log_accept = proposed_loglik - current_loglik + prior_ratio.ln();

            if log_accept > 0.0 || rng.f64() < log_accept.exp() {
                included[j] = proposed_included[j];
            }
        }

        let active_vars: Vec<usize> = (0..n_covariates).filter(|&j| included[j]).collect();

        if !active_vars.is_empty() {
            for &j in &active_vars {
                let (grad, hess) = compute_gradient_hessian_single(
                    &time,
                    &event,
                    &covariates,
                    n_covariates,
                    &beta,
                    j,
                );

                if hess.abs() > 1e-10 {
                    let prior_var = 10.0;
                    let posterior_var = 1.0 / (hess.abs() + 1.0 / prior_var);
                    let posterior_mean = posterior_var * (hess * beta[j] + grad);

                    beta[j] = posterior_mean + sample_normal(&mut rng) * posterior_var.sqrt();
                }
            }
        }

        for j in 0..n_covariates {
            if !included[j] {
                beta[j] = 0.0;
            }
        }

        if iter >= config.burnin {
            for j in 0..n_covariates {
                if included[j] {
                    inclusion_counts[j] += 1;
                }
            }
            beta_samples.push(beta.clone());
            *model_counts.entry(included.clone()).or_insert(0) += 1;
        }
    }

    let n_post = (config.n_iter - config.burnin) as f64;
    let posterior_inclusion_prob: Vec<f64> = inclusion_counts
        .iter()
        .map(|&c| c as f64 / n_post)
        .collect();

    let posterior_mean_coef: Vec<f64> = (0..n_covariates)
        .map(|j| beta_samples.iter().map(|b| b[j]).sum::<f64>() / n_post)
        .collect();

    let posterior_sd_coef: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let mean = posterior_mean_coef[j];
            let var: f64 = beta_samples
                .iter()
                .map(|b| (b[j] - mean).powi(2))
                .sum::<f64>()
                / n_post;
            var.sqrt()
        })
        .collect();

    let total_model_counts: usize = model_counts.values().sum();
    let mut model_probs: Vec<(Vec<bool>, f64)> = model_counts
        .iter()
        .map(|(m, &c)| (m.clone(), c as f64 / total_model_counts as f64))
        .collect();
    model_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let best_model = if !model_probs.is_empty() {
        model_probs[0].0.clone()
    } else {
        vec![false; n_covariates]
    };

    let best_model_indices: Vec<usize> = (0..n_covariates).filter(|&j| best_model[j]).collect();

    let model_posterior_probs: Vec<f64> = model_probs.iter().take(10).map(|(_, p)| *p).collect();

    let bayes_factor_vs_null: Vec<f64> = (0..n_covariates)
        .map(|j| {
            let pip = posterior_inclusion_prob[j];
            let prior_odds = config.prior_inclusion_prob / (1.0 - config.prior_inclusion_prob);
            let posterior_odds = pip / (1.0 - pip + 1e-10);
            posterior_odds / prior_odds
        })
        .collect();

    Ok(BayesianModelAveragingResult {
        posterior_inclusion_prob,
        posterior_mean_coef,
        posterior_sd_coef,
        model_posterior_probs,
        best_model_indices,
        bayes_factor_vs_null,
        n_models_visited: model_counts.len(),
        n_vars: n_covariates,
    })
}

fn compute_cox_loglik(
    time: &[f64],
    event: &[i32],
    beta: &[f64],
    included: &[bool],
    covariates: &[f64],
    n_covariates: usize,
) -> f64 {
    let n = time.len();

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                if included[j] {
                    e += covariates[i * n_covariates + j] * beta[j];
                }
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut log_lik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &sorted_indices {
        risk_sum += exp_eta[i];
        if event[i] == 1 && risk_sum > 0.0 {
            log_lik += eta[i] - risk_sum.ln();
        }
    }

    log_lik
}

fn compute_gradient_hessian_single(
    time: &[f64],
    event: &[i32],
    covariates: &[f64],
    n_covariates: usize,
    beta: &[f64],
    var_idx: usize,
) -> (f64, f64) {
    let n = time.len();

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..n_covariates {
                e += covariates[i * n_covariates + j] * beta[j];
            }
            e.clamp(-500.0, 500.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut gradient = 0.0;
    let mut hessian = 0.0;
    let mut risk_sum = 0.0;
    let mut weighted_x = 0.0;
    let mut weighted_xx = 0.0;

    for &i in &sorted_indices {
        let x_j = covariates[i * n_covariates + var_idx];
        risk_sum += exp_eta[i];
        weighted_x += exp_eta[i] * x_j;
        weighted_xx += exp_eta[i] * x_j * x_j;

        if event[i] == 1 && risk_sum > 0.0 {
            let x_bar = weighted_x / risk_sum;
            let xx_bar = weighted_xx / risk_sum;
            gradient += x_j - x_bar;
            hessian -= xx_bar - x_bar * x_bar;
        }
    }

    (gradient, hessian)
}
