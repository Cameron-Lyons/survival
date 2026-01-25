#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    dead_code,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use crate::utilities::statistical::sample_normal;
use pyo3::prelude::*;
use rayon::prelude::*;

fn sample_gamma(rng: &mut fastrand::Rng, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.f64();
        return sample_gamma(rng, shape + 1.0, scale) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);

        if v > 0.0 {
            let u = rng.f64();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v * scale;
            }
            if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DirichletProcessConfig {
    #[pyo3(get, set)]
    pub concentration: f64,
    #[pyo3(get, set)]
    pub n_components: usize,
    #[pyo3(get, set)]
    pub n_iter: usize,
    #[pyo3(get, set)]
    pub burnin: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl DirichletProcessConfig {
    #[new]
    #[pyo3(signature = (concentration=1.0, n_components=10, n_iter=1000, burnin=500, seed=None))]
    pub fn new(
        concentration: f64,
        n_components: usize,
        n_iter: usize,
        burnin: usize,
        seed: Option<u64>,
    ) -> Self {
        DirichletProcessConfig {
            concentration,
            n_components,
            n_iter,
            burnin,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DirichletProcessResult {
    #[pyo3(get)]
    pub cluster_assignments: Vec<usize>,
    #[pyo3(get)]
    pub cluster_sizes: Vec<usize>,
    #[pyo3(get)]
    pub cluster_survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub posterior_mean_survival: Vec<f64>,
    #[pyo3(get)]
    pub posterior_lower: Vec<f64>,
    #[pyo3(get)]
    pub posterior_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_clusters: usize,
    #[pyo3(get)]
    pub concentration_posterior: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, n_covariates, config))]
pub fn dirichlet_process_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    n_covariates: usize,
    config: &DirichletProcessConfig,
) -> PyResult<DirichletProcessResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }

    let mut rng = match config.seed {
        Some(s) => fastrand::Rng::with_seed(s),
        None => fastrand::Rng::new(),
    };

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_grid = 50;
    let eval_times: Vec<f64> = (0..n_grid)
        .map(|i| (i as f64 + 1.0) / n_grid as f64 * max_time)
        .collect();

    let mut cluster_assignments = vec![0; n];
    for i in 0..n {
        cluster_assignments[i] = rng.usize(0..config.n_components);
    }

    let mut cluster_params: Vec<(f64, f64)> =
        (0..config.n_components).map(|_| (1.0, 1.0)).collect();

    let mut concentration = config.concentration;
    let mut survival_samples: Vec<Vec<f64>> = Vec::new();

    for iter in 0..config.n_iter {
        for i in 0..n {
            let mut probs = vec![0.0_f64; config.n_components + 1];

            for k in 0..config.n_components {
                let n_k: usize = cluster_assignments
                    .iter()
                    .enumerate()
                    .filter(|&(j, &c)| j != i && c == k)
                    .count();

                if n_k > 0 {
                    let (shape, rate) = cluster_params[k];
                    let log_lik = compute_weibull_loglik(time[i], event[i], shape, rate);
                    probs[k] = (n_k as f64).ln() + log_lik;
                } else {
                    probs[k] = f64::NEG_INFINITY;
                }
            }

            let prior_shape = 1.0;
            let prior_rate = 1.0;
            let log_lik_new = compute_weibull_loglik(time[i], event[i], prior_shape, prior_rate);
            probs[config.n_components] = concentration.ln() + log_lik_new;

            let max_log = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = probs.iter().map(|&p| (p - max_log).exp()).sum();
            let probs_normalized: Vec<f64> = probs
                .iter()
                .map(|&p| (p - max_log).exp() / sum_exp)
                .collect();

            let u: f64 = rng.f64();
            let mut cumsum = 0.0;
            let mut new_cluster = config.n_components;
            for k in 0..=config.n_components {
                cumsum += probs_normalized[k];
                if u <= cumsum {
                    new_cluster = k;
                    break;
                }
            }

            if new_cluster == config.n_components && config.n_components < 50 {
                cluster_params.push((1.0, 1.0));
            }

            cluster_assignments[i] = new_cluster.min(cluster_params.len() - 1);
        }

        for k in 0..cluster_params.len() {
            let cluster_obs: Vec<usize> = (0..n).filter(|&i| cluster_assignments[i] == k).collect();

            if !cluster_obs.is_empty() {
                let (new_shape, new_rate) = sample_weibull_posterior(
                    &cluster_obs,
                    &time,
                    &event,
                    cluster_params[k].0,
                    cluster_params[k].1,
                    &mut rng,
                );
                cluster_params[k] = (new_shape, new_rate);
            }
        }

        let a_alpha = 1.0;
        let b_alpha = 1.0;
        let n_clusters_used = cluster_params.len();
        concentration = sample_gamma(
            &mut rng,
            a_alpha + n_clusters_used as f64,
            1.0 / (b_alpha + harmonic(n)),
        )
        .max(0.1);

        if iter >= config.burnin {
            let survival: Vec<f64> = eval_times
                .iter()
                .map(|&t| {
                    let mut s = 0.0;
                    let mut total = 0.0;
                    for k in 0..cluster_params.len() {
                        let n_k = cluster_assignments.iter().filter(|&&c| c == k).count();
                        if n_k > 0 {
                            let (shape, rate) = cluster_params[k];
                            let surv_k = (-(t * rate).powf(shape)).exp();
                            s += n_k as f64 * surv_k;
                            total += n_k as f64;
                        }
                    }
                    if total > 0.0 { s / total } else { 1.0 }
                })
                .collect();
            survival_samples.push(survival);
        }
    }

    let n_samples = survival_samples.len();
    let posterior_mean_survival: Vec<f64> = (0..n_grid)
        .map(|t| survival_samples.iter().map(|s| s[t]).sum::<f64>() / n_samples as f64)
        .collect();

    let posterior_lower: Vec<f64> = (0..n_grid)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.025 * n_samples as f64) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let posterior_upper: Vec<f64> = (0..n_grid)
        .map(|t| {
            let mut vals: Vec<f64> = survival_samples.iter().map(|s| s[t]).collect();
            vals.sort_by(|a, b| a.total_cmp(b));
            let idx = ((0.975 * n_samples as f64) as usize).min(vals.len().saturating_sub(1));
            vals.get(idx).copied().unwrap_or(0.0)
        })
        .collect();

    let mut cluster_sizes: Vec<usize> = vec![0; cluster_params.len()];
    for &c in &cluster_assignments {
        if c < cluster_sizes.len() {
            cluster_sizes[c] += 1;
        }
    }
    let n_clusters = cluster_sizes.iter().filter(|&&s| s > 0).count();

    let cluster_survival: Vec<Vec<f64>> = cluster_params
        .iter()
        .map(|(shape, rate)| {
            eval_times
                .iter()
                .map(|&t| (-(t * rate).powf(*shape)).exp())
                .collect()
        })
        .collect();

    Ok(DirichletProcessResult {
        cluster_assignments,
        cluster_sizes,
        cluster_survival,
        eval_times,
        posterior_mean_survival,
        posterior_lower,
        posterior_upper,
        n_clusters,
        concentration_posterior: concentration,
    })
}

fn compute_weibull_loglik(t: f64, event: i32, shape: f64, rate: f64) -> f64 {
    let log_surv = -(t * rate).powf(shape);
    if event == 1 {
        shape.ln() + (shape - 1.0) * t.max(1e-10).ln() + shape * rate.ln() + log_surv
    } else {
        log_surv
    }
}

fn sample_weibull_posterior(
    obs_idx: &[usize],
    time: &[f64],
    event: &[i32],
    shape: f64,
    rate: f64,
    rng: &mut fastrand::Rng,
) -> (f64, f64) {
    let sum_events: f64 = obs_idx.iter().map(|&i| event[i] as f64).sum();
    let sum_t_shape: f64 = obs_idx.iter().map(|&i| time[i].powf(shape)).sum();

    let new_rate_shape = 1.0 + sum_events;
    let new_rate_rate = 1.0 + sum_t_shape;

    let new_rate = sample_gamma(rng, new_rate_shape, 1.0 / new_rate_rate).max(0.01);

    let new_shape = shape + sample_normal(rng) * 0.1;
    let new_shape = new_shape.clamp(0.1, 10.0);

    (new_shape, new_rate)
}

fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
    model_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

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
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

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

#[derive(Debug, Clone)]
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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
#[pyclass]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_process() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = DirichletProcessConfig::new(1.0, 5, 100, 50, Some(42));
        let result = dirichlet_process_survival(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.cluster_assignments.len(), 8);
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            1.0, 0.5, 0.0, 0.8, 1.0, 0.3, 0.0, 0.9, 1.0, 0.2, 0.0, 0.7, 1.0, 0.1, 0.0, 0.6,
        ];

        let config = BayesianModelAveragingConfig::new(200, 100, 0.5, Some(42));
        let result = bayesian_model_averaging_cox(time, event, covariates, 2, &config).unwrap();

        assert_eq!(result.n_vars, 2);
        assert_eq!(result.posterior_inclusion_prob.len(), 2);
    }

    #[test]
    fn test_spike_slab() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = SpikeSlabConfig::new(0.001, 10.0, 0.5, 200, 100, Some(42));
        let result = spike_slab_cox(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.posterior_inclusion_prob.len(), 1);
        assert_eq!(result.posterior_mean.len(), 1);
    }

    #[test]
    fn test_horseshoe() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = HorseshoeConfig::new(1.0, 200, 100, Some(42));
        let result = horseshoe_cox(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.posterior_mean.len(), 1);
        assert_eq!(result.shrinkage_factors.len(), 1);
        assert!(result.effective_df >= 0.0);
    }
}
