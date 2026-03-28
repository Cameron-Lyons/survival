

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
#[pyclass(from_py_object)]
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
#[pyclass(from_py_object)]
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
    if covariates.len() != n * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates length must equal n_obs * n_covariates",
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
    for assignment in cluster_assignments.iter_mut().take(n) {
        *assignment = rng.usize(0..config.n_components);
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
            for (k, prob) in probs_normalized
                .iter()
                .enumerate()
                .take(config.n_components + 1)
            {
                cumsum += *prob;
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

        for (k, params) in cluster_params.iter_mut().enumerate() {
            let cluster_obs: Vec<usize> = (0..n).filter(|&i| cluster_assignments[i] == k).collect();

            if !cluster_obs.is_empty() {
                let (new_shape, new_rate) = sample_weibull_posterior(
                    &cluster_obs,
                    &time,
                    &event,
                    params.0,
                    params.1,
                    &mut rng,
                );
                *params = (new_shape, new_rate);
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
                    for (k, &(shape, rate)) in cluster_params.iter().enumerate() {
                        let n_k = cluster_assignments.iter().filter(|&&c| c == k).count();
                        if n_k > 0 {
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
    _rate: f64,
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
