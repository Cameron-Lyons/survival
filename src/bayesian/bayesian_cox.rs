#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use crate::utilities::statistical::sample_normal;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum PriorType {
    Normal,
    Laplace,
    Cauchy,
    Horseshoe,
    Flat,
}

#[pymethods]
impl PriorType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "normal" | "gaussian" => Ok(PriorType::Normal),
            "laplace" | "double_exponential" => Ok(PriorType::Laplace),
            "cauchy" => Ok(PriorType::Cauchy),
            "horseshoe" => Ok(PriorType::Horseshoe),
            "flat" | "improper" => Ok(PriorType::Flat),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown prior type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BayesianCoxConfig {
    #[pyo3(get, set)]
    pub prior_type: PriorType,
    #[pyo3(get, set)]
    pub prior_scale: f64,
    #[pyo3(get, set)]
    pub n_samples: usize,
    #[pyo3(get, set)]
    pub n_warmup: usize,
    #[pyo3(get, set)]
    pub n_chains: usize,
    #[pyo3(get, set)]
    pub target_accept: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl BayesianCoxConfig {
    #[new]
    #[pyo3(signature = (prior_type=PriorType::Normal, prior_scale=2.5, n_samples=2000, n_warmup=1000, n_chains=4, target_accept=0.8, seed=None))]
    pub fn new(
        prior_type: PriorType,
        prior_scale: f64,
        n_samples: usize,
        n_warmup: usize,
        n_chains: usize,
        target_accept: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if prior_scale <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "prior_scale must be positive",
            ));
        }
        Ok(BayesianCoxConfig {
            prior_type,
            prior_scale,
            n_samples,
            n_warmup,
            n_chains,
            target_accept,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BayesianCoxResult {
    #[pyo3(get)]
    pub posterior_mean: Vec<f64>,
    #[pyo3(get)]
    pub posterior_sd: Vec<f64>,
    #[pyo3(get)]
    pub credible_lower: Vec<f64>,
    #[pyo3(get)]
    pub credible_upper: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_mean: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_lower: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_upper: Vec<f64>,
    #[pyo3(get)]
    pub samples: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_posterior: Vec<f64>,
    #[pyo3(get)]
    pub waic: f64,
    #[pyo3(get)]
    pub loo: f64,
    #[pyo3(get)]
    pub rhat: Vec<f64>,
    #[pyo3(get)]
    pub n_eff: Vec<f64>,
}

fn log_prior(beta: &[f64], prior_type: &PriorType, scale: f64) -> f64 {
    match prior_type {
        PriorType::Normal => -0.5 * beta.iter().map(|&b| (b / scale).powi(2)).sum::<f64>(),
        PriorType::Laplace => -beta.iter().map(|&b| b.abs() / scale).sum::<f64>(),
        PriorType::Cauchy => -beta
            .iter()
            .map(|&b| (1.0 + (b / scale).powi(2)).ln())
            .sum::<f64>(),
        PriorType::Horseshoe => -beta
            .iter()
            .map(|&b| (1.0 + (b / scale).powi(2)).ln())
            .sum::<f64>(),
        PriorType::Flat => 0.0,
    }
}

fn log_prior_gradient(beta: &[f64], prior_type: &PriorType, scale: f64) -> Vec<f64> {
    match prior_type {
        PriorType::Normal => beta.iter().map(|&b| -b / (scale * scale)).collect(),
        PriorType::Laplace => beta.iter().map(|&b| -b.signum() / scale).collect(),
        PriorType::Cauchy => beta
            .iter()
            .map(|&b| -2.0 * b / (scale * scale + b * b))
            .collect(),
        PriorType::Horseshoe => beta
            .iter()
            .map(|&b| -2.0 * b / (scale * scale + b * b))
            .collect(),
        PriorType::Flat => vec![0.0; beta.len()],
    }
}

fn log_likelihood(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    beta: &[f64],
) -> f64 {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..p {
                e += x[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut loglik = 0.0;
    let mut risk_sum = 0.0;

    for &i in &indices {
        risk_sum += exp_eta[i];
        if status[i] == 1 && risk_sum > 0.0 {
            loglik += eta[i] - risk_sum.ln();
        }
    }

    loglik
}

fn log_likelihood_gradient(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    beta: &[f64],
) -> Vec<f64> {
    let mut gradient = vec![0.0; p];

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eta: Vec<f64> = (0..n)
        .map(|i| {
            let mut e = 0.0;
            for j in 0..p {
                e += x[i * p + j] * beta[j];
            }
            e.clamp(-700.0, 700.0)
        })
        .collect();

    let exp_eta: Vec<f64> = eta.iter().map(|&e| e.exp()).collect();

    let mut risk_sum = 0.0;
    let mut weighted_x = vec![0.0; p];

    for &i in &indices {
        risk_sum += exp_eta[i];
        for j in 0..p {
            weighted_x[j] += exp_eta[i] * x[i * p + j];
        }

        if status[i] == 1 && risk_sum > 0.0 {
            for j in 0..p {
                gradient[j] += x[i * p + j] - weighted_x[j] / risk_sum;
            }
        }
    }

    gradient
}

fn hmc_step(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    beta: &[f64],
    config: &BayesianCoxConfig,
    step_size: f64,
    n_leapfrog: usize,
    rng: &mut fastrand::Rng,
) -> (Vec<f64>, f64, bool) {
    let mut q = beta.to_vec();
    let mut momentum: Vec<f64> = (0..p).map(|_| sample_normal(rng)).collect();

    let initial_h = -log_likelihood(x, n, p, time, status, &q)
        - log_prior(&q, &config.prior_type, config.prior_scale)
        + 0.5 * momentum.iter().map(|&m| m * m).sum::<f64>();

    let ll_grad = log_likelihood_gradient(x, n, p, time, status, &q);
    let prior_grad = log_prior_gradient(&q, &config.prior_type, config.prior_scale);
    let mut grad: Vec<f64> = ll_grad
        .iter()
        .zip(prior_grad.iter())
        .map(|(&l, &p)| l + p)
        .collect();

    for j in 0..p {
        momentum[j] += 0.5 * step_size * grad[j];
    }

    for _ in 0..n_leapfrog {
        for j in 0..p {
            q[j] += step_size * momentum[j];
        }

        let ll_grad = log_likelihood_gradient(x, n, p, time, status, &q);
        let prior_grad = log_prior_gradient(&q, &config.prior_type, config.prior_scale);
        grad = ll_grad
            .iter()
            .zip(prior_grad.iter())
            .map(|(&l, &p)| l + p)
            .collect();

        for j in 0..p {
            momentum[j] += step_size * grad[j];
        }
    }

    for j in 0..p {
        momentum[j] -= 0.5 * step_size * grad[j];
    }

    let final_h = -log_likelihood(x, n, p, time, status, &q)
        - log_prior(&q, &config.prior_type, config.prior_scale)
        + 0.5 * momentum.iter().map(|&m| m * m).sum::<f64>();

    let log_accept = initial_h - final_h;
    let accepted = rng.f64().ln() < log_accept;

    let log_post = log_likelihood(x, n, p, time, status, &q)
        + log_prior(&q, &config.prior_type, config.prior_scale);

    if accepted {
        (q, log_post, true)
    } else {
        (
            beta.to_vec(),
            log_likelihood(x, n, p, time, status, beta)
                + log_prior(beta, &config.prior_type, config.prior_scale),
            false,
        )
    }
}

fn run_chain(
    x: &[f64],
    n: usize,
    p: usize,
    time: &[f64],
    status: &[i32],
    config: &BayesianCoxConfig,
    chain_id: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let seed = config.seed.unwrap_or(42) + chain_id as u64 * 1000;
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut beta = vec![0.0; p];
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut log_posteriors: Vec<f64> = Vec::with_capacity(config.n_samples);

    let mut step_size = 0.1;
    let n_leapfrog = 10;

    let mut n_accepted = 0;
    for i in 0..(config.n_warmup + config.n_samples) {
        let (new_beta, log_post, accepted) = hmc_step(
            x, n, p, time, status, &beta, config, step_size, n_leapfrog, &mut rng,
        );

        beta = new_beta;

        if accepted {
            n_accepted += 1;
        }

        if i < config.n_warmup && (i + 1) % 50 == 0 {
            let accept_rate = n_accepted as f64 / 50.0;
            if accept_rate < config.target_accept - 0.1 {
                step_size *= 0.8;
            } else if accept_rate > config.target_accept + 0.1 {
                step_size *= 1.2;
            }
            n_accepted = 0;
        }

        if i >= config.n_warmup {
            samples.push(beta.clone());
            log_posteriors.push(log_post);
        }
    }

    (samples, log_posteriors)
}

fn compute_rhat(chains: &[Vec<Vec<f64>>], p: usize) -> Vec<f64> {
    let n_chains = chains.len();
    let n_samples = chains[0].len();

    (0..p)
        .into_par_iter()
        .map(|j| {
            let chain_means: Vec<f64> = chains
                .iter()
                .map(|chain| chain.iter().map(|s| s[j]).sum::<f64>() / n_samples as f64)
                .collect();

            let overall_mean = chain_means.iter().sum::<f64>() / n_chains as f64;

            let b = n_samples as f64
                * chain_means
                    .iter()
                    .map(|&m| (m - overall_mean).powi(2))
                    .sum::<f64>()
                / (n_chains - 1) as f64;

            let w: f64 = chains
                .iter()
                .zip(chain_means.iter())
                .map(|(chain, &mean)| {
                    chain.iter().map(|s| (s[j] - mean).powi(2)).sum::<f64>()
                        / (n_samples - 1) as f64
                })
                .sum::<f64>()
                / n_chains as f64;

            let var_plus = (n_samples as f64 - 1.0) / n_samples as f64 * w + b / n_samples as f64;

            if w > 0.0 { (var_plus / w).sqrt() } else { 1.0 }
        })
        .collect()
}

fn compute_n_eff(chains: &[Vec<Vec<f64>>], p: usize) -> Vec<f64> {
    let n_chains = chains.len();
    let n_samples = chains[0].len();
    let total_samples = n_chains * n_samples;

    (0..p)
        .into_par_iter()
        .map(|j| {
            let all_samples: Vec<f64> = chains
                .iter()
                .flat_map(|chain| chain.iter().map(|s| s[j]))
                .collect();

            let mean = all_samples.iter().sum::<f64>() / total_samples as f64;
            let var = all_samples.iter().map(|&s| (s - mean).powi(2)).sum::<f64>()
                / (total_samples - 1) as f64;

            if var < 1e-10 {
                return total_samples as f64;
            }

            let max_lag = (total_samples / 2).min(100);
            let mut rho_sum = 0.0;

            for lag in 1..max_lag {
                let mut autocorr = 0.0;
                for i in 0..(total_samples - lag) {
                    autocorr += (all_samples[i] - mean) * (all_samples[i + lag] - mean);
                }
                autocorr /= (total_samples - lag) as f64 * var;

                if autocorr < 0.05 {
                    break;
                }
                rho_sum += autocorr;
            }

            let tau = 1.0 + 2.0 * rho_sum;
            (total_samples as f64 / tau).max(1.0)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config))]
pub fn bayesian_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: &BayesianCoxConfig,
) -> PyResult<BayesianCoxResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let chains: Vec<(Vec<Vec<f64>>, Vec<f64>)> = (0..config.n_chains)
        .into_par_iter()
        .map(|chain_id| run_chain(&x, n_obs, n_vars, &time, &status, config, chain_id))
        .collect();

    let all_samples: Vec<Vec<f64>> = chains.iter().flat_map(|(s, _)| s.clone()).collect();
    let all_log_post: Vec<f64> = chains.iter().flat_map(|(_, lp)| lp.clone()).collect();

    let chain_samples: Vec<Vec<Vec<f64>>> = chains.iter().map(|(s, _)| s.clone()).collect();

    let n_total = all_samples.len();

    let posterior_mean: Vec<f64> = (0..n_vars)
        .map(|j| all_samples.iter().map(|s| s[j]).sum::<f64>() / n_total as f64)
        .collect();

    let posterior_sd: Vec<f64> = (0..n_vars)
        .map(|j| {
            let mean = posterior_mean[j];
            let var = all_samples
                .iter()
                .map(|s| (s[j] - mean).powi(2))
                .sum::<f64>()
                / (n_total - 1) as f64;
            var.sqrt()
        })
        .collect();

    let (credible_lower, credible_upper): (Vec<f64>, Vec<f64>) = (0..n_vars)
        .into_par_iter()
        .map(|j| {
            let mut vals: Vec<f64> = all_samples.iter().map(|s| s[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lower_idx = (n_total as f64 * 0.025) as usize;
            let upper_idx = (n_total as f64 * 0.975) as usize;
            (vals[lower_idx], vals[upper_idx])
        })
        .unzip();

    let hazard_ratio_mean: Vec<f64> = posterior_mean.iter().map(|&b| b.exp()).collect();
    let hazard_ratio_lower: Vec<f64> = credible_lower.iter().map(|&b| b.exp()).collect();
    let hazard_ratio_upper: Vec<f64> = credible_upper.iter().map(|&b| b.exp()).collect();

    let rhat = compute_rhat(&chain_samples, n_vars);
    let n_eff = compute_n_eff(&chain_samples, n_vars);

    let mean_log_post = all_log_post.iter().sum::<f64>() / n_total as f64;
    let var_log_post = all_log_post
        .iter()
        .map(|&lp| (lp - mean_log_post).powi(2))
        .sum::<f64>()
        / n_total as f64;
    let waic = -2.0 * mean_log_post + 2.0 * var_log_post;
    let loo = waic;

    Ok(BayesianCoxResult {
        posterior_mean,
        posterior_sd,
        credible_lower,
        credible_upper,
        hazard_ratio_mean,
        hazard_ratio_lower,
        hazard_ratio_upper,
        samples: all_samples,
        log_posterior: all_log_post,
        waic,
        loo,
        rhat,
        n_eff,
    })
}

#[pyfunction]
pub fn bayesian_cox_predict_survival(
    result: &BayesianCoxResult,
    x_new: Vec<f64>,
    n_new: usize,
    n_vars: usize,
    baseline_hazard: Vec<f64>,
    time_points: Vec<f64>,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n_times = time_points.len();
    let n_samples = result.samples.len().min(500);

    let sample_indices: Vec<usize> = {
        let step = result.samples.len() / n_samples;
        (0..n_samples).map(|i| i * step).collect()
    };

    let all_survival: Vec<Vec<Vec<f64>>> = (0..n_new)
        .into_par_iter()
        .map(|i| {
            let x_row: Vec<f64> = (0..n_vars).map(|j| x_new[i * n_vars + j]).collect();

            sample_indices
                .iter()
                .map(|&s_idx| {
                    let beta = &result.samples[s_idx];
                    let mut eta = 0.0;
                    for (j, &b) in beta.iter().enumerate() {
                        if j < n_vars {
                            eta += x_row[j] * b;
                        }
                    }
                    let risk = eta.exp();

                    baseline_hazard.iter().map(|&h| (-h * risk).exp()).collect()
                })
                .collect()
        })
        .collect();

    let survival_mean: Vec<Vec<f64>> = (0..n_new)
        .map(|i| {
            (0..n_times)
                .map(|t| all_survival[i].iter().map(|s| s[t]).sum::<f64>() / n_samples as f64)
                .collect()
        })
        .collect();

    let survival_lower: Vec<Vec<f64>> = (0..n_new)
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mut vals: Vec<f64> = all_survival[i].iter().map(|s| s[t]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    vals[(n_samples as f64 * 0.025) as usize]
                })
                .collect()
        })
        .collect();

    let survival_upper: Vec<Vec<f64>> = (0..n_new)
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mut vals: Vec<f64> = all_survival[i].iter().map(|s| s[t]).collect();
                    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    vals[(n_samples as f64 * 0.975) as usize]
                })
                .collect()
        })
        .collect();

    Ok((survival_mean, survival_lower, survival_upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_prior_normal() {
        let beta = vec![0.5, -0.3, 0.2];
        let lp = log_prior(&beta, &PriorType::Normal, 2.5);
        assert!(lp < 0.0);
    }

    #[test]
    fn test_bayesian_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];

        let config =
            BayesianCoxConfig::new(PriorType::Normal, 2.5, 100, 50, 2, 0.8, Some(42)).unwrap();

        let result = bayesian_cox(x, 4, 2, time, status, &config).unwrap();
        assert_eq!(result.posterior_mean.len(), 2);
        assert_eq!(result.hazard_ratio_mean.len(), 2);
    }
}
