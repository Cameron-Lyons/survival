#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use crate::utilities::statistical::{normal_cdf, sample_normal};
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum BayesianDistribution {
    Weibull,
    LogNormal,
    LogLogistic,
    Exponential,
}

#[pymethods]
impl BayesianDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "weibull" => Ok(BayesianDistribution::Weibull),
            "lognormal" | "log_normal" => Ok(BayesianDistribution::LogNormal),
            "loglogistic" | "log_logistic" => Ok(BayesianDistribution::LogLogistic),
            "exponential" | "exp" => Ok(BayesianDistribution::Exponential),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown distribution",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BayesianParametricConfig {
    #[pyo3(get, set)]
    pub distribution: BayesianDistribution,
    #[pyo3(get, set)]
    pub beta_prior_scale: f64,
    #[pyo3(get, set)]
    pub shape_prior_mean: f64,
    #[pyo3(get, set)]
    pub shape_prior_sd: f64,
    #[pyo3(get, set)]
    pub n_samples: usize,
    #[pyo3(get, set)]
    pub n_warmup: usize,
    #[pyo3(get, set)]
    pub n_chains: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl BayesianParametricConfig {
    #[new]
    #[pyo3(signature = (distribution=BayesianDistribution::Weibull, beta_prior_scale=2.5, shape_prior_mean=1.0, shape_prior_sd=1.0, n_samples=2000, n_warmup=1000, n_chains=4, seed=None))]
    pub fn new(
        distribution: BayesianDistribution,
        beta_prior_scale: f64,
        shape_prior_mean: f64,
        shape_prior_sd: f64,
        n_samples: usize,
        n_warmup: usize,
        n_chains: usize,
        seed: Option<u64>,
    ) -> Self {
        BayesianParametricConfig {
            distribution,
            beta_prior_scale,
            shape_prior_mean,
            shape_prior_sd,
            n_samples,
            n_warmup,
            n_chains,
            seed,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BayesianParametricResult {
    #[pyo3(get)]
    pub beta_mean: Vec<f64>,
    #[pyo3(get)]
    pub beta_sd: Vec<f64>,
    #[pyo3(get)]
    pub beta_lower: Vec<f64>,
    #[pyo3(get)]
    pub beta_upper: Vec<f64>,
    #[pyo3(get)]
    pub shape_mean: f64,
    #[pyo3(get)]
    pub shape_sd: f64,
    #[pyo3(get)]
    pub shape_lower: f64,
    #[pyo3(get)]
    pub shape_upper: f64,
    #[pyo3(get)]
    pub acceleration_factor_mean: Vec<f64>,
    #[pyo3(get)]
    pub acceleration_factor_lower: Vec<f64>,
    #[pyo3(get)]
    pub acceleration_factor_upper: Vec<f64>,
    #[pyo3(get)]
    pub beta_samples: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub shape_samples: Vec<f64>,
    #[pyo3(get)]
    pub log_posterior: Vec<f64>,
    #[pyo3(get)]
    pub dic: f64,
    #[pyo3(get)]
    pub waic: f64,
}

fn weibull_log_lik(time: f64, status: i32, scale: f64, shape: f64) -> f64 {
    if time <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let log_t = time.ln();
    let log_scale = scale.ln();
    let z = (log_t - log_scale) * shape;

    if status == 1 {
        shape.ln() - log_scale + (shape - 1.0) * (log_t - log_scale) - z.exp()
    } else {
        -z.exp()
    }
}

fn lognormal_log_lik(time: f64, status: i32, mu: f64, sigma: f64) -> f64 {
    if time <= 0.0 || sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let log_t = time.ln();
    let z = (log_t - mu) / sigma;

    if status == 1 {
        -0.5 * z * z - log_t.ln() - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    } else {
        (1.0 - normal_cdf(z)).max(1e-300).ln()
    }
}

fn loglogistic_log_lik(time: f64, status: i32, scale: f64, shape: f64) -> f64 {
    if time <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let z = (time / scale).powf(shape);

    if status == 1 {
        shape.ln() - scale.ln() + (shape - 1.0) * (time / scale).ln() - 2.0 * (1.0 + z).ln()
    } else {
        -(1.0 + z).ln()
    }
}

fn exponential_log_lik(time: f64, status: i32, rate: f64) -> f64 {
    if time <= 0.0 || rate <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if status == 1 {
        rate.ln() - rate * time
    } else {
        -rate * time
    }
}

fn log_likelihood(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    shape: f64,
    dist: &BayesianDistribution,
) -> f64 {
    (0..n)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[i * p + j] * beta[j];
            }
            let scale = eta.exp();

            match dist {
                BayesianDistribution::Weibull => weibull_log_lik(time[i], status[i], scale, shape),
                BayesianDistribution::LogNormal => {
                    lognormal_log_lik(time[i], status[i], eta, shape)
                }
                BayesianDistribution::LogLogistic => {
                    loglogistic_log_lik(time[i], status[i], scale, shape)
                }
                BayesianDistribution::Exponential => {
                    exponential_log_lik(time[i], status[i], 1.0 / scale)
                }
            }
        })
        .sum()
}

fn log_prior(beta: &[f64], shape: f64, config: &BayesianParametricConfig) -> f64 {
    let beta_prior: f64 = -0.5
        * beta
            .iter()
            .map(|&b| (b / config.beta_prior_scale).powi(2))
            .sum::<f64>();

    let shape_prior = if shape > 0.0 {
        let z = (shape.ln() - config.shape_prior_mean.ln()) / config.shape_prior_sd;
        -0.5 * z * z - shape.ln()
    } else {
        f64::NEG_INFINITY
    };

    beta_prior + shape_prior
}

fn metropolis_step(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n: usize,
    p: usize,
    beta: &[f64],
    shape: f64,
    config: &BayesianParametricConfig,
    proposal_sd: &[f64],
    shape_proposal_sd: f64,
    rng: &mut fastrand::Rng,
) -> (Vec<f64>, f64, f64, bool) {
    let mut new_beta = beta.to_vec();
    for j in 0..p {
        new_beta[j] += proposal_sd[j] * sample_normal(rng);
    }
    let new_shape = (shape + shape_proposal_sd * sample_normal(rng)).max(0.01);

    let current_log_post = log_likelihood(time, status, x, n, p, beta, shape, &config.distribution)
        + log_prior(beta, shape, config);

    let new_log_post = log_likelihood(
        time,
        status,
        x,
        n,
        p,
        &new_beta,
        new_shape,
        &config.distribution,
    ) + log_prior(&new_beta, new_shape, config);

    let log_accept = new_log_post - current_log_post;

    if rng.f64().ln() < log_accept {
        (new_beta, new_shape, new_log_post, true)
    } else {
        (beta.to_vec(), shape, current_log_post, false)
    }
}

fn run_chain(
    time: &[f64],
    status: &[i32],
    x: &[f64],
    n: usize,
    p: usize,
    config: &BayesianParametricConfig,
    chain_id: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let seed = config.seed.unwrap_or(42) + chain_id as u64 * 1000;
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut beta = vec![0.0; p];
    let mut shape = config.shape_prior_mean;

    let mut proposal_sd = vec![0.1; p];
    let mut shape_proposal_sd = 0.1;

    let mut beta_samples: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut shape_samples: Vec<f64> = Vec::with_capacity(config.n_samples);
    let mut log_posteriors: Vec<f64> = Vec::with_capacity(config.n_samples);

    let mut n_accepted = 0;

    for i in 0..(config.n_warmup + config.n_samples) {
        let (new_beta, new_shape, log_post, accepted) = metropolis_step(
            time,
            status,
            x,
            n,
            p,
            &beta,
            shape,
            config,
            &proposal_sd,
            shape_proposal_sd,
            &mut rng,
        );

        beta = new_beta;
        shape = new_shape;

        if accepted {
            n_accepted += 1;
        }

        if i < config.n_warmup && (i + 1) % 100 == 0 {
            let accept_rate = n_accepted as f64 / 100.0;
            let adjustment = if accept_rate < 0.2 {
                0.8
            } else if accept_rate > 0.4 {
                1.2
            } else {
                1.0
            };

            for sd in &mut proposal_sd {
                *sd *= adjustment;
            }
            shape_proposal_sd *= adjustment;
            n_accepted = 0;
        }

        if i >= config.n_warmup {
            beta_samples.push(beta.clone());
            shape_samples.push(shape);
            log_posteriors.push(log_post);
        }
    }

    (beta_samples, shape_samples, log_posteriors)
}

#[pyfunction]
#[pyo3(signature = (time, status, x, n_obs, n_vars, config))]
pub fn bayesian_parametric(
    time: Vec<f64>,
    status: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    config: &BayesianParametricConfig,
) -> PyResult<BayesianParametricResult> {
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let chains: Vec<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> = (0..config.n_chains)
        .into_par_iter()
        .map(|chain_id| run_chain(&time, &status, &x, n_obs, n_vars, config, chain_id))
        .collect();

    let all_beta: Vec<Vec<f64>> = chains.iter().flat_map(|(b, _, _)| b.clone()).collect();
    let all_shape: Vec<f64> = chains.iter().flat_map(|(_, s, _)| s.clone()).collect();
    let all_log_post: Vec<f64> = chains.iter().flat_map(|(_, _, lp)| lp.clone()).collect();

    let n_total = all_beta.len();

    let beta_mean: Vec<f64> = (0..n_vars)
        .map(|j| all_beta.iter().map(|b| b[j]).sum::<f64>() / n_total as f64)
        .collect();

    let beta_sd: Vec<f64> = (0..n_vars)
        .map(|j| {
            let mean = beta_mean[j];
            let var =
                all_beta.iter().map(|b| (b[j] - mean).powi(2)).sum::<f64>() / (n_total - 1) as f64;
            var.sqrt()
        })
        .collect();

    let beta_lower: Vec<f64> = (0..n_vars)
        .map(|j| {
            let mut vals: Vec<f64> = all_beta.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_total as f64 * 0.025) as usize]
        })
        .collect();

    let beta_upper: Vec<f64> = (0..n_vars)
        .map(|j| {
            let mut vals: Vec<f64> = all_beta.iter().map(|b| b[j]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals[(n_total as f64 * 0.975) as usize]
        })
        .collect();

    let shape_mean = all_shape.iter().sum::<f64>() / n_total as f64;
    let shape_sd = (all_shape
        .iter()
        .map(|&s| (s - shape_mean).powi(2))
        .sum::<f64>()
        / (n_total - 1) as f64)
        .sqrt();

    let mut sorted_shape = all_shape.clone();
    sorted_shape.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let shape_lower = sorted_shape[(n_total as f64 * 0.025) as usize];
    let shape_upper = sorted_shape[(n_total as f64 * 0.975) as usize];

    let acceleration_factor_mean: Vec<f64> = beta_mean.iter().map(|&b| b.exp()).collect();
    let acceleration_factor_lower: Vec<f64> = beta_lower.iter().map(|&b| b.exp()).collect();
    let acceleration_factor_upper: Vec<f64> = beta_upper.iter().map(|&b| b.exp()).collect();

    let mean_log_post = all_log_post.iter().sum::<f64>() / n_total as f64;
    let var_log_post = all_log_post
        .iter()
        .map(|&lp| (lp - mean_log_post).powi(2))
        .sum::<f64>()
        / n_total as f64;
    let waic = -2.0 * mean_log_post + 2.0 * var_log_post;
    let dic = -2.0 * mean_log_post + var_log_post;

    Ok(BayesianParametricResult {
        beta_mean,
        beta_sd,
        beta_lower,
        beta_upper,
        shape_mean,
        shape_sd,
        shape_lower,
        shape_upper,
        acceleration_factor_mean,
        acceleration_factor_lower,
        acceleration_factor_upper,
        beta_samples: all_beta,
        shape_samples: all_shape,
        log_posterior: all_log_post,
        dic,
        waic,
    })
}

#[pyfunction]
pub fn bayesian_parametric_predict(
    result: &BayesianParametricResult,
    x_new: Vec<f64>,
    n_new: usize,
    n_vars: usize,
    time_points: Vec<f64>,
    distribution: &BayesianDistribution,
) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let n_times = time_points.len();
    let n_samples = result.beta_samples.len().min(500);

    let sample_indices: Vec<usize> = {
        let step = result.beta_samples.len() / n_samples;
        (0..n_samples).map(|i| i * step).collect()
    };

    let compute_survival = |t: f64, scale: f64, shape: f64| -> f64 {
        match distribution {
            BayesianDistribution::Weibull => (-(t / scale).powf(shape)).exp(),
            BayesianDistribution::LogNormal => {
                let z = (t.ln() - scale.ln()) / shape;
                1.0 - normal_cdf(z)
            }
            BayesianDistribution::LogLogistic => 1.0 / (1.0 + (t / scale).powf(shape)),
            BayesianDistribution::Exponential => (-t / scale).exp(),
        }
    };

    let all_survival: Vec<Vec<Vec<f64>>> = (0..n_new)
        .into_par_iter()
        .map(|i| {
            let x_row: Vec<f64> = (0..n_vars).map(|j| x_new[i * n_vars + j]).collect();

            sample_indices
                .iter()
                .map(|&s_idx| {
                    let beta = &result.beta_samples[s_idx];
                    let shape = result.shape_samples[s_idx];

                    let mut eta = 0.0;
                    for (j, &b) in beta.iter().enumerate() {
                        if j < n_vars {
                            eta += x_row[j] * b;
                        }
                    }
                    let scale = eta.exp();

                    time_points
                        .iter()
                        .map(|&t| compute_survival(t, scale, shape))
                        .collect()
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
    fn test_weibull_log_lik() {
        let ll = weibull_log_lik(5.0, 1, 3.0, 1.5);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_bayesian_parametric_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5];

        let config = BayesianParametricConfig::new(
            BayesianDistribution::Weibull,
            2.5,
            1.0,
            1.0,
            100,
            50,
            2,
            Some(42),
        );

        let result = bayesian_parametric(time, status, x, 5, 1, &config).unwrap();
        assert_eq!(result.beta_mean.len(), 1);
        assert!(result.shape_mean > 0.0);
    }
}
