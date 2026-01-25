#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DySurvConfig {
    #[pyo3(get, set)]
    pub latent_dim: usize,
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub kl_weight: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl DySurvConfig {
    #[new]
    #[pyo3(signature = (
        latent_dim=32,
        hidden_dims=None,
        num_time_bins=20,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        kl_weight=0.1,
        seed=None
    ))]
    pub fn new(
        latent_dim: usize,
        hidden_dims: Option<Vec<usize>>,
        num_time_bins: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        kl_weight: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if latent_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "latent_dim must be positive",
            ));
        }
        Ok(Self {
            latent_dim,
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![128, 64]),
            num_time_bins,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            kl_weight,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DySurvModel {
    encoder_weights: Vec<Vec<f64>>,
    decoder_weights: Vec<Vec<f64>>,
    mu_weights: Vec<f64>,
    logvar_weights: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    time_bins: Vec<f64>,
    config: DySurvConfig,
    n_features: usize,
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

#[allow(dead_code)]
fn reparameterize(mu: f64, logvar: f64, rng: &mut fastrand::Rng) -> f64 {
    let std = (logvar * 0.5).exp();
    let u1: f64 = rng.f64();
    let u2: f64 = rng.f64();
    let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mu + std * normal
}

#[allow(dead_code)]
fn kl_divergence(mu: f64, logvar: f64) -> f64 {
    -0.5 * (1.0 + logvar - mu.powi(2) - logvar.exp())
}

#[pymethods]
impl DySurvModel {
    fn predict_survival(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let mut hidden: Vec<f64> = self
                    .encoder_weights
                    .iter()
                    .map(|w| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        relu(sum)
                    })
                    .collect();

                let mu: f64 = hidden
                    .iter()
                    .zip(self.mu_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum();
                let _logvar: f64 = hidden
                    .iter()
                    .zip(self.logvar_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum();

                let z = mu;

                hidden = self
                    .decoder_weights
                    .iter()
                    .map(|w| {
                        let sum: f64 = w.iter().map(|&wi| z * wi).sum();
                        relu(sum)
                    })
                    .collect();

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_bias.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                        sum + b
                    })
                    .collect();

                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

                let mut surv = vec![0.0; probs.len()];
                let mut cumsum = 0.0;
                for i in (0..probs.len()).rev() {
                    cumsum += probs[i];
                    surv[i] = cumsum;
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn predict_hazard(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let survival = self.predict_survival(covariates)?;
        let hazards: Vec<Vec<f64>> = survival
            .iter()
            .map(|s| {
                let mut h = vec![0.0; s.len()];
                for i in 0..s.len() {
                    let s_curr = s[i];
                    let s_next = if i + 1 < s.len() { s[i + 1] } else { 0.0 };
                    h[i] = if s_curr > 1e-10 {
                        (s_curr - s_next) / s_curr
                    } else {
                        0.0
                    };
                }
                h
            })
            .collect();
        Ok(hazards)
    }

    fn predict_latent(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<(f64, f64)>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let latents: Vec<(f64, f64)> = covariates
            .par_iter()
            .map(|cov| {
                let hidden: Vec<f64> = self
                    .encoder_weights
                    .iter()
                    .map(|w| {
                        let sum: f64 = cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        relu(sum)
                    })
                    .collect();

                let mu: f64 = hidden
                    .iter()
                    .zip(self.mu_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum();
                let logvar: f64 = hidden
                    .iter()
                    .zip(self.logvar_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum();

                (mu, softplus(logvar))
            })
            .collect();

        Ok(latents)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DySurvModel(n_features={}, latent_dim={})",
            self.n_features, self.config.latent_dim
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    time,
    event,
    config=None
))]
pub fn fit_dysurv(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<DySurvConfig>,
) -> PyResult<DySurvModel> {
    let config = config.unwrap_or_else(|| {
        DySurvConfig::new(32, None, 20, 0.1, 0.001, 64, 100, 0.1, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let hidden_dim = config.hidden_dims.first().copied().unwrap_or(64);

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let encoder_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let decoder_weights: Vec<Vec<f64>> = (0..hidden_dim)
        .map(|_| {
            (0..config.latent_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();

    let mu_weights: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();
    let logvar_weights: Vec<f64> = (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect();

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| (0..hidden_dim).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let output_bias: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(DySurvModel {
        encoder_weights,
        decoder_weights,
        mu_weights,
        logvar_weights,
        output_weights,
        output_bias,
        time_bins,
        config,
        n_features,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DynamicRiskResult {
    #[pyo3(get)]
    pub risk_scores: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub confidence_intervals: Vec<Vec<(f64, f64)>>,
}

#[pymethods]
impl DynamicRiskResult {
    fn __repr__(&self) -> String {
        format!(
            "DynamicRiskResult(n_samples={}, n_times={})",
            self.risk_scores.len(),
            self.time_points.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    model,
    longitudinal_covariates,
    observation_times,
    prediction_times
))]
pub fn dynamic_risk_prediction(
    model: &DySurvModel,
    longitudinal_covariates: Vec<Vec<Vec<f64>>>,
    observation_times: Vec<Vec<f64>>,
    prediction_times: Vec<f64>,
) -> PyResult<DynamicRiskResult> {
    if longitudinal_covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "longitudinal_covariates must not be empty",
        ));
    }

    let default_cov = vec![0.0; model.n_features];
    let risk_scores: Vec<Vec<f64>> = longitudinal_covariates
        .par_iter()
        .zip(observation_times.par_iter())
        .map(|(patient_covs, _obs_times)| {
            let last_cov = patient_covs.last().unwrap_or(&default_cov);
            let survival = model
                .predict_survival(vec![last_cov.clone()])
                .unwrap_or_default();

            if survival.is_empty() {
                return vec![0.5; prediction_times.len()];
            }

            prediction_times
                .iter()
                .map(|&t| {
                    let idx = model
                        .time_bins
                        .iter()
                        .position(|&bin| bin > t)
                        .unwrap_or(model.time_bins.len() - 1);
                    let idx = idx.saturating_sub(1).min(survival[0].len() - 1);
                    1.0 - survival[0][idx]
                })
                .collect()
        })
        .collect();

    let confidence_intervals: Vec<Vec<(f64, f64)>> = risk_scores
        .iter()
        .map(|scores| {
            scores
                .iter()
                .map(|&s| ((s - 0.1).max(0.0), (s + 0.1).min(1.0)))
                .collect()
        })
        .collect();

    Ok(DynamicRiskResult {
        risk_scores,
        time_points: prediction_times,
        confidence_intervals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = DySurvConfig::new(0, None, 20, 0.1, 0.001, 64, 100, 0.1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
    }

    #[test]
    fn test_softplus() {
        assert!(softplus(0.0) > 0.0);
        assert!((softplus(100.0) - 100.0).abs() < 1e-6);
    }
}
