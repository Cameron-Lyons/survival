#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvivalTransformerConfig {
    #[pyo3(get, set)]
    pub d_model: usize,
    #[pyo3(get, set)]
    pub n_heads: usize,
    #[pyo3(get, set)]
    pub n_layers: usize,
    #[pyo3(get, set)]
    pub d_ff: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub max_time: f64,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl SurvivalTransformerConfig {
    #[new]
    #[pyo3(signature = (
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout_rate=0.1,
        max_time=100.0,
        num_time_bins=20,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        dropout_rate: f64,
        max_time: f64,
        num_time_bins: usize,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !d_model.is_multiple_of(n_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "d_model must be divisible by n_heads",
            ));
        }
        if n_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_layers must be at least 1",
            ));
        }
        Ok(Self {
            d_model,
            n_heads,
            n_layers,
            d_ff,
            dropout_rate,
            max_time,
            num_time_bins,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

#[allow(dead_code)]
fn positional_encoding(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
    (0..seq_len)
        .map(|pos| {
            (0..d_model)
                .map(|i| {
                    let angle =
                        pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / d_model as f64);
                    if i.is_multiple_of(2) {
                        angle.sin()
                    } else {
                        angle.cos()
                    }
                })
                .collect()
        })
        .collect()
}

fn softmax_attention(query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
    let d_k = query.len() as f64;
    let scores: Vec<f64> = keys
        .iter()
        .map(|k| {
            query
                .iter()
                .zip(k.iter())
                .map(|(&q, &ki)| q * ki)
                .sum::<f64>()
                / d_k.sqrt()
        })
        .collect();

    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let attention_weights: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    let d_v = values.first().map(|v| v.len()).unwrap_or(0);
    let mut result = vec![0.0; d_v];
    for (weight, value) in attention_weights.iter().zip(values.iter()) {
        for (j, &v) in value.iter().enumerate() {
            result[j] += weight * v;
        }
    }
    result
}

fn layer_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = (var + eps).sqrt();
    x.iter().map(|&xi| (xi - mean) / std).collect()
}

fn feed_forward(x: &[f64], w1: &[Vec<f64>], b1: &[f64], w2: &[Vec<f64>], b2: &[f64]) -> Vec<f64> {
    let hidden: Vec<f64> = w1
        .iter()
        .zip(b1.iter())
        .map(|(w, &b)| {
            let linear: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| xi * wi).sum();
            (linear + b).max(0.0)
        })
        .collect();

    w2.iter()
        .zip(b2.iter())
        .map(|(w, &b)| {
            let linear: f64 = hidden.iter().zip(w.iter()).map(|(&hi, &wi)| hi * wi).sum();
            linear + b
        })
        .collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvivalTransformerModel {
    embed_weights: Vec<Vec<f64>>,
    ff_w1: Vec<Vec<Vec<f64>>>,
    ff_b1: Vec<Vec<f64>>,
    ff_w2: Vec<Vec<Vec<f64>>>,
    ff_b2: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    time_bins: Vec<f64>,
    config: SurvivalTransformerConfig,
    n_features: usize,
}

#[pymethods]
impl SurvivalTransformerModel {
    fn predict_survival(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let mut hidden: Vec<f64> = self
                    .embed_weights
                    .iter()
                    .map(|w| cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum())
                    .collect();

                for layer in 0..self.config.n_layers {
                    let keys: Vec<Vec<f64>> = vec![hidden.clone()];
                    let values: Vec<Vec<f64>> = vec![hidden.clone()];
                    let attn_out = softmax_attention(&hidden, &keys, &values);

                    hidden = hidden
                        .iter()
                        .zip(attn_out.iter())
                        .map(|(&h, &a)| h + a)
                        .collect();
                    hidden = layer_norm(&hidden, 1e-6);

                    let ff_out = feed_forward(
                        &hidden,
                        &self.ff_w1[layer],
                        &self.ff_b1[layer],
                        &self.ff_w2[layer],
                        &self.ff_b2[layer],
                    );

                    hidden = hidden
                        .iter()
                        .zip(ff_out.iter())
                        .map(|(&h, &f)| h + f)
                        .collect();
                    hidden = layer_norm(&hidden, 1e-6);
                }

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_bias.iter())
                    .map(|(w, &b)| {
                        let linear: f64 = hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                        linear + b
                    })
                    .collect();

                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

                let mut survival = vec![0.0; probs.len()];
                let mut cumsum = 0.0;
                for i in (0..probs.len()).rev() {
                    cumsum += probs[i];
                    survival[i] = cumsum;
                }
                survival
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

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurvivalTransformerModel(n_features={}, d_model={}, n_layers={})",
            self.n_features, self.config.d_model, self.config.n_layers
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
pub fn fit_survival_transformer(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<SurvivalTransformerConfig>,
) -> PyResult<SurvivalTransformerModel> {
    let config = config.unwrap_or_else(|| {
        SurvivalTransformerConfig::new(64, 4, 2, 128, 0.1, 100.0, 20, 0.001, 64, 100, None).unwrap()
    });

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let embed_weights: Vec<Vec<f64>> = (0..config.d_model)
        .map(|_| (0..n_features).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();

    let mut ff_w1 = Vec::new();
    let mut ff_b1 = Vec::new();
    let mut ff_w2 = Vec::new();
    let mut ff_b2 = Vec::new();

    for _ in 0..config.n_layers {
        ff_w1.push(
            (0..config.d_ff)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );
        ff_b1.push((0..config.d_ff).map(|_| rng.f64() * 0.1 - 0.05).collect());
        ff_w2.push(
            (0..config.d_model)
                .map(|_| (0..config.d_ff).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
        );
        ff_b2.push(
            (0..config.d_model)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect(),
        );
    }

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| {
            (0..config.d_model)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let output_bias: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(SurvivalTransformerModel {
        embed_weights,
        ff_w1,
        ff_b1,
        ff_w2,
        ff_b2,
        output_weights,
        output_bias,
        time_bins,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result =
            SurvivalTransformerConfig::new(64, 5, 2, 128, 0.1, 100.0, 20, 0.001, 64, 100, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let normed = layer_norm(&x, 1e-6);
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-6);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = positional_encoding(10, 8);
        assert_eq!(pe.len(), 10);
        assert_eq!(pe[0].len(), 8);
    }
}
