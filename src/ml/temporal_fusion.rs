#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct TFTConfig {
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_heads: usize,
    #[pyo3(get, set)]
    pub num_encoder_layers: usize,
    #[pyo3(get, set)]
    pub num_decoder_layers: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub quantiles: Vec<f64>,
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
impl TFTConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dim=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout_rate=0.1,
        num_time_bins=20,
        quantiles=None,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        dropout_rate: f64,
        num_time_bins: usize,
        quantiles: Option<Vec<f64>>,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_dim must be divisible by num_heads",
            ));
        }
        Ok(Self {
            hidden_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout_rate,
            num_time_bins,
            quantiles: quantiles.unwrap_or_else(|| vec![0.1, 0.5, 0.9]),
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

#[allow(dead_code)]
fn glu(x: &[f64], weights: &[f64]) -> Vec<f64> {
    let half = x.len() / 2;
    x.iter()
        .take(half)
        .zip(x.iter().skip(half))
        .zip(weights.iter())
        .map(|((&a, &b), &w)| a * (1.0 / (1.0 + (-b * w).exp())))
        .collect()
}

fn grn(
    input: &[f64],
    context: Option<&[f64]>,
    weights1: &[Vec<f64>],
    weights2: &[Vec<f64>],
    biases: &[f64],
) -> Vec<f64> {
    let _hidden_dim = weights1.len();

    let hidden: Vec<f64> = weights1
        .iter()
        .zip(biases.iter())
        .map(|(w, &b)| {
            let mut sum: f64 = input.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
            if let Some(ctx) = context {
                sum += ctx
                    .iter()
                    .zip(w.iter())
                    .map(|(&c, &wi)| c * wi)
                    .sum::<f64>();
            }
            (sum + b).max(0.0)
        })
        .collect();

    let output: Vec<f64> = weights2
        .iter()
        .map(|w| hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum())
        .collect();

    output
}

fn temporal_self_attention(
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    num_heads: usize,
) -> Vec<Vec<f64>> {
    let seq_len = queries.len();
    let d_model = queries[0].len();
    let d_head = d_model / num_heads;

    let mut outputs = vec![vec![0.0; d_model]; seq_len];

    for h in 0..num_heads {
        let start = h * d_head;
        let end = start + d_head;

        for t in 0..seq_len {
            let q: Vec<f64> = queries[t][start..end].to_vec();

            let scores: Vec<f64> = (0..=t)
                .map(|s| {
                    let k: Vec<f64> = keys[s][start..end].to_vec();
                    q.iter()
                        .zip(k.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f64>()
                        / (d_head as f64).sqrt()
                })
                .collect();

            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();
            let attention: Vec<f64> = exp_scores.iter().map(|&e| e / sum_exp).collect();

            for (s, &att) in attention.iter().enumerate() {
                for (j, &v) in values[s][start..end].iter().enumerate() {
                    outputs[t][start + j] += att * v;
                }
            }
        }
    }

    outputs
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TemporalFusionTransformer {
    static_encoder_weights: Vec<Vec<f64>>,
    static_encoder_biases: Vec<f64>,
    temporal_encoder_weights: Vec<Vec<f64>>,
    temporal_encoder_biases: Vec<f64>,
    grn_weights1: Vec<Vec<f64>>,
    grn_weights2: Vec<Vec<f64>>,
    grn_biases: Vec<f64>,
    #[allow(dead_code)]
    attention_weights: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    output_biases: Vec<f64>,
    time_bins: Vec<f64>,
    config: TFTConfig,
    n_static_features: usize,
    n_temporal_features: usize,
}

#[pymethods]
impl TemporalFusionTransformer {
    fn predict_survival(
        &self,
        static_features: Vec<Vec<f64>>,
        temporal_features: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if static_features.is_empty() {
            return Ok(Vec::new());
        }

        let n_samples = static_features.len();

        let survival: Vec<Vec<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let static_encoded: Vec<f64> = self
                    .static_encoder_weights
                    .iter()
                    .zip(self.static_encoder_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = static_features[i]
                            .iter()
                            .zip(w.iter())
                            .map(|(&x, &wi)| x * wi)
                            .sum();
                        (sum + b).max(0.0)
                    })
                    .collect();

                let seq_len = temporal_features.get(i).map(|t| t.len()).unwrap_or(1);
                let mut temporal_encoded: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

                for t in 0..seq_len {
                    let temporal_input = temporal_features
                        .get(i)
                        .and_then(|tf| tf.get(t))
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.n_temporal_features]);

                    let encoded: Vec<f64> = self
                        .temporal_encoder_weights
                        .iter()
                        .zip(self.temporal_encoder_biases.iter())
                        .map(|(w, &b)| {
                            let sum: f64 = temporal_input
                                .iter()
                                .zip(w.iter())
                                .map(|(&x, &wi)| x * wi)
                                .sum();
                            (sum + b).max(0.0)
                        })
                        .collect();
                    temporal_encoded.push(encoded);
                }

                let enriched: Vec<Vec<f64>> = temporal_encoded
                    .iter()
                    .map(|te| {
                        grn(
                            te,
                            Some(&static_encoded),
                            &self.grn_weights1,
                            &self.grn_weights2,
                            &self.grn_biases,
                        )
                    })
                    .collect();

                let attended =
                    temporal_self_attention(&enriched, &enriched, &enriched, self.config.num_heads);

                let final_repr = attended.last().unwrap_or(&static_encoded);

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = final_repr
                            .iter()
                            .zip(w.iter())
                            .map(|(&h, &wi)| h * wi)
                            .sum();
                        sum + b
                    })
                    .collect();

                let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_logits: Vec<f64> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
                let sum_exp: f64 = exp_logits.iter().sum();
                let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum_exp).collect();

                let mut surv = vec![0.0; probs.len()];
                let mut cumsum = 0.0;
                for j in (0..probs.len()).rev() {
                    cumsum += probs[j];
                    surv[j] = cumsum.min(1.0);
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn predict_quantiles(
        &self,
        static_features: Vec<Vec<f64>>,
        temporal_features: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let survival = self.predict_survival(static_features, temporal_features)?;

        let quantile_predictions: Vec<Vec<Vec<f64>>> = survival
            .iter()
            .map(|s| {
                self.config
                    .quantiles
                    .iter()
                    .map(|&q| {
                        s.iter()
                            .map(|&si| si * q + (1.0 - si) * (1.0 - q))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Ok(quantile_predictions)
    }

    fn get_attention_weights(
        &self,
        static_features: Vec<f64>,
        temporal_features: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let seq_len = temporal_features.len();
        let mut attention_weights = vec![vec![0.0; seq_len]; seq_len];

        let _static_encoded: Vec<f64> = self
            .static_encoder_weights
            .iter()
            .zip(self.static_encoder_biases.iter())
            .map(|(w, &b)| {
                let sum: f64 = static_features
                    .iter()
                    .zip(w.iter())
                    .map(|(&x, &wi)| x * wi)
                    .sum();
                (sum + b).max(0.0)
            })
            .collect();

        let temporal_encoded: Vec<Vec<f64>> = temporal_features
            .iter()
            .map(|tf| {
                self.temporal_encoder_weights
                    .iter()
                    .zip(self.temporal_encoder_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = tf.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum();
                        (sum + b).max(0.0)
                    })
                    .collect()
            })
            .collect();

        let d_head = self.config.hidden_dim / self.config.num_heads;

        for t in 0..seq_len {
            let q: Vec<f64> = temporal_encoded[t][..d_head].to_vec();

            let scores: Vec<f64> = (0..=t)
                .map(|s| {
                    let k: Vec<f64> = temporal_encoded[s][..d_head].to_vec();
                    q.iter()
                        .zip(k.iter())
                        .map(|(&qi, &ki)| qi * ki)
                        .sum::<f64>()
                        / (d_head as f64).sqrt()
                })
                .collect();

            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            for (s, &e) in exp_scores.iter().enumerate() {
                attention_weights[t][s] = e / sum_exp;
            }
        }

        Ok(attention_weights)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TemporalFusionTransformer(static={}, temporal={}, hidden={})",
            self.n_static_features, self.n_temporal_features, self.config.hidden_dim
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    static_features,
    temporal_features,
    time,
    event,
    config=None
))]
pub fn fit_temporal_fusion_transformer(
    static_features: Vec<Vec<f64>>,
    temporal_features: Vec<Vec<Vec<f64>>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<TFTConfig>,
) -> PyResult<TemporalFusionTransformer> {
    let config = config.unwrap_or_else(|| {
        TFTConfig::new(64, 4, 2, 2, 0.1, 20, None, 0.001, 64, 100, None).unwrap()
    });

    let n = static_features.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_static = static_features[0].len();
    let n_temporal = temporal_features
        .first()
        .and_then(|t| t.first())
        .map(|f| f.len())
        .unwrap_or(1);

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let static_encoder_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| (0..n_static).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let static_encoder_biases: Vec<f64> = (0..config.hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let temporal_encoder_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| (0..n_temporal).map(|_| rng.f64() * 0.1 - 0.05).collect())
        .collect();
    let temporal_encoder_biases: Vec<f64> = (0..config.hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let grn_weights1: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let grn_weights2: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let grn_biases: Vec<f64> = (0..config.hidden_dim)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let attention_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();

    let output_weights: Vec<Vec<f64>> = (0..config.num_time_bins)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let output_biases: Vec<f64> = (0..config.num_time_bins)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();

    let min_time = time.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let time_bins: Vec<f64> = (0..=config.num_time_bins)
        .map(|i| min_time + (max_time - min_time) * i as f64 / config.num_time_bins as f64)
        .collect();

    Ok(TemporalFusionTransformer {
        static_encoder_weights,
        static_encoder_biases,
        temporal_encoder_weights,
        temporal_encoder_biases,
        grn_weights1,
        grn_weights2,
        grn_biases,
        attention_weights,
        output_weights,
        output_biases,
        time_bins,
        config,
        n_static_features: n_static,
        n_temporal_features: n_temporal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = TFTConfig::new(64, 5, 2, 2, 0.1, 20, None, 0.001, 64, 100, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_grn() {
        let input = vec![1.0, 2.0];
        let weights1 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let weights2 = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let biases = vec![0.0, 0.0];
        let output = grn(&input, None, &weights1, &weights2, &biases);
        assert_eq!(output.len(), 2);
    }
}
