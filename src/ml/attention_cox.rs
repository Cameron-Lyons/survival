#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct AttentionCoxConfig {
    #[pyo3(get, set)]
    pub d_model: usize,
    #[pyo3(get, set)]
    pub n_heads: usize,
    #[pyo3(get, set)]
    pub n_layers: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
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
impl AttentionCoxConfig {
    #[new]
    #[pyo3(signature = (
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        dropout_rate: f64,
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
        Ok(Self {
            d_model,
            n_heads,
            n_layers,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

fn scaled_dot_product_attention(query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>]) -> Vec<f64> {
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

fn multi_head_attention(
    query: &[f64],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    n_heads: usize,
    w_q: &[Vec<f64>],
    w_k: &[Vec<f64>],
    w_v: &[Vec<f64>],
    w_o: &[Vec<f64>],
) -> Vec<f64> {
    let d_model = query.len();
    let d_head = d_model / n_heads;

    let mut all_heads = Vec::new();

    for h in 0..n_heads {
        let start = h * d_head;
        let end = start + d_head;

        let q_h: Vec<f64> = w_q[start..end]
            .iter()
            .map(|w| query.iter().zip(w.iter()).map(|(&q, &wi)| q * wi).sum())
            .collect();

        let k_h: Vec<Vec<f64>> = keys
            .iter()
            .map(|k| {
                w_k[start..end]
                    .iter()
                    .map(|w| k.iter().zip(w.iter()).map(|(&ki, &wi)| ki * wi).sum())
                    .collect()
            })
            .collect();

        let v_h: Vec<Vec<f64>> = values
            .iter()
            .map(|v| {
                w_v[start..end]
                    .iter()
                    .map(|w| v.iter().zip(w.iter()).map(|(&vi, &wi)| vi * wi).sum())
                    .collect()
            })
            .collect();

        let head_output = scaled_dot_product_attention(&q_h, &k_h, &v_h);
        all_heads.extend(head_output);
    }

    w_o.iter()
        .map(|w| all_heads.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum())
        .collect()
}

fn layer_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std = (var + eps).sqrt();
    x.iter().map(|&xi| (xi - mean) / std).collect()
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AttentionCoxModel {
    embed_weights: Vec<Vec<f64>>,
    w_q: Vec<Vec<Vec<f64>>>,
    w_k: Vec<Vec<Vec<f64>>>,
    w_v: Vec<Vec<Vec<f64>>>,
    w_o: Vec<Vec<Vec<f64>>>,
    ff_w1: Vec<Vec<Vec<f64>>>,
    ff_b1: Vec<Vec<f64>>,
    ff_w2: Vec<Vec<Vec<f64>>>,
    ff_b2: Vec<Vec<f64>>,
    output_weights: Vec<f64>,
    output_bias: f64,
    config: AttentionCoxConfig,
    n_features: usize,
}

#[pymethods]
impl AttentionCoxModel {
    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        if covariates.is_empty() {
            return Ok(Vec::new());
        }

        let risks: Vec<f64> = covariates
            .par_iter()
            .map(|cov| {
                let mut hidden: Vec<f64> = self
                    .embed_weights
                    .iter()
                    .map(|w| cov.iter().zip(w.iter()).map(|(&x, &wi)| x * wi).sum())
                    .collect();

                for layer in 0..self.config.n_layers {
                    let keys = vec![hidden.clone()];
                    let values = vec![hidden.clone()];

                    let attn_out = multi_head_attention(
                        &hidden,
                        &keys,
                        &values,
                        self.config.n_heads,
                        &self.w_q[layer],
                        &self.w_k[layer],
                        &self.w_v[layer],
                        &self.w_o[layer],
                    );

                    hidden = hidden
                        .iter()
                        .zip(attn_out.iter())
                        .map(|(&h, &a)| h + a)
                        .collect();
                    hidden = layer_norm(&hidden, 1e-6);

                    let ff_hidden: Vec<f64> = self.ff_w1[layer]
                        .iter()
                        .zip(self.ff_b1[layer].iter())
                        .map(|(w, &b)| {
                            let sum: f64 =
                                hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                            (sum + b).max(0.0)
                        })
                        .collect();

                    let ff_out: Vec<f64> = self.ff_w2[layer]
                        .iter()
                        .zip(self.ff_b2[layer].iter())
                        .map(|(w, &b)| {
                            let sum: f64 =
                                ff_hidden.iter().zip(w.iter()).map(|(&h, &wi)| h * wi).sum();
                            sum + b
                        })
                        .collect();

                    hidden = hidden
                        .iter()
                        .zip(ff_out.iter())
                        .map(|(&h, &f)| h + f)
                        .collect();
                    hidden = layer_norm(&hidden, 1e-6);
                }

                let log_risk: f64 = hidden
                    .iter()
                    .zip(self.output_weights.iter())
                    .map(|(&h, &w)| h * w)
                    .sum::<f64>()
                    + self.output_bias;

                log_risk.exp()
            })
            .collect();

        Ok(risks)
    }

    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        times: Vec<f64>,
        baseline_hazard: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let risks = self.predict_risk(covariates)?;

        let survival: Vec<Vec<f64>> = risks
            .iter()
            .map(|&risk| {
                times
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let cumhaz: f64 = baseline_hazard.iter().take(i + 1).sum::<f64>() * risk;
                        (-cumhaz).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    fn get_attention_weights(&self, covariates: Vec<f64>, layer: usize) -> PyResult<Vec<Vec<f64>>> {
        if layer >= self.config.n_layers {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "layer index out of bounds",
            ));
        }

        let mut hidden: Vec<f64> = self
            .embed_weights
            .iter()
            .map(|w| {
                covariates
                    .iter()
                    .zip(w.iter())
                    .map(|(&x, &wi)| x * wi)
                    .sum()
            })
            .collect();

        for l in 0..=layer {
            if l == layer {
                let d_head = self.config.d_model / self.config.n_heads;
                let mut attention_weights = Vec::new();

                for h in 0..self.config.n_heads {
                    let start = h * d_head;
                    let end = start + d_head;

                    let q_h: Vec<f64> = self.w_q[l][start..end]
                        .iter()
                        .map(|w| hidden.iter().zip(w.iter()).map(|(&q, &wi)| q * wi).sum())
                        .collect();

                    let k_h: Vec<f64> = self.w_k[l][start..end]
                        .iter()
                        .map(|w| hidden.iter().zip(w.iter()).map(|(&k, &wi)| k * wi).sum())
                        .collect();

                    let score: f64 = q_h
                        .iter()
                        .zip(k_h.iter())
                        .map(|(&q, &k)| q * k)
                        .sum::<f64>()
                        / (d_head as f64).sqrt();

                    attention_weights.push(vec![score.exp()]);
                }

                return Ok(attention_weights);
            }

            let keys = vec![hidden.clone()];
            let values = vec![hidden.clone()];

            let attn_out = multi_head_attention(
                &hidden,
                &keys,
                &values,
                self.config.n_heads,
                &self.w_q[l],
                &self.w_k[l],
                &self.w_v[l],
                &self.w_o[l],
            );

            hidden = hidden
                .iter()
                .zip(attn_out.iter())
                .map(|(&h, &a)| h + a)
                .collect();
            hidden = layer_norm(&hidden, 1e-6);
        }

        Ok(Vec::new())
    }

    fn __repr__(&self) -> String {
        format!(
            "AttentionCoxModel(n_features={}, d_model={}, n_heads={}, n_layers={})",
            self.n_features, self.config.d_model, self.config.n_heads, self.config.n_layers
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
pub fn fit_attention_cox(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<AttentionCoxConfig>,
) -> PyResult<AttentionCoxModel> {
    let config = config
        .unwrap_or_else(|| AttentionCoxConfig::new(64, 4, 2, 0.1, 0.001, 64, 100, None).unwrap());

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

    let mut w_q = Vec::new();
    let mut w_k = Vec::new();
    let mut w_v = Vec::new();
    let mut w_o = Vec::new();
    let mut ff_w1 = Vec::new();
    let mut ff_b1 = Vec::new();
    let mut ff_w2 = Vec::new();
    let mut ff_b2 = Vec::new();

    for _ in 0..config.n_layers {
        w_q.push(
            (0..config.d_model)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );
        w_k.push(
            (0..config.d_model)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );
        w_v.push(
            (0..config.d_model)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );
        w_o.push(
            (0..config.d_model)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );

        let ff_dim = config.d_model * 4;
        ff_w1.push(
            (0..ff_dim)
                .map(|_| {
                    (0..config.d_model)
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect(),
        );
        ff_b1.push((0..ff_dim).map(|_| rng.f64() * 0.1 - 0.05).collect());
        ff_w2.push(
            (0..config.d_model)
                .map(|_| (0..ff_dim).map(|_| rng.f64() * 0.1 - 0.05).collect())
                .collect(),
        );
        ff_b2.push(
            (0..config.d_model)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect(),
        );
    }

    let output_weights: Vec<f64> = (0..config.d_model)
        .map(|_| rng.f64() * 0.1 - 0.05)
        .collect();
    let output_bias = rng.f64() * 0.1 - 0.05;

    Ok(AttentionCoxModel {
        embed_weights,
        w_q,
        w_k,
        w_v,
        w_o,
        ff_w1,
        ff_b1,
        ff_w2,
        ff_b2,
        output_weights,
        output_bias,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = AttentionCoxConfig::new(64, 5, 2, 0.1, 0.001, 64, 100, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let normed = layer_norm(&x, 1e-6);
        let mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
        assert!(mean.abs() < 1e-6);
    }
}
