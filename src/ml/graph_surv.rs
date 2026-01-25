#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct GraphSurvConfig {
    #[pyo3(get, set)]
    pub hidden_dim: usize,
    #[pyo3(get, set)]
    pub num_layers: usize,
    #[pyo3(get, set)]
    pub num_heads: usize,
    #[pyo3(get, set)]
    pub num_time_bins: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub aggregation: String,
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
impl GraphSurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        num_time_bins=20,
        dropout_rate=0.1,
        aggregation=None,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        seed=None
    ))]
    pub fn new(
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        num_time_bins: usize,
        dropout_rate: f64,
        aggregation: Option<String>,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let aggregation = aggregation.unwrap_or_else(|| "mean".to_string());
        if !["mean", "max", "sum", "attention"].contains(&aggregation.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "aggregation must be one of: mean, max, sum, attention",
            ));
        }
        Ok(Self {
            hidden_dim,
            num_layers,
            num_heads,
            num_time_bins,
            dropout_rate,
            aggregation,
            learning_rate,
            batch_size,
            n_epochs,
            seed,
        })
    }
}

fn message_passing(
    node_features: &[Vec<f64>],
    adjacency: &[Vec<f64>],
    weights: &[Vec<f64>],
    biases: &[f64],
) -> Vec<Vec<f64>> {
    let n_nodes = node_features.len();
    let hidden_dim = weights.len();

    let mut output = vec![vec![0.0; hidden_dim]; n_nodes];

    for i in 0..n_nodes {
        let mut aggregated = vec![0.0; node_features[0].len()];
        let mut neighbor_count = 0.0;

        for j in 0..n_nodes {
            if adjacency[i][j] > 0.0 {
                for (k, &f) in node_features[j].iter().enumerate() {
                    aggregated[k] += adjacency[i][j] * f;
                }
                neighbor_count += adjacency[i][j];
            }
        }

        if neighbor_count > 0.0 {
            for a in &mut aggregated {
                *a /= neighbor_count;
            }
        }

        for (a, &self_f) in aggregated.iter_mut().zip(node_features[i].iter()) {
            *a += self_f;
        }

        for (h, (w, &b)) in output[i].iter_mut().zip(weights.iter().zip(biases.iter())) {
            let sum: f64 = aggregated
                .iter()
                .zip(w.iter())
                .map(|(&a, &wi)| a * wi)
                .sum();
            *h = (sum + b).max(0.0);
        }
    }

    output
}

fn graph_attention(
    node_features: &[Vec<f64>],
    adjacency: &[Vec<f64>],
    query_weights: &[Vec<f64>],
    key_weights: &[Vec<f64>],
    value_weights: &[Vec<f64>],
    num_heads: usize,
) -> Vec<Vec<f64>> {
    let n_nodes = node_features.len();
    let d_model = node_features[0].len();
    let d_head = d_model / num_heads.max(1);

    let mut output = vec![vec![0.0; d_model]; n_nodes];

    for i in 0..n_nodes {
        let query: Vec<f64> = query_weights
            .iter()
            .map(|w| {
                node_features[i]
                    .iter()
                    .zip(w.iter())
                    .map(|(&f, &wi)| f * wi)
                    .sum()
            })
            .collect();

        let mut attention_sum = vec![0.0; d_model];
        let mut attention_weights = Vec::new();

        for j in 0..n_nodes {
            if adjacency[i][j] > 0.0 || i == j {
                let key: Vec<f64> = key_weights
                    .iter()
                    .map(|w| {
                        node_features[j]
                            .iter()
                            .zip(w.iter())
                            .map(|(&f, &wi)| f * wi)
                            .sum()
                    })
                    .collect();

                let score: f64 = query
                    .iter()
                    .zip(key.iter())
                    .map(|(&q, &k)| q * k)
                    .sum::<f64>()
                    / (d_head as f64).sqrt();

                attention_weights.push((j, score));
            }
        }

        if !attention_weights.is_empty() {
            let max_score = attention_weights
                .iter()
                .map(|(_, s)| *s)
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = attention_weights
                .iter()
                .map(|(_, s)| (s - max_score).exp())
                .sum();

            for (j, score) in &attention_weights {
                let att = (score - max_score).exp() / exp_sum;

                let value: Vec<f64> = value_weights
                    .iter()
                    .map(|w| {
                        node_features[*j]
                            .iter()
                            .zip(w.iter())
                            .map(|(&f, &wi)| f * wi)
                            .sum()
                    })
                    .collect();

                for (k, &v) in value.iter().enumerate() {
                    attention_sum[k] += att * v;
                }
            }
        }

        output[i] = attention_sum;
    }

    output
}

fn global_pooling(node_features: &[Vec<f64>], method: &str) -> Vec<f64> {
    if node_features.is_empty() {
        return Vec::new();
    }

    let dim = node_features[0].len();
    let n_nodes = node_features.len();

    match method {
        "max" => (0..dim)
            .map(|d| {
                node_features
                    .iter()
                    .map(|n| n[d])
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect(),
        "sum" => (0..dim)
            .map(|d| node_features.iter().map(|n| n[d]).sum())
            .collect(),
        _ => (0..dim)
            .map(|d| node_features.iter().map(|n| n[d]).sum::<f64>() / n_nodes as f64)
            .collect(),
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GraphSurvModel {
    layer_weights: Vec<Vec<Vec<f64>>>,
    layer_biases: Vec<Vec<f64>>,
    query_weights: Vec<Vec<f64>>,
    key_weights: Vec<Vec<f64>>,
    value_weights: Vec<Vec<f64>>,
    output_weights: Vec<Vec<f64>>,
    output_biases: Vec<f64>,
    time_bins: Vec<f64>,
    config: GraphSurvConfig,
    #[allow(dead_code)]
    n_node_features: usize,
}

#[pymethods]
impl GraphSurvModel {
    fn predict_survival(
        &self,
        node_features: Vec<Vec<Vec<f64>>>,
        adjacency_matrices: Vec<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if node_features.is_empty() {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = node_features
            .par_iter()
            .zip(adjacency_matrices.par_iter())
            .map(|(nodes, adj)| {
                let mut hidden = nodes.clone();

                for layer in 0..self.config.num_layers {
                    hidden = if self.config.aggregation == "attention" {
                        graph_attention(
                            &hidden,
                            adj,
                            &self.query_weights,
                            &self.key_weights,
                            &self.value_weights,
                            self.config.num_heads,
                        )
                    } else {
                        message_passing(
                            &hidden,
                            adj,
                            &self.layer_weights[layer],
                            &self.layer_biases[layer],
                        )
                    };
                }

                let graph_repr = global_pooling(&hidden, &self.config.aggregation);

                let logits: Vec<f64> = self
                    .output_weights
                    .iter()
                    .zip(self.output_biases.iter())
                    .map(|(w, &b)| {
                        let sum: f64 = graph_repr
                            .iter()
                            .zip(w.iter())
                            .map(|(&g, &wi)| g * wi)
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
                for i in (0..probs.len()).rev() {
                    cumsum += probs[i];
                    surv[i] = cumsum.min(1.0);
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    fn get_node_embeddings(
        &self,
        node_features: Vec<Vec<f64>>,
        adjacency: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let mut hidden = node_features;

        for layer in 0..self.config.num_layers {
            hidden = if self.config.aggregation == "attention" {
                graph_attention(
                    &hidden,
                    &adjacency,
                    &self.query_weights,
                    &self.key_weights,
                    &self.value_weights,
                    self.config.num_heads,
                )
            } else {
                message_passing(
                    &hidden,
                    &adjacency,
                    &self.layer_weights[layer],
                    &self.layer_biases[layer],
                )
            };
        }

        Ok(hidden)
    }

    fn get_time_bins(&self) -> Vec<f64> {
        self.time_bins.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphSurvModel(hidden={}, layers={}, aggregation={})",
            self.config.hidden_dim, self.config.num_layers, self.config.aggregation
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    node_features,
    adjacency_matrices,
    time,
    event,
    config=None
))]
pub fn fit_graph_surv(
    node_features: Vec<Vec<Vec<f64>>>,
    adjacency_matrices: Vec<Vec<Vec<f64>>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<GraphSurvConfig>,
) -> PyResult<GraphSurvModel> {
    let config = config.unwrap_or_else(|| {
        GraphSurvConfig::new(64, 3, 4, 20, 0.1, None, 0.001, 64, 100, None).unwrap()
    });

    let n = node_features.len();
    if n == 0 || time.len() != n || event.len() != n || adjacency_matrices.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_node_features = node_features[0]
        .first()
        .map(|n| n.len())
        .unwrap_or(config.hidden_dim);

    let mut rng = fastrand::Rng::new();
    if let Some(seed) = config.seed {
        rng.seed(seed);
    }

    let layer_weights: Vec<Vec<Vec<f64>>> = (0..config.num_layers)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| {
                    (0..n_node_features.max(config.hidden_dim))
                        .map(|_| rng.f64() * 0.1 - 0.05)
                        .collect()
                })
                .collect()
        })
        .collect();

    let layer_biases: Vec<Vec<f64>> = (0..config.num_layers)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();

    let query_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let key_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
        .map(|_| {
            (0..config.hidden_dim)
                .map(|_| rng.f64() * 0.1 - 0.05)
                .collect()
        })
        .collect();
    let value_weights: Vec<Vec<f64>> = (0..config.hidden_dim)
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

    Ok(GraphSurvModel {
        layer_weights,
        layer_biases,
        query_weights,
        key_weights,
        value_weights,
        output_weights,
        output_biases,
        time_bins,
        config,
        n_node_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let result = GraphSurvConfig::new(
            64,
            3,
            4,
            20,
            0.1,
            Some("invalid".to_string()),
            0.001,
            64,
            100,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_global_pooling() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mean = global_pooling(&features, "mean");
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((mean[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_message_passing() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let weights = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let biases = vec![0.0, 0.0];
        let output = message_passing(&features, &adj, &weights, &biases);
        assert_eq!(output.len(), 2);
    }
}
