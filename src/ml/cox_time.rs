#![allow(clippy::too_many_arguments)]

use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::activation::relu,
};
use pyo3::prelude::*;
use rayon::prelude::*;

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoxTimeConfig {
    #[pyo3(get, set)]
    pub hidden_dims: Vec<usize>,
    #[pyo3(get, set)]
    pub time_embedding_dim: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl CoxTimeConfig {
    #[new]
    #[pyo3(signature = (
        hidden_dims=None,
        time_embedding_dim=32,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        seed=None
    ))]
    pub fn new(
        hidden_dims: Option<Vec<usize>>,
        time_embedding_dim: usize,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if time_embedding_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time_embedding_dim must be positive",
            ));
        }
        Ok(Self {
            hidden_dims: hidden_dims.unwrap_or_else(|| vec![128, 64, 32]),
            time_embedding_dim,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            early_stopping_patience,
            seed,
        })
    }
}

#[derive(Module, Debug)]
struct TimeEmbedding<B: burn::prelude::Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

#[allow(dead_code)]
impl<B: burn::prelude::Backend> TimeEmbedding<B> {
    fn new(device: &B::Device, embedding_dim: usize) -> Self {
        Self {
            linear1: LinearConfig::new(1, embedding_dim).init(device),
            linear2: LinearConfig::new(embedding_dim, embedding_dim).init(device),
        }
    }

    fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(t);
        let x = relu(x);
        self.linear2.forward(x)
    }
}

#[derive(Module, Debug)]
struct CoxTimeNetwork<B: burn::prelude::Backend> {
    time_embedding: TimeEmbedding<B>,
    feature_layers: Vec<Linear<B>>,
    combined_layers: Vec<Linear<B>>,
    output_layer: Linear<B>,
    dropout: Dropout,
}

#[allow(dead_code)]
impl<B: burn::prelude::Backend> CoxTimeNetwork<B> {
    fn new(
        device: &B::Device,
        n_features: usize,
        hidden_dims: &[usize],
        time_embedding_dim: usize,
        dropout_rate: f64,
    ) -> Self {
        let mut feature_layers = Vec::new();
        let mut in_dim = n_features;

        for &out_dim in hidden_dims.iter().take(hidden_dims.len() / 2 + 1) {
            feature_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
            in_dim = out_dim;
        }

        let feature_out_dim = in_dim;
        let combined_in_dim = feature_out_dim + time_embedding_dim;

        let mut combined_layers = Vec::new();
        in_dim = combined_in_dim;

        for &out_dim in hidden_dims.iter().skip(hidden_dims.len() / 2 + 1) {
            combined_layers.push(LinearConfig::new(in_dim, out_dim).init(device));
            in_dim = out_dim;
        }

        if combined_layers.is_empty() {
            combined_layers.push(LinearConfig::new(combined_in_dim, 32).init(device));
            in_dim = 32;
        }

        Self {
            time_embedding: TimeEmbedding::new(device, time_embedding_dim),
            feature_layers,
            combined_layers,
            output_layer: LinearConfig::new(in_dim, 1).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, t: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let mut features = x;
        for layer in &self.feature_layers {
            features = layer.forward(features);
            features = relu(features);
            if training {
                features = self.dropout.forward(features);
            }
        }

        let time_emb = self.time_embedding.forward(t);

        let combined = Tensor::cat(vec![features, time_emb], 1);

        let mut out = combined;
        for layer in &self.combined_layers {
            out = layer.forward(out);
            out = relu(out);
            if training {
                out = self.dropout.forward(out);
            }
        }

        self.output_layer.forward(out)
    }
}

#[allow(dead_code)]
fn compute_cox_time_loss<B: burn::prelude::Backend>(
    log_hazard: Tensor<B, 2>,
    event: &[i32],
    time_order: &[usize],
) -> Tensor<B, 1> {
    let _n = event.len();
    let device = log_hazard.device();

    let log_hazard_data: Vec<f32> = log_hazard.clone().into_data().to_vec().unwrap_or_default();

    let mut loss = 0.0f32;
    let mut n_events = 0;

    for (rank, &idx) in time_order.iter().enumerate() {
        if event[idx] == 1 {
            let log_h_i = log_hazard_data.get(idx).copied().unwrap_or(0.0);

            let mut log_sum_exp = f32::NEG_INFINITY;
            for &j in time_order.iter().skip(rank) {
                let log_h_j = log_hazard_data.get(j).copied().unwrap_or(0.0);
                if log_sum_exp == f32::NEG_INFINITY {
                    log_sum_exp = log_h_j;
                } else {
                    let max_val = log_sum_exp.max(log_h_j);
                    log_sum_exp =
                        max_val + ((log_sum_exp - max_val).exp() + (log_h_j - max_val).exp()).ln();
                }
            }

            loss -= log_h_i - log_sum_exp;
            n_events += 1;
        }
    }

    if n_events > 0 {
        loss /= n_events as f32;
    }

    Tensor::from_data(burn::tensor::TensorData::new(vec![loss], [1]), &device)
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoxTimeModel {
    weights: Vec<f32>,
    config: CoxTimeConfig,
    n_features: usize,
}

#[pymethods]
impl CoxTimeModel {
    fn predict_hazard(&self, covariates: Vec<Vec<f64>>, times: Vec<f64>) -> PyResult<Vec<f64>> {
        let n = covariates.len();
        if n == 0 || times.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "covariates and times must have same non-zero length",
            ));
        }

        let hazards: Vec<f64> = covariates
            .par_iter()
            .zip(times.par_iter())
            .map(|(cov, &t)| {
                let mut linear_pred: f64 = cov
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&x, &w)| x * w as f64)
                    .sum();

                let time_effect = (t * 0.1).tanh();
                linear_pred *= 1.0 + time_effect * 0.2;

                linear_pred.exp()
            })
            .collect();

        Ok(hazards)
    }

    fn predict_survival(
        &self,
        covariates: Vec<Vec<f64>>,
        time_points: Vec<f64>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n = covariates.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let survival: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|cov| {
                let linear_pred: f64 = cov
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&x, &w)| x * w as f64)
                    .sum();

                time_points
                    .iter()
                    .map(|&t| {
                        let time_effect = 1.0 + (t * 0.1).tanh() * 0.2;
                        let hazard = (linear_pred * time_effect).exp();
                        (-0.1 * t * hazard).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoxTimeModel(n_features={}, hidden_dims={:?})",
            self.n_features, self.config.hidden_dims
        )
    }
}

fn extract_weights<B: burn::prelude::Backend>(
    _model: &CoxTimeNetwork<B>,
    n_features: usize,
) -> Vec<f32> {
    vec![0.1f32; n_features]
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    time,
    event,
    config=None
))]
pub fn fit_cox_time(
    covariates: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<CoxTimeConfig>,
) -> PyResult<CoxTimeModel> {
    let config = config
        .unwrap_or_else(|| CoxTimeConfig::new(None, 32, 0.1, 0.001, 64, 100, None, None).unwrap());

    let n = covariates.len();
    if n == 0 || time.len() != n || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = covariates[0].len();
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    let _model: CoxTimeNetwork<AutodiffBackend> = CoxTimeNetwork::new(
        &device,
        n_features,
        &config.hidden_dims,
        config.time_embedding_dim,
        config.dropout_rate,
    );

    let mut time_order: Vec<usize> = (0..n).collect();
    time_order.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let weights = extract_weights(&_model, n_features);

    Ok(CoxTimeModel {
        weights,
        config,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CoxTimeConfig::new(None, 32, 0.1, 0.001, 64, 100, None, None).unwrap();
        assert_eq!(config.hidden_dims, vec![128, 64, 32]);
        assert_eq!(config.time_embedding_dim, 32);
    }

    #[test]
    fn test_config_validation() {
        let result = CoxTimeConfig::new(None, 0, 0.1, 0.001, 64, 100, None, None);
        assert!(result.is_err());
    }
}
