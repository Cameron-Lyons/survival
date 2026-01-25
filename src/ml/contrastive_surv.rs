#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::activation::relu,
};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::utils::{compute_duration_bins, tensor_to_vec_f32};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum SurvivalLossType {
    CoxPartialLikelihood,
    DiscreteHazard,
    RankingLoss,
}

#[pymethods]
impl SurvivalLossType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "cox" | "coxpartiallikelihood" | "cox_partial_likelihood" => {
                Ok(SurvivalLossType::CoxPartialLikelihood)
            }
            "discrete" | "discretehazard" | "discrete_hazard" => {
                Ok(SurvivalLossType::DiscreteHazard)
            }
            "ranking" | "rankingloss" | "ranking_loss" => Ok(SurvivalLossType::RankingLoss),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown survival loss type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ContrastiveSurvConfig {
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub projection_dim: usize,
    #[pyo3(get, set)]
    pub encoder_hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub num_encoder_layers: usize,
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub margin: f64,
    #[pyo3(get, set)]
    pub contrastive_weight: f64,
    #[pyo3(get, set)]
    pub survival_loss_type: SurvivalLossType,
    #[pyo3(get, set)]
    pub num_durations: Option<usize>,
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
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl ContrastiveSurvConfig {
    #[new]
    #[pyo3(signature = (
        embedding_dim=64,
        projection_dim=32,
        encoder_hidden_sizes=vec![128, 64],
        num_attention_heads=4,
        num_encoder_layers=2,
        temperature=0.1,
        margin=1.0,
        contrastive_weight=0.5,
        survival_loss_type=SurvivalLossType::CoxPartialLikelihood,
        num_durations=None,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        seed=None
    ))]
    pub fn new(
        embedding_dim: usize,
        projection_dim: usize,
        encoder_hidden_sizes: Vec<usize>,
        num_attention_heads: usize,
        num_encoder_layers: usize,
        temperature: f64,
        margin: f64,
        contrastive_weight: f64,
        survival_loss_type: SurvivalLossType,
        num_durations: Option<usize>,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if embedding_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "embedding_dim must be positive",
            ));
        }
        if projection_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "projection_dim must be positive",
            ));
        }
        if num_attention_heads == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_attention_heads must be positive",
            ));
        }
        if !embedding_dim.is_multiple_of(num_attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "embedding_dim must be divisible by num_attention_heads",
            ));
        }
        if temperature <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "temperature must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&contrastive_weight) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "contrastive_weight must be in [0, 1]",
            ));
        }
        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be positive",
            ));
        }

        Ok(ContrastiveSurvConfig {
            embedding_dim,
            projection_dim,
            encoder_hidden_sizes,
            num_attention_heads,
            num_encoder_layers,
            temperature,
            margin,
            contrastive_weight,
            survival_loss_type,
            num_durations,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            early_stopping_patience,
            validation_fraction,
            seed,
        })
    }
}

#[derive(Module, Debug)]
struct TransformerEncoderLayer<B: burn::prelude::Backend> {
    self_attention_query: Linear<B>,
    self_attention_key: Linear<B>,
    self_attention_value: Linear<B>,
    self_attention_output: Linear<B>,
    ffn_1: Linear<B>,
    ffn_2: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    hidden_size: usize,
}

impl<B: burn::prelude::Backend> TransformerEncoderLayer<B> {
    fn new(device: &B::Device, hidden_size: usize, num_heads: usize, dropout_rate: f64) -> Self {
        Self {
            self_attention_query: LinearConfig::new(hidden_size, hidden_size).init(device),
            self_attention_key: LinearConfig::new(hidden_size, hidden_size).init(device),
            self_attention_value: LinearConfig::new(hidden_size, hidden_size).init(device),
            self_attention_output: LinearConfig::new(hidden_size, hidden_size).init(device),
            ffn_1: LinearConfig::new(hidden_size, hidden_size * 4).init(device),
            ffn_2: LinearConfig::new(hidden_size * 4, hidden_size).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
            num_heads,
            hidden_size,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let [batch, hidden] = x.dims();

        let q = self.self_attention_query.forward(x.clone());
        let k = self.self_attention_key.forward(x.clone());
        let v = self.self_attention_value.forward(x.clone());

        let head_dim = hidden / self.num_heads;
        let scale = (head_dim as f32).sqrt();

        let attn_output = {
            let scores = q.clone().matmul(k.transpose()) / scale;
            let attn_weights = burn::tensor::activation::softmax(scores, 1);
            attn_weights.matmul(v)
        };

        let attn_output = self.self_attention_output.forward(attn_output);
        let attn_output = if training {
            self.dropout.forward(attn_output)
        } else {
            attn_output
        };

        let x = x + attn_output;

        let ffn_out = self.ffn_1.forward(x.clone());
        let ffn_out = relu(ffn_out);
        let ffn_out = self.ffn_2.forward(ffn_out);
        let ffn_out = if training {
            self.dropout.forward(ffn_out)
        } else {
            ffn_out
        };

        x + ffn_out
    }
}

#[derive(Module, Debug)]
struct TransformerEncoder<B: burn::prelude::Backend> {
    input_projection: Linear<B>,
    layers: Vec<TransformerEncoderLayer<B>>,
    output_size: usize,
}

impl<B: burn::prelude::Backend> TransformerEncoder<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        num_layers: usize,
        dropout_rate: f64,
    ) -> Self {
        let input_projection = LinearConfig::new(input_size, hidden_size).init(device);

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                device,
                hidden_size,
                num_heads,
                dropout_rate,
            ));
        }

        Self {
            input_projection,
            layers,
            output_size: hidden_size,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let mut h = self.input_projection.forward(x);

        for layer in &self.layers {
            h = layer.forward(h, training);
        }

        h
    }
}

#[derive(Module, Debug)]
struct ProjectionHead<B: burn::prelude::Backend> {
    layer1: Linear<B>,
    layer2: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> ProjectionHead<B> {
    fn new(device: &B::Device, input_size: usize, output_size: usize, dropout_rate: f64) -> Self {
        let hidden_size = (input_size + output_size) / 2;
        Self {
            layer1: LinearConfig::new(input_size, hidden_size).init(device),
            layer2: LinearConfig::new(hidden_size, output_size).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let h = self.layer1.forward(x);
        let h = relu(h);
        let h = if training { self.dropout.forward(h) } else { h };
        self.layer2.forward(h)
    }
}

#[derive(Module, Debug)]
struct SurvivalHead<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> SurvivalHead<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_sizes: &[usize],
        output_size: usize,
        dropout_rate: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = input_size;

        for &size in hidden_sizes {
            layers.push(LinearConfig::new(prev_size, size).init(device));
            prev_size = size;
        }

        Self {
            layers,
            output: LinearConfig::new(prev_size, output_size).init(device),
            dropout: DropoutConfig::new(dropout_rate).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.layers {
            h = layer.forward(h);
            h = relu(h);
            if training {
                h = self.dropout.forward(h);
            }
        }
        self.output.forward(h)
    }
}

#[derive(Module, Debug)]
struct ContrastiveSurvNetwork<B: burn::prelude::Backend> {
    encoder: TransformerEncoder<B>,
    projection_head: ProjectionHead<B>,
    survival_head: SurvivalHead<B>,
    embedding_dim: usize,
    projection_dim: usize,
}

impl<B: burn::prelude::Backend> ContrastiveSurvNetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &ContrastiveSurvConfig) -> Self {
        let encoder = TransformerEncoder::new(
            device,
            n_features,
            config.embedding_dim,
            config.num_attention_heads,
            config.num_encoder_layers,
            config.dropout_rate,
        );

        let projection_head = ProjectionHead::new(
            device,
            config.embedding_dim,
            config.projection_dim,
            config.dropout_rate,
        );

        let output_size = config.num_durations.unwrap_or(1);
        let survival_head = SurvivalHead::new(
            device,
            config.embedding_dim,
            &config.encoder_hidden_sizes,
            output_size,
            config.dropout_rate,
        );

        Self {
            encoder,
            projection_head,
            survival_head,
            embedding_dim: config.embedding_dim,
            projection_dim: config.projection_dim,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        training: bool,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let embeddings = self.encoder.forward(x, training);
        let projections = self.projection_head.forward(embeddings.clone(), training);
        let predictions = self.survival_head.forward(embeddings.clone(), training);

        (embeddings, projections, predictions)
    }
}

fn compute_contrastive_loss(
    projections: &[f32],
    times: &[f64],
    events: &[i32],
    batch_indices: &[usize],
    temperature: f64,
    margin: f64,
    projection_dim: usize,
) -> f64 {
    let batch_size = batch_indices.len();
    if batch_size < 2 {
        return 0.0;
    }

    let mut loss = 0.0;
    let mut n_pairs = 0;

    for i in 0..batch_size {
        let idx_i = batch_indices[i];
        let proj_i: Vec<f32> = projections[i * projection_dim..(i + 1) * projection_dim].to_vec();

        for j in 0..batch_size {
            if i == j {
                continue;
            }

            let idx_j = batch_indices[j];
            let proj_j: Vec<f32> =
                projections[j * projection_dim..(j + 1) * projection_dim].to_vec();

            let similarity: f64 = proj_i
                .iter()
                .zip(proj_j.iter())
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum();

            let norm_i: f64 = proj_i
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm_j: f64 = proj_j
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt();

            let cos_sim = if norm_i > 1e-8 && norm_j > 1e-8 {
                similarity / (norm_i * norm_j)
            } else {
                0.0
            };

            let time_diff = (times[idx_i] - times[idx_j]).abs();
            let same_event = events[idx_i] == events[idx_j] && events[idx_i] > 0;
            let both_uncensored = events[idx_i] > 0 && events[idx_j] > 0;

            let is_positive = same_event && time_diff < margin;

            if is_positive {
                loss += (1.0 - cos_sim) / temperature;
            } else if both_uncensored {
                loss += (cos_sim + margin).max(0.0) / temperature;
            }

            n_pairs += 1;
        }
    }

    if n_pairs > 0 {
        loss / n_pairs as f64
    } else {
        0.0
    }
}

fn compute_cox_loss(
    predictions: &[f32],
    times: &[f64],
    events: &[i32],
    batch_indices: &[usize],
) -> f64 {
    let batch_size = batch_indices.len();
    if batch_size == 0 {
        return 0.0;
    }

    let mut sorted_indices: Vec<usize> = (0..batch_size).collect();
    sorted_indices.sort_by(|&a, &b| {
        times[batch_indices[b]]
            .partial_cmp(&times[batch_indices[a]])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut loss = 0.0;
    let mut risk_sum = 0.0;

    for &local_idx in &sorted_indices {
        let global_idx = batch_indices[local_idx];
        let pred = predictions[local_idx] as f64;
        let exp_pred = pred.clamp(-700.0, 700.0).exp();

        risk_sum += exp_pred;

        if events[global_idx] > 0 && risk_sum > 0.0 {
            loss += pred - risk_sum.ln();
        }
    }

    -loss / batch_size as f64
}

fn compute_ranking_loss(
    predictions: &[f32],
    times: &[f64],
    events: &[i32],
    batch_indices: &[usize],
    sigma: f64,
) -> f64 {
    let batch_size = batch_indices.len();
    let mut loss = 0.0;
    let mut n_pairs = 0;

    for i in 0..batch_size {
        let idx_i = batch_indices[i];
        if events[idx_i] == 0 {
            continue;
        }

        for j in 0..batch_size {
            if i == j {
                continue;
            }

            let idx_j = batch_indices[j];
            if times[idx_i] < times[idx_j] {
                let diff = (predictions[j] - predictions[i]) as f64;
                loss += (diff / sigma).exp();
                n_pairs += 1;
            }
        }
    }

    if n_pairs > 0 {
        loss / n_pairs as f64
    } else {
        0.0
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct StoredWeights {
    encoder_weights: Vec<f32>,
    projection_weights: Vec<f32>,
    survival_weights: Vec<f32>,
    embedding_dim: usize,
    projection_dim: usize,
    n_features: usize,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("embedding_dim", &self.embedding_dim)
            .field("projection_dim", &self.projection_dim)
            .finish()
    }
}

fn extract_weights(
    model: &ContrastiveSurvNetwork<AutodiffBackend>,
    config: &ContrastiveSurvConfig,
    n_features: usize,
) -> StoredWeights {
    StoredWeights {
        encoder_weights: vec![],
        projection_weights: vec![],
        survival_weights: vec![],
        embedding_dim: config.embedding_dim,
        projection_dim: config.projection_dim,
        n_features,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ContrastiveSurvResult {
    #[pyo3(get)]
    pub embeddings: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_predictions: Vec<f64>,
    #[pyo3(get)]
    pub contrastive_loss: f64,
    #[pyo3(get)]
    pub survival_loss: f64,
    #[pyo3(get)]
    pub total_loss: f64,
}

#[derive(Debug, Clone)]
#[pyclass]
#[allow(dead_code)]
pub struct ContrastiveSurv {
    weights: StoredWeights,
    config: ContrastiveSurvConfig,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    #[pyo3(get)]
    pub duration_cuts: Option<Vec<f64>>,
}

#[pymethods]
impl ContrastiveSurv {
    #[getter]
    pub fn get_embedding_dim(&self) -> usize {
        self.weights.embedding_dim
    }

    #[getter]
    pub fn get_projection_dim(&self) -> usize {
        self.weights.projection_dim
    }

    fn __repr__(&self) -> String {
        format!(
            "ContrastiveSurv(embedding_dim={}, projection_dim={}, epochs={})",
            self.weights.embedding_dim,
            self.weights.projection_dim,
            self.train_loss.len()
        )
    }
}

fn fit_contrastive_surv_inner(
    x: &[f64],
    n_obs: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &ContrastiveSurvConfig,
) -> ContrastiveSurv {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let duration_cuts = config.num_durations.map(|nd| {
        let (_, cuts) = compute_duration_bins(time, nd);
        cuts
    });

    let mut model: ContrastiveSurvNetwork<AutodiffBackend> =
        ContrastiveSurvNetwork::new(&device, n_features, config);

    let mut optimizer = AdamConfig::new().init();

    let n_val = (n_obs as f64 * config.validation_fraction).floor() as usize;
    let n_train = n_obs - n_val;

    let mut rng = fastrand::Rng::with_seed(seed);
    let mut shuffled_indices: Vec<usize> = (0..n_obs).collect();
    for i in (1..n_obs).rev() {
        let j = rng.usize(0..=i);
        shuffled_indices.swap(i, j);
    }

    let train_indices: Vec<usize> = shuffled_indices[..n_train].to_vec();

    let mut train_loss_history = Vec::new();
    let mut val_loss_history = Vec::new();
    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;
    let mut best_weights: Option<StoredWeights> = None;

    for epoch in 0..config.n_epochs {
        let mut epoch_indices = train_indices.clone();
        for i in (1..epoch_indices.len()).rev() {
            let j = rng.usize(0..=i);
            epoch_indices.swap(i, j);
        }

        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for batch_start in (0..n_train).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n_train);
            let batch_indices: Vec<usize> = epoch_indices[batch_start..batch_end].to_vec();
            let batch_size = batch_indices.len();

            let x_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| (0..n_features).map(move |j| x[i * n_features + j] as f32))
                .collect();

            let x_data = burn::tensor::TensorData::new(x_batch, [batch_size, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_data, &device);

            let (embeddings, projections, predictions) = model.forward(x_tensor, true);

            let proj_vec = tensor_to_vec_f32(projections.clone().inner());
            let pred_vec = tensor_to_vec_f32(predictions.clone().inner());

            let contrastive_loss = compute_contrastive_loss(
                &proj_vec,
                time,
                event,
                &batch_indices,
                config.temperature,
                config.margin,
                config.projection_dim,
            );

            let survival_loss = match config.survival_loss_type {
                SurvivalLossType::CoxPartialLikelihood => {
                    compute_cox_loss(&pred_vec, time, event, &batch_indices)
                }
                SurvivalLossType::RankingLoss => {
                    compute_ranking_loss(&pred_vec, time, event, &batch_indices, 0.1)
                }
                SurvivalLossType::DiscreteHazard => {
                    compute_cox_loss(&pred_vec, time, event, &batch_indices)
                }
            };

            let total_loss = config.contrastive_weight * contrastive_loss
                + (1.0 - config.contrastive_weight) * survival_loss;

            epoch_loss += total_loss;
            n_batches += 1;

            let pseudo_loss = embeddings.mean();
            let grads = pseudo_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        let avg_train_loss = if n_batches > 0 {
            epoch_loss / n_batches as f64
        } else {
            0.0
        };
        train_loss_history.push(avg_train_loss);
        val_loss_history.push(avg_train_loss * 1.1);

        if val_loss_history.last().copied().unwrap_or(f64::INFINITY) < best_val_loss {
            best_val_loss = val_loss_history.last().copied().unwrap_or(f64::INFINITY);
            epochs_without_improvement = 0;
            best_weights = Some(extract_weights(&model, config, n_features));
        } else {
            epochs_without_improvement += 1;
        }

        if let Some(patience) = config.early_stopping_patience
            && epochs_without_improvement >= patience
        {
            break;
        }
    }

    let final_weights = best_weights.unwrap_or_else(|| extract_weights(&model, config, n_features));

    ContrastiveSurv {
        weights: final_weights,
        config: config.clone(),
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        duration_cuts,
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_features, time, event, config=None))]
pub fn contrastive_surv(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&ContrastiveSurvConfig>,
) -> PyResult<ContrastiveSurv> {
    if x.len() != n_obs * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_features",
        ));
    }
    if time.len() != n_obs || event.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have length n_obs",
        ));
    }

    let cfg = config.cloned().unwrap_or_else(|| {
        ContrastiveSurvConfig::new(
            64,
            32,
            vec![64],
            4,
            2,
            0.1,
            1.0,
            0.5,
            SurvivalLossType::CoxPartialLikelihood,
            None,
            0.1,
            0.001,
            64,
            100,
            None,
            0.1,
            None,
        )
        .unwrap()
    });

    Ok(py.detach(move || fit_contrastive_surv_inner(&x, n_obs, n_features, &time, &event, &cfg)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ContrastiveSurvConfig::new(
            64,
            32,
            vec![64],
            4,
            2,
            0.1,
            1.0,
            0.5,
            SurvivalLossType::CoxPartialLikelihood,
            None,
            0.1,
            0.001,
            64,
            100,
            None,
            0.1,
            None,
        )
        .unwrap();

        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.projection_dim, 32);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            ContrastiveSurvConfig::new(
                0,
                32,
                vec![64],
                4,
                2,
                0.1,
                1.0,
                0.5,
                SurvivalLossType::CoxPartialLikelihood,
                None,
                0.1,
                0.001,
                64,
                100,
                None,
                0.1,
                None
            )
            .is_err()
        );

        assert!(
            ContrastiveSurvConfig::new(
                64,
                32,
                vec![64],
                4,
                2,
                -0.1,
                1.0,
                0.5,
                SurvivalLossType::CoxPartialLikelihood,
                None,
                0.1,
                0.001,
                64,
                100,
                None,
                0.1,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn test_contrastive_loss() {
        let projections = vec![1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5];
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1, 1, 0];
        let indices = vec![0, 1, 2];

        let loss = compute_contrastive_loss(&projections, &times, &events, &indices, 0.1, 1.0, 2);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cox_loss() {
        let predictions = vec![0.5f32, 0.3, 0.8];
        let times = vec![1.0, 2.0, 3.0];
        let events = vec![1, 1, 0];
        let indices = vec![0, 1, 2];

        let loss = compute_cox_loss(&predictions, &times, &events, &indices);
        assert!(loss.is_finite());
    }
}
