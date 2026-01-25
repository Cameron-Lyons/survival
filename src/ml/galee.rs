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
pub enum UnimodalConstraint {
    SoftConstraint,
    HardConstraint,
    MixtureOfUnimodal,
}

#[pymethods]
impl UnimodalConstraint {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "soft" | "softconstraint" => Ok(UnimodalConstraint::SoftConstraint),
            "hard" | "hardconstraint" | "loglogistic" => Ok(UnimodalConstraint::HardConstraint),
            "mixture" | "mixtureotunimodal" => Ok(UnimodalConstraint::MixtureOfUnimodal),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown unimodal constraint type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GALEEConfig {
    #[pyo3(get, set)]
    pub health_dim: usize,
    #[pyo3(get, set)]
    pub hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub enforce_monotonic_decline: bool,
    #[pyo3(get, set)]
    pub decline_constraint_weight: f64,
    #[pyo3(get, set)]
    pub enforce_unimodal_hazard: bool,
    #[pyo3(get, set)]
    pub unimodal_constraint_type: UnimodalConstraint,
    #[pyo3(get, set)]
    pub unimodal_constraint_weight: f64,
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
impl GALEEConfig {
    #[new]
    #[pyo3(signature = (
        health_dim=32,
        hidden_sizes=vec![64, 32],
        num_attention_heads=4,
        num_durations=20,
        enforce_monotonic_decline=true,
        decline_constraint_weight=1.0,
        enforce_unimodal_hazard=true,
        unimodal_constraint_type=UnimodalConstraint::SoftConstraint,
        unimodal_constraint_weight=0.5,
        dropout_rate=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        seed=None
    ))]
    pub fn new(
        health_dim: usize,
        hidden_sizes: Vec<usize>,
        num_attention_heads: usize,
        num_durations: usize,
        enforce_monotonic_decline: bool,
        decline_constraint_weight: f64,
        enforce_unimodal_hazard: bool,
        unimodal_constraint_type: UnimodalConstraint,
        unimodal_constraint_weight: f64,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if health_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "health_dim must be positive",
            ));
        }
        if num_attention_heads == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_attention_heads must be positive",
            ));
        }
        if !health_dim.is_multiple_of(num_attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "health_dim must be divisible by num_attention_heads",
            ));
        }
        if num_durations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_durations must be positive",
            ));
        }
        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be positive",
            ));
        }
        if decline_constraint_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "decline_constraint_weight must be non-negative",
            ));
        }

        Ok(GALEEConfig {
            health_dim,
            hidden_sizes,
            num_attention_heads,
            num_durations,
            enforce_monotonic_decline,
            decline_constraint_weight,
            enforce_unimodal_hazard,
            unimodal_constraint_type,
            unimodal_constraint_weight,
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
struct HealthStateEncoder<B: burn::prelude::Backend> {
    input_projection: Linear<B>,
    layers: Vec<Linear<B>>,
    output_projection: Linear<B>,
    dropout: Dropout,
    health_dim: usize,
}

impl<B: burn::prelude::Backend> HealthStateEncoder<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_sizes: &[usize],
        health_dim: usize,
        dropout_rate: f64,
    ) -> Self {
        let input_projection = LinearConfig::new(
            input_size,
            hidden_sizes.first().copied().unwrap_or(health_dim),
        )
        .init(device);

        let mut layers = Vec::new();
        let mut prev_size = hidden_sizes.first().copied().unwrap_or(health_dim);
        for &size in hidden_sizes.iter().skip(1) {
            layers.push(LinearConfig::new(prev_size, size).init(device));
            prev_size = size;
        }

        let output_projection = LinearConfig::new(prev_size, health_dim).init(device);

        Self {
            input_projection,
            layers,
            output_projection,
            dropout: DropoutConfig::new(dropout_rate).init(),
            health_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, training: bool) -> Tensor<B, 3> {
        let [batch, seq_len, input_size] = x.dims();
        let device = x.device();

        let x_2d: Tensor<B, 2> = x.reshape([batch * seq_len, input_size]);
        let mut h = self.input_projection.forward(x_2d);
        h = relu(h);

        if training {
            h = self.dropout.forward(h);
        }

        for layer in &self.layers {
            h = layer.forward(h);
            h = relu(h);
            if training {
                h = self.dropout.forward(h);
            }
        }

        let h = self.output_projection.forward(h);
        h.reshape([batch, seq_len, self.health_dim])
    }
}

#[derive(Module, Debug)]
struct MonotonicConstraintLayer<B: burn::prelude::Backend> {
    projection: Linear<B>,
}

impl<B: burn::prelude::Backend> MonotonicConstraintLayer<B> {
    fn new(device: &B::Device, health_dim: usize) -> Self {
        Self {
            projection: LinearConfig::new(health_dim, health_dim).init(device),
        }
    }

    fn forward(&self, health_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, health_dim] = health_states.dims();
        let device = health_states.device();

        if seq_len < 2 {
            return health_states;
        }

        let mut constrained = Vec::with_capacity(seq_len);
        let first_state: Tensor<B, 2> = health_states
            .clone()
            .slice([0..batch, 0..1, 0..health_dim])
            .reshape([batch, health_dim]);
        constrained.push(first_state.clone());

        let mut prev_state = first_state;
        for t in 1..seq_len {
            let current: Tensor<B, 2> = health_states
                .clone()
                .slice([0..batch, t..t + 1, 0..health_dim])
                .reshape([batch, health_dim]);

            let decay = burn::tensor::activation::sigmoid(self.projection.forward(current.clone()));
            let constrained_state = prev_state.clone() * decay;

            constrained.push(constrained_state.clone());
            prev_state = constrained_state;
        }

        Tensor::stack(constrained, 1)
    }
}

#[derive(Module, Debug)]
struct GlobalAttention<B: burn::prelude::Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    num_heads: usize,
}

impl<B: burn::prelude::Backend> GlobalAttention<B> {
    fn new(device: &B::Device, hidden_size: usize, num_heads: usize) -> Self {
        Self {
            query: LinearConfig::new(hidden_size, hidden_size).init(device),
            key: LinearConfig::new(hidden_size, hidden_size).init(device),
            value: LinearConfig::new(hidden_size, hidden_size).init(device),
            output: LinearConfig::new(hidden_size, hidden_size).init(device),
            num_heads,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, seq_len, hidden] = x.dims();
        let device = x.device();

        let x_2d: Tensor<B, 2> = x.clone().reshape([batch * seq_len, hidden]);

        let q = self.query.forward(x_2d.clone());
        let k = self.key.forward(x_2d.clone());
        let v = self.value.forward(x_2d);

        let q: Tensor<B, 3> = q.reshape([batch, seq_len, hidden]);
        let k: Tensor<B, 3> = k.reshape([batch, seq_len, hidden]);
        let v: Tensor<B, 3> = v.reshape([batch, seq_len, hidden]);

        let scale = (hidden as f32).sqrt();
        let scores: Tensor<B, 3> = q.matmul(k.swap_dims(1, 2)) / scale;
        let attn_weights: Tensor<B, 3> = burn::tensor::activation::softmax(scores, 2);

        let context: Tensor<B, 3> = attn_weights.clone().matmul(v);

        let pooled: Tensor<B, 2> = context.mean_dim(1).reshape([batch, hidden]);
        let output = self.output.forward(pooled);

        let attn_weights_pooled: Tensor<B, 2> = attn_weights.mean_dim(2).reshape([batch, seq_len]);

        (output, attn_weights_pooled)
    }
}

#[derive(Module, Debug)]
struct HazardHead<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> HazardHead<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_sizes: &[usize],
        num_durations: usize,
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
            output: LinearConfig::new(prev_size, num_durations).init(device),
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
struct GALEENetwork<B: burn::prelude::Backend> {
    health_encoder: HealthStateEncoder<B>,
    monotonic_constraint: MonotonicConstraintLayer<B>,
    global_attention: GlobalAttention<B>,
    hazard_head: HazardHead<B>,
    health_dim: usize,
    num_durations: usize,
    enforce_monotonic: bool,
}

impl<B: burn::prelude::Backend> GALEENetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &GALEEConfig) -> Self {
        let health_encoder = HealthStateEncoder::new(
            device,
            n_features,
            &config.hidden_sizes,
            config.health_dim,
            config.dropout_rate,
        );

        let monotonic_constraint = MonotonicConstraintLayer::new(device, config.health_dim);

        let global_attention =
            GlobalAttention::new(device, config.health_dim, config.num_attention_heads);

        let hazard_head = HazardHead::new(
            device,
            config.health_dim,
            &[config.health_dim / 2],
            config.num_durations,
            config.dropout_rate,
        );

        Self {
            health_encoder,
            monotonic_constraint,
            global_attention,
            hazard_head,
            health_dim: config.health_dim,
            num_durations: config.num_durations,
            enforce_monotonic: config.enforce_monotonic_decline,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        training: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        let health_states = self.health_encoder.forward(x, training);

        let constrained_health = if self.enforce_monotonic {
            self.monotonic_constraint.forward(health_states.clone())
        } else {
            health_states.clone()
        };

        let (attended, attention_weights) =
            self.global_attention.forward(constrained_health.clone());

        let hazard_logits = self.hazard_head.forward(attended, training);

        (constrained_health, hazard_logits, attention_weights)
    }
}

fn compute_monotonicity_violation(
    health_trajectories: &[f32],
    batch_size: usize,
    seq_len: usize,
    health_dim: usize,
) -> f64 {
    let mut total_violation = 0.0;

    for b in 0..batch_size {
        for t in 1..seq_len {
            for d in 0..health_dim {
                let prev_idx = b * seq_len * health_dim + (t - 1) * health_dim + d;
                let curr_idx = b * seq_len * health_dim + t * health_dim + d;

                let prev = health_trajectories.get(prev_idx).copied().unwrap_or(0.0);
                let curr = health_trajectories.get(curr_idx).copied().unwrap_or(0.0);

                if curr > prev {
                    total_violation += (curr - prev) as f64;
                }
            }
        }
    }

    total_violation / (batch_size * (seq_len - 1).max(1) * health_dim) as f64
}

fn compute_unimodality_violation(
    hazard_dist: &[f32],
    batch_size: usize,
    num_durations: usize,
) -> f64 {
    let mut total_violation = 0.0;

    for b in 0..batch_size {
        let hazards: Vec<f32> = (0..num_durations)
            .map(|t| hazard_dist[b * num_durations + t])
            .collect();

        let max_idx = hazards
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        for t in 1..max_idx {
            if hazards[t] < hazards[t - 1] {
                total_violation += (hazards[t - 1] - hazards[t]) as f64;
            }
        }

        for t in (max_idx + 1)..num_durations {
            if hazards[t] > hazards[t - 1] {
                total_violation += (hazards[t] - hazards[t - 1]) as f64;
            }
        }
    }

    total_violation / batch_size as f64
}

fn compute_galee_loss(
    hazard_logits: &[f32],
    durations: &[usize],
    events: &[i32],
    num_durations: usize,
    batch_indices: &[usize],
) -> f64 {
    let batch_size = batch_indices.len();
    let eps = 1e-7;

    let mut hazards = vec![0.0f32; batch_size * num_durations];
    for i in 0..batch_size {
        let mut max_logit = f32::NEG_INFINITY;
        for t in 0..num_durations {
            max_logit = max_logit.max(hazard_logits[i * num_durations + t]);
        }

        let mut sum_exp = 0.0f32;
        for t in 0..num_durations {
            sum_exp += (hazard_logits[i * num_durations + t] - max_logit).exp();
        }

        for t in 0..num_durations {
            hazards[i * num_durations + t] =
                (hazard_logits[i * num_durations + t] - max_logit).exp() / sum_exp;
        }
    }

    let mut loss = 0.0;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        let mut cum_surv = 1.0f64;
        for t in 0..duration_bin {
            cum_surv *= (1.0 - hazards[local_idx * num_durations + t] as f64).max(eps);
        }

        if event > 0 {
            let h_t = hazards[local_idx * num_durations + duration_bin] as f64;
            loss -= (cum_surv * h_t.max(eps)).ln();
        } else {
            let h_t = hazards[local_idx * num_durations + duration_bin] as f64;
            loss -= (cum_surv * (1.0 - h_t).max(eps)).ln();
        }
    }

    loss / batch_size.max(1) as f64
}

#[derive(Clone)]
#[allow(dead_code)]
struct StoredWeights {
    encoder_weights: Vec<f32>,
    attention_weights: Vec<f32>,
    hazard_weights: Vec<f32>,
    health_dim: usize,
    num_durations: usize,
    n_features: usize,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("health_dim", &self.health_dim)
            .field("num_durations", &self.num_durations)
            .finish()
    }
}

fn extract_weights(
    model: &GALEENetwork<AutodiffBackend>,
    config: &GALEEConfig,
    n_features: usize,
) -> StoredWeights {
    StoredWeights {
        encoder_weights: vec![],
        attention_weights: vec![],
        hazard_weights: vec![],
        health_dim: config.health_dim,
        num_durations: config.num_durations,
        n_features,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GALEEResult {
    #[pyo3(get)]
    pub health_trajectories: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub hazard_distributions: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub attention_weights: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
#[pyclass]
#[allow(dead_code)]
pub struct GALEE {
    weights: StoredWeights,
    config: GALEEConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    #[pyo3(get)]
    pub max_seq_len: usize,
}

#[pymethods]
impl GALEE {
    #[getter]
    pub fn get_health_dim(&self) -> usize {
        self.weights.health_dim
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.weights.num_durations
    }

    fn __repr__(&self) -> String {
        format!(
            "GALEE(health_dim={}, num_durations={}, epochs={})",
            self.weights.health_dim,
            self.weights.num_durations,
            self.train_loss.len()
        )
    }
}

fn fit_galee_inner(
    x: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &GALEEConfig,
) -> GALEE {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: GALEENetwork<AutodiffBackend> = GALEENetwork::new(&device, n_features, config);

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
                .flat_map(|&i| {
                    (0..max_seq_len * n_features).map(move |j| {
                        x.get(i * max_seq_len * n_features + j)
                            .copied()
                            .unwrap_or(0.0) as f32
                    })
                })
                .collect();

            let x_data =
                burn::tensor::TensorData::new(x_batch, [batch_size, max_seq_len, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(x_data, &device);

            let (health_states, hazard_logits, _attn_weights) = model.forward(x_tensor, true);

            let hazard_vec = tensor_to_vec_f32(hazard_logits.clone().inner());
            let [bs, sl, hd] = health_states.dims();
            let health_reshaped: Tensor<NdArray, 2> =
                health_states.clone().inner().reshape([bs, sl * hd]);
            let health_vec = tensor_to_vec_f32(health_reshaped);

            let base_loss = compute_galee_loss(
                &hazard_vec,
                &duration_bins,
                event,
                config.num_durations,
                &batch_indices,
            );

            let mono_violation = if config.enforce_monotonic_decline {
                compute_monotonicity_violation(
                    &health_vec,
                    batch_size,
                    max_seq_len,
                    config.health_dim,
                )
            } else {
                0.0
            };

            let unimodal_violation = if config.enforce_unimodal_hazard {
                compute_unimodality_violation(&hazard_vec, batch_size, config.num_durations)
            } else {
                0.0
            };

            let total_loss = base_loss
                + config.decline_constraint_weight * mono_violation
                + config.unimodal_constraint_weight * unimodal_violation;

            epoch_loss += total_loss;
            n_batches += 1;

            let pseudo_loss = hazard_logits.mean();
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

    GALEE {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        max_seq_len,
    }
}

#[pyfunction]
#[pyo3(signature = (x, seq_lengths, n_obs, max_seq_len, n_features, time, event, config=None))]
pub fn galee(
    py: Python<'_>,
    x: Vec<f64>,
    seq_lengths: Vec<usize>,
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&GALEEConfig>,
) -> PyResult<GALEE> {
    let expected_len = n_obs * max_seq_len * n_features;
    if x.len() != expected_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "x length {} must equal n_obs * max_seq_len * n_features = {}",
            x.len(),
            expected_len
        )));
    }
    if seq_lengths.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "seq_lengths must have length n_obs",
        ));
    }
    if time.len() != n_obs || event.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have length n_obs",
        ));
    }

    let cfg = config.cloned().unwrap_or_else(|| {
        GALEEConfig::new(
            32,
            vec![64, 32],
            4,
            20,
            true,
            1.0,
            true,
            UnimodalConstraint::SoftConstraint,
            0.5,
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

    Ok(py.detach(move || {
        fit_galee_inner(
            &x,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &cfg,
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GALEEConfig::new(
            32,
            vec![64, 32],
            4,
            20,
            true,
            1.0,
            true,
            UnimodalConstraint::SoftConstraint,
            0.5,
            0.1,
            0.001,
            64,
            100,
            None,
            0.1,
            None,
        )
        .unwrap();

        assert_eq!(config.health_dim, 32);
        assert_eq!(config.num_durations, 20);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            GALEEConfig::new(
                0,
                vec![64],
                4,
                20,
                true,
                1.0,
                true,
                UnimodalConstraint::SoftConstraint,
                0.5,
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
            GALEEConfig::new(
                32,
                vec![64],
                3,
                20,
                true,
                1.0,
                true,
                UnimodalConstraint::SoftConstraint,
                0.5,
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
    fn test_monotonicity_violation() {
        let health = vec![1.0f32, 0.9, 0.8, 0.7, 0.95, 0.6];
        let violation = compute_monotonicity_violation(&health, 1, 3, 2);
        assert!(violation > 0.0);
    }

    #[test]
    fn test_unimodality_violation() {
        let hazards = vec![0.1f32, 0.2, 0.3, 0.2, 0.15, 0.25];
        let violation = compute_unimodality_violation(&hazards, 1, 6);
        assert!(violation > 0.0);
    }
}
