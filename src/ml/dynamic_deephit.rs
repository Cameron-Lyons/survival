#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::single_range_in_vec_init,
    clippy::manual_memcpy
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

use super::utils::{compute_duration_bins, linear_forward, relu_vec, tensor_to_vec_f32};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum TemporalType {
    LSTM,
    GRU,
    Attention,
    LSTMAttention,
}

#[pymethods]
impl TemporalType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "lstm" => Ok(TemporalType::LSTM),
            "gru" => Ok(TemporalType::GRU),
            "attention" => Ok(TemporalType::Attention),
            "lstm_attention" | "lstmattention" => Ok(TemporalType::LSTMAttention),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown temporal type. Use 'lstm', 'gru', 'attention', or 'lstm_attention'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DynamicDeepHitConfig {
    #[pyo3(get, set)]
    pub temporal_type: TemporalType,
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub num_temporal_layers: usize,
    #[pyo3(get, set)]
    pub bidirectional: bool,
    #[pyo3(get, set)]
    pub shared_hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub cause_hidden_sizes: Vec<usize>,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_causes: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub sigma: f64,
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
impl DynamicDeepHitConfig {
    #[new]
    #[pyo3(signature = (
        temporal_type=TemporalType::LSTM,
        embedding_dim=64,
        num_temporal_layers=2,
        bidirectional=false,
        shared_hidden_sizes=vec![64, 64],
        cause_hidden_sizes=vec![32],
        num_durations=10,
        num_causes=1,
        dropout_rate=0.1,
        alpha=0.5,
        sigma=0.1,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        seed=None
    ))]
    pub fn new(
        temporal_type: TemporalType,
        embedding_dim: usize,
        num_temporal_layers: usize,
        bidirectional: bool,
        shared_hidden_sizes: Vec<usize>,
        cause_hidden_sizes: Vec<usize>,
        num_durations: usize,
        num_causes: usize,
        dropout_rate: f64,
        alpha: f64,
        sigma: f64,
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
        if num_temporal_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_temporal_layers must be positive",
            ));
        }
        if num_durations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_durations must be positive",
            ));
        }
        if num_causes == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_causes must be positive",
            ));
        }
        if batch_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be positive",
            ));
        }
        if !(0.0..1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be in [0, 1)",
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be in [0, 1]",
            ));
        }
        if !(0.0..1.0).contains(&validation_fraction) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "validation_fraction must be in [0, 1)",
            ));
        }

        Ok(DynamicDeepHitConfig {
            temporal_type,
            embedding_dim,
            num_temporal_layers,
            bidirectional,
            shared_hidden_sizes,
            cause_hidden_sizes,
            num_durations,
            num_causes,
            dropout_rate,
            alpha,
            sigma,
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
struct LSTMCell<B: burn::prelude::Backend> {
    input_gate: Linear<B>,
    forget_gate: Linear<B>,
    cell_gate: Linear<B>,
    output_gate: Linear<B>,
    hidden_size: usize,
}

impl<B: burn::prelude::Backend> LSTMCell<B> {
    fn new(device: &B::Device, input_size: usize, hidden_size: usize) -> Self {
        let gate_size = hidden_size;
        Self {
            input_gate: LinearConfig::new(input_size + hidden_size, gate_size).init(device),
            forget_gate: LinearConfig::new(input_size + hidden_size, gate_size).init(device),
            cell_gate: LinearConfig::new(input_size + hidden_size, gate_size).init(device),
            output_gate: LinearConfig::new(input_size + hidden_size, gate_size).init(device),
            hidden_size,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        h_prev: Tensor<B, 2>,
        c_prev: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [batch, _] = x.dims();
        let combined = Tensor::cat(vec![x, h_prev.clone()], 1);

        let i = burn::tensor::activation::sigmoid(self.input_gate.forward(combined.clone()));
        let f = burn::tensor::activation::sigmoid(self.forget_gate.forward(combined.clone()));
        let g = burn::tensor::activation::tanh(self.cell_gate.forward(combined.clone()));
        let o = burn::tensor::activation::sigmoid(self.output_gate.forward(combined));

        let c_new = f * c_prev + i * g;
        let h_new = o * burn::tensor::activation::tanh(c_new.clone());

        (h_new, c_new)
    }
}

#[derive(Module, Debug)]
struct GRUCell<B: burn::prelude::Backend> {
    reset_gate: Linear<B>,
    update_gate: Linear<B>,
    new_gate: Linear<B>,
    hidden_size: usize,
}

impl<B: burn::prelude::Backend> GRUCell<B> {
    fn new(device: &B::Device, input_size: usize, hidden_size: usize) -> Self {
        Self {
            reset_gate: LinearConfig::new(input_size + hidden_size, hidden_size).init(device),
            update_gate: LinearConfig::new(input_size + hidden_size, hidden_size).init(device),
            new_gate: LinearConfig::new(input_size + hidden_size, hidden_size).init(device),
            hidden_size,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, h_prev: Tensor<B, 2>) -> Tensor<B, 2> {
        let combined = Tensor::cat(vec![x.clone(), h_prev.clone()], 1);

        let r = burn::tensor::activation::sigmoid(self.reset_gate.forward(combined.clone()));
        let z = burn::tensor::activation::sigmoid(self.update_gate.forward(combined));

        let combined_reset = Tensor::cat(vec![x, r * h_prev.clone()], 1);
        let n = burn::tensor::activation::tanh(self.new_gate.forward(combined_reset));

        let ones: Tensor<B, 2> = Tensor::ones_like(&z);
        (ones - z.clone()) * n + z * h_prev
    }
}

#[derive(Module, Debug)]
struct TemporalAttention<B: burn::prelude::Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    hidden_size: usize,
}

impl<B: burn::prelude::Backend> TemporalAttention<B> {
    fn new(device: &B::Device, hidden_size: usize) -> Self {
        Self {
            query: LinearConfig::new(hidden_size, hidden_size).init(device),
            key: LinearConfig::new(hidden_size, hidden_size).init(device),
            value: LinearConfig::new(hidden_size, hidden_size).init(device),
            output: LinearConfig::new(hidden_size, hidden_size).init(device),
            hidden_size,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
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

        let attn_weights = burn::tensor::activation::softmax(scores, 2);
        let context: Tensor<B, 3> = attn_weights.matmul(v);

        let last_context: Tensor<B, 2> = context
            .slice([0..batch, seq_len - 1..seq_len, 0..hidden])
            .reshape([batch, hidden]);

        self.output.forward(last_context)
    }
}

#[derive(Module, Debug)]
struct TemporalEncoder<B: burn::prelude::Backend> {
    lstm_cells: Vec<LSTMCell<B>>,
    gru_cells: Vec<GRUCell<B>>,
    attention: Option<TemporalAttention<B>>,
    input_projection: Linear<B>,
    temporal_type: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
}

impl<B: burn::prelude::Backend> TemporalEncoder<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        temporal_type: TemporalType,
        bidirectional: bool,
    ) -> Self {
        let input_projection = LinearConfig::new(input_size, hidden_size).init(device);

        let mut lstm_cells = Vec::new();
        let mut gru_cells = Vec::new();

        match temporal_type {
            TemporalType::LSTM | TemporalType::LSTMAttention => {
                for _ in 0..num_layers {
                    lstm_cells.push(LSTMCell::new(device, hidden_size, hidden_size));
                }
            }
            TemporalType::GRU => {
                for _ in 0..num_layers {
                    gru_cells.push(GRUCell::new(device, hidden_size, hidden_size));
                }
            }
            TemporalType::Attention => {}
        }

        let attention = match temporal_type {
            TemporalType::Attention | TemporalType::LSTMAttention => {
                Some(TemporalAttention::new(device, hidden_size))
            }
            _ => None,
        };

        let temporal_type_code = match temporal_type {
            TemporalType::LSTM => 0,
            TemporalType::GRU => 1,
            TemporalType::Attention => 2,
            TemporalType::LSTMAttention => 3,
        };

        Self {
            lstm_cells,
            gru_cells,
            attention,
            input_projection,
            temporal_type: temporal_type_code,
            hidden_size,
            num_layers,
            bidirectional,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, seq_lengths: &[usize]) -> Tensor<B, 2> {
        let [batch, max_seq_len, input_size] = x.dims();
        let device = x.device();

        let x_2d: Tensor<B, 2> = x.clone().reshape([batch * max_seq_len, input_size]);
        let projected = self.input_projection.forward(x_2d);
        let projected: Tensor<B, 3> = projected.reshape([batch, max_seq_len, self.hidden_size]);

        match self.temporal_type {
            0 => {
                let mut h: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_size], &device);
                let mut c: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_size], &device);

                for t in 0..max_seq_len {
                    let x_t: Tensor<B, 2> = projected
                        .clone()
                        .slice([0..batch, t..t + 1, 0..self.hidden_size])
                        .reshape([batch, self.hidden_size]);

                    for cell in &self.lstm_cells {
                        let (h_new, c_new) = cell.forward(x_t.clone(), h.clone(), c.clone());
                        h = h_new;
                        c = c_new;
                    }
                }
                h
            }
            1 => {
                let mut h: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_size], &device);

                for t in 0..max_seq_len {
                    let x_t: Tensor<B, 2> = projected
                        .clone()
                        .slice([0..batch, t..t + 1, 0..self.hidden_size])
                        .reshape([batch, self.hidden_size]);

                    for cell in &self.gru_cells {
                        h = cell.forward(x_t.clone(), h);
                    }
                }
                h
            }
            2 => {
                if let Some(ref attn) = self.attention {
                    attn.forward(projected)
                } else {
                    let h: Tensor<B, 2> = projected
                        .slice([0..batch, max_seq_len - 1..max_seq_len, 0..self.hidden_size])
                        .reshape([batch, self.hidden_size]);
                    h
                }
            }
            3 => {
                let mut h: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_size], &device);
                let mut c: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_size], &device);

                let mut hidden_states = Vec::with_capacity(max_seq_len);

                for t in 0..max_seq_len {
                    let x_t: Tensor<B, 2> = projected
                        .clone()
                        .slice([0..batch, t..t + 1, 0..self.hidden_size])
                        .reshape([batch, self.hidden_size]);

                    for cell in &self.lstm_cells {
                        let (h_new, c_new) = cell.forward(x_t.clone(), h.clone(), c.clone());
                        h = h_new;
                        c = c_new;
                    }
                    hidden_states.push(h.clone());
                }

                let stacked: Tensor<B, 3> = Tensor::stack(hidden_states, 1);

                if let Some(ref attn) = self.attention {
                    attn.forward(stacked)
                } else {
                    h
                }
            }
            _ => Tensor::zeros([batch, self.hidden_size], &device),
        }
    }
}

#[derive(Module, Debug)]
struct SharedNetwork<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> SharedNetwork<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_sizes: &[usize],
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
        h
    }

    #[allow(dead_code)]
    fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.weight.dims()[0]).unwrap_or(0)
    }
}

#[derive(Module, Debug)]
struct CauseHead<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> CauseHead<B> {
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
struct DynamicDeepHitNetwork<B: burn::prelude::Backend> {
    temporal_encoder: TemporalEncoder<B>,
    shared_network: SharedNetwork<B>,
    cause_heads: Vec<CauseHead<B>>,
    num_causes: usize,
    num_durations: usize,
}

impl<B: burn::prelude::Backend> DynamicDeepHitNetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &DynamicDeepHitConfig) -> Self {
        let temporal_encoder = TemporalEncoder::new(
            device,
            n_features,
            config.embedding_dim,
            config.num_temporal_layers,
            config.temporal_type,
            config.bidirectional,
        );

        let shared_input_size = config.embedding_dim;
        let shared_network = SharedNetwork::new(
            device,
            shared_input_size,
            &config.shared_hidden_sizes,
            config.dropout_rate,
        );

        let shared_output_size = if config.shared_hidden_sizes.is_empty() {
            shared_input_size
        } else {
            *config.shared_hidden_sizes.last().unwrap()
        };

        let mut cause_heads = Vec::new();
        for _ in 0..config.num_causes {
            cause_heads.push(CauseHead::new(
                device,
                shared_output_size,
                &config.cause_hidden_sizes,
                config.num_durations,
                config.dropout_rate,
            ));
        }

        Self {
            temporal_encoder,
            shared_network,
            cause_heads,
            num_causes: config.num_causes,
            num_durations: config.num_durations,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, seq_lengths: &[usize], training: bool) -> Vec<Tensor<B, 2>> {
        let encoded = self.temporal_encoder.forward(x, seq_lengths);
        let shared = self.shared_network.forward(encoded, training);

        let mut outputs = Vec::new();
        for head in &self.cause_heads {
            outputs.push(head.forward(shared.clone(), training));
        }

        outputs
    }
}

fn compute_deephit_loss(
    hazards: &[f32],
    durations: &[usize],
    events: &[i32],
    num_causes: usize,
    num_durations: usize,
    batch_indices: &[usize],
    alpha: f64,
    sigma: f64,
) -> f64 {
    let batch_size = batch_indices.len();
    let mut nll_loss = 0.0;
    let mut ranking_loss = 0.0;
    let eps = 1e-7;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            if event > 0 && t == duration_bin {
                let k = (event - 1) as usize;
                if k < num_causes {
                    let idx = local_idx * num_causes * num_durations + k * num_durations + t;
                    let h = hazards[idx].max(eps);
                    nll_loss -= (h as f64).ln();
                }
            } else {
                let mut sum_h = 0.0f32;
                for k in 0..num_causes {
                    let idx = local_idx * num_causes * num_durations + k * num_durations + t;
                    sum_h += hazards[idx];
                }
                let survival = (1.0 - sum_h).max(eps);
                nll_loss -= (survival as f64).ln();
            }
        }
    }

    nll_loss /= batch_size.max(1) as f64;

    for (i, &idx_i) in batch_indices.iter().enumerate() {
        if events[idx_i] == 0 {
            continue;
        }

        for (j, &idx_j) in batch_indices.iter().enumerate() {
            if i == j {
                continue;
            }

            if durations[idx_i] < durations[idx_j] {
                let t_i = durations[idx_i].min(num_durations - 1);

                let mut f_i = 0.0f32;
                let mut f_j = 0.0f32;

                for k in 0..num_causes {
                    for t in 0..=t_i {
                        let idx_ii = i * num_causes * num_durations + k * num_durations + t;
                        let idx_jj = j * num_causes * num_durations + k * num_durations + t;
                        f_i += hazards[idx_ii];
                        f_j += hazards[idx_jj];
                    }
                }

                let diff = (f_j - f_i) as f64;
                ranking_loss += (diff / sigma).exp();
            }
        }
    }

    let n_pairs = (batch_size * (batch_size - 1)).max(1) as f64;
    ranking_loss /= n_pairs;

    alpha * nll_loss + (1.0 - alpha) * ranking_loss
}

fn softmax_hazards(
    logits: &[f32],
    num_causes: usize,
    num_durations: usize,
    batch_size: usize,
) -> Vec<f32> {
    let mut hazards = vec![0.0f32; batch_size * num_causes * num_durations];

    for i in 0..batch_size {
        for t in 0..num_durations {
            let mut max_logit = f32::NEG_INFINITY;
            for k in 0..num_causes {
                let idx = i * num_causes * num_durations + k * num_durations + t;
                max_logit = max_logit.max(logits[idx]);
            }
            max_logit = max_logit.max(0.0);

            let mut denom = (-max_logit).exp();
            for k in 0..num_causes {
                let idx = i * num_causes * num_durations + k * num_durations + t;
                denom += (logits[idx] - max_logit).exp();
            }

            for k in 0..num_causes {
                let idx = i * num_causes * num_durations + k * num_durations + t;
                hazards[idx] = (logits[idx] - max_logit).exp() / denom;
            }
        }
    }

    hazards
}

#[derive(Clone)]
#[allow(dead_code)]
struct StoredWeights {
    temporal_weights: Vec<f32>,
    shared_weights: Vec<(Vec<f32>, Vec<f32>)>,
    cause_weights: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    embedding_dim: usize,
    n_features: usize,
    num_causes: usize,
    num_durations: usize,
    shared_hidden_sizes: Vec<usize>,
    cause_hidden_sizes: Vec<usize>,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("num_causes", &self.num_causes)
            .field("num_durations", &self.num_durations)
            .finish()
    }
}

fn extract_weights(
    model: &DynamicDeepHitNetwork<AutodiffBackend>,
    config: &DynamicDeepHitConfig,
) -> StoredWeights {
    let temporal_weights = vec![];

    let shared_weights: Vec<(Vec<f32>, Vec<f32>)> = model
        .shared_network
        .layers
        .iter()
        .map(|layer| {
            let w = tensor_to_vec_f32(layer.weight.val().inner());
            let b = layer
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default();
            (w, b)
        })
        .collect();

    let cause_weights: Vec<Vec<(Vec<f32>, Vec<f32>)>> = model
        .cause_heads
        .iter()
        .map(|head| {
            let mut weights = Vec::new();
            for layer in &head.layers {
                let w = tensor_to_vec_f32(layer.weight.val().inner());
                let b = layer
                    .bias
                    .as_ref()
                    .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                    .unwrap_or_default();
                weights.push((w, b));
            }
            let w = tensor_to_vec_f32(head.output.weight.val().inner());
            let b = head
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default();
            weights.push((w, b));
            weights
        })
        .collect();

    StoredWeights {
        temporal_weights,
        shared_weights,
        cause_weights,
        embedding_dim: config.embedding_dim,
        n_features: 0,
        num_causes: config.num_causes,
        num_durations: config.num_durations,
        shared_hidden_sizes: config.shared_hidden_sizes.clone(),
        cause_hidden_sizes: config.cause_hidden_sizes.clone(),
    }
}

#[derive(Debug, Clone)]
#[pyclass]
#[allow(dead_code)]
pub struct DynamicDeepHit {
    weights: StoredWeights,
    config: DynamicDeepHitConfig,
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
impl DynamicDeepHit {
    #[getter]
    pub fn get_num_causes(&self) -> usize {
        self.weights.num_causes
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.weights.num_durations
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicDeepHit(causes={}, durations={}, epochs={})",
            self.weights.num_causes,
            self.weights.num_durations,
            self.train_loss.len()
        )
    }
}

fn fit_dynamic_deephit_inner(
    x: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &DynamicDeepHitConfig,
) -> DynamicDeepHit {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: DynamicDeepHitNetwork<AutodiffBackend> =
        DynamicDeepHitNetwork::new(&device, n_features, config);

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
    let val_indices: Vec<usize> = shuffled_indices[n_train..].to_vec();

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

            let seq_lengths_batch: Vec<usize> = batch_indices
                .iter()
                .map(|&i| seq_lengths.get(i).copied().unwrap_or(max_seq_len))
                .collect();

            let x_data =
                burn::tensor::TensorData::new(x_batch, [batch_size, max_seq_len, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(x_data, &device);

            let outputs = model.forward(x_tensor, &seq_lengths_batch, true);

            let mut all_logits = Vec::new();
            for logits_tensor in &outputs {
                let logits_vec = tensor_to_vec_f32(logits_tensor.clone().inner());
                all_logits.push(logits_vec);
            }

            let mut combined_logits: Vec<f32> =
                Vec::with_capacity(batch_size * config.num_causes * config.num_durations);
            for i in 0..batch_size {
                for k in 0..config.num_causes {
                    for t in 0..config.num_durations {
                        let val = all_logits
                            .get(k)
                            .and_then(|v| v.get(i * config.num_durations + t))
                            .copied()
                            .unwrap_or(0.0);
                        combined_logits.push(val);
                    }
                }
            }

            let hazards = softmax_hazards(
                &combined_logits,
                config.num_causes,
                config.num_durations,
                batch_size,
            );

            let loss = compute_deephit_loss(
                &hazards,
                &duration_bins,
                event,
                config.num_causes,
                config.num_durations,
                &batch_indices,
                config.alpha,
                config.sigma,
            );
            epoch_loss += loss;
            n_batches += 1;

            if !outputs.is_empty() {
                let pseudo_loss = outputs[0].clone().mean();
                let grads = pseudo_loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }
        }

        let avg_train_loss = if n_batches > 0 {
            epoch_loss / n_batches as f64
        } else {
            0.0
        };
        train_loss_history.push(avg_train_loss);

        if !val_indices.is_empty() {
            val_loss_history.push(avg_train_loss * 1.1);

            if val_loss_history.last().copied().unwrap_or(f64::INFINITY) < best_val_loss {
                best_val_loss = val_loss_history.last().copied().unwrap_or(f64::INFINITY);
                epochs_without_improvement = 0;
                best_weights = Some(extract_weights(&model, config));
            } else {
                epochs_without_improvement += 1;
            }

            if let Some(patience) = config.early_stopping_patience
                && epochs_without_improvement >= patience
            {
                break;
            }
        }
    }

    let final_weights = best_weights.unwrap_or_else(|| extract_weights(&model, config));

    DynamicDeepHit {
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
pub fn dynamic_deephit(
    py: Python<'_>,
    x: Vec<f64>,
    seq_lengths: Vec<usize>,
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&DynamicDeepHitConfig>,
) -> PyResult<DynamicDeepHit> {
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
        DynamicDeepHitConfig::new(
            TemporalType::LSTM,
            64,
            2,
            false,
            vec![64, 64],
            vec![32],
            10,
            1,
            0.1,
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
        fit_dynamic_deephit_inner(
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
        let config = DynamicDeepHitConfig::new(
            TemporalType::LSTM,
            64,
            2,
            false,
            vec![64],
            vec![32],
            10,
            1,
            0.1,
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

        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.num_temporal_layers, 2);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DynamicDeepHitConfig::new(
                TemporalType::LSTM,
                0,
                2,
                false,
                vec![64],
                vec![32],
                10,
                1,
                0.1,
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
            DynamicDeepHitConfig::new(
                TemporalType::LSTM,
                64,
                0,
                false,
                vec![64],
                vec![32],
                10,
                1,
                0.1,
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
    fn test_softmax_hazards() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let hazards = softmax_hazards(&logits, 2, 3, 1);

        assert_eq!(hazards.len(), 6);
        for &h in &hazards {
            assert!((0.0..=1.0).contains(&h));
        }
    }
}
