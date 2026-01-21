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
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{activation::softplus, backend::AutodiffBackend as AutodiffBackendTrait},
};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::utils::{
    compute_duration_bins, gelu_cpu, layer_norm_cpu, linear_forward, tensor_to_vec_f32,
};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

fn gelu<B: burn::prelude::Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sqrt_2 = (2.0_f32).sqrt();
    let cdf = (x.clone() / sqrt_2).erf().add_scalar(1.0) * 0.5;
    x * cdf
}

fn layer_norm<B: burn::prelude::Backend>(
    x: Tensor<B, 2>,
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 2> {
    let [batch, hidden] = x.dims();
    let mean = x.clone().mean_dim(1);
    let var = x.clone().var(1);
    let x_norm = (x - mean) / (var + eps).sqrt();
    let gamma_expanded: Tensor<B, 2> = gamma.reshape([1, hidden]);
    let beta_expanded: Tensor<B, 2> = beta.reshape([1, hidden]);
    x_norm * gamma_expanded + beta_expanded
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum SurvTraceActivation {
    GELU,
    ReLU,
}

#[pymethods]
impl SurvTraceActivation {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "gelu" => Ok(SurvTraceActivation::GELU),
            "relu" => Ok(SurvTraceActivation::ReLU),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown activation. Use 'gelu' or 'relu'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvTraceConfig {
    #[pyo3(get, set)]
    pub hidden_size: usize,
    #[pyo3(get, set)]
    pub num_hidden_layers: usize,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub intermediate_size: usize,
    #[pyo3(get, set)]
    pub hidden_dropout_prob: f64,
    #[pyo3(get, set)]
    pub attention_dropout_prob: f64,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_events: usize,
    #[pyo3(get, set)]
    pub vocab_size: usize,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub layer_norm_eps: f32,
}

#[pymethods]
impl SurvTraceConfig {
    #[new]
    #[pyo3(signature = (
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.1,
        num_durations=5,
        num_events=1,
        vocab_size=8,
        learning_rate=0.001,
        batch_size=64,
        n_epochs=100,
        weight_decay=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1,
        layer_norm_eps=1e-12
    ))]
    pub fn new(
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        intermediate_size: usize,
        hidden_dropout_prob: f64,
        attention_dropout_prob: f64,
        num_durations: usize,
        num_events: usize,
        vocab_size: usize,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        weight_decay: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        layer_norm_eps: f32,
    ) -> PyResult<Self> {
        if hidden_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_size must be positive",
            ));
        }
        if num_hidden_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_hidden_layers must be positive",
            ));
        }
        if num_attention_heads == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_attention_heads must be positive",
            ));
        }
        if !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hidden_size must be divisible by num_attention_heads",
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
        if n_epochs == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_epochs must be positive",
            ));
        }
        if !(0.0..1.0).contains(&validation_fraction) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "validation_fraction must be in [0, 1)",
            ));
        }

        Ok(SurvTraceConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_dropout_prob,
            attention_dropout_prob,
            num_durations,
            num_events,
            vocab_size,
            learning_rate,
            batch_size,
            n_epochs,
            weight_decay,
            seed,
            early_stopping_patience,
            validation_fraction,
            layer_norm_eps,
        })
    }
}

#[derive(Module, Debug)]
struct MultiHeadAttention<B: burn::prelude::Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
}

impl<B: burn::prelude::Backend> MultiHeadAttention<B> {
    fn new(device: &B::Device, hidden_size: usize, num_heads: usize, dropout_prob: f64) -> Self {
        let head_dim = hidden_size / num_heads;

        Self {
            query: LinearConfig::new(hidden_size, hidden_size).init(device),
            key: LinearConfig::new(hidden_size, hidden_size).init(device),
            value: LinearConfig::new(hidden_size, hidden_size).init(device),
            output: LinearConfig::new(hidden_size, hidden_size).init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
            num_heads,
            head_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let [batch_size, hidden_size] = x.dims();
        let seq_len = 1;

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let q = q
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn_weights = burn::tensor::activation::softmax(scores, 3);

        let attn_weights = if training {
            self.dropout.forward(attn_weights)
        } else {
            attn_weights
        };

        let context = attn_weights.matmul(v);
        let context = context.swap_dims(1, 2).reshape([batch_size, hidden_size]);

        self.output.forward(context)
    }
}

#[derive(Module, Debug)]
struct TransformerLayer<B: burn::prelude::Backend> {
    attention: MultiHeadAttention<B>,
    intermediate: Linear<B>,
    output_dense: Linear<B>,
    layer_norm1_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm1_beta: burn::module::Param<Tensor<B, 1>>,
    layer_norm2_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm2_beta: burn::module::Param<Tensor<B, 1>>,
    dropout: Dropout,
    layer_norm_eps: f32,
}

impl<B: burn::prelude::Backend> TransformerLayer<B> {
    fn new(
        device: &B::Device,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        hidden_dropout_prob: f64,
        attention_dropout_prob: f64,
        layer_norm_eps: f32,
    ) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                device,
                hidden_size,
                num_heads,
                attention_dropout_prob,
            ),
            intermediate: LinearConfig::new(hidden_size, intermediate_size).init(device),
            output_dense: LinearConfig::new(intermediate_size, hidden_size).init(device),
            layer_norm1_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [hidden_size],
                device,
            )),
            layer_norm1_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [hidden_size],
                device,
            )),
            layer_norm2_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [hidden_size],
                device,
            )),
            layer_norm2_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [hidden_size],
                device,
            )),
            dropout: DropoutConfig::new(hidden_dropout_prob).init(),
            layer_norm_eps,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let attn_output = self.attention.forward(x.clone(), training);
        let attn_output = if training {
            self.dropout.forward(attn_output)
        } else {
            attn_output
        };
        let x = layer_norm(
            x + attn_output,
            self.layer_norm1_gamma.val(),
            self.layer_norm1_beta.val(),
            self.layer_norm_eps,
        );

        let intermediate = self.intermediate.forward(x.clone());
        let intermediate = gelu(intermediate);
        let output = self.output_dense.forward(intermediate);
        let output = if training {
            self.dropout.forward(output)
        } else {
            output
        };

        layer_norm(
            x + output,
            self.layer_norm2_gamma.val(),
            self.layer_norm2_beta.val(),
            self.layer_norm_eps,
        )
    }
}

#[derive(Module, Debug)]
struct SurvTraceNetwork<B: burn::prelude::Backend> {
    cat_embeddings: Vec<Embedding<B>>,
    num_projection: Linear<B>,
    transformer_layers: Vec<TransformerLayer<B>>,
    output_heads: Vec<Linear<B>>,
    hidden_size: usize,
    num_cat_features: usize,
    num_num_features: usize,
    num_events: usize,
    num_durations: usize,
}

impl<B: burn::prelude::Backend> SurvTraceNetwork<B> {
    fn new(
        device: &B::Device,
        num_cat_features: usize,
        num_num_features: usize,
        cat_cardinalities: &[usize],
        config: &SurvTraceConfig,
    ) -> Self {
        let mut cat_embeddings = Vec::new();
        for &card in cat_cardinalities {
            cat_embeddings.push(EmbeddingConfig::new(card.max(2), config.hidden_size).init(device));
        }

        let num_projection = if num_num_features > 0 {
            LinearConfig::new(num_num_features, config.hidden_size).init(device)
        } else {
            LinearConfig::new(1, config.hidden_size).init(device)
        };

        let mut transformer_layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            transformer_layers.push(TransformerLayer::new(
                device,
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout_prob,
                config.attention_dropout_prob,
                config.layer_norm_eps,
            ));
        }

        let mut output_heads = Vec::new();
        let num_events = config.num_events.max(1);
        for _ in 0..num_events {
            output_heads
                .push(LinearConfig::new(config.hidden_size, config.num_durations).init(device));
        }

        Self {
            cat_embeddings,
            num_projection,
            transformer_layers,
            output_heads,
            hidden_size: config.hidden_size,
            num_cat_features,
            num_num_features,
            num_events,
            num_durations: config.num_durations,
        }
    }

    fn forward(
        &self,
        x_cat: Option<Tensor<B, 2, Int>>,
        x_num: Tensor<B, 2>,
        training: bool,
    ) -> Vec<Tensor<B, 2>> {
        let [batch_size, _] = x_num.dims();
        let device = x_num.device();

        let mut embeddings: Tensor<B, 2> = Tensor::zeros([batch_size, self.hidden_size], &device);

        if let Some(x_cat) = x_cat {
            for (i, emb) in self.cat_embeddings.iter().enumerate() {
                let cat_slice: Tensor<B, 2, Int> = x_cat.clone().slice([0..batch_size, i..i + 1]);
                let cat_emb_3d: Tensor<B, 3> = emb.forward(cat_slice);
                let cat_emb: Tensor<B, 2> = cat_emb_3d.squeeze::<2>();
                embeddings = embeddings + cat_emb;
            }
        }

        if self.num_num_features > 0 {
            let num_emb = self.num_projection.forward(x_num);
            embeddings = embeddings + num_emb;
        }

        let mut hidden = embeddings;
        for layer in &self.transformer_layers {
            hidden = layer.forward(hidden, training);
        }

        let mut outputs = Vec::new();
        for head in &self.output_heads {
            let logits = head.forward(hidden.clone());
            outputs.push(logits);
        }

        outputs
    }

    fn forward_inference(
        &self,
        x_cat: Option<Tensor<B, 2, Int>>,
        x_num: Tensor<B, 2>,
    ) -> Vec<Tensor<B, 2>> {
        self.forward(x_cat, x_num, false)
    }
}

fn compute_nll_logistic_hazard_loss(
    logits: &[f32],
    durations: &[usize],
    events: &[i32],
    num_durations: usize,
    batch_indices: &[usize],
) -> f64 {
    let mut total_loss = 0.0;
    let mut n_events = 0;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            let logit = logits[local_idx * num_durations + t];
            let target = if t == duration_bin && event == 1 {
                1.0
            } else {
                0.0
            };

            let loss = if target > 0.5 {
                (1.0 + (-logit).exp()).ln()
            } else {
                logit + (1.0 + (-logit).exp()).ln()
            };
            total_loss += loss as f64;
        }

        if event == 1 {
            n_events += 1;
        }
    }

    if n_events > 0 {
        total_loss / n_events as f64
    } else {
        total_loss / batch_indices.len().max(1) as f64
    }
}

fn compute_nll_logistic_hazard_gradient(
    logits: &[f32],
    durations: &[usize],
    events: &[i32],
    num_durations: usize,
    batch_indices: &[usize],
) -> Vec<f32> {
    let batch_size = batch_indices.len();
    let mut gradients = vec![0.0f32; batch_size * num_durations];

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            let logit = logits[local_idx * num_durations + t];
            let pred = 1.0 / (1.0 + (-logit).exp());
            let target = if t == duration_bin && event == 1 {
                1.0
            } else {
                0.0
            };
            gradients[local_idx * num_durations + t] = pred - target;
        }
    }

    let n_events: i32 = batch_indices.iter().map(|&i| events[i]).sum();
    let divisor = if n_events > 0 {
        n_events as f32
    } else {
        batch_size.max(1) as f32
    };

    for g in &mut gradients {
        *g /= divisor;
    }

    gradients
}

#[derive(Clone)]
struct StoredWeights {
    cat_embeddings: Vec<Vec<f32>>,
    cat_embedding_dims: Vec<(usize, usize)>,
    num_projection_weights: Vec<f32>,
    num_projection_bias: Vec<f32>,
    num_projection_dims: (usize, usize),
    transformer_layers: Vec<TransformerLayerWeights>,
    output_heads: Vec<(Vec<f32>, Vec<f32>, usize, usize)>,
    hidden_size: usize,
    num_cat_features: usize,
    num_num_features: usize,
    num_events: usize,
}

#[derive(Clone)]
struct TransformerLayerWeights {
    query_w: Vec<f32>,
    query_b: Vec<f32>,
    key_w: Vec<f32>,
    key_b: Vec<f32>,
    value_w: Vec<f32>,
    value_b: Vec<f32>,
    output_w: Vec<f32>,
    output_b: Vec<f32>,
    intermediate_w: Vec<f32>,
    intermediate_b: Vec<f32>,
    output_dense_w: Vec<f32>,
    output_dense_b: Vec<f32>,
    ln1_gamma: Vec<f32>,
    ln1_beta: Vec<f32>,
    ln2_gamma: Vec<f32>,
    ln2_beta: Vec<f32>,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("num_transformer_layers", &self.transformer_layers.len())
            .field("num_events", &self.num_events)
            .finish()
    }
}

fn extract_weights(
    model: &SurvTraceNetwork<AutodiffBackend>,
    config: &SurvTraceConfig,
    cat_cardinalities: &[usize],
) -> StoredWeights {
    let mut cat_embeddings = Vec::new();
    let mut cat_embedding_dims = Vec::new();

    for (i, emb) in model.cat_embeddings.iter().enumerate() {
        let w: Vec<f32> = emb
            .weight
            .val()
            .inner()
            .into_data()
            .to_vec()
            .unwrap_or_default();
        cat_embeddings.push(w);
        cat_embedding_dims.push((
            cat_cardinalities.get(i).copied().unwrap_or(2),
            config.hidden_size,
        ));
    }

    let num_proj_w: Tensor<AutodiffBackend, 2> = model.num_projection.weight.val();
    let num_projection_weights: Vec<f32> = tensor_to_vec_f32(num_proj_w.inner());
    let num_projection_bias: Vec<f32> = model
        .num_projection
        .bias
        .as_ref()
        .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
        .unwrap_or_default();
    let num_projection_dims = (model.num_num_features.max(1), config.hidden_size);

    let mut transformer_layers = Vec::new();
    for layer in &model.transformer_layers {
        let tlw = TransformerLayerWeights {
            query_w: tensor_to_vec_f32(layer.attention.query.weight.val().inner()),
            query_b: layer
                .attention
                .query
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            key_w: tensor_to_vec_f32(layer.attention.key.weight.val().inner()),
            key_b: layer
                .attention
                .key
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            value_w: tensor_to_vec_f32(layer.attention.value.weight.val().inner()),
            value_b: layer
                .attention
                .value
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            output_w: tensor_to_vec_f32(layer.attention.output.weight.val().inner()),
            output_b: layer
                .attention
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            intermediate_w: tensor_to_vec_f32(layer.intermediate.weight.val().inner()),
            intermediate_b: layer
                .intermediate
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            output_dense_w: tensor_to_vec_f32(layer.output_dense.weight.val().inner()),
            output_dense_b: layer
                .output_dense
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            ln1_gamma: layer
                .layer_norm1_gamma
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln1_beta: layer
                .layer_norm1_beta
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln2_gamma: layer
                .layer_norm2_gamma
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln2_beta: layer
                .layer_norm2_beta
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_heads: config.num_attention_heads,
        };
        transformer_layers.push(tlw);
    }

    let mut output_heads = Vec::new();
    for head in &model.output_heads {
        let w: Vec<f32> = tensor_to_vec_f32(head.weight.val().inner());
        let b: Vec<f32> = head
            .bias
            .as_ref()
            .map(|bias| bias.val().inner().into_data().to_vec().unwrap_or_default())
            .unwrap_or_default();
        output_heads.push((w, b, config.hidden_size, config.num_durations));
    }

    StoredWeights {
        cat_embeddings,
        cat_embedding_dims,
        num_projection_weights,
        num_projection_bias,
        num_projection_dims,
        transformer_layers,
        output_heads,
        hidden_size: config.hidden_size,
        num_cat_features: model.num_cat_features,
        num_num_features: model.num_num_features,
        num_events: model.num_events,
    }
}

fn predict_with_weights(
    x_cat: Option<&[i64]>,
    x_num: &[f64],
    n: usize,
    weights: &StoredWeights,
    layer_norm_eps: f32,
) -> Vec<Vec<f64>> {
    let hidden_size = weights.hidden_size;
    let num_num = weights.num_num_features;
    let num_cat = weights.num_cat_features;

    let mut all_outputs: Vec<Vec<f64>> = vec![Vec::new(); weights.num_events];

    for i in 0..n {
        let mut hidden = vec![0.0f64; hidden_size];

        if let Some(cats) = x_cat {
            for (feat_idx, emb_weights) in weights.cat_embeddings.iter().enumerate() {
                let (vocab_size, emb_dim) = weights.cat_embedding_dims[feat_idx];
                let cat_val = cats[i * num_cat + feat_idx] as usize;
                let cat_val = cat_val.min(vocab_size - 1);
                for j in 0..emb_dim {
                    hidden[j] += emb_weights[cat_val * emb_dim + j] as f64;
                }
            }
        }

        if num_num > 0 {
            let (in_dim, out_dim) = weights.num_projection_dims;
            for j in 0..out_dim {
                let mut sum = if !weights.num_projection_bias.is_empty() {
                    weights.num_projection_bias[j] as f64
                } else {
                    0.0
                };
                for k in 0..in_dim.min(num_num) {
                    sum += x_num[i * num_num + k]
                        * weights.num_projection_weights[j * in_dim + k] as f64;
                }
                hidden[j] += sum;
            }
        }

        for layer in &weights.transformer_layers {
            hidden = apply_transformer_layer_cpu(&hidden, layer, layer_norm_eps);
        }

        for (event_idx, (w, b, in_dim, out_dim)) in weights.output_heads.iter().enumerate() {
            let mut logits = Vec::with_capacity(*out_dim);
            for j in 0..*out_dim {
                let mut sum = if !b.is_empty() { b[j] as f64 } else { 0.0 };
                for k in 0..*in_dim {
                    sum += hidden[k] * w[j * in_dim + k] as f64;
                }
                logits.push(sum);
            }
            all_outputs[event_idx].extend(logits);
        }
    }

    all_outputs
}

fn apply_transformer_layer_cpu(
    hidden: &[f64],
    layer: &TransformerLayerWeights,
    eps: f32,
) -> Vec<f64> {
    let h = layer.hidden_size;

    let q = linear_forward(hidden, &layer.query_w, &layer.query_b, h, h);
    let k = linear_forward(hidden, &layer.key_w, &layer.key_b, h, h);
    let v = linear_forward(hidden, &layer.value_w, &layer.value_b, h, h);

    let head_dim = h / layer.num_heads;
    let mut attn_output = vec![0.0f64; h];

    for head in 0..layer.num_heads {
        let start = head * head_dim;
        let end = start + head_dim;

        let mut score = 0.0;
        for i in start..end {
            score += q[i] * k[i];
        }
        score /= (head_dim as f64).sqrt();
        let attn_weight = 1.0;

        for i in start..end {
            attn_output[i] = attn_weight * v[i];
        }
    }

    let attn_proj = linear_forward(&attn_output, &layer.output_w, &layer.output_b, h, h);

    let mut residual1: Vec<f64> = hidden.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
    residual1 = layer_norm_cpu(&residual1, &layer.ln1_gamma, &layer.ln1_beta, eps);

    let intermediate = linear_forward(
        &residual1,
        &layer.intermediate_w,
        &layer.intermediate_b,
        h,
        layer.intermediate_size,
    );
    let intermediate: Vec<f64> = intermediate.iter().map(|&x| gelu_cpu(x)).collect();

    let output = linear_forward(
        &intermediate,
        &layer.output_dense_w,
        &layer.output_dense_b,
        layer.intermediate_size,
        h,
    );

    let mut residual2: Vec<f64> = residual1.iter().zip(&output).map(|(a, b)| a + b).collect();
    residual2 = layer_norm_cpu(&residual2, &layer.ln2_gamma, &layer.ln2_beta, eps);

    residual2
}

fn fit_survtrace_inner(
    x_cat: Option<&[i64]>,
    x_num: &[f64],
    n_obs: usize,
    num_cat_features: usize,
    num_num_features: usize,
    cat_cardinalities: &[usize],
    time: &[f64],
    event: &[i32],
    config: &SurvTraceConfig,
) -> SurvTrace {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: SurvTraceNetwork<AutodiffBackend> = SurvTraceNetwork::new(
        &device,
        num_cat_features,
        num_num_features,
        cat_cardinalities,
        config,
    );

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        )))
        .init();

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

            let x_num_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| {
                    (0..num_num_features).map(move |j| x_num[i * num_num_features + j] as f32)
                })
                .collect();

            let x_num_data = burn::tensor::TensorData::new(
                x_num_batch.clone(),
                [batch_size, num_num_features.max(1)],
            );
            let x_num_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_num_data, &device);

            let x_cat_tensor: Option<Tensor<AutodiffBackend, 2, Int>> = if num_cat_features > 0 {
                if let Some(cats) = x_cat {
                    let x_cat_batch: Vec<i64> = batch_indices
                        .iter()
                        .flat_map(|&i| {
                            (0..num_cat_features).map(move |j| cats[i * num_cat_features + j])
                        })
                        .collect();
                    let x_cat_data =
                        burn::tensor::TensorData::new(x_cat_batch, [batch_size, num_cat_features]);
                    Some(Tensor::from_data(x_cat_data, &device))
                } else {
                    None
                }
            } else {
                None
            };

            let outputs = model.forward(x_cat_tensor, x_num_tensor, true);

            let mut total_loss = 0.0;
            let mut all_grads: Vec<Vec<f32>> = Vec::new();

            for (event_idx, logits_tensor) in outputs.iter().enumerate() {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());

                let loss = compute_nll_logistic_hazard_loss(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &batch_indices,
                );
                total_loss += loss;

                let grads = compute_nll_logistic_hazard_gradient(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &batch_indices,
                );
                all_grads.push(grads);
            }

            epoch_loss += total_loss;
            n_batches += 1;

            if !all_grads.is_empty() {
                let grad_data = burn::tensor::TensorData::new(
                    all_grads[0].clone(),
                    [batch_size, config.num_durations],
                );
                let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

                let pseudo_loss = (outputs[0].clone() * grad_tensor).mean();
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
            let x_num_val: Vec<f32> = val_indices
                .iter()
                .flat_map(|&i| {
                    (0..num_num_features).map(move |j| x_num[i * num_num_features + j] as f32)
                })
                .collect();

            let x_num_val_data =
                burn::tensor::TensorData::new(x_num_val, [n_val, num_num_features.max(1)]);
            let x_num_val_tensor: Tensor<AutodiffBackend, 2> =
                Tensor::from_data(x_num_val_data, &device);

            let x_cat_val_tensor: Option<Tensor<AutodiffBackend, 2, Int>> = if num_cat_features > 0
            {
                if let Some(cats) = x_cat {
                    let x_cat_val: Vec<i64> = val_indices
                        .iter()
                        .flat_map(|&i| {
                            (0..num_cat_features).map(move |j| cats[i * num_cat_features + j])
                        })
                        .collect();
                    let x_cat_val_data =
                        burn::tensor::TensorData::new(x_cat_val, [n_val, num_cat_features]);
                    Some(Tensor::from_data(x_cat_val_data, &device))
                } else {
                    None
                }
            } else {
                None
            };

            let val_outputs = model.forward_inference(x_cat_val_tensor, x_num_val_tensor);

            let mut val_loss = 0.0;
            for logits_tensor in &val_outputs {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());
                val_loss += compute_nll_logistic_hazard_loss(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &val_indices,
                );
            }
            val_loss_history.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
                best_weights = Some(extract_weights(&model, config, cat_cardinalities));
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

    let final_weights =
        best_weights.unwrap_or_else(|| extract_weights(&model, config, cat_cardinalities));

    SurvTrace {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        cat_cardinalities: cat_cardinalities.to_vec(),
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvTrace {
    weights: StoredWeights,
    config: SurvTraceConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    #[pyo3(get)]
    pub cat_cardinalities: Vec<usize>,
}

#[pymethods]
impl SurvTrace {
    #[staticmethod]
    #[pyo3(signature = (x_cat, x_num, n_obs, num_cat_features, num_num_features, cat_cardinalities, time, event, config))]
    pub fn fit(
        py: Python<'_>,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_obs: usize,
        num_cat_features: usize,
        num_num_features: usize,
        cat_cardinalities: Vec<usize>,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &SurvTraceConfig,
    ) -> PyResult<Self> {
        if x_num.len() != n_obs * num_num_features.max(1) && num_num_features > 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_num length must equal n_obs * num_num_features",
            ));
        }
        if time.len() != n_obs || event.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and event must have length n_obs",
            ));
        }
        if let Some(ref cats) = x_cat
            && cats.len() != n_obs * num_cat_features
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_cat length must equal n_obs * num_cat_features",
            ));
        }

        let config = config.clone();
        let x_cat_clone = x_cat.clone();
        Ok(py.detach(move || {
            fit_survtrace_inner(
                x_cat_clone.as_deref(),
                &x_num,
                n_obs,
                num_cat_features,
                num_num_features,
                &cat_cardinalities,
                &time,
                &event,
                &config,
            )
        }))
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_hazard(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            x_cat.as_deref(),
            &x_num,
            n_new,
            &self.weights,
            self.config.layer_norm_eps,
        );

        if event_idx >= outputs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "event_idx out of range",
            ));
        }

        let logits = &outputs[event_idx];
        let num_durations = self.config.num_durations;

        let hazards: Vec<Vec<f64>> = (0..n_new)
            .map(|i| {
                (0..num_durations)
                    .map(|t| {
                        let logit = logits[i * num_durations + t];
                        1.0 / (1.0 + (-logit).exp())
                    })
                    .collect()
            })
            .collect();

        Ok(hazards)
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_survival(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let hazards = self.predict_hazard(x_cat, x_num, n_new, event_idx)?;

        let survival: Vec<Vec<f64>> = hazards
            .par_iter()
            .map(|h| {
                let mut surv = Vec::with_capacity(h.len());
                let mut cum_surv = 1.0;
                for &haz in h {
                    cum_surv *= 1.0 - haz;
                    surv.push(cum_surv);
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_risk(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<f64>> {
        let survival = self.predict_survival(x_cat, x_num, n_new, event_idx)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| {
                let final_surv = s.last().copied().unwrap_or(1.0);
                1.0 - final_surv
            })
            .collect();

        Ok(risks)
    }

    #[pyo3(signature = (x_cat, x_num, n_new))]
    pub fn predict_cumulative_incidence(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let num_events = self.weights.num_events;
        let num_durations = self.config.num_durations;

        let outputs = predict_with_weights(
            x_cat.as_deref(),
            &x_num,
            n_new,
            &self.weights,
            self.config.layer_norm_eps,
        );

        let mut all_hazards: Vec<Vec<Vec<f64>>> = Vec::new();
        for event_idx in 0..num_events {
            let logits = &outputs[event_idx];
            let hazards: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    (0..num_durations)
                        .map(|t| {
                            let logit = logits[i * num_durations + t];
                            1.0 / (1.0 + (-logit).exp())
                        })
                        .collect()
                })
                .collect();
            all_hazards.push(hazards);
        }

        let cifs: Vec<Vec<Vec<f64>>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut overall_surv = vec![1.0; num_durations + 1];
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for event_idx in 0..num_events {
                        total_haz += all_hazards[event_idx][i][t];
                    }
                    overall_surv[t + 1] = overall_surv[t] * (1.0 - total_haz.min(1.0));
                }

                let mut event_cifs = Vec::new();
                for event_idx in 0..num_events {
                    let mut cif = Vec::with_capacity(num_durations);
                    let mut cum_inc = 0.0;
                    for t in 0..num_durations {
                        cum_inc += overall_surv[t] * all_hazards[event_idx][i][t];
                        cif.push(cum_inc);
                    }
                    event_cifs.push(cif);
                }
                event_cifs
            })
            .collect();

        Ok(cifs)
    }

    #[getter]
    pub fn get_num_events(&self) -> usize {
        self.weights.num_events
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.config.num_durations
    }

    #[getter]
    pub fn get_hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[getter]
    pub fn get_num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

#[pyfunction]
#[pyo3(signature = (x_cat, x_num, n_obs, num_cat_features, num_num_features, cat_cardinalities, time, event, config=None))]
pub fn survtrace(
    py: Python<'_>,
    x_cat: Option<Vec<i64>>,
    x_num: Vec<f64>,
    n_obs: usize,
    num_cat_features: usize,
    num_num_features: usize,
    cat_cardinalities: Vec<usize>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&SurvTraceConfig>,
) -> PyResult<SurvTrace> {
    let cfg = config.cloned().unwrap_or_else(|| {
        SurvTraceConfig::new(
            16, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12,
        )
        .unwrap()
    });

    SurvTrace::fit(
        py,
        x_cat,
        x_num,
        n_obs,
        num_cat_features,
        num_num_features,
        cat_cardinalities,
        time,
        event,
        &cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SurvTraceConfig::new(
            16,
            3,
            2,
            64,
            0.0,
            0.1,
            5,
            1,
            8,
            0.001,
            64,
            100,
            0.0001,
            Some(42),
            Some(5),
            0.1,
            1e-12,
        )
        .unwrap();
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.num_hidden_layers, 3);
        assert_eq!(config.num_attention_heads, 2);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            SurvTraceConfig::new(
                0, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
        assert!(
            SurvTraceConfig::new(
                15, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
        assert!(
            SurvTraceConfig::new(
                16, 0, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
    }

    #[test]
    fn test_survtrace_basic() {
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 1,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(None, &x_num, 6, 0, 2, &[], &time, &event, &config);
        assert_eq!(model.get_num_events(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_survtrace_with_categorical() {
        let x_cat = vec![0i64, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1];
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];
        let cat_cardinalities = vec![2, 2];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 1,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(
            Some(&x_cat),
            &x_num,
            6,
            2,
            2,
            &cat_cardinalities,
            &time,
            &event,
            &config,
        );
        assert_eq!(model.get_num_events(), 1);
    }

    #[test]
    fn test_survtrace_competing_risks() {
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 2,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(None, &x_num, 6, 0, 2, &[], &time, &event, &config);
        assert_eq!(model.get_num_events(), 2);
    }

    #[test]
    fn test_duration_bins() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (bins, cuts) = compute_duration_bins(&times, 5);

        assert_eq!(bins.len(), 10);
        assert_eq!(cuts.len(), 6);

        for &bin in &bins {
            assert!(bin < 5);
        }
    }

    #[test]
    fn test_nll_loss() {
        let logits = vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4];
        let durations = vec![1, 0, 2];
        let events = vec![1, 0, 1];
        let indices: Vec<usize> = vec![0, 1, 2];

        let loss = compute_nll_logistic_hazard_loss(&logits, &durations, &events, 2, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_gelu_cpu() {
        let x = 0.5;
        let result = gelu_cpu(x);
        assert!(result > 0.0);
        assert!(result < x);
    }

    #[test]
    fn test_layer_norm_cpu() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
        let beta = vec![0.0f32, 0.0, 0.0, 0.0];

        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-12);

        assert_eq!(result.len(), 4);
        let mean: f64 = result.iter().sum::<f64>() / 4.0;
        assert!((mean).abs() < 1e-6);
    }
}
