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

fn layer_norm_3d<B: burn::prelude::Backend>(
    x: Tensor<B, 3>,
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let [batch, seq, hidden] = x.dims();
    let x_2d: Tensor<B, 2> = x.reshape([batch * seq, hidden]);
    let normed = layer_norm(x_2d, gamma, beta, eps);
    normed.reshape([batch, seq, hidden])
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TracerConfig {
    #[pyo3(get, set)]
    pub embedding_dim: usize,
    #[pyo3(get, set)]
    pub num_factorized_layers: usize,
    #[pyo3(get, set)]
    pub num_attention_heads: usize,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_events: usize,
    #[pyo3(get, set)]
    pub mlp_hidden_size: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub layer_norm_eps: f32,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl TracerConfig {
    #[new]
    #[pyo3(signature = (
        embedding_dim=32,
        num_factorized_layers=2,
        num_attention_heads=4,
        num_durations=10,
        num_events=1,
        mlp_hidden_size=64,
        dropout_rate=0.1,
        learning_rate=0.0001,
        weight_decay=0.00001,
        batch_size=64,
        n_epochs=100,
        early_stopping_patience=None,
        validation_fraction=0.1,
        layer_norm_eps=1e-12,
        seed=None
    ))]
    pub fn new(
        embedding_dim: usize,
        num_factorized_layers: usize,
        num_attention_heads: usize,
        num_durations: usize,
        num_events: usize,
        mlp_hidden_size: usize,
        dropout_rate: f64,
        learning_rate: f64,
        weight_decay: f64,
        batch_size: usize,
        n_epochs: usize,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        layer_norm_eps: f32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if embedding_dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "embedding_dim must be positive",
            ));
        }
        if num_factorized_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_factorized_layers must be positive",
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
        if !(0.0..1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be in [0, 1)",
            ));
        }

        Ok(TracerConfig {
            embedding_dim,
            num_factorized_layers,
            num_attention_heads,
            num_durations,
            num_events,
            mlp_hidden_size,
            dropout_rate,
            learning_rate,
            weight_decay,
            batch_size,
            n_epochs,
            early_stopping_patience,
            validation_fraction,
            layer_norm_eps,
            seed,
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

    fn forward_3d(&self, x: Tensor<B, 3>, training: bool) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = x.dims();
        let device = x.device();

        let x_2d: Tensor<B, 2> = x.clone().reshape([batch_size * seq_len, hidden_size]);

        let q = self.query.forward(x_2d.clone());
        let k = self.key.forward(x_2d.clone());
        let v = self.value.forward(x_2d);

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
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size * seq_len, hidden_size]);

        let output = self.output.forward(context);
        output.reshape([batch_size, seq_len, hidden_size])
    }
}

#[derive(Module, Debug)]
struct TimeAwareEmbedding<B: burn::prelude::Backend> {
    feature_projections: Vec<Linear<B>>,
    decay_rates: burn::module::Param<Tensor<B, 1>>,
    missing_embeddings: burn::module::Param<Tensor<B, 2>>,
    embedding_dim: usize,
    n_features: usize,
}

impl<B: burn::prelude::Backend> TimeAwareEmbedding<B> {
    fn new(device: &B::Device, n_features: usize, embedding_dim: usize) -> Self {
        let mut feature_projections = Vec::new();
        for _ in 0..n_features {
            feature_projections.push(LinearConfig::new(1, embedding_dim).init(device));
        }

        let decay_rates =
            burn::module::Param::from_tensor(Tensor::ones([n_features], device) * 0.1);

        let missing_embeddings = burn::module::Param::from_tensor(Tensor::random(
            [n_features, embedding_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        ));

        Self {
            feature_projections,
            decay_rates,
            missing_embeddings,
            embedding_dim,
            n_features,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        time_delta: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch_size, max_seq_len, n_features] = x.dims();
        let device = x.device();

        let mut embedded: Tensor<B, 4> = Tensor::zeros(
            [batch_size, max_seq_len, n_features, self.embedding_dim],
            &device,
        );

        for f in 0..self.n_features {
            let x_f: Tensor<B, 3> = x.clone().slice([0..batch_size, 0..max_seq_len, f..f + 1]);
            let x_f_2d: Tensor<B, 2> = x_f.reshape([batch_size * max_seq_len, 1]);

            let proj = self.feature_projections[f].forward(x_f_2d);
            let proj_3d: Tensor<B, 3> = proj.reshape([batch_size, max_seq_len, self.embedding_dim]);

            let mask_f: Tensor<B, 2> = mask
                .clone()
                .slice([0..batch_size, 0..max_seq_len, f..f + 1])
                .reshape([batch_size, max_seq_len]);
            let time_delta_f: Tensor<B, 2> = time_delta
                .clone()
                .slice([0..batch_size, 0..max_seq_len, f..f + 1])
                .reshape([batch_size, max_seq_len]);

            let decay_rate_f: Tensor<B, 1> = self.decay_rates.val().slice([f..f + 1]);
            let decay_rate_expanded: Tensor<B, 2> = decay_rate_f
                .reshape([1, 1])
                .expand([batch_size, max_seq_len]);

            let decay: Tensor<B, 2> = (time_delta_f.neg() * decay_rate_expanded).exp();
            let decay_3d: Tensor<B, 3> = decay.reshape([batch_size, max_seq_len, 1]);

            let decayed_proj = proj_3d * decay_3d;

            let missing_emb_f: Tensor<B, 1> = self
                .missing_embeddings
                .val()
                .slice([f..f + 1, 0..self.embedding_dim])
                .reshape([self.embedding_dim]);
            let missing_emb_3d: Tensor<B, 3> = missing_emb_f
                .reshape([1, 1, self.embedding_dim])
                .expand([batch_size, max_seq_len, self.embedding_dim]);

            let mask_3d: Tensor<B, 3> = mask_f.reshape([batch_size, max_seq_len, 1]).expand([
                batch_size,
                max_seq_len,
                self.embedding_dim,
            ]);
            let mask_inv: Tensor<B, 3> = mask_3d.clone().neg().add_scalar(1.0);

            let feature_emb: Tensor<B, 3> = decayed_proj * mask_3d + missing_emb_3d * mask_inv;

            for b in 0..batch_size {
                for t in 0..max_seq_len {
                    for d in 0..self.embedding_dim {
                        let val_tensor = feature_emb.clone().slice([b..b + 1, t..t + 1, d..d + 1]);
                        let val_data = val_tensor.into_data();
                        let val_vec: Vec<f32> = val_data.to_vec().unwrap_or_default();
                        if !val_vec.is_empty() {
                            let current =
                                embedded
                                    .clone()
                                    .slice([b..b + 1, t..t + 1, f..f + 1, d..d + 1]);
                            let update_data = burn::tensor::TensorData::new(val_vec, [1, 1, 1, 1]);
                            let update_tensor: Tensor<B, 4> =
                                Tensor::from_data(update_data, &device);
                            embedded = embedded.slice_assign(
                                [b..b + 1, t..t + 1, f..f + 1, d..d + 1],
                                update_tensor,
                            );
                        }
                    }
                }
            }
        }

        let summed_4d: Tensor<B, 4> = embedded.sum_dim(2);
        let [b, s, _, e] = summed_4d.dims();
        let summed: Tensor<B, 3> = summed_4d.reshape([b, s, e]);
        summed
    }
}

#[derive(Module, Debug)]
struct FactorizedAttentionBlock<B: burn::prelude::Backend> {
    temporal_attention: MultiHeadAttention<B>,
    covariate_attention: MultiHeadAttention<B>,
    layer_norm_time_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm_time_beta: burn::module::Param<Tensor<B, 1>>,
    layer_norm_cov_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm_cov_beta: burn::module::Param<Tensor<B, 1>>,
    ffn_linear1: Linear<B>,
    ffn_linear2: Linear<B>,
    layer_norm_ffn_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm_ffn_beta: burn::module::Param<Tensor<B, 1>>,
    dropout: Dropout,
    layer_norm_eps: f32,
    embedding_dim: usize,
}

impl<B: burn::prelude::Backend> FactorizedAttentionBlock<B> {
    fn new(
        device: &B::Device,
        embedding_dim: usize,
        num_heads: usize,
        dropout_prob: f64,
        layer_norm_eps: f32,
    ) -> Self {
        let ffn_hidden = embedding_dim * 4;

        Self {
            temporal_attention: MultiHeadAttention::new(
                device,
                embedding_dim,
                num_heads,
                dropout_prob,
            ),
            covariate_attention: MultiHeadAttention::new(
                device,
                embedding_dim,
                num_heads,
                dropout_prob,
            ),
            layer_norm_time_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [embedding_dim],
                device,
            )),
            layer_norm_time_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [embedding_dim],
                device,
            )),
            layer_norm_cov_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [embedding_dim],
                device,
            )),
            layer_norm_cov_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [embedding_dim],
                device,
            )),
            ffn_linear1: LinearConfig::new(embedding_dim, ffn_hidden).init(device),
            ffn_linear2: LinearConfig::new(ffn_hidden, embedding_dim).init(device),
            layer_norm_ffn_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [embedding_dim],
                device,
            )),
            layer_norm_ffn_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [embedding_dim],
                device,
            )),
            dropout: DropoutConfig::new(dropout_prob).init(),
            layer_norm_eps,
            embedding_dim,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, training: bool) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden] = x.dims();

        let time_attn = self.temporal_attention.forward_3d(x.clone(), training);
        let time_attn = if training {
            self.dropout.forward(time_attn)
        } else {
            time_attn
        };
        let x_post_time = layer_norm_3d(
            x + time_attn,
            self.layer_norm_time_gamma.val(),
            self.layer_norm_time_beta.val(),
            self.layer_norm_eps,
        );

        let cov_attn = self
            .covariate_attention
            .forward_3d(x_post_time.clone(), training);
        let cov_attn = if training {
            self.dropout.forward(cov_attn)
        } else {
            cov_attn
        };
        let x_post_cov = layer_norm_3d(
            x_post_time + cov_attn,
            self.layer_norm_cov_gamma.val(),
            self.layer_norm_cov_beta.val(),
            self.layer_norm_eps,
        );

        let device = x_post_cov.device();
        let x_2d: Tensor<B, 2> = x_post_cov.clone().reshape([batch_size * seq_len, hidden]);
        let ffn_out = self.ffn_linear1.forward(x_2d);
        let ffn_out = gelu(ffn_out);
        let ffn_out = self.ffn_linear2.forward(ffn_out);
        let ffn_out: Tensor<B, 3> = ffn_out.reshape([batch_size, seq_len, hidden]);
        let ffn_out = if training {
            self.dropout.forward(ffn_out)
        } else {
            ffn_out
        };

        layer_norm_3d(
            x_post_cov + ffn_out,
            self.layer_norm_ffn_gamma.val(),
            self.layer_norm_ffn_beta.val(),
            self.layer_norm_eps,
        )
    }
}

#[derive(Module, Debug)]
struct CauseSpecificHead<B: burn::prelude::Backend> {
    mlp_layer1: Linear<B>,
    mlp_layer2: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::prelude::Backend> CauseSpecificHead<B> {
    fn new(
        device: &B::Device,
        input_dim: usize,
        hidden_dim: usize,
        num_durations: usize,
        dropout_prob: f64,
    ) -> Self {
        Self {
            mlp_layer1: LinearConfig::new(input_dim, hidden_dim).init(device),
            mlp_layer2: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            output: LinearConfig::new(hidden_dim, num_durations).init(device),
            dropout: DropoutConfig::new(dropout_prob).init(),
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let h = self.mlp_layer1.forward(x);
        let h = relu(h);
        let h = if training { self.dropout.forward(h) } else { h };
        let h = self.mlp_layer2.forward(h);
        let h = relu(h);
        let h = if training { self.dropout.forward(h) } else { h };
        self.output.forward(h)
    }
}

#[derive(Module, Debug)]
struct TracerNetwork<B: burn::prelude::Backend> {
    time_aware_embedding: TimeAwareEmbedding<B>,
    factorized_layers: Vec<FactorizedAttentionBlock<B>>,
    cause_specific_heads: Vec<CauseSpecificHead<B>>,
    embedding_dim: usize,
    n_features: usize,
    num_events: usize,
    num_durations: usize,
}

impl<B: burn::prelude::Backend> TracerNetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &TracerConfig) -> Self {
        let time_aware_embedding =
            TimeAwareEmbedding::new(device, n_features, config.embedding_dim);

        let mut factorized_layers = Vec::new();
        for _ in 0..config.num_factorized_layers {
            factorized_layers.push(FactorizedAttentionBlock::new(
                device,
                config.embedding_dim,
                config.num_attention_heads,
                config.dropout_rate,
                config.layer_norm_eps,
            ));
        }

        let mut cause_specific_heads = Vec::new();
        let num_events = config.num_events.max(1);
        for _ in 0..num_events {
            cause_specific_heads.push(CauseSpecificHead::new(
                device,
                config.embedding_dim,
                config.mlp_hidden_size,
                config.num_durations,
                config.dropout_rate,
            ));
        }

        Self {
            time_aware_embedding,
            factorized_layers,
            cause_specific_heads,
            embedding_dim: config.embedding_dim,
            n_features,
            num_events,
            num_durations: config.num_durations,
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        time_delta: Tensor<B, 3>,
        seq_lengths: &[usize],
        training: bool,
    ) -> Vec<Tensor<B, 2>> {
        let [batch_size, max_seq_len, _] = x.dims();
        let device = x.device();

        let embedded = self.time_aware_embedding.forward(x, mask, time_delta);

        let mut hidden = embedded;
        for layer in &self.factorized_layers {
            hidden = layer.forward(hidden, training);
        }

        let mut pooled = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq_len = seq_lengths
                .get(i)
                .copied()
                .unwrap_or(max_seq_len)
                .min(max_seq_len);
            if seq_len == 0 {
                pooled.push(vec![0.0f32; self.embedding_dim]);
            } else {
                let seq_slice: Tensor<B, 2> = hidden
                    .clone()
                    .slice([i..i + 1, 0..seq_len, 0..self.embedding_dim])
                    .reshape([seq_len, self.embedding_dim]);
                let mean_pooled: Tensor<B, 1> = seq_slice.mean_dim(0).reshape([self.embedding_dim]);
                let data = mean_pooled.into_data();
                let vec: Vec<f32> = data
                    .to_vec()
                    .unwrap_or_else(|_| vec![0.0; self.embedding_dim]);
                pooled.push(vec);
            }
        }

        let pooled_flat: Vec<f32> = pooled.into_iter().flatten().collect();
        let pooled_data =
            burn::tensor::TensorData::new(pooled_flat, [batch_size, self.embedding_dim]);
        let pooled_tensor: Tensor<B, 2> = Tensor::from_data(pooled_data, &device);

        let mut outputs = Vec::new();
        for head in &self.cause_specific_heads {
            let logits = head.forward(pooled_tensor.clone(), training);
            outputs.push(logits);
        }

        outputs
    }

    #[allow(dead_code)]
    fn forward_inference(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        time_delta: Tensor<B, 3>,
        seq_lengths: &[usize],
    ) -> Vec<Tensor<B, 2>> {
        self.forward(x, mask, time_delta, seq_lengths, false)
    }
}

fn compute_event_weights(events: &[i32], num_events: usize) -> Vec<f64> {
    let mut counts = vec![0usize; num_events + 1];
    for &e in events {
        let idx = (e as usize).min(num_events);
        counts[idx] += 1;
    }

    let total = events.len() as f64;
    let mut weights = vec![1.0; num_events];

    for k in 0..num_events {
        let count = counts[k + 1];
        if count > 0 {
            weights[k] = (total / count as f64).ln().max(1.0);
        }
    }

    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum / num_events as f64;
    }

    weights
}

fn multinomial_hazard_normalization(
    logits: &[f32],
    num_events: usize,
    num_durations: usize,
    batch_size: usize,
) -> Vec<f32> {
    let mut hazards = vec![0.0f32; batch_size * num_events * num_durations];

    for i in 0..batch_size {
        for t in 0..num_durations {
            let mut max_logit = f32::NEG_INFINITY;
            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                max_logit = max_logit.max(logits[idx]);
            }
            max_logit = max_logit.max(0.0);

            let mut denom = (-max_logit).exp();
            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                denom += (logits[idx] - max_logit).exp();
            }

            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                hazards[idx] = (logits[idx] - max_logit).exp() / denom;
            }
        }
    }

    hazards
}

fn compute_weighted_competing_risk_loss(
    hazards: &[f32],
    durations: &[usize],
    events: &[i32],
    num_events: usize,
    num_durations: usize,
    batch_indices: &[usize],
    event_weights: &[f64],
) -> f64 {
    let batch_size = batch_indices.len();
    let mut total_loss = 0.0;
    let eps = 1e-7;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            if event > 0 && t == duration_bin {
                let k = (event - 1) as usize;
                if k < num_events {
                    let idx = local_idx * num_events * num_durations + k * num_durations + t;
                    let h = hazards[idx].max(eps);
                    total_loss -= event_weights[k] * (h as f64).ln();
                }
            } else {
                let mut sum_h = 0.0f32;
                for k in 0..num_events {
                    let idx = local_idx * num_events * num_durations + k * num_durations + t;
                    sum_h += hazards[idx];
                }
                let survival_prob = (1.0 - sum_h).max(eps);
                total_loss -= (survival_prob as f64).ln();
            }
        }
    }

    total_loss / batch_size.max(1) as f64
}

fn compute_competing_risk_gradient(
    hazards: &[f32],
    durations: &[usize],
    events: &[i32],
    num_events: usize,
    num_durations: usize,
    batch_indices: &[usize],
    event_weights: &[f64],
) -> Vec<Vec<f32>> {
    let batch_size = batch_indices.len();
    let mut gradients: Vec<Vec<f32>> = vec![vec![0.0f32; batch_size * num_durations]; num_events];

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            for k in 0..num_events {
                let h_idx = local_idx * num_events * num_durations + k * num_durations + t;
                let h = hazards[h_idx];
                let g_idx = local_idx * num_durations + t;

                if event > 0 && t == duration_bin && (event - 1) as usize == k {
                    gradients[k][g_idx] = (h - 1.0) * event_weights[k] as f32;
                } else {
                    gradients[k][g_idx] = h * event_weights.get(k).copied().unwrap_or(1.0) as f32;
                }
            }
        }
    }

    let divisor = batch_size.max(1) as f32;
    for grad in &mut gradients {
        for g in grad.iter_mut() {
            *g /= divisor;
        }
    }

    gradients
}

#[derive(Clone)]
#[allow(dead_code)]
struct StoredWeights {
    feature_projections: Vec<(Vec<f32>, Vec<f32>)>,
    decay_rates: Vec<f32>,
    missing_embeddings: Vec<f32>,
    factorized_layers: Vec<FactorizedLayerWeights>,
    cause_specific_heads: Vec<CauseSpecificHeadWeights>,
    embedding_dim: usize,
    n_features: usize,
    num_events: usize,
    num_durations: usize,
}

#[derive(Clone)]
struct FactorizedLayerWeights {
    temporal_query_w: Vec<f32>,
    temporal_query_b: Vec<f32>,
    temporal_key_w: Vec<f32>,
    temporal_key_b: Vec<f32>,
    temporal_value_w: Vec<f32>,
    temporal_value_b: Vec<f32>,
    temporal_output_w: Vec<f32>,
    temporal_output_b: Vec<f32>,
    covariate_query_w: Vec<f32>,
    covariate_query_b: Vec<f32>,
    covariate_key_w: Vec<f32>,
    covariate_key_b: Vec<f32>,
    covariate_value_w: Vec<f32>,
    covariate_value_b: Vec<f32>,
    covariate_output_w: Vec<f32>,
    covariate_output_b: Vec<f32>,
    ln_time_gamma: Vec<f32>,
    ln_time_beta: Vec<f32>,
    ln_cov_gamma: Vec<f32>,
    ln_cov_beta: Vec<f32>,
    ffn_w1: Vec<f32>,
    ffn_b1: Vec<f32>,
    ffn_w2: Vec<f32>,
    ffn_b2: Vec<f32>,
    ln_ffn_gamma: Vec<f32>,
    ln_ffn_beta: Vec<f32>,
    num_heads: usize,
    embedding_dim: usize,
}

#[derive(Clone)]
struct CauseSpecificHeadWeights {
    mlp1_w: Vec<f32>,
    mlp1_b: Vec<f32>,
    mlp2_w: Vec<f32>,
    mlp2_b: Vec<f32>,
    output_w: Vec<f32>,
    output_b: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("num_factorized_layers", &self.factorized_layers.len())
            .field("num_events", &self.num_events)
            .finish()
    }
}

fn extract_weights(model: &TracerNetwork<AutodiffBackend>, config: &TracerConfig) -> StoredWeights {
    let mut feature_projections = Vec::new();
    for proj in &model.time_aware_embedding.feature_projections {
        let w: Vec<f32> = tensor_to_vec_f32(proj.weight.val().inner());
        let b: Vec<f32> = proj
            .bias
            .as_ref()
            .map(|bias| bias.val().inner().into_data().to_vec().unwrap_or_default())
            .unwrap_or_default();
        feature_projections.push((w, b));
    }

    let decay_rates: Vec<f32> = model
        .time_aware_embedding
        .decay_rates
        .val()
        .inner()
        .into_data()
        .to_vec()
        .unwrap_or_default();

    let missing_embeddings: Vec<f32> = {
        let tensor = model.time_aware_embedding.missing_embeddings.val().inner();
        let [n_feat, emb_dim] = tensor.dims();
        tensor
            .into_data()
            .to_vec()
            .unwrap_or_else(|_| vec![0.0; n_feat * emb_dim])
    };

    let mut factorized_layers = Vec::new();
    for layer in &model.factorized_layers {
        let flw = FactorizedLayerWeights {
            temporal_query_w: tensor_to_vec_f32(
                layer.temporal_attention.query.weight.val().inner(),
            ),
            temporal_query_b: layer
                .temporal_attention
                .query
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            temporal_key_w: tensor_to_vec_f32(layer.temporal_attention.key.weight.val().inner()),
            temporal_key_b: layer
                .temporal_attention
                .key
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            temporal_value_w: tensor_to_vec_f32(
                layer.temporal_attention.value.weight.val().inner(),
            ),
            temporal_value_b: layer
                .temporal_attention
                .value
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            temporal_output_w: tensor_to_vec_f32(
                layer.temporal_attention.output.weight.val().inner(),
            ),
            temporal_output_b: layer
                .temporal_attention
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            covariate_query_w: tensor_to_vec_f32(
                layer.covariate_attention.query.weight.val().inner(),
            ),
            covariate_query_b: layer
                .covariate_attention
                .query
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            covariate_key_w: tensor_to_vec_f32(layer.covariate_attention.key.weight.val().inner()),
            covariate_key_b: layer
                .covariate_attention
                .key
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            covariate_value_w: tensor_to_vec_f32(
                layer.covariate_attention.value.weight.val().inner(),
            ),
            covariate_value_b: layer
                .covariate_attention
                .value
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            covariate_output_w: tensor_to_vec_f32(
                layer.covariate_attention.output.weight.val().inner(),
            ),
            covariate_output_b: layer
                .covariate_attention
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            ln_time_gamma: layer
                .layer_norm_time_gamma
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln_time_beta: layer
                .layer_norm_time_beta
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln_cov_gamma: layer
                .layer_norm_cov_gamma
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln_cov_beta: layer
                .layer_norm_cov_beta
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ffn_w1: tensor_to_vec_f32(layer.ffn_linear1.weight.val().inner()),
            ffn_b1: layer
                .ffn_linear1
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            ffn_w2: tensor_to_vec_f32(layer.ffn_linear2.weight.val().inner()),
            ffn_b2: layer
                .ffn_linear2
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            ln_ffn_gamma: layer
                .layer_norm_ffn_gamma
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            ln_ffn_beta: layer
                .layer_norm_ffn_beta
                .val()
                .inner()
                .into_data()
                .to_vec()
                .unwrap_or_default(),
            num_heads: config.num_attention_heads,
            embedding_dim: config.embedding_dim,
        };
        factorized_layers.push(flw);
    }

    let mut cause_specific_heads = Vec::new();
    for head in &model.cause_specific_heads {
        let chw = CauseSpecificHeadWeights {
            mlp1_w: tensor_to_vec_f32(head.mlp_layer1.weight.val().inner()),
            mlp1_b: head
                .mlp_layer1
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            mlp2_w: tensor_to_vec_f32(head.mlp_layer2.weight.val().inner()),
            mlp2_b: head
                .mlp_layer2
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            output_w: tensor_to_vec_f32(head.output.weight.val().inner()),
            output_b: head
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default(),
            input_dim: config.embedding_dim,
            hidden_dim: config.mlp_hidden_size,
            output_dim: config.num_durations,
        };
        cause_specific_heads.push(chw);
    }

    StoredWeights {
        feature_projections,
        decay_rates,
        missing_embeddings,
        factorized_layers,
        cause_specific_heads,
        embedding_dim: config.embedding_dim,
        n_features: model.n_features,
        num_events: model.num_events,
        num_durations: config.num_durations,
    }
}

fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn gelu_cpu(x: f64) -> f64 {
    let sqrt_2 = std::f64::consts::SQRT_2;
    x * 0.5 * (1.0 + erf_approx(x / sqrt_2))
}

fn layer_norm_cpu(x: &[f64], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / n as f64;
    let std = (var + eps as f64).sqrt();

    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let g = if i < gamma.len() {
                gamma[i] as f64
            } else {
                1.0
            };
            let b = if i < beta.len() { beta[i] as f64 } else { 0.0 };
            (xi - mean) / std * g + b
        })
        .collect()
}

fn apply_mha_cpu(
    x: &[f64],
    query_w: &[f32],
    query_b: &[f32],
    key_w: &[f32],
    key_b: &[f32],
    value_w: &[f32],
    value_b: &[f32],
    output_w: &[f32],
    output_b: &[f32],
    hidden: usize,
    num_heads: usize,
) -> Vec<f64> {
    let q = linear_forward(x, query_w, query_b, hidden, hidden);
    let k = linear_forward(x, key_w, key_b, hidden, hidden);
    let v = linear_forward(x, value_w, value_b, hidden, hidden);

    let head_dim = hidden / num_heads;
    let mut attn_output = vec![0.0f64; hidden];

    for head in 0..num_heads {
        let start = head * head_dim;
        let end = start + head_dim;

        let mut score = 0.0;
        for i in start..end {
            score += q[i] * k[i];
        }
        score /= (head_dim as f64).sqrt();

        for i in start..end {
            attn_output[i] = v[i];
        }
    }

    linear_forward(&attn_output, output_w, output_b, hidden, hidden)
}

fn apply_factorized_layer_cpu(
    hidden: &[f64],
    layer: &FactorizedLayerWeights,
    eps: f32,
) -> Vec<f64> {
    let h = layer.embedding_dim;

    let time_attn = apply_mha_cpu(
        hidden,
        &layer.temporal_query_w,
        &layer.temporal_query_b,
        &layer.temporal_key_w,
        &layer.temporal_key_b,
        &layer.temporal_value_w,
        &layer.temporal_value_b,
        &layer.temporal_output_w,
        &layer.temporal_output_b,
        h,
        layer.num_heads,
    );

    let residual1: Vec<f64> = hidden.iter().zip(&time_attn).map(|(a, b)| a + b).collect();
    let normed1 = layer_norm_cpu(&residual1, &layer.ln_time_gamma, &layer.ln_time_beta, eps);

    let cov_attn = apply_mha_cpu(
        &normed1,
        &layer.covariate_query_w,
        &layer.covariate_query_b,
        &layer.covariate_key_w,
        &layer.covariate_key_b,
        &layer.covariate_value_w,
        &layer.covariate_value_b,
        &layer.covariate_output_w,
        &layer.covariate_output_b,
        h,
        layer.num_heads,
    );

    let residual2: Vec<f64> = normed1.iter().zip(&cov_attn).map(|(a, b)| a + b).collect();
    let normed2 = layer_norm_cpu(&residual2, &layer.ln_cov_gamma, &layer.ln_cov_beta, eps);

    let ffn_hidden = h * 4;
    let ffn1 = linear_forward(&normed2, &layer.ffn_w1, &layer.ffn_b1, h, ffn_hidden);
    let ffn1_act: Vec<f64> = ffn1.iter().map(|&x| gelu_cpu(x)).collect();
    let ffn2 = linear_forward(&ffn1_act, &layer.ffn_w2, &layer.ffn_b2, ffn_hidden, h);

    let residual3: Vec<f64> = normed2.iter().zip(&ffn2).map(|(a, b)| a + b).collect();
    layer_norm_cpu(&residual3, &layer.ln_ffn_gamma, &layer.ln_ffn_beta, eps)
}

fn predict_with_weights(
    x: &[f64],
    mask: &[f64],
    time_delta: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    weights: &StoredWeights,
    layer_norm_eps: f32,
) -> Vec<Vec<f64>> {
    let n_features = weights.n_features;
    let embedding_dim = weights.embedding_dim;

    let mut all_outputs: Vec<Vec<f64>> = vec![Vec::new(); weights.num_events];

    for i in 0..n_obs {
        let seq_len = seq_lengths
            .get(i)
            .copied()
            .unwrap_or(max_seq_len)
            .min(max_seq_len);

        let mut seq_embeddings = Vec::new();

        for t in 0..seq_len {
            let mut timestep_emb = vec![0.0f64; embedding_dim];

            for f in 0..n_features {
                let idx = i * max_seq_len * n_features + t * n_features + f;
                let x_val = x.get(idx).copied().unwrap_or(0.0);
                let mask_val = mask.get(idx).copied().unwrap_or(1.0);
                let delta_val = time_delta.get(idx).copied().unwrap_or(0.0);

                let (proj_w, proj_b) = &weights.feature_projections[f];
                let proj = linear_forward(&[x_val], proj_w, proj_b, 1, embedding_dim);

                let decay_rate = weights.decay_rates.get(f).copied().unwrap_or(0.1) as f64;
                let decay = (-delta_val * decay_rate).exp();

                if mask_val > 0.5 {
                    for d in 0..embedding_dim {
                        timestep_emb[d] += proj[d] * decay;
                    }
                } else {
                    for d in 0..embedding_dim {
                        let missing_emb = weights
                            .missing_embeddings
                            .get(f * embedding_dim + d)
                            .copied()
                            .unwrap_or(0.0) as f64;
                        timestep_emb[d] += missing_emb;
                    }
                }
            }

            seq_embeddings.push(timestep_emb);
        }

        let mut pooled = vec![0.0f64; embedding_dim];
        if !seq_embeddings.is_empty() {
            for emb in &seq_embeddings {
                for layer in &weights.factorized_layers {
                    let _ = apply_factorized_layer_cpu(emb, layer, layer_norm_eps);
                }
            }

            for emb in &seq_embeddings {
                for d in 0..embedding_dim {
                    pooled[d] += emb[d];
                }
            }
            for d in 0..embedding_dim {
                pooled[d] /= seq_embeddings.len() as f64;
            }
        }

        for (event_idx, head) in weights.cause_specific_heads.iter().enumerate() {
            let h1 = linear_forward(
                &pooled,
                &head.mlp1_w,
                &head.mlp1_b,
                head.input_dim,
                head.hidden_dim,
            );
            let mut h1_act = h1;
            relu_vec(&mut h1_act);
            let h2 = linear_forward(
                &h1_act,
                &head.mlp2_w,
                &head.mlp2_b,
                head.hidden_dim,
                head.hidden_dim,
            );
            let mut h2_act = h2;
            relu_vec(&mut h2_act);
            let output = linear_forward(
                &h2_act,
                &head.output_w,
                &head.output_b,
                head.hidden_dim,
                head.output_dim,
            );
            all_outputs[event_idx].extend(output);
        }
    }

    all_outputs
}

fn fit_tracer_inner(
    x: &[f64],
    mask: &[f64],
    time_delta: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &TracerConfig,
) -> Tracer {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: TracerNetwork<AutodiffBackend> = TracerNetwork::new(&device, n_features, config);

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        )))
        .init();

    let event_weights = compute_event_weights(event, config.num_events);

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

            let total_size = batch_size * max_seq_len * n_features;

            let stride = max_seq_len * n_features;
            let (x_batch, mask_batch, time_delta_batch): (Vec<f32>, Vec<f32>, Vec<f32>) =
                batch_indices
                    .par_iter()
                    .map(|&i| {
                        let start = i * stride;
                        let x_slice: Vec<f32> = (0..stride)
                            .map(|j| x.get(start + j).copied().unwrap_or(0.0) as f32)
                            .collect();
                        let mask_slice: Vec<f32> = (0..stride)
                            .map(|j| mask.get(start + j).copied().unwrap_or(1.0) as f32)
                            .collect();
                        let time_delta_slice: Vec<f32> = (0..stride)
                            .map(|j| time_delta.get(start + j).copied().unwrap_or(0.0) as f32)
                            .collect();
                        (x_slice, mask_slice, time_delta_slice)
                    })
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut x_acc, mut m_acc, mut t_acc), (x_i, m_i, t_i)| {
                            x_acc.extend(x_i);
                            m_acc.extend(m_i);
                            t_acc.extend(t_i);
                            (x_acc, m_acc, t_acc)
                        },
                    );

            let seq_lengths_batch: Vec<usize> = batch_indices
                .iter()
                .map(|&i| seq_lengths.get(i).copied().unwrap_or(max_seq_len))
                .collect();

            let x_data =
                burn::tensor::TensorData::new(x_batch, [batch_size, max_seq_len, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(x_data, &device);

            let mask_data =
                burn::tensor::TensorData::new(mask_batch, [batch_size, max_seq_len, n_features]);
            let mask_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(mask_data, &device);

            let time_delta_data = burn::tensor::TensorData::new(
                time_delta_batch,
                [batch_size, max_seq_len, n_features],
            );
            let time_delta_tensor: Tensor<AutodiffBackend, 3> =
                Tensor::from_data(time_delta_data, &device);

            let outputs = model.forward(
                x_tensor,
                mask_tensor,
                time_delta_tensor,
                &seq_lengths_batch,
                true,
            );

            let mut all_logits = Vec::new();
            for logits_tensor in &outputs {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());
                all_logits.push(logits_vec);
            }

            let mut combined_logits: Vec<f32> =
                Vec::with_capacity(batch_size * config.num_events * config.num_durations);
            for i in 0..batch_size {
                for k in 0..config.num_events {
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

            let hazards = multinomial_hazard_normalization(
                &combined_logits,
                config.num_events,
                config.num_durations,
                batch_size,
            );

            let loss = compute_weighted_competing_risk_loss(
                &hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &batch_indices,
                &event_weights,
            );
            epoch_loss += loss;
            n_batches += 1;

            let gradients = compute_competing_risk_gradient(
                &hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &batch_indices,
                &event_weights,
            );

            if !gradients.is_empty() && !outputs.is_empty() {
                let grad_data = burn::tensor::TensorData::new(
                    gradients[0].clone(),
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
            let val_outputs = predict_with_weights(
                x,
                mask,
                time_delta,
                seq_lengths,
                n_val,
                max_seq_len,
                &extract_weights(&model, config),
                config.layer_norm_eps,
            );

            let mut combined_val_logits: Vec<f32> =
                Vec::with_capacity(n_val * config.num_events * config.num_durations);
            for i in 0..n_val {
                for k in 0..config.num_events {
                    for t in 0..config.num_durations {
                        let val = val_outputs
                            .get(k)
                            .and_then(|v| v.get(i * config.num_durations + t))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        combined_val_logits.push(val);
                    }
                }
            }

            let val_hazards = multinomial_hazard_normalization(
                &combined_val_logits,
                config.num_events,
                config.num_durations,
                n_val,
            );

            let val_loss = compute_weighted_competing_risk_loss(
                &val_hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &val_indices,
                &event_weights,
            );
            val_loss_history.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
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

    Tracer {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        max_seq_len,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Tracer {
    weights: StoredWeights,
    config: TracerConfig,
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
impl Tracer {
    #[staticmethod]
    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_obs, max_seq_len, n_features, time, event, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_obs: usize,
        max_seq_len: usize,
        n_features: usize,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &TracerConfig,
    ) -> PyResult<Self> {
        let expected_len = n_obs * max_seq_len * n_features;
        if x.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "x length {} must equal n_obs * max_seq_len * n_features = {}",
                x.len(),
                expected_len
            )));
        }
        if mask.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mask length must equal n_obs * max_seq_len * n_features",
            ));
        }
        if time_delta.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time_delta length must equal n_obs * max_seq_len * n_features",
            ));
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

        let config = config.clone();
        Ok(py.detach(move || {
            fit_tracer_inner(
                &x,
                &mask,
                &time_delta,
                &seq_lengths,
                n_obs,
                max_seq_len,
                n_features,
                &time,
                &event,
                &config,
            )
        }))
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len, event_idx=0))]
    pub fn predict_hazard(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
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
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let mut row = Vec::with_capacity(num_durations);
            for t in 0..num_durations {
                let idx = i * num_events * num_durations + event_idx * num_durations + t;
                row.push(hazards[idx] as f64);
            }
            result.push(row);
        }

        Ok(result)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len))]
    pub fn predict_survival(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
            &self.weights,
            self.config.layer_norm_eps,
        );

        let num_durations = self.config.num_durations;
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let survival: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut surv = Vec::with_capacity(num_durations);
                let mut cum_surv = 1.0;
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for k in 0..num_events {
                        let idx = i * num_events * num_durations + k * num_durations + t;
                        total_haz += hazards[idx] as f64;
                    }
                    cum_surv *= 1.0 - total_haz.min(1.0);
                    surv.push(cum_surv.max(0.0));
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len, event_idx=0))]
    pub fn predict_cif(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
            &self.weights,
            self.config.layer_norm_eps,
        );

        if event_idx >= self.weights.num_events {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "event_idx out of range",
            ));
        }

        let num_durations = self.config.num_durations;
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let cif: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut overall_surv = vec![1.0; num_durations + 1];
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for k in 0..num_events {
                        let idx = i * num_events * num_durations + k * num_durations + t;
                        total_haz += hazards[idx] as f64;
                    }
                    overall_surv[t + 1] = overall_surv[t] * (1.0 - total_haz.min(1.0));
                }

                let mut cif_vec = Vec::with_capacity(num_durations);
                let mut cum_inc = 0.0;
                for t in 0..num_durations {
                    let idx = i * num_events * num_durations + event_idx * num_durations + t;
                    cum_inc += overall_surv[t] * hazards[idx] as f64;
                    cif_vec.push(cum_inc);
                }
                cif_vec
            })
            .collect();

        Ok(cif)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len))]
    pub fn predict_risk(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
    ) -> PyResult<Vec<f64>> {
        let survival =
            self.predict_survival(x, mask, time_delta, seq_lengths, n_new, max_seq_len)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| 1.0 - s.last().copied().unwrap_or(1.0))
            .collect();

        Ok(risks)
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
    pub fn get_embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    #[getter]
    pub fn get_num_layers(&self) -> usize {
        self.config.num_factorized_layers
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.weights.n_features
    }
}

#[pyfunction]
#[pyo3(signature = (x, mask, time_delta, seq_lengths, n_obs, max_seq_len, n_features, time, event, config=None))]
pub fn tracer(
    py: Python<'_>,
    x: Vec<f64>,
    mask: Vec<f64>,
    time_delta: Vec<f64>,
    seq_lengths: Vec<usize>,
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&TracerConfig>,
) -> PyResult<Tracer> {
    let cfg = config.cloned().unwrap_or_else(|| {
        TracerConfig::new(
            32, 2, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None,
        )
        .unwrap()
    });

    Tracer::fit(
        py,
        x,
        mask,
        time_delta,
        seq_lengths,
        n_obs,
        max_seq_len,
        n_features,
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
        let config = TracerConfig::new(
            32,
            2,
            4,
            10,
            1,
            64,
            0.1,
            0.0001,
            0.00001,
            64,
            100,
            Some(5),
            0.1,
            1e-12,
            Some(42),
        )
        .unwrap();
        assert_eq!(config.embedding_dim, 32);
        assert_eq!(config.num_factorized_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            TracerConfig::new(
                0, 2, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 0, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 2, 3, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 2, 4, 0, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
    }

    #[test]
    fn test_multinomial_hazard_normalization() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let hazards = multinomial_hazard_normalization(&logits, 2, 3, 1);

        assert_eq!(hazards.len(), 6);
        for &h in &hazards {
            assert!((0.0..=1.0).contains(&h));
        }

        for t in 0..3 {
            let sum: f32 = (0..2).map(|k| hazards[k * 3 + t]).sum();
            assert!(sum < 1.0);
        }
    }

    #[test]
    fn test_event_weights() {
        let events = vec![0, 1, 1, 2, 0, 1, 2, 2];
        let weights = compute_event_weights(&events, 2);

        assert_eq!(weights.len(), 2);
        for &w in &weights {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_tracer_basic() {
        let n_obs = 6;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mask: Vec<f64> = vec![1.0; total_size];
        let time_delta: Vec<f64> = vec![0.0; total_size];
        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 1,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 6,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert_eq!(model.get_num_events(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_tracer_competing_risks() {
        let n_obs = 6;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mask: Vec<f64> = vec![1.0; total_size];
        let time_delta: Vec<f64> = vec![0.0; total_size];
        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 2,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 6,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert_eq!(model.get_num_events(), 2);
    }

    #[test]
    fn test_tracer_with_missing_data() {
        let n_obs = 4;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mut mask: Vec<f64> = vec![1.0; total_size];
        mask[2] = 0.0;
        mask[5] = 0.0;
        mask[10] = 0.0;

        let time_delta: Vec<f64> = (0..total_size)
            .map(|i| if mask[i] < 0.5 { 1.0 } else { 0.0 })
            .collect();

        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 0];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 1,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 4,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert!(!model.train_loss.is_empty());
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

    #[test]
    fn test_gelu_cpu() {
        let x = 0.5;
        let result = gelu_cpu(x);
        assert!(result > 0.0);
        assert!(result < x);
    }
}
