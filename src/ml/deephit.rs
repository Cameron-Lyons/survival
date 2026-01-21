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

use super::utils::{compute_duration_bins, linear_forward, relu_vec, tensor_to_vec_f32};

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepHitConfig {
    #[pyo3(get, set)]
    pub shared_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub cause_specific_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub num_durations: usize,
    #[pyo3(get, set)]
    pub num_risks: usize,
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
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
    #[pyo3(get, set)]
    pub use_batch_norm: bool,
}

#[pymethods]
impl DeepHitConfig {
    #[new]
    #[pyo3(signature = (
        shared_layers=None,
        cause_specific_layers=None,
        num_durations=10,
        num_risks=1,
        dropout_rate=0.1,
        alpha=0.2,
        sigma=0.1,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=100,
        weight_decay=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1,
        use_batch_norm=true
    ))]
    pub fn new(
        shared_layers: Option<Vec<usize>>,
        cause_specific_layers: Option<Vec<usize>>,
        num_durations: usize,
        num_risks: usize,
        dropout_rate: f64,
        alpha: f64,
        sigma: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        weight_decay: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
        use_batch_norm: bool,
    ) -> PyResult<Self> {
        if num_durations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_durations must be positive",
            ));
        }
        if num_risks == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_risks must be positive",
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
        if sigma <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sigma must be positive",
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

        Ok(DeepHitConfig {
            shared_layers: shared_layers.unwrap_or_else(|| vec![64, 64]),
            cause_specific_layers: cause_specific_layers.unwrap_or_else(|| vec![32]),
            num_durations,
            num_risks,
            dropout_rate,
            alpha,
            sigma,
            learning_rate,
            batch_size,
            n_epochs,
            weight_decay,
            seed,
            early_stopping_patience,
            validation_fraction,
            use_batch_norm,
        })
    }
}

#[derive(Module, Debug)]
struct SharedNetwork<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    dropouts: Vec<Dropout>,
}

impl<B: burn::prelude::Backend> SharedNetwork<B> {
    fn new(
        device: &B::Device,
        n_features: usize,
        hidden_layers: &[usize],
        dropout_rate: f64,
        _use_batch_norm: bool,
    ) -> Self {
        let mut layers = Vec::new();
        let mut dropouts = Vec::new();

        let mut input_size = n_features;
        for &hidden_size in hidden_layers {
            layers.push(LinearConfig::new(input_size, hidden_size).init(device));
            dropouts.push(DropoutConfig::new(dropout_rate).init());
            input_size = hidden_size;
        }

        Self { layers, dropouts }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let mut current = x;

        for i in 0..self.layers.len() {
            current = self.layers[i].forward(current);
            current = relu(current);
            if training {
                current = self.dropouts[i].forward(current);
            }
        }

        current
    }

    fn output_size(&self, hidden_layers: &[usize]) -> usize {
        *hidden_layers.last().unwrap_or(&0)
    }
}

#[derive(Module, Debug)]
struct CauseSpecificNetwork<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    dropouts: Vec<Dropout>,
    output: Linear<B>,
}

impl<B: burn::prelude::Backend> CauseSpecificNetwork<B> {
    fn new(
        device: &B::Device,
        input_size: usize,
        hidden_layers: &[usize],
        num_durations: usize,
        dropout_rate: f64,
    ) -> Self {
        let mut layers = Vec::new();
        let mut dropouts = Vec::new();

        let mut current_size = input_size;
        for &hidden_size in hidden_layers {
            layers.push(LinearConfig::new(current_size, hidden_size).init(device));
            dropouts.push(DropoutConfig::new(dropout_rate).init());
            current_size = hidden_size;
        }

        let output = LinearConfig::new(current_size, num_durations).init(device);

        Self {
            layers,
            dropouts,
            output,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let mut current = x;

        for i in 0..self.layers.len() {
            current = self.layers[i].forward(current);
            current = relu(current);
            if training {
                current = self.dropouts[i].forward(current);
            }
        }

        self.output.forward(current)
    }
}

#[derive(Module, Debug)]
struct DeepHitNetwork<B: burn::prelude::Backend> {
    shared: SharedNetwork<B>,
    cause_specific: Vec<CauseSpecificNetwork<B>>,
    num_risks: usize,
    num_durations: usize,
}

impl<B: burn::prelude::Backend> DeepHitNetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &DeepHitConfig) -> Self {
        let shared = SharedNetwork::new(
            device,
            n_features,
            &config.shared_layers,
            config.dropout_rate,
            config.use_batch_norm,
        );

        let shared_output_size = shared.output_size(&config.shared_layers);
        let shared_output = if shared_output_size == 0 {
            n_features
        } else {
            shared_output_size
        };

        let mut cause_specific = Vec::new();
        for _ in 0..config.num_risks {
            cause_specific.push(CauseSpecificNetwork::new(
                device,
                shared_output,
                &config.cause_specific_layers,
                config.num_durations,
                config.dropout_rate,
            ));
        }

        Self {
            shared,
            cause_specific,
            num_risks: config.num_risks,
            num_durations: config.num_durations,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let [batch_size, _] = x.dims();
        let device = x.device();

        let shared_out = if self.shared.layers.is_empty() {
            x
        } else {
            self.shared.forward(x, training)
        };

        let mut all_outputs = Vec::new();
        for cs_net in &self.cause_specific {
            let out = cs_net.forward(shared_out.clone(), training);
            all_outputs.push(out);
        }

        let total_outputs = self.num_risks * self.num_durations;
        let mut combined_data = vec![0.0f32; batch_size * total_outputs];

        for (risk_idx, out) in all_outputs.into_iter().enumerate() {
            let out_data = out.into_data();
            let out_vec: Vec<f32> = out_data.to_vec().unwrap_or_default();

            for i in 0..batch_size {
                for t in 0..self.num_durations {
                    let src_idx = i * self.num_durations + t;
                    let dst_idx = i * total_outputs + risk_idx * self.num_durations + t;
                    if src_idx < out_vec.len() {
                        combined_data[dst_idx] = out_vec[src_idx];
                    }
                }
            }
        }

        let combined_tensor_data =
            burn::tensor::TensorData::new(combined_data, [batch_size, total_outputs]);
        Tensor::from_data(combined_tensor_data, &device)
    }

    fn forward_inference(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(x, false)
    }
}

fn softmax_pmf(
    logits: &[f32],
    num_risks: usize,
    num_durations: usize,
    batch_size: usize,
) -> Vec<f32> {
    let total_outputs = num_risks * num_durations;
    let mut pmf = vec![0.0f32; batch_size * total_outputs];

    for i in 0..batch_size {
        let start = i * total_outputs;
        let end = start + total_outputs;

        let max_val = logits[start..end]
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut exp_sum = 0.0f32;
        for j in start..end {
            exp_sum += (logits[j] - max_val).exp();
        }

        for j in start..end {
            pmf[j] = (logits[j] - max_val).exp() / exp_sum;
        }
    }

    pmf
}

fn compute_nll_loss(
    pmf: &[f32],
    durations: &[usize],
    events: &[i32],
    num_risks: usize,
    num_durations: usize,
    batch_indices: &[usize],
) -> f64 {
    let batch_size = batch_indices.len();
    let total_outputs = num_risks * num_durations;
    let mut total_loss = 0.0f64;
    let eps = 1e-7;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];
        let pmf_start = local_idx * total_outputs;

        if event > 0 {
            let risk_idx = (event - 1) as usize;
            if risk_idx < num_risks {
                let pmf_val = pmf[pmf_start + risk_idx * num_durations + duration_bin];
                total_loss -= (pmf_val.max(eps) as f64).ln();
            }
        } else {
            let mut survival_prob = 1.0f32;
            for risk in 0..num_risks {
                for t in 0..=duration_bin {
                    survival_prob -= pmf[pmf_start + risk * num_durations + t];
                }
            }
            survival_prob = survival_prob.max(eps);
            total_loss -= (survival_prob as f64).ln();
        }
    }

    total_loss / batch_size.max(1) as f64
}

fn compute_ranking_loss(
    pmf: &[f32],
    durations: &[usize],
    events: &[i32],
    num_risks: usize,
    num_durations: usize,
    batch_indices: &[usize],
    sigma: f64,
) -> f64 {
    let batch_size = batch_indices.len();
    if batch_size < 2 {
        return 0.0;
    }

    let total_outputs = num_risks * num_durations;
    let mut total_loss = 0.0f64;
    let mut n_pairs = 0;

    for (i, &idx_i) in batch_indices.iter().enumerate() {
        let event_i = events[idx_i];
        if event_i == 0 {
            continue;
        }

        let duration_i = durations[idx_i].min(num_durations - 1);
        let risk_i = (event_i - 1) as usize;
        if risk_i >= num_risks {
            continue;
        }

        for (j, &idx_j) in batch_indices.iter().enumerate() {
            if i == j {
                continue;
            }

            let duration_j = durations[idx_j].min(num_durations - 1);

            if duration_i >= duration_j {
                continue;
            }

            let pmf_start_i = i * total_outputs;
            let pmf_start_j = j * total_outputs;

            let mut cif_i = 0.0f32;
            for t in 0..=duration_i {
                cif_i += pmf[pmf_start_i + risk_i * num_durations + t];
            }

            let mut cif_j = 0.0f32;
            for t in 0..=duration_i {
                cif_j += pmf[pmf_start_j + risk_i * num_durations + t];
            }

            let diff = cif_j - cif_i;
            let exp_term = (diff as f64 / sigma).exp();
            total_loss += exp_term;
            n_pairs += 1;
        }
    }

    if n_pairs > 0 {
        total_loss / n_pairs as f64
    } else {
        0.0
    }
}

fn compute_combined_gradient(
    logits: &[f32],
    pmf: &[f32],
    durations: &[usize],
    events: &[i32],
    num_risks: usize,
    num_durations: usize,
    batch_indices: &[usize],
    alpha: f64,
    sigma: f64,
) -> Vec<f32> {
    let batch_size = batch_indices.len();
    let total_outputs = num_risks * num_durations;
    let mut gradients = vec![0.0f32; batch_size * total_outputs];
    let eps = 1e-7;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];
        let start = local_idx * total_outputs;

        if event > 0 {
            let risk_idx = (event - 1) as usize;
            if risk_idx < num_risks {
                let target_idx = start + risk_idx * num_durations + duration_bin;
                for j in start..start + total_outputs {
                    if j == target_idx {
                        gradients[j] = pmf[j] - 1.0;
                    } else {
                        gradients[j] = pmf[j];
                    }
                }
            }
        } else {
            for j in start..start + total_outputs {
                let risk = (j - start) / num_durations;
                let t = (j - start) % num_durations;
                if t <= duration_bin {
                    gradients[j] = pmf[j];
                }
            }
        }
    }

    let scale = 1.0 / batch_size.max(1) as f32;
    for g in &mut gradients {
        *g *= scale;
    }

    gradients
}

#[derive(Clone)]
struct StoredWeights {
    shared_weights: Vec<Vec<f32>>,
    shared_biases: Vec<Vec<f32>>,
    shared_dims: Vec<(usize, usize)>,
    cause_specific_weights: Vec<Vec<Vec<f32>>>,
    cause_specific_biases: Vec<Vec<Vec<f32>>>,
    cause_specific_dims: Vec<Vec<(usize, usize)>>,
    output_weights: Vec<Vec<f32>>,
    output_biases: Vec<Vec<f32>>,
    output_dims: Vec<(usize, usize)>,
    num_risks: usize,
    num_durations: usize,
    n_features: usize,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("num_risks", &self.num_risks)
            .field("num_durations", &self.num_durations)
            .finish()
    }
}

fn extract_weights(
    model: &DeepHitNetwork<AutodiffBackend>,
    config: &DeepHitConfig,
    n_features: usize,
) -> StoredWeights {
    let mut shared_weights = Vec::new();
    let mut shared_biases = Vec::new();
    let mut shared_dims = Vec::new();

    let mut input_size = n_features;
    for (i, layer) in model.shared.layers.iter().enumerate() {
        let output_size = config.shared_layers[i];
        let w: Vec<f32> = tensor_to_vec_f32(layer.weight.val().inner());
        shared_weights.push(w);
        shared_dims.push((input_size, output_size));

        let b: Vec<f32> = layer
            .bias
            .as_ref()
            .map(|bias| bias.val().inner().into_data().to_vec().unwrap_or_default())
            .unwrap_or_default();
        shared_biases.push(b);
        input_size = output_size;
    }

    let shared_output_size = if config.shared_layers.is_empty() {
        n_features
    } else {
        *config.shared_layers.last().unwrap()
    };

    let mut cause_specific_weights = Vec::new();
    let mut cause_specific_biases = Vec::new();
    let mut cause_specific_dims = Vec::new();
    let mut output_weights = Vec::new();
    let mut output_biases = Vec::new();
    let mut output_dims = Vec::new();

    for cs_net in &model.cause_specific {
        let mut cs_w = Vec::new();
        let mut cs_b = Vec::new();
        let mut cs_d = Vec::new();

        let mut in_size = shared_output_size;
        for (i, layer) in cs_net.layers.iter().enumerate() {
            let out_size = config.cause_specific_layers[i];
            let w: Vec<f32> = tensor_to_vec_f32(layer.weight.val().inner());
            cs_w.push(w);
            cs_d.push((in_size, out_size));

            let b: Vec<f32> = layer
                .bias
                .as_ref()
                .map(|bias| bias.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default();
            cs_b.push(b);
            in_size = out_size;
        }

        cause_specific_weights.push(cs_w);
        cause_specific_biases.push(cs_b);
        cause_specific_dims.push(cs_d);

        let final_in_size = if config.cause_specific_layers.is_empty() {
            shared_output_size
        } else {
            *config.cause_specific_layers.last().unwrap()
        };

        let out_w: Vec<f32> = tensor_to_vec_f32(cs_net.output.weight.val().inner());
        output_weights.push(out_w);
        output_dims.push((final_in_size, config.num_durations));

        let out_b: Vec<f32> = cs_net
            .output
            .bias
            .as_ref()
            .map(|bias| bias.val().inner().into_data().to_vec().unwrap_or_default())
            .unwrap_or_default();
        output_biases.push(out_b);
    }

    StoredWeights {
        shared_weights,
        shared_biases,
        shared_dims,
        cause_specific_weights,
        cause_specific_biases,
        cause_specific_dims,
        output_weights,
        output_biases,
        output_dims,
        num_risks: config.num_risks,
        num_durations: config.num_durations,
        n_features,
    }
}

fn predict_with_weights(x: &[f64], n: usize, weights: &StoredWeights) -> Vec<f64> {
    let total_outputs = weights.num_risks * weights.num_durations;
    let mut all_logits = vec![0.0f64; n * total_outputs];

    for i in 0..n {
        let mut current: Vec<f64> = (0..weights.n_features)
            .map(|j| x[i * weights.n_features + j])
            .collect();

        for layer_idx in 0..weights.shared_weights.len() {
            let (in_dim, out_dim) = weights.shared_dims[layer_idx];
            current = linear_forward(
                &current,
                &weights.shared_weights[layer_idx],
                &weights.shared_biases[layer_idx],
                in_dim,
                out_dim,
            );
            relu_vec(&mut current);
        }

        let shared_out = current;

        for (risk_idx, (cs_weights, cs_biases)) in weights
            .cause_specific_weights
            .iter()
            .zip(&weights.cause_specific_biases)
            .enumerate()
        {
            let mut cs_current = shared_out.clone();

            for layer_idx in 0..cs_weights.len() {
                let (in_dim, out_dim) = weights.cause_specific_dims[risk_idx][layer_idx];
                cs_current = linear_forward(
                    &cs_current,
                    &cs_weights[layer_idx],
                    &cs_biases[layer_idx],
                    in_dim,
                    out_dim,
                );
                relu_vec(&mut cs_current);
            }

            let (in_dim, out_dim) = weights.output_dims[risk_idx];
            let output = linear_forward(
                &cs_current,
                &weights.output_weights[risk_idx],
                &weights.output_biases[risk_idx],
                in_dim,
                out_dim,
            );

            for (t, &val) in output.iter().enumerate() {
                all_logits[i * total_outputs + risk_idx * weights.num_durations + t] = val;
            }
        }
    }

    all_logits
}

fn fit_deephit_inner(
    x: &[f64],
    n_obs: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &DeepHitConfig,
) -> DeepHit {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(42);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: DeepHitNetwork<AutodiffBackend> =
        DeepHitNetwork::new(&device, n_features, config);

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

            let x_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| (0..n_features).map(move |j| x[i * n_features + j] as f32))
                .collect();

            let x_data = burn::tensor::TensorData::new(x_batch, [batch_size, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_data, &device);

            let logits = model.forward(x_tensor, true);
            let logits_vec: Vec<f32> = tensor_to_vec_f32(logits.clone().inner());

            let pmf = softmax_pmf(
                &logits_vec,
                config.num_risks,
                config.num_durations,
                batch_size,
            );

            let nll_loss = compute_nll_loss(
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
            );

            let ranking_loss = compute_ranking_loss(
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
                config.sigma,
            );

            let total_loss = (1.0 - config.alpha) * nll_loss + config.alpha * ranking_loss;
            epoch_loss += total_loss;
            n_batches += 1;

            let gradients = compute_combined_gradient(
                &logits_vec,
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
                config.alpha,
                config.sigma,
            );

            let total_outputs = config.num_risks * config.num_durations;
            let grad_data = burn::tensor::TensorData::new(gradients, [batch_size, total_outputs]);
            let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

            let pseudo_loss = (logits * grad_tensor).mean();
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

        if !val_indices.is_empty() {
            let x_val: Vec<f32> = val_indices
                .iter()
                .flat_map(|&i| (0..n_features).map(move |j| x[i * n_features + j] as f32))
                .collect();

            let x_val_data = burn::tensor::TensorData::new(x_val, [n_val, n_features]);
            let x_val_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_val_data, &device);

            let val_logits = model.forward_inference(x_val_tensor);
            let val_logits_vec: Vec<f32> = tensor_to_vec_f32(val_logits.inner());

            let val_pmf = softmax_pmf(
                &val_logits_vec,
                config.num_risks,
                config.num_durations,
                n_val,
            );

            let val_loss = compute_nll_loss(
                &val_pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &val_indices,
            );
            val_loss_history.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
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
    }

    let final_weights = best_weights.unwrap_or_else(|| extract_weights(&model, config, n_features));

    DeepHit {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepHit {
    weights: StoredWeights,
    config: DeepHitConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
}

#[pymethods]
impl DeepHit {
    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_features, time, event, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        n_obs: usize,
        n_features: usize,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &DeepHitConfig,
    ) -> PyResult<Self> {
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

        let config = config.clone();
        Ok(py.detach(move || fit_deephit_inner(&x, n_obs, n_features, &time, &event, &config)))
    }

    #[pyo3(signature = (x_new, n_new, risk_idx=None))]
    pub fn predict_pmf(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        risk_idx: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let logits = predict_with_weights(&x_new, n_new, &self.weights);
        let logits_f32: Vec<f32> = logits.iter().map(|&x| x as f32).collect();

        let pmf = softmax_pmf(
            &logits_f32,
            self.weights.num_risks,
            self.weights.num_durations,
            n_new,
        );

        let total_outputs = self.weights.num_risks * self.weights.num_durations;

        if let Some(risk) = risk_idx {
            if risk >= self.weights.num_risks {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "risk_idx out of range",
                ));
            }

            let result: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    let start = i * total_outputs + risk * self.weights.num_durations;
                    (0..self.weights.num_durations)
                        .map(|t| pmf[start + t] as f64)
                        .collect()
                })
                .collect();
            Ok(result)
        } else {
            let result: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    let start = i * total_outputs;
                    (0..total_outputs).map(|t| pmf[start + t] as f64).collect()
                })
                .collect();
            Ok(result)
        }
    }

    #[pyo3(signature = (x_new, n_new, risk_idx=None))]
    pub fn predict_cif(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        risk_idx: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let pmf_all = self.predict_pmf(x_new, n_new, None)?;

        if let Some(risk) = risk_idx {
            if risk >= self.weights.num_risks {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "risk_idx out of range",
                ));
            }

            let cif: Vec<Vec<f64>> = pmf_all
                .par_iter()
                .map(|pmf| {
                    let mut cumulative = Vec::with_capacity(self.weights.num_durations);
                    let mut sum = 0.0;
                    for t in 0..self.weights.num_durations {
                        sum += pmf[risk * self.weights.num_durations + t];
                        cumulative.push(sum);
                    }
                    cumulative
                })
                .collect();
            Ok(cif)
        } else {
            let cif: Vec<Vec<f64>> = pmf_all
                .par_iter()
                .map(|pmf| {
                    let mut all_cif = Vec::new();
                    for risk in 0..self.weights.num_risks {
                        let mut sum = 0.0;
                        for t in 0..self.weights.num_durations {
                            sum += pmf[risk * self.weights.num_durations + t];
                            all_cif.push(sum);
                        }
                    }
                    all_cif
                })
                .collect();
            Ok(cif)
        }
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let pmf_all = self.predict_pmf(x_new, n_new, None)?;

        let survival: Vec<Vec<f64>> = pmf_all
            .par_iter()
            .map(|pmf| {
                let mut surv = Vec::with_capacity(self.weights.num_durations);
                for t in 0..self.weights.num_durations {
                    let mut total_pmf_up_to_t = 0.0;
                    for risk in 0..self.weights.num_risks {
                        for tau in 0..=t {
                            total_pmf_up_to_t += pmf[risk * self.weights.num_durations + tau];
                        }
                    }
                    surv.push((1.0 - total_pmf_up_to_t).max(0.0));
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        let survival = self.predict_survival(x_new, n_new)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| 1.0 - s.last().copied().unwrap_or(1.0))
            .collect();

        Ok(risks)
    }

    #[getter]
    pub fn get_num_risks(&self) -> usize {
        self.weights.num_risks
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.weights.num_durations
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.weights.n_features
    }

    #[getter]
    pub fn get_config(&self) -> DeepHitConfig {
        self.config.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_features, time, event, config=None))]
pub fn deephit(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&DeepHitConfig>,
) -> PyResult<DeepHit> {
    let cfg = config.cloned().unwrap_or_else(|| {
        DeepHitConfig::new(
            None, None, 10, 1, 0.1, 0.2, 0.1, 0.001, 256, 100, 0.0001, None, None, 0.1, true,
        )
        .unwrap()
    });

    DeepHit::fit(py, x, n_obs, n_features, time, event, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeepHitConfig::new(
            Some(vec![64, 32]),
            Some(vec![16]),
            10,
            2,
            0.1,
            0.2,
            0.1,
            0.001,
            64,
            100,
            0.0001,
            Some(42),
            Some(5),
            0.1,
            true,
        )
        .unwrap();
        assert_eq!(config.shared_layers, vec![64, 32]);
        assert_eq!(config.cause_specific_layers, vec![16]);
        assert_eq!(config.num_risks, 2);
        assert_eq!(config.num_durations, 10);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DeepHitConfig::new(
                None, None, 0, 1, 0.1, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 0, 0.1, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 1, 1.5, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 1, 0.1, 1.5, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
    }

    #[test]
    fn test_softmax_pmf() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let pmf = softmax_pmf(&logits, 2, 3, 1);

        assert_eq!(pmf.len(), 6);
        let sum: f32 = pmf.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        for &p in &pmf {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_deephit_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = DeepHitConfig {
            shared_layers: vec![8],
            cause_specific_layers: vec![4],
            num_durations: 3,
            num_risks: 1,
            dropout_rate: 0.0,
            alpha: 0.2,
            sigma: 0.1,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            use_batch_norm: false,
        };

        let model = fit_deephit_inner(&x, 6, 2, &time, &event, &config);
        assert_eq!(model.get_num_risks(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_deephit_competing_risks() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = DeepHitConfig {
            shared_layers: vec![8],
            cause_specific_layers: vec![4],
            num_durations: 3,
            num_risks: 2,
            dropout_rate: 0.0,
            alpha: 0.2,
            sigma: 0.1,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            use_batch_norm: false,
        };

        let model = fit_deephit_inner(&x, 6, 2, &time, &event, &config);
        assert_eq!(model.get_num_risks(), 2);
    }

    #[test]
    fn test_nll_loss() {
        let pmf = vec![0.1f32, 0.2, 0.3, 0.05, 0.15, 0.2, 0.1, 0.3, 0.2];
        let durations = vec![1, 0, 2];
        let events = vec![1, 0, 1];
        let indices: Vec<usize> = vec![0, 1, 2];

        let loss = compute_nll_loss(&pmf, &durations, &events, 1, 3, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
