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
    tensor::{
        activation::{relu, tanh},
        backend::AutodiffBackend as AutodiffBackendTrait,
    },
};
use pyo3::prelude::*;
use rayon::prelude::*;

use super::utils::tensor_to_vec_f32;

type Backend = NdArray;
type AutodiffBackend = Autodiff<Backend>;

fn selu_activation<B: burn::prelude::Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let alpha: f32 = 1.673_263_2;
    let scale: f32 = 1.050_701;
    let positive = x.clone().clamp_min(0.0);
    let negative = (x.clamp_max(0.0).exp() - 1.0) * alpha;
    (positive + negative) * scale
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum Activation {
    ReLU,
    SELU,
    Tanh,
}

#[pymethods]
impl Activation {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "relu" => Ok(Activation::ReLU),
            "selu" => Ok(Activation::SELU),
            "tanh" => Ok(Activation::Tanh),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown activation function. Use 'relu', 'selu', or 'tanh'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepSurvConfig {
    #[pyo3(get, set)]
    pub hidden_layers: Vec<usize>,
    #[pyo3(get, set)]
    pub activation: Activation,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub n_epochs: usize,
    #[pyo3(get, set)]
    pub l2_reg: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
    #[pyo3(get, set)]
    pub early_stopping_patience: Option<usize>,
    #[pyo3(get, set)]
    pub validation_fraction: f64,
}

#[pymethods]
impl DeepSurvConfig {
    #[new]
    #[pyo3(signature = (
        hidden_layers=None,
        activation=Activation::SELU,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=100,
        l2_reg=0.0001,
        seed=None,
        early_stopping_patience=None,
        validation_fraction=0.1
    ))]
    pub fn new(
        hidden_layers: Option<Vec<usize>>,
        activation: Activation,
        dropout_rate: f64,
        learning_rate: f64,
        batch_size: usize,
        n_epochs: usize,
        l2_reg: f64,
        seed: Option<u64>,
        early_stopping_patience: Option<usize>,
        validation_fraction: f64,
    ) -> PyResult<Self> {
        if !(0.0..1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be in [0, 1)",
            ));
        }
        if learning_rate <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "learning_rate must be positive",
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

        Ok(DeepSurvConfig {
            hidden_layers: hidden_layers.unwrap_or_else(|| vec![64, 32]),
            activation,
            dropout_rate,
            learning_rate,
            batch_size,
            n_epochs,
            l2_reg,
            seed,
            early_stopping_patience,
            validation_fraction,
        })
    }
}

#[derive(Module, Debug)]
struct DeepSurvNetwork<B: burn::prelude::Backend> {
    layers: Vec<Linear<B>>,
    dropouts: Vec<Dropout>,
    output: Linear<B>,
}

fn apply_activation<B: burn::prelude::Backend>(
    x: Tensor<B, 2>,
    activation: Activation,
) -> Tensor<B, 2> {
    match activation {
        Activation::ReLU => relu(x),
        Activation::SELU => selu_activation(x),
        Activation::Tanh => tanh(x),
    }
}

impl<B: burn::prelude::Backend> DeepSurvNetwork<B> {
    fn new(device: &B::Device, n_features: usize, config: &DeepSurvConfig) -> Self {
        let mut layers = Vec::new();
        let mut dropouts = Vec::new();

        let mut input_size = n_features;
        for &hidden_size in &config.hidden_layers {
            layers.push(
                LinearConfig::new(input_size, hidden_size)
                    .with_bias(true)
                    .init(device),
            );
            dropouts.push(DropoutConfig::new(config.dropout_rate).init());
            input_size = hidden_size;
        }

        let output = LinearConfig::new(input_size, 1)
            .with_bias(false)
            .init(device);

        Self {
            layers,
            dropouts,
            output,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, activation: Activation) -> Tensor<B, 2> {
        let mut current = x;

        for (layer, dropout) in self.layers.iter().zip(self.dropouts.iter()) {
            current = layer.forward(current);
            current = apply_activation(current, activation);
            current = dropout.forward(current);
        }

        self.output.forward(current)
    }

    fn forward_inference(&self, x: Tensor<B, 2>, activation: Activation) -> Tensor<B, 2> {
        let mut current = x;

        for layer in self.layers.iter() {
            current = layer.forward(current);
            current = apply_activation(current, activation);
        }

        self.output.forward(current)
    }
}

fn compute_cox_gradient_cpu(
    risk_scores: &[f32],
    time: &[f64],
    status: &[i32],
    batch_indices: &[usize],
) -> Vec<f32> {
    let n = batch_indices.len();
    if n == 0 {
        return Vec::new();
    }

    let mut sorted_order: Vec<usize> = (0..n).collect();
    sorted_order.sort_by(|&a, &b| {
        let ta = time[batch_indices[b]];
        let tb = time[batch_indices[a]];
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_risk = risk_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_risks: Vec<f32> = risk_scores
        .iter()
        .map(|&r| (r - max_risk).clamp(-700.0, 700.0).exp())
        .collect();

    let mut cumsum_exp = vec![0.0f32; n];
    let mut running_sum = 0.0f32;
    for (i, &sorted_idx) in sorted_order.iter().enumerate() {
        running_sum += exp_risks[sorted_idx];
        cumsum_exp[i] = running_sum;
    }

    let mut gradients = vec![0.0f32; n];
    let mut cumsum_d_over_riskset = 0.0f32;

    for (sorted_pos, &sorted_idx) in sorted_order.iter().enumerate().rev() {
        let batch_idx = batch_indices[sorted_idx];
        let risk_set_sum = cumsum_exp[sorted_pos];

        if status[batch_idx] == 1 && risk_set_sum > 0.0 {
            cumsum_d_over_riskset += 1.0 / risk_set_sum;
        }

        if status[batch_idx] == 1 {
            gradients[sorted_idx] = exp_risks[sorted_idx] * cumsum_d_over_riskset - 1.0;
        } else {
            gradients[sorted_idx] = exp_risks[sorted_idx] * cumsum_d_over_riskset;
        }
    }

    let n_events: i32 = batch_indices.iter().map(|&i| status[i]).sum();
    if n_events > 0 {
        for g in &mut gradients {
            *g /= n_events as f32;
        }
    }

    gradients
}

fn compute_cox_loss_cpu(
    risk_scores: &[f32],
    time: &[f64],
    status: &[i32],
    indices: &[usize],
) -> f64 {
    let n = indices.len();
    if n == 0 {
        return 0.0;
    }

    let mut sorted_order: Vec<usize> = (0..n).collect();
    sorted_order.sort_by(|&a, &b| {
        let ta = time[indices[b]];
        let tb = time[indices[a]];
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_risk = risk_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_risks: Vec<f32> = risk_scores
        .iter()
        .map(|&r| (r - max_risk).clamp(-700.0, 700.0).exp())
        .collect();

    let mut log_likelihood = 0.0f64;
    let mut n_events = 0;
    let mut cumsum_exp = 0.0f32;

    for &sorted_idx in &sorted_order {
        let idx = indices[sorted_idx];
        cumsum_exp += exp_risks[sorted_idx];

        if status[idx] == 1 {
            let log_risk_sum = (cumsum_exp as f64).ln() + max_risk as f64;
            log_likelihood += risk_scores[sorted_idx] as f64 - log_risk_sum;
            n_events += 1;
        }
    }

    if n_events == 0 {
        return 0.0;
    }

    -log_likelihood / n_events as f64
}

#[derive(Clone)]
struct StoredWeights {
    layer_weights: Vec<Vec<f32>>,
    layer_biases: Vec<Vec<f32>>,
    output_weights: Vec<f32>,
    layer_input_sizes: Vec<usize>,
    layer_output_sizes: Vec<usize>,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("n_layers", &self.layer_weights.len())
            .finish()
    }
}

fn extract_weights_from_autodiff(
    model: &DeepSurvNetwork<AutodiffBackend>,
    hidden_layers: &[usize],
    n_features: usize,
) -> StoredWeights {
    let mut layer_weights = Vec::new();
    let mut layer_biases = Vec::new();
    let mut layer_input_sizes = Vec::new();
    let mut layer_output_sizes = Vec::new();

    let mut input_size = n_features;
    for (i, layer) in model.layers.iter().enumerate() {
        let output_size = hidden_layers[i];
        let w_tensor: Tensor<AutodiffBackend, 2> = layer.weight.val();
        let w: Vec<f32> = tensor_to_vec_f32(w_tensor.inner());
        layer_weights.push(w);
        layer_input_sizes.push(input_size);
        layer_output_sizes.push(output_size);

        if let Some(ref bias) = layer.bias {
            let b_tensor: Tensor<AutodiffBackend, 1> = bias.val();
            let b: Vec<f32> = b_tensor.inner().into_data().to_vec().unwrap_or_default();
            layer_biases.push(b);
        } else {
            layer_biases.push(Vec::new());
        }
        input_size = output_size;
    }

    let out_tensor: Tensor<AutodiffBackend, 2> = model.output.weight.val();
    let output_weights: Vec<f32> = tensor_to_vec_f32(out_tensor.inner());

    StoredWeights {
        layer_weights,
        layer_biases,
        output_weights,
        layer_input_sizes,
        layer_output_sizes,
    }
}

fn predict_with_weights(
    x: &[f64],
    n: usize,
    p: usize,
    weights: &StoredWeights,
    activation: Activation,
) -> Vec<f64> {
    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let mut current: Vec<f64> = (0..p).map(|j| x[i * p + j]).collect();

        for layer_idx in 0..weights.layer_weights.len() {
            let input_size = weights.layer_input_sizes[layer_idx];
            let output_size = weights.layer_output_sizes[layer_idx];
            let w = &weights.layer_weights[layer_idx];
            let b = &weights.layer_biases[layer_idx];

            let mut next = vec![0.0; output_size];
            for h in 0..output_size {
                let mut sum = if !b.is_empty() { b[h] as f64 } else { 0.0 };
                for k in 0..input_size {
                    sum += current[k] * w[h * input_size + k] as f64;
                }

                next[h] = match activation {
                    Activation::ReLU => sum.max(0.0),
                    Activation::SELU => {
                        let alpha = 1.6732632423543772;
                        let scale = 1.0507009873554805;
                        if sum > 0.0 {
                            scale * sum
                        } else {
                            scale * alpha * (sum.exp() - 1.0)
                        }
                    }
                    Activation::Tanh => sum.tanh(),
                };
            }
            current = next;
        }

        let input_size = current.len();
        let mut output = 0.0;
        for k in 0..input_size {
            output += current[k] * weights.output_weights[k] as f64;
        }
        results.push(output);
    }

    results
}

fn compute_baseline_hazard(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    unique_times: &[f64],
) -> Vec<f64> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let exp_risks: Vec<f64> = risk_scores
        .iter()
        .map(|&r| r.clamp(-700.0, 700.0).exp())
        .collect();

    let mut risk_sum = exp_risks.iter().sum::<f64>();
    let mut baseline_hazard = Vec::with_capacity(unique_times.len());
    let mut cum_haz = 0.0;

    let mut time_idx = 0;

    for &ut in unique_times {
        while time_idx < n && time[indices[time_idx]] <= ut {
            let idx = indices[time_idx];
            if status[idx] == 1 && risk_sum > 0.0 {
                cum_haz += 1.0 / risk_sum;
            }
            risk_sum -= exp_risks[idx];
            time_idx += 1;
        }
        baseline_hazard.push(cum_haz);
    }

    baseline_hazard
}

fn fit_deep_surv_inner(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    config: &DeepSurvConfig,
) -> DeepSurv {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();

    let seed = config.seed.unwrap_or(42);

    let mut model: DeepSurvNetwork<AutodiffBackend> = DeepSurvNetwork::new(&device, n_vars, config);

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.l2_reg as f32,
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

    let activation = config.activation;

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
                .flat_map(|&i| (0..n_vars).map(move |j| x[i * n_vars + j] as f32))
                .collect();

            let x_data = burn::tensor::TensorData::new(x_batch, [batch_size, n_vars]);
            let x_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_data, &device);

            let risk_scores = model.forward(x_tensor, activation);
            let risk_vec: Vec<f32> = tensor_to_vec_f32(risk_scores.clone().inner());

            let gradients = compute_cox_gradient_cpu(&risk_vec, time, status, &batch_indices);

            let grad_data = burn::tensor::TensorData::new(gradients, [batch_size, 1]);
            let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

            let pseudo_loss = (risk_scores * grad_tensor).mean();

            let loss_val = compute_cox_loss_cpu(&risk_vec, time, status, &batch_indices);
            epoch_loss += loss_val;
            n_batches += 1;

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
                .flat_map(|&i| (0..n_vars).map(move |j| x[i * n_vars + j] as f32))
                .collect();

            let x_val_data = burn::tensor::TensorData::new(x_val, [n_val, n_vars]);
            let x_val_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_val_data, &device);

            let val_risk = model.forward_inference(x_val_tensor, activation);
            let val_risk_vec: Vec<f32> = tensor_to_vec_f32(val_risk.inner());

            let val_loss = compute_cox_loss_cpu(&val_risk_vec, time, status, &val_indices);
            val_loss_history.push(val_loss);

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
                best_weights = Some(extract_weights_from_autodiff(
                    &model,
                    &config.hidden_layers,
                    n_vars,
                ));
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

    let final_weights = if let Some(weights) = best_weights {
        weights
    } else {
        extract_weights_from_autodiff(&model, &config.hidden_layers, n_vars)
    };

    let all_risks = predict_with_weights(x, n_obs, n_vars, &final_weights, config.activation);

    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let baseline_hazard = compute_baseline_hazard(time, status, &all_risks, &unique_times);

    DeepSurv {
        weights: final_weights,
        hidden_layers: config.hidden_layers.clone(),
        activation: config.activation,
        baseline_hazard,
        unique_times,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        n_vars,
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeepSurv {
    weights: StoredWeights,
    hidden_layers: Vec<usize>,
    activation: Activation,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub unique_times: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    n_vars: usize,
}

#[pymethods]
impl DeepSurv {
    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_vars, time, status, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
        time: Vec<f64>,
        status: Vec<i32>,
        config: &DeepSurvConfig,
    ) -> PyResult<Self> {
        if x.len() != n_obs * n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x length must equal n_obs * n_vars",
            ));
        }
        if time.len() != n_obs || status.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and status must have length n_obs",
            ));
        }

        let config = config.clone();
        Ok(py.detach(move || fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config)))
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        if x_new.len() != n_new * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_new dimensions don't match",
            ));
        }

        Ok(predict_with_weights(
            &x_new,
            n_new,
            self.n_vars,
            &self.weights,
            self.activation,
        ))
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let survival: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| (-h * risk.exp()).exp())
                    .collect()
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_cumulative_hazard(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.predict_risk(x_new, n_new)?;

        let cumhaz: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk| {
                self.baseline_hazard
                    .iter()
                    .map(|&h| h * risk.exp())
                    .collect()
            })
            .collect();

        Ok(cumhaz)
    }

    #[pyo3(signature = (x_new, n_new, percentile=0.5))]
    pub fn predict_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        percentile: f64,
    ) -> PyResult<Vec<Option<f64>>> {
        if !(0.0..=1.0).contains(&percentile) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "percentile must be between 0 and 1",
            ));
        }

        let survival = self.predict_survival(x_new, n_new)?;

        let times: Vec<Option<f64>> = survival
            .par_iter()
            .map(|surv| {
                for (i, &s) in surv.iter().enumerate() {
                    if s <= percentile && i < self.unique_times.len() {
                        return Some(self.unique_times[i]);
                    }
                }
                None
            })
            .collect();

        Ok(times)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_median_survival_time(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Option<f64>>> {
        self.predict_survival_time(x_new, n_new, 0.5)
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.n_vars
    }

    #[getter]
    pub fn get_hidden_layers(&self) -> Vec<usize> {
        self.hidden_layers.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, status, config=None))]
pub fn deep_surv(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    status: Vec<i32>,
    config: Option<&DeepSurvConfig>,
) -> PyResult<DeepSurv> {
    let cfg = config.cloned().unwrap_or_else(|| {
        DeepSurvConfig::new(
            None,
            Activation::SELU,
            0.2,
            0.001,
            256,
            100,
            0.0001,
            None,
            None,
            0.1,
        )
        .unwrap()
    });

    DeepSurv::fit(py, x, n_obs, n_vars, time, status, &cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_data() -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let x = vec![
            1.0, 0.5, 0.2, 0.8, 0.3, 0.1, 0.6, 0.7, 0.4, 0.4, 0.2, 0.8, 0.9, 0.1, 0.3, 0.3, 0.8,
            0.5, 0.7, 0.4, 0.6, 0.2, 0.6, 0.9,
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1];
        (x, time, status)
    }

    #[test]
    fn test_config_default() {
        let config = DeepSurvConfig::new(
            Some(vec![32, 16]),
            Activation::ReLU,
            0.1,
            0.01,
            64,
            50,
            0.001,
            Some(42),
            Some(10),
            0.2,
        )
        .unwrap();
        assert_eq!(config.hidden_layers, vec![32, 16]);
        assert_eq!(config.n_epochs, 50);
        assert_eq!(config.dropout_rate, 0.1);
        assert_eq!(config.learning_rate, 0.01);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                -0.1,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                1.0,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                -0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                0,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                64,
                0,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                1.0
            )
            .is_err()
        );
    }

    #[test]
    fn test_deep_surv_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, 6, 2, &time, &status, &config);
        assert_eq!(model.get_n_features(), 2);
        assert_eq!(model.get_hidden_layers(), vec![4]);
        assert!(!model.train_loss.is_empty());
        assert_eq!(model.train_loss.len(), 5);
        assert!(!model.unique_times.is_empty());
        assert!(!model.baseline_hazard.is_empty());
    }

    #[test]
    fn test_predict_risk() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, 6, 2, &time, &status, &config);
        let risks = model.predict_risk(x.clone(), 6).unwrap();
        assert_eq!(risks.len(), 6);
        for risk in &risks {
            assert!(risk.is_finite());
        }
    }

    #[test]
    fn test_predict_survival() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4, 2],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        let survival = model.predict_survival(x.clone(), n_obs).unwrap();

        assert_eq!(survival.len(), n_obs);
        for (i, surv) in survival.iter().enumerate() {
            assert_eq!(surv.len(), model.unique_times.len());
            for s in surv {
                assert!(
                    *s >= 0.0 && *s <= 1.0,
                    "Survival prob at row {} should be in [0,1]",
                    i
                );
            }
            for j in 1..surv.len() {
                assert!(
                    surv[j] <= surv[j - 1] + 1e-10,
                    "Survival should be monotonically decreasing"
                );
            }
        }
    }

    #[test]
    fn test_predict_cumulative_hazard() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::Tanh,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        let cumhaz = model.predict_cumulative_hazard(x.clone(), n_obs).unwrap();

        assert_eq!(cumhaz.len(), n_obs);
        for (i, ch) in cumhaz.iter().enumerate() {
            assert_eq!(ch.len(), model.unique_times.len());
            for c in ch {
                assert!(
                    *c >= 0.0,
                    "Cumulative hazard at row {} should be non-negative",
                    i
                );
            }
            for j in 1..ch.len() {
                assert!(
                    ch[j] >= ch[j - 1] - 1e-10,
                    "Cumulative hazard should be monotonically increasing"
                );
            }
        }
    }

    #[test]
    fn test_predict_survival_time() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 10,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);

        let median_times = model
            .predict_median_survival_time(x.clone(), n_obs)
            .unwrap();
        assert_eq!(median_times.len(), n_obs);

        let q75_times = model.predict_survival_time(x.clone(), n_obs, 0.75).unwrap();
        assert_eq!(q75_times.len(), n_obs);

        let q25_times = model.predict_survival_time(x.clone(), n_obs, 0.25).unwrap();
        assert_eq!(q25_times.len(), n_obs);
    }

    #[test]
    fn test_validation_and_early_stopping() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 4,
            n_epochs: 50,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: Some(5),
            validation_fraction: 0.25,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);

        assert!(!model.val_loss.is_empty());
        assert_eq!(model.train_loss.len(), model.val_loss.len());
        assert!(model.train_loss.len() <= 50);
    }

    #[test]
    fn test_all_activations() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 1];

        for activation in [Activation::ReLU, Activation::SELU, Activation::Tanh] {
            let config = DeepSurvConfig {
                hidden_layers: vec![2],
                activation,
                dropout_rate: 0.0,
                learning_rate: 0.01,
                batch_size: 3,
                n_epochs: 2,
                l2_reg: 0.0,
                seed: Some(42),
                early_stopping_patience: None,
                validation_fraction: 0.0,
            };

            let model = fit_deep_surv_inner(&x, 3, 2, &time, &status, &config);
            let risks = model.predict_risk(x.clone(), 3).unwrap();
            assert_eq!(risks.len(), 3);
            for risk in &risks {
                assert!(
                    risk.is_finite(),
                    "Risk should be finite for {:?}",
                    activation
                );
            }
        }
    }

    #[test]
    fn test_multi_layer_network() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![16, 8, 4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        assert_eq!(model.get_hidden_layers(), vec![16, 8, 4]);
        let risks = model.predict_risk(x.clone(), n_obs).unwrap();
        assert_eq!(risks.len(), n_obs);
    }

    #[test]
    fn test_cox_gradient_computation() {
        let risk_scores = vec![0.5f32, -0.3, 0.1, 0.8];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 1];
        let indices: Vec<usize> = (0..4).collect();

        let gradients = compute_cox_gradient_cpu(&risk_scores, &time, &status, &indices);
        assert_eq!(gradients.len(), 4);
        for g in &gradients {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_cox_loss_computation() {
        let risk_scores = vec![0.5f32, -0.3, 0.1, 0.8];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 1];
        let indices: Vec<usize> = (0..4).collect();

        let loss = compute_cox_loss_cpu(&risk_scores, &time, &status, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_baseline_hazard_computation() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 1, 0];
        let risk_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let unique_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let baseline = compute_baseline_hazard(&time, &status, &risk_scores, &unique_times);

        assert_eq!(baseline.len(), unique_times.len());
        for b in &baseline {
            assert!(*b >= 0.0);
        }
        for i in 1..baseline.len() {
            assert!(
                baseline[i] >= baseline[i - 1],
                "Baseline hazard should be monotonically increasing"
            );
        }
    }
}
