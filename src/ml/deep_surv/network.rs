
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
        for (&value, &weight) in current
            .iter()
            .zip(weights.output_weights.iter())
            .take(input_size)
        {
            output += value * weight as f64;
        }
        results.push(output);
    }

    results
}

