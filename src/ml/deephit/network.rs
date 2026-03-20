
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
        for &logit in logits.iter().take(end).skip(start) {
            exp_sum += (logit - max_val).exp();
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

#[allow(clippy::too_many_arguments)]
fn compute_combined_gradient(
    _logits: &[f32],
    pmf: &[f32],
    durations: &[usize],
    events: &[i32],
    num_risks: usize,
    num_durations: usize,
    batch_indices: &[usize],
    _alpha: f64,
    _sigma: f64,
) -> Vec<f32> {
    let batch_size = batch_indices.len();
    let total_outputs = num_risks * num_durations;
    let mut gradients = vec![0.0f32; batch_size * total_outputs];

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
