
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
        let [_batch, _] = x.dims();
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

    fn forward(&self, x: Tensor<B, 3>, _seq_lengths: &[usize]) -> Tensor<B, 2> {
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

    fn _output_size(&self) -> usize {
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

        let shared_output_size = config
            .shared_hidden_sizes
            .last()
            .copied()
            .unwrap_or(shared_input_size);

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

#[allow(clippy::too_many_arguments)]
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
