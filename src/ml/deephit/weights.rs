
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

    let shared_output_size = config.shared_layers.last().copied().unwrap_or(n_features);

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

        let final_in_size = config
            .cause_specific_layers
            .last()
            .copied()
            .unwrap_or(shared_output_size);

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

