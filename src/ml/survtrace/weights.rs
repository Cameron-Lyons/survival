
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
            for (j, hidden_j) in hidden.iter_mut().enumerate().take(out_dim) {
                let mut sum = if !weights.num_projection_bias.is_empty() {
                    weights.num_projection_bias[j] as f64
                } else {
                    0.0
                };
                for k in 0..in_dim.min(num_num) {
                    sum += x_num[i * num_num + k]
                        * weights.num_projection_weights[j * in_dim + k] as f64;
                }
                *hidden_j += sum;
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

        let mut _score = 0.0;
        for i in start..end {
            _score += q[i] * k[i];
        }
        _score /= (head_dim as f64).sqrt();
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
