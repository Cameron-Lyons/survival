
#[derive(Clone)]
struct StoredWeights {
    feature_projections: Vec<(Vec<f32>, Vec<f32>)>,
    decay_rates: Vec<f32>,
    missing_embeddings: Vec<f32>,
    factorized_layers: Vec<FactorizedLayerWeights>,
    cause_specific_heads: Vec<CauseSpecificHeadWeights>,
    embedding_dim: usize,
    n_features: usize,
    num_events: usize,
    _num_durations: usize,
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
        _num_durations: config.num_durations,
    }
}

#[allow(clippy::too_many_arguments)]
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

        let mut _score = 0.0;
        for i in start..end {
            _score += q[i] * k[i];
        }
        _score /= (head_dim as f64).sqrt();

        attn_output[start..end].copy_from_slice(&v[start..end]);
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

#[allow(clippy::too_many_arguments)]
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
                    for (d, timestep_emb_d) in
                        timestep_emb.iter_mut().enumerate().take(embedding_dim)
                    {
                        let missing_emb = weights
                            .missing_embeddings
                            .get(f * embedding_dim + d)
                            .copied()
                            .unwrap_or(0.0) as f64;
                        *timestep_emb_d += missing_emb;
                    }
                }
            }

            seq_embeddings.push(timestep_emb);
        }

        let mut pooled = vec![0.0f64; embedding_dim];
        if !seq_embeddings.is_empty() {
            for emb in &seq_embeddings {
                let mut transformed = emb.clone();
                for layer in &weights.factorized_layers {
                    transformed = apply_factorized_layer_cpu(&transformed, layer, layer_norm_eps);
                }
                for (pooled_d, &val) in pooled
                    .iter_mut()
                    .zip(transformed.iter())
                    .take(embedding_dim)
                {
                    *pooled_d += val;
                }
            }
            for pooled_d in pooled.iter_mut().take(embedding_dim) {
                *pooled_d /= seq_embeddings.len() as f64;
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
