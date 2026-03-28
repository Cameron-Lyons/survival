
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

            let decay_rate_f: Tensor<B, 1> = self.decay_rates.val().slice_dim(0, f);
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

    fn _forward_inference(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 3>,
        time_delta: Tensor<B, 3>,
        seq_lengths: &[usize],
    ) -> Vec<Tensor<B, 2>> {
        self.forward(x, mask, time_delta, seq_lengths, false)
    }
}

