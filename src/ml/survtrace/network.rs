
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

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let [batch_size, hidden_size] = x.dims();
        let seq_len = 1;

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

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
        let context = context.swap_dims(1, 2).reshape([batch_size, hidden_size]);

        self.output.forward(context)
    }
}

#[derive(Module, Debug)]
struct TransformerLayer<B: burn::prelude::Backend> {
    attention: MultiHeadAttention<B>,
    intermediate: Linear<B>,
    output_dense: Linear<B>,
    layer_norm1_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm1_beta: burn::module::Param<Tensor<B, 1>>,
    layer_norm2_gamma: burn::module::Param<Tensor<B, 1>>,
    layer_norm2_beta: burn::module::Param<Tensor<B, 1>>,
    dropout: Dropout,
    layer_norm_eps: f32,
}

impl<B: burn::prelude::Backend> TransformerLayer<B> {
    fn new(
        device: &B::Device,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        hidden_dropout_prob: f64,
        attention_dropout_prob: f64,
        layer_norm_eps: f32,
    ) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                device,
                hidden_size,
                num_heads,
                attention_dropout_prob,
            ),
            intermediate: LinearConfig::new(hidden_size, intermediate_size).init(device),
            output_dense: LinearConfig::new(intermediate_size, hidden_size).init(device),
            layer_norm1_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [hidden_size],
                device,
            )),
            layer_norm1_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [hidden_size],
                device,
            )),
            layer_norm2_gamma: burn::module::Param::from_tensor(Tensor::ones(
                [hidden_size],
                device,
            )),
            layer_norm2_beta: burn::module::Param::from_tensor(Tensor::zeros(
                [hidden_size],
                device,
            )),
            dropout: DropoutConfig::new(hidden_dropout_prob).init(),
            layer_norm_eps,
        }
    }

    fn forward(&self, x: Tensor<B, 2>, training: bool) -> Tensor<B, 2> {
        let attn_output = self.attention.forward(x.clone(), training);
        let attn_output = if training {
            self.dropout.forward(attn_output)
        } else {
            attn_output
        };
        let x = layer_norm(
            x + attn_output,
            self.layer_norm1_gamma.val(),
            self.layer_norm1_beta.val(),
            self.layer_norm_eps,
        );

        let intermediate = self.intermediate.forward(x.clone());
        let intermediate = gelu(intermediate);
        let output = self.output_dense.forward(intermediate);
        let output = if training {
            self.dropout.forward(output)
        } else {
            output
        };

        layer_norm(
            x + output,
            self.layer_norm2_gamma.val(),
            self.layer_norm2_beta.val(),
            self.layer_norm_eps,
        )
    }
}

#[derive(Module, Debug)]
struct SurvTraceNetwork<B: burn::prelude::Backend> {
    cat_embeddings: Vec<Embedding<B>>,
    num_projection: Linear<B>,
    transformer_layers: Vec<TransformerLayer<B>>,
    output_heads: Vec<Linear<B>>,
    hidden_size: usize,
    num_cat_features: usize,
    num_num_features: usize,
    num_events: usize,
    num_durations: usize,
}

impl<B: burn::prelude::Backend> SurvTraceNetwork<B> {
    fn new(
        device: &B::Device,
        num_cat_features: usize,
        num_num_features: usize,
        cat_cardinalities: &[usize],
        config: &SurvTraceConfig,
    ) -> Self {
        let mut cat_embeddings = Vec::new();
        for &card in cat_cardinalities {
            cat_embeddings.push(EmbeddingConfig::new(card.max(2), config.hidden_size).init(device));
        }

        let num_projection = if num_num_features > 0 {
            LinearConfig::new(num_num_features, config.hidden_size).init(device)
        } else {
            LinearConfig::new(1, config.hidden_size).init(device)
        };

        let mut transformer_layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            transformer_layers.push(TransformerLayer::new(
                device,
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.hidden_dropout_prob,
                config.attention_dropout_prob,
                config.layer_norm_eps,
            ));
        }

        let mut output_heads = Vec::new();
        let num_events = config.num_events.max(1);
        for _ in 0..num_events {
            output_heads
                .push(LinearConfig::new(config.hidden_size, config.num_durations).init(device));
        }

        Self {
            cat_embeddings,
            num_projection,
            transformer_layers,
            output_heads,
            hidden_size: config.hidden_size,
            num_cat_features,
            num_num_features,
            num_events,
            num_durations: config.num_durations,
        }
    }

    fn forward(
        &self,
        x_cat: Option<Tensor<B, 2, Int>>,
        x_num: Tensor<B, 2>,
        training: bool,
    ) -> Vec<Tensor<B, 2>> {
        let [batch_size, _] = x_num.dims();
        let device = x_num.device();

        let mut embeddings: Tensor<B, 2> = Tensor::zeros([batch_size, self.hidden_size], &device);

        if let Some(x_cat) = x_cat {
            for (i, emb) in self.cat_embeddings.iter().enumerate() {
                let cat_slice: Tensor<B, 2, Int> = x_cat.clone().slice([0..batch_size, i..i + 1]);
                let cat_emb_3d: Tensor<B, 3> = emb.forward(cat_slice);
                let cat_emb: Tensor<B, 2> = cat_emb_3d.squeeze::<2>();
                embeddings = embeddings + cat_emb;
            }
        }

        if self.num_num_features > 0 {
            let num_emb = self.num_projection.forward(x_num);
            embeddings = embeddings + num_emb;
        }

        let mut hidden = embeddings;
        for layer in &self.transformer_layers {
            hidden = layer.forward(hidden, training);
        }

        let mut outputs = Vec::new();
        for head in &self.output_heads {
            let logits = head.forward(hidden.clone());
            outputs.push(logits);
        }

        outputs
    }

    fn forward_inference(
        &self,
        x_cat: Option<Tensor<B, 2, Int>>,
        x_num: Tensor<B, 2>,
    ) -> Vec<Tensor<B, 2>> {
        self.forward(x_cat, x_num, false)
    }
}

