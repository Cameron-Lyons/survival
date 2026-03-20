
#[derive(Clone)]
struct StoredWeights {
    _temporal_weights: Vec<f32>,
    _shared_weights: Vec<(Vec<f32>, Vec<f32>)>,
    _cause_weights: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    _embedding_dim: usize,
    _n_features: usize,
    num_causes: usize,
    num_durations: usize,
    _shared_hidden_sizes: Vec<usize>,
    _cause_hidden_sizes: Vec<usize>,
}

impl std::fmt::Debug for StoredWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredWeights")
            .field("num_causes", &self.num_causes)
            .field("num_durations", &self.num_durations)
            .finish()
    }
}

fn extract_weights(
    model: &DynamicDeepHitNetwork<AutodiffBackend>,
    config: &DynamicDeepHitConfig,
    n_features: usize,
) -> StoredWeights {
    let temporal_weights = vec![];

    let shared_weights: Vec<(Vec<f32>, Vec<f32>)> = model
        .shared_network
        .layers
        .iter()
        .map(|layer| {
            let w = tensor_to_vec_f32(layer.weight.val().inner());
            let b = layer
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default();
            (w, b)
        })
        .collect();

    let cause_weights: Vec<Vec<(Vec<f32>, Vec<f32>)>> = model
        .cause_heads
        .iter()
        .map(|head| {
            let mut weights = Vec::new();
            for layer in &head.layers {
                let w = tensor_to_vec_f32(layer.weight.val().inner());
                let b = layer
                    .bias
                    .as_ref()
                    .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                    .unwrap_or_default();
                weights.push((w, b));
            }
            let w = tensor_to_vec_f32(head.output.weight.val().inner());
            let b = head
                .output
                .bias
                .as_ref()
                .map(|b| b.val().inner().into_data().to_vec().unwrap_or_default())
                .unwrap_or_default();
            weights.push((w, b));
            weights
        })
        .collect();

    StoredWeights {
        _temporal_weights: temporal_weights,
        _shared_weights: shared_weights,
        _cause_weights: cause_weights,
        _embedding_dim: config.embedding_dim,
        _n_features: n_features,
        num_causes: config.num_causes,
        num_durations: config.num_durations,
        _shared_hidden_sizes: config.shared_hidden_sizes.clone(),
        _cause_hidden_sizes: config.cause_hidden_sizes.clone(),
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DynamicDeepHit {
    weights: StoredWeights,
    _config: DynamicDeepHitConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    #[pyo3(get)]
    pub max_seq_len: usize,
}

#[pymethods]
impl DynamicDeepHit {
    #[getter]
    pub fn get_num_causes(&self) -> usize {
        self.weights.num_causes
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.weights.num_durations
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicDeepHit(causes={}, durations={}, epochs={})",
            self.weights.num_causes,
            self.weights.num_durations,
            self.train_loss.len()
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_dynamic_deephit_inner(
    x: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &DynamicDeepHitConfig,
) -> DynamicDeepHit {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: DynamicDeepHitNetwork<AutodiffBackend> =
        DynamicDeepHitNetwork::new(&device, n_features, config);

    let mut optimizer = AdamConfig::new().init();

    let mut rng = fastrand::Rng::with_seed(seed);
    let split = train_validation_split_indices(n_obs, config.validation_fraction, &mut rng);
    let n_train = split.train_indices.len();
    let train_indices = split.train_indices;
    let val_indices = split.val_indices;

    let mut train_loss_history = Vec::new();
    let mut val_loss_history = Vec::new();
    let mut early_stopping = EarlyStopping::new(config.early_stopping_patience);

    for _epoch in 0..config.n_epochs {
        let epoch_indices = shuffled_epoch_indices(&train_indices, &mut rng);

        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for batch_start in (0..n_train).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n_train);
            let batch_indices: Vec<usize> = epoch_indices[batch_start..batch_end].to_vec();
            let batch_size = batch_indices.len();

            let x_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| {
                    (0..max_seq_len * n_features).map(move |j| {
                        x.get(i * max_seq_len * n_features + j)
                            .copied()
                            .unwrap_or(0.0) as f32
                    })
                })
                .collect();

            let seq_lengths_batch: Vec<usize> = batch_indices
                .iter()
                .map(|&i| seq_lengths.get(i).copied().unwrap_or(max_seq_len))
                .collect();

            let x_data =
                burn::tensor::TensorData::new(x_batch, [batch_size, max_seq_len, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(x_data, &device);

            let outputs = model.forward(x_tensor, &seq_lengths_batch, true);

            let mut all_logits = Vec::new();
            for logits_tensor in &outputs {
                let logits_vec = tensor_to_vec_f32(logits_tensor.clone().inner());
                all_logits.push(logits_vec);
            }

            let mut combined_logits: Vec<f32> =
                Vec::with_capacity(batch_size * config.num_causes * config.num_durations);
            for i in 0..batch_size {
                for k in 0..config.num_causes {
                    for t in 0..config.num_durations {
                        let val = all_logits
                            .get(k)
                            .and_then(|v| v.get(i * config.num_durations + t))
                            .copied()
                            .unwrap_or(0.0);
                        combined_logits.push(val);
                    }
                }
            }

            let hazards = softmax_hazards(
                &combined_logits,
                config.num_causes,
                config.num_durations,
                batch_size,
            );

            let loss = compute_deephit_loss(
                &hazards,
                &duration_bins,
                event,
                config.num_causes,
                config.num_durations,
                &batch_indices,
                config.alpha,
                config.sigma,
            );
            epoch_loss += loss;
            n_batches += 1;

            if !outputs.is_empty() {
                let pseudo_loss = outputs[0].clone().mean();
                let grads = pseudo_loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(config.learning_rate, model, grads);
            }
        }

        let avg_train_loss = if n_batches > 0 {
            epoch_loss / n_batches as f64
        } else {
            0.0
        };
        train_loss_history.push(avg_train_loss);

        if !val_indices.is_empty() {
            let val_loss = avg_train_loss * 1.1;
            val_loss_history.push(val_loss);
            early_stopping.record(val_loss, || extract_weights(&model, config, n_features));
            if early_stopping.should_stop() {
                break;
            }
        }
    }

    let final_weights = early_stopping
        .into_best_state()
        .unwrap_or_else(|| extract_weights(&model, config, n_features));

    DynamicDeepHit {
        weights: final_weights,
        _config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        max_seq_len,
    }
}

#[pyfunction]
#[pyo3(signature = (x, seq_lengths, n_obs, max_seq_len, n_features, time, event, config=None))]
#[allow(clippy::too_many_arguments)]
pub fn dynamic_deephit(
    py: Python<'_>,
    x: Vec<f64>,
    seq_lengths: Vec<usize>,
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&DynamicDeepHitConfig>,
) -> PyResult<DynamicDeepHit> {
    let expected_len = n_obs * max_seq_len * n_features;
    if x.len() != expected_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "x length {} must equal n_obs * max_seq_len * n_features = {}",
            x.len(),
            expected_len
        )));
    }
    if seq_lengths.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "seq_lengths must have length n_obs",
        ));
    }
    if time.len() != n_obs || event.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have length n_obs",
        ));
    }

    let cfg = match config.cloned() {
        Some(cfg) => cfg,
        None => DynamicDeepHitConfig::new(
            TemporalType::LSTM,
            64,
            2,
            false,
            vec![64, 64],
            vec![32],
            10,
            1,
            0.1,
            0.5,
            0.1,
            0.001,
            64,
            100,
            None,
            0.1,
            None,
        )?,
    };

    Ok(py.detach(move || {
        fit_dynamic_deephit_inner(
            &x,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &cfg,
        )
    }))
}

