
#[allow(clippy::too_many_arguments)]
fn fit_tracer_inner(
    x: &[f64],
    mask: &[f64],
    time_delta: &[f64],
    seq_lengths: &[usize],
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &TracerConfig,
) -> Tracer {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: TracerNetwork<AutodiffBackend> = TracerNetwork::new(&device, n_features, config);

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        )))
        .init();

    let event_weights = compute_event_weights(event, config.num_events);

    let mut rng = fastrand::Rng::with_seed(seed);
    let split = train_validation_split_indices(n_obs, config.validation_fraction, &mut rng);
    let n_train = split.train_indices.len();
    let n_val = split.val_indices.len();
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

            let stride = max_seq_len * n_features;
            let (x_batch, mask_batch, time_delta_batch): (Vec<f32>, Vec<f32>, Vec<f32>) =
                batch_indices
                    .par_iter()
                    .map(|&i| {
                        let start = i * stride;
                        let x_slice: Vec<f32> = (0..stride)
                            .map(|j| x.get(start + j).copied().unwrap_or(0.0) as f32)
                            .collect();
                        let mask_slice: Vec<f32> = (0..stride)
                            .map(|j| mask.get(start + j).copied().unwrap_or(1.0) as f32)
                            .collect();
                        let time_delta_slice: Vec<f32> = (0..stride)
                            .map(|j| time_delta.get(start + j).copied().unwrap_or(0.0) as f32)
                            .collect();
                        (x_slice, mask_slice, time_delta_slice)
                    })
                    .reduce(
                        || (Vec::new(), Vec::new(), Vec::new()),
                        |(mut x_acc, mut m_acc, mut t_acc), (x_i, m_i, t_i)| {
                            x_acc.extend(x_i);
                            m_acc.extend(m_i);
                            t_acc.extend(t_i);
                            (x_acc, m_acc, t_acc)
                        },
                    );

            let seq_lengths_batch: Vec<usize> = batch_indices
                .iter()
                .map(|&i| seq_lengths.get(i).copied().unwrap_or(max_seq_len))
                .collect();

            let x_data =
                burn::tensor::TensorData::new(x_batch, [batch_size, max_seq_len, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(x_data, &device);

            let mask_data =
                burn::tensor::TensorData::new(mask_batch, [batch_size, max_seq_len, n_features]);
            let mask_tensor: Tensor<AutodiffBackend, 3> = Tensor::from_data(mask_data, &device);

            let time_delta_data = burn::tensor::TensorData::new(
                time_delta_batch,
                [batch_size, max_seq_len, n_features],
            );
            let time_delta_tensor: Tensor<AutodiffBackend, 3> =
                Tensor::from_data(time_delta_data, &device);

            let outputs = model.forward(
                x_tensor,
                mask_tensor,
                time_delta_tensor,
                &seq_lengths_batch,
                true,
            );

            let mut all_logits = Vec::new();
            for logits_tensor in &outputs {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());
                all_logits.push(logits_vec);
            }

            let mut combined_logits: Vec<f32> =
                Vec::with_capacity(batch_size * config.num_events * config.num_durations);
            for i in 0..batch_size {
                for k in 0..config.num_events {
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

            let hazards = multinomial_hazard_normalization(
                &combined_logits,
                config.num_events,
                config.num_durations,
                batch_size,
            );

            let loss = compute_weighted_competing_risk_loss(
                &hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &batch_indices,
                &event_weights,
            );
            epoch_loss += loss;
            n_batches += 1;

            let gradients = compute_competing_risk_gradient(
                &hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &batch_indices,
                &event_weights,
            );

            if !gradients.is_empty() && !outputs.is_empty() {
                let grad_data = burn::tensor::TensorData::new(
                    gradients[0].clone(),
                    [batch_size, config.num_durations],
                );
                let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

                let pseudo_loss = (outputs[0].clone() * grad_tensor).mean();
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
            let val_outputs = predict_with_weights(
                x,
                mask,
                time_delta,
                seq_lengths,
                n_val,
                max_seq_len,
                &extract_weights(&model, config),
                config.layer_norm_eps,
            );

            let mut combined_val_logits: Vec<f32> =
                Vec::with_capacity(n_val * config.num_events * config.num_durations);
            for i in 0..n_val {
                for k in 0..config.num_events {
                    for t in 0..config.num_durations {
                        let val = val_outputs
                            .get(k)
                            .and_then(|v| v.get(i * config.num_durations + t))
                            .copied()
                            .unwrap_or(0.0) as f32;
                        combined_val_logits.push(val);
                    }
                }
            }

            let val_hazards = multinomial_hazard_normalization(
                &combined_val_logits,
                config.num_events,
                config.num_durations,
                n_val,
            );

            let val_loss = compute_weighted_competing_risk_loss(
                &val_hazards,
                &duration_bins,
                event,
                config.num_events,
                config.num_durations,
                &val_indices,
                &event_weights,
            );
            val_loss_history.push(val_loss);

            early_stopping.record(val_loss, || extract_weights(&model, config));
            if early_stopping.should_stop() {
                break;
            }
        }
    }

    let final_weights = early_stopping
        .into_best_state()
        .unwrap_or_else(|| extract_weights(&model, config));

    Tracer {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        max_seq_len,
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct Tracer {
    weights: StoredWeights,
    config: TracerConfig,
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
impl Tracer {
    #[staticmethod]
    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_obs, max_seq_len, n_features, time, event, config))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_obs: usize,
        max_seq_len: usize,
        n_features: usize,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &TracerConfig,
    ) -> PyResult<Self> {
        let expected_len = n_obs * max_seq_len * n_features;
        if x.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "x length {} must equal n_obs * max_seq_len * n_features = {}",
                x.len(),
                expected_len
            )));
        }
        if mask.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mask length must equal n_obs * max_seq_len * n_features",
            ));
        }
        if time_delta.len() != expected_len {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time_delta length must equal n_obs * max_seq_len * n_features",
            ));
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

        let config = config.clone();
        Ok(py.detach(move || {
            fit_tracer_inner(
                &x,
                &mask,
                &time_delta,
                &seq_lengths,
                n_obs,
                max_seq_len,
                n_features,
                &time,
                &event,
                &config,
            )
        }))
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len, event_idx=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn predict_hazard(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
            &self.weights,
            self.config.layer_norm_eps,
        );

        if event_idx >= outputs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "event_idx out of range",
            ));
        }

        let num_durations = self.config.num_durations;
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let mut row = Vec::with_capacity(num_durations);
            for t in 0..num_durations {
                let idx = i * num_events * num_durations + event_idx * num_durations + t;
                row.push(hazards[idx] as f64);
            }
            result.push(row);
        }

        Ok(result)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len))]
    pub fn predict_survival(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
            &self.weights,
            self.config.layer_norm_eps,
        );

        let num_durations = self.config.num_durations;
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let survival: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut surv = Vec::with_capacity(num_durations);
                let mut cum_surv = 1.0;
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for k in 0..num_events {
                        let idx = i * num_events * num_durations + k * num_durations + t;
                        total_haz += hazards[idx] as f64;
                    }
                    cum_surv *= 1.0 - total_haz.min(1.0);
                    surv.push(cum_surv.max(0.0));
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len, event_idx=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn predict_cif(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_new,
            max_seq_len,
            &self.weights,
            self.config.layer_norm_eps,
        );

        if event_idx >= self.weights.num_events {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "event_idx out of range",
            ));
        }

        let num_durations = self.config.num_durations;
        let num_events = self.config.num_events;

        let mut combined_logits: Vec<f32> = Vec::with_capacity(n_new * num_events * num_durations);
        for i in 0..n_new {
            for k in 0..num_events {
                for t in 0..num_durations {
                    let val = outputs
                        .get(k)
                        .and_then(|v| v.get(i * num_durations + t))
                        .copied()
                        .unwrap_or(0.0) as f32;
                    combined_logits.push(val);
                }
            }
        }

        let hazards =
            multinomial_hazard_normalization(&combined_logits, num_events, num_durations, n_new);

        let cif: Vec<Vec<f64>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut overall_surv = vec![1.0; num_durations + 1];
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for k in 0..num_events {
                        let idx = i * num_events * num_durations + k * num_durations + t;
                        total_haz += hazards[idx] as f64;
                    }
                    overall_surv[t + 1] = overall_surv[t] * (1.0 - total_haz.min(1.0));
                }

                let mut cif_vec = Vec::with_capacity(num_durations);
                let mut cum_inc = 0.0;
                let hazard_start = i * num_events * num_durations + event_idx * num_durations;
                let hazard_end = hazard_start + num_durations;
                for (t, &hazard) in hazards[hazard_start..hazard_end].iter().enumerate() {
                    cum_inc += overall_surv[t] * hazard as f64;
                    cif_vec.push(cum_inc);
                }
                cif_vec
            })
            .collect();

        Ok(cif)
    }

    #[pyo3(signature = (x, mask, time_delta, seq_lengths, n_new, max_seq_len))]
    pub fn predict_risk(
        &self,
        x: Vec<f64>,
        mask: Vec<f64>,
        time_delta: Vec<f64>,
        seq_lengths: Vec<usize>,
        n_new: usize,
        max_seq_len: usize,
    ) -> PyResult<Vec<f64>> {
        let survival =
            self.predict_survival(x, mask, time_delta, seq_lengths, n_new, max_seq_len)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| 1.0 - s.last().copied().unwrap_or(1.0))
            .collect();

        Ok(risks)
    }

    #[getter]
    pub fn get_num_events(&self) -> usize {
        self.weights.num_events
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.config.num_durations
    }

    #[getter]
    pub fn get_embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    #[getter]
    pub fn get_num_layers(&self) -> usize {
        self.config.num_factorized_layers
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.weights.n_features
    }
}

#[pyfunction]
#[pyo3(signature = (x, mask, time_delta, seq_lengths, n_obs, max_seq_len, n_features, time, event, config=None))]
#[allow(clippy::too_many_arguments)]
pub fn tracer(
    py: Python<'_>,
    x: Vec<f64>,
    mask: Vec<f64>,
    time_delta: Vec<f64>,
    seq_lengths: Vec<usize>,
    n_obs: usize,
    max_seq_len: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&TracerConfig>,
) -> PyResult<Tracer> {
    let cfg = match config.cloned() {
        Some(cfg) => cfg,
        None => TracerConfig::new(
            32, 2, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None,
        )?,
    };

    Tracer::fit(
        py,
        x,
        mask,
        time_delta,
        seq_lengths,
        n_obs,
        max_seq_len,
        n_features,
        time,
        event,
        &cfg,
    )
}

