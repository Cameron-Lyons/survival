
#[allow(clippy::too_many_arguments)]
fn fit_survtrace_inner(
    x_cat: Option<&[i64]>,
    x_num: &[f64],
    n_obs: usize,
    num_cat_features: usize,
    num_num_features: usize,
    cat_cardinalities: &[usize],
    time: &[f64],
    event: &[i32],
    config: &SurvTraceConfig,
) -> SurvTrace {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: SurvTraceNetwork<AutodiffBackend> = SurvTraceNetwork::new(
        &device,
        num_cat_features,
        num_num_features,
        cat_cardinalities,
        config,
    );

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.weight_decay as f32,
        )))
        .init();

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

            let x_num_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| {
                    (0..num_num_features).map(move |j| x_num[i * num_num_features + j] as f32)
                })
                .collect();

            let x_num_data = burn::tensor::TensorData::new(
                x_num_batch.clone(),
                [batch_size, num_num_features.max(1)],
            );
            let x_num_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_num_data, &device);

            let x_cat_tensor: Option<Tensor<AutodiffBackend, 2, Int>> = if num_cat_features > 0 {
                if let Some(cats) = x_cat {
                    let x_cat_batch: Vec<i64> = batch_indices
                        .iter()
                        .flat_map(|&i| {
                            (0..num_cat_features).map(move |j| cats[i * num_cat_features + j])
                        })
                        .collect();
                    let x_cat_data =
                        burn::tensor::TensorData::new(x_cat_batch, [batch_size, num_cat_features]);
                    Some(Tensor::from_data(x_cat_data, &device))
                } else {
                    None
                }
            } else {
                None
            };

            let outputs = model.forward(x_cat_tensor, x_num_tensor, true);

            let mut total_loss = 0.0;
            let mut all_grads: Vec<Vec<f32>> = Vec::new();

            for logits_tensor in outputs.iter() {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());

                let loss = compute_nll_logistic_hazard_loss(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &batch_indices,
                );
                total_loss += loss;

                let grads = compute_nll_logistic_hazard_gradient(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &batch_indices,
                );
                all_grads.push(grads);
            }

            epoch_loss += total_loss;
            n_batches += 1;

            if !all_grads.is_empty() {
                let grad_data = burn::tensor::TensorData::new(
                    all_grads[0].clone(),
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
            let x_num_val: Vec<f32> = val_indices
                .iter()
                .flat_map(|&i| {
                    (0..num_num_features).map(move |j| x_num[i * num_num_features + j] as f32)
                })
                .collect();

            let x_num_val_data =
                burn::tensor::TensorData::new(x_num_val, [n_val, num_num_features.max(1)]);
            let x_num_val_tensor: Tensor<AutodiffBackend, 2> =
                Tensor::from_data(x_num_val_data, &device);

            let x_cat_val_tensor: Option<Tensor<AutodiffBackend, 2, Int>> = if num_cat_features > 0
            {
                if let Some(cats) = x_cat {
                    let x_cat_val: Vec<i64> = val_indices
                        .iter()
                        .flat_map(|&i| {
                            (0..num_cat_features).map(move |j| cats[i * num_cat_features + j])
                        })
                        .collect();
                    let x_cat_val_data =
                        burn::tensor::TensorData::new(x_cat_val, [n_val, num_cat_features]);
                    Some(Tensor::from_data(x_cat_val_data, &device))
                } else {
                    None
                }
            } else {
                None
            };

            let val_outputs = model.forward_inference(x_cat_val_tensor, x_num_val_tensor);

            let mut val_loss = 0.0;
            for logits_tensor in &val_outputs {
                let logits_vec: Vec<f32> = tensor_to_vec_f32(logits_tensor.clone().inner());
                val_loss += compute_nll_logistic_hazard_loss(
                    &logits_vec,
                    &duration_bins,
                    event,
                    config.num_durations,
                    &val_indices,
                );
            }
            val_loss_history.push(val_loss);

            early_stopping.record(val_loss, || {
                extract_weights(&model, config, cat_cardinalities)
            });
            if early_stopping.should_stop() {
                break;
            }
        }
    }

    let final_weights = early_stopping
        .into_best_state()
        .unwrap_or_else(|| extract_weights(&model, config, cat_cardinalities));

    SurvTrace {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        cat_cardinalities: cat_cardinalities.to_vec(),
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvTrace {
    weights: StoredWeights,
    config: SurvTraceConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
    #[pyo3(get)]
    pub cat_cardinalities: Vec<usize>,
}

#[pymethods]
impl SurvTrace {
    #[staticmethod]
    #[pyo3(signature = (x_cat, x_num, n_obs, num_cat_features, num_num_features, cat_cardinalities, time, event, config))]
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        py: Python<'_>,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_obs: usize,
        num_cat_features: usize,
        num_num_features: usize,
        cat_cardinalities: Vec<usize>,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &SurvTraceConfig,
    ) -> PyResult<Self> {
        if x_num.len() != n_obs * num_num_features.max(1) && num_num_features > 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_num length must equal n_obs * num_num_features",
            ));
        }
        if time.len() != n_obs || event.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and event must have length n_obs",
            ));
        }
        if let Some(ref cats) = x_cat
            && cats.len() != n_obs * num_cat_features
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_cat length must equal n_obs * num_cat_features",
            ));
        }

        let config = config.clone();
        let x_cat_clone = x_cat.clone();
        Ok(py.detach(move || {
            fit_survtrace_inner(
                x_cat_clone.as_deref(),
                &x_num,
                n_obs,
                num_cat_features,
                num_num_features,
                &cat_cardinalities,
                &time,
                &event,
                &config,
            )
        }))
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_hazard(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let outputs = predict_with_weights(
            x_cat.as_deref(),
            &x_num,
            n_new,
            &self.weights,
            self.config.layer_norm_eps,
        );

        if event_idx >= outputs.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "event_idx out of range",
            ));
        }

        let logits = &outputs[event_idx];
        let num_durations = self.config.num_durations;

        let hazards: Vec<Vec<f64>> = (0..n_new)
            .map(|i| {
                (0..num_durations)
                    .map(|t| {
                        let logit = logits[i * num_durations + t];
                        1.0 / (1.0 + (-logit).exp())
                    })
                    .collect()
            })
            .collect();

        Ok(hazards)
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_survival(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<Vec<f64>>> {
        let hazards = self.predict_hazard(x_cat, x_num, n_new, event_idx)?;

        let survival: Vec<Vec<f64>> = hazards
            .par_iter()
            .map(|h| {
                let mut surv = Vec::with_capacity(h.len());
                let mut cum_surv = 1.0;
                for &haz in h {
                    cum_surv *= 1.0 - haz;
                    surv.push(cum_surv);
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_cat, x_num, n_new, event_idx=0))]
    pub fn predict_risk(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
        event_idx: usize,
    ) -> PyResult<Vec<f64>> {
        let survival = self.predict_survival(x_cat, x_num, n_new, event_idx)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| {
                let final_surv = s.last().copied().unwrap_or(1.0);
                1.0 - final_surv
            })
            .collect();

        Ok(risks)
    }

    #[pyo3(signature = (x_cat, x_num, n_new))]
    pub fn predict_cumulative_incidence(
        &self,
        x_cat: Option<Vec<i64>>,
        x_num: Vec<f64>,
        n_new: usize,
    ) -> PyResult<Vec<Vec<Vec<f64>>>> {
        let num_events = self.weights.num_events;
        let num_durations = self.config.num_durations;

        let outputs = predict_with_weights(
            x_cat.as_deref(),
            &x_num,
            n_new,
            &self.weights,
            self.config.layer_norm_eps,
        );

        let mut all_hazards: Vec<Vec<Vec<f64>>> = Vec::new();
        for logits in outputs.iter().take(num_events) {
            let hazards: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    (0..num_durations)
                        .map(|t| {
                            let logit = logits[i * num_durations + t];
                            1.0 / (1.0 + (-logit).exp())
                        })
                        .collect()
                })
                .collect();
            all_hazards.push(hazards);
        }

        let cifs: Vec<Vec<Vec<f64>>> = (0..n_new)
            .into_par_iter()
            .map(|i| {
                let mut overall_surv = vec![1.0; num_durations + 1];
                for t in 0..num_durations {
                    let mut total_haz = 0.0;
                    for event_hazards in all_hazards.iter().take(num_events) {
                        total_haz += event_hazards[i][t];
                    }
                    overall_surv[t + 1] = overall_surv[t] * (1.0 - total_haz.min(1.0));
                }

                let mut event_cifs = Vec::new();
                for event_hazards in all_hazards.iter().take(num_events) {
                    let mut cif = Vec::with_capacity(num_durations);
                    let mut cum_inc = 0.0;
                    for t in 0..num_durations {
                        cum_inc += overall_surv[t] * event_hazards[i][t];
                        cif.push(cum_inc);
                    }
                    event_cifs.push(cif);
                }
                event_cifs
            })
            .collect();

        Ok(cifs)
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
    pub fn get_hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    #[getter]
    pub fn get_num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
}

#[pyfunction]
#[pyo3(signature = (x_cat, x_num, n_obs, num_cat_features, num_num_features, cat_cardinalities, time, event, config=None))]
#[allow(clippy::too_many_arguments)]
pub fn survtrace(
    py: Python<'_>,
    x_cat: Option<Vec<i64>>,
    x_num: Vec<f64>,
    n_obs: usize,
    num_cat_features: usize,
    num_num_features: usize,
    cat_cardinalities: Vec<usize>,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&SurvTraceConfig>,
) -> PyResult<SurvTrace> {
    let cfg = match config.cloned() {
        Some(cfg) => cfg,
        None => SurvTraceConfig::new(
            16, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12,
        )?,
    };

    SurvTrace::fit(
        py,
        x_cat,
        x_num,
        n_obs,
        num_cat_features,
        num_num_features,
        cat_cardinalities,
        time,
        event,
        &cfg,
    )
}

