fn fit_deephit_inner(
    x: &[f64],
    n_obs: usize,
    n_features: usize,
    time: &[f64],
    event: &[i32],
    config: &DeepHitConfig,
) -> DeepHit {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let (duration_bins, cuts) = compute_duration_bins(time, config.num_durations);

    let mut model: DeepHitNetwork<AutodiffBackend> =
        DeepHitNetwork::new(&device, n_features, config);

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

            let x_batch: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| (0..n_features).map(move |j| x[i * n_features + j] as f32))
                .collect();

            let x_data = burn::tensor::TensorData::new(x_batch, [batch_size, n_features]);
            let x_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_data, &device);

            let logits = model.forward(x_tensor, true);
            let logits_vec: Vec<f32> = tensor_to_vec_f32(logits.clone().inner());

            let pmf = softmax_pmf(
                &logits_vec,
                config.num_risks,
                config.num_durations,
                batch_size,
            );

            let nll_loss = compute_nll_loss(
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
            );

            let ranking_loss = compute_ranking_loss(
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
                config.sigma,
            );

            let total_loss = (1.0 - config.alpha) * nll_loss + config.alpha * ranking_loss;
            epoch_loss += total_loss;
            n_batches += 1;

            let gradients = compute_combined_gradient(
                &logits_vec,
                &pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &batch_indices,
                config.alpha,
                config.sigma,
            );

            let total_outputs = config.num_risks * config.num_durations;
            let grad_data = burn::tensor::TensorData::new(gradients, [batch_size, total_outputs]);
            let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

            let pseudo_loss = (logits * grad_tensor).mean();
            let grads = pseudo_loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads);
        }

        let avg_train_loss = if n_batches > 0 {
            epoch_loss / n_batches as f64
        } else {
            0.0
        };
        train_loss_history.push(avg_train_loss);

        if !val_indices.is_empty() {
            let x_val: Vec<f32> = val_indices
                .iter()
                .flat_map(|&i| (0..n_features).map(move |j| x[i * n_features + j] as f32))
                .collect();

            let x_val_data = burn::tensor::TensorData::new(x_val, [n_val, n_features]);
            let x_val_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_val_data, &device);

            let val_logits = model.forward_inference(x_val_tensor);
            let val_logits_vec: Vec<f32> = tensor_to_vec_f32(val_logits.inner());

            let val_pmf = softmax_pmf(
                &val_logits_vec,
                config.num_risks,
                config.num_durations,
                n_val,
            );

            let val_loss = compute_nll_loss(
                &val_pmf,
                &duration_bins,
                event,
                config.num_risks,
                config.num_durations,
                &val_indices,
            );
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

    DeepHit {
        weights: final_weights,
        config: config.clone(),
        duration_cuts: cuts,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct DeepHit {
    weights: StoredWeights,
    config: DeepHitConfig,
    #[pyo3(get)]
    pub duration_cuts: Vec<f64>,
    #[pyo3(get)]
    pub train_loss: Vec<f64>,
    #[pyo3(get)]
    pub val_loss: Vec<f64>,
}

#[pymethods]
impl DeepHit {
    #[staticmethod]
    #[pyo3(signature = (x, n_obs, n_features, time, event, config))]
    pub fn fit(
        py: Python<'_>,
        x: Vec<f64>,
        n_obs: usize,
        n_features: usize,
        time: Vec<f64>,
        event: Vec<i32>,
        config: &DeepHitConfig,
    ) -> PyResult<Self> {
        if x.len() != n_obs * n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x length must equal n_obs * n_features",
            ));
        }
        if time.len() != n_obs || event.len() != n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and event must have length n_obs",
            ));
        }

        let config = config.clone();
        Ok(py.detach(move || fit_deephit_inner(&x, n_obs, n_features, &time, &event, &config)))
    }

    #[pyo3(signature = (x_new, n_new, risk_idx=None))]
    pub fn predict_pmf(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        risk_idx: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let logits = predict_with_weights(&x_new, n_new, &self.weights);
        let logits_f32: Vec<f32> = logits.iter().map(|&x| x as f32).collect();

        let pmf = softmax_pmf(
            &logits_f32,
            self.weights.num_risks,
            self.weights.num_durations,
            n_new,
        );

        let total_outputs = self.weights.num_risks * self.weights.num_durations;

        if let Some(risk) = risk_idx {
            if risk >= self.weights.num_risks {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "risk_idx out of range",
                ));
            }

            let result: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    let start = i * total_outputs + risk * self.weights.num_durations;
                    (0..self.weights.num_durations)
                        .map(|t| pmf[start + t] as f64)
                        .collect()
                })
                .collect();
            Ok(result)
        } else {
            let result: Vec<Vec<f64>> = (0..n_new)
                .map(|i| {
                    let start = i * total_outputs;
                    (0..total_outputs).map(|t| pmf[start + t] as f64).collect()
                })
                .collect();
            Ok(result)
        }
    }

    #[pyo3(signature = (x_new, n_new, risk_idx=None))]
    pub fn predict_cif(
        &self,
        x_new: Vec<f64>,
        n_new: usize,
        risk_idx: Option<usize>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let pmf_all = self.predict_pmf(x_new, n_new, None)?;

        if let Some(risk) = risk_idx {
            if risk >= self.weights.num_risks {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "risk_idx out of range",
                ));
            }

            let cif: Vec<Vec<f64>> = pmf_all
                .par_iter()
                .map(|pmf| {
                    let mut cumulative = Vec::with_capacity(self.weights.num_durations);
                    let mut sum = 0.0;
                    for t in 0..self.weights.num_durations {
                        sum += pmf[risk * self.weights.num_durations + t];
                        cumulative.push(sum);
                    }
                    cumulative
                })
                .collect();
            Ok(cif)
        } else {
            let cif: Vec<Vec<f64>> = pmf_all
                .par_iter()
                .map(|pmf| {
                    let mut all_cif = Vec::new();
                    for risk in 0..self.weights.num_risks {
                        let mut sum = 0.0;
                        for t in 0..self.weights.num_durations {
                            sum += pmf[risk * self.weights.num_durations + t];
                            all_cif.push(sum);
                        }
                    }
                    all_cif
                })
                .collect();
            Ok(cif)
        }
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_survival(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<Vec<f64>>> {
        let pmf_all = self.predict_pmf(x_new, n_new, None)?;

        let survival: Vec<Vec<f64>> = pmf_all
            .par_iter()
            .map(|pmf| {
                let mut surv = Vec::with_capacity(self.weights.num_durations);
                for t in 0..self.weights.num_durations {
                    let mut total_pmf_up_to_t = 0.0;
                    for risk in 0..self.weights.num_risks {
                        for tau in 0..=t {
                            total_pmf_up_to_t += pmf[risk * self.weights.num_durations + tau];
                        }
                    }
                    surv.push((1.0 - total_pmf_up_to_t).max(0.0));
                }
                surv
            })
            .collect();

        Ok(survival)
    }

    #[pyo3(signature = (x_new, n_new))]
    pub fn predict_risk(&self, x_new: Vec<f64>, n_new: usize) -> PyResult<Vec<f64>> {
        let survival = self.predict_survival(x_new, n_new)?;

        let risks: Vec<f64> = survival
            .par_iter()
            .map(|s| 1.0 - s.last().copied().unwrap_or(1.0))
            .collect();

        Ok(risks)
    }

    #[getter]
    pub fn get_num_risks(&self) -> usize {
        self.weights.num_risks
    }

    #[getter]
    pub fn get_num_durations(&self) -> usize {
        self.weights.num_durations
    }

    #[getter]
    pub fn get_n_features(&self) -> usize {
        self.weights.n_features
    }

    #[getter]
    pub fn get_config(&self) -> DeepHitConfig {
        self.config.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_features, time, event, config=None))]
pub fn deephit(
    py: Python<'_>,
    x: Vec<f64>,
    n_obs: usize,
    n_features: usize,
    time: Vec<f64>,
    event: Vec<i32>,
    config: Option<&DeepHitConfig>,
) -> PyResult<DeepHit> {
    let cfg = match config.cloned() {
        Some(cfg) => cfg,
        None => DeepHitConfig::new(
            None, None, 10, 1, 0.1, 0.2, 0.1, 0.001, 256, 100, 0.0001, None, None, 0.1, true,
        )?,
    };

    DeepHit::fit(py, x, n_obs, n_features, time, event, &cfg)
}

