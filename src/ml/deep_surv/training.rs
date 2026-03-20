fn compute_baseline_hazard(
    time: &[f64],
    status: &[i32],
    risk_scores: &[f64],
    unique_times: &[f64],
) -> Vec<f64> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let exp_risks: Vec<f64> = risk_scores
        .iter()
        .map(|&r| r.clamp(-700.0, 700.0).exp())
        .collect();

    let mut risk_sum = exp_risks.iter().sum::<f64>();
    let mut baseline_hazard = Vec::with_capacity(unique_times.len());
    let mut cum_haz = 0.0;

    let mut time_idx = 0;

    for &ut in unique_times {
        while time_idx < n && time[indices[time_idx]] <= ut {
            let idx = indices[time_idx];
            if status[idx] == 1 && risk_sum > 0.0 {
                cum_haz += 1.0 / risk_sum;
            }
            risk_sum -= exp_risks[idx];
            time_idx += 1;
        }
        baseline_hazard.push(cum_haz);
    }

    baseline_hazard
}

fn fit_deep_surv_inner(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    status: &[i32],
    config: &DeepSurvConfig,
) -> DeepSurv {
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();

    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let mut model: DeepSurvNetwork<AutodiffBackend> = DeepSurvNetwork::new(&device, n_vars, config);

    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
            config.l2_reg as f32,
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

    let activation = config.activation;

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
                .flat_map(|&i| (0..n_vars).map(move |j| x[i * n_vars + j] as f32))
                .collect();

            let x_data = burn::tensor::TensorData::new(x_batch, [batch_size, n_vars]);
            let x_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_data, &device);

            let risk_scores = model.forward(x_tensor, activation);
            let risk_vec: Vec<f32> = tensor_to_vec_f32(risk_scores.clone().inner());

            let gradients = compute_cox_gradient_cpu(&risk_vec, time, status, &batch_indices);

            let grad_data = burn::tensor::TensorData::new(gradients, [batch_size, 1]);
            let grad_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(grad_data, &device);

            let pseudo_loss = (risk_scores * grad_tensor).mean();

            let loss_val = compute_cox_loss_cpu(&risk_vec, time, status, &batch_indices);
            epoch_loss += loss_val;
            n_batches += 1;

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
                .flat_map(|&i| (0..n_vars).map(move |j| x[i * n_vars + j] as f32))
                .collect();

            let x_val_data = burn::tensor::TensorData::new(x_val, [n_val, n_vars]);
            let x_val_tensor: Tensor<AutodiffBackend, 2> = Tensor::from_data(x_val_data, &device);

            let val_risk = model.forward_inference(x_val_tensor, activation);
            let val_risk_vec: Vec<f32> = tensor_to_vec_f32(val_risk.inner());

            let val_loss = compute_cox_loss_cpu(&val_risk_vec, time, status, &val_indices);
            val_loss_history.push(val_loss);

            early_stopping.record(val_loss, || {
                extract_weights_from_autodiff(&model, &config.hidden_layers, n_vars)
            });
            if early_stopping.should_stop() {
                break;
            }
        }
    }

    let final_weights = early_stopping
        .into_best_state()
        .unwrap_or_else(|| extract_weights_from_autodiff(&model, &config.hidden_layers, n_vars));

    let all_risks = predict_with_weights(x, n_obs, n_vars, &final_weights, config.activation);

    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let baseline_hazard = compute_baseline_hazard(time, status, &all_risks, &unique_times);

    DeepSurv {
        weights: final_weights,
        hidden_layers: config.hidden_layers.clone(),
        activation: config.activation,
        baseline_hazard,
        unique_times,
        train_loss: train_loss_history,
        val_loss: val_loss_history,
        n_vars,
    }
}
