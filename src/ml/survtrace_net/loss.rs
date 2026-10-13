fn compute_nll_logistic_hazard_loss(
    logits: &[f32],
    durations: &[usize],
    events: &[i32],
    num_durations: usize,
    batch_indices: &[usize],
) -> f64 {
    let mut total_loss = 0.0;
    let mut n_events = 0;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            let logit = logits[local_idx * num_durations + t];
            let target = if t == duration_bin && event == 1 {
                1.0
            } else {
                0.0
            };

            let loss = if target > 0.5 {
                (1.0 + (-logit).exp()).ln()
            } else {
                logit + (1.0 + (-logit).exp()).ln()
            };
            total_loss += loss as f64;
        }

        if event == 1 {
            n_events += 1;
        }
    }

    if n_events > 0 {
        total_loss / n_events as f64
    } else {
        total_loss / batch_indices.len().max(1) as f64
    }
}

fn compute_nll_logistic_hazard_gradient(
    logits: &[f32],
    durations: &[usize],
    events: &[i32],
    num_durations: usize,
    batch_indices: &[usize],
) -> Vec<f32> {
    let batch_size = batch_indices.len();
    let mut gradients = vec![0.0f32; batch_size * num_durations];

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            let logit = logits[local_idx * num_durations + t];
            let pred = 1.0 / (1.0 + (-logit).exp());
            let target = if t == duration_bin && event == 1 {
                1.0
            } else {
                0.0
            };
            gradients[local_idx * num_durations + t] = pred - target;
        }
    }

    let n_events: i32 = batch_indices.iter().map(|&i| events[i]).sum();
    let divisor = if n_events > 0 {
        n_events as f32
    } else {
        batch_size.max(1) as f32
    };

    for g in &mut gradients {
        *g /= divisor;
    }

    gradients
}
