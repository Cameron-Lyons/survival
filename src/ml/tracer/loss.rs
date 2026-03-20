fn compute_event_weights(events: &[i32], num_events: usize) -> Vec<f64> {
    let mut counts = vec![0usize; num_events + 1];
    for &e in events {
        let idx = (e as usize).min(num_events);
        counts[idx] += 1;
    }

    let total = events.len() as f64;
    let mut weights = vec![1.0; num_events];

    for k in 0..num_events {
        let count = counts[k + 1];
        if count > 0 {
            weights[k] = (total / count as f64).ln().max(1.0);
        }
    }

    let sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum / num_events as f64;
    }

    weights
}

fn multinomial_hazard_normalization(
    logits: &[f32],
    num_events: usize,
    num_durations: usize,
    batch_size: usize,
) -> Vec<f32> {
    let mut hazards = vec![0.0f32; batch_size * num_events * num_durations];

    for i in 0..batch_size {
        for t in 0..num_durations {
            let mut max_logit = f32::NEG_INFINITY;
            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                max_logit = max_logit.max(logits[idx]);
            }
            max_logit = max_logit.max(0.0);

            let mut denom = (-max_logit).exp();
            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                denom += (logits[idx] - max_logit).exp();
            }

            for k in 0..num_events {
                let idx = i * num_events * num_durations + k * num_durations + t;
                hazards[idx] = (logits[idx] - max_logit).exp() / denom;
            }
        }
    }

    hazards
}

fn compute_weighted_competing_risk_loss(
    hazards: &[f32],
    durations: &[usize],
    events: &[i32],
    num_events: usize,
    num_durations: usize,
    batch_indices: &[usize],
    event_weights: &[f64],
) -> f64 {
    let batch_size = batch_indices.len();
    let mut total_loss = 0.0;
    let eps = 1e-7;

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            if event > 0 && t == duration_bin {
                let k = (event - 1) as usize;
                if k < num_events {
                    let idx = local_idx * num_events * num_durations + k * num_durations + t;
                    let h = hazards[idx].max(eps);
                    total_loss -= event_weights[k] * (h as f64).ln();
                }
            } else {
                let mut sum_h = 0.0f32;
                for k in 0..num_events {
                    let idx = local_idx * num_events * num_durations + k * num_durations + t;
                    sum_h += hazards[idx];
                }
                let survival_prob = (1.0 - sum_h).max(eps);
                total_loss -= (survival_prob as f64).ln();
            }
        }
    }

    total_loss / batch_size.max(1) as f64
}

fn compute_competing_risk_gradient(
    hazards: &[f32],
    durations: &[usize],
    events: &[i32],
    num_events: usize,
    num_durations: usize,
    batch_indices: &[usize],
    event_weights: &[f64],
) -> Vec<Vec<f32>> {
    let batch_size = batch_indices.len();
    let mut gradients: Vec<Vec<f32>> = vec![vec![0.0f32; batch_size * num_durations]; num_events];

    for (local_idx, &global_idx) in batch_indices.iter().enumerate() {
        let duration_bin = durations[global_idx].min(num_durations - 1);
        let event = events[global_idx];

        for t in 0..=duration_bin {
            for k in 0..num_events {
                let h_idx = local_idx * num_events * num_durations + k * num_durations + t;
                let h = hazards[h_idx];
                let g_idx = local_idx * num_durations + t;

                if event > 0 && t == duration_bin && (event - 1) as usize == k {
                    gradients[k][g_idx] = (h - 1.0) * event_weights[k] as f32;
                } else {
                    gradients[k][g_idx] = h * event_weights.get(k).copied().unwrap_or(1.0) as f32;
                }
            }
        }
    }

    let divisor = batch_size.max(1) as f32;
    for grad in &mut gradients {
        for g in grad.iter_mut() {
            *g /= divisor;
        }
    }

    gradients
}
