fn compute_gradient_survival(x: &[f64], coefficients: &[f64], time: f64, event: usize) -> Vec<f64> {
    let linear_pred: f64 = x
        .iter()
        .zip(coefficients.iter())
        .map(|(&xi, &c)| xi * c)
        .sum();
    let exp_pred = linear_pred
        .clamp(
            crate::constants::LINEAR_PRED_CLAMP_MIN,
            crate::constants::LINEAR_PRED_CLAMP_MAX,
        )
        .exp();

    let p = x.len();
    let mut gradient = vec![0.0; p];

    for j in 0..p {
        if event == 1 {
            gradient[j] = x[j] * (1.0 - exp_pred * time);
        } else {
            gradient[j] = -x[j] * exp_pred * time;
        }
    }

    gradient
}

fn fgsm_attack(
    x: &[f64],
    coefficients: &[f64],
    time: f64,
    event: usize,
    epsilon: f64,
    config: &AdversarialAttackConfig,
) -> Vec<f64> {
    let gradient = compute_gradient_survival(x, coefficients, time, event);

    let sign_direction = if config.targeted { -1.0 } else { 1.0 };

    let perturbed: Vec<f64> = x
        .iter()
        .zip(gradient.iter())
        .map(|(&xi, &gi)| {
            let sign = if gi > 0.0 {
                1.0
            } else if gi < 0.0 {
                -1.0
            } else {
                0.0
            };
            (xi + sign_direction * epsilon * sign).clamp(config.clip_min, config.clip_max)
        })
        .collect();

    perturbed
}

fn pgd_attack(
    x: &[f64],
    coefficients: &[f64],
    time: f64,
    event: usize,
    config: &AdversarialAttackConfig,
) -> Vec<f64> {
    let mut perturbed = x.to_vec();
    let sign_direction = if config.targeted { -1.0 } else { 1.0 };

    for _ in 0..config.n_iterations {
        let gradient = compute_gradient_survival(&perturbed, coefficients, time, event);

        for j in 0..perturbed.len() {
            let sign = if gradient[j] > 0.0 {
                1.0
            } else if gradient[j] < 0.0 {
                -1.0
            } else {
                0.0
            };
            perturbed[j] += sign_direction * config.step_size * sign;

            let delta = perturbed[j] - x[j];
            perturbed[j] = x[j] + delta.clamp(-config.epsilon, config.epsilon);
            perturbed[j] = perturbed[j].clamp(config.clip_min, config.clip_max);
        }
    }

    perturbed
}

fn deepfool_attack(
    x: &[f64],
    coefficients: &[f64],
    time: f64,
    event: usize,
    config: &AdversarialAttackConfig,
) -> Vec<f64> {
    let mut perturbed = x.to_vec();

    for _ in 0..config.n_iterations {
        let gradient = compute_gradient_survival(&perturbed, coefficients, time, event);
        let grad_norm: f64 = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();

        if grad_norm < crate::constants::DIVISION_FLOOR {
            break;
        }

        let linear_pred: f64 = perturbed
            .iter()
            .zip(coefficients.iter())
            .map(|(&xi, &c)| xi * c)
            .sum();
        let current_loss = if event == 1 {
            -linear_pred + linear_pred.exp() * time
        } else {
            linear_pred.exp() * time
        };

        let perturbation_size = (current_loss.abs() + 1e-4) / (grad_norm * grad_norm);

        for j in 0..perturbed.len() {
            perturbed[j] += perturbation_size * gradient[j];
            let delta = perturbed[j] - x[j];
            perturbed[j] = x[j] + delta.clamp(-config.epsilon, config.epsilon);
            perturbed[j] = perturbed[j].clamp(config.clip_min, config.clip_max);
        }
    }

    perturbed
}

fn predict_risk(x: &[f64], coefficients: &[f64]) -> f64 {
    let linear_pred: f64 = x
        .iter()
        .zip(coefficients.iter())
        .map(|(&xi, &c)| xi * c)
        .sum();
    linear_pred
        .clamp(
            crate::constants::LINEAR_PRED_CLAMP_MIN,
            crate::constants::LINEAR_PRED_CLAMP_MAX,
        )
        .exp()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}
