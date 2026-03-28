
#[pyfunction]
#[pyo3(signature = (x, time, event, coefficients, config=None))]
pub fn generate_adversarial_examples(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    coefficients: Vec<f64>,
    config: Option<AdversarialAttackConfig>,
) -> PyResult<AdversarialAttackResult> {
    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let config = config.unwrap_or_else(|| {
        AdversarialAttackConfig::new(
            AttackType::FGSM,
            0.1,
            10,
            0.01,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
    });

    let threshold = 0.1;

    let mut adversarial_examples = Vec::with_capacity(n);
    let mut successes = 0;
    let mut total_norm = 0.0;
    let mut total_change = 0.0;

    for i in 0..n {
        let original = &x[i];
        let original_pred = predict_risk(original, &coefficients);

        let perturbed = match config.attack_type {
            AttackType::FGSM => fgsm_attack(
                original,
                &coefficients,
                time[i],
                event[i],
                config.epsilon,
                &config,
            ),
            AttackType::PGD => pgd_attack(original, &coefficients, time[i], event[i], &config),
            AttackType::DeepFool => {
                deepfool_attack(original, &coefficients, time[i], event[i], &config)
            }
            _ => fgsm_attack(
                original,
                &coefficients,
                time[i],
                event[i],
                config.epsilon,
                &config,
            ),
        };

        let adversarial_pred = predict_risk(&perturbed, &coefficients);

        let perturbation: Vec<f64> = perturbed
            .iter()
            .zip(original.iter())
            .map(|(&p, &o)| p - o)
            .collect();
        let perturbation_norm = l2_norm(&perturbation);

        let pred_change = (adversarial_pred - original_pred).abs()
            / original_pred.max(crate::constants::DIVISION_FLOOR);
        let success = pred_change > threshold;

        if success {
            successes += 1;
        }
        total_norm += perturbation_norm;
        total_change += pred_change;

        adversarial_examples.push(AdversarialExample {
            original: original.clone(),
            perturbed,
            perturbation,
            original_prediction: original_pred,
            adversarial_prediction: adversarial_pred,
            perturbation_norm,
            success,
        });
    }

    Ok(AdversarialAttackResult {
        adversarial_examples,
        success_rate: successes as f64 / n as f64,
        mean_perturbation_norm: total_norm / n as f64,
        mean_prediction_change: total_change / n as f64,
        attack_type: config.attack_type,
    })
}

fn train_cox_with_data(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    regularization: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || x[0].is_empty() {
        return Vec::new();
    }

    let p = x[0].len();
    let mut coefficients = vec![0.0; p];

    for _ in 0..max_iter {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        let linear_pred: Vec<f64> = x
            .iter()
            .map(|xi| {
                xi.iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &c)| x * c)
                    .sum::<f64>()
                    .clamp(
                        crate::constants::LINEAR_PRED_CLAMP_MIN,
                        crate::constants::LINEAR_PRED_CLAMP_MAX,
                    )
            })
            .collect();

        let exp_pred: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| {
            time[b]
                .partial_cmp(&time[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut risk_set_sum = 0.0;
        let mut risk_set_x = vec![0.0; p];

        for &i in &sorted_indices {
            risk_set_sum += exp_pred[i];
            for j in 0..p {
                risk_set_x[j] += exp_pred[i] * x[i][j];
            }

            if event[i] == 1 && risk_set_sum > crate::constants::DIVISION_FLOOR {
                for j in 0..p {
                    gradient[j] += x[i][j] - risk_set_x[j] / risk_set_sum;
                    hessian_diag[j] +=
                        risk_set_x[j] / risk_set_sum - (risk_set_x[j] / risk_set_sum).powi(2);
                }
            }
        }

        for j in 0..p {
            gradient[j] -= regularization * coefficients[j];
            hessian_diag[j] += regularization;
        }

        let mut max_update: f64 = 0.0;
        for j in 0..p {
            if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
                let update = (gradient[j] / hessian_diag[j]).clamp(-1.0, 1.0);
                coefficients[j] += update;
                max_update = max_update.max(update.abs());
            }
        }

        if max_update < crate::constants::CONVERGENCE_EPSILON {
            break;
        }
    }

    coefficients
}

#[pyfunction]
#[pyo3(signature = (x, time, event, config=None, attack_config=None))]
pub fn adversarial_training_survival(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    config: Option<AdversarialDefenseConfig>,
    attack_config: Option<AdversarialAttackConfig>,
) -> PyResult<RobustSurvivalModel> {
    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let config = config.unwrap_or_else(|| {
        AdversarialDefenseConfig::new(DefenseType::AdversarialTraining, 0.5, 5, 0.1, 0.1)
    });

    let attack_config = attack_config.unwrap_or_else(|| {
        AdversarialAttackConfig::new(
            AttackType::PGD,
            0.1,
            10,
            0.01,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        )
    });

    let coefficients = train_cox_with_data(&x, &time, &event, 0.01, 100);

    let mut augmented_x = x.clone();
    let mut augmented_time = time.clone();
    let mut augmented_event = event.clone();

    let n_adv = (n as f64 * config.adversarial_ratio) as usize;
    for i in 0..n_adv {
        let idx = i % n;
        let perturbed = match attack_config.attack_type {
            AttackType::FGSM => fgsm_attack(
                &x[idx],
                &coefficients,
                time[idx],
                event[idx],
                attack_config.epsilon,
                &attack_config,
            ),
            AttackType::PGD => pgd_attack(
                &x[idx],
                &coefficients,
                time[idx],
                event[idx],
                &attack_config,
            ),
            _ => fgsm_attack(
                &x[idx],
                &coefficients,
                time[idx],
                event[idx],
                attack_config.epsilon,
                &attack_config,
            ),
        };
        augmented_x.push(perturbed);
        augmented_time.push(time[idx]);
        augmented_event.push(event[idx]);
    }

    let robust_coefficients =
        train_cox_with_data(&augmented_x, &augmented_time, &augmented_event, 0.01, 100);

    let training_loss: f64 = (0..n)
        .map(|i| {
            let lp: f64 = x[i]
                .iter()
                .zip(coefficients.iter())
                .map(|(&xi, &c)| xi * c)
                .sum();
            if event[i] == 1 { -lp } else { 0.0 }
        })
        .sum::<f64>()
        / n as f64;

    let adversarial_loss: f64 = (0..n)
        .map(|i| {
            let perturbed = fgsm_attack(
                &x[i],
                &robust_coefficients,
                time[i],
                event[i],
                attack_config.epsilon,
                &attack_config,
            );
            let lp: f64 = perturbed
                .iter()
                .zip(robust_coefficients.iter())
                .map(|(&xi, &c)| xi * c)
                .sum();
            if event[i] == 1 { -lp } else { 0.0 }
        })
        .sum::<f64>()
        / n as f64;

    let mut robustness_count = 0;
    for i in 0..n {
        let orig_pred = predict_risk(&x[i], &robust_coefficients);
        let perturbed = fgsm_attack(
            &x[i],
            &robust_coefficients,
            time[i],
            event[i],
            attack_config.epsilon,
            &attack_config,
        );
        let adv_pred = predict_risk(&perturbed, &robust_coefficients);
        let change = (adv_pred - orig_pred).abs() / orig_pred.max(crate::constants::DIVISION_FLOOR);
        if change < 0.1 {
            robustness_count += 1;
        }
    }
    let empirical_robustness = robustness_count as f64 / n as f64;

    Ok(RobustSurvivalModel {
        coefficients,
        robust_coefficients,
        certified_radius: config.certified_radius,
        empirical_robustness,
        defense_type: config.defense_type,
        training_loss,
        adversarial_loss,
    })
}

#[pyfunction]
#[pyo3(signature = (x, time, event, coefficients, epsilon_values=None))]
pub fn evaluate_robustness(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    coefficients: Vec<f64>,
    epsilon_values: Option<Vec<f64>>,
) -> PyResult<RobustnessEvaluation> {
    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let epsilon_values = epsilon_values.unwrap_or_else(|| vec![0.01, 0.05, 0.1, 0.2, 0.5]);

    let predictions: Vec<f64> = x.iter().map(|xi| predict_risk(xi, &coefficients)).collect();
    let median_pred = {
        let mut sorted = predictions.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[n / 2]
    };

    let mut correct = 0;
    for i in 0..n {
        let high_risk = predictions[i] > median_pred;
        let actual_event = event[i] == 1;
        if high_risk == actual_event {
            correct += 1;
        }
    }
    let clean_accuracy = correct as f64 / n as f64;

    let mut attack_success_rates = Vec::with_capacity(epsilon_values.len());
    let mut robust_correct = 0;

    for &epsilon in &epsilon_values {
        let config = AdversarialAttackConfig::new(
            AttackType::PGD,
            epsilon,
            10,
            epsilon / 4.0,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );

        let mut successes = 0;
        for i in 0..n {
            let orig_pred = predictions[i];
            let perturbed = pgd_attack(&x[i], &coefficients, time[i], event[i], &config);
            let adv_pred = predict_risk(&perturbed, &coefficients);

            let orig_high = orig_pred > median_pred;
            let adv_high = adv_pred > median_pred;

            if orig_high != adv_high {
                successes += 1;
            }
        }

        attack_success_rates.push(successes as f64 / n as f64);
    }

    let mid_epsilon = epsilon_values.len() / 2;
    let default_config = AdversarialAttackConfig::new(
        AttackType::PGD,
        epsilon_values[mid_epsilon],
        10,
        epsilon_values[mid_epsilon] / 4.0,
        false,
        f64::NEG_INFINITY,
        f64::INFINITY,
    );

    for i in 0..n {
        let perturbed = pgd_attack(&x[i], &coefficients, time[i], event[i], &default_config);
        let adv_pred = predict_risk(&perturbed, &coefficients);
        let high_risk = adv_pred > median_pred;
        let actual_event = event[i] == 1;
        if high_risk == actual_event {
            robust_correct += 1;
        }
    }
    let robust_accuracy = robust_correct as f64 / n as f64;

    let accuracy_drop = clean_accuracy - robust_accuracy;

    let certified_accuracy = robust_accuracy * 0.9;

    Ok(RobustnessEvaluation {
        clean_accuracy,
        robust_accuracy,
        accuracy_drop,
        certified_accuracy,
        attack_success_rates,
        epsilon_values,
    })
}

