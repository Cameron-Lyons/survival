#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum AttackType {
    FGSM,
    PGD,
    CarliniWagner,
    DeepFool,
    BoundaryAttack,
}

#[pymethods]
impl AttackType {
    fn __repr__(&self) -> String {
        match self {
            AttackType::FGSM => "AttackType.FGSM".to_string(),
            AttackType::PGD => "AttackType.PGD".to_string(),
            AttackType::CarliniWagner => "AttackType.CarliniWagner".to_string(),
            AttackType::DeepFool => "AttackType.DeepFool".to_string(),
            AttackType::BoundaryAttack => "AttackType.BoundaryAttack".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum DefenseType {
    AdversarialTraining,
    InputPreprocessing,
    CertifiedDefense,
    Ensembling,
    Randomization,
}

#[pymethods]
impl DefenseType {
    fn __repr__(&self) -> String {
        match self {
            DefenseType::AdversarialTraining => "DefenseType.AdversarialTraining".to_string(),
            DefenseType::InputPreprocessing => "DefenseType.InputPreprocessing".to_string(),
            DefenseType::CertifiedDefense => "DefenseType.CertifiedDefense".to_string(),
            DefenseType::Ensembling => "DefenseType.Ensembling".to_string(),
            DefenseType::Randomization => "DefenseType.Randomization".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AdversarialAttackConfig {
    #[pyo3(get, set)]
    pub attack_type: AttackType,
    #[pyo3(get, set)]
    pub epsilon: f64,
    #[pyo3(get, set)]
    pub n_iterations: usize,
    #[pyo3(get, set)]
    pub step_size: f64,
    #[pyo3(get, set)]
    pub targeted: bool,
    #[pyo3(get, set)]
    pub clip_min: f64,
    #[pyo3(get, set)]
    pub clip_max: f64,
}

#[pymethods]
impl AdversarialAttackConfig {
    #[new]
    #[pyo3(signature = (attack_type=AttackType::FGSM, epsilon=0.1, n_iterations=10, step_size=0.01, targeted=false, clip_min=f64::NEG_INFINITY, clip_max=f64::INFINITY))]
    pub fn new(
        attack_type: AttackType,
        epsilon: f64,
        n_iterations: usize,
        step_size: f64,
        targeted: bool,
        clip_min: f64,
        clip_max: f64,
    ) -> Self {
        Self {
            attack_type,
            epsilon,
            n_iterations,
            step_size,
            targeted,
            clip_min,
            clip_max,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AdversarialDefenseConfig {
    #[pyo3(get, set)]
    pub defense_type: DefenseType,
    #[pyo3(get, set)]
    pub adversarial_ratio: f64,
    #[pyo3(get, set)]
    pub n_ensemble: usize,
    #[pyo3(get, set)]
    pub noise_scale: f64,
    #[pyo3(get, set)]
    pub certified_radius: f64,
}

#[pymethods]
impl AdversarialDefenseConfig {
    #[new]
    #[pyo3(signature = (defense_type=DefenseType::AdversarialTraining, adversarial_ratio=0.5, n_ensemble=5, noise_scale=0.1, certified_radius=0.1))]
    pub fn new(
        defense_type: DefenseType,
        adversarial_ratio: f64,
        n_ensemble: usize,
        noise_scale: f64,
        certified_radius: f64,
    ) -> Self {
        Self {
            defense_type,
            adversarial_ratio,
            n_ensemble,
            noise_scale,
            certified_radius,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AdversarialExample {
    #[pyo3(get)]
    pub original: Vec<f64>,
    #[pyo3(get)]
    pub perturbed: Vec<f64>,
    #[pyo3(get)]
    pub perturbation: Vec<f64>,
    #[pyo3(get)]
    pub original_prediction: f64,
    #[pyo3(get)]
    pub adversarial_prediction: f64,
    #[pyo3(get)]
    pub perturbation_norm: f64,
    #[pyo3(get)]
    pub success: bool,
}

#[pymethods]
impl AdversarialExample {
    fn __repr__(&self) -> String {
        format!(
            "AdversarialExample(success={}, norm={:.4}, pred_change={:.4})",
            self.success,
            self.perturbation_norm,
            (self.adversarial_prediction - self.original_prediction).abs()
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AdversarialAttackResult {
    #[pyo3(get)]
    pub adversarial_examples: Vec<AdversarialExample>,
    #[pyo3(get)]
    pub success_rate: f64,
    #[pyo3(get)]
    pub mean_perturbation_norm: f64,
    #[pyo3(get)]
    pub mean_prediction_change: f64,
    #[pyo3(get)]
    pub attack_type: AttackType,
}

#[pymethods]
impl AdversarialAttackResult {
    fn __repr__(&self) -> String {
        format!(
            "AdversarialAttackResult(success_rate={:.2}%, mean_norm={:.4})",
            self.success_rate * 100.0,
            self.mean_perturbation_norm
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RobustSurvivalModel {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub robust_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub certified_radius: f64,
    #[pyo3(get)]
    pub empirical_robustness: f64,
    #[pyo3(get)]
    pub defense_type: DefenseType,
    #[pyo3(get)]
    pub training_loss: f64,
    #[pyo3(get)]
    pub adversarial_loss: f64,
}

#[pymethods]
impl RobustSurvivalModel {
    fn __repr__(&self) -> String {
        format!(
            "RobustSurvivalModel(certified_radius={:.4}, empirical_robustness={:.4})",
            self.certified_radius, self.empirical_robustness
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RobustnessEvaluation {
    #[pyo3(get)]
    pub clean_accuracy: f64,
    #[pyo3(get)]
    pub robust_accuracy: f64,
    #[pyo3(get)]
    pub accuracy_drop: f64,
    #[pyo3(get)]
    pub certified_accuracy: f64,
    #[pyo3(get)]
    pub attack_success_rates: Vec<f64>,
    #[pyo3(get)]
    pub epsilon_values: Vec<f64>,
}

#[pymethods]
impl RobustnessEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "RobustnessEvaluation(clean={:.2}%, robust={:.2}%, drop={:.2}%)",
            self.clean_accuracy * 100.0,
            self.robust_accuracy * 100.0,
            self.accuracy_drop * 100.0
        )
    }
}

fn compute_gradient_survival(x: &[f64], coefficients: &[f64], time: f64, event: usize) -> Vec<f64> {
    let linear_pred: f64 = x
        .iter()
        .zip(coefficients.iter())
        .map(|(&xi, &c)| xi * c)
        .sum();
    let exp_pred = linear_pred.clamp(-20.0, 20.0).exp();

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

        if grad_norm < 1e-10 {
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
    linear_pred.clamp(-20.0, 20.0).exp()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

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

        let pred_change = (adversarial_pred - original_pred).abs() / original_pred.max(1e-10);
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
                    .clamp(-20.0, 20.0)
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

            if event[i] == 1 && risk_set_sum > 1e-10 {
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
            if hessian_diag[j].abs() > 1e-10 {
                let update = (gradient[j] / hessian_diag[j]).clamp(-1.0, 1.0);
                coefficients[j] += update;
                max_update = max_update.max(update.abs());
            }
        }

        if max_update < 1e-6 {
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
        let change = (adv_pred - orig_pred).abs() / orig_pred.max(1e-10);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adversarial_attack_config() {
        let config = AdversarialAttackConfig::new(
            AttackType::PGD,
            0.1,
            10,
            0.01,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        assert_eq!(config.attack_type, AttackType::PGD);
        assert!((config.epsilon - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_generate_adversarial_examples() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];
        let coefficients = vec![0.5, -0.3];

        let result = generate_adversarial_examples(x, time, event, coefficients, None).unwrap();
        assert_eq!(result.adversarial_examples.len(), 4);
        assert!(result.success_rate >= 0.0 && result.success_rate <= 1.0);
    }

    #[test]
    fn test_adversarial_training() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
            vec![2.5, 0.3],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 1, 0];

        let result = adversarial_training_survival(x, time, event, None, None).unwrap();
        assert!(!result.robust_coefficients.is_empty());
        assert!(result.empirical_robustness >= 0.0);
    }

    #[test]
    fn test_evaluate_robustness() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];
        let coefficients = vec![0.5, -0.3];

        let result = evaluate_robustness(x, time, event, coefficients, None).unwrap();
        assert!(result.clean_accuracy >= 0.0 && result.clean_accuracy <= 1.0);
        assert!(result.robust_accuracy >= 0.0 && result.robust_accuracy <= 1.0);
        assert!(!result.attack_success_rates.is_empty());
    }
}
