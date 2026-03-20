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

    #[test]
    fn test_fgsm_perturbation_bounded() {
        let x = vec![1.0, 0.5, -0.3];
        let coefficients = vec![0.5, -0.3, 0.2];
        let config =
            AdversarialAttackConfig::new(AttackType::FGSM, 0.1, 10, 0.01, false, -10.0, 10.0);

        let perturbed = fgsm_attack(&x, &coefficients, 1.0, 1, 0.1, &config);
        assert_eq!(perturbed.len(), 3);
        for (i, &p) in perturbed.iter().enumerate() {
            assert!((p - x[i]).abs() <= 0.1 + 1e-10);
            assert!((-10.0..=10.0).contains(&p));
        }
    }

    #[test]
    fn test_pgd_perturbation_bounded() {
        let x = vec![1.0, 0.5];
        let coefficients = vec![0.5, -0.3];
        let config = AdversarialAttackConfig::new(
            AttackType::PGD,
            0.1,
            20,
            0.01,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );

        let perturbed = pgd_attack(&x, &coefficients, 1.0, 1, &config);
        assert_eq!(perturbed.len(), 2);
        for (i, &p) in perturbed.iter().enumerate() {
            assert!((p - x[i]).abs() <= 0.1 + 1e-10);
        }
    }

    #[test]
    fn test_generate_adversarial_empty_input() {
        let result = generate_adversarial_examples(vec![], vec![], vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_adversarial_training_empty_input() {
        let result = adversarial_training_survival(vec![], vec![], vec![], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_robustness_empty_input() {
        let result = evaluate_robustness(vec![], vec![], vec![], vec![], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_deepfool_attack() {
        let x = vec![1.0, 0.5, 0.3];
        let coefficients = vec![0.5, -0.3, 0.2];
        let config = AdversarialAttackConfig::new(
            AttackType::DeepFool,
            0.5,
            10,
            0.01,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );

        let perturbed = deepfool_attack(&x, &coefficients, 1.0, 1, &config);
        assert_eq!(perturbed.len(), 3);
        for (i, &p) in perturbed.iter().enumerate() {
            assert!((p - x[i]).abs() <= 0.5 + 1e-10);
        }
    }

    #[test]
    fn test_adversarial_pgd_attack_result() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
            vec![2.5, 0.3],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 1, 0];
        let coefficients = vec![0.5, -0.3];

        let config = AdversarialAttackConfig::new(
            AttackType::PGD,
            0.2,
            20,
            0.05,
            false,
            f64::NEG_INFINITY,
            f64::INFINITY,
        );
        let result =
            generate_adversarial_examples(x, time, event, coefficients, Some(config)).unwrap();
        assert_eq!(result.adversarial_examples.len(), 5);
        assert!(result.mean_perturbation_norm >= 0.0);
        assert!(result.mean_prediction_change >= 0.0);
    }

    #[test]
    fn test_evaluate_robustness_custom_epsilons() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];
        let coefficients = vec![0.5, -0.3];
        let epsilons = vec![0.01, 0.1, 1.0];

        let result =
            evaluate_robustness(x, time, event, coefficients, Some(epsilons.clone())).unwrap();
        assert_eq!(result.attack_success_rates.len(), 3);
        assert_eq!(result.epsilon_values, epsilons);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-10);

        let zeros = vec![0.0, 0.0, 0.0];
        assert!((l2_norm(&zeros)).abs() < 1e-10);
    }

    #[test]
    fn test_predict_risk_fn() {
        let x = vec![1.0, 2.0];
        let coefficients = vec![0.0, 0.0];
        let risk = predict_risk(&x, &coefficients);
        assert!((risk - 1.0).abs() < 1e-10);
    }
}
