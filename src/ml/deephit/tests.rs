#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeepHitConfig::new(
            Some(vec![64, 32]),
            Some(vec![16]),
            10,
            2,
            0.1,
            0.2,
            0.1,
            0.001,
            64,
            100,
            0.0001,
            Some(42),
            Some(5),
            0.1,
            true,
        )
        .unwrap();
        assert_eq!(config.shared_layers, vec![64, 32]);
        assert_eq!(config.cause_specific_layers, vec![16]);
        assert_eq!(config.num_risks, 2);
        assert_eq!(config.num_durations, 10);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DeepHitConfig::new(
                None, None, 0, 1, 0.1, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 0, 0.1, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 1, 1.5, 0.2, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
        assert!(
            DeepHitConfig::new(
                None, None, 10, 1, 0.1, 1.5, 0.1, 0.001, 64, 100, 0.0001, None, None, 0.1, true
            )
            .is_err()
        );
    }

    #[test]
    fn test_softmax_pmf() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let pmf = softmax_pmf(&logits, 2, 3, 1);

        assert_eq!(pmf.len(), 6);
        let sum: f32 = pmf.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        for &p in &pmf {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_deephit_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = DeepHitConfig {
            shared_layers: vec![8],
            cause_specific_layers: vec![4],
            num_durations: 3,
            num_risks: 1,
            dropout_rate: 0.0,
            alpha: 0.2,
            sigma: 0.1,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            use_batch_norm: false,
        };

        let model = fit_deephit_inner(&x, 6, 2, &time, &event, &config);
        assert_eq!(model.get_num_risks(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_deephit_competing_risks() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = DeepHitConfig {
            shared_layers: vec![8],
            cause_specific_layers: vec![4],
            num_durations: 3,
            num_risks: 2,
            dropout_rate: 0.0,
            alpha: 0.2,
            sigma: 0.1,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            use_batch_norm: false,
        };

        let model = fit_deephit_inner(&x, 6, 2, &time, &event, &config);
        assert_eq!(model.get_num_risks(), 2);
    }

    #[test]
    fn test_nll_loss() {
        let pmf = vec![0.1f32, 0.2, 0.3, 0.05, 0.15, 0.2, 0.1, 0.3, 0.2];
        let durations = vec![1, 0, 2];
        let events = vec![1, 0, 1];
        let indices: Vec<usize> = vec![0, 1, 2];

        let loss = compute_nll_loss(&pmf, &durations, &events, 1, 3, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
