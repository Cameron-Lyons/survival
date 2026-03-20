#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TracerConfig::new(
            32,
            2,
            4,
            10,
            1,
            64,
            0.1,
            0.0001,
            0.00001,
            64,
            100,
            Some(5),
            0.1,
            1e-12,
            Some(42),
        )
        .unwrap();
        assert_eq!(config.embedding_dim, 32);
        assert_eq!(config.num_factorized_layers, 2);
        assert_eq!(config.num_attention_heads, 4);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            TracerConfig::new(
                0, 2, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 0, 4, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 2, 3, 10, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
        assert!(
            TracerConfig::new(
                32, 2, 4, 0, 1, 64, 0.1, 0.0001, 0.00001, 64, 100, None, 0.1, 1e-12, None
            )
            .is_err()
        );
    }

    #[test]
    fn test_multinomial_hazard_normalization() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let hazards = multinomial_hazard_normalization(&logits, 2, 3, 1);

        assert_eq!(hazards.len(), 6);
        for &h in &hazards {
            assert!((0.0..=1.0).contains(&h));
        }

        for t in 0..3 {
            let sum: f32 = (0..2).map(|k| hazards[k * 3 + t]).sum();
            assert!(sum < 1.0);
        }
    }

    #[test]
    fn test_event_weights() {
        let events = vec![0, 1, 1, 2, 0, 1, 2, 2];
        let weights = compute_event_weights(&events, 2);

        assert_eq!(weights.len(), 2);
        for &w in &weights {
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_tracer_basic() {
        let n_obs = 6;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mask: Vec<f64> = vec![1.0; total_size];
        let time_delta: Vec<f64> = vec![0.0; total_size];
        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 1,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 6,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert_eq!(model.get_num_events(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_tracer_competing_risks() {
        let n_obs = 6;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mask: Vec<f64> = vec![1.0; total_size];
        let time_delta: Vec<f64> = vec![0.0; total_size];
        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 2,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 6,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert_eq!(model.get_num_events(), 2);
    }

    #[test]
    fn test_tracer_with_missing_data() {
        let n_obs = 4;
        let max_seq_len = 3;
        let n_features = 2;
        let total_size = n_obs * max_seq_len * n_features;

        let x: Vec<f64> = (0..total_size).map(|i| (i as f64) * 0.1).collect();
        let mut mask: Vec<f64> = vec![1.0; total_size];
        mask[2] = 0.0;
        mask[5] = 0.0;
        mask[10] = 0.0;

        let time_delta: Vec<f64> = (0..total_size)
            .map(|i| if mask[i] < 0.5 { 1.0 } else { 0.0 })
            .collect();

        let seq_lengths: Vec<usize> = vec![3, 2, 3, 2];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 0];

        let config = TracerConfig {
            embedding_dim: 8,
            num_factorized_layers: 1,
            num_attention_heads: 2,
            num_durations: 3,
            num_events: 1,
            mlp_hidden_size: 8,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            weight_decay: 0.0,
            batch_size: 4,
            n_epochs: 2,
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
            seed: Some(42),
        };

        let model = fit_tracer_inner(
            &x,
            &mask,
            &time_delta,
            &seq_lengths,
            n_obs,
            max_seq_len,
            n_features,
            &time,
            &event,
            &config,
        );

        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_layer_norm_cpu() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0f32, 1.0, 1.0, 1.0];
        let beta = vec![0.0f32, 0.0, 0.0, 0.0];

        let result = layer_norm_cpu(&x, &gamma, &beta, 1e-12);

        assert_eq!(result.len(), 4);
        let mean: f64 = result.iter().sum::<f64>() / 4.0;
        assert!((mean).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_cpu() {
        let x = 0.5;
        let result = gelu_cpu(x);
        assert!(result > 0.0);
        assert!(result < x);
    }
}
