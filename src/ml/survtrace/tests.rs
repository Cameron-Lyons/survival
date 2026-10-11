#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SurvTraceConfig::new(
            16,
            3,
            2,
            64,
            0.0,
            0.1,
            5,
            1,
            8,
            0.001,
            64,
            100,
            0.0001,
            Some(42),
            Some(5),
            0.1,
            1e-12,
        )
        .unwrap();
        assert_eq!(config.hidden_size, 16);
        assert_eq!(config.num_hidden_layers, 3);
        assert_eq!(config.num_attention_heads, 2);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            SurvTraceConfig::new(
                0, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
        assert!(
            SurvTraceConfig::new(
                15, 3, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
        assert!(
            SurvTraceConfig::new(
                16, 0, 2, 64, 0.0, 0.1, 5, 1, 8, 0.001, 64, 100, 0.0001, None, None, 0.1, 1e-12
            )
            .is_err()
        );
    }

    #[test]
    fn test_optimizer_updates_output_head_and_exports_weights() {
        let config = SurvTraceConfig::new(
            4,
            1,
            2,
            8,
            0.0,
            0.0,
            3,
            1,
            4,
            0.01,
            2,
            1,
            0.0,
            Some(42),
            None,
            0.0,
            1e-6,
        )
        .unwrap();
        let device = super::super::utils::training_device();
        let mut network = SurvTraceNetwork::new(&device, 1, 2, &[2], &config);
        let bias = network.output_heads[0].bias.as_ref().unwrap().val();
        assert!(bias.is_autodiff());
        assert!(bias.is_require_grad());
        let before: Vec<f32> = bias.clone().into_data().try_to_vec().unwrap();
        let categorical: Tensor<2, Int> = Tensor::from_data([[0i64], [1]], &device);
        let numeric: Tensor<2> = Tensor::from_data([[1.0f32, 0.5], [0.0, 1.0]], &device);
        let loss = network
            .forward(Some(categorical), numeric, true)
            .remove(0)
            .sum();
        let gradients = loss.backward();
        assert!(bias.grad(&gradients).is_some());
        let gradients = GradientsParams::from_grads(gradients, &network);
        network = AdamConfig::new()
            .init()
            .step(config.learning_rate, network, gradients);

        let after: Vec<f32> = network.output_heads[0]
            .bias
            .as_ref()
            .unwrap()
            .val()
            .into_data()
            .try_to_vec()
            .unwrap();
        assert!(
            before
                .iter()
                .zip(&after)
                .all(|(before, after)| after < before)
        );
        let weights = extract_weights(&network, &config, &[2]);
        assert_eq!(weights.output_heads[0].1, after);
        assert_eq!(weights.output_heads[0].0.len(), 4 * 3);
        assert_eq!(weights.cat_embeddings[0].len(), 2 * 4);
    }

    #[test]
    fn test_survtrace_basic() {
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 1,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(None, &x_num, 6, 0, 2, &[], &time, &event, &config);
        assert_eq!(model.get_num_events(), 1);
        assert_eq!(model.get_num_durations(), 3);
        assert!(!model.train_loss.is_empty());
    }

    #[test]
    fn test_single_row_forward_and_validation() {
        let config = SurvTraceConfig::new(
            4,
            1,
            2,
            8,
            0.0,
            0.0,
            3,
            2,
            4,
            0.01,
            3,
            1,
            0.0,
            Some(42),
            None,
            0.25,
            1e-6,
        )
        .unwrap();
        let device = super::super::utils::training_device();
        for num_cat_features in [0, 1] {
            let cardinalities = vec![2; num_cat_features];
            let network =
                SurvTraceNetwork::new(&device, num_cat_features, 2, &cardinalities, &config);
            let categorical =
                (num_cat_features > 0).then(|| Tensor::<2, Int>::from_data([[1i64]], &device));
            let numeric: Tensor<2> = Tensor::from_data([[1.0f32, 0.5]], &device);
            for training in [true, false] {
                let outputs = network.forward(categorical.clone(), numeric.clone(), training);
                assert_eq!(outputs.len(), 2);
                for output in outputs {
                    assert_eq!(output.dims(), [1, 3]);
                    let values: Vec<f32> = output.into_data().try_to_vec().unwrap();
                    assert!(values.iter().all(|value| value.is_finite()));
                }
            }
        }

        let model = fit_survtrace_inner(
            Some(&[0i64, 1, 0, 1]),
            &[1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            4,
            1,
            2,
            &[2],
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 0, 2, 1],
            &config,
        );
        assert_eq!(model.val_loss.len(), 1);
        assert!(model.val_loss[0].is_finite());
    }

    #[test]
    fn test_survtrace_with_categorical() {
        let x_cat = vec![0i64, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1];
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 0, 1];
        let cat_cardinalities = vec![2, 2];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 1,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(
            Some(&x_cat),
            &x_num,
            6,
            2,
            2,
            &cat_cardinalities,
            &time,
            &event,
            &config,
        );
        assert_eq!(model.get_num_events(), 1);
    }

    #[test]
    fn test_survtrace_competing_risks() {
        let x_num = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 2, 0, 1, 2, 1];

        let config = SurvTraceConfig {
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            num_durations: 3,
            num_events: 2,
            vocab_size: 4,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            weight_decay: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
            layer_norm_eps: 1e-12,
        };

        let model = fit_survtrace_inner(None, &x_num, 6, 0, 2, &[], &time, &event, &config);
        assert_eq!(model.get_num_events(), 2);
    }

    #[test]
    fn test_duration_bins() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (bins, cuts) = compute_duration_bins(&times, 5);

        assert_eq!(bins.len(), 10);
        assert_eq!(cuts.len(), 6);

        for &bin in &bins {
            assert!(bin < 5);
        }
    }

    #[test]
    fn test_nll_loss() {
        let logits = vec![0.5f32, -0.3, 0.1, 0.8, -0.2, 0.4];
        let durations = vec![1, 0, 2];
        let events = vec![1, 0, 1];
        let indices: Vec<usize> = vec![0, 1, 2];

        let loss = compute_nll_logistic_hazard_loss(&logits, &durations, &events, 2, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_gelu_cpu() {
        let x = 0.5;
        let result = gelu_cpu(x);
        assert!(result > 0.0);
        assert!(result < x);
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
}
