#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_data() -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let x = vec![
            1.0, 0.5, 0.2, 0.8, 0.3, 0.1, 0.6, 0.7, 0.4, 0.4, 0.2, 0.8, 0.9, 0.1, 0.3, 0.3, 0.8,
            0.5, 0.7, 0.4, 0.6, 0.2, 0.6, 0.9,
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1];
        (x, time, status)
    }

    #[test]
    fn test_config_default() {
        let config = DeepSurvConfig::new(
            Some(vec![32, 16]),
            Activation::ReLU,
            0.1,
            0.01,
            64,
            50,
            0.001,
            Some(42),
            Some(10),
            0.2,
        )
        .unwrap();
        assert_eq!(config.hidden_layers, vec![32, 16]);
        assert_eq!(config.n_epochs, 50);
        assert_eq!(config.dropout_rate, 0.1);
        assert_eq!(config.learning_rate, 0.01);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                -0.1,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                1.0,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                -0.01,
                64,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                0,
                50,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                64,
                0,
                0.0,
                None,
                None,
                0.1
            )
            .is_err()
        );
        assert!(
            DeepSurvConfig::new(
                None,
                Activation::ReLU,
                0.1,
                0.01,
                64,
                50,
                0.0,
                None,
                None,
                1.0
            )
            .is_err()
        );
    }

    #[test]
    fn test_deep_surv_basic() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, 6, 2, &time, &status, &config);
        assert_eq!(model.get_n_features(), 2);
        assert_eq!(model.get_hidden_layers(), vec![4]);
        assert!(!model.train_loss.is_empty());
        assert_eq!(model.train_loss.len(), 5);
        assert!(!model.unique_times.is_empty());
        assert!(!model.baseline_hazard.is_empty());
    }

    #[test]
    fn test_predict_risk() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 6,
            n_epochs: 3,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, 6, 2, &time, &status, &config);
        let risks = model.predict_risk(x.clone(), 6).unwrap();
        assert_eq!(risks.len(), 6);
        for risk in &risks {
            assert!(risk.is_finite());
        }
    }

    #[test]
    fn test_predict_survival() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4, 2],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        let survival = model.predict_survival(x.clone(), n_obs).unwrap();

        assert_eq!(survival.len(), n_obs);
        for (i, surv) in survival.iter().enumerate() {
            assert_eq!(surv.len(), model.unique_times.len());
            for s in surv {
                assert!(
                    *s >= 0.0 && *s <= 1.0,
                    "Survival prob at row {} should be in [0,1]",
                    i
                );
            }
            for j in 1..surv.len() {
                assert!(
                    surv[j] <= surv[j - 1] + 1e-10,
                    "Survival should be monotonically decreasing"
                );
            }
        }
    }

    #[test]
    fn test_predict_cumulative_hazard() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::Tanh,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        let cumhaz = model.predict_cumulative_hazard(x.clone(), n_obs).unwrap();

        assert_eq!(cumhaz.len(), n_obs);
        for (i, ch) in cumhaz.iter().enumerate() {
            assert_eq!(ch.len(), model.unique_times.len());
            for c in ch {
                assert!(
                    *c >= 0.0,
                    "Cumulative hazard at row {} should be non-negative",
                    i
                );
            }
            for j in 1..ch.len() {
                assert!(
                    ch[j] >= ch[j - 1] - 1e-10,
                    "Cumulative hazard should be monotonically increasing"
                );
            }
        }
    }

    #[test]
    fn test_predict_survival_time() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 10,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);

        let median_times = model
            .predict_median_survival_time(x.clone(), n_obs)
            .unwrap();
        assert_eq!(median_times.len(), n_obs);

        let q75_times = model.predict_survival_time(x.clone(), n_obs, 0.75).unwrap();
        assert_eq!(q75_times.len(), n_obs);

        let q25_times = model.predict_survival_time(x.clone(), n_obs, 0.25).unwrap();
        assert_eq!(q25_times.len(), n_obs);
    }

    #[test]
    fn test_validation_and_early_stopping() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![4],
            activation: Activation::ReLU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 4,
            n_epochs: 50,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: Some(5),
            validation_fraction: 0.25,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);

        assert!(!model.val_loss.is_empty());
        assert_eq!(model.train_loss.len(), model.val_loss.len());
        assert!(model.train_loss.len() <= 50);
    }

    #[test]
    fn test_all_activations() {
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 1];

        for activation in [Activation::ReLU, Activation::SELU, Activation::Tanh] {
            let config = DeepSurvConfig {
                hidden_layers: vec![2],
                activation,
                dropout_rate: 0.0,
                learning_rate: 0.01,
                batch_size: 3,
                n_epochs: 2,
                l2_reg: 0.0,
                seed: Some(42),
                early_stopping_patience: None,
                validation_fraction: 0.0,
            };

            let model = fit_deep_surv_inner(&x, 3, 2, &time, &status, &config);
            let risks = model.predict_risk(x.clone(), 3).unwrap();
            assert_eq!(risks.len(), 3);
            for risk in &risks {
                assert!(
                    risk.is_finite(),
                    "Risk should be finite for {:?}",
                    activation
                );
            }
        }
    }

    #[test]
    fn test_multi_layer_network() {
        let (x, time, status) = get_test_data();
        let n_obs = 8;
        let n_vars = 3;

        let config = DeepSurvConfig {
            hidden_layers: vec![16, 8, 4],
            activation: Activation::SELU,
            dropout_rate: 0.0,
            learning_rate: 0.01,
            batch_size: 8,
            n_epochs: 5,
            l2_reg: 0.0,
            seed: Some(42),
            early_stopping_patience: None,
            validation_fraction: 0.0,
        };

        let model = fit_deep_surv_inner(&x, n_obs, n_vars, &time, &status, &config);
        assert_eq!(model.get_hidden_layers(), vec![16, 8, 4]);
        let risks = model.predict_risk(x.clone(), n_obs).unwrap();
        assert_eq!(risks.len(), n_obs);
    }

    #[test]
    fn test_cox_gradient_computation() {
        let risk_scores = vec![0.5f32, -0.3, 0.1, 0.8];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 1];
        let indices: Vec<usize> = (0..4).collect();

        let gradients = compute_cox_gradient_cpu(&risk_scores, &time, &status, &indices);
        assert_eq!(gradients.len(), 4);
        for g in &gradients {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_cox_loss_computation() {
        let risk_scores = vec![0.5f32, -0.3, 0.1, 0.8];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 1];
        let indices: Vec<usize> = (0..4).collect();

        let loss = compute_cox_loss_cpu(&risk_scores, &time, &status, &indices);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_baseline_hazard_computation() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 1, 0];
        let risk_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let unique_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let baseline = compute_baseline_hazard(&time, &status, &risk_scores, &unique_times);

        assert_eq!(baseline.len(), unique_times.len());
        for b in &baseline {
            assert!(*b >= 0.0);
        }
        for i in 1..baseline.len() {
            assert!(
                baseline[i] >= baseline[i - 1],
                "Baseline hazard should be monotonically increasing"
            );
        }
    }
}
