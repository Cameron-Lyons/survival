#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_super_learner() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = SuperLearnerConfig::new(3, "nnls", false, true, Some(42)).unwrap();
        let result = super_learner_survival(
            time,
            event,
            covariates,
            vec![pred1, pred2],
            vec!["model1".to_string(), "model2".to_string()],
            config,
        )
        .unwrap();

        assert_eq!(result.weights.len(), 2);
        assert!((result.weights.iter().sum::<f64>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_stacking() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = StackingConfig::new(3, "cox", true, Some(42)).unwrap();
        let result = stacking_survival(time, event, vec![pred1, pred2], config).unwrap();

        assert_eq!(result.meta_coefficients.len(), 2);
    }

    #[test]
    fn test_componentwise_boosting() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, Some(10), 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates, config).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert!(!result.selected_features.is_empty());
    }

    #[test]
    fn test_blending() {
        let val_time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let val_event = vec![1, 0, 1, 0, 1];
        let val_pred1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let val_pred2 = vec![0.15, 0.25, 0.35, 0.45, 0.55];
        let test_pred1 = vec![0.2, 0.3, 0.4];
        let test_pred2 = vec![0.25, 0.35, 0.45];

        let result = blending_survival(
            val_time,
            val_event,
            vec![val_pred1, val_pred2],
            vec![test_pred1, test_pred2],
        )
        .unwrap();

        assert_eq!(result.blend_weights.len(), 2);
        assert_eq!(result.blended_predictions.len(), 3);
    }

    #[test]
    fn test_super_learner_config_validation() {
        let result = SuperLearnerConfig::new(1, "nnls", false, true, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stacking_config_validation() {
        let result = StackingConfig::new(1, "cox", true, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_componentwise_boosting_config_validation() {
        let result = ComponentwiseBoostingConfig::new(100, 0.0, None, 1.0, None);
        assert!(result.is_err());

        let result = ComponentwiseBoostingConfig::new(100, 1.5, None, 1.0, None);
        assert!(result.is_err());

        let result = ComponentwiseBoostingConfig::new(100, 0.1, None, 0.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_super_learner_empty_input() {
        let config = SuperLearnerConfig::new(3, "nnls", false, true, Some(42)).unwrap();
        let result = super_learner_survival(vec![], vec![], vec![], vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_super_learner_uniform_weights() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
        let pred1: Vec<f64> = (0..10).map(|i| 0.1 + i as f64 * 0.05).collect();
        let pred2: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.03).collect();

        let config = SuperLearnerConfig::new(3, "nnls", false, false, Some(42)).unwrap();
        let result = super_learner_survival(
            time,
            event,
            covariates,
            vec![pred1, pred2],
            vec!["m1".to_string(), "m2".to_string()],
            config,
        )
        .unwrap();

        assert!((result.weights[0] - 0.5).abs() < 1e-6);
        assert!((result.weights[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_componentwise_boosting_predict_risk() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, Some(10), 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates.clone(), config).unwrap();

        let risks = result.predict_risk(covariates);
        assert_eq!(risks.len(), 10);
        assert!(risks.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_stacking_empty_input() {
        let config = StackingConfig::new(3, "cox", true, Some(42)).unwrap();
        let result = stacking_survival(vec![], vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_blending_empty_input() {
        let result = blending_survival(vec![], vec![], vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_blending_mismatched_models() {
        let val_time = vec![1.0, 2.0, 3.0];
        let val_event = vec![1, 0, 1];
        let val_preds = vec![vec![0.1, 0.2, 0.3]];
        let test_preds = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let result = blending_survival(val_time, val_event, val_preds, test_preds);
        assert!(result.is_err());
    }

    #[test]
    fn test_componentwise_boosting_feature_importance() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1, 0.5])
            .collect();

        let config = ComponentwiseBoostingConfig::new(50, 0.1, None, 1.0, Some(42)).unwrap();
        let result = componentwise_boosting(time, event, covariates, config).unwrap();

        assert_eq!(result.feature_importance.len(), 3);
        let total: f64 = result.feature_importance.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
