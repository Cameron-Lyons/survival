#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 4), 5);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 5), 252);
    }

    #[test]
    fn test_sample_coalitions() {
        let (coalitions, sizes) = sample_coalitions(5, 100, 42);
        assert_eq!(coalitions.len(), 100);
        assert_eq!(sizes.len(), 100);

        assert!(coalitions[0].iter().all(|&c| !c));
        assert_eq!(sizes[0], 0);

        assert!(coalitions[1].iter().all(|&c| c));
        assert_eq!(sizes[1], 5);

        for (i, (coalition, &size)) in coalitions.iter().zip(sizes.iter()).enumerate() {
            let actual_size = coalition.iter().filter(|&&c| c).count();
            assert_eq!(actual_size, size, "Coalition {} size mismatch", i);
        }
    }

    #[test]
    fn test_kernel_weights() {
        let weights = compute_shapley_kernel_weights(4, &[0, 1, 2, 3, 4]);

        assert!(weights[0].is_infinite());
        assert!(weights[4].is_infinite());

        assert!(weights[1] > 0.0);
        assert!(weights[2] > 0.0);
        assert!(weights[3] > 0.0);
    }

    #[test]
    fn test_weighted_least_squares() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0, 1.0];

        let result = weighted_least_squares(&x, &y, &weights, 3, 2);
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_survshap_basic() {
        let n_explain = 2;
        let n_background = 3;
        let n_features = 3;
        let n_times = 4;

        let x_explain: Vec<f64> = (0..(n_explain * n_features))
            .map(|i| (i as f64) * 0.1)
            .collect();
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.05)
            .collect();

        let predictions_explain: Vec<f64> = (0..(n_explain * n_times))
            .map(|i| 1.0 - (i as f64) * 0.05)
            .collect();
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.9 - (i as f64) * 0.02)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let config = SurvShapConfig::new(100, n_background, Some(42), false).unwrap();

        let result = survshap(
            x_explain,
            x_background,
            predictions_explain.clone(),
            predictions_background,
            time_points.clone(),
            n_explain,
            n_background,
            n_features,
            Some(&config),
            Some(AggregationMethod::Mean),
        )
        .unwrap();

        assert_eq!(result.shap_values.len(), n_explain);
        assert_eq!(result.shap_values[0].len(), n_features);
        assert_eq!(result.shap_values[0][0].len(), n_times);
        assert_eq!(result.base_value.len(), n_times);
        assert_eq!(result.time_points.len(), n_times);
        assert!(result.aggregated_importance.is_some());
        assert_eq!(result.aggregated_importance.unwrap().len(), n_features);
    }

    #[test]
    fn test_aggregation_methods() {
        let shap_values = vec![vec![vec![0.1, 0.2, 0.3, 0.4]; 2]; 3];
        let time_points = vec![1.0, 2.0, 3.0, 4.0];

        for method in [
            AggregationMethod::Mean,
            AggregationMethod::Integral,
            AggregationMethod::MaxAbsolute,
            AggregationMethod::TimeWeighted,
        ] {
            let result =
                aggregate_survshap(shap_values.clone(), time_points.clone(), method).unwrap();
            assert_eq!(result.len(), 2);
            assert!(result.iter().all(|&v| v.is_finite() && v >= 0.0));
        }
    }

    #[test]
    fn test_shap_additivity() {
        let n_explain = 1;
        let n_background = 10;
        let n_features = 2;
        let n_times = 3;

        let x_explain: Vec<f64> = vec![0.5, 0.5];
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.1 % 1.0)
            .collect();

        let predictions_explain: Vec<f64> = vec![0.9, 0.8, 0.7];
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.95 - (i as f64) * 0.01)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0];

        let config = SurvShapConfig::new(500, n_background, Some(42), false).unwrap();

        let result = survshap(
            x_explain,
            x_background,
            predictions_explain.clone(),
            predictions_background,
            time_points,
            n_explain,
            n_background,
            n_features,
            Some(&config),
            None,
        )
        .unwrap();

        for (t, &pred) in predictions_explain.iter().enumerate().take(n_times) {
            let shap_sum: f64 = (0..n_features).map(|f| result.shap_values[0][f][t]).sum();
            let reconstructed = result.base_value[t] + shap_sum;
            let error = (reconstructed - pred).abs();
            assert!(
                error < 0.5,
                "Additivity check failed at t={}: reconstructed={}, actual={}, error={}",
                t,
                reconstructed,
                pred,
                error
            );
        }
    }

    #[test]
    fn test_config_validation() {
        assert!(SurvShapConfig::new(1, 100, None, true).is_err());
        assert!(SurvShapConfig::new(100, 0, None, true).is_err());
        assert!(SurvShapConfig::new(100, 50, Some(42), false).is_ok());
    }

    #[test]
    fn test_feature_ranking() {
        let n_explain = 3;
        let n_features = 4;
        let n_times = 3;

        let mut shap_values = vec![vec![vec![0.0; n_times]; n_features]; n_explain];
        for sample in shap_values.iter_mut() {
            for (f, feature) in sample.iter_mut().enumerate() {
                for (t, val) in feature.iter_mut().enumerate() {
                    *val = (f + 1) as f64 * 0.1 + (t as f64) * 0.01;
                }
            }
        }

        let result = SurvShapResult {
            shap_values,
            base_value: vec![0.5; n_times],
            time_points: vec![1.0, 2.0, 3.0],
            aggregated_importance: None,
        };

        let ranking = result
            .feature_ranking(AggregationMethod::Mean, Some(2))
            .unwrap();
        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0].feature_idx, 3);
        assert_eq!(ranking[1].feature_idx, 2);
    }

    #[test]
    fn test_bootstrap_survshap() {
        let n_explain = 2;
        let n_background = 5;
        let n_features = 2;
        let n_times = 3;

        let x_explain: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let x_background: Vec<f64> = (0..(n_background * n_features))
            .map(|i| (i as f64) * 0.1)
            .collect();

        let predictions_explain: Vec<f64> = vec![0.9, 0.8, 0.7, 0.85, 0.75, 0.65];
        let predictions_background: Vec<f64> = (0..(n_background * n_times))
            .map(|i| 0.9 - (i as f64) * 0.02)
            .collect();

        let time_points: Vec<f64> = vec![1.0, 2.0, 3.0];

        let config = SurvShapConfig::new(50, n_background, Some(42), false).unwrap();

        let result = survshap_bootstrap(
            x_explain,
            x_background,
            predictions_explain,
            predictions_background,
            time_points,
            n_explain,
            n_background,
            n_features,
            10,
            0.95,
            Some(&config),
        )
        .unwrap();

        assert_eq!(result.shap_values_mean.len(), n_explain);
        assert_eq!(result.shap_values_std.len(), n_explain);
        assert_eq!(result.shap_values_lower.len(), n_explain);
        assert_eq!(result.shap_values_upper.len(), n_explain);
        assert_eq!(result.n_bootstrap, 10);
        assert!((result.confidence_level - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_shap_interactions() {
        let shap_values = vec![
            vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.3, 0.4]],
            vec![vec![0.15, 0.25, 0.35], vec![0.25, 0.35, 0.45]],
        ];
        let time_points = vec![1.0, 2.0, 3.0];

        let result =
            compute_shap_interactions(shap_values, time_points, 2, Some(AggregationMethod::Mean))
                .unwrap();

        assert_eq!(result.interaction_values.len(), 2);
        assert_eq!(result.interaction_values[0].len(), 2);
        assert_eq!(result.interaction_values[0][0].len(), 3);
        assert!(result.aggregated_interactions.is_some());

        let top = result.top_interactions(1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_concordance_index() {
        let predictions = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let times = vec![10.0, 5.0, 15.0];
        let events = vec![1, 1, 0];

        let c_index = compute_concordance_index(&predictions, &times, &events, 3, 3);
        assert!((0.0..=1.0).contains(&c_index));
    }
}
