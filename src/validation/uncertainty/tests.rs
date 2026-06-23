#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_dropout_config() {
        let config = MCDropoutConfig::new(100, 0.1, None).unwrap();
        assert_eq!(config.n_samples, 100);
    }

    #[test]
    fn test_mc_dropout_config_validation() {
        let result = MCDropoutConfig::new(0, 0.1, None);
        assert!(result.is_err());

        assert!(MCDropoutConfig::new(100, 1.0, None).is_err());
        assert!(MCDropoutConfig::new(100, f64::NAN, None).is_err());
    }

    #[test]
    fn test_mc_dropout_uncertainty_validates_prediction_cube() {
        let predictions = vec![
            vec![vec![0.9, 0.8], vec![0.7, 0.6]],
            vec![vec![0.85, 0.75], vec![0.65, 0.55]],
        ];
        let result = mc_dropout_uncertainty(predictions).unwrap();

        assert_eq!(result.mean_prediction.len(), 2);
        assert_eq!(result.mean_prediction[0].len(), 2);
        assert!(result
            .total_uncertainty()
            .into_iter()
            .all(|value| value.is_finite()));

        assert!(mc_dropout_uncertainty(vec![]).is_err());
        assert!(mc_dropout_uncertainty(vec![vec![]]).is_err());
        assert!(mc_dropout_uncertainty(vec![vec![vec![]]]).is_err());
        assert!(mc_dropout_uncertainty(vec![vec![vec![0.5]], vec![]]).is_err());
        assert!(mc_dropout_uncertainty(vec![vec![vec![f64::NAN]]]).is_err());
        assert!(mc_dropout_uncertainty(vec![vec![vec![1.2]]]).is_err());
    }

    #[test]
    fn test_compute_quantile() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(compute_quantile(&mut values, 0.5), 3.0);
    }

    #[test]
    fn test_ensemble_uncertainty_validates_prediction_cube() {
        let predictions = vec![
            vec![vec![0.9, 0.8], vec![0.7, 0.6]],
            vec![vec![0.85, 0.75], vec![0.65, 0.55]],
        ];
        let result = ensemble_uncertainty(predictions, 0.95).unwrap();

        assert_eq!(result.mean_prediction.len(), 2);
        assert_eq!(result.mean_prediction[0].len(), 2);
        assert!(result.model_disagreement.iter().all(|value| value.is_finite()));

        assert!(ensemble_uncertainty(vec![], 0.95).is_err());
        assert!(ensemble_uncertainty(vec![vec![vec![0.5]], vec![]], 0.95).is_err());
        assert!(ensemble_uncertainty(vec![vec![vec![f64::NAN]], vec![vec![0.5]]], 0.95).is_err());
        assert!(ensemble_uncertainty(vec![vec![vec![0.5]], vec![vec![0.6]]], 1.0).is_err());
    }

    #[test]
    fn test_quantile_regression_intervals_validate_inputs() {
        let predictions = vec![
            vec![vec![0.9, 0.8], vec![0.7, 0.6]],
            vec![vec![0.85, 0.75], vec![0.65, 0.55]],
            vec![vec![0.95, 0.82], vec![0.75, 0.62]],
        ];
        let result =
            quantile_regression_intervals(predictions.clone(), Some(vec![0.1, 0.5, 0.9])).unwrap();

        assert_eq!(result.median.len(), 2);
        assert_eq!(result.quantiles, vec![0.1, 0.5, 0.9]);

        assert!(quantile_regression_intervals(predictions.clone(), Some(vec![0.1, 0.9])).is_err());
        assert!(
            quantile_regression_intervals(predictions.clone(), Some(vec![0.5, 0.1, 0.9]))
                .is_err()
        );
        assert!(quantile_regression_intervals(vec![vec![vec![]]], None).is_err());
        assert!(quantile_regression_intervals(vec![vec![vec![f64::INFINITY]]], None).is_err());
    }

    #[test]
    fn test_calibrate_prediction_intervals_validates_inputs() {
        let result = calibrate_prediction_intervals(
            vec![1.0, 2.0, 3.0],
            vec![1, 0, 1],
            vec![0.5, 1.5, 2.5],
            vec![1.5, 2.5, 3.5],
            0.9,
        )
        .unwrap();

        assert_eq!(result.observed_coverage, 1.0);

        assert!(
            calibrate_prediction_intervals(vec![1.0], vec![2], vec![0.0], vec![2.0], 0.9)
                .is_err()
        );
        assert!(
            calibrate_prediction_intervals(
                vec![1.0],
                vec![1],
                vec![f64::NAN],
                vec![2.0],
                0.9
            )
            .is_err()
        );
        assert!(
            calibrate_prediction_intervals(vec![1.0], vec![1], vec![2.0], vec![1.0], 0.9)
                .is_err()
        );
        assert!(
            calibrate_prediction_intervals(vec![1.0], vec![1], vec![0.0], vec![2.0], 1.0)
                .is_err()
        );
    }

    #[test]
    fn test_apply_dropout() {
        let mut rng = fastrand::Rng::new();
        rng.seed(42);
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let dropped = _apply_dropout(&values, 0.5, &mut rng);
        assert_eq!(dropped.len(), 5);
    }

    #[test]
    fn test_conformal_survival_basic() {
        let cal_time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let cal_event = vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 0];
        let cal_predictions = vec![1.5, 2.2, 2.8, 4.5, 4.8, 5.5, 7.5, 7.8, 8.5, 10.5];
        let test_predictions = vec![3.0, 6.0, 9.0];

        let config = ConformalSurvivalConfig::new(0.1, "cqr".to_string(), 100, Some(42));
        let result = conformal_survival(
            cal_time,
            cal_event,
            cal_predictions,
            test_predictions,
            config,
        )
        .unwrap();

        assert_eq!(result.lower_bounds.len(), 3);
        assert_eq!(result.upper_bounds.len(), 3);

        for i in 0..3 {
            assert!(result.lower_bounds[i] <= result.point_predictions[i]);
            assert!(result.upper_bounds[i] >= result.point_predictions[i]);
        }
    }

    #[test]
    fn test_conformal_survival_validates_public_inputs() {
        let config = ConformalSurvivalConfig::new(0.1, "cqr".to_string(), 100, Some(42));

        let err = conformal_survival(vec![1.0], vec![1, 0], vec![1.0], vec![1.0], config.clone())
            .expect_err("cal_event length mismatch should fail");
        assert!(err.to_string().contains("cal_event length mismatch"));

        let err = conformal_survival(
            vec![1.0],
            vec![1],
            vec![f64::NAN],
            vec![1.0],
            config.clone(),
        )
        .expect_err("non-finite calibration predictions should fail");
        assert!(err
            .to_string()
            .contains("cal_predictions contains non-finite"));

        let err = conformal_survival(
            vec![1.0],
            vec![2],
            vec![1.0],
            vec![1.0],
            config.clone(),
        )
        .expect_err("non-binary event should fail");
        assert!(err.to_string().contains("cal_event must contain only 0/1"));

        let err = conformal_survival(
            vec![1.0],
            vec![1],
            vec![1.0],
            vec![f64::INFINITY],
            config.clone(),
        )
        .expect_err("non-finite test predictions should fail");
        assert!(err
            .to_string()
            .contains("test_predictions contains non-finite"));

        let err = conformal_survival(
            vec![1.0],
            vec![1],
            vec![1.0],
            vec![1.0],
            ConformalSurvivalConfig::new(0.0, "cqr".to_string(), 100, None),
        )
        .expect_err("invalid alpha should fail");
        assert!(err.to_string().contains("alpha"));

        let err = conformal_survival(
            vec![1.0],
            vec![1],
            vec![1.0],
            vec![1.0],
            ConformalSurvivalConfig::new(0.1, "unknown".to_string(), 100, None),
        )
        .expect_err("unknown method should fail");
        assert!(err.to_string().contains("method must be"));

        let err = conformal_survival(
            vec![0.0],
            vec![0],
            vec![0.0],
            vec![1.0],
            ConformalSurvivalConfig::new(0.1, "censoring_adjusted".to_string(), 100, None),
        )
        .expect_err("zero calibration times should fail for censoring adjusted method");
        assert!(err.to_string().contains("positive value"));
    }

    #[test]
    fn test_bayesian_bootstrap_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 0, 1, 0];
        let eval_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let config = BayesianBootstrapConfig::new(100, 0.95, Some(42));
        let result = bayesian_bootstrap_survival(time, event, eval_times, config).unwrap();

        assert_eq!(result.mean_survival.len(), 5);
        assert_eq!(result.lower_ci.len(), 5);
        assert_eq!(result.upper_ci.len(), 5);

        for i in 0..5 {
            assert!(result.lower_ci[i] <= result.mean_survival[i]);
            assert!(result.upper_ci[i] >= result.mean_survival[i]);
            assert!(result.mean_survival[i] >= 0.0 && result.mean_survival[i] <= 1.0);
        }
    }

    #[test]
    fn test_weighted_kaplan_meier_groups_tied_event_times() {
        let time = vec![1.0, 1.0, 2.0];
        let event = vec![1, 1, 0];
        let weights = vec![0.2, 0.3, 0.5];
        let eval_times = vec![
            0.5,
            1.0,
            1.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
        ];

        let survival = weighted_kaplan_meier(&time, &event, &weights, &eval_times);

        assert_eq!(survival[0], 1.0);
        for value in &survival[1..] {
            assert!((*value - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_bayesian_bootstrap_survival_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![
            1.0,
            1.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
        ];
        let event = vec![1, 1, 0, 0];
        let eval_times = vec![1.0, 2.0, 3.0];
        let config = BayesianBootstrapConfig::new(25, 0.8, Some(7));

        let exact = bayesian_bootstrap_survival(
            exact_time,
            event.clone(),
            eval_times.clone(),
            config.clone(),
        )
        .unwrap();
        let near = bayesian_bootstrap_survival(near_time, event, eval_times, config).unwrap();

        assert_eq!(near.mean_survival, exact.mean_survival);
        assert_eq!(near.lower_ci, exact.lower_ci);
        assert_eq!(near.upper_ci, exact.upper_ci);
        assert_eq!(near.posterior_samples, exact.posterior_samples);
    }

    #[test]
    fn test_bayesian_bootstrap_survival_validates_inputs() {
        pyo3::Python::initialize();
        let config = BayesianBootstrapConfig::new(10, 0.95, Some(1));

        let err =
            bayesian_bootstrap_survival(vec![1.0], vec![1, 0], vec![1.0], config.clone())
                .expect_err("event length mismatch should fail");
        assert!(err.to_string().contains("event length mismatch"));

        let err = bayesian_bootstrap_survival(
            vec![1.0],
            vec![2],
            vec![1.0],
            config.clone(),
        )
        .expect_err("non-binary event should fail");
        assert!(err.to_string().contains("event must contain only 0/1"));

        let err = bayesian_bootstrap_survival(
            vec![f64::NAN],
            vec![1],
            vec![1.0],
            config.clone(),
        )
        .expect_err("non-finite time should fail");
        assert!(err.to_string().contains("time contains non-finite"));

        let err = bayesian_bootstrap_survival(
            vec![1.0],
            vec![1],
            vec![1.0],
            BayesianBootstrapConfig::new(0, 0.95, Some(1)),
        )
        .expect_err("zero bootstrap count should fail");
        assert!(err.to_string().contains("n_bootstrap must be positive"));
    }

    #[test]
    fn test_jackknife_plus_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 1, 1, 0, 1, 1, 0, 1, 1, 0];
        let covariates = vec![
            vec![0.5, 1.0],
            vec![1.0, 1.2],
            vec![1.5, 1.4],
            vec![2.0, 1.6],
            vec![2.5, 1.8],
            vec![3.0, 2.0],
            vec![3.5, 2.2],
            vec![4.0, 2.4],
            vec![4.5, 2.6],
            vec![5.0, 2.8],
        ];

        let config = JackknifePlusConfig::new(0.1, true, 5);
        let result = jackknife_plus_survival(time, event, covariates, config).unwrap();

        assert_eq!(result.lower_bounds.len(), 10);
        assert_eq!(result.upper_bounds.len(), 10);
        assert_eq!(result.residuals.len(), 10);

        for i in 0..10 {
            assert!(result.lower_bounds[i] <= result.upper_bounds[i]);
        }
    }

    #[test]
    fn test_jackknife_plus_survival_validates_public_inputs() {
        let config = JackknifePlusConfig::new(0.1, true, 5);
        let covariates = vec![vec![0.0], vec![1.0]];

        let err = jackknife_plus_survival(vec![1.0], vec![1], vec![vec![0.0]], config.clone())
            .expect_err("too few observations should fail");
        assert!(err
            .to_string()
            .contains("Need at least 2 observations for jackknife"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1],
            covariates.clone(),
            config.clone(),
        )
        .expect_err("event length mismatch should fail");
        assert!(err.to_string().contains("event length mismatch"));

        let err = jackknife_plus_survival(
            vec![1.0, f64::INFINITY],
            vec![1, 0],
            covariates.clone(),
            config.clone(),
        )
        .expect_err("non-finite time should fail");
        assert!(err.to_string().contains("time contains non-finite"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 2],
            covariates.clone(),
            config.clone(),
        )
        .expect_err("non-binary event should fail");
        assert!(err.to_string().contains("event must contain only 0/1"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 0],
            vec![vec![0.0]],
            config.clone(),
        )
        .expect_err("covariate row mismatch should fail");
        assert!(err.to_string().contains("one row per observation"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 0],
            vec![vec![0.0], vec![1.0, 2.0]],
            config.clone(),
        )
        .expect_err("ragged covariates should fail");
        assert!(err.to_string().contains("covariates must be rectangular"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 0],
            vec![vec![0.0], vec![f64::NAN]],
            config.clone(),
        )
        .expect_err("non-finite covariates should fail");
        assert!(err
            .to_string()
            .contains("covariates contains non-finite"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 0],
            covariates.clone(),
            JackknifePlusConfig::new(1.0, true, 5),
        )
        .expect_err("invalid alpha should fail");
        assert!(err.to_string().contains("alpha"));

        let err = jackknife_plus_survival(
            vec![1.0, 2.0],
            vec![1, 0],
            covariates,
            JackknifePlusConfig::new(0.1, true, 0),
        )
        .expect_err("zero cv folds should fail");
        assert!(err.to_string().contains("cv_folds must be positive"));

        let zero_column = jackknife_plus_survival(
            vec![1.0, 2.0, 3.0],
            vec![1, 0, 1],
            vec![vec![], vec![], vec![]],
            config,
        )
        .expect("zero-column covariates should use the baseline fallback");
        assert_eq!(zero_column.point_predictions.len(), 3);
    }

    #[test]
    fn test_conformity_score_methods() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 0, 1, 0];
        let predictions = vec![1.5, 2.2, 2.8, 4.5, 4.8];

        let cqr_scores = compute_conformity_scores(&time, &event, &predictions, "cqr");
        let weighted_scores = compute_conformity_scores(&time, &event, &predictions, "weighted");
        let censoring_scores =
            compute_conformity_scores(&time, &event, &predictions, "censoring_adjusted");

        assert_eq!(cqr_scores.len(), 5);
        assert_eq!(weighted_scores.len(), 5);
        assert_eq!(censoring_scores.len(), 5);

        for i in 0..5 {
            if event[i] == 0 {
                assert!(
                    weighted_scores[i] < cqr_scores[i] || weighted_scores[i] == cqr_scores[i] * 0.5
                );
            }
        }
    }
}
