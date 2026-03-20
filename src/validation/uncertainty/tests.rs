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
    }

    #[test]
    fn test_compute_quantile() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(compute_quantile(&mut values, 0.5), 3.0);
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
