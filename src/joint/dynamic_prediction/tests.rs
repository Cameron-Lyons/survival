#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_prediction_basic() {
        let result = dynamic_prediction(
            vec![0.5, 0.3],
            vec![0.2],
            0.1,
            vec![0.0, 0.0],
            vec![0.01, 0.02, 0.03],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 0.5, 0.3],
            vec![1.0, 0.5, 1.0, 0.3, 1.0, 0.7],
            3,
            2,
            vec![0.5],
            1,
            2.0,
            vec![3.0, 4.0, 5.0],
            100,
        )
        .unwrap();

        assert!(!result.survival_mean.is_empty());
    }

    #[test]
    fn test_time_varying_auc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![1.5, 2.5, 3.5];

        let result = time_varying_auc(
            risk_scores,
            event_time,
            event_status,
            eval_times,
            1.0,
            "cumulative/dynamic",
        )
        .unwrap();

        assert_eq!(result.times.len(), 3);
        assert_eq!(result.auc_values.len(), 3);
    }

    #[test]
    fn test_dynamic_c_index() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];

        let result =
            dynamic_c_index(risk_scores, event_time, event_status, 0.0, 6.0, None).unwrap();

        assert!(result.c_index >= 0.0 && result.c_index <= 1.0);
        assert!(result.n_pairs > 0);
    }

    #[test]
    fn test_ipcw_auc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![2.0, 3.0, 4.0];

        let result = ipcw_auc(risk_scores, event_time, event_status, eval_times).unwrap();

        assert_eq!(result.times.len(), 3);
        assert!(result.integrated_auc >= 0.0 && result.integrated_auc <= 1.0);
    }

    #[test]
    fn test_super_landmark() {
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event_status = vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9, 0.5, 0.4, 0.6, 0.3, 0.8, 0.2, 0.7, 0.5,
            0.3, 0.6, 0.4,
        ];
        let landmark_times = vec![0.0, 2.0, 4.0];

        let result = super_landmark_model(
            event_time,
            event_status,
            covariates,
            2,
            landmark_times,
            12.0,
            30,
        )
        .unwrap();

        assert_eq!(result.landmark_times.len(), 3);
        assert_eq!(result.coefficients.len(), 3);
    }

    #[test]
    fn test_time_dependent_roc() {
        let risk_scores = vec![0.8, 0.6, 0.4, 0.2, 0.9, 0.3];
        let event_time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 5.0];
        let event_status = vec![1, 1, 0, 1, 1, 0];
        let eval_times = vec![2.0, 3.5];

        let result =
            time_dependent_roc(risk_scores, event_time, event_status, eval_times, 20).unwrap();

        assert_eq!(result.times.len(), 2);
        assert_eq!(result.sensitivity.len(), 2);
        assert_eq!(result.thresholds.len(), 20);
    }
}
