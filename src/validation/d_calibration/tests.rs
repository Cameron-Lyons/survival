#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_calibration_uniform() {
        let survival_probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let status = vec![1; 100];

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert!(result.p_value > 0.05);
        assert!(result.is_calibrated);
        assert_eq!(result.n_events, 100);
        assert_eq!(result.n_bins, 10);
    }

    #[test]
    fn test_d_calibration_non_uniform() {
        let mut survival_probs = vec![0.1; 50];
        survival_probs.extend(vec![0.9; 50]);
        let status = vec![1; 100];

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert!(result.p_value < 0.05);
        assert!(!result.is_calibrated);
    }

    #[test]
    fn test_d_calibration_with_censoring() {
        let survival_probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let mut status = vec![1; 100];
        for i in (0..100).step_by(2) {
            status[i] = 0;
        }

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert_eq!(result.n_events, 50);
    }

    #[test]
    fn test_d_calibration_empty() {
        let result = d_calibration_core(&[], &[], 10);
        assert_eq!(result.n_events, 0);
        assert!(result.is_calibrated);
    }

    #[test]
    fn test_one_calibration_basic() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = time.iter().map(|&t| (-0.01 * t).exp()).collect();

        let result = one_calibration_core(&time, &status, &predicted, 50.0, 5);

        assert_eq!(result.n_groups, 5);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_calibration_plot_basic() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = time.iter().map(|&t| (-0.01 * t).exp()).collect();

        let result = calibration_plot_data_core(&time, &status, &predicted, 50.0, 5);

        assert_eq!(result.predicted.len(), 5);
        assert_eq!(result.observed.len(), 5);
        assert!(result.ici >= 0.0);
        assert!(result.emax >= result.e90);
        assert!(result.e90 >= result.e50);
    }

    #[test]
    fn test_calibration_metrics() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = (0..100).map(|i| 1.0 - i as f64 / 100.0).collect();

        let result = calibration_plot_data_core(&time, &status, &predicted, 50.0, 10);

        assert!(result.ici >= 0.0 && result.ici <= 1.0);
    }

    #[test]
    fn test_timepoint_calibration_groups_near_tied_time_points() {
        let exact_time = vec![1.0, 2.0, 2.0, 2.0, 3.0, 1.5, 2.0, 2.5, 3.5, 4.0];
        let near_time = vec![
            1.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
            1.5,
            2.0,
            2.5,
            3.5,
            4.0,
        ];
        let status = vec![1, 1, 0, 0, 1, 0, 1, 0, 1, 0];
        let predicted = vec![0.92, 0.84, 0.78, 0.7, 0.62, 0.56, 0.48, 0.4, 0.32, 0.24];
        let time_point = 2.0;
        let n_groups = 2;

        let exact_one = one_calibration_core(&exact_time, &status, &predicted, time_point, n_groups);
        let near_one = one_calibration_core(&near_time, &status, &predicted, time_point, n_groups);
        assert_eq!(near_one.n_events_per_group, exact_one.n_events_per_group);
        assert_eq!(near_one.n_per_group, exact_one.n_per_group);
        assert_eq!(near_one.observed_survival, exact_one.observed_survival);
        assert!((near_one.statistic - exact_one.statistic).abs() < 1e-12);

        let exact_plot =
            calibration_plot_data_core(&exact_time, &status, &predicted, time_point, n_groups);
        let near_plot =
            calibration_plot_data_core(&near_time, &status, &predicted, time_point, n_groups);
        assert_eq!(near_plot.n_per_group, exact_plot.n_per_group);
        assert_eq!(near_plot.observed, exact_plot.observed);
        assert!((near_plot.ici - exact_plot.ici).abs() < 1e-12);
        assert!((near_plot.e50 - exact_plot.e50).abs() < 1e-12);
        assert!((near_plot.e90 - exact_plot.e90).abs() < 1e-12);
        assert!((near_plot.emax - exact_plot.emax).abs() < 1e-12);

        let exact_brier =
            brier_calibration_core(&exact_time, &status, &predicted, time_point, n_groups);
        let near_brier =
            brier_calibration_core(&near_time, &status, &predicted, time_point, n_groups);
        assert!((near_brier.brier_score - exact_brier.brier_score).abs() < 1e-12);
        assert_eq!(near_brier.observed, exact_brier.observed);

        let exact_smooth =
            smoothed_calibration_core(&exact_time, &status, &predicted, time_point, 10, Some(0.2));
        let near_smooth =
            smoothed_calibration_core(&near_time, &status, &predicted, time_point, 10, Some(0.2));
        assert_eq!(near_smooth.predicted_grid, exact_smooth.predicted_grid);
        for (actual, expected) in near_smooth
            .smoothed_observed
            .iter()
            .zip(exact_smooth.smoothed_observed.iter())
        {
            assert!((actual - expected).abs() < 1e-12);
        }

        let survival_predictions = predicted
            .iter()
            .map(|&value| vec![value, (value - 0.1_f64).max(0.0)])
            .collect::<Vec<_>>();
        let prediction_times = vec![2.0, 3.0];
        let exact_multi = multi_time_calibration_core(
            &exact_time,
            &status,
            &survival_predictions,
            &prediction_times,
            n_groups,
        );
        let near_multi = multi_time_calibration_core(
            &near_time,
            &status,
            &survival_predictions,
            &prediction_times,
            n_groups,
        );
        assert_eq!(near_multi.time_points, exact_multi.time_points);
        assert_eq!(near_multi.brier_scores, exact_multi.brier_scores);
        assert_eq!(near_multi.ici_values, exact_multi.ici_values);
        assert!((near_multi.integrated_brier - exact_multi.integrated_brier).abs() < 1e-12);
    }

    #[test]
    fn public_d_calibration_validates_probabilities_and_status() {
        let err = d_calibration(vec![1.2], vec![1], None).unwrap_err();
        assert!(err.to_string().contains("probabilities between 0 and 1"));

        let err = d_calibration(vec![0.8], vec![2], None).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));
    }

    #[test]
    fn public_timepoint_calibration_validates_shared_inputs() {
        let err = one_calibration(vec![f64::NAN], vec![1], vec![0.8], 1.0, Some(2)).unwrap_err();
        assert!(err.to_string().contains("time contains non-finite"));

        let err = calibration_plot(vec![1.0], vec![2], vec![0.8], 1.0, Some(2)).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = brier_calibration(vec![1.0], vec![1], vec![1.2], 1.0, Some(2)).unwrap_err();
        assert!(err.to_string().contains("probabilities between 0 and 1"));

        let err = smoothed_calibration(
            vec![1.0],
            vec![1],
            vec![0.8],
            f64::INFINITY,
            Some(10),
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("time_point contains non-finite"));
    }

    #[test]
    fn public_multi_time_calibration_validates_times_and_rows() {
        let err =
            multi_time_calibration(vec![1.0], vec![1], vec![vec![0.8, 0.7]], vec![2.0, 1.0], Some(2))
                .unwrap_err();
        assert!(err.to_string().contains("prediction_times must be sorted"));

        let err =
            multi_time_calibration(vec![1.0], vec![1], vec![vec![1.2]], vec![1.0], Some(2))
                .unwrap_err();
        assert!(err.to_string().contains("probabilities between 0 and 1"));
    }

    #[test]
    fn public_smoothed_calibration_validates_bandwidth() {
        let err =
            smoothed_calibration(vec![1.0], vec![1], vec![0.8], 1.0, Some(10), Some(f64::NAN))
                .unwrap_err();
        assert!(err.to_string().contains("bandwidth contains non-finite"));

        let err =
            smoothed_calibration(vec![1.0], vec![1], vec![0.8], 1.0, Some(10), Some(0.0))
                .unwrap_err();
        assert!(err.to_string().contains("bandwidth must be positive"));
    }
}
