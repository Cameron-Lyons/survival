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
}
