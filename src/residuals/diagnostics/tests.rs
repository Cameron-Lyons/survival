#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfbeta() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = dfbeta_cox(time, event, covariates, 1, coefficients, None).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.n_vars, 1);
        assert_eq!(result.dfbeta.len(), 8);
    }

    #[test]
    fn test_leverage() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = leverage_cox(time, event, covariates, 1, coefficients, 2.0).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.leverage.len(), 8);
        assert!(result.mean_leverage >= 0.0);
    }

    #[test]
    fn test_schoenfeld_smooth() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let schoenfeld = vec![0.1, -0.1, 0.2, -0.2, 0.1];
        let coefficients = vec![0.5];

        let result =
            smooth_schoenfeld(event_times, schoenfeld, 1, coefficients, None, "identity").unwrap();

        assert_eq!(result.n_events, 5);
        assert_eq!(result.n_vars, 1);
        assert_eq!(result.smoothed_residuals.len(), 5);
    }

    #[test]
    fn test_outlier_detection() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = outlier_detection_cox(time, event, covariates, 1, coefficients, 3.0).unwrap();

        assert_eq!(result.martingale_residuals.len(), 8);
        assert_eq!(result.deviance_residuals.len(), 8);
    }

    #[test]
    fn test_model_influence() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = model_influence_cox(time, event, covariates, 1, coefficients).unwrap();

        assert_eq!(result.n_obs, 8);
        assert_eq!(result.cooks_distance.len(), 8);
        assert_eq!(result.covratio.len(), 8);
    }

    #[test]
    fn test_goodness_of_fit() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let coefficients = vec![0.5];

        let result = goodness_of_fit_cox(time, event, covariates, 1, coefficients).unwrap();

        assert_eq!(result.n_obs, 8);
        assert!(result.global_p_value >= 0.0 && result.global_p_value <= 1.0);
    }
}
