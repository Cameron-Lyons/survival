#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_lasso() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2, 0.3, 0.6, 0.5, 0.4, 0.3, 0.8, 0.6, 0.2, 0.4,
            0.5, 0.7, 0.3, 0.2, 0.4, 0.8, 0.5,
        ];
        let groups = vec![0, 0, 1];

        let config = GroupLassoConfig::new(0.1, 100, 1e-4, true, None).unwrap();
        let result = group_lasso_cox(time, event, covariates, groups, &config).unwrap();

        assert_eq!(result.coefficients.len(), 3);
        assert!(result.n_groups == 2);
    }

    #[test]
    fn test_sparse_boosting() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.4, 0.8, 0.2, 0.6, 0.5, 0.3, 0.8, 0.4, 0.5, 0.7, 0.3, 0.4, 0.8,
        ];

        let config = SparseBoostingConfig::new(50, 0.1, 0.8, 10, 0.01, Some(42)).unwrap();
        let result = sparse_boosting_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert!(!result.iteration_scores.is_empty());
    }

    #[test]
    fn test_sis() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.2, 0.7, 0.4, 0.1, 0.8, 0.2, 0.3, 0.6, 0.5, 0.4, 0.3, 0.8, 0.6, 0.2, 0.4,
            0.5, 0.7, 0.3, 0.2, 0.4, 0.8, 0.5,
        ];

        let config = SISConfig::new(Some(2), false, 5, 0.0).unwrap();
        let result = sis_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.n_selected, 2);
        assert_eq!(result.ranking.len(), 3);
    }

    #[test]
    fn test_stability_selection() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            0.5, 0.3, 0.7, 0.4, 0.8, 0.2, 0.6, 0.5, 0.3, 0.8, 0.4, 0.5, 0.7, 0.3, 0.4, 0.8,
        ];

        let config =
            StabilitySelectionConfig::new(20, 0.5, Some(vec![0.1, 0.5]), 0.3, Some(42)).unwrap();
        let result = stability_selection_cox(time, event, covariates, &config).unwrap();

        assert_eq!(result.selection_probabilities.len(), 2);
    }
}
