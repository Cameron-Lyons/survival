#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_process() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = DirichletProcessConfig::new(1.0, 5, 100, 50, Some(42));
        let result = dirichlet_process_survival(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.cluster_assignments.len(), 8);
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_bayesian_model_averaging() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![
            1.0, 0.5, 0.0, 0.8, 1.0, 0.3, 0.0, 0.9, 1.0, 0.2, 0.0, 0.7, 1.0, 0.1, 0.0, 0.6,
        ];

        let config = BayesianModelAveragingConfig::new(200, 100, 0.5, Some(42));
        let result = bayesian_model_averaging_cox(time, event, covariates, 2, &config).unwrap();

        assert_eq!(result.n_vars, 2);
        assert_eq!(result.posterior_inclusion_prob.len(), 2);
    }

    #[test]
    fn test_spike_slab() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = SpikeSlabConfig::new(0.001, 10.0, 0.5, 200, 100, Some(42));
        let result = spike_slab_cox(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.posterior_inclusion_prob.len(), 1);
        assert_eq!(result.posterior_mean.len(), 1);
    }

    #[test]
    fn test_horseshoe() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let config = HorseshoeConfig::new(1.0, 200, 100, Some(42));
        let result = horseshoe_cox(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.posterior_mean.len(), 1);
        assert_eq!(result.shrinkage_factors.len(), 1);
        assert!(result.effective_df >= 0.0);
    }

    #[test]
    fn test_dirichlet_process_output_dimensions() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0];

        let config = DirichletProcessConfig::new(1.0, 5, 200, 100, Some(42));
        let result = dirichlet_process_survival(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.cluster_assignments.len(), 10);
        assert_eq!(result.eval_times.len(), 50);
        assert_eq!(result.posterior_mean_survival.len(), 50);
        assert_eq!(result.posterior_lower.len(), 50);
        assert_eq!(result.posterior_upper.len(), 50);
        assert!(result.n_clusters >= 1);
        assert!(result.concentration_posterior > 0.0);

        for t in 0..50 {
            assert!(result.posterior_mean_survival[t] >= 0.0);
            assert!(result.posterior_mean_survival[t] <= 1.0);
            assert!(result.posterior_lower[t] <= result.posterior_upper[t]);
        }
    }

    #[test]
    fn test_dirichlet_process_high_concentration() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0];

        let config_high = DirichletProcessConfig::new(10.0, 10, 200, 100, Some(42));
        let result_high = dirichlet_process_survival(
            time.clone(),
            event.clone(),
            covariates.clone(),
            1,
            &config_high,
        )
        .unwrap();

        let config_low = DirichletProcessConfig::new(0.1, 10, 200, 100, Some(42));
        let result_low =
            dirichlet_process_survival(time, event, covariates, 1, &config_low).unwrap();

        assert!(result_high.n_clusters >= 1);
        assert!(result_low.n_clusters >= 1);
    }

    #[test]
    fn test_dirichlet_process_survival_monotone() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let covariates = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0];

        let config = DirichletProcessConfig::new(1.0, 5, 300, 150, Some(42));
        let result = dirichlet_process_survival(time, event, covariates, 1, &config).unwrap();

        for s in &result.posterior_mean_survival {
            assert!(*s >= 0.0 && *s <= 1.0);
        }

        for surv in &result.cluster_survival {
            for t in 1..surv.len() {
                assert!(surv[t] <= surv[t - 1] + 1e-10);
            }
        }
    }

    #[test]
    fn test_bayesian_model_averaging_output_validation() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![
            1.0, 0.2, 0.5, 0.8, 0.0, 0.5, 1.0, 0.3, 0.5, 0.9, 0.0, 0.7, 1.0, 0.1, 0.5, 0.6, 0.0,
            0.4, 0.0, 0.8, 1.0, 0.3, 0.0, 0.5, 0.5, 0.6, 1.0, 0.2, 0.0, 0.9,
        ];
        let n_covariates = 3;

        let config = BayesianModelAveragingConfig::new(400, 200, 0.5, Some(42));
        let result =
            bayesian_model_averaging_cox(time, event, covariates, n_covariates, &config).unwrap();

        assert_eq!(result.n_vars, n_covariates);
        assert_eq!(result.posterior_inclusion_prob.len(), n_covariates);
        assert_eq!(result.posterior_mean_coef.len(), n_covariates);
        assert_eq!(result.posterior_sd_coef.len(), n_covariates);
        assert_eq!(result.bayes_factor_vs_null.len(), n_covariates);

        for j in 0..n_covariates {
            assert!(result.posterior_inclusion_prob[j] >= 0.0);
            assert!(result.posterior_inclusion_prob[j] <= 1.0);
            assert!(result.posterior_sd_coef[j] >= 0.0);
            assert!(result.bayes_factor_vs_null[j] >= 0.0);
        }

        assert!(result.n_models_visited >= 1);
        assert!(!result.model_posterior_probs.is_empty());
    }

    #[test]
    fn test_bayesian_model_averaging_input_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];
        let covariates = vec![1.0, 2.0];

        let config = BayesianModelAveragingConfig::new(100, 50, 0.5, Some(42));
        let result = bayesian_model_averaging_cox(time, event, covariates, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_bayesian_model_averaging_prior_inclusion_extremes() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![
            1.0, 0.2, 0.5, 0.8, 0.0, 0.5, 1.0, 0.3, 0.5, 0.9, 0.0, 0.7, 1.0, 0.1, 0.5, 0.6, 0.0,
            0.4, 0.0, 0.8, 1.0, 0.3, 0.0, 0.5, 0.5, 0.6, 1.0, 0.2, 0.0, 0.9,
        ];

        let config_low = BayesianModelAveragingConfig::new(300, 150, 0.1, Some(42));
        let result_low = bayesian_model_averaging_cox(
            time.clone(),
            event.clone(),
            covariates.clone(),
            3,
            &config_low,
        )
        .unwrap();

        let config_high = BayesianModelAveragingConfig::new(300, 150, 0.9, Some(42));
        let result_high =
            bayesian_model_averaging_cox(time, event, covariates, 3, &config_high).unwrap();

        for j in 0..3 {
            assert!(result_low.posterior_inclusion_prob[j] >= 0.0);
            assert!(result_high.posterior_inclusion_prob[j] >= 0.0);
        }
    }

    #[test]
    fn test_spike_slab_output_dimensions() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![
            1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0,
            1.0, 0.0, 0.0,
        ];
        let n_covariates = 2;

        let config = SpikeSlabConfig::new(0.001, 10.0, 0.5, 400, 200, Some(42));
        let result = spike_slab_cox(time, event, covariates, n_covariates, &config).unwrap();

        assert_eq!(result.posterior_inclusion_prob.len(), n_covariates);
        assert_eq!(result.posterior_mean.len(), n_covariates);
        assert_eq!(result.posterior_sd.len(), n_covariates);
        assert_eq!(result.credible_lower.len(), n_covariates);
        assert_eq!(result.credible_upper.len(), n_covariates);

        for j in 0..n_covariates {
            assert!(result.posterior_inclusion_prob[j] >= 0.0);
            assert!(result.posterior_inclusion_prob[j] <= 1.0);
            assert!(result.posterior_sd[j] >= 0.0);
            assert!(result.credible_lower[j] <= result.credible_upper[j]);
        }

        assert!(result.n_selected <= n_covariates);
        for &idx in &result.selected_variables {
            assert!(idx < n_covariates);
            assert!(result.posterior_inclusion_prob[idx] > 0.5);
        }

        assert!(result.log_marginal_likelihood.is_finite());
    }

    #[test]
    fn test_spike_slab_input_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0];
        let covariates = vec![1.0, 2.0, 3.0];

        let config = SpikeSlabConfig::new(0.001, 10.0, 0.5, 100, 50, Some(42));
        let result = spike_slab_cox(time, event, covariates, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_spike_slab_narrow_slab() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0];

        let config_narrow = SpikeSlabConfig::new(0.001, 1.0, 0.5, 300, 150, Some(42));
        let result_narrow = spike_slab_cox(
            time.clone(),
            event.clone(),
            covariates.clone(),
            1,
            &config_narrow,
        )
        .unwrap();

        let config_wide = SpikeSlabConfig::new(0.001, 100.0, 0.5, 300, 150, Some(42));
        let result_wide = spike_slab_cox(time, event, covariates, 1, &config_wide).unwrap();

        assert!(result_narrow.posterior_sd[0] >= 0.0);
        assert!(result_wide.posterior_sd[0] >= 0.0);
    }

    #[test]
    fn test_horseshoe_output_dimensions_multi_covariate() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![
            1.0, 0.2, 0.5, 0.8, 0.0, 0.5, 1.0, 0.3, 0.5, 0.9, 0.0, 0.7, 1.0, 0.1, 0.5, 0.6, 0.0,
            0.4, 0.0, 0.8, 1.0, 0.3, 0.0, 0.5, 0.5, 0.6, 1.0, 0.2, 0.0, 0.9,
        ];
        let n_covariates = 3;

        let config = HorseshoeConfig::new(1.0, 400, 200, Some(42));
        let result = horseshoe_cox(time, event, covariates, n_covariates, &config).unwrap();

        assert_eq!(result.posterior_mean.len(), n_covariates);
        assert_eq!(result.posterior_sd.len(), n_covariates);
        assert_eq!(result.credible_lower.len(), n_covariates);
        assert_eq!(result.credible_upper.len(), n_covariates);
        assert_eq!(result.shrinkage_factors.len(), n_covariates);
        assert_eq!(result.local_scales.len(), n_covariates);

        for j in 0..n_covariates {
            assert!(result.posterior_sd[j] >= 0.0);
            assert!(result.credible_lower[j] <= result.credible_upper[j]);
            assert!(result.shrinkage_factors[j] >= 0.0);
            assert!(result.shrinkage_factors[j] <= 1.0);
            assert!(result.local_scales[j] > 0.0);
        }

        assert!(result.global_scale > 0.0);
        assert!(result.effective_df >= 0.0);
        assert!(result.effective_df <= n_covariates as f64);
    }

    #[test]
    fn test_horseshoe_input_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];
        let covariates = vec![1.0, 2.0];

        let config = HorseshoeConfig::new(1.0, 100, 50, Some(42));
        let result = horseshoe_cox(time, event, covariates, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_horseshoe_small_global_tau() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0];

        let config = HorseshoeConfig::new(0.01, 300, 150, Some(42));
        let result = horseshoe_cox(time, event, covariates, 1, &config).unwrap();

        assert_eq!(result.posterior_mean.len(), 1);
        assert!(result.posterior_sd[0] >= 0.0);
        assert!(result.global_scale > 0.0);
    }

    #[test]
    fn test_horseshoe_credible_intervals_bracket_mean() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let covariates = vec![
            1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.5, 1.0,
            1.0, 0.0, 0.0,
        ];

        let config = HorseshoeConfig::new(1.0, 400, 200, Some(42));
        let result = horseshoe_cox(time, event, covariates, 2, &config).unwrap();

        for j in 0..2 {
            assert!(
                result.credible_lower[j] <= result.posterior_mean[j],
                "credible_lower[{j}]={} > posterior_mean[{j}]={}",
                result.credible_lower[j],
                result.posterior_mean[j]
            );
            assert!(
                result.credible_upper[j] >= result.posterior_mean[j],
                "credible_upper[{j}]={} < posterior_mean[{j}]={}",
                result.credible_upper[j],
                result.posterior_mean[j]
            );
        }
    }

    #[test]
    fn test_dirichlet_process_input_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0];
        let covariates = vec![1.0, 2.0, 3.0];

        let config = DirichletProcessConfig::new(1.0, 5, 100, 50, Some(42));
        let result = dirichlet_process_survival(time, event, covariates, 1, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_harmonic_function() {
        assert!((harmonic(1) - 1.0).abs() < 1e-10);
        assert!((harmonic(2) - 1.5).abs() < 1e-10);
        assert!((harmonic(3) - (1.0 + 0.5 + 1.0 / 3.0)).abs() < 1e-10);
    }
}
