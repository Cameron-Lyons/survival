#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iv_cox() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let instruments = vec![0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.8];

        let config = IVCoxConfig::new(50, 1e-4, true, true);
        let result = iv_cox(time, event, treatment, instruments, vec![], &config).unwrap();

        assert!(result.first_stage_r2 >= 0.0 && result.first_stage_r2 <= 1.0);
    }

    #[test]
    fn test_rd_survival() {
        let time = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
            8.5, 9.5, 10.5,
        ];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
        let running_var = vec![
            0.35, 0.38, 0.41, 0.52, 0.55, 0.58, 0.44, 0.61, 0.47, 0.64, 0.32, 0.36, 0.43, 0.53,
            0.57, 0.62, 0.39, 0.59, 0.46, 0.67,
        ];
        let treatment = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
        ];

        let config = RDSurvivalConfig::new(Some(0.2), "triangular", 1, false).unwrap();
        let result =
            rd_survival(time, event, running_var, 0.5, treatment, vec![], &config).unwrap();

        assert!(result.n_left >= 10);
        assert!(result.n_right >= 10);
    }

    #[test]
    fn test_mediation_survival() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mediator = vec![0.5, 1.2, 0.4, 1.1, 0.3, 1.3, 0.6, 1.0];

        let config = MediationSurvivalConfig::new(50, 1e-4, 50, Some(42));
        let result = mediation_survival(time, event, treatment, mediator, vec![], &config).unwrap();

        assert!(result.proportion_mediated >= -1.0 && result.proportion_mediated <= 2.0);
    }

    #[test]
    fn test_g_estimation() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let config = GEstimationConfig::new(50, 1e-4, "aft");
        let result = g_estimation_aft(time, event, treatment, vec![], &config).unwrap();

        assert!(!result.psi.is_empty());
        assert!(result.treatment_effect_ratio > 0.0);
    }

    #[test]
    fn test_iv_cox_dimension_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1, 1];
        let treatment = vec![0.0, 1.0, 0.0];
        let instruments = vec![0.5, 0.8, 0.3];
        let config = IVCoxConfig::new(50, 1e-4, true, true);
        let result = iv_cox(time, event, treatment, instruments, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_iv_cox_no_instruments() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0];
        let config = IVCoxConfig::new(50, 1e-4, true, true);
        let result = iv_cox(time, event, treatment, vec![], vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_iv_cox_output_properties() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let instruments = vec![0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.8, 0.15, 0.85];

        let config = IVCoxConfig::new(100, 1e-6, true, true);
        let result = iv_cox(time, event, treatment, instruments, vec![], &config).unwrap();

        assert!(result.first_stage_r2 >= 0.0 && result.first_stage_r2 <= 1.0);
        assert!(result.treatment_pvalue >= 0.0 && result.treatment_pvalue <= 1.0);
        assert!(result.sargan_pvalue >= 0.0 && result.sargan_pvalue <= 1.0);
        assert!(result.treatment_coef.is_finite());
        assert!(result.n_iter > 0);
    }

    #[test]
    fn test_iv_cox_with_covariates() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let instruments = vec![0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.2, 0.8];
        let covariates = vec![0.5, 1.2, 0.8, 0.3, 0.9, 0.4, 1.1, 0.7];

        let config = IVCoxConfig::new(50, 1e-4, true, true);
        let result = iv_cox(time, event, treatment, instruments, covariates, &config).unwrap();

        assert_eq!(result.covariate_coef.len(), 1);
        assert_eq!(result.covariate_se.len(), 1);
        assert!(result.treatment_coef.is_finite());
    }

    #[test]
    fn test_rd_survival_insufficient_observations() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];
        let running_var = vec![0.3, 0.4, 0.6, 0.7];
        let treatment = vec![0.0, 0.0, 1.0, 1.0];

        let config = RDSurvivalConfig::new(Some(0.2), "triangular", 1, false).unwrap();
        let result = rd_survival(time, event, running_var, 0.5, treatment, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_rd_survival_dimension_mismatch() {
        let time = vec![1.0, 2.0];
        let event = vec![1, 0, 1];
        let running_var = vec![0.3, 0.7];
        let treatment = vec![0.0, 1.0];

        let config = RDSurvivalConfig::new(Some(0.5), "triangular", 1, false).unwrap();
        let result = rd_survival(time, event, running_var, 0.5, treatment, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_rd_survival_kernel_types() {
        let n = 40;
        let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let event: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let running_var: Vec<f64> = (0..n).map(|i| 0.3 + 0.4 * i as f64 / n as f64).collect();
        let treatment: Vec<f64> = (0..n)
            .map(|i| if running_var[i] >= 0.5 { 1.0 } else { 0.0 })
            .collect();

        for kernel in &["triangular", "uniform", "epanechnikov"] {
            let config = RDSurvivalConfig::new(Some(0.25), kernel, 1, false).unwrap();
            let result = rd_survival(
                time.clone(),
                event.clone(),
                running_var.clone(),
                0.5,
                treatment.clone(),
                vec![],
                &config,
            )
            .unwrap();
            assert!(result.survival_left >= 0.0 && result.survival_left <= 1.0);
            assert!(result.survival_right >= 0.0 && result.survival_right <= 1.0);
        }
    }

    #[test]
    fn test_mediation_dimension_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1, 1];
        let treatment = vec![0.0, 1.0, 0.0];
        let mediator = vec![0.5, 1.2, 0.4];
        let config = MediationSurvivalConfig::new(50, 1e-4, 10, Some(42));
        let result = mediation_survival(time, event, treatment, mediator, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mediation_output_properties() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mediator = vec![0.5, 1.2, 0.4, 1.1, 0.3, 1.3, 0.6, 1.0, 0.45, 1.15];

        let config = MediationSurvivalConfig::new(50, 1e-4, 20, Some(42));
        let result = mediation_survival(time, event, treatment, mediator, vec![], &config).unwrap();

        assert!(result.total_se >= 0.0);
        assert!(result.direct_se >= 0.0);
        assert!(result.indirect_se >= 0.0);
        assert!(result.proportion_mediated >= -1.0 && result.proportion_mediated <= 2.0);
        assert!(result.total_pvalue >= 0.0 && result.total_pvalue <= 1.0);
        assert!(result.direct_pvalue >= 0.0 && result.direct_pvalue <= 1.0);
        assert!(result.indirect_pvalue >= 0.0 && result.indirect_pvalue <= 1.0);
    }

    #[test]
    fn test_g_estimation_dimension_mismatch() {
        let time = vec![1.0, 2.0];
        let event = vec![1, 0, 1];
        let treatment = vec![0.0, 1.0];
        let config = GEstimationConfig::new(50, 1e-4, "aft");
        let result = g_estimation_aft(time, event, treatment, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_g_estimation_output_properties() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

        let config = GEstimationConfig::new(100, 1e-6, "aft");
        let result = g_estimation_aft(time, event, treatment, vec![], &config).unwrap();

        assert_eq!(result.psi.len(), 1);
        assert_eq!(result.se.len(), 1);
        assert_eq!(result.z_scores.len(), 1);
        assert_eq!(result.p_values.len(), 1);
        assert_eq!(result.counterfactual_times.len(), 10);
        assert!(result.treatment_effect_ratio > 0.0);
        assert!(result.p_values.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!(result.counterfactual_times.iter().all(|t| t.is_finite()));
    }

    #[test]
    fn test_g_estimation_with_covariates() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let covariates = vec![0.5, 1.2, 0.8, 0.3, 0.9, 0.4, 1.1, 0.7];

        let config = GEstimationConfig::new(50, 1e-4, "aft");
        let result = g_estimation_aft(time, event, treatment, covariates, &config).unwrap();

        assert_eq!(result.psi.len(), 2);
        assert_eq!(result.se.len(), 2);
    }

    #[test]
    fn test_kernel_weight_outside_bandwidth() {
        assert_eq!(kernel_weight(2.0, 1.0, "triangular"), 0.0);
        assert_eq!(kernel_weight(-2.0, 1.0, "triangular"), 0.0);
    }

    #[test]
    fn test_kernel_weight_at_center() {
        assert!((kernel_weight(0.0, 1.0, "triangular") - 1.0).abs() < 1e-10);
        assert!((kernel_weight(0.0, 1.0, "uniform") - 1.0).abs() < 1e-10);
        assert!((kernel_weight(0.0, 1.0, "epanechnikov") - 0.75).abs() < 1e-10);
    }
}
