#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copula_density() {
        let d = copula_density(0.5, 0.5, 2.0, &CopulaType::Clayton);
        assert!(d > 0.0);

        let d_indep = copula_density(0.5, 0.5, 0.0, &CopulaType::Independent);
        assert!((d_indep - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kendall_tau() {
        let tau = kendall_tau_from_theta(2.0, &CopulaType::Clayton);
        assert!((tau - 0.5).abs() < 1e-6);

        let tau_gumbel = kendall_tau_from_theta(2.0, &CopulaType::Gumbel);
        assert!((tau_gumbel - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sensitivity_bounds() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let treatment = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let config = SensitivityBoundsConfig::new(Some(vec![1.0, 2.0]), 50, "rosenbaum");
        let result =
            sensitivity_bounds_survival(time, event, treatment, vec![], 10.0, &config).unwrap();

        assert_eq!(result.gamma_values.len(), 2);
        assert_eq!(result.rmst_lower.len(), 2);
    }

    #[test]
    fn test_mnar_sensitivity() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];

        let config = MNARSurvivalConfig::new(Some(vec![-0.5, 0.0, 0.5]), "tilt");
        let result = mnar_sensitivity_survival(time, event, vec![], &config).unwrap();

        assert_eq!(result.delta_values.len(), 3);
        assert_eq!(result.adjusted_survival.len(), 3);
    }

    #[test]
    fn test_copula_type_parsing() {
        assert_eq!(CopulaType::new("clayton").unwrap(), CopulaType::Clayton);
        assert_eq!(CopulaType::new("frank").unwrap(), CopulaType::Frank);
        assert_eq!(CopulaType::new("gumbel").unwrap(), CopulaType::Gumbel);
        assert_eq!(CopulaType::new("gaussian").unwrap(), CopulaType::Gaussian);
        assert_eq!(CopulaType::new("normal").unwrap(), CopulaType::Gaussian);
        assert_eq!(
            CopulaType::new("independent").unwrap(),
            CopulaType::Independent
        );
        assert_eq!(CopulaType::new("indep").unwrap(), CopulaType::Independent);
        assert!(CopulaType::new("unknown_type").is_err());
    }

    #[test]
    fn test_copula_cdf_independence() {
        let u = 0.3;
        let v = 0.7;
        let cdf = copula_cdf(u, v, 0.0, &CopulaType::Independent);
        assert!((cdf - u * v).abs() < 1e-6);
    }

    #[test]
    fn test_copula_cdf_boundary_values() {
        for copula in &[CopulaType::Clayton, CopulaType::Gumbel] {
            let theta = match copula {
                CopulaType::Clayton => 2.0,
                CopulaType::Gumbel => 2.0,
                _ => 1.0,
            };
            let cdf = copula_cdf(0.5, 0.5, theta, copula);
            assert!((0.0..=1.0).contains(&cdf));
        }
    }

    #[test]
    fn test_copula_censoring_model_dimension_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1, 1];
        let censoring = vec![0, 1, 0];
        let config = CopulaCensoringConfig::new(CopulaType::Clayton, Some(1.0), 50, 1e-4, 20);
        let result = copula_censoring_model(time, event, censoring, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_copula_censoring_model_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let censoring = vec![0, 1, 0, 0, 1, 0, 1, 0];

        let config = CopulaCensoringConfig::new(CopulaType::Clayton, Some(1.0), 20, 1e-4, 20);
        let result = copula_censoring_model(time, event, censoring, vec![], &config).unwrap();

        assert!(result.theta > 0.0);
        assert!(result.theta_se > 0.0);
        assert!(result.kendall_tau >= -1.0 && result.kendall_tau <= 1.0);
        assert_eq!(result.eval_times.len(), 20);
        assert_eq!(result.marginal_survival_t.len(), 20);
        assert_eq!(result.marginal_survival_c.len(), 20);
        assert_eq!(result.joint_survival.len(), 20);
        assert!(result.n_iter > 0);
    }

    #[test]
    fn test_sensitivity_bounds_dimension_mismatch() {
        let time = vec![1.0, 2.0];
        let event = vec![1, 0, 1];
        let treatment = vec![0, 1];
        let config = SensitivityBoundsConfig::new(Some(vec![1.0, 2.0]), 50, "rosenbaum");
        let result = sensitivity_bounds_survival(time, event, treatment, vec![], 10.0, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_sensitivity_bounds_output_ranges() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let treatment = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let config = SensitivityBoundsConfig::new(Some(vec![1.0, 1.5, 2.0, 3.0]), 20, "rosenbaum");
        let result =
            sensitivity_bounds_survival(time, event, treatment, vec![], 10.0, &config).unwrap();

        assert_eq!(result.gamma_values.len(), 4);
        assert_eq!(result.rmst_lower.len(), 4);
        assert_eq!(result.rmst_upper.len(), 4);
        assert_eq!(result.hazard_ratio_lower.len(), 4);
        assert_eq!(result.hazard_ratio_upper.len(), 4);

        for i in 0..4 {
            assert!(result.rmst_lower[i] <= result.rmst_upper[i]);
            assert!(result.hazard_ratio_lower[i] <= result.hazard_ratio_upper[i]);
            assert!(result.hazard_ratio_lower[i] > 0.0);
        }

        assert!(result.point_estimate.is_finite());
    }

    #[test]
    fn test_mnar_sensitivity_dimension_mismatch() {
        let time = vec![1.0, 2.0];
        let event = vec![1, 0, 1];
        let config = MNARSurvivalConfig::new(None, "tilt");
        let result = mnar_sensitivity_survival(time, event, vec![], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mnar_zero_delta_matches_reference() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let event = vec![1, 0, 1, 1, 0, 1, 0, 1];

        let config = MNARSurvivalConfig::new(Some(vec![0.0]), "tilt");
        let result = mnar_sensitivity_survival(time, event, vec![], &config).unwrap();

        assert_eq!(result.adjusted_survival.len(), 1);
        for (adj, ref_s) in result.adjusted_survival[0]
            .iter()
            .zip(result.reference_survival.iter())
        {
            assert!((adj - ref_s).abs() < 1e-6);
        }
    }

    #[test]
    fn test_probit_symmetric() {
        let p1 = probit(0.25);
        let p2 = probit(0.75);
        assert!((p1 + p2).abs() < 1e-6);
    }

    #[test]
    fn test_probit_median() {
        let p = probit(0.5);
        assert!(p.abs() < 1e-6);
    }
}
