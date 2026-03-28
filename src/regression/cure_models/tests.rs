#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_functions() {
        let logit = LinkFunction::Logit;
        assert!((logit.inv_link(0.0) - 0.5).abs() < 1e-6);
        assert!((logit.inv_link(logit.link(0.7)) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_weibull_surv() {
        assert!((weibull_surv(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10);
        assert!(weibull_surv(10.0, 1.0, 1.0) < 0.001);
    }

    #[test]
    fn test_mixture_cure_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = MixtureCureConfig::new(
            CureDistribution::Weibull,
            LinkFunction::Logit,
            50,
            1e-4,
            100,
        );

        let result = mixture_cure_model(time, status, vec![], vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
    }

    #[test]
    fn test_bounded_cumulative_hazard() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = BoundedCumulativeHazardConfig::new(CureDistribution::Weibull, 100, 1e-4, 1.0);

        let result = bounded_cumulative_hazard_model(time, status, vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
        assert!(result.alpha > 0.0);
    }

    #[test]
    fn test_non_mixture_cure() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];
        let config = NonMixtureCureConfig::new(
            NonMixtureType::GeometricGeneralized,
            CureDistribution::Weibull,
            100,
            1e-4,
            1.0,
        );

        let result = non_mixture_cure_model(time, status, vec![], &config).unwrap();
        assert!(result.cure_fraction >= 0.0 && result.cure_fraction <= 1.0);
        assert!(result.theta > 0.0);
    }

    #[test]
    fn test_non_mixture_types() {
        let time = vec![1.0, 2.0, 3.0, 5.0, 8.0, 12.0];
        let status = vec![1, 1, 0, 0, 0, 0];

        for model_type in [
            NonMixtureType::GeometricGeneralized,
            NonMixtureType::NegativeBinomial,
            NonMixtureType::Poisson,
            NonMixtureType::Destructive,
        ] {
            let config =
                NonMixtureCureConfig::new(model_type, CureDistribution::Weibull, 100, 1e-4, 1.0);
            let result = non_mixture_cure_model(time.clone(), status.clone(), vec![], &config);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_compare_cure_models() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 0, 0, 0, 0, 0];

        let result = compare_cure_models(time, status, vec![], None).unwrap();
        assert!(!result.model_names.is_empty());
        assert_eq!(result.model_names.len(), result.aic_values.len());
    }
}
