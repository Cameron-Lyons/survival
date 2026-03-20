#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(5.0, 2.0) - 3.0).abs() < 1e-10);
        assert!((soft_threshold(-5.0, 2.0) - (-3.0)).abs() < 1e-10);
        assert!((soft_threshold(1.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_config() {
        let config = FastCoxConfig::new(
            0.1,
            1.0,
            1000,
            1e-7,
            ScreeningRule::Strong,
            None,
            10,
            true,
            true,
        )
        .unwrap();
        assert_eq!(config.lambda, 0.1);
        assert_eq!(config.l1_ratio, 1.0);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            FastCoxConfig::new(
                -0.1,
                1.0,
                1000,
                1e-7,
                ScreeningRule::None,
                None,
                10,
                true,
                true
            )
            .is_err()
        );
        assert!(
            FastCoxConfig::new(
                0.1,
                1.5,
                1000,
                1e-7,
                ScreeningRule::None,
                None,
                10,
                true,
                true
            )
            .is_err()
        );
        assert!(
            FastCoxConfig::new(0.1, 1.0, 0, 1e-7, ScreeningRule::None, None, 10, true, true)
                .is_err()
        );
    }

    #[test]
    fn test_fast_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 1];
        let config = FastCoxConfig::new(
            0.1,
            1.0,
            100,
            1e-5,
            ScreeningRule::None,
            None,
            10,
            true,
            true,
        )
        .unwrap();

        let result = fast_cox(x, 4, 2, time, status, &config, None, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
    }

    #[test]
    fn test_screening_rules() {
        let gradient = vec![0.5, 0.1, 0.8, 0.05];
        let lambda = 0.3;
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let safe = apply_safe_screening(&gradient, lambda, &beta, 4);
        let strong = apply_strong_screening(&gradient, lambda, None, &beta, 4);

        assert!(safe.contains(&0));
        assert!(safe.contains(&2));
        assert!(!safe.contains(&3));

        assert!(strong.contains(&0));
        assert!(strong.contains(&2));
    }

    #[test]
    fn test_fast_cox_path() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.1).collect();
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 0];

        let path = fast_cox_path(
            x,
            10,
            4,
            time,
            status,
            1.0,
            10,
            None,
            None,
            100,
            1e-5,
            ScreeningRule::Strong,
        )
        .unwrap();

        assert_eq!(path.lambdas.len(), 10);
        assert_eq!(path.coefficients.len(), 10);
        assert!(path.lambdas[0] >= path.lambdas[9]);
    }
}
