#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uno_c_index_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!(result.c_index > 0.9);
        assert!(result.std_error >= 0.0);
        assert!(result.ci_lower <= result.c_index);
        assert!(result.ci_upper >= result.c_index);
    }

    #[test]
    fn test_uno_c_index_random_prediction() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4, 0.7, 0.35];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
    }

    #[test]
    fn test_uno_c_index_with_tau() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result_full = uno_c_index_core(&time, &status, &risk_score, None);
        let result_tau = uno_c_index_core(&time, &status, &risk_score, Some(5.0));

        assert!(result_tau.tau <= 5.0);
        assert!(result_tau.comparable_pairs <= result_full.comparable_pairs);
    }

    #[test]
    fn test_uno_c_index_heavy_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 0, 1, 0, 0, 1, 0];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
    }

    #[test]
    fn test_uno_c_index_empty() {
        let result = uno_c_index_core(&[], &[], &[], None);
        assert!((result.c_index - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compare_uno_c_indices() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score_1 = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let risk_score_2 = vec![0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1];

        let result = compare_uno_c_indices_core(&time, &status, &risk_score_1, &risk_score_2, None);

        assert!((0.0..=1.0).contains(&result.c_index_1));
        assert!((0.0..=1.0).contains(&result.c_index_2));
        assert!((result.difference - (result.c_index_1 - result.c_index_2)).abs() < 1e-10);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_censoring_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let (times, values) = compute_censoring_km(&time, &status);

        assert!(!times.is_empty());
        assert_eq!(times.len(), values.len());
        for &v in &values {
            assert!((0.0..=1.0).contains(&v));
        }
        for i in 1..values.len() {
            assert!(values[i] <= values[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_c_index_decomposition_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!((0.0..=1.0).contains(&result.c_index_ee));
        assert!((0.0..=1.0).contains(&result.c_index_ec));
        assert!((0.0..=1.0).contains(&result.alpha));
        assert!(result.n_event_event_pairs > 0);
        assert!(result.n_event_censored_pairs > 0);
    }

    #[test]
    fn test_c_index_decomposition_all_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let risk_score = vec![0.9, 0.7, 0.5, 0.3, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!(result.c_index > 0.9);
        assert!(result.c_index_ee > 0.9);
        assert_eq!(result.n_event_censored_pairs, 0);
        assert!((result.alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_c_index_decomposition_heavy_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 0, 0, 1, 0, 0, 0];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!(result.n_event_censored_pairs > result.n_event_event_pairs);
        assert!(result.alpha < 0.5);
    }

    #[test]
    fn test_c_index_decomposition_empty() {
        let result = c_index_decomposition_core(&[], &[], &[], None);

        assert!((result.c_index - 0.5).abs() < 1e-10);
        assert!((result.c_index_ee - 0.5).abs() < 1e-10);
        assert!((result.c_index_ec - 0.5).abs() < 1e-10);
        assert_eq!(result.n_event_event_pairs, 0);
        assert_eq!(result.n_event_censored_pairs, 0);
    }

    #[test]
    fn test_c_index_decomposition_consistency() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];
        let risk_score = vec![0.9, 0.7, 0.8, 0.5, 0.4, 0.2];

        let decomp = c_index_decomposition_core(&time, &status, &risk_score, None);
        let uno = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((decomp.c_index - uno.c_index).abs() < 0.01);
    }

    #[test]
    fn test_gonen_heller_basic() {
        let lp = vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5];

        let result = gonen_heller_core(&lp);

        assert!((0.0..=1.0).contains(&result.cpe));
        assert!(result.cpe > 0.5);
        assert!(result.n_pairs > 0);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_gonen_heller_good_discrimination() {
        let lp = vec![10.0, 8.0, 6.0, 4.0, 2.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!(result.cpe > 0.9);
        assert_eq!(result.n_ties, 0);
    }

    #[test]
    fn test_gonen_heller_no_discrimination() {
        let lp = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!((result.cpe - 0.5).abs() < 1e-10);
        assert_eq!(result.n_pairs, 0);
        assert!(result.n_ties > 0);
    }

    #[test]
    fn test_gonen_heller_symmetric() {
        let lp1 = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let lp2: Vec<f64> = lp1.iter().map(|x| -x).collect();

        let result1 = gonen_heller_core(&lp1);
        let result2 = gonen_heller_core(&lp2);

        assert!((result1.cpe - result2.cpe).abs() < 1e-10);
    }

    #[test]
    fn test_gonen_heller_small_sample() {
        let lp = vec![1.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!((0.0..=1.0).contains(&result.cpe));
        assert_eq!(result.n_pairs, 1);
    }

    #[test]
    fn test_gonen_heller_single_element() {
        let lp = vec![1.0];

        let result = gonen_heller_core(&lp);

        assert!((result.cpe - 0.5).abs() < 1e-10);
        assert_eq!(result.n_pairs, 0);
    }
}
