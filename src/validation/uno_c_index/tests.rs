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
    fn test_uno_c_index_parallel_path_is_finite() {
        let n = PARALLEL_THRESHOLD_LARGE + 5;
        let time: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let status = vec![1; n];
        let risk_score: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!(result.c_index.is_finite());
        assert!((result.c_index - 1.0).abs() < 1e-12);
        assert!(result.comparable_pairs > 0.0);
        assert!(result.std_error.is_finite());
    }

    #[test]
    fn test_uno_public_apis_reject_malformed_inputs() {
        let err = uno_c_index(vec![], vec![], vec![], None).unwrap_err();
        assert!(err.to_string().contains("time cannot be empty"));

        let err = uno_c_index(vec![f64::NAN], vec![1], vec![0.5], None).unwrap_err();
        assert!(err.to_string().contains("time contains NaN"));

        let err = uno_c_index(vec![1.0], vec![2], vec![0.5], None).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = uno_c_index(vec![1.0], vec![1], vec![f64::INFINITY], None).unwrap_err();
        assert!(err.to_string().contains("risk_score contains non-finite"));

        let err = uno_c_index(vec![1.0], vec![1], vec![0.5], Some(-1.0)).unwrap_err();
        assert!(err.to_string().contains("tau must be non-negative"));

        let err = compare_uno_c_indices(
            vec![1.0],
            vec![1],
            vec![0.5],
            vec![f64::NAN],
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("risk_score_2 contains NaN"));

        let err = c_index_decomposition(vec![1.0], vec![1], vec![0.5], Some(f64::INFINITY))
            .unwrap_err();
        assert!(err.to_string().contains("tau must be finite"));
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
    fn test_compare_uno_c_indices_parallel_path_is_finite() {
        let n = PARALLEL_THRESHOLD_LARGE + 5;
        let time: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let status = vec![1; n];
        let risk_score_1: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        let risk_score_2: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let result =
            compare_uno_c_indices_core(&time, &status, &risk_score_1, &risk_score_2, None);

        assert!((result.c_index_1 - 1.0).abs() < 1e-12);
        assert!(result.c_index_2.abs() < 1e-12);
        assert!(result.difference.is_finite());
        assert!(result.std_error_diff.is_finite());
    }

    #[test]
    fn test_uno_family_groups_near_tied_event_and_tau_times() {
        let exact_time = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let near_time = vec![
            1.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
            4.0,
        ];
        let status = vec![1, 1, 1, 0, 0];
        let risk_score = vec![0.9, 0.7, 0.8, 0.4, 0.2];
        let reverse_risk_score = vec![0.2, 0.4, 0.3, 0.7, 0.9];
        let tau = Some(2.0);

        let expected = uno_c_index_core(&exact_time, &status, &risk_score, tau);
        let actual = uno_c_index_core(&near_time, &status, &risk_score, tau);
        assert_eq!(actual.comparable_pairs, expected.comparable_pairs);
        assert_eq!(actual.concordant, expected.concordant);
        assert_eq!(actual.discordant, expected.discordant);
        assert_eq!(actual.tied_risk, expected.tied_risk);
        assert!((actual.c_index - expected.c_index).abs() < 1e-12);

        let expected_comparison =
            compare_uno_c_indices_core(&exact_time, &status, &risk_score, &reverse_risk_score, tau);
        let actual_comparison =
            compare_uno_c_indices_core(&near_time, &status, &risk_score, &reverse_risk_score, tau);
        assert!((actual_comparison.c_index_1 - expected_comparison.c_index_1).abs() < 1e-12);
        assert!((actual_comparison.c_index_2 - expected_comparison.c_index_2).abs() < 1e-12);
        assert!((actual_comparison.difference - expected_comparison.difference).abs() < 1e-12);
        assert!(
            (actual_comparison.variance_diff - expected_comparison.variance_diff).abs() < 1e-12
        );

        let expected_decomp = c_index_decomposition_core(&exact_time, &status, &risk_score, tau);
        let actual_decomp = c_index_decomposition_core(&near_time, &status, &risk_score, tau);
        assert_eq!(
            actual_decomp.n_event_event_pairs,
            expected_decomp.n_event_event_pairs
        );
        assert_eq!(
            actual_decomp.n_event_censored_pairs,
            expected_decomp.n_event_censored_pairs
        );
        assert!((actual_decomp.c_index - expected_decomp.c_index).abs() < 1e-12);
        assert!((actual_decomp.c_index_ee - expected_decomp.c_index_ee).abs() < 1e-12);
        assert!((actual_decomp.c_index_ec - expected_decomp.c_index_ec).abs() < 1e-12);
        assert!((actual_decomp.alpha - expected_decomp.alpha).abs() < 1e-12);
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
    fn test_c_index_decomposition_parallel_path_is_finite() {
        let n = PARALLEL_THRESHOLD_LARGE + 5;
        let time: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let status = vec![1; n];
        let risk_score: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!((result.c_index - 1.0).abs() < 1e-12);
        assert!((result.c_index_ee - 1.0).abs() < 1e-12);
        assert_eq!(result.n_event_event_pairs, n * (n - 1) / 2);
        assert_eq!(result.n_event_censored_pairs, 0);
        assert!(result.alpha.is_finite());
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

    #[test]
    fn test_gonen_heller_parallel_path_is_finite() {
        let n = PARALLEL_THRESHOLD_LARGE + 5;
        let lp: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();

        let result = gonen_heller_core(&lp);

        assert!(result.cpe.is_finite());
        assert!((0.0..=1.0).contains(&result.cpe));
        assert_eq!(result.n_pairs, n * (n - 1) / 2);
        assert_eq!(result.n_ties, 0);
        assert!(result.std_error.is_finite());
    }

    #[test]
    fn test_gonen_heller_public_api_rejects_malformed_inputs() {
        let err = gonen_heller_concordance(vec![]).unwrap_err();
        assert!(err.to_string().contains("linear_predictor must not be empty"));

        let err = gonen_heller_concordance(vec![0.1, f64::NAN]).unwrap_err();
        assert!(err.to_string().contains("linear_predictor contains NaN"));

        let err = gonen_heller_concordance(vec![0.1, f64::INFINITY]).unwrap_err();
        assert!(
            err.to_string()
                .contains("linear_predictor contains non-finite")
        );
    }
}
