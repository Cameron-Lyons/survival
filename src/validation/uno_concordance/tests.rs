#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            (actual - expected).abs() <= 1e-11 * scale,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    fn assert_uno_accumulators_close(
        actual: &UnoCIndexAccumulator,
        expected: &UnoCIndexAccumulator,
    ) {
        assert_close(actual.concordant, expected.concordant);
        assert_close(actual.discordant, expected.discordant);
        assert_close(actual.tied, expected.tied);
        assert_close(actual.total_weight, expected.total_weight);
        assert_eq!(actual.influence_sums.len(), expected.influence_sums.len());
        for (&actual_value, &expected_value) in actual
            .influence_sums
            .iter()
            .zip(&expected.influence_sums)
        {
            assert_close(actual_value, expected_value);
        }
        assert_close(
            actual.concordant + actual.discordant + actual.tied,
            actual.total_weight,
        );
        let influence_sum = actual.influence_sums.iter().sum::<f64>();
        assert!(
            influence_sum.abs() <= 1e-11 * actual.total_weight.max(1.0),
            "influence sum={influence_sum:?}, total weight={:?}",
            actual.total_weight
        );
    }

    fn assert_uno_results_close(actual: &UnoCIndexResult, expected: &UnoCIndexResult) {
        assert_close(actual.c_index, expected.c_index);
        assert_close(actual.concordant, expected.concordant);
        assert_close(actual.discordant, expected.discordant);
        assert_close(actual.tied_risk, expected.tied_risk);
        assert_close(actual.comparable_pairs, expected.comparable_pairs);
        assert_close(actual.variance, expected.variance);
        assert_close(actual.std_error, expected.std_error);
        assert_close(actual.ci_lower, expected.ci_lower);
        assert_close(actual.ci_upper, expected.ci_upper);
        assert_close(actual.tau, expected.tau);
    }

    fn assert_ranked_uno_matches_quadratic(
        time: &[f64],
        status: &[i32],
        risk_score: &[f64],
        tau: Option<f64>,
    ) {
        let tau_value = tau.unwrap_or_else(|| time.iter().copied().fold(0.0, f64::max));
        let (km_times, km_values) = compute_censoring_km(time, status);
        let event_weights = uno_event_weights(time, status, &km_times, &km_values, tau_value);
        let ranked = uno_c_index_ranked_accumulator(time, risk_score, &event_weights);
        let quadratic = uno_c_index_quadratic_accumulator(time, risk_score, &event_weights);
        assert_uno_accumulators_close(&ranked, &quadratic);
        assert_uno_results_close(
            &uno_c_index_core(time, status, risk_score, tau),
            &uno_c_index_core_quadratic(time, status, risk_score, tau),
        );
    }

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
    fn test_ranked_uno_matches_quadratic_at_time_tau_and_risk_boundaries() {
        let epsilon = crate::constants::TIME_EPSILON;
        let time = vec![
            1.0,
            1.0 + 0.5 * epsilon,
            1.0 + epsilon,
            1.0 + 1.5 * epsilon,
            2.0,
            2.0,
            3.0,
            4.0,
        ];
        let status = vec![1, 1, 1, 0, 1, 0, 1, 0];
        let risk_score = vec![-0.0, 0.0, 1.0, 1.0, -1.0, 2.0, -1.0, 0.5];

        for tau in [
            None,
            Some(0.0),
            Some(1.0),
            Some(1.0 + 0.5 * epsilon),
            Some(1.0 + epsilon),
            Some(2.5),
        ] {
            assert_ranked_uno_matches_quadratic(&time, &status, &risk_score, tau);
        }
    }

    #[test]
    fn test_ranked_uno_matches_quadratic_at_ipcw_survival_floor() {
        let n = 202;
        let time: Vec<f64> = (1..=n).map(|value| value as f64).collect();
        let mut status = vec![0; n];
        status[n - 2] = 1;
        status[n - 1] = 1;
        let risk_score: Vec<f64> = (0..n).map(|idx| ((idx * 37) % n) as f64).collect();

        let (km_times, km_values) = compute_censoring_km(&time, &status);
        let event_weights = uno_event_weights(
            &time,
            &status,
            &km_times,
            &km_values,
            *time.last().expect("fixture is non-empty"),
        );
        assert_close(
            event_weights[n - 2],
            1.0 / (IPCW_SURVIVAL_FLOOR * IPCW_SURVIVAL_FLOOR),
        );
        assert_ranked_uno_matches_quadratic(&time, &status, &risk_score, None);
    }

    #[test]
    fn test_ranked_uno_matches_quadratic_on_deterministic_random_inputs() {
        let epsilon = crate::constants::TIME_EPSILON;
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        for n in 1..=64 {
            for case_idx in 0..12 {
                let mut time = Vec::with_capacity(n);
                let mut status = Vec::with_capacity(n);
                let mut risk_score = Vec::with_capacity(n);
                for _ in 0..n {
                    crate::internal::statistical::lcg64_next(&mut state);
                    let bucket = (state % 11) as f64;
                    let offset = match (state >> 8) % 4 {
                        0 => 0.0,
                        1 => 0.5 * epsilon,
                        2 => epsilon,
                        _ => 1.5 * epsilon,
                    };
                    time.push(bucket + offset);
                    status.push(i32::from((state >> 16).is_multiple_of(3)));
                    let risk = match (state >> 24) % 9 {
                        0 => -0.0,
                        1 => 0.0,
                        value => value as f64 - 5.0,
                    };
                    risk_score.push(risk);
                }
                let tau = match case_idx % 4 {
                    0 => None,
                    1 => Some(4.0),
                    2 => Some(4.0 + 0.5 * epsilon),
                    _ => Some(4.0 + epsilon),
                };
                assert_ranked_uno_matches_quadratic(&time, &status, &risk_score, tau);
            }
        }
    }

    #[test]
    fn test_ranked_uno_matches_quadratic_across_old_parallel_threshold() {
        for n in [
            PARALLEL_THRESHOLD_LARGE - 1,
            PARALLEL_THRESHOLD_LARGE,
            PARALLEL_THRESHOLD_LARGE + 1,
        ] {
            let time: Vec<f64> = (0..n)
                .map(|idx| 1.0 + (idx % 127) as f64 * 0.25 + (idx / 127) as f64 * 0.01)
                .collect();
            let status: Vec<i32> = (0..n).map(|idx| i32::from(idx % 5 != 0)).collect();
            let risk_score: Vec<f64> = (0..n)
                .map(|idx| ((idx.wrapping_mul(37) + 7) % 41) as f64)
                .collect();
            assert_ranked_uno_matches_quadratic(&time, &status, &risk_score, Some(20.0));
        }
    }

    #[test]
    fn test_uno_c_index_large_input_is_finite() {
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
