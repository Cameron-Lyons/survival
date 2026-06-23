#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rmst_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let tau = 5.0;

        let result = compute_rmst(&time, &status, tau, 0.95);

        assert!(result.rmst > 0.0);
        assert!(result.rmst <= tau);
        assert!(result.se >= 0.0);
        assert!(result.ci_lower <= result.rmst);
        assert!(result.ci_upper >= result.rmst);
        assert_eq!(result.tau, tau);
    }

    #[test]
    fn test_compute_rmst_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_rmst(&time, &status, 5.0, 0.95);

        assert_eq!(result.rmst, 0.0);
        assert_eq!(result.variance, 0.0);
    }

    #[test]
    fn test_compute_rmst_no_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = compute_rmst(&time, &status, 5.0, 0.95);

        assert_eq!(result.rmst, 5.0);
    }

    #[test]
    fn test_compute_rmst_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_rmst(&exact_time, &status, 4.0, 0.95);
        let actual = compute_rmst(&near_time, &status, 4.0, 0.95);

        assert!((actual.rmst - expected.rmst).abs() < 1e-12);
        assert!((actual.variance - expected.variance).abs() < 1e-12);
        assert!((actual.se - expected.se).abs() < 1e-12);
        assert!((actual.ci_lower - expected.ci_lower).abs() < 1e-12);
        assert!((actual.ci_upper - expected.ci_upper).abs() < 1e-12);
    }

    #[test]
    fn test_compare_rmst_two_groups() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 0];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);

        assert!(result.rmst_group1.rmst > 0.0);
        assert!(result.rmst_group2.rmst > 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_compare_rmst_single_group() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 0];
        let group = vec![0, 0, 0];

        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);

        assert_eq!(result.rmst_diff, 0.0);
        assert_eq!(result.rmst_ratio, 1.0);
    }

    #[test]
    fn test_survival_quantile_median() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        assert!(result.median.is_some());
        assert_eq!(result.quantile, 0.5);
    }

    #[test]
    fn test_survival_quantile_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        assert!(result.median.is_none());
    }

    #[test]
    fn test_survival_quantile_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];

        let expected = compute_survival_quantile(&exact_time, &status, 0.5, 0.95);
        let actual = compute_survival_quantile(&near_time, &status, 0.5, 0.95);

        assert_eq!(actual.median, expected.median);
        assert_eq!(actual.ci_lower, expected.ci_lower);
        assert_eq!(actual.ci_upper, expected.ci_upper);
    }

    #[test]
    fn test_cumulative_incidence_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 2, 1, 0, 2];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(!result.time.is_empty());
        assert_eq!(result.event_types.len(), 2);
        assert_eq!(result.cif.len(), 2);
    }

    #[test]
    fn test_cumulative_incidence_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(result.time.is_empty());
        assert!(result.cif.is_empty());
    }

    #[test]
    fn test_cumulative_incidence_no_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = compute_cumulative_incidence(&time, &status);

        assert!(result.event_types.is_empty());
    }

    #[test]
    fn test_cumulative_incidence_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 2, 0, 0];

        let expected = compute_cumulative_incidence(&exact_time, &status);
        let actual = compute_cumulative_incidence(&near_time, &status);

        assert_eq!(actual.time, expected.time);
        assert_eq!(actual.n_risk, expected.n_risk);
        assert_eq!(actual.event_types, expected.event_types);
        for (actual_curve, expected_curve) in actual.cif.iter().zip(expected.cif.iter()) {
            for (actual_value, expected_value) in actual_curve.iter().zip(expected_curve.iter()) {
                assert!((*actual_value - *expected_value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_compute_nnt_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 1, 1, 0];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = compute_nnt(&time, &status, &group, 5.0, 0.95);

        assert!(result.nnt.is_finite() || result.nnt.is_infinite());
        assert_eq!(result.time_horizon, 5.0);
    }

    #[test]
    fn test_compute_nnt_groups_near_tied_event_times_at_horizon() {
        let exact_time = vec![1.0, 1.0, 3.0, 2.0, 2.0, 3.0];
        let near_time = vec![
            1.0,
            1.0 + crate::constants::TIME_EPSILON / 2.0,
            3.0,
            2.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
        ];
        let status = vec![1, 1, 0, 1, 0, 0];
        let group = vec![0, 0, 0, 1, 1, 1];

        let expected = compute_nnt(&exact_time, &status, &group, 2.0, 0.95);
        let actual = compute_nnt(&near_time, &status, &group, 2.0, 0.95);

        assert!((actual.nnt - expected.nnt).abs() < 1e-12);
        assert!(
            (actual.absolute_risk_reduction - expected.absolute_risk_reduction).abs() < 1e-12
        );
        assert!((actual.arr_ci_lower - expected.arr_ci_lower).abs() < 1e-12);
        assert!((actual.arr_ci_upper - expected.arr_ci_upper).abs() < 1e-12);
    }

    #[test]
    fn test_rmst_result_new() {
        let result = RMSTResult::new(3.5, 0.25, 0.5, 2.5, 4.5, 5.0);

        assert_eq!(result.rmst, 3.5);
        assert_eq!(result.variance, 0.25);
        assert_eq!(result.se, 0.5);
        assert_eq!(result.ci_lower, 2.5);
        assert_eq!(result.ci_upper, 4.5);
        assert_eq!(result.tau, 5.0);
    }

    #[test]
    fn test_chi_squared_cdf() {
        let cdf_0 = chi2_cdf(0.0, 1.0);
        assert!(cdf_0.abs() < 0.01);
        let cdf_small = chi2_cdf(0.5, 1.0);
        assert!(cdf_small > 0.0 && cdf_small < 1.0);
        let cdf_large = chi2_cdf(10.0, 1.0);
        assert!(cdf_large > 0.99);
    }

    #[test]
    fn test_piecewise_exp_likelihood_no_changepoints() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let censor_times = vec![2.5, 4.5];
        let lik = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
        assert!(lik.is_finite());
    }

    #[test]
    fn test_piecewise_exp_likelihood_with_changepoint() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let censor_times = vec![2.5, 7.5];
        let lik_null = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[]);
        let lik_alt = compute_piecewise_exp_likelihood(&event_times, &censor_times, &[5.0]);
        assert!(lik_null.is_finite());
        assert!(lik_alt.is_finite());
    }

    #[test]
    fn test_hazard_in_interval() {
        let event_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let censor_times = vec![2.5, 4.5];
        let hazard = compute_hazard_in_interval(&event_times, &censor_times, 0.0, 3.0);
        assert!(hazard >= 0.0);
    }

    #[test]
    fn test_rmst_optimal_threshold_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 0.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_rmst_optimal_threshold_no_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0, 0, 0, 0, 0];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 5.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_rmst_optimal_threshold_basic() {
        let mut time: Vec<f64> = Vec::new();
        let mut status: Vec<i32> = Vec::new();
        for i in 1..=20 {
            time.push(i as f64);
            status.push(1);
        }
        for i in 1..=10 {
            time.push(i as f64 + 0.5);
            status.push(0);
        }
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 3, 0.95);
        assert!(result.optimal_tau > 0.0);
        assert!(result.optimal_tau <= result.max_followup);
        assert!(result.rmst_at_optimal.rmst > 0.0);
    }

    #[test]
    fn test_rmst_optimal_threshold_few_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let result = compute_rmst_optimal_threshold(&time, &status, 0.05, 5, 0.95);
        assert_eq!(result.optimal_tau, 5.0);
        assert_eq!(result.n_changepoints, 0);
    }

    #[test]
    fn test_rmst_optimal_threshold_groups_near_tied_event_candidates() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0, 4.0, 4.5];
        let near_time = vec![
            1.0,
            1.0 + crate::constants::TIME_EPSILON / 2.0,
            2.0,
            3.0,
            4.0,
            4.5,
        ];
        let status = vec![1, 1, 1, 1, 0, 0];

        let expected = compute_rmst_optimal_threshold(&exact_time, &status, 1.0, 2, 0.95);
        let actual = compute_rmst_optimal_threshold(&near_time, &status, 1.0, 2, 0.95);

        assert!((actual.optimal_tau - expected.optimal_tau).abs() < crate::constants::TIME_EPSILON);
        assert_eq!(actual.n_changepoints, expected.n_changepoints);
        assert!((actual.rmst_at_optimal.rmst - expected.rmst_at_optimal.rmst).abs() < 1e-12);
    }

    #[test]
    fn test_changepoint_info_new() {
        let info = ChangepointInfo::new(5.0, 0.1, 0.2, 4.5, 0.03);
        assert_eq!(info.time, 5.0);
        assert_eq!(info.hazard_before, 0.1);
        assert_eq!(info.hazard_after, 0.2);
        assert_eq!(info.likelihood_ratio, 4.5);
        assert_eq!(info.p_value, 0.03);
    }

    #[test]
    fn test_rmst_optimal_threshold_result_new() {
        let rmst = RMSTResult::new(3.5, 0.25, 0.5, 2.5, 4.5, 5.0);
        let changepoints = vec![ChangepointInfo::new(3.0, 0.1, 0.2, 4.5, 0.03)];
        let result = RMSTOptimalThresholdResult::new(3.0, 5.0, changepoints, 1, rmst);
        assert_eq!(result.optimal_tau, 3.0);
        assert_eq!(result.max_followup, 5.0);
        assert_eq!(result.n_changepoints, 1);
        assert_eq!(result.changepoints.len(), 1);
    }

    #[test]
    fn public_rmst_wrappers_validate_inputs() {
        let err = rmst(vec![f64::NAN], vec![1], 1.0, None).unwrap_err();
        assert!(err.to_string().contains("time contains non-finite"));

        let err = rmst(vec![1.0], vec![2], 1.0, None).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = rmst(vec![1.0], vec![1], -1.0, None).unwrap_err();
        assert!(err.to_string().contains("tau must be non-negative"));

        let err = rmst(vec![1.0], vec![1], 1.0, Some(1.0)).unwrap_err();
        assert!(err.to_string().contains("confidence_level must be greater than 0"));
    }

    #[test]
    fn public_grouped_rmst_wrappers_validate_lengths_and_parameters() {
        let err = rmst_comparison(vec![1.0], vec![1], vec![0, 1], 1.0, None).unwrap_err();
        assert!(
            err.to_string()
                .contains("time, status, and group must have the same length")
        );

        let err = number_needed_to_treat(vec![1.0], vec![1], vec![0], f64::INFINITY, None)
            .unwrap_err();
        assert!(err.to_string().contains("time_horizon contains non-finite"));
    }

    #[test]
    fn public_quantile_and_incidence_wrappers_validate_inputs() {
        let err = survival_quantile(vec![1.0], vec![1], Some(0.0), None).unwrap_err();
        assert!(err.to_string().contains("quantile must be greater than 0"));

        let err = cumulative_incidence(vec![1.0], vec![-1]).unwrap_err();
        assert!(
            err.to_string()
                .contains("status must contain non-negative event codes")
        );

        let result = cumulative_incidence(vec![1.0, 2.0], vec![1, 2]).unwrap();
        assert_eq!(result.event_types, vec![1, 2]);
    }

    #[test]
    fn public_rmst_optimal_threshold_validates_parameters() {
        let err = rmst_optimal_threshold(vec![1.0], vec![1], Some(0.0), None, None).unwrap_err();
        assert!(err.to_string().contains("alpha must be greater than 0"));

        let err = rmst_optimal_threshold(vec![1.0], vec![1], None, Some(1), None).unwrap_err();
        assert!(
            err.to_string()
                .contains("min_events_per_interval must be at least 2")
        );
    }
}
