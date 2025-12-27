//! Validation tests comparing outputs against known statistical results.
//! Reference values computed using R's survival package and other established implementations.

#[cfg(test)]
mod tests {
    use crate::surv_analysis::nelson_aalen::{nelson_aalen, stratified_km};
    use crate::validation::calibration::{calibration_curve, stratify_risk, time_dependent_auc};
    use crate::validation::landmark::{
        compute_conditional_survival, compute_hazard_ratio, compute_landmark, compute_life_table,
        compute_survival_at_times,
    };
    use crate::validation::logrank::{WeightType, logrank_trend_test, weighted_logrank_test};
    use crate::validation::power::{power_logrank, sample_size_freedman, sample_size_logrank};
    use crate::validation::rmst::{
        compare_rmst, compute_cumulative_incidence, compute_rmst, compute_survival_quantile,
    };

    const TOLERANCE: f64 = 1e-4;
    const LOOSE_TOLERANCE: f64 = 1e-2;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol || (a.is_nan() && b.is_nan())
    }

    // ==================== NELSON-AALEN TESTS ====================
    // Reference: R survival package - survfit(..., type="fh")

    #[test]
    fn test_nelson_aalen_simple() {
        // Simple dataset: times with events
        // R: library(survival); fit <- survfit(Surv(time, status) ~ 1, type="fh")
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1]; // all events

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(result.time.len(), 5);
        assert_eq!(result.n_events, vec![1, 1, 1, 1, 1]);

        // Cumulative hazard: H(t) = sum(d_i / n_i)
        // H(1) = 1/5 = 0.2
        // H(2) = 1/5 + 1/4 = 0.45
        // H(3) = 0.45 + 1/3 = 0.7833...
        // H(4) = 0.7833 + 1/2 = 1.2833...
        // H(5) = 1.2833 + 1/1 = 2.2833...
        assert!(approx_eq(result.cumulative_hazard[0], 0.2, TOLERANCE));
        assert!(approx_eq(result.cumulative_hazard[1], 0.45, TOLERANCE));
        assert!(approx_eq(
            result.cumulative_hazard[2],
            0.7833,
            LOOSE_TOLERANCE
        ));
        assert!(approx_eq(
            result.cumulative_hazard[3],
            1.2833,
            LOOSE_TOLERANCE
        ));
        assert!(approx_eq(
            result.cumulative_hazard[4],
            2.2833,
            LOOSE_TOLERANCE
        ));
    }

    #[test]
    fn test_nelson_aalen_with_censoring() {
        // Dataset with censoring
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0]; // alternating events and censoring

        let result = nelson_aalen(&time, &status, None, 0.95);

        // Only event times should appear
        assert_eq!(result.time, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.n_events, vec![1, 1, 1]);

        // H(1) = 1/6
        // After t=1: 5 at risk, t=2 censored -> 4 at risk
        // H(3) = 1/6 + 1/4
        // After t=3: 3 at risk, t=4 censored -> 2 at risk
        // H(5) = 1/6 + 1/4 + 1/2
        assert!(approx_eq(result.cumulative_hazard[0], 1.0 / 6.0, TOLERANCE));
        assert!(approx_eq(
            result.cumulative_hazard[1],
            1.0 / 6.0 + 1.0 / 4.0,
            TOLERANCE
        ));
        assert!(approx_eq(
            result.cumulative_hazard[2],
            1.0 / 6.0 + 1.0 / 4.0 + 1.0 / 2.0,
            TOLERANCE
        ));
    }

    #[test]
    fn test_nelson_aalen_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = nelson_aalen(&time, &status, None, 0.95);
        assert!(result.time.is_empty());
        assert!(result.cumulative_hazard.is_empty());
    }

    #[test]
    fn test_nelson_aalen_all_censored() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = nelson_aalen(&time, &status, None, 0.95);
        assert!(result.time.is_empty()); // No event times
    }

    #[test]
    fn test_stratified_km_two_groups() {
        // Two strata with different survival
        let time = vec![1.0, 2.0, 3.0, 1.0, 3.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 1];
        let strata = vec![0, 0, 0, 1, 1, 1];

        let result = stratified_km(&time, &status, &strata, 0.95);

        assert_eq!(result.strata, vec![0, 1]);
        assert_eq!(result.times.len(), 2);

        // Stratum 0: events at t=1,2; censored at t=3
        // S(1) = 2/3, S(2) = 1/3
        assert!(approx_eq(result.survival[0][0], 2.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[0][1], 1.0 / 3.0, TOLERANCE));

        // Stratum 1: events at t=1,3,5
        // S(1) = 2/3, S(3) = 1/3, S(5) = 0
        assert!(approx_eq(result.survival[1][0], 2.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[1][1], 1.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[1][2], 0.0, TOLERANCE));
    }

    // ==================== RMST TESTS ====================
    // Reference: R survRM2 package

    #[test]
    fn test_rmst_simple() {
        // Simple case: constant survival for verification
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = compute_rmst(&time, &status, 5.0, 0.95);

        // RMST is area under survival curve up to tau
        // With all events, survival drops stepwise
        assert!(result.rmst > 0.0);
        assert!(result.rmst < 5.0); // Must be less than tau
        assert!(result.se > 0.0);
        assert!(result.ci_lower < result.rmst);
        assert!(result.ci_upper > result.rmst);
    }

    #[test]
    fn test_rmst_no_events_before_tau() {
        // All censored before tau - RMST should equal tau
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];

        let result = compute_rmst(&time, &status, 10.0, 0.95);

        assert!(approx_eq(result.rmst, 10.0, TOLERANCE));
    }

    #[test]
    fn test_rmst_comparison() {
        // Two groups with different survival
        let time = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);

        // Group 1 has longer times, so higher RMST
        assert!(result.rmst_group2.rmst > result.rmst_group1.rmst);
        assert!(result.rmst_diff < 0.0); // group1 - group2 < 0
    }

    #[test]
    fn test_survival_quantile_median() {
        // Test median survival time
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        // Median is when S(t) <= 0.5
        // S(1)=5/6, S(2)=4/6, S(3)=3/6=0.5
        assert!(result.median.is_some());
        assert!(approx_eq(result.median.unwrap(), 3.0, TOLERANCE));
    }

    #[test]
    fn test_survival_quantile_not_reached() {
        // High survival - median not reached
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 0]; // Only one event

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        // S never drops below 0.5, so median is None
        // S(1) = 2/3 > 0.5
        assert!(result.median.is_none());
    }

    // ==================== CUMULATIVE INCIDENCE TESTS ====================

    #[test]
    fn test_cumulative_incidence_single_event_type() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = compute_cumulative_incidence(&time, &status);

        assert_eq!(result.event_types, vec![1]);
        assert_eq!(result.cif.len(), 1);

        // CIF should equal 1 - KM survival
        // At t=5, everyone has had the event
        let last_cif = result.cif[0].last().unwrap();
        assert!(approx_eq(*last_cif, 1.0, TOLERANCE));
    }

    #[test]
    fn test_cumulative_incidence_competing_risks() {
        // Two competing event types
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 2, 1, 2, 1, 0]; // 1=event A, 2=event B, 0=censored

        let result = compute_cumulative_incidence(&time, &status);

        assert_eq!(result.event_types.len(), 2);
        assert!(result.event_types.contains(&1));
        assert!(result.event_types.contains(&2));

        // Sum of CIFs should be <= 1 - S(t)
        let last_idx = result.time.len() - 1;
        let sum_cif: f64 = result.cif.iter().map(|c| c[last_idx]).sum();
        assert!(sum_cif <= 1.0 + TOLERANCE);
    }

    // ==================== LOG-RANK TEST TESTS ====================
    // Reference: R survival::survdiff

    #[test]
    fn test_logrank_identical_groups() {
        // Identical survival in both groups - p-value should be high
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        // With identical survival, chi-square should be near 0, p near 1
        assert!(result.statistic < 1.0);
        assert!(result.p_value > 0.3);
    }

    #[test]
    fn test_logrank_different_groups() {
        // Very different survival - low p-value expected
        let time = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        // Very different survival should give significant result
        assert!(result.statistic > 3.0);
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_logrank_wilcoxon_weight() {
        // Wilcoxon weights early differences more heavily
        let time = vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let lr_result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        let wilcox_result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        // Both should give a result
        assert!(lr_result.statistic >= 0.0);
        assert!(wilcox_result.statistic >= 0.0);
        assert_eq!(wilcox_result.weight_type, "Wilcoxon");
    }

    #[test]
    fn test_logrank_trend() {
        // Test for trend across ordered groups
        // Group 0 has early events, group 2 has late events
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let result = logrank_trend_test(&time, &status, &group, None);

        // Higher group number = later events = better survival = fewer events than expected
        // This means U statistic is negative (decreasing hazard with group)
        assert!(result.trend_direction == "increasing" || result.trend_direction == "decreasing");
        // Just verify the test runs and gives a reasonable p-value
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    // ==================== HAZARD RATIO TESTS ====================

    #[test]
    fn test_hazard_ratio_equal_groups() {
        // Equal survival - HR should be near 1
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = compute_hazard_ratio(&time, &status, &group, 0.95);

        // HR should be close to 1
        assert!(approx_eq(result.hazard_ratio, 1.0, 0.5));
        assert!(result.ci_lower < 1.0 && result.ci_upper > 1.0); // CI should include 1
    }

    #[test]
    fn test_hazard_ratio_higher_risk() {
        // Group 0 has much higher hazard
        let time = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = compute_hazard_ratio(&time, &status, &group, 0.95);

        // Group 0 has earlier events, so HR > 1
        assert!(result.hazard_ratio > 1.0);
    }

    // ==================== SAMPLE SIZE / POWER TESTS ====================
    // Reference: R powerSurvEpi package

    #[test]
    fn test_sample_size_schoenfeld() {
        // Classic Schoenfeld formula
        // HR=0.5, power=0.8, alpha=0.05, 1:1 allocation
        let result = sample_size_logrank(0.5, 0.8, 0.05, 1.0, 2);

        // Expected ~65 events for HR=0.5, 80% power (from R)
        assert!(result.n_events >= 50 && result.n_events <= 80);
        assert!(result.n_per_group.len() == 2);
    }

    #[test]
    fn test_sample_size_smaller_effect() {
        // Smaller effect size needs more events
        let result_large = sample_size_logrank(0.5, 0.8, 0.05, 1.0, 2);
        let result_small = sample_size_logrank(0.7, 0.8, 0.05, 1.0, 2);

        assert!(result_small.n_events > result_large.n_events);
    }

    #[test]
    fn test_sample_size_higher_power() {
        // Higher power needs more events
        let result_80 = sample_size_logrank(0.6, 0.8, 0.05, 1.0, 2);
        let result_90 = sample_size_logrank(0.6, 0.9, 0.05, 1.0, 2);

        assert!(result_90.n_events > result_80.n_events);
    }

    #[test]
    fn test_sample_size_freedman() {
        // Freedman method with event probability
        let result = sample_size_freedman(0.6, 0.8, 0.05, 0.3, 1.0, 2);

        assert!(result.n_total > 0);
        assert!(result.n_events > 0);
        assert_eq!(result.method, "Freedman");
    }

    #[test]
    fn test_power_calculation() {
        // Calculate power for given sample size
        let power = power_logrank(100, 0.6, 0.05, 1.0, 2);

        assert!(power > 0.0 && power < 1.0);

        // More events = more power
        let power_more = power_logrank(200, 0.6, 0.05, 1.0, 2);
        assert!(power_more > power);
    }

    #[test]
    fn test_power_approaches_one() {
        // Very large sample should have power near 1
        let power = power_logrank(1000, 0.5, 0.05, 1.0, 2);
        assert!(power > 0.99);
    }

    // ==================== LANDMARK ANALYSIS TESTS ====================

    #[test]
    fn test_landmark_excludes_early_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = compute_landmark(&time, &status, 2.5);

        // Should exclude times <= 2.5
        assert_eq!(result.n_excluded, 2);
        assert_eq!(result.n_at_risk, 3);
        assert_eq!(result.time, vec![0.5, 1.5, 2.5]); // Shifted by landmark
    }

    #[test]
    fn test_landmark_all_excluded() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];

        let result = compute_landmark(&time, &status, 5.0);

        assert_eq!(result.n_at_risk, 0);
        assert_eq!(result.n_excluded, 3);
    }

    #[test]
    fn test_conditional_survival() {
        // S(5|3) = S(5)/S(3)
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];

        let result = compute_conditional_survival(&time, &status, 2.0, 5.0, 0.95);

        // S(2) = 4/6, S(5) = 1/6
        // S(5|2) = (1/6)/(4/6) = 1/4 = 0.25
        assert!(approx_eq(result.conditional_survival, 0.25, TOLERANCE));
    }

    #[test]
    fn test_survival_at_specific_times() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let results = compute_survival_at_times(&time, &status, &[1.0, 3.0, 5.0], 0.95);

        assert_eq!(results.len(), 3);

        // S(1) = 4/5 = 0.8
        assert!(approx_eq(results[0].survival, 0.8, TOLERANCE));

        // S(3) = 2/5 = 0.4
        assert!(approx_eq(results[1].survival, 0.4, TOLERANCE));

        // S(5) = 0
        assert!(approx_eq(results[2].survival, 0.0, TOLERANCE));
    }

    // ==================== LIFE TABLE TESTS ====================

    #[test]
    fn test_life_table_intervals() {
        let time = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let status = vec![1, 1, 1, 1, 1];
        let breaks = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        let result = compute_life_table(&time, &status, &breaks);

        assert_eq!(result.interval_start, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.interval_end, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.n_deaths, vec![1.0, 1.0, 1.0, 1.0, 1.0]);

        // Survival should decrease
        for i in 1..result.survival.len() {
            assert!(result.survival[i] <= result.survival[i - 1]);
        }
    }

    // ==================== CALIBRATION TESTS ====================

    #[test]
    fn test_calibration_curve_basic() {
        // Test basic calibration curve functionality
        let predicted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let observed = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let result = calibration_curve(&predicted, &observed, 5);

        // Should produce groups
        assert!(!result.risk_groups.is_empty());
        assert!(!result.predicted.is_empty());
        assert!(!result.observed.is_empty());

        // Predicted and observed should be between 0 and 1
        for &p in &result.predicted {
            assert!((0.0..=1.0).contains(&p));
        }
        for &o in &result.observed {
            assert!((0.0..=1.0).contains(&o));
        }

        // HL test should produce valid p-value
        assert!((0.0..=1.0).contains(&result.hosmer_lemeshow_pvalue));
    }

    #[test]
    fn test_risk_stratification() {
        let risk_scores = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let events = vec![0, 0, 0, 1, 1, 1];

        let result = stratify_risk(&risk_scores, &events, 2);

        assert_eq!(result.cutpoints.len(), 1);
        assert_eq!(result.group_sizes.len(), 2);

        // High risk group should have higher event rate
        assert!(result.group_event_rates[1] >= result.group_event_rates[0]);
    }

    // ==================== TIME-DEPENDENT AUC TESTS ====================

    #[test]
    fn test_td_auc_perfect_discrimination() {
        // Perfect discrimination
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 0];
        let risk_score = vec![0.9, 0.8, 0.2, 0.1]; // High risk = early event

        let result = time_dependent_auc(&time, &status, &risk_score, &[2.5]);

        // Should have high AUC
        assert!(result.auc[0] > 0.8);
    }

    #[test]
    fn test_td_auc_random_discrimination() {
        // Random risk scores
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 0, 0, 0];
        let risk_score = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // All same

        let result = time_dependent_auc(&time, &status, &risk_score, &[3.5]);

        // Should be around 0.5 (random)
        assert!(approx_eq(result.auc[0], 0.5, 0.1));
    }

    // ==================== EDGE CASE TESTS ====================

    #[test]
    fn test_single_observation() {
        let time = vec![5.0];
        let status = vec![1];

        let na_result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(na_result.cumulative_hazard, vec![1.0]); // 1/1

        let rmst_result = compute_rmst(&time, &status, 10.0, 0.95);
        assert!(rmst_result.rmst > 0.0);
    }

    #[test]
    fn test_tied_event_times() {
        // Multiple events at same time
        let time = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(result.time, vec![1.0, 2.0]);
        assert_eq!(result.n_events, vec![3, 2]);

        // H(1) = 3/5 = 0.6
        assert!(approx_eq(result.cumulative_hazard[0], 0.6, TOLERANCE));
    }

    #[test]
    fn test_very_small_sample() {
        let time = vec![1.0, 2.0];
        let status = vec![1, 1];

        let result = compute_hazard_ratio(&time, &status, &[0, 1], 0.95);

        // Should handle without panic
        assert!(result.hazard_ratio > 0.0);
    }

    // ==================== NUMERICAL STABILITY TESTS ====================

    #[test]
    fn test_large_sample() {
        // Test with larger sample size
        let n = 1000;
        let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert!(!result.cumulative_hazard.is_empty());
        assert!(result.cumulative_hazard.last().unwrap().is_finite());
    }

    #[test]
    fn test_extreme_hazard_ratio() {
        // Very different groups
        let time = vec![0.1, 0.1, 0.1, 100.0, 100.0, 100.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = compute_hazard_ratio(&time, &status, &group, 0.95);

        assert!(result.hazard_ratio.is_finite());
        assert!(result.hazard_ratio > 1.0);
    }
}
