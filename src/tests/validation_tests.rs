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
    #[test]
    fn test_nelson_aalen_simple() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(result.time.len(), 5);
        assert_eq!(result.n_events, vec![1, 1, 1, 1, 1]);
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
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 0, 1, 0, 1, 0];
        let result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(result.time, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.n_events, vec![1, 1, 1]);
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
        assert!(result.time.is_empty());
    }
    #[test]
    fn test_stratified_km_two_groups() {
        let time = vec![1.0, 2.0, 3.0, 1.0, 3.0, 5.0];
        let status = vec![1, 1, 0, 1, 1, 1];
        let strata = vec![0, 0, 0, 1, 1, 1];
        let result = stratified_km(&time, &status, &strata, 0.95);
        assert_eq!(result.strata, vec![0, 1]);
        assert_eq!(result.times.len(), 2);
        assert!(approx_eq(result.survival[0][0], 2.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[0][1], 1.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[1][0], 2.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[1][1], 1.0 / 3.0, TOLERANCE));
        assert!(approx_eq(result.survival[1][2], 0.0, TOLERANCE));
    }
    #[test]
    fn test_rmst_simple() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = compute_rmst(&time, &status, 5.0, 0.95);
        assert!(result.rmst > 0.0);
        assert!(result.rmst < 5.0);
        assert!(result.se > 0.0);
        assert!(result.ci_lower < result.rmst);
        assert!(result.ci_upper > result.rmst);
    }
    #[test]
    fn test_rmst_no_events_before_tau() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![0, 0, 0];
        let result = compute_rmst(&time, &status, 10.0, 0.95);
        assert!(approx_eq(result.rmst, 10.0, TOLERANCE));
    }
    #[test]
    fn test_rmst_comparison() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let result = compare_rmst(&time, &status, &group, 5.0, 0.95);
        assert!(result.rmst_group2.rmst > result.rmst_group1.rmst);
        assert!(result.rmst_diff < 0.0);
    }
    #[test]
    fn test_survival_quantile_median() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);
        assert!(result.median.is_some());
        assert!(approx_eq(result.median.unwrap(), 3.0, TOLERANCE));
    }
    #[test]
    fn test_survival_quantile_not_reached() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 0];
        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);
        assert!(result.median.is_none());
    }
    #[test]
    fn test_cumulative_incidence_single_event_type() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = compute_cumulative_incidence(&time, &status);
        assert_eq!(result.event_types, vec![1]);
        assert_eq!(result.cif.len(), 1);
        let last_cif = result.cif[0].last().unwrap();
        assert!(approx_eq(*last_cif, 1.0, TOLERANCE));
    }
    #[test]
    fn test_cumulative_incidence_competing_risks() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 2, 1, 2, 1, 0];
        let result = compute_cumulative_incidence(&time, &status);
        assert_eq!(result.event_types.len(), 2);
        assert!(result.event_types.contains(&1));
        assert!(result.event_types.contains(&2));
        let last_idx = result.time.len() - 1;
        let sum_cif: f64 = result.cif.iter().map(|c| c[last_idx]).sum();
        assert!(sum_cif <= 1.0 + TOLERANCE);
    }
    #[test]
    fn test_logrank_identical_groups() {
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        assert!(result.statistic < 1.0);
        assert!(result.p_value > 0.3);
    }
    #[test]
    fn test_logrank_different_groups() {
        let time = vec![1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        assert!(result.statistic > 3.0);
        assert!(result.p_value < 0.1);
    }
    #[test]
    fn test_logrank_wilcoxon_weight() {
        let time = vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let lr_result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        let wilcox_result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);
        assert!(lr_result.statistic >= 0.0);
        assert!(wilcox_result.statistic >= 0.0);
        assert_eq!(wilcox_result.weight_type, "Wilcoxon");
    }
    #[test]
    fn test_logrank_trend() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let result = logrank_trend_test(&time, &status, &group, None);
        assert!(result.trend_direction == "increasing" || result.trend_direction == "decreasing");
        assert!((0.0..=1.0).contains(&result.p_value));
    }
    #[test]
    fn test_hazard_ratio_equal_groups() {
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];
        let result = compute_hazard_ratio(&time, &status, &group, 0.95);
        assert!(approx_eq(result.hazard_ratio, 1.0, 0.5));
        assert!(result.ci_lower < 1.0 && result.ci_upper > 1.0);
    }
    #[test]
    fn test_hazard_ratio_higher_risk() {
        let time = vec![1.0, 1.0, 1.0, 5.0, 5.0, 5.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];
        let result = compute_hazard_ratio(&time, &status, &group, 0.95);
        assert!(result.hazard_ratio > 1.0);
    }
    #[test]
    fn test_sample_size_schoenfeld() {
        let result = sample_size_logrank(0.5, 0.8, 0.05, 1.0, 2);
        assert!(result.n_events >= 50 && result.n_events <= 80);
        assert!(result.n_per_group.len() == 2);
    }
    #[test]
    fn test_sample_size_smaller_effect() {
        let result_large = sample_size_logrank(0.5, 0.8, 0.05, 1.0, 2);
        let result_small = sample_size_logrank(0.7, 0.8, 0.05, 1.0, 2);
        assert!(result_small.n_events > result_large.n_events);
    }
    #[test]
    fn test_sample_size_higher_power() {
        let result_80 = sample_size_logrank(0.6, 0.8, 0.05, 1.0, 2);
        let result_90 = sample_size_logrank(0.6, 0.9, 0.05, 1.0, 2);
        assert!(result_90.n_events > result_80.n_events);
    }
    #[test]
    fn test_sample_size_freedman() {
        let result = sample_size_freedman(0.6, 0.8, 0.05, 0.3, 1.0, 2);
        assert!(result.n_total > 0);
        assert!(result.n_events > 0);
        assert_eq!(result.method, "Freedman");
    }
    #[test]
    fn test_power_calculation() {
        let power = power_logrank(100, 0.6, 0.05, 1.0, 2);
        assert!(power > 0.0 && power < 1.0);
        let power_more = power_logrank(200, 0.6, 0.05, 1.0, 2);
        assert!(power_more > power);
    }
    #[test]
    fn test_power_approaches_one() {
        let power = power_logrank(1000, 0.5, 0.05, 1.0, 2);
        assert!(power > 0.99);
    }
    #[test]
    fn test_landmark_excludes_early_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = compute_landmark(&time, &status, 2.5);
        assert_eq!(result.n_excluded, 2);
        assert_eq!(result.n_at_risk, 3);
        assert_eq!(result.time, vec![0.5, 1.5, 2.5]);
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
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let result = compute_conditional_survival(&time, &status, 2.0, 5.0, 0.95);
        assert!(approx_eq(result.conditional_survival, 0.25, TOLERANCE));
    }
    #[test]
    fn test_survival_at_specific_times() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let results = compute_survival_at_times(&time, &status, &[1.0, 3.0, 5.0], 0.95);
        assert_eq!(results.len(), 3);
        assert!(approx_eq(results[0].survival, 0.8, TOLERANCE));
        assert!(approx_eq(results[1].survival, 0.4, TOLERANCE));
        assert!(approx_eq(results[2].survival, 0.0, TOLERANCE));
    }
    #[test]
    fn test_life_table_intervals() {
        let time = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let status = vec![1, 1, 1, 1, 1];
        let breaks = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_life_table(&time, &status, &breaks);
        assert_eq!(result.interval_start, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.interval_end, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.n_deaths, vec![1.0, 1.0, 1.0, 1.0, 1.0]);
        for i in 1..result.survival.len() {
            assert!(result.survival[i] <= result.survival[i - 1]);
        }
    }
    #[test]
    fn test_calibration_curve_basic() {
        let predicted = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let observed = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let result = calibration_curve(&predicted, &observed, 5);
        assert!(!result.risk_groups.is_empty());
        assert!(!result.predicted.is_empty());
        assert!(!result.observed.is_empty());
        for &p in &result.predicted {
            assert!((0.0..=1.0).contains(&p));
        }
        for &o in &result.observed {
            assert!((0.0..=1.0).contains(&o));
        }
        assert!((0.0..=1.0).contains(&result.hosmer_lemeshow_pvalue));
    }
    #[test]
    fn test_risk_stratification() {
        let risk_scores = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let events = vec![0, 0, 0, 1, 1, 1];
        let result = stratify_risk(&risk_scores, &events, 2);
        assert_eq!(result.cutpoints.len(), 1);
        assert_eq!(result.group_sizes.len(), 2);
        assert!(result.group_event_rates[1] >= result.group_event_rates[0]);
    }
    #[test]
    fn test_td_auc_perfect_discrimination() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 0];
        let risk_score = vec![0.9, 0.8, 0.2, 0.1];
        let result = time_dependent_auc(&time, &status, &risk_score, &[2.5]);
        assert!(result.auc[0] > 0.8);
    }
    #[test]
    fn test_td_auc_random_discrimination() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 0, 0, 0];
        let risk_score = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let result = time_dependent_auc(&time, &status, &risk_score, &[3.5]);
        assert!(approx_eq(result.auc[0], 0.5, 0.1));
    }
    #[test]
    fn test_single_observation() {
        let time = vec![5.0];
        let status = vec![1];
        let na_result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(na_result.cumulative_hazard, vec![1.0]);
        let rmst_result = compute_rmst(&time, &status, 10.0, 0.95);
        assert!(rmst_result.rmst > 0.0);
    }
    #[test]
    fn test_tied_event_times() {
        let time = vec![1.0, 1.0, 1.0, 2.0, 2.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(result.time, vec![1.0, 2.0]);
        assert_eq!(result.n_events, vec![3, 2]);
        assert!(approx_eq(result.cumulative_hazard[0], 0.6, TOLERANCE));
    }
    #[test]
    fn test_very_small_sample() {
        let time = vec![1.0, 2.0];
        let status = vec![1, 1];
        let result = compute_hazard_ratio(&time, &status, &[0, 1], 0.95);
        assert!(result.hazard_ratio > 0.0);
    }
    #[test]
    fn test_large_sample() {
        let n = 1000;
        let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();
        let result = nelson_aalen(&time, &status, None, 0.95);
        assert!(!result.cumulative_hazard.is_empty());
        assert!(result.cumulative_hazard.last().unwrap().is_finite());
    }
    #[test]
    fn test_extreme_hazard_ratio() {
        let time = vec![0.1, 0.1, 0.1, 100.0, 100.0, 100.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];
        let result = compute_hazard_ratio(&time, &status, &group, 0.95);
        assert!(result.hazard_ratio.is_finite());
        assert!(result.hazard_ratio > 1.0);
    }
}
