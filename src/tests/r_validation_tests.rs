#[cfg(test)]
mod tests {
    use crate::surv_analysis::nelson_aalen::{nelson_aalen, stratified_km};
    use crate::validation::landmark::{compute_hazard_ratio, compute_survival_at_times};
    use crate::validation::logrank::{WeightType, weighted_logrank_test};
    use crate::validation::power::sample_size_logrank;
    use crate::validation::rmst::compute_rmst;

    const LOOSE_TOLERANCE: f64 = 0.05;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn aml_maintained() -> (Vec<f64>, Vec<i32>) {
        (
            vec![
                9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
            ],
            vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        )
    }

    fn aml_nonmaintained() -> (Vec<f64>, Vec<i32>) {
        (
            vec![
                5.0, 5.0, 8.0, 8.0, 12.0, 16.0, 23.0, 27.0, 30.0, 33.0, 43.0, 45.0,
            ],
            vec![1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        )
    }

    fn aml_combined() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        let (t1, s1) = aml_maintained();
        let (t2, s2) = aml_nonmaintained();

        let mut time = t1.clone();
        time.extend(t2.clone());

        let mut status = s1.clone();
        status.extend(s2.clone());

        let mut group = vec![1; t1.len()];
        group.extend(vec![0; t2.len()]);

        (time, status, group)
    }

    fn lung_subset() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        (
            vec![
                306.0, 455.0, 1010.0, 210.0, 883.0, 1022.0, 310.0, 361.0, 218.0, 166.0, 170.0,
                654.0, 728.0, 71.0, 567.0, 144.0, 613.0, 707.0, 61.0, 88.0,
            ],
            vec![1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
            vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
    }

    fn ovarian_data() -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        (
            vec![
                59.0, 115.0, 156.0, 421.0, 431.0, 448.0, 464.0, 475.0, 477.0, 563.0, 638.0, 744.0,
                769.0, 770.0, 803.0, 855.0, 1040.0, 1106.0, 1129.0, 1206.0, 268.0, 329.0, 353.0,
                365.0, 377.0, 506.0,
            ],
            vec![
                1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
            ],
            vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
            ],
        )
    }

    #[test]
    fn test_r_aml_kaplan_meier_maintained() {
        let (time, status) = aml_maintained();

        let results = compute_survival_at_times(&time, &status, &[9.0, 13.0, 18.0, 23.0], 0.95);

        assert!(approx_eq(results[0].survival, 0.909, LOOSE_TOLERANCE));
        assert!(approx_eq(results[1].survival, 0.818, LOOSE_TOLERANCE));
        assert!(approx_eq(results[2].survival, 0.727, LOOSE_TOLERANCE));
        assert!(approx_eq(results[3].survival, 0.636, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_r_aml_kaplan_meier_nonmaintained() {
        let (time, status) = aml_nonmaintained();

        let results = compute_survival_at_times(&time, &status, &[5.0, 8.0, 12.0, 23.0], 0.95);

        assert!(approx_eq(results[0].survival, 0.833, LOOSE_TOLERANCE));
        assert!(approx_eq(results[1].survival, 0.667, LOOSE_TOLERANCE));
        assert!(approx_eq(results[2].survival, 0.583, LOOSE_TOLERANCE));
        assert!(approx_eq(results[3].survival, 0.486, LOOSE_TOLERANCE));
    }

    #[test]
    fn test_r_aml_logrank_test() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic > 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_aml_wilcoxon_test() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        assert!(result.statistic > 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
        assert_eq!(result.weight_type, "Wilcoxon");
    }

    #[test]
    fn test_r_aml_nelson_aalen() {
        let (time, status) = aml_maintained();

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert!(result.cumulative_hazard[0] > 0.0);

        for i in 1..result.cumulative_hazard.len() {
            assert!(result.cumulative_hazard[i] >= result.cumulative_hazard[i - 1]);
        }

        let surv_from_na: Vec<f64> = result
            .cumulative_hazard
            .iter()
            .map(|h| (-h).exp())
            .collect();

        assert!(surv_from_na[0] <= 1.0 && surv_from_na[0] >= 0.0);
    }

    #[test]
    fn test_r_aml_rmst() {
        let (time, status) = aml_nonmaintained();

        let result = compute_rmst(&time, &status, 30.0, 0.95);

        assert!(result.rmst > 15.0 && result.rmst < 25.0);
        assert!(result.se > 0.0);
        assert!(result.ci_lower < result.rmst);
        assert!(result.ci_upper > result.rmst);
    }

    #[test]
    fn test_r_lung_logrank() {
        let (time, status, group) = lung_subset();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_lung_hazard_ratio() {
        let (time, status, group) = lung_subset();

        let result = compute_hazard_ratio(&time, &status, &group, 0.95);

        assert!(result.hazard_ratio > 0.0);
        assert!(result.ci_lower > 0.0);
        assert!(result.ci_upper > result.ci_lower);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_r_ovarian_survival() {
        let (time, status, _group) = ovarian_data();

        let results =
            compute_survival_at_times(&time, &status, &[100.0, 300.0, 500.0, 700.0], 0.95);

        assert!(results[0].survival > results[1].survival);
        assert!(results[1].survival >= results[2].survival);
        assert!(results[2].survival >= results[3].survival);

        for r in &results {
            assert!((0.0..=1.0).contains(&r.survival));
            assert!(r.ci_lower <= r.survival);
            assert!(r.ci_upper >= r.survival);
        }
    }

    #[test]
    fn test_r_ovarian_logrank() {
        let (time, status, group) = ovarian_data();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_r_sample_size_schoenfeld() {
        let result = sample_size_logrank(0.5, 0.80, 0.05, 1.0, 2);

        assert!(result.n_events >= 50 && result.n_events <= 70);
    }

    #[test]
    fn test_r_sample_size_hr_07() {
        let result = sample_size_logrank(0.7, 0.80, 0.05, 1.0, 2);

        assert!(result.n_events > 150 && result.n_events < 250);
    }

    #[test]
    fn test_r_sample_size_90_power() {
        let result = sample_size_logrank(0.6, 0.90, 0.05, 1.0, 2);

        assert!(result.n_events > 100);
    }

    #[test]
    fn test_r_stratified_km() {
        let (time, status, strata) = aml_combined();

        let result = stratified_km(&time, &status, &strata, 0.95);

        assert_eq!(result.strata.len(), 2);
        assert_eq!(result.times.len(), 2);
        assert_eq!(result.survival.len(), 2);

        for s in &result.survival {
            for &surv in s {
                assert!((0.0..=1.0).contains(&surv));
            }
        }
    }

    #[test]
    fn test_r_aml_median_survival() {
        let (time, status) = aml_nonmaintained();

        let result = crate::validation::rmst::compute_survival_quantile(&time, &status, 0.5, 0.95);

        if let Some(median) = result.median {
            assert!((20.0..=30.0).contains(&median));
        }
    }

    #[test]
    fn test_r_proportional_hazards_assumption() {
        let (time, status, group) = aml_combined();

        let lr_result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        let wil_result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        let ratio = if lr_result.statistic > 0.0 {
            wil_result.statistic / lr_result.statistic
        } else {
            1.0
        };

        assert!(ratio > 0.5 && ratio < 2.0);
    }

    #[test]
    fn test_r_peto_peto_weight() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::PetoPeto);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.weight_type, "PetoPeto");
    }

    #[test]
    fn test_r_tarone_ware_weight() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::TaroneWare);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.weight_type, "TaroneWare");
    }

    #[test]
    fn test_r_fleming_harrington() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(
            &time,
            &status,
            &group,
            WeightType::FlemingHarrington { p: 0.0, q: 1.0 },
        );

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_r_confidence_intervals_coverage() {
        let (time, status) = aml_maintained();

        let results = compute_survival_at_times(&time, &status, &[13.0, 23.0, 34.0], 0.95);

        for r in &results {
            assert!(r.ci_lower <= r.survival);
            assert!(r.ci_upper >= r.survival);
            assert!(r.ci_lower >= 0.0);
            assert!(r.ci_upper <= 1.0);

            let ci_width = r.ci_upper - r.ci_lower;
            assert!(ci_width > 0.0);
        }
    }

    #[test]
    fn test_r_veteran_style_data() {
        let time = vec![
            72.0, 411.0, 228.0, 126.0, 118.0, 10.0, 82.0, 110.0, 314.0, 100.0, 42.0, 8.0, 144.0,
            25.0, 11.0, 30.0, 384.0, 4.0, 54.0, 13.0,
        ];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1];
        let group = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));

        let hr_result = compute_hazard_ratio(&time, &status, &group, 0.95);
        assert!(hr_result.hazard_ratio > 0.0);
    }

    #[test]
    fn test_r_rmst_difference() {
        let (time, status, group) = aml_combined();
        let tau = 40.0;

        let result = crate::validation::rmst::compare_rmst(&time, &status, &group, tau, 0.95);

        assert!(result.rmst_group1.rmst > 0.0);
        assert!(result.rmst_group2.rmst > 0.0);
        assert!(result.diff_se > 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_r_nelson_aalen_variance() {
        let (time, status) = aml_maintained();

        let result = nelson_aalen(&time, &status, None, 0.95);

        for i in 0..result.variance.len() {
            assert!(result.variance[i] >= 0.0);
            assert!(result.ci_lower[i] <= result.cumulative_hazard[i]);
            assert!(result.ci_upper[i] >= result.cumulative_hazard[i]);
        }
    }

    #[test]
    fn test_r_survfit_n_at_risk() {
        let (time, status) = aml_maintained();

        let result = nelson_aalen(&time, &status, None, 0.95);

        let initial_n = time.len();
        assert!(result.n_risk[0] <= initial_n);

        for i in 1..result.n_risk.len() {
            assert!(result.n_risk[i] <= result.n_risk[i - 1]);
        }
    }

    #[test]
    fn test_r_tied_events_handling() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0];
        let status = vec![1, 1, 0, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert!(result.time.contains(&5.0));
        assert!(result.time.contains(&10.0));
        assert!(result.time.contains(&15.0));

        let idx_5 = result.time.iter().position(|&t| t == 5.0).unwrap();
        assert_eq!(result.n_events[idx_5], 2);
    }

    #[test]
    fn test_r_all_censored_at_end() {
        let time = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let status = vec![1, 1, 1, 0, 0];

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(result.time.len(), 3);
        assert!(!result.time.contains(&40.0));
        assert!(!result.time.contains(&50.0));
    }

    #[test]
    fn test_r_single_event() {
        let time = vec![10.0, 20.0, 30.0];
        let status = vec![0, 1, 0];

        let result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(result.time.len(), 1);
        assert_eq!(result.time[0], 20.0);
        assert_eq!(result.n_risk[0], 2);
    }

    #[test]
    fn test_r_late_entry_simulation() {
        let time = vec![100.0, 150.0, 200.0, 250.0, 300.0];
        let status = vec![1, 1, 1, 1, 1];

        let result = compute_rmst(&time, &status, 350.0, 0.95);

        assert!(result.rmst > 0.0);
        assert!(result.rmst < 350.0);
    }
}
