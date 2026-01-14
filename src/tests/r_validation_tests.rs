#[cfg(test)]
mod tests {
    use crate::surv_analysis::nelson_aalen::{nelson_aalen, stratified_km};
    use crate::tests::common::{
        LOOSE_TOL, STANDARD_TOL, STRICT_TOL, aml_combined, aml_maintained, aml_nonmaintained,
        approx_eq, lung_subset, ovarian_data,
    };
    use crate::validation::landmark::{compute_hazard_ratio, compute_survival_at_times};
    use crate::validation::logrank::{WeightType, weighted_logrank_test};
    use crate::validation::power::sample_size_logrank;
    use crate::validation::rmst::compute_rmst;

    #[test]
    fn test_r_aml_kaplan_meier_maintained() {
        let (time, status) = aml_maintained();
        let results = compute_survival_at_times(&time, &status, &[9.0, 13.0, 18.0, 23.0], 0.95);

        assert!(approx_eq(results[0].survival, 0.90909090909, STANDARD_TOL));
        assert!(approx_eq(results[1].survival, 0.81818181818, STANDARD_TOL));
        assert!(approx_eq(results[2].survival, 0.71590909091, STANDARD_TOL));
        assert!(approx_eq(results[3].survival, 0.61363636364, STANDARD_TOL));
    }

    #[test]
    fn test_r_aml_kaplan_meier_nonmaintained() {
        let (time, status) = aml_nonmaintained();
        let results = compute_survival_at_times(&time, &status, &[5.0, 8.0, 12.0, 23.0], 0.95);

        assert!(approx_eq(results[0].survival, 0.8333333, STANDARD_TOL));
        assert!(approx_eq(results[1].survival, 0.6666667, STANDARD_TOL));
        assert!(approx_eq(results[2].survival, 0.5833333, STANDARD_TOL));
        assert!(approx_eq(results[3].survival, 0.4861111, STANDARD_TOL));
    }

    #[test]
    fn test_r_aml_logrank_test() {
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert_eq!(result.df, 1);
        assert!(approx_eq(result.statistic, 3.4, 0.5));
        assert!(approx_eq(result.p_value, 0.0653, LOOSE_TOL));
    }

    #[test]
    fn test_r_aml_wilcoxon_test() {
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        assert!(result.statistic > 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
        assert_eq!(result.weight_type, "Wilcoxon");
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_aml_nelson_aalen() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

        for i in 1..result.cumulative_hazard.len() {
            assert!(result.cumulative_hazard[i] >= result.cumulative_hazard[i - 1]);
        }

        assert!(approx_eq(
            result.cumulative_hazard[0],
            1.0 / 11.0,
            STRICT_TOL
        ));

        let surv_from_na: Vec<f64> = result
            .cumulative_hazard
            .iter()
            .map(|h| (-h).exp())
            .collect();

        for s in &surv_from_na {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    #[test]
    fn test_r_aml_rmst() {
        let (time, status) = aml_nonmaintained();
        let result = compute_rmst(&time, &status, 30.0, 0.95);

        assert!(result.rmst > 15.0 && result.rmst < 25.0);
        assert!(result.se > 0.0);
        assert!(result.ci_lower < result.rmst);
        assert!(result.ci_upper > result.rmst);

        assert!(result.ci_lower > 0.0);
        assert!(result.ci_upper < 30.0);
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
            assert!(r.ci_lower >= 0.0);
            assert!(r.ci_upper <= 1.0);
        }
    }

    #[test]
    fn test_r_ovarian_logrank() {
        let (time, status, group) = ovarian_data();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.df, 1);
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
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_tarone_ware_weight() {
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::TaroneWare);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.weight_type, "TaroneWare");
        assert_eq!(result.df, 1);
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
        assert_eq!(result.df, 1);
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
        assert_eq!(result.df, 1);

        let hr_result = compute_hazard_ratio(&time, &status, &group, 0.95);
        assert!(hr_result.hazard_ratio > 0.0);
        assert!(hr_result.ci_lower > 0.0);
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

    #[test]
    fn test_r_aml_km_extended() {
        let (time, status) = aml_maintained();
        let results = compute_survival_at_times(
            &time,
            &status,
            &[9.0, 13.0, 18.0, 23.0, 31.0, 34.0, 48.0],
            0.95,
        );

        let expected = [0.9091, 0.8182, 0.7159, 0.6136, 0.4909, 0.3682, 0.1841];

        for (i, &exp) in expected.iter().enumerate() {
            if i < results.len() {
                assert!(
                    approx_eq(results[i].survival, exp, STANDARD_TOL),
                    "Mismatch at time index {}: expected {}, got {}",
                    i,
                    exp,
                    results[i].survival
                );
            }
        }
    }

    #[test]
    fn test_r_exact_nelson_aalen_values() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

        if !result.cumulative_hazard.is_empty() {
            assert!(approx_eq(
                result.cumulative_hazard[0],
                1.0 / 11.0,
                STRICT_TOL
            ));
        }
        if result.cumulative_hazard.len() > 1 {
            assert!(approx_eq(
                result.cumulative_hazard[1],
                1.0 / 11.0 + 1.0 / 10.0,
                STRICT_TOL
            ));
        }
    }

    #[test]
    fn test_r_survdiff_exact_chisq() {
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(
            result.statistic > 2.5 && result.statistic < 4.5,
            "Chi-squared {} not in expected range [2.5, 4.5]",
            result.statistic
        );

        assert!(
            result.p_value > 0.04 && result.p_value < 0.15,
            "P-value {} not in expected range [0.04, 0.15]",
            result.p_value
        );
    }
}
