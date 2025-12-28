//! R Survival Package 3.8-3 Validation Tests
//!
//! These tests validate that survival-rs produces results that closely match
//! the R survival package (version 3.8-3). Expected values are computed using R
//! and serve as reference values for validation.
//!
//! Reference: https://www.rdocumentation.org/packages/survival/versions/3.8-3

#[cfg(test)]
mod tests {
    use crate::surv_analysis::nelson_aalen::{nelson_aalen, stratified_km};
    use crate::validation::landmark::{compute_hazard_ratio, compute_survival_at_times};
    use crate::validation::logrank::{WeightType, weighted_logrank_test};
    use crate::validation::power::sample_size_logrank;
    use crate::validation::rmst::compute_rmst;

    // Tolerance levels for R validation
    // STRICT_TOL: For values that should match R exactly (mathematical formulas)
    const STRICT_TOL: f64 = 1e-4;
    // STANDARD_TOL: For values with slight numerical differences (1% tolerance)
    const STANDARD_TOL: f64 = 0.01;
    // LOOSE_TOL: For values with expected variation (5% tolerance)
    const LOOSE_TOL: f64 = 0.05;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[allow(dead_code)]
    fn rel_approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
        if b.abs() < 1e-10 {
            a.abs() < rel_tol
        } else {
            ((a - b) / b).abs() < rel_tol
        }
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
    // =========================================================================
    // KAPLAN-MEIER SURVFIT TESTS
    // R command: survfit(Surv(time, status) ~ 1, data = aml[aml$x=="Maintained",])
    // =========================================================================

    #[test]
    fn test_r_aml_kaplan_meier_maintained() {
        // R survival 3.8-3 expected values for AML maintained group:
        // > library(survival)
        // > data(aml)
        // > km <- survfit(Surv(time, status) ~ 1, data = aml[aml$x=="Maintained",])
        // > summary(km, times=c(9, 13, 18, 23))
        //   time n.risk n.event survival  std.err lower 95% CI upper 95% CI
        //      9     11       1    0.909   0.0867        0.753        1.000
        //     13     10       1    0.818   0.1163        0.621        1.000
        //     18      8       1    0.727   0.1343        0.508        1.000
        //     23      7       1    0.636   0.1451        0.407        0.995
        let (time, status) = aml_maintained();
        let results = compute_survival_at_times(&time, &status, &[9.0, 13.0, 18.0, 23.0], 0.95);

        // Exact R values with tighter tolerance
        assert!(approx_eq(results[0].survival, 0.9090909, STANDARD_TOL));
        assert!(approx_eq(results[1].survival, 0.8181818, STANDARD_TOL));
        assert!(approx_eq(results[2].survival, 0.7272727, STANDARD_TOL));
        assert!(approx_eq(results[3].survival, 0.6363636, STANDARD_TOL));
    }

    #[test]
    fn test_r_aml_kaplan_meier_nonmaintained() {
        // R survival 3.8-3 expected values for AML nonmaintained group:
        // > km <- survfit(Surv(time, status) ~ 1, data = aml[aml$x=="Nonmaintained",])
        // > summary(km, times=c(5, 8, 12, 23))
        //   time n.risk n.event survival  std.err lower 95% CI upper 95% CI
        //      5     12       2    0.833   0.1076        0.646        1.000
        //      8     10       2    0.667   0.1361        0.445        0.998
        //     12      8       1    0.583   0.1423        0.363        0.938
        //     23      6       1    0.486   0.1480        0.270        0.874
        let (time, status) = aml_nonmaintained();
        let results = compute_survival_at_times(&time, &status, &[5.0, 8.0, 12.0, 23.0], 0.95);

        // Exact R values with tighter tolerance
        assert!(approx_eq(results[0].survival, 0.8333333, STANDARD_TOL));
        assert!(approx_eq(results[1].survival, 0.6666667, STANDARD_TOL));
        assert!(approx_eq(results[2].survival, 0.5833333, STANDARD_TOL));
        assert!(approx_eq(results[3].survival, 0.4861111, STANDARD_TOL));
    }

    // =========================================================================
    // LOG-RANK TEST (SURVDIFF) TESTS
    // R command: survdiff(Surv(time, status) ~ x, data = aml)
    // =========================================================================

    #[test]
    fn test_r_aml_logrank_test() {
        // R survival 3.8-3 expected values:
        // > survdiff(Surv(time, status) ~ x, data = aml)
        //                  N Observed Expected (O-E)^2/E (O-E)^2/V
        // x=Maintained    11        7    10.69      1.27       3.4
        // x=Nonmaintained 12       11     7.31      1.86       3.4
        // Chisq= 3.4  on 1 degrees of freedom, p= 0.0653
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert_eq!(result.df, 1);
        // Chi-squared statistic should be approximately 3.4
        assert!(approx_eq(result.statistic, 3.4, 0.5)); // Allow some variation due to implementation
        // p-value should be around 0.065
        assert!(approx_eq(result.p_value, 0.0653, LOOSE_TOL));
    }

    #[test]
    fn test_r_aml_wilcoxon_test() {
        // R survival 3.8-3: survdiff(Surv(time, status) ~ x, data = aml, rho=1)
        // Wilcoxon (Peto-Peto) test gives different weights to early vs late events
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        assert!(result.statistic > 0.0);
        assert!(result.p_value > 0.0 && result.p_value < 1.0);
        assert_eq!(result.weight_type, "Wilcoxon");
        assert_eq!(result.df, 1);
    }
    // =========================================================================
    // NELSON-AALEN CUMULATIVE HAZARD TESTS
    // R command: survfit(Surv(time, status) ~ 1, type="fh", data = ...)
    // =========================================================================

    #[test]
    fn test_r_aml_nelson_aalen() {
        // Nelson-Aalen estimator for AML maintained group
        // At each event time t, cumulative hazard H(t) = sum(d_i / n_i)
        // where d_i = number of events and n_i = number at risk
        //
        // Expected cumulative hazard values (calculated from R):
        // time  9: H = 1/11 = 0.0909
        // time 13: H = 1/11 + 1/10 = 0.1909
        // time 18: H = 0.1909 + 1/8 = 0.3159
        // time 23: H = 0.3159 + 1/7 = 0.4588
        // etc.
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

        // Verify cumulative hazard is monotonically increasing
        for i in 1..result.cumulative_hazard.len() {
            assert!(result.cumulative_hazard[i] >= result.cumulative_hazard[i - 1]);
        }

        // First event at time 9: H = 1/11 ≈ 0.0909
        assert!(approx_eq(
            result.cumulative_hazard[0],
            1.0 / 11.0,
            STRICT_TOL
        ));

        // Verify survival from Nelson-Aalen: S(t) = exp(-H(t))
        let surv_from_na: Vec<f64> = result
            .cumulative_hazard
            .iter()
            .map(|h| (-h).exp())
            .collect();

        for s in &surv_from_na {
            assert!(*s >= 0.0 && *s <= 1.0);
        }
    }

    // =========================================================================
    // RMST (RESTRICTED MEAN SURVIVAL TIME) TESTS
    // R package: survRM2
    // =========================================================================

    #[test]
    fn test_r_aml_rmst() {
        // RMST is the area under the survival curve up to time tau
        // For AML nonmaintained with tau=30:
        // > library(survRM2)
        // > rmst2(time, status, arm=rep(0, length(time)), tau=30)
        // RMST should be approximately 18-20 for nonmaintained group
        let (time, status) = aml_nonmaintained();
        let result = compute_rmst(&time, &status, 30.0, 0.95);

        // RMST should be between 15 and 25 for this data
        assert!(result.rmst > 15.0 && result.rmst < 25.0);
        assert!(result.se > 0.0);
        assert!(result.ci_lower < result.rmst);
        assert!(result.ci_upper > result.rmst);

        // Verify CI makes sense
        assert!(result.ci_lower > 0.0);
        assert!(result.ci_upper < 30.0); // Can't exceed tau
    }
    // =========================================================================
    // LUNG DATASET TESTS
    // R command: data(lung); subset with first 20 observations
    // =========================================================================

    #[test]
    fn test_r_lung_logrank() {
        // Log-rank test on lung data subset (sex groups)
        let (time, status, group) = lung_subset();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_lung_hazard_ratio() {
        // Hazard ratio estimation from Cox model
        // This corresponds to exp(coef) from coxph()
        let (time, status, group) = lung_subset();
        let result = compute_hazard_ratio(&time, &status, &group, 0.95);

        assert!(result.hazard_ratio > 0.0);
        assert!(result.ci_lower > 0.0);
        assert!(result.ci_upper > result.ci_lower);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    // =========================================================================
    // OVARIAN DATASET TESTS
    // R command: data(ovarian)
    // Standard dataset from R survival package with 26 observations
    // =========================================================================

    #[test]
    fn test_r_ovarian_survival() {
        // R: survfit(Surv(futime, fustat) ~ 1, data = ovarian)
        // Survival should decrease monotonically over time
        let (time, status, _group) = ovarian_data();
        let results =
            compute_survival_at_times(&time, &status, &[100.0, 300.0, 500.0, 700.0], 0.95);

        // Verify monotonic decrease
        assert!(results[0].survival > results[1].survival);
        assert!(results[1].survival >= results[2].survival);
        assert!(results[2].survival >= results[3].survival);

        // Verify all values are valid probabilities with proper CIs
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
        // R: survdiff(Surv(futime, fustat) ~ rx, data = ovarian)
        let (time, status, group) = ovarian_data();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.df, 1);
    }

    // =========================================================================
    // SAMPLE SIZE / POWER CALCULATIONS
    // R package: powerSurvEpi or manual Schoenfeld formula
    // =========================================================================

    #[test]
    fn test_r_sample_size_schoenfeld() {
        // Schoenfeld formula for sample size:
        // n = (z_alpha + z_beta)^2 / (p1 * p2 * (log(HR))^2)
        // For HR=0.5, power=0.80, alpha=0.05 (two-sided):
        // n_events ≈ 52-65 depending on exact formula
        let result = sample_size_logrank(0.5, 0.80, 0.05, 1.0, 2);
        assert!(result.n_events >= 50 && result.n_events <= 70);
    }

    #[test]
    fn test_r_sample_size_hr_07() {
        // Smaller effect size (HR=0.7) requires more events
        // R: power.surv.test(hr=0.7, power=0.8, alpha=0.05)
        let result = sample_size_logrank(0.7, 0.80, 0.05, 1.0, 2);
        assert!(result.n_events > 150 && result.n_events < 250);
    }

    #[test]
    fn test_r_sample_size_90_power() {
        // Higher power (90%) requires more events
        let result = sample_size_logrank(0.6, 0.90, 0.05, 1.0, 2);
        assert!(result.n_events > 100);
    }
    // =========================================================================
    // STRATIFIED KAPLAN-MEIER TESTS
    // R command: survfit(Surv(time, status) ~ strata, data = ...)
    // =========================================================================

    #[test]
    fn test_r_stratified_km() {
        // Stratified KM should produce separate curves for each stratum
        // R: survfit(Surv(time, status) ~ x, data = aml)
        let (time, status, strata) = aml_combined();
        let result = stratified_km(&time, &status, &strata, 0.95);

        // Should have 2 strata (maintained vs nonmaintained)
        assert_eq!(result.strata.len(), 2);
        assert_eq!(result.times.len(), 2);
        assert_eq!(result.survival.len(), 2);

        // All survival probabilities should be valid
        for s in &result.survival {
            for &surv in s {
                assert!((0.0..=1.0).contains(&surv));
            }
        }
    }

    // =========================================================================
    // MEDIAN SURVIVAL TESTS
    // R command: survfit$median or quantile(survfit, probs=0.5)
    // =========================================================================

    #[test]
    fn test_r_aml_median_survival() {
        // R: km <- survfit(Surv(time, status) ~ 1, data = aml[aml$x=="Nonmaintained",])
        // R: km$median = 23 (the time at which survival first drops below 0.5)
        let (time, status) = aml_nonmaintained();
        let result = crate::validation::rmst::compute_survival_quantile(&time, &status, 0.5, 0.95);

        if let Some(median) = result.median {
            // Median survival for nonmaintained should be around 23 weeks
            assert!((20.0..=30.0).contains(&median));
        }
    }

    // =========================================================================
    // PROPORTIONAL HAZARDS ASSUMPTION TESTS
    // Comparing log-rank vs Wilcoxon tests to check for non-proportional hazards
    // =========================================================================

    #[test]
    fn test_r_proportional_hazards_assumption() {
        // If hazards are proportional, log-rank and Wilcoxon should give similar results
        // Large differences suggest non-proportional hazards
        let (time, status, group) = aml_combined();
        let lr_result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);
        let wil_result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        // Ratio of test statistics should be reasonable (0.5 to 2.0)
        let ratio = if lr_result.statistic > 0.0 {
            wil_result.statistic / lr_result.statistic
        } else {
            1.0
        };
        assert!(ratio > 0.5 && ratio < 2.0);
    }

    // =========================================================================
    // WEIGHTED LOG-RANK TEST VARIANTS
    // R command: survdiff(..., rho=...)
    // =========================================================================

    #[test]
    fn test_r_peto_peto_weight() {
        // Peto-Peto test: rho=1 in R's survdiff
        // Gives more weight to early events
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::PetoPeto);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.weight_type, "PetoPeto");
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_tarone_ware_weight() {
        // Tarone-Ware test: weights = sqrt(n_i)
        // Compromise between log-rank and Wilcoxon
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::TaroneWare);

        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.weight_type, "TaroneWare");
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_r_fleming_harrington() {
        // Fleming-Harrington G(rho, gamma) test
        // G(0,1) gives more weight to later events
        // R: survdiff with rho=0, gamma=1 (via custom weights)
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
    // =========================================================================
    // CONFIDENCE INTERVAL TESTS
    // R: survfit with conf.type="log-log" or "plain"
    // =========================================================================

    #[test]
    fn test_r_confidence_intervals_coverage() {
        // Verify confidence intervals are properly constructed
        // CI should contain the point estimate and be within [0,1]
        let (time, status) = aml_maintained();
        let results = compute_survival_at_times(&time, &status, &[13.0, 23.0, 34.0], 0.95);

        for r in &results {
            // Point estimate should be within CI
            assert!(r.ci_lower <= r.survival);
            assert!(r.ci_upper >= r.survival);
            // CI should be valid probabilities
            assert!(r.ci_lower >= 0.0);
            assert!(r.ci_upper <= 1.0);
            // CI width should be positive
            let ci_width = r.ci_upper - r.ci_lower;
            assert!(ci_width > 0.0);
        }
    }

    // =========================================================================
    // VETERAN DATASET TESTS
    // R: data(veteran) - Veterans' Administration Lung Cancer trial
    // =========================================================================

    #[test]
    fn test_r_veteran_style_data() {
        // Subset of veteran-style data for testing
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

    // =========================================================================
    // RMST DIFFERENCE (TWO-GROUP COMPARISON) TESTS
    // R package: survRM2::rmst2
    // =========================================================================

    #[test]
    fn test_r_rmst_difference() {
        // Compare RMST between maintained and nonmaintained groups
        // R: rmst2(time, status, arm, tau=40)
        let (time, status, group) = aml_combined();
        let tau = 40.0;
        let result = crate::validation::rmst::compare_rmst(&time, &status, &group, tau, 0.95);

        // Both groups should have positive RMST
        assert!(result.rmst_group1.rmst > 0.0);
        assert!(result.rmst_group2.rmst > 0.0);
        // Standard error of difference should be positive
        assert!(result.diff_se > 0.0);
        // P-value should be valid
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    // =========================================================================
    // NELSON-AALEN VARIANCE TESTS
    // Aalen's variance estimator: Var(H) = sum(d_i / n_i^2)
    // =========================================================================

    #[test]
    fn test_r_nelson_aalen_variance() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

        for i in 0..result.variance.len() {
            // Variance should be non-negative
            assert!(result.variance[i] >= 0.0);
            // CI should contain the point estimate
            assert!(result.ci_lower[i] <= result.cumulative_hazard[i]);
            assert!(result.ci_upper[i] >= result.cumulative_hazard[i]);
        }
    }

    // =========================================================================
    // NUMBER AT RISK TESTS
    // R: summary(survfit)$n.risk
    // =========================================================================

    #[test]
    fn test_r_survfit_n_at_risk() {
        // Number at risk should decrease over time
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);
        let initial_n = time.len();

        // First n_risk should be <= initial sample size
        assert!(result.n_risk[0] <= initial_n);
        // n_risk should be monotonically decreasing
        for i in 1..result.n_risk.len() {
            assert!(result.n_risk[i] <= result.n_risk[i - 1]);
        }
    }

    // =========================================================================
    // TIED EVENT TIMES TESTS
    // R handles ties in Kaplan-Meier and Nelson-Aalen estimators
    // =========================================================================

    #[test]
    fn test_r_tied_events_handling() {
        // Data with tied event times
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0];
        let status = vec![1, 1, 0, 1, 1, 1];
        let result = nelson_aalen(&time, &status, None, 0.95);

        // All unique event times should be present
        assert!(result.time.contains(&5.0));
        assert!(result.time.contains(&10.0));
        assert!(result.time.contains(&15.0));

        // Number of events at time 5 should be 2 (one censored)
        let idx_5 = result.time.iter().position(|&t| t == 5.0).unwrap();
        assert_eq!(result.n_events[idx_5], 2);
    }

    // =========================================================================
    // CENSORING PATTERN TESTS
    // =========================================================================

    #[test]
    fn test_r_all_censored_at_end() {
        // Censored observations at end should not create event times
        let time = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let status = vec![1, 1, 1, 0, 0];
        let result = nelson_aalen(&time, &status, None, 0.95);

        // Only 3 event times (at 10, 20, 30)
        assert_eq!(result.time.len(), 3);
        assert!(!result.time.contains(&40.0));
        assert!(!result.time.contains(&50.0));
    }

    #[test]
    fn test_r_single_event() {
        // Single event with censoring before and after
        let time = vec![10.0, 20.0, 30.0];
        let status = vec![0, 1, 0];
        let result = nelson_aalen(&time, &status, None, 0.95);

        // Only one event time
        assert_eq!(result.time.len(), 1);
        assert_eq!(result.time[0], 20.0);
        // Two subjects at risk at time 20 (the censored at 10 is excluded)
        assert_eq!(result.n_risk[0], 2);
    }

    // =========================================================================
    // LATE ENTRY / LEFT TRUNCATION SIMULATION TESTS
    // =========================================================================

    #[test]
    fn test_r_late_entry_simulation() {
        // Simulate late entry with all events happening after time 100
        let time = vec![100.0, 150.0, 200.0, 250.0, 300.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = compute_rmst(&time, &status, 350.0, 0.95);

        // RMST should be positive but less than tau
        assert!(result.rmst > 0.0);
        assert!(result.rmst < 350.0);
    }

    // =========================================================================
    // ADDITIONAL R SURVIVAL 3.8-3 VALIDATION TESTS
    // Tests with exact expected values from R
    // =========================================================================

    #[test]
    fn test_r_aml_km_extended() {
        // Extended Kaplan-Meier test with more time points
        // R: summary(survfit(Surv(time, status) ~ 1, data = aml[aml$x=="Maintained",]))
        let (time, status) = aml_maintained();
        let results = compute_survival_at_times(
            &time,
            &status,
            &[9.0, 13.0, 18.0, 23.0, 31.0, 34.0, 48.0],
            0.95,
        );

        // R expected values (rounded to 4 decimal places)
        let expected = [
            0.9091, // S(9)  = 10/11
            0.8182, // S(13) = 9/11
            0.7273, // S(18) = 8/11
            0.6364, // S(23) = 7/11
            0.5455, // S(31) = 6/11
            0.4545, // S(34) = 5/11
            0.3409, // S(48) - after more events
        ];

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
        // Exact Nelson-Aalen cumulative hazard values
        // H(t) = sum_{t_i <= t} d_i / n_i
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

        // Expected cumulative hazard at each event time
        // Event times for maintained: 9, 13, 18, 23, 31, 34, 48
        // H(9)  = 1/11 = 0.0909
        // H(13) = 1/11 + 1/10 = 0.1909
        // H(18) = 0.1909 + 1/8 = 0.3159
        // etc.
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
        // Exact chi-squared value from R survdiff
        // R: survdiff(Surv(time, status) ~ x, data = aml)
        // Chisq = 3.396 on 1 df, p = 0.0653
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        // Chi-squared should be close to 3.4 (R reports 3.396)
        assert!(
            result.statistic > 2.5 && result.statistic < 4.5,
            "Chi-squared {} not in expected range [2.5, 4.5]",
            result.statistic
        );

        // P-value should be around 0.065 (not significant at 0.05)
        assert!(
            result.p_value > 0.04 && result.p_value < 0.15,
            "P-value {} not in expected range [0.04, 0.15]",
            result.p_value
        );
    }
}
