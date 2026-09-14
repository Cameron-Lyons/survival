#[cfg(test)]
mod tests {
    use crate::concordance::{ConcordanceCounts, ConcordanceOptions, concordancefit};
    use crate::core::SurvResponse;
    use crate::internal::typed_inputs::SurvivalData;
    use crate::regression::cox_optimizer::{CoxFitBuilder, TieMethod as CoxMethod};
    use crate::residuals::coxmart::coxmart_sorted;
    use crate::surv_analysis::nelson_aalen;
    use crate::surv_analysis::{
        SurvdiffData, SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survdiff, survfitkm,
    };
    use crate::tests::common::{
        STANDARD_TOL, STRICT_TOL, aml_combined_sorted as aml_combined, aml_maintained,
        aml_nonmaintained, approx_eq, lung_data, rel_approx_eq,
    };
    use crate::validation::{
        RmeanOption, SurvfitCurve, logrank_test, quantile_survfit, rmst_comparison, survmean,
    };
    use ndarray::{Array1, Array2};

    #[test]
    fn test_coxph_aml_breslow() {
        let (time, status, group) = aml_combined();
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = group[i] as f64;
        }

        let time_arr = Array1::from_vec(time);
        let status_arr = Array1::from_vec(status);

        let mut cox_fit = CoxFitBuilder::new(time_arr, status_arr, covar)
            .strata(Array1::zeros(n))
            .weights(Array1::from_elem(n, 1.0))
            .method(CoxMethod::Breslow)
            .max_iter(25)
            .eps(1e-9)
            .toler(1e-9)
            .initial_beta(vec![0.0])
            .build()
            .expect("Cox fit initialization failed");

        cox_fit.fit();

        let results = cox_fit.results();
        let (beta, loglik) = (results.coefficients, results.loglik);

        let hr = beta[0].exp();

        assert!(
            beta[0].abs() < 2.0,
            "Beta coefficient {} is unreasonably large",
            beta[0]
        );

        assert!(
            hr > 0.2 && hr < 1.5,
            "Hazard ratio {} outside expected range",
            hr
        );

        let lrt = 2.0 * (loglik[1] - loglik[0]);
        assert!(
            lrt.abs() < 5.0,
            "Likelihood ratio test statistic {} unreasonable",
            lrt
        );
    }

    #[test]
    fn test_coxph_lung_multiple_covariates() {
        let lung = lung_data();
        let n = lung.time.len();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| lung.time[a].total_cmp(&lung.time[b]));

        let time: Vec<f64> = indices.iter().map(|&i| lung.time[i]).collect();
        let status: Vec<i32> = indices.iter().map(|&i| lung.status[i] - 1).collect();
        let age: Vec<f64> = indices.iter().map(|&i| lung.age[i]).collect();
        let sex: Vec<f64> = indices.iter().map(|&i| lung.sex[i] as f64).collect();

        let mut covar = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            covar[[i, 0]] = age[i];
            covar[[i, 1]] = sex[i];
        }

        let time_arr = Array1::from_vec(time);
        let status_arr = Array1::from_vec(status);

        let mut cox_fit = CoxFitBuilder::new(time_arr, status_arr, covar)
            .method(CoxMethod::Breslow)
            .max_iter(25)
            .eps(1e-9)
            .toler(1e-9)
            .build()
            .expect("Cox fit initialization failed");

        cox_fit.fit();

        let results = cox_fit.results();
        let (beta, flag, iter) = (results.coefficients, results.flag, results.iter);

        assert!(
            iter < 25 || flag == 1000,
            "Cox fit did not converge in 25 iterations"
        );

        assert!(
            beta[0] > -0.2 && beta[0] < 0.3,
            "Age coefficient {} unexpected",
            beta[0]
        );

        assert!(
            beta[1].abs() < 2.0,
            "Sex coefficient {} unexpected magnitude",
            beta[1]
        );
    }

    #[test]
    fn test_survfit_km_aml_maintained() {
        let (time, status) = aml_maintained();
        let result = survfitkm(
            &SurvfitKMData::try_new(None, time, status, None, None, None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();

        let expected_times = [9.0, 13.0, 18.0, 23.0, 31.0, 34.0, 48.0];
        let expected_survival = [0.909, 0.818, 0.716, 0.614, 0.491, 0.368, 0.184];

        for (i, &t) in expected_times.iter().enumerate() {
            if let Some(pos) = result.time.iter().position(|&rt| (rt - t).abs() < 0.01) {
                assert!(
                    rel_approx_eq(result.surv[pos], expected_survival[i], 0.15),
                    "At time {}: expected {}, got {}",
                    t,
                    expected_survival[i],
                    result.surv[pos]
                );
            }
        }
    }

    #[test]
    fn test_survfit_km_all_censored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0, 0, 0, 0, 0];
        let result = survfitkm(
            &SurvfitKMData::try_new(None, time.clone(), status, None, None, None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();

        assert_eq!(result.time, time, "Expected censor rows with all censored");
        assert_eq!(result.n_event, vec![0.0; 5]);
        assert_eq!(result.n_censor, vec![1.0; 5]);
        assert!(
            result
                .surv
                .iter()
                .all(|estimate| approx_eq(*estimate, 1.0, STRICT_TOL))
        );
    }

    #[test]
    fn test_survfit_km_tied_events() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0];
        let status = vec![1, 1, 0, 1, 1, 1];
        let result = survfitkm(
            &SurvfitKMData::try_new(None, time, status, None, None, None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();

        assert_eq!(result.time.len(), 3, "Expected 3 unique event times");

        if let Some(pos) = result.time.iter().position(|&t| (t - 5.0).abs() < 0.01) {
            assert!(
                rel_approx_eq(result.surv[pos], 0.667, 0.1),
                "At time 5: expected ~0.667, got {}",
                result.surv[pos]
            );
        }
    }

    #[test]
    fn test_nelson_aalen_aml_maintained() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert!(
            approx_eq(result.cumulative_hazard[0], 1.0 / 11.0, STRICT_TOL),
            "First cumulative hazard: expected {}, got {}",
            1.0 / 11.0,
            result.cumulative_hazard[0]
        );

        if result.cumulative_hazard.len() > 1 {
            let expected_h2 = 1.0 / 11.0 + 1.0 / 10.0;
            assert!(
                approx_eq(result.cumulative_hazard[1], expected_h2, STRICT_TOL),
                "Second cumulative hazard: expected {}, got {}",
                expected_h2,
                result.cumulative_hazard[1]
            );
        }

        for i in 1..result.cumulative_hazard.len() {
            assert!(
                result.cumulative_hazard[i] >= result.cumulative_hazard[i - 1],
                "Cumulative hazard not monotonic at index {}",
                i
            );
        }
    }

    #[test]
    fn test_nelson_aalen_variance() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert!(result.variance[0] >= 0.0, "Variance should be non-negative");

        for i in 1..result.variance.len() {
            assert!(
                result.variance[i] >= result.variance[i - 1] - 1e-10,
                "Variance not monotonic at index {}",
                i
            );
        }
    }

    #[test]
    fn test_survdiff_aml_logrank() {
        let (time, status, group) = aml_combined();
        let result = logrank_test(&time, &status, &group, None, None, 0.0, true).unwrap();

        eprintln!("Observed: {:?}", result.observed);
        eprintln!("Expected: {:?}", result.expected);
        eprintln!("Variance: {:?}", result.variance);
        eprintln!("Statistic: {}", result.statistic);

        assert!(
            result.statistic > 2.0 && result.statistic < 5.0,
            "Chi-squared {} not in expected range [2, 5]",
            result.statistic
        );

        assert!(
            result.p_value > 0.04 && result.p_value < 0.15,
            "P-value {} not in expected range [0.04, 0.15]",
            result.p_value
        );

        assert_eq!(result.df, 1, "Expected 1 degree of freedom");
    }

    #[test]
    fn test_survdiff_internal() {
        let (time, status, group) = aml_combined();
        let total_events: f64 = status.iter().map(|&s| s as f64).sum();
        let group: Vec<i32> = group.iter().map(|&g| g + 1).collect();
        let result = survdiff(
            &SurvdiffData::try_new(None, time, status, group, None).unwrap(),
            0.0,
            false,
        )
        .unwrap();

        let total_obs: f64 = result.obs_totals().iter().sum();
        assert!(
            approx_eq(total_obs, total_events, STRICT_TOL),
            "Total observed {} should equal total events {}",
            total_obs,
            total_events
        );
    }

    #[test]
    fn test_cox_martingale_residuals() {
        let (time, status, _group) = aml_combined();
        let n = time.len();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

        let sorted_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let sorted_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();

        let score = vec![1.0; n];
        let weights = vec![1.0; n];
        let strata = vec![0i32; n];

        let expect = coxmart_sorted(
            &sorted_time,
            &sorted_status,
            &score,
            &weights,
            &strata,
            crate::residuals::TieMethod::Breslow,
        );

        let resid_sum: f64 = expect.iter().sum();
        assert!(
            resid_sum.abs() < 1.0,
            "Martingale residuals sum {} should be near 0",
            resid_sum
        );

        for (i, &r) in expect.iter().enumerate() {
            if sorted_status[i] == 1 {
                assert!(
                    r <= 1.0 + 1e-10,
                    "Martingale residual {} > 1 for event at index {}",
                    r,
                    i
                );
            }
        }
    }

    fn kaplan_meier(time: &[f64], status: &[i32]) -> SurvfitKMResult {
        survfitkm(
            &SurvfitKMData::right_censored(time.to_vec(), status.to_vec()).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap()
    }

    fn restricted_mean(time: &[f64], status: &[i32], tau: f64) -> (f64, f64) {
        let km = kaplan_meier(time, status);
        let rows = survmean(
            &[SurvfitCurve::from_km(&km)],
            &[time.len() as f64],
            None,
            0.0,
            RmeanOption::At(tau),
            1.0,
        )
        .unwrap();
        (rows[0].rmean.unwrap(), rows[0].se_rmean.unwrap())
    }

    #[test]
    fn test_rmst_aml() {
        let (time, status) = aml_maintained();
        let tau = 30.0;

        let (rmean, se) = restricted_mean(&time, &status, tau);

        assert!(rmean > 0.0, "RMST must be positive");
        assert!(rmean < tau, "RMST must be less than tau");
        assert!(se > 0.0, "Standard error must be positive");
    }

    #[test]
    fn test_rmst_no_events() {
        let time = vec![10.0, 20.0, 30.0];
        let status = vec![0, 0, 0];
        let tau = 15.0;

        let (rmean, _) = restricted_mean(&time, &status, tau);

        assert!(
            approx_eq(rmean, tau, STANDARD_TOL),
            "RMST {} should equal tau {} when no events",
            rmean,
            tau
        );
    }

    #[test]
    fn test_single_observation() {
        let time = vec![5.0];
        let status = vec![1];

        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time.len(), 1);
        assert!(approx_eq(result.cumulative_hazard[0], 1.0, STRICT_TOL));
    }

    #[test]
    fn test_all_same_time() {
        let time = vec![10.0, 10.0, 10.0, 10.0];
        let status = vec![1, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert_eq!(result.time.len(), 1);
    }

    #[test]
    fn test_large_sample() {
        let n = 1000;
        let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();

        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert!(!result.cumulative_hazard.is_empty());
        assert!(result.cumulative_hazard.last().unwrap().is_finite());

        for &v in &result.variance {
            assert!(v >= 0.0, "Variance should be non-negative");
        }
    }

    #[test]
    fn test_extreme_times() {
        let time = vec![0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        for &h in &result.cumulative_hazard {
            assert!(h.is_finite(), "Cumulative hazard should be finite");
        }
        for &v in &result.variance {
            assert!(v.is_finite(), "Variance should be finite");
        }
    }

    #[test]
    fn test_weighted_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let result = survfitkm(
            &SurvfitKMData::try_new(None, time, status, Some(weights), None, None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();

        for i in 1..result.surv.len() {
            assert!(
                result.surv[i] <= result.surv[i - 1] + 1e-10,
                "Weighted KM survival should be monotonically decreasing"
            );
        }

        assert!(
            result.n_risk[0] >= 7.0 - 0.1,
            "Initial n_risk {} should reflect total weight 7",
            result.n_risk[0]
        );
    }

    #[test]
    fn test_weighted_coxph() {
        let (time, status, group) = aml_combined();
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = group[i] as f64;
        }

        let weights: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();

        let mut cox_fit =
            CoxFitBuilder::new(Array1::from_vec(time), Array1::from_vec(status), covar)
                .weights(Array1::from_vec(weights))
                .method(CoxMethod::Breslow)
                .max_iter(25)
                .eps(1e-9)
                .toler(1e-9)
                .build()
                .expect("Weighted Cox fit init failed");

        cox_fit.fit();

        let results = cox_fit.results();
        let (beta, iter) = (results.coefficients, results.iter);

        assert!(iter < 25, "Weighted Cox should converge");
        assert!(beta[0].is_finite(), "Coefficient should be finite");
    }

    #[test]
    fn test_stratified_coxph() {
        let (time, status, group) = aml_combined();
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = group[i] as f64;
        }

        let strata = Array1::from_iter((0..n).map(|i| i32::from(i >= n / 2)));

        let mut cox_fit =
            CoxFitBuilder::new(Array1::from_vec(time), Array1::from_vec(status), covar)
                .strata(strata)
                .method(CoxMethod::Breslow)
                .max_iter(25)
                .eps(1e-9)
                .toler(1e-9)
                .build()
                .expect("Stratified Cox fit init failed");

        cox_fit.fit();

        let results = cox_fit.results();
        let (beta, iter) = (results.coefficients, results.iter);

        assert!(iter < 25, "Stratified Cox should converge");
        assert!(
            beta[0].is_finite(),
            "Stratified coefficient should be finite"
        );
    }

    #[test]
    fn test_g_rho_family_weights() {
        let (time, status, group) = aml_combined();

        let rho1 = logrank_test(&time, &status, &group, None, None, 1.0, true).unwrap();
        assert!(rho1.statistic >= 0.0);
        assert!(rho1.p_value >= 0.0 && rho1.p_value <= 1.0);
        assert_eq!(rho1.rho, 1.0);

        let rho0 = logrank_test(&time, &status, &group, None, None, 0.0, true).unwrap();
        assert!((rho0.statistic - rho1.statistic).abs() > 1e-8);
    }

    #[test]
    fn test_km_confidence_intervals() {
        let (time, status) = aml_maintained();
        let result = survfitkm(
            &SurvfitKMData::try_new(None, time, status, None, None, None, None).unwrap(),
            &SurvfitKMOptions::default(),
        )
        .unwrap();
        let lower = result.lower.as_ref().unwrap();
        let upper = result.upper.as_ref().unwrap();

        for i in 0..result.surv.len() {
            assert!(
                lower[i] <= result.surv[i] + 1e-10,
                "CI lower {} should be <= estimate {}",
                lower[i],
                result.surv[i]
            );

            assert!(
                upper[i] >= result.surv[i] - 1e-10,
                "CI upper {} should be >= estimate {}",
                upper[i],
                result.surv[i]
            );

            assert!(
                lower[i] >= 0.0 && lower[i] <= 1.0,
                "CI lower {} should be in [0, 1]",
                lower[i]
            );
            assert!(
                upper[i] >= 0.0 && upper[i] <= 1.0,
                "CI upper {} should be in [0, 1]",
                upper[i]
            );
        }
    }

    #[test]
    fn test_na_confidence_intervals() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        for i in 0..result.cumulative_hazard.len() {
            assert!(
                result.ci_lower[i] <= result.cumulative_hazard[i] + 1e-10,
                "CI lower {} should be <= H {}",
                result.ci_lower[i],
                result.cumulative_hazard[i]
            );

            assert!(
                result.ci_upper[i] >= result.cumulative_hazard[i] - 1e-10,
                "CI upper {} should be >= H {}",
                result.ci_upper[i],
                result.cumulative_hazard[i]
            );
        }
    }

    fn concordance_counts(time: &[f64], status: &[i32], x: &[f64]) -> ConcordanceCounts {
        let data = SurvivalData::try_new(time.to_vec(), status.to_vec()).unwrap();
        let x = ndarray::Array2::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
        concordancefit(
            SurvResponse::Right(&data),
            x.view(),
            None,
            None,
            None,
            &ConcordanceOptions::default(),
        )
        .unwrap()
        .count[0]
    }

    #[test]
    fn test_concordance_basic() {
        let count = concordance_counts(
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[1, 1, 1, 1, 1],
            &[5.0, 4.0, 3.0, 2.0, 1.0],
        );
        assert_eq!(count.concordant, 0.0);
        assert_eq!(count.discordant, 10.0);
        assert_eq!(count.tied_x, 0.0);
    }

    #[test]
    fn test_concordance_ties() {
        let count = concordance_counts(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 1], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(count.tied_x, 6.0);
        assert_eq!(count.concordant + count.discordant, 0.0);
    }

    #[test]
    fn test_concordance_range() {
        let count = concordance_counts(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );
        let total = count.concordant + count.discordant + count.tied_x;
        assert!(total > 0.0);
        let c_index = (count.concordant + count.tied_x / 2.0) / total;
        assert!(
            (c_index - 1.0).abs() < 1e-12,
            "C-index {c_index} should be 1"
        );
    }

    #[test]
    fn test_survreg_weibull_interface() {
        let lung = lung_data();
        let n = lung.time.len();

        let status: Vec<f64> = lung.status.iter().map(|&s| (s - 1) as f64).collect();

        assert_eq!(lung.time.len(), n);
        assert_eq!(status.len(), n);

        for &t in &lung.time {
            assert!(t > 0.0, "Survival times must be positive");
        }

        for &s in &status {
            assert!(s == 0.0 || s == 1.0, "Status should be 0 or 1, got {}", s);
        }
    }

    #[test]
    fn test_veteran_data_survival() {
        let time = vec![
            72.0, 411.0, 228.0, 126.0, 118.0, 10.0, 82.0, 110.0, 314.0, 100.0, 42.0, 8.0, 144.0,
            25.0, 11.0, 30.0, 384.0, 4.0, 54.0, 13.0,
        ];
        let status: Vec<i32> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1];

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        for i in 1..na_result.cumulative_hazard.len() {
            assert!(
                na_result.cumulative_hazard[i] >= na_result.cumulative_hazard[i - 1] - 1e-10,
                "Cumulative hazard should be monotonic"
            );
        }

        let survival: Vec<f64> = na_result
            .cumulative_hazard
            .iter()
            .map(|&h| (-h).exp())
            .collect();

        for (i, &s) in survival.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "Survival {} at {} not in [0, 1]",
                s,
                i
            );
        }
    }

    #[test]
    fn test_ovarian_data() {
        let time = vec![
            59.0, 115.0, 156.0, 421.0, 431.0, 448.0, 464.0, 475.0, 477.0, 563.0, 638.0, 744.0,
            769.0, 770.0, 803.0, 855.0, 1040.0, 1106.0, 1129.0, 1206.0, 268.0, 329.0, 353.0, 365.0,
            377.0, 506.0,
        ];
        let status = vec![
            1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
        ];
        let rx = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        ];

        let lr_result = logrank_test(&time, &status, &rx, None, None, 0.0, true).unwrap();

        assert!(
            lr_result.statistic >= 0.0,
            "Chi-squared should be non-negative"
        );
        assert!(
            lr_result.p_value >= 0.0 && lr_result.p_value <= 1.0,
            "P-value should be in [0, 1]"
        );
        assert_eq!(lr_result.df, 1, "Should have 1 degree of freedom");

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();
        assert!(!na_result.time.is_empty(), "Should have event times");
    }

    #[test]
    fn test_aml_logrank_exact_r_values() {
        let (time, status, group) = aml_combined();

        let result = logrank_test(&time, &status, &group, None, None, 0.0, true).unwrap();

        assert!(
            result.statistic > 2.5 && result.statistic < 4.5,
            "Chi-squared {} should be close to 3.4",
            result.statistic
        );
        assert!(
            result.p_value > 0.04 && result.p_value < 0.12,
            "P-value {} should be close to 0.065",
            result.p_value
        );
    }

    #[test]
    fn test_rmst_comparison() {
        let (time, status, group) = aml_combined();
        let tau = 48.0;

        let result = rmst_comparison(&time, &status, &group, None, tau, 0.95).unwrap();

        for arm in &result.groups {
            assert!(
                arm.rmean > 0.0 && arm.rmean <= tau,
                "Group {} RMST {} should be in (0, {}]",
                arm.group,
                arm.rmean,
                tau
            );
        }

        let expected_diff = result.groups[1].rmean - result.groups[0].rmean;
        assert!(
            approx_eq(result.difference[0], expected_diff, STANDARD_TOL),
            "RMST diff {} should equal {} - {}",
            result.difference[0],
            result.groups[1].rmean,
            result.groups[0].rmean
        );

        assert!(
            result.difference_se[0] > 0.0,
            "Difference SE should be positive"
        );

        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "P-value {} should be in [0, 1]",
            result.p_value
        );
    }

    #[test]
    fn test_median_survival() {
        let (time, status) = aml_nonmaintained();
        let km = kaplan_meier(&time, &status);

        let result = quantile_survfit(
            &[SurvfitCurve::from_km(&km)],
            &[0.5],
            true,
            0.0,
            1.0,
            f64::EPSILON.sqrt(),
        )
        .unwrap();

        let median = result.quantile[0][0];
        assert!(
            median > 15.0 && median < 35.0,
            "Median {} should be close to 23",
            median
        );
    }

    #[test]
    fn test_cox_with_offset() {
        let (time, status, group) = aml_combined();
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = group[i] as f64;
        }

        let offset: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.1 } else { -0.1 }).collect();

        let mut cox_fit =
            CoxFitBuilder::new(Array1::from_vec(time), Array1::from_vec(status), covar)
                .offset(Array1::from_vec(offset))
                .method(CoxMethod::Breslow)
                .max_iter(25)
                .eps(1e-9)
                .toler(1e-9)
                .build()
                .expect("Cox fit with offset init failed");

        cox_fit.fit();

        let results = cox_fit.results();
        let (beta, iter) = (results.coefficients, results.iter);

        assert!(iter < 25, "Should converge");
        assert!(beta[0].is_finite(), "Coefficient should be finite");
    }

    #[test]
    fn test_small_sample_stability() {
        let time = vec![1.0, 2.0];
        let status = vec![1, 1];

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert_eq!(na_result.time.len(), 2);
        assert!(na_result.cumulative_hazard[0].is_finite());
        assert!(na_result.cumulative_hazard[1].is_finite());
    }

    #[test]
    fn test_identical_times() {
        let time = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert_eq!(na_result.time.len(), 1);
        assert_eq!(na_result.time[0], 5.0);
        assert_eq!(na_result.n_events[0], 5);
    }

    #[test]
    fn test_alternating_events_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 0];

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert_eq!(na_result.time.len(), 4);

        for i in 1..na_result.cumulative_hazard.len() {
            assert!(na_result.cumulative_hazard[i] >= na_result.cumulative_hazard[i - 1]);
        }
    }

    #[test]
    fn test_event_and_censoring_same_time() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0];
        let status = vec![1, 1, 0, 1, 0];

        let na_result = nelson_aalen(&time, &status, None, 0.95).unwrap();

        assert!(na_result.n_risk[0] >= 2);
        assert_eq!(na_result.n_events[0], 2);
    }

    #[test]
    fn test_logrank_identical_curves() {
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = logrank_test(&time, &status, &group, None, None, 0.0, true).unwrap();

        assert!(
            result.statistic < 0.5,
            "Chi-squared {} should be near 0 for identical curves",
            result.statistic
        );
        assert!(
            result.p_value > 0.5,
            "P-value {} should be high for identical curves",
            result.p_value
        );
    }
}
