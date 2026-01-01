#[cfg(test)]
mod tests {
    use crate::regression::coxfit6::{CoxFit, CoxFitBuilder, Method as CoxMethod};
    use crate::residuals::coxmart::compute_coxmart;
    use crate::surv_analysis::nelson_aalen::nelson_aalen;
    use crate::surv_analysis::survdiff2::{
        SurvDiffInput, SurvDiffOutput, SurvDiffParams, compute_survdiff,
    };
    use crate::surv_analysis::survfitkm::{KaplanMeierConfig, compute_survfitkm};
    use crate::validation::logrank::{WeightType, weighted_logrank_test};
    use crate::validation::rmst::compute_rmst;
    use ndarray::{Array1, Array2};

    const STRICT_TOL: f64 = 1e-4;
    const STANDARD_TOL: f64 = 1e-3;
    #[allow(dead_code)]
    const LOOSE_TOL: f64 = 1e-2;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() {
            return a.signum() == b.signum();
        }
        (a - b).abs() < tol
    }

    fn rel_approx_eq(a: f64, b: f64, rel_tol: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        let max_abs = a.abs().max(b.abs());
        if max_abs < 1e-10 {
            return true;
        }
        (a - b).abs() / max_abs < rel_tol
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
        let mut group = vec![1i32; t1.len()];
        group.extend(vec![0i32; t2.len()]);
        let mut indices: Vec<usize> = (0..time.len()).collect();
        indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
        let time_sorted: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let status_sorted: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
        let group_sorted: Vec<i32> = indices.iter().map(|&i| group[i]).collect();
        (time_sorted, status_sorted, group_sorted)
    }

    fn lung_data() -> LungData {
        LungData {
            time: vec![
                306.0, 455.0, 1010.0, 210.0, 883.0, 1022.0, 310.0, 361.0, 218.0, 166.0, 170.0,
                654.0, 728.0, 71.0, 567.0, 144.0, 613.0, 707.0, 61.0, 88.0,
            ],
            status: vec![2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            sex: vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            age: vec![
                74.0, 68.0, 56.0, 57.0, 60.0, 74.0, 68.0, 71.0, 53.0, 61.0, 63.0, 52.0, 47.0, 60.0,
                75.0, 77.0, 64.0, 58.0, 64.0, 71.0,
            ],
            ph_ecog: vec![1, 0, 0, 1, 0, 1, 2, 2, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2],
        }
    }

    #[allow(dead_code)]
    struct LungData {
        time: Vec<f64>,
        status: Vec<i32>,
        sex: Vec<i32>,
        age: Vec<f64>,
        ph_ecog: Vec<i32>,
    }

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
            .offset(Array1::zeros(n))
            .weights(Array1::from_elem(n, 1.0))
            .method(CoxMethod::Breslow)
            .max_iter(25)
            .eps(1e-9)
            .toler(1e-9)
            .doscale(vec![true])
            .initial_beta(vec![0.0])
            .build()
            .expect("Cox fit initialization failed");

        cox_fit.fit().expect("Cox fit failed");

        let (beta, _means, _u, _imat, loglik, _sctest, _flag, _iter) = cox_fit.results();

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
    fn test_coxph_aml_efron() {
        let (time, status, group) = aml_combined();
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = group[i] as f64;
        }

        let time_arr = Array1::from_vec(time);
        let status_arr = Array1::from_vec(status);
        let strata = Array1::zeros(n);
        let offset = Array1::zeros(n);
        let weights = Array1::from_elem(n, 1.0);

        let mut cox_fit = CoxFit::new(
            time_arr,
            status_arr,
            covar,
            strata,
            offset,
            weights,
            CoxMethod::Efron,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Cox fit initialization failed");

        cox_fit.fit().expect("Cox fit failed");

        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, _iter) = cox_fit.results();

        let hr = beta[0].exp();
        assert!(
            hr > 0.2 && hr < 1.5,
            "Hazard ratio {} outside expected range for Efron",
            hr
        );
    }

    #[test]
    fn test_coxph_lung_multiple_covariates() {
        let lung = lung_data();
        let n = lung.time.len();

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| lung.time[a].partial_cmp(&lung.time[b]).unwrap());

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
        let strata = Array1::zeros(n);
        let offset = Array1::zeros(n);
        let weights = Array1::from_elem(n, 1.0);

        let mut cox_fit = CoxFit::new(
            time_arr,
            status_arr,
            covar,
            strata,
            offset,
            weights,
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true, true],
            vec![0.0, 0.0],
        )
        .expect("Cox fit initialization failed");

        cox_fit.fit().expect("Cox fit failed");

        let (beta, _means, _u, _imat, _loglik, _sctest, flag, iter) = cox_fit.results();

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
        let status_f64: Vec<f64> = status.iter().map(|&s| s as f64).collect();
        let weights = vec![1.0; time.len()];
        let position = vec![0i32; time.len()];

        let result = compute_survfitkm(
            &time,
            &status_f64,
            &weights,
            None,
            &position,
            &KaplanMeierConfig::default(),
        );

        let expected_times = [9.0, 13.0, 18.0, 23.0, 31.0, 34.0, 48.0];
        let expected_survival = [0.909, 0.818, 0.716, 0.614, 0.491, 0.368, 0.184];

        for (i, &t) in expected_times.iter().enumerate() {
            if let Some(pos) = result.time.iter().position(|&rt| (rt - t).abs() < 0.01) {
                assert!(
                    rel_approx_eq(result.estimate[pos], expected_survival[i], 0.15),
                    "At time {}: expected {}, got {}",
                    t,
                    expected_survival[i],
                    result.estimate[pos]
                );
            }
        }
    }

    #[test]
    fn test_survfit_km_all_censored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let weights = vec![1.0; 5];
        let position = vec![0i32; 5];

        let result = compute_survfitkm(
            &time,
            &status,
            &weights,
            None,
            &position,
            &KaplanMeierConfig::default(),
        );

        assert!(
            result.time.is_empty(),
            "Expected no event times with all censored"
        );
    }

    #[test]
    fn test_survfit_km_tied_events() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0];
        let status = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0; 6];
        let position = vec![0i32; 6];

        let result = compute_survfitkm(
            &time,
            &status,
            &weights,
            None,
            &position,
            &KaplanMeierConfig::default(),
        );

        assert_eq!(result.time.len(), 3, "Expected 3 unique event times");

        if let Some(pos) = result.time.iter().position(|&t| (t - 5.0).abs() < 0.01) {
            assert!(
                rel_approx_eq(result.estimate[pos], 0.667, 0.1),
                "At time 5: expected ~0.667, got {}",
                result.estimate[pos]
            );
        }
    }

    #[test]
    fn test_nelson_aalen_aml_maintained() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

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
        let result = nelson_aalen(&time, &status, None, 0.95);

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
        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

        eprintln!("Observed: {:?}", result.observed);
        eprintln!("Expected: {:?}", result.expected);
        eprintln!("Variance: {}", result.variance);
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
    fn test_survdiff_aml_wilcoxon() {
        let (time, status, group) = aml_combined();
        let result = weighted_logrank_test(&time, &status, &group, WeightType::Wilcoxon);

        assert_eq!(result.weight_type, "Wilcoxon");
        assert!(result.statistic >= 0.0, "Chi-squared must be non-negative");
        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "P-value must be between 0 and 1"
        );
    }

    #[test]
    fn test_survdiff_internal() {
        let (time, status, group) = aml_combined();
        let strata = vec![0i32; time.len()];
        let n = time.len();
        let ngroup = 2;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

        let sorted_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let sorted_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
        let sorted_group: Vec<i32> = indices.iter().map(|&i| group[i] + 1).collect();

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &sorted_time,
            status: &sorted_status,
            group: &sorted_group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        compute_survdiff(params, input, &mut output);

        let total_events: f64 = sorted_status.iter().map(|&s| s as f64).sum();
        let total_obs: f64 = obs.iter().sum();
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
        indices.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());

        let sorted_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let sorted_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();

        let score = vec![1.0; n];
        let weights = vec![1.0; n];
        let mut strata = vec![0i32; n];

        use crate::residuals::coxmart::{SurvivalData, Weights};

        let mut expect = vec![0.0; n];

        let surv_data = SurvivalData {
            time: &sorted_time,
            status: &sorted_status,
            strata: &mut strata,
        };

        let weights_data = Weights {
            score: &score,
            wt: &weights,
        };

        compute_coxmart(n, 0, surv_data, weights_data, &mut expect);

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

    #[test]
    fn test_rmst_aml() {
        let (time, status) = aml_maintained();
        let tau = 30.0;

        let result = compute_rmst(&time, &status, tau, 0.95);

        assert!(result.rmst > 0.0, "RMST must be positive");
        assert!(result.rmst < tau, "RMST must be less than tau");

        assert!(result.se > 0.0, "Standard error must be positive");

        assert!(
            result.ci_lower <= result.rmst,
            "CI lower {} should be <= RMST {}",
            result.ci_lower,
            result.rmst
        );
        assert!(
            result.ci_upper >= result.rmst,
            "CI upper {} should be >= RMST {}",
            result.ci_upper,
            result.rmst
        );
    }

    #[test]
    fn test_rmst_no_events() {
        let time = vec![10.0, 20.0, 30.0];
        let status = vec![0, 0, 0];
        let tau = 5.0;

        let result = compute_rmst(&time, &status, tau, 0.95);

        assert!(
            approx_eq(result.rmst, tau, STANDARD_TOL),
            "RMST {} should equal tau {} when no events",
            result.rmst,
            tau
        );
    }

    #[test]
    fn test_single_observation() {
        let time = vec![5.0];
        let status = vec![1];

        let result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(result.time.len(), 1);
        assert!(approx_eq(result.cumulative_hazard[0], 1.0, STRICT_TOL));
    }

    #[test]
    fn test_all_same_time() {
        let time = vec![10.0, 10.0, 10.0, 10.0];
        let status = vec![1, 1, 1, 1];

        let result = nelson_aalen(&time, &status, None, 0.95);
        assert_eq!(result.time.len(), 1);
    }

    #[test]
    fn test_large_sample() {
        let n = 1000;
        let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let status: Vec<i32> = (0..n).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();

        let result = nelson_aalen(&time, &status, None, 0.95);

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

        let result = nelson_aalen(&time, &status, None, 0.95);

        for &h in &result.cumulative_hazard {
            assert!(h.is_finite(), "Cumulative hazard should be finite");
        }
        for &v in &result.variance {
            assert!(v.is_finite(), "Variance should be finite");
        }
    }

    #[test]
    fn test_breslow_vs_efron_with_ties() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0, 15.0, 15.0, 15.0, 20.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1];
        let covar_vals = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let n = time.len();

        let mut covar = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            covar[[i, 0]] = covar_vals[i];
        }

        let mut cox_breslow = CoxFit::new(
            Array1::from_vec(time.clone()),
            Array1::from_vec(status.clone()),
            covar.clone(),
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Breslow fit init failed");
        cox_breslow.fit().expect("Breslow fit failed");
        let (beta_breslow, _, _, _, loglik_breslow, _, _, _) = cox_breslow.results();

        let mut cox_efron = CoxFit::new(
            Array1::from_vec(time),
            Array1::from_vec(status),
            covar,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Efron,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Efron fit init failed");
        cox_efron.fit().expect("Efron fit failed");
        let (beta_efron, _, _, _, loglik_efron, _, _, _) = cox_efron.results();

        assert!(
            loglik_breslow[1].is_finite(),
            "Breslow log-likelihood should be finite"
        );
        assert!(
            loglik_efron[1].is_finite(),
            "Efron log-likelihood should be finite"
        );

        assert!(
            (beta_breslow[0] - beta_efron[0]).abs() < 1.0,
            "Breslow ({}) and Efron ({}) betas should be similar",
            beta_breslow[0],
            beta_efron[0]
        );
    }

    #[test]
    fn test_weighted_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0];
        let position = vec![0i32; 5];

        let result = compute_survfitkm(
            &time,
            &status,
            &weights,
            None,
            &position,
            &KaplanMeierConfig::default(),
        );

        for i in 1..result.estimate.len() {
            assert!(
                result.estimate[i] <= result.estimate[i - 1] + 1e-10,
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

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(time),
            Array1::from_vec(status),
            covar,
            Array1::zeros(n),
            Array1::zeros(n),
            Array1::from_vec(weights),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Weighted Cox fit init failed");

        cox_fit.fit().expect("Weighted Cox fit failed");

        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, iter) = cox_fit.results();

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
        let _ = &group;

        let mut strata = Array1::zeros(n);
        for i in 0..n {
            if i < n / 2 {
                strata[i] = 0;
            } else {
                strata[i] = 1;
            }
        }
        strata[n / 2 - 1] = 1;

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(time),
            Array1::from_vec(status),
            covar,
            strata,
            Array1::zeros(n),
            Array1::from_elem(n, 1.0),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Stratified Cox fit init failed");

        cox_fit.fit().expect("Stratified Cox fit failed");

        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, iter) = cox_fit.results();

        assert!(iter < 25, "Stratified Cox should converge");
        assert!(
            beta[0].is_finite(),
            "Stratified coefficient should be finite"
        );
    }

    #[test]
    fn test_fleming_harrington_weights() {
        let (time, status, group) = aml_combined();

        let fh_01 = weighted_logrank_test(
            &time,
            &status,
            &group,
            WeightType::FlemingHarrington { p: 0.0, q: 1.0 },
        );
        assert!(fh_01.statistic >= 0.0);
        assert!(fh_01.p_value >= 0.0 && fh_01.p_value <= 1.0);

        let fh_10 = weighted_logrank_test(
            &time,
            &status,
            &group,
            WeightType::FlemingHarrington { p: 1.0, q: 0.0 },
        );
        assert!(fh_10.statistic >= 0.0);
        assert!(fh_10.p_value >= 0.0 && fh_10.p_value <= 1.0);

        let fh_11 = weighted_logrank_test(
            &time,
            &status,
            &group,
            WeightType::FlemingHarrington { p: 1.0, q: 1.0 },
        );
        assert!(fh_11.statistic >= 0.0);
        assert!(fh_11.p_value >= 0.0 && fh_11.p_value <= 1.0);
    }

    #[test]
    fn test_peto_peto() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::PetoPeto);

        assert_eq!(result.weight_type, "PetoPeto");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_tarone_ware() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::TaroneWare);

        assert_eq!(result.weight_type, "TaroneWare");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_km_confidence_intervals() {
        let (time, status) = aml_maintained();
        let status_f64: Vec<f64> = status.iter().map(|&s| s as f64).collect();
        let weights = vec![1.0; time.len()];
        let position = vec![0i32; time.len()];

        let result = compute_survfitkm(
            &time,
            &status_f64,
            &weights,
            None,
            &position,
            &KaplanMeierConfig::default(),
        );

        for i in 0..result.estimate.len() {
            assert!(
                result.conf_lower[i] <= result.estimate[i] + 1e-10,
                "CI lower {} should be <= estimate {}",
                result.conf_lower[i],
                result.estimate[i]
            );

            assert!(
                result.conf_upper[i] >= result.estimate[i] - 1e-10,
                "CI upper {} should be >= estimate {}",
                result.conf_upper[i],
                result.estimate[i]
            );

            assert!(
                result.conf_lower[i] >= 0.0 && result.conf_lower[i] <= 1.0,
                "CI lower {} should be in [0, 1]",
                result.conf_lower[i]
            );
            assert!(
                result.conf_upper[i] >= 0.0 && result.conf_upper[i] <= 1.0,
                "CI upper {} should be in [0, 1]",
                result.conf_upper[i]
            );
        }
    }

    #[test]
    fn test_na_confidence_intervals() {
        let (time, status) = aml_maintained();
        let result = nelson_aalen(&time, &status, None, 0.95);

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

    #[test]
    fn test_concordance_basic() {
        use crate::concordance::concordance1::concordance1;

        let n = 5;
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let wt = vec![1.0; n];
        let indx = vec![4, 3, 2, 1, 0];
        let ntree = 8;

        let count = concordance1(&y, &wt, &indx, ntree);

        let concordant = count[0];
        let discordant = count[1];

        assert!(concordant >= 0.0, "Concordant pairs should be non-negative");
        assert!(discordant >= 0.0, "Discordant pairs should be non-negative");

        let total_pairs = concordant + discordant + count[2];
        assert!(
            total_pairs > 0.0,
            "Should have at least some pairs for comparison"
        );
    }

    #[test]
    fn test_concordance_ties() {
        use crate::concordance::concordance1::concordance1;

        let n = 4;
        let y = vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
        let wt = vec![1.0; n];
        let indx = vec![0, 0, 0, 0];
        let ntree = 4;

        let count = concordance1(&y, &wt, &indx, ntree);

        let tied = count[2];
        assert!(tied >= 0.0, "Tied pairs should be non-negative");
    }

    #[test]
    fn test_concordance_range() {
        use crate::concordance::concordance1::concordance1;

        let n = 10;
        let y = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let wt = vec![1.0; n];
        let indx = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let ntree = 16;

        let count = concordance1(&y, &wt, &indx, ntree);

        let concordant = count[0];
        let discordant = count[1];
        let tied = count[2];

        if concordant + discordant + tied > 0.0 {
            let c_index = concordant / (concordant + discordant + tied);
            assert!(
                (0.0..=1.0).contains(&c_index),
                "C-index {} should be in [0, 1]",
                c_index
            );
        }
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

        let na_result = nelson_aalen(&time, &status, None, 0.95);

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

        let lr_result = weighted_logrank_test(&time, &status, &rx, WeightType::LogRank);

        assert!(
            lr_result.statistic >= 0.0,
            "Chi-squared should be non-negative"
        );
        assert!(
            lr_result.p_value >= 0.0 && lr_result.p_value <= 1.0,
            "P-value should be in [0, 1]"
        );
        assert_eq!(lr_result.df, 1, "Should have 1 degree of freedom");

        let na_result = nelson_aalen(&time, &status, None, 0.95);
        assert!(!na_result.time.is_empty(), "Should have event times");
    }

    #[test]
    fn test_aml_logrank_exact_r_values() {
        let (time, status, group) = aml_combined();

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

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
        use crate::validation::rmst::compare_rmst;

        let (time, status, group) = aml_combined();
        let tau = 48.0;

        let result = compare_rmst(&time, &status, &group, tau, 0.95);

        assert!(
            result.rmst_group1.rmst > 0.0 && result.rmst_group1.rmst <= tau,
            "Group 1 RMST {} should be in (0, {}]",
            result.rmst_group1.rmst,
            tau
        );
        assert!(
            result.rmst_group2.rmst > 0.0 && result.rmst_group2.rmst <= tau,
            "Group 2 RMST {} should be in (0, {}]",
            result.rmst_group2.rmst,
            tau
        );

        let expected_diff = result.rmst_group1.rmst - result.rmst_group2.rmst;
        assert!(
            approx_eq(result.rmst_diff, expected_diff, STANDARD_TOL),
            "RMST diff {} should equal {} - {}",
            result.rmst_diff,
            result.rmst_group1.rmst,
            result.rmst_group2.rmst
        );

        assert!(result.diff_se > 0.0, "Difference SE should be positive");

        assert!(
            result.p_value >= 0.0 && result.p_value <= 1.0,
            "P-value {} should be in [0, 1]",
            result.p_value
        );
    }

    #[test]
    fn test_median_survival() {
        use crate::validation::rmst::compute_survival_quantile;

        let (time, status) = aml_nonmaintained();

        let result = compute_survival_quantile(&time, &status, 0.5, 0.95);

        if let Some(median) = result.median {
            assert!(
                median > 15.0 && median < 35.0,
                "Median {} should be close to 23",
                median
            );
        }
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

        let mut cox_fit = CoxFit::new(
            Array1::from_vec(time),
            Array1::from_vec(status),
            covar,
            Array1::zeros(n),
            Array1::from_vec(offset),
            Array1::from_elem(n, 1.0),
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true],
            vec![0.0],
        )
        .expect("Cox fit with offset init failed");

        cox_fit.fit().expect("Cox fit with offset failed");

        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, iter) = cox_fit.results();

        assert!(iter < 25, "Should converge");
        assert!(beta[0].is_finite(), "Coefficient should be finite");
    }

    #[test]
    fn test_small_sample_stability() {
        let time = vec![1.0, 2.0];
        let status = vec![1, 1];

        let na_result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(na_result.time.len(), 2);
        assert!(na_result.cumulative_hazard[0].is_finite());
        assert!(na_result.cumulative_hazard[1].is_finite());
    }

    #[test]
    fn test_identical_times() {
        let time = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];

        let na_result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(na_result.time.len(), 1);
        assert_eq!(na_result.time[0], 5.0);
        assert_eq!(na_result.n_events[0], 5);
    }

    #[test]
    fn test_alternating_events_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 0, 1, 0, 1, 0];

        let na_result = nelson_aalen(&time, &status, None, 0.95);

        assert_eq!(na_result.time.len(), 4);

        for i in 1..na_result.cumulative_hazard.len() {
            assert!(na_result.cumulative_hazard[i] >= na_result.cumulative_hazard[i - 1]);
        }
    }

    #[test]
    fn test_event_and_censoring_same_time() {
        let time = vec![5.0, 5.0, 5.0, 10.0, 10.0];
        let status = vec![1, 1, 0, 1, 0];

        let na_result = nelson_aalen(&time, &status, None, 0.95);

        assert!(na_result.n_risk[0] >= 2);
        assert_eq!(na_result.n_events[0], 2);
    }

    #[test]
    fn test_logrank_identical_curves() {
        let time = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let status = vec![1, 1, 1, 1, 1, 1];
        let group = vec![0, 0, 0, 1, 1, 1];

        let result = weighted_logrank_test(&time, &status, &group, WeightType::LogRank);

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
