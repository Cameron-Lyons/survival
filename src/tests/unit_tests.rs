//! Unit tests for survival analysis functions

#[cfg(test)]
mod tests {
    use crate::surv_analysis::survdiff2::{
        SurvDiffInput, SurvDiffOutput, SurvDiffParams, survdiff2_internal,
    };

    /// Test survdiff2_internal doesn't panic on standard input
    #[test]
    fn test_survdiff2_standard() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1];
        let group = vec![1, 2, 1, 2, 1];
        let strata = vec![0, 0, 0, 0, 0];

        let ngroup = 2;
        let n = 5;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // This should not panic
        survdiff2_internal(params, input, &mut output);

        // Verify some output
        assert!(
            obs.iter().any(|&x| x > 0.0),
            "Should have some observations"
        );
    }

    /// Test survdiff2_internal with all same times (edge case that previously caused panic)
    #[test]
    fn test_survdiff2_same_times() {
        let time = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let status = vec![1, 1, 1, 1, 1];
        let group = vec![1, 1, 2, 2, 2];
        let strata = vec![0, 0, 0, 0, 0];

        let ngroup = 2;
        let n = 5;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // This previously caused an infinite loop or panic
        survdiff2_internal(params, input, &mut output);

        // Should complete without hanging
        assert!(obs[0] > 0.0 || obs[1] > 0.0, "Should have observations");
    }

    /// Test survdiff2_internal with single element (edge case)
    #[test]
    fn test_survdiff2_single_element() {
        let time = vec![1.0];
        let status = vec![1];
        let group = vec![1];
        let strata = vec![0];

        let ngroup = 1;
        let n = 1;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // Should not panic
        survdiff2_internal(params, input, &mut output);
    }

    /// Test with rho != 0 (weighted log-rank / Peto-Peto test)
    #[test]
    fn test_survdiff2_weighted() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 1];
        let group = vec![1, 2, 1, 2, 1];
        let strata = vec![0, 0, 0, 0, 0];

        let ngroup = 2;
        let n = 5;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 1.0, // Peto-Peto test
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        survdiff2_internal(params, input, &mut output);

        // Kaplan-Meier weights should be computed
        assert!(
            kaplan.iter().any(|&k| k > 0.0),
            "Kaplan weights should be set"
        );
    }

    /// Test survdiff2 with two elements at same time (regression test)
    #[test]
    fn test_survdiff2_two_same_time() {
        let time = vec![1.0, 1.0];
        let status = vec![1, 1];
        let group = vec![1, 2];
        let strata = vec![0, 0];

        let ngroup = 2;
        let n = 2;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // This should not panic
        survdiff2_internal(params, input, &mut output);

        // Both groups should have one observation
        assert!(obs[0] > 0.0, "Group 1 should have observation");
        assert!(obs[1] > 0.0, "Group 2 should have observation");
    }

    /// Test survdiff2 with 10 elements (matches the Python test case size)
    #[test]
    fn test_survdiff2_ten_elements() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let group = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
        let strata = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let ngroup = 2;
        let n = 10;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // This matches the Python test that was failing
        survdiff2_internal(params, input, &mut output);

        // Verify observations were calculated
        let total_obs: f64 = obs.iter().sum();
        assert!(total_obs > 0.0, "Total observations should be positive");
    }

    /// Test survdiff2 with no events (all censored)
    #[test]
    fn test_survdiff2_all_censored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0, 0, 0, 0, 0]; // All censored
        let group = vec![1, 2, 1, 2, 1];
        let strata = vec![0, 0, 0, 0, 0];

        let ngroup = 2;
        let n = 5;

        let mut obs = vec![0.0; ngroup];
        let mut exp = vec![0.0; ngroup];
        let mut var = vec![0.0; ngroup * ngroup];
        let mut risk = vec![0.0; ngroup];
        let mut kaplan = vec![0.0; n];

        let params = SurvDiffParams {
            nn: n as i32,
            nngroup: ngroup as i32,
            _nstrat: 1,
            rho: 0.0,
        };

        let input = SurvDiffInput {
            time: &time,
            status: &status,
            group: &group,
            strata: &strata,
        };

        let mut output = SurvDiffOutput {
            obs: &mut obs,
            exp: &mut exp,
            var: &mut var,
            risk: &mut risk,
            kaplan: &mut kaplan,
        };

        // Should handle all-censored case without panic
        survdiff2_internal(params, input, &mut output);

        // No observations expected when all are censored
        let total_obs: f64 = obs.iter().sum();
        assert_eq!(total_obs, 0.0, "No observations expected when all censored");
    }
}
