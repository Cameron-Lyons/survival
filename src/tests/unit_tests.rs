#[cfg(test)]
mod tests {
    use crate::matrix::chinv2::chinv2;
    use crate::matrix::cholesky2::cholesky2;
    use crate::surv_analysis::survdiff2::{
        SurvDiffInput, SurvDiffOutput, SurvDiffParams, compute_survdiff,
    };
    use crate::utilities::survsplit::survsplit;
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
        compute_survdiff(params, input, &mut output);
        assert!(
            obs.iter().any(|&x| x > 0.0),
            "Should have some observations"
        );
    }
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
        compute_survdiff(params, input, &mut output);
        assert!(obs[0] > 0.0 || obs[1] > 0.0, "Should have observations");
    }
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
        compute_survdiff(params, input, &mut output);
    }
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
            rho: 1.0,
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
        compute_survdiff(params, input, &mut output);
        assert!(
            kaplan.iter().any(|&k| k > 0.0),
            "Kaplan weights should be set"
        );
    }
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
        compute_survdiff(params, input, &mut output);
        assert!(obs[0] > 0.0, "Group 1 should have observation");
        assert!(obs[1] > 0.0, "Group 2 should have observation");
    }
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
        compute_survdiff(params, input, &mut output);
        let total_obs: f64 = obs.iter().sum();
        assert!(total_obs > 0.0, "Total observations should be positive");
    }
    #[test]
    fn test_survdiff2_all_censored() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![0, 0, 0, 0, 0];
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
        compute_survdiff(params, input, &mut output);
        let total_obs: f64 = obs.iter().sum();
        assert_eq!(total_obs, 0.0, "No observations expected when all censored");
    }
    #[test]
    fn test_survsplit_with_nan_start() {
        let tstart = vec![f64::NAN, 1.0, 2.0];
        let tstop = vec![5.0, 3.0, 4.0];
        let cut = vec![2.5];
        let result = survsplit(tstart, tstop, cut);
        assert_eq!(result.row.len(), 5);
        assert!(result.start[0].is_nan());
    }
    #[test]
    fn test_survsplit_with_nan_stop() {
        let tstart = vec![1.0, 2.0, 3.0];
        let tstop = vec![f64::NAN, 4.0, 5.0];
        let cut = vec![3.5];
        let result = survsplit(tstart, tstop, cut);
        assert!(result.end[0].is_nan());
    }
    #[test]
    fn test_survsplit_with_nan_in_cuts() {
        let tstart = vec![1.0, 2.0];
        let tstop = vec![5.0, 6.0];
        let cut = vec![f64::NAN, 3.0, 4.0];
        let result = survsplit(tstart, tstop, cut);
        assert!(!result.row.is_empty());
    }
    #[test]
    fn test_survsplit_all_nan() {
        let tstart = vec![f64::NAN, f64::NAN];
        let tstop = vec![f64::NAN, f64::NAN];
        let cut = vec![1.0, 2.0];
        let result = survsplit(tstart, tstop, cut);
        assert_eq!(result.row.len(), 2);
        assert!(result.start.iter().all(|x| x.is_nan()));
        assert!(result.end.iter().all(|x| x.is_nan()));
    }
    #[test]
    fn test_survsplit_normal_operation() {
        let tstart = vec![0.0, 0.0];
        let tstop = vec![5.0, 10.0];
        let cut = vec![2.0, 4.0, 6.0, 8.0];
        let result = survsplit(tstart, tstop, cut);
        assert_eq!(result.row.len(), 8);
    }
    #[test]
    fn test_cholesky2_identity() {
        let mut matrix = vec![1.0, 0.0, 0.0, 1.0];
        let rank = cholesky2(&mut matrix, 2, 1e-10);
        assert_eq!(rank, 2);
    }
    #[test]
    fn test_cholesky2_positive_definite() {
        let mut matrix = vec![4.0, 2.0, 2.0, 5.0];
        let rank = cholesky2(&mut matrix, 2, 1e-10);
        assert_eq!(rank, 2);
    }
    #[test]
    fn test_cholesky2_singular() {
        let mut matrix = vec![1.0, 1.0, 1.0, 1.0];
        let rank = cholesky2(&mut matrix, 2, 1e-10);
        assert_eq!(rank, 1);
    }
    #[test]
    fn test_cholesky2_3x3() {
        let mut matrix = vec![4.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 6.0];
        let rank = cholesky2(&mut matrix, 3, 1e-10);
        assert_eq!(rank, 3);
    }
    #[test]
    fn test_cholesky2_with_nan() {
        let mut matrix = vec![f64::NAN, 0.0, 0.0, 1.0];
        let rank = cholesky2(&mut matrix, 2, 1e-10);
        assert!(rank <= 2);
    }
    #[test]
    fn test_chinv2_identity() {
        let mut matrix = vec![1.0, 0.0, 0.0, 1.0];
        chinv2(&mut matrix, 2);
        assert!((matrix[0] - 1.0).abs() < 1e-10);
        assert!((matrix[3] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_chinv2_diagonal() {
        let mut matrix = vec![2.0, 0.0, 0.0, 4.0];
        chinv2(&mut matrix, 2);
        assert!((matrix[0] - 0.5).abs() < 1e-10);
        assert!((matrix[3] - 0.25).abs() < 1e-10);
    }
    #[test]
    fn test_chinv2_zero_diagonal() {
        let mut matrix = vec![0.0, 0.0, 0.0, 1.0];
        chinv2(&mut matrix, 2);
        assert_eq!(matrix[0], 0.0);
        assert_eq!(matrix[1], 0.0);
        assert_eq!(matrix[2], 0.0);
    }
}
