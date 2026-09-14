#[cfg(test)]
mod tests {
    use crate::data_prep::{SurvSplitResponse, survsplit};
    use crate::surv_analysis::{SurvdiffData, agsurv4, survdiff};

    #[test]
    fn test_agsurv4_public_alias() {
        let result = agsurv4(vec![0, 1], vec![2.0], vec![0.5], 2, vec![1.0, 4.0]).unwrap();
        assert_eq!(
            result.len(),
            2,
            "Backward-compatible agsurv4 alias should remain callable"
        );
    }

    fn survdiff_right(
        time: Vec<f64>,
        status: Vec<i32>,
        group: Vec<i32>,
        rho: f64,
    ) -> crate::surv_analysis::SurvDiffResult {
        survdiff(
            &SurvdiffData::try_new(None, time, status, group, None).unwrap(),
            rho,
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_survdiff_standard() {
        let result = survdiff_right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 0, 1, 1],
            vec![1, 2, 1, 2, 1],
            0.0,
        );
        assert_eq!(result.obs.len(), 2);
        assert!(
            result.obs_totals().iter().any(|&x| x > 0.0),
            "Should have some observations"
        );
        assert_eq!(result.df, 1);
    }
    #[test]
    fn test_survdiff_same_times_is_singular() {
        // everyone dies at once: a zero variance, which R's solve() rejects
        let err = survdiff(
            &SurvdiffData::try_new(
                None,
                vec![1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1, 1, 1, 1, 1],
                vec![1, 1, 2, 2, 2],
                None,
            )
            .unwrap(),
            0.0,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("singular"));
    }
    #[test]
    fn test_survdiff_single_group_is_rejected() {
        let err = survdiff(
            &SurvdiffData::try_new(None, vec![1.0], vec![1], vec![1], None).unwrap(),
            0.0,
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("only 1 group"));
    }
    #[test]
    fn test_survdiff_weighted() {
        let logrank = survdiff_right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 0, 1, 1],
            vec![1, 2, 1, 2, 1],
            0.0,
        );
        let wilcoxon = survdiff_right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 0, 1, 1],
            vec![1, 2, 1, 2, 1],
            1.0,
        );
        assert!(
            wilcoxon.obs_totals()[0] < logrank.obs_totals()[0],
            "Kaplan-Meier weights should down-weight later events"
        );
    }
    #[test]
    fn test_survdiff_two_same_time() {
        let result = survdiff_right(vec![1.0, 1.0, 2.0], vec![1, 1, 0], vec![1, 2, 1], 0.0);
        let obs = result.obs_totals();
        assert!(obs[0] > 0.0, "Group 1 should have observation");
        assert!(obs[1] > 0.0, "Group 2 should have observation");
    }
    #[test]
    fn test_survdiff_ten_elements() {
        let result = survdiff_right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            0.0,
        );
        let total_obs: f64 = result.obs_totals().iter().sum();
        assert!(total_obs > 0.0, "Total observations should be positive");
    }
    #[test]
    fn test_survdiff_all_censored() {
        let result = survdiff_right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0, 0, 0, 0, 0],
            vec![1, 2, 1, 2, 1],
            0.0,
        );
        let total_obs: f64 = result.obs_totals().iter().sum();
        assert_eq!(total_obs, 0.0, "No observations expected when all censored");
        assert_eq!(result.chisq, 0.0);
    }
    fn split_counting(
        tstart: &[f64],
        tstop: &[f64],
        cut: &[f64],
    ) -> crate::error::SurvivalResult<crate::data_prep::SurvSplitResult> {
        let status = vec![1.0; tstart.len()];
        let response: SurvSplitResponse<'_, i64> = SurvSplitResponse::Counting {
            start: tstart,
            stop: tstop,
            status: &status,
        };
        survsplit(response, cut, 0.0, false)
    }
    #[test]
    fn test_survsplit_with_nan_start() {
        let result = split_counting(&[f64::NAN, 1.0, 2.0], &[5.0, 3.0, 4.0], &[2.5]).unwrap();
        assert_eq!(result.row.len(), 5);
        assert!(result.start[0].is_nan());
    }
    #[test]
    fn test_survsplit_with_nan_stop() {
        let result = split_counting(&[1.0, 2.0, 3.0], &[f64::NAN, 4.0, 5.0], &[3.5]).unwrap();
        assert!(result.end[0].is_nan());
    }
    #[test]
    fn test_survsplit_with_nan_in_cuts() {
        let err = split_counting(&[1.0, 2.0], &[5.0, 6.0], &[f64::NAN, 3.0, 4.0]).unwrap_err();
        assert!(
            err.to_string()
                .contains("cut must be a vector of finite numbers")
        );
    }
    #[test]
    fn test_survsplit_all_nan() {
        let result =
            split_counting(&[f64::NAN, f64::NAN], &[f64::NAN, f64::NAN], &[1.0, 2.0]).unwrap();
        assert_eq!(result.row.len(), 2);
        assert!(result.start.iter().all(|x| x.is_nan()));
        assert!(result.end.iter().all(|x| x.is_nan()));
    }
    #[test]
    fn test_survsplit_normal_operation() {
        let result = split_counting(&[0.0, 0.0], &[5.0, 10.0], &[2.0, 4.0, 6.0, 8.0]).unwrap();
        assert_eq!(result.row.len(), 8);
    }
}
