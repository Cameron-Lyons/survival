use crate::error::SurvivalResult;
use crate::surv_analysis::{SurvdiffData, survdiff};

#[doc = " Result of [`logrank_test`]: R's `survdiff` components."]
#[derive(Debug, Clone, PartialEq)]
pub struct LogRankResult {
    #[doc = " The distinct group labels, sorted; observed/expected follow this order."]
    pub groups: Vec<i32>,
    pub observed: Vec<f64>,
    pub expected: Vec<f64>,
    #[doc = " Full variance-covariance matrix of `observed - expected`."]
    pub variance: Vec<Vec<f64>>,
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    pub rho: f64,
}

pub fn logrank_test(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    entry_times: Option<&[f64]>,
    strata: Option<&[i32]>,
    rho: f64,
    timefix: bool,
) -> SurvivalResult<LogRankResult> {
    let data = SurvdiffData::try_new(
        entry_times.map(<[f64]>::to_vec),
        time.to_vec(),
        status.to_vec(),
        group.to_vec(),
        strata.map(<[i32]>::to_vec),
    )?;
    fit_logrank(&data, rho, timefix)
}

fn fit_logrank(data: &SurvdiffData, rho: f64, timefix: bool) -> SurvivalResult<LogRankResult> {
    let result = survdiff(data, rho, timefix)?;
    let observed = result.obs_totals();
    let expected = result.exp_totals();
    Ok(LogRankResult {
        groups: result.group_codes,
        observed,
        expected,
        variance: result.var,
        statistic: result.chisq,
        df: result.df,
        p_value: result.pvalue,
        rho,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_survdiff_on_a_small_example() {
        let time = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 5.0, 7.0];
        let status = [1, 1, 0, 1, 0, 1, 1, 1, 0];
        let group = [10, 10, 10, 20, 20, 20, 30, 30, 30];
        let result = logrank_test(&time, &status, &group, None, None, 0.0, true).unwrap();
        assert_eq!(result.groups, vec![10, 20, 30]);
        assert_eq!(result.df, 2);
        assert_eq!(result.observed, vec![2.0, 2.0, 2.0]);
        assert!((result.expected[0] - 1.0).abs() < 1e-12);
        assert!((result.expected[1] - 2.25).abs() < 1e-12);
        assert!((result.expected[2] - 2.75).abs() < 1e-12);
        assert!((result.variance[0][0] - 0.6825396825396826).abs() < 1e-12);
        assert!((result.statistic - 1.5105257668985863).abs() < 1e-12);
        assert!((result.p_value - 0.4698870729581883).abs() < 1e-9);
    }

    #[test]
    fn strata_and_delayed_entry_are_routed_to_the_kernel() {
        let time = [2.0, 4.0, 3.0, 5.0];
        let status = [1, 0, 1, 1];
        let group = [1, 0, 1, 0];
        let entry = [0.0, 0.0, 1.0, 2.0];
        let result = logrank_test(&time, &status, &group, Some(&entry), None, 0.0, true).unwrap();
        assert_eq!(result.observed, vec![1.0, 2.0]);
        assert!((result.expected[0] - 2.0).abs() < 1e-12);
        assert!((result.expected[1] - 1.0).abs() < 1e-12);
        assert!((result.statistic - 2.25).abs() < 1e-10);

        let strata = [7, 7, 3, 3];
        let stratified =
            logrank_test(&time, &status, &group, None, Some(&strata), 0.0, true).unwrap();
        assert_eq!(stratified.observed, vec![1.0, 2.0]);
    }

    #[test]
    fn a_single_group_is_rejected() {
        assert!(logrank_test(&[1.0, 2.0], &[1, 1], &[1, 1], None, None, 0.0, true).is_err());
    }
}
