//! `logrank_test`: a typed convenience entry over the `survdiff` kernel in
//! `surv_analysis::logrank_components` for callers holding raw group
//! labels.  It returns the G-rho family test of R's `survdiff`
//! (`R/survdiff.R`, `src/survdiff2.c`) for right-censored or (start, stop]
//! data (the latter an extension: R's `survdiff` refuses counting-process
//! data), with observed and expected counts summed over strata.

use crate::data_types::{FloatVec, IntVec};
use crate::error::SurvivalResult;
use crate::surv_analysis::{SurvdiffData, survdiff};
use pyo3::prelude::*;

/// Result of [`logrank_test`]: R's `survdiff` components.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct LogRankResult {
    /// The distinct group labels, sorted; observed/expected follow this order.
    pub groups: Vec<i32>,
    pub observed: Vec<f64>,
    pub expected: Vec<f64>,
    /// Full variance-covariance matrix of `observed - expected`.
    pub variance: Vec<Vec<f64>>,
    pub statistic: f64,
    pub df: usize,
    pub p_value: f64,
    pub rho: f64,
}

/// G-rho log-rank test of `group` on right-censored (`entry_times = None`)
/// or (start, stop] data, optionally stratified.  `timefix` applies R's
/// near-tie rounding before comparing times.
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
    let result = survdiff(&data, rho, timefix)?;
    Ok(LogRankResult {
        groups: result.group_codes.clone(),
        observed: result.obs_totals(),
        expected: result.exp_totals(),
        variance: result.var,
        statistic: result.chisq,
        df: result.df,
        p_value: result.pvalue,
        rho,
    })
}

/// Python entry point: `logrank_test(time, status, group, rho=0.0,
/// strata=None, entry_times=None, timefix=True)`.
#[pyfunction(name = "logrank_test")]
#[pyo3(signature = (time, status, group, rho=0.0, strata=None, entry_times=None, timefix=true))]
pub fn logrank_test_py(
    time: FloatVec,
    status: IntVec,
    group: IntVec,
    rho: f64,
    strata: Option<IntVec>,
    entry_times: Option<FloatVec>,
    timefix: bool,
) -> PyResult<LogRankResult> {
    Ok(logrank_test(
        &time,
        &status,
        &group,
        entry_times.as_deref(),
        strata.as_deref(),
        rho,
        timefix,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_survdiff_on_a_small_example() {
        // Same data as the survdiff2 unit tests: two groups, no strata.
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
