//! The G-rho family of tests for a difference between survival curves:
//! R's `survdiff` (`R/survdiff.R`, `R/survdiff.fit.R`) and its C kernel
//! `survdiff2` (`src/survdiff2.c`), including the one-sample test against
//! expected survival probabilities.

use super::survfitkm::strata_index;
use crate::constants::PARALLEL_THRESHOLD_LARGE;
use crate::core::strata_order::validate_intervals;
use crate::data_prep::aeq_counting;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::pchisq;
use crate::internal::matrix::LuDecomposition;
use crate::internal::numpy_utils::{FloatVec, IntVec};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_non_empty,
};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// The data of a `survdiff(Surv(...) ~ group + strata(s))` call.
///
/// `group` and `strata` are integer codes; groups and strata are ordered by
/// ascending code (R's factor levels).  `start` extends the test to
/// `(start, stop]` intervals, which R itself refuses: the risk set at each
/// event time is then the set of intervals containing it.
#[derive(Debug, Clone)]
pub struct SurvdiffData {
    pub start: Option<Vec<f64>>,
    pub time: Vec<f64>,
    pub status: Vec<i32>,
    pub group: Vec<i32>,
    pub strata: Option<Vec<i32>>,
}

impl SurvdiffData {
    pub fn try_new(
        start: Option<Vec<f64>>,
        time: Vec<f64>,
        status: Vec<i32>,
        group: Vec<i32>,
        strata: Option<Vec<i32>>,
    ) -> SurvivalResult<Self> {
        let data = Self {
            start,
            time,
            status,
            group,
            strata,
        };
        data.validate()?;
        Ok(data)
    }

    /// Validate values and row alignment, including after public fields have
    /// been modified. [`survdiff`] checks these invariants before fitting.
    pub fn validate(&self) -> SurvivalResult<()> {
        validate_non_empty(&self.time, "time")?;
        validate_finite(&self.time, "time")?;
        validate_length(self.time.len(), self.status.len(), "status")?;
        validate_binary_i32(&self.status, "status")?;
        validate_length(self.time.len(), self.group.len(), "group")?;
        if let Some(start) = &self.start {
            validate_length(self.time.len(), start.len(), "start")?;
            validate_finite(start, "start")?;
            validate_intervals(start, &self.time)?;
        }
        if let Some(strata) = &self.strata {
            validate_length(self.time.len(), strata.len(), "strata")?;
        }
        Ok(())
    }
}

/// A `survdiff` object.  `obs` and `exp` are `groups x strata` (one column
/// without strata), `var` is the `groups x groups` covariance of `obs - exp`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvDiffResult {
    /// Observations per group (`table(groups)`).
    #[pyo3(get)]
    pub n: Vec<usize>,
    #[pyo3(get)]
    pub obs: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub exp: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub var: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub chisq: f64,
    #[pyo3(get)]
    pub pvalue: f64,
    #[pyo3(get)]
    pub df: usize,
    /// Observations per stratum when strata were given.
    #[pyo3(get)]
    pub strata: Option<Vec<usize>>,
    /// The group code of each row of `obs`, in ascending order.
    #[pyo3(get)]
    pub group_codes: Vec<i32>,
}

impl SurvDiffResult {
    /// `obs` summed over strata, one entry per group.
    pub fn obs_totals(&self) -> Vec<f64> {
        self.obs.iter().map(|row| row.iter().sum()).collect()
    }

    /// `exp` summed over strata, one entry per group.
    pub fn exp_totals(&self) -> Vec<f64> {
        self.exp.iter().map(|row| row.iter().sum()).collect()
    }
}

/// The left-continuous KM values in this stratum's existing stop-time order.
/// The reverse test sweep reads them directly, as in R's `survdiff2.c`.
/// Counting-process risk sets exclude starts at the current event time.
fn left_continuous_km(
    rows: &[usize],
    entries_desc: &[usize],
    start: Option<&[f64]>,
    time: &[f64],
    status: &[i32],
) -> Vec<f64> {
    let n = rows.len();
    let mut kaplan = vec![1.0; n];
    let mut km = 1.0;
    let mut entered = 0;
    let mut i = 0;
    while i < n {
        let current = time[rows[i]];
        let nrisk = if let Some(start) = start {
            while entered < n && start[entries_desc[n - 1 - entered]] < current {
                entered += 1;
            }
            // Every row that already ended also entered before this time.
            entered - i
        } else {
            n - i
        };
        let mut end = i;
        let mut deaths = 0.0;
        while end < n && time[rows[end]] == current {
            kaplan[end] = km;
            deaths += f64::from(status[rows[end]]);
            end += 1;
        }
        if deaths > 0.0 {
            let nrisk = nrisk as f64;
            // Keep the multiplication order used by the shared KM engine.
            km *= (nrisk - deaths) / nrisk;
        }
        i = end;
    }
    kaplan
}

/// Port of `survdiff2` (`src/survdiff2.c`) for one stratum, given its rows
/// ordered by `(time, -status)`. For `rho != 0`, reuse this ordering to
/// compute Kaplan-Meier weights. Accumulates into `obs[group][stratum]`, `exp[group][stratum]`
/// and `var`.
#[allow(clippy::too_many_arguments)]
fn survdiff_stratum(
    rows: &[usize],
    stratum: usize,
    start: Option<&[f64]>,
    time: &[f64],
    status: &[i32],
    group: &[usize],
    rho: f64,
    obs: &mut [Vec<f64>],
    exp: &mut [Vec<f64>],
    var: &mut Array2<f64>,
) {
    let n = rows.len();
    let ngroup = obs.len();
    // entry times of the stratum, descending, for the counting-process case
    let entries_desc: Vec<usize> = match start {
        Some(start) => {
            let mut entries = rows.to_vec();
            entries.sort_by(|&a, &b| start[b].total_cmp(&start[a]));
            entries
        }
        None => Vec::new(),
    };
    let kaplan = (rho != 0.0).then(|| left_continuous_km(rows, &entries_desc, start, time, status));

    // Walk backwards so risk sets accumulate.
    let mut risk = vec![0.0; ngroup];
    let mut left = 0; // intervals with start >= current time, already removed
    let mut i = n;
    while i > 0 {
        let current = time[rows[i - 1]];
        // the G-rho weight is the left-continuous Kaplan-Meier, S(t-)^rho
        let wt = match &kaplan {
            Some(kaplan) => kaplan[i - 1].powf(rho),
            None => 1.0,
        };
        let mut deaths = 0.0;
        let mut j = i;
        while j > 0 && time[rows[j - 1]] == current {
            let row = rows[j - 1];
            let k = group[row];
            deaths += f64::from(status[row]);
            risk[k] += 1.0;
            obs[k][stratum] += f64::from(status[row]) * wt;
            j -= 1;
        }
        i = j;
        if let Some(start) = start {
            // intervals that begin at or after this time are not at risk
            while left < n && start[entries_desc[left]] >= current {
                risk[group[entries_desc[left]]] -= 1.0;
                left += 1;
            }
        }
        let nrisk: f64 = risk.iter().sum();
        if deaths > 0.0 {
            for k in 0..ngroup {
                exp[k][stratum] += wt * deaths * risk[k] / nrisk;
            }
            if nrisk == 1.0 {
                continue; // only 1 subject, so no variance
            }
            let wt2 = wt * wt;
            for j in 0..ngroup {
                let tmp = wt2 * deaths * risk[j] * (nrisk - deaths) / (nrisk * (nrisk - 1.0));
                var[[j, j]] += tmp;
                for k in 0..ngroup {
                    var[[j, k]] -= tmp * risk[k] / nrisk;
                }
            }
        }
    }
}

/// R's `survdiff` chi-square: groups with no expected events are dropped,
/// the first remaining group is the reference.
fn survdiff_chisq(
    obs_totals: &[f64],
    exp_totals: &[f64],
    var: &Array2<f64>,
) -> SurvivalResult<(f64, usize)> {
    let keep: Vec<usize> = (0..exp_totals.len())
        .filter(|&k| exp_totals[k] > 0.0)
        .collect();
    let df = keep.len().saturating_sub(1);
    if keep.len() < 2 {
        return Ok((0.0, df)); // No test, actually
    }
    let contrast: Vec<f64> = keep[1..]
        .iter()
        .map(|&k| obs_totals[k] - exp_totals[k])
        .collect();
    let mut vv = Array2::zeros((df, df));
    for (r, &j) in keep[1..].iter().enumerate() {
        for (c, &k) in keep[1..].iter().enumerate() {
            vv[[r, c]] = var[[j, k]];
        }
    }
    let solution = LuDecomposition::decompose(&vv)?.solve(&contrast)?;
    let chisq = solution
        .iter()
        .zip(&contrast)
        .map(|(s, c)| s * c)
        .sum::<f64>();
    Ok((chisq, df))
}

/// Port of `survdiff` (`R/survdiff.R`) for the k-sample test.
///
/// `rho = 0` is the log-rank test, `rho = 1` the Peto & Peto modification
/// of the Gehan-Wilcoxon test, whose weights are the left-continuous
/// Kaplan-Meier curve of each stratum.
pub fn survdiff(data: &SurvdiffData, rho: f64, timefix: bool) -> SurvivalResult<SurvDiffResult> {
    data.validate()?;
    if !rho.is_finite() {
        return Err(SurvivalError::invalid_input("rho must be finite"));
    }
    let (start, time) = if timefix {
        aeq_counting(data.start.as_deref(), &data.time)?
    } else {
        (data.start.clone(), data.time.clone())
    };
    let n = time.len();
    let (group_levels, group) = strata_index(Some(&data.group), n);
    let ngroup = group_levels.len();
    if ngroup < 2 {
        return Err(SurvivalError::invalid_input("There is only 1 group"));
    }
    let (strata_levels, stratum) = strata_index(data.strata.as_deref(), n);
    let nstrat = strata_levels.len();
    // the rows of each stratum in order(strat, time, -status): one pass
    // to bucket them, then a sort per stratum; sorting (time, -status,
    // row) keys rather than indices keeps the comparisons local in memory
    let mut rows_by_stratum: Vec<Vec<usize>> = vec![Vec::new(); nstrat];
    for (i, &s) in stratum.iter().enumerate() {
        rows_by_stratum[s].push(i);
    }
    let rows_by_stratum: Vec<Vec<usize>> = rows_by_stratum
        .into_iter()
        .map(|rows| {
            let mut keys: Vec<(f64, i32, usize)> = rows
                .into_iter()
                .map(|i| (time[i], -data.status[i], i))
                .collect();
            let order = |a: &(f64, i32, usize), b: &(f64, i32, usize)| {
                a.0.total_cmp(&b.0)
                    .then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
            };
            if keys.len() > PARALLEL_THRESHOLD_LARGE {
                keys.par_sort_unstable_by(order);
            } else {
                keys.sort_unstable_by(order);
            }
            keys.into_iter().map(|(_, _, i)| i).collect()
        })
        .collect();

    let mut obs = vec![vec![0.0; nstrat]; ngroup];
    let mut exp = vec![vec![0.0; nstrat]; ngroup];
    let mut var = Array2::zeros((ngroup, ngroup));
    let strata_counts: Vec<usize> = rows_by_stratum.iter().map(Vec::len).collect();
    for (s, rows) in rows_by_stratum.iter().enumerate() {
        survdiff_stratum(
            rows,
            s,
            start.as_deref(),
            &time,
            &data.status,
            &group,
            rho,
            &mut obs,
            &mut exp,
            &mut var,
        );
    }
    let obs_totals: Vec<f64> = obs.iter().map(|row| row.iter().sum()).collect();
    let exp_totals: Vec<f64> = exp.iter().map(|row| row.iter().sum()).collect();
    let (chisq, df) = survdiff_chisq(&obs_totals, &exp_totals, &var)?;
    let mut counts = vec![0usize; ngroup];
    for &g in &group {
        counts[g] += 1;
    }
    Ok(SurvDiffResult {
        n: counts,
        obs,
        exp,
        var: var.outer_iter().map(|row| row.to_vec()).collect(),
        chisq,
        pvalue: pchisq(chisq, df as f64, false, false),
        df,
        strata: data.strata.is_some().then_some(strata_counts),
        group_codes: group_levels,
    })
}

/// The one-sample test of `survdiff(Surv(time, status) ~ offset(expected))`:
/// observed events against the expected number `sum(-log(expected))` from
/// the survival probabilities `expected` (usually `survexp(...,
/// cohort = FALSE)`).
pub fn survdiff_one_sample(
    status: &[i32],
    expected: &[f64],
    rho: f64,
) -> SurvivalResult<SurvDiffResult> {
    validate_non_empty(status, "status")?;
    validate_binary_i32(status, "status")?;
    validate_length(status.len(), expected.len(), "expected")?;
    if !rho.is_finite() {
        return Err(SurvivalError::invalid_input("rho must be finite"));
    }
    if expected.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return Err(SurvivalError::invalid_input(
            "The offset must be a survival probability",
        ));
    }
    let exp: f64 = expected.iter().map(|p| -p.ln()).sum();
    let obs: f64 = status.iter().map(|&s| f64::from(s)).sum();
    let (num, var) = if rho != 0.0 {
        let num = status
            .iter()
            .zip(expected)
            .map(|(&s, &p)| 1.0 / rho - (1.0 / rho + f64::from(s)) * p.powf(rho))
            .sum::<f64>();
        let var = expected
            .iter()
            .map(|p| 1.0 - p.powf(2.0 * rho))
            .sum::<f64>()
            / (2.0 * rho);
        (num, var)
    } else {
        (obs - exp, exp)
    };
    let chisq = num * num / var;
    Ok(SurvDiffResult {
        n: vec![status.len()],
        obs: vec![vec![obs]],
        exp: vec![vec![exp]],
        var: vec![vec![var]],
        chisq,
        pvalue: pchisq(chisq, 1.0, false, false),
        df: 1,
        strata: None,
        group_codes: vec![1],
    })
}

/// Python binding of [`survdiff`].
#[pyfunction(name = "survdiff")]
#[pyo3(signature = (time, status, group, start=None, strata=None, rho=0.0, timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn survdiff_py(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    group: IntVec,
    start: Option<FloatVec>,
    strata: Option<IntVec>,
    rho: f64,
    timefix: bool,
) -> PyResult<SurvDiffResult> {
    let data = SurvdiffData::try_new(
        start.map(FloatVec::into_inner),
        time.into_inner(),
        status.into_inner(),
        group.into_inner(),
        strata.map(IntVec::into_inner),
    )?;
    Ok(py.detach(|| survdiff(&data, rho, timefix))?)
}

/// Python binding of [`survdiff_one_sample`].
#[pyfunction(name = "survdiff_one_sample")]
#[pyo3(signature = (status, expected, rho=0.0))]
pub fn survdiff_one_sample_py(
    py: Python<'_>,
    status: IntVec,
    expected: FloatVec,
    rho: f64,
) -> PyResult<SurvDiffResult> {
    Ok(py.detach(|| survdiff_one_sample(&status, &expected, rho))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn right(time: Vec<f64>, status: Vec<i32>, group: Vec<i32>) -> SurvdiffData {
        SurvdiffData::try_new(None, time, status, group, None).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn three_group_logrank_matches_r() {
        // survdiff(Surv(time, status) ~ g) with the data below
        let result = survdiff(
            &right(
                vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                vec![1, 1, 1, 0, 1, 0, 1, 1, 0],
                vec![1, 1, 2, 1, 3, 2, 3, 2, 3],
            ),
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(result.n, vec![3, 3, 3]);
        assert_eq!(result.df, 2);
        assert_eq!(result.obs_totals(), vec![2.0, 2.0, 2.0]);
        assert_eq!(result.exp_totals(), vec![1.0, 2.25, 2.75]);
        assert!(close(result.var[0][0], 0.6825396825396826));
        assert!(close(result.var[0][1], -0.3273809523809524));
        assert!(close(result.chisq, 1.5105257668985863));
        assert!(result.pvalue > 0.0 && result.pvalue < 1.0);
        assert!(result.strata.is_none());
    }

    #[test]
    fn two_group_test_by_hand() {
        let result = survdiff(
            &right(
                vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
                vec![1, 1, 0, 1],
                vec![1, 2, 1, 2],
            ),
            0.0,
            false,
        )
        .unwrap();
        assert_eq!(result.obs_totals(), vec![1.0, 2.0]);
        assert!(close(result.exp_totals()[0], 5.0 / 6.0));
        assert!(close(result.exp_totals()[1], 13.0 / 6.0));
        assert!(close(result.var[0][0], 17.0 / 36.0));
        assert!(close(result.chisq, 1.0 / 17.0));
        // timefix bins the near-tie and the groups balance exactly
        let fixed = survdiff(
            &right(
                vec![1.0, 1.0 + 5e-10, 2.0, 3.0],
                vec![1, 1, 0, 1],
                vec![1, 2, 1, 2],
            ),
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(fixed.chisq, 0.0);
    }

    #[test]
    fn strata_accumulate_per_stratum_columns() {
        let stratified = survdiff(
            &SurvdiffData::try_new(
                None,
                vec![1.0, 2.0, 1.0, 2.0],
                vec![1, 0, 0, 1],
                vec![1, 2, 1, 2],
                Some(vec![7, 7, 9, 9]),
            )
            .unwrap(),
            0.0,
            true,
        )
        .unwrap();
        let first = survdiff(&right(vec![1.0, 2.0], vec![1, 0], vec![1, 2]), 0.0, true).unwrap();
        let second = survdiff(&right(vec![1.0, 2.0], vec![0, 1], vec![1, 2]), 0.0, true).unwrap();
        assert_eq!(stratified.strata, Some(vec![2, 2]));
        for g in 0..2 {
            assert_eq!(stratified.obs[g], vec![first.obs[g][0], second.obs[g][0]]);
            assert!(close(
                stratified.exp[g][0] + stratified.exp[g][1],
                first.exp[g][0] + second.exp[g][0]
            ));
            for k in 0..2 {
                assert!(close(
                    stratified.var[g][k],
                    first.var[g][k] + second.var[g][k]
                ));
            }
        }
    }

    #[test]
    fn rho_weights_by_the_left_continuous_kaplan_meier() {
        let data = right(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 1, 0, 1, 1],
            vec![1, 2, 1, 2, 1, 2],
        );
        let logrank = survdiff(&data, 0.0, true).unwrap();
        let wilcoxon = survdiff(&data, 1.0, true).unwrap();
        // the first event has weight 1 under both tests
        assert!(close(logrank.obs[0][0], 3.0));
        assert!(wilcoxon.obs[0][0] < logrank.obs[0][0]);
        assert!(wilcoxon.obs[0][0] > 1.0);
    }

    #[test]
    fn counting_process_intervals_shrink_the_risk_set() {
        let right_censored = survdiff(
            &right(vec![2.0, 4.0, 3.0, 5.0], vec![1, 0, 1, 1], vec![1, 2, 1, 2]),
            0.0,
            true,
        )
        .unwrap();
        let delayed = survdiff(
            &SurvdiffData::try_new(
                Some(vec![0.0, 0.0, 2.5, 0.0]),
                vec![2.0, 4.0, 3.0, 5.0],
                vec![1, 0, 1, 1],
                vec![1, 2, 1, 2],
                None,
            )
            .unwrap(),
            0.0,
            true,
        )
        .unwrap();
        // the third subject enters after the first event
        assert!(delayed.exp_totals()[0] < right_censored.exp_totals()[0]);
        assert_eq!(delayed.obs_totals(), right_censored.obs_totals());
    }

    #[test]
    fn all_censored_gives_no_test() {
        let result = survdiff(
            &right(vec![1.0, 2.0, 3.0, 4.0], vec![0, 0, 0, 0], vec![1, 1, 2, 2]),
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(result.chisq, 0.0);
        assert_eq!(result.df, 0);
    }

    #[test]
    fn weighted_tests_ignore_a_stratum_without_events() {
        for counting in [false, true] {
            let event_data = SurvdiffData::try_new(
                counting.then(|| vec![0.0, 0.0, 1.0, 2.0]),
                vec![1.0, 2.0, 3.0, 4.0],
                vec![1, 1, 0, 1],
                vec![0, 1, 0, 1],
                None,
            )
            .unwrap();
            let mixed_data = SurvdiffData::try_new(
                counting.then(|| vec![0.0, 0.0, 1.0, 2.0, 0.0, 1.0]),
                vec![1.0, 2.0, 3.0, 4.0, 2.0, 5.0],
                vec![1, 1, 0, 1, 0, 0],
                vec![0, 1, 0, 1, 0, 1],
                Some(vec![7, 7, 7, 7, 9, 9]),
            )
            .unwrap();
            for rho in [-1.0, 0.25, 1.0] {
                let event = survdiff(&event_data, rho, false).unwrap();
                let mixed = survdiff(&mixed_data, rho, false).unwrap();
                for group in 0..2 {
                    assert_eq!(mixed.obs[group], vec![event.obs[group][0], 0.0]);
                    assert_eq!(mixed.exp[group], vec![event.exp[group][0], 0.0]);
                }
                assert_eq!(mixed.var, event.var);
                assert_eq!(mixed.chisq, event.chisq);
                assert_eq!(mixed.pvalue, event.pvalue);
            }
        }
    }

    #[test]
    fn one_sample_test_matches_r_formulas() {
        let status = [1, 0, 1, 1];
        let expected = [0.9, 0.8, 0.7, 0.95];
        let result = survdiff_one_sample(&status, &expected, 0.0).unwrap();
        let exp: f64 = expected.iter().map(|p| -p.ln()).sum();
        assert_eq!(result.n, vec![4]);
        assert_eq!(result.obs, vec![vec![3.0]]);
        assert!(close(result.exp[0][0], exp));
        assert!(close(result.chisq, (3.0 - exp).powi(2) / exp));
        let rho = survdiff_one_sample(&status, &expected, 0.5).unwrap();
        assert!(rho.chisq.is_finite());
        assert!(survdiff_one_sample(&status, &[1.5, 0.8, 0.7, 0.9], 0.0).is_err());
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(SurvdiffData::try_new(None, vec![1.0], vec![2], vec![1], None).is_err());
        assert!(SurvdiffData::try_new(None, vec![1.0, 2.0], vec![1, 0], vec![1], None).is_err());
        assert!(
            survdiff(&right(vec![1.0, 2.0], vec![1, 0], vec![1, 1]), 0.0, true)
                .unwrap_err()
                .to_string()
                .contains("only 1 group")
        );
        assert!(
            survdiff(
                &right(vec![1.0, 2.0], vec![1, 0], vec![1, 2]),
                f64::NAN,
                true
            )
            .is_err()
        );
    }
}
