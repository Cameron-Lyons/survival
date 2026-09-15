//! The `survmean` table (`R/print.survfit.R`) from stacked curve vectors,
//! and a comparison of the restricted means of several groups built on it.
//! The port itself is `surv_analysis::survmean`.

use super::{StackedCurves, kaplan_meier, stacked_curves};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::{pchisq, pnorm, qnorm};
use crate::internal::matrix::{cholesky2, chsolve2};
use crate::internal::validation::validate_length;
use crate::surv_analysis::{RmeanOption, SurvmeanTable, survmean};
use ndarray::Array2;
use pyo3::prelude::*;

/// One row of the table.  `NaN` stands for R's `NA` (a median the curve
/// never reaches); the restricted mean is absent when `rmean` is `None`
/// and the limits when the curve has none.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvfitSummaryRow {
    pub records: f64,
    pub n_max: f64,
    pub n_start: f64,
    pub events: f64,
    pub rmean: Option<f64>,
    pub se_rmean: Option<f64>,
    /// The truncation time used for the restricted mean.
    pub end_time: Option<f64>,
    pub median: f64,
    pub lower: Option<f64>,
    pub upper: Option<f64>,
}

impl SurvfitSummaryRow {
    /// The rows of a [`SurvmeanTable`], one per curve.
    fn from_table(table: &SurvmeanTable) -> Vec<Self> {
        let column = |values: &Option<Vec<f64>>, curve: usize| values.as_ref().map(|v| v[curve]);
        (0..table.records.len())
            .map(|curve| Self {
                records: table.records[curve],
                n_max: table.n_max[curve],
                n_start: table.n_start[curve],
                events: table.events[curve],
                rmean: column(&table.rmean, curve),
                se_rmean: column(&table.se_rmean, curve),
                end_time: Some(table.end_time[curve]).filter(|t| !t.is_nan()),
                median: table.median[curve],
                lower: column(&table.lower, curve),
                upper: column(&table.upper, curve),
            })
            .collect()
    }
}

/// Restricted mean of one group with a normal confidence interval.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct RmstGroupResult {
    pub group: i32,
    pub n: usize,
    pub events: f64,
    pub rmean: f64,
    pub se_rmean: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Comparison of the restricted means of `k` groups: each group against
/// the first (reference) group, and a global Wald test on `k - 1` degrees
/// of freedom.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct RmstComparisonResult {
    pub tau: f64,
    pub groups: Vec<RmstGroupResult>,
    /// `rmean[g] - rmean[reference]` for every non-reference group.
    pub difference: Vec<f64>,
    pub difference_se: Vec<f64>,
    pub difference_lower: Vec<f64>,
    pub difference_upper: Vec<f64>,
    pub difference_p_value: Vec<f64>,
    pub chisq: f64,
    pub df: usize,
    pub p_value: f64,
}

/// Compare the restricted mean survival times (up to `tau`) of the groups
/// of right-censored data.  Groups are ordered by their sorted labels;
/// the first is the reference.  Each group's mean and standard error are
/// `summary(survfit(Surv(time, status) ~ 1, subset = group), rmean =
/// tau)$table`.
pub fn rmst_comparison(
    time: &[f64],
    status: &[i32],
    group: &[i32],
    weights: Option<&[f64]>,
    tau: f64,
    conf_level: f64,
) -> SurvivalResult<RmstComparisonResult> {
    validate_length(time.len(), group.len(), "group")?;
    if !tau.is_finite() || tau <= 0.0 {
        return Err(SurvivalError::invalid_input("tau must be positive"));
    }
    let mut labels = group.to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return Err(SurvivalError::invalid_input(
            "at least two groups are needed for a comparison",
        ));
    }
    let z = qnorm((1.0 + conf_level) / 2.0, true, false);
    let mut groups = Vec::with_capacity(labels.len());
    for &label in &labels {
        let rows: Vec<usize> = (0..time.len()).filter(|&i| group[i] == label).collect();
        let time: Vec<f64> = rows.iter().map(|&i| time[i]).collect();
        let status: Vec<i32> = rows.iter().map(|&i| status[i]).collect();
        let weights: Option<Vec<f64>> =
            weights.map(|weights| rows.iter().map(|&i| weights[i]).collect());
        let km = kaplan_meier(&time, &status, weights.as_deref(), conf_level)?;
        let table = survmean(&km, 1.0, RmeanOption::At(tau))?;
        let rmean = table.rmean.as_ref().map_or(f64::NAN, |v| v[0]);
        let se = table.se_rmean.as_ref().map_or(f64::NAN, |v| v[0]);
        groups.push(RmstGroupResult {
            group: label,
            n: time.len(),
            events: table.events[0],
            rmean,
            se_rmean: se,
            lower: rmean - z * se,
            upper: rmean + z * se,
        });
    }
    let reference = &groups[0];
    let k = groups.len() - 1;
    let difference: Vec<f64> = groups[1..]
        .iter()
        .map(|g| g.rmean - reference.rmean)
        .collect();
    let difference_se: Vec<f64> = groups[1..]
        .iter()
        .map(|g| (g.se_rmean.powi(2) + reference.se_rmean.powi(2)).sqrt())
        .collect();
    let difference_p_value = difference
        .iter()
        .zip(&difference_se)
        .map(|(d, se)| 2.0 * pnorm((d / se).abs(), false, false))
        .collect();
    // Wald test: the differences share the reference group's variance.
    let mut covariance = Array2::from_elem((k, k), reference.se_rmean.powi(2));
    for (i, g) in groups[1..].iter().enumerate() {
        covariance[[i, i]] += g.se_rmean.powi(2);
    }
    cholesky2(&mut covariance, 1e-9);
    let mut solution = difference.clone();
    chsolve2(&covariance, &mut solution);
    let chisq: f64 = difference.iter().zip(&solution).map(|(d, s)| d * s).sum();
    Ok(RmstComparisonResult {
        tau,
        groups,
        difference_lower: difference
            .iter()
            .zip(&difference_se)
            .map(|(d, se)| d - z * se)
            .collect(),
        difference_upper: difference
            .iter()
            .zip(&difference_se)
            .map(|(d, se)| d + z * se)
            .collect(),
        difference,
        difference_se,
        difference_p_value,
        chisq,
        df: k,
        p_value: pchisq(chisq, k as f64, false, false),
    })
}

/// Python entry point of `survmean` on the stacked vectors of one
/// `survfit` object: `strata` gives the number of rows of each curve
/// (`None` for a single curve), `n` is `fit$n`, `n_id` the optional
/// `fit$n.id` and `start_time` the `t0` the area under the curve starts
/// from.  `rmean` is `"none"`, `"common"`, `"individual"` or a number;
/// `rmean_at` gives a numeric truncation time directly.
#[pyfunction(name = "survmean_curves")]
#[pyo3(signature = (time, surv, n_risk, n_event, n, lower=None, upper=None, strata=None, n_id=None, start_time=0.0, rmean="common", rmean_at=None, scale=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn survmean_curves_py(
    time: Vec<f64>,
    surv: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n: Vec<f64>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    n_id: Option<Vec<f64>>,
    start_time: f64,
    rmean: &str,
    rmean_at: Option<f64>,
    scale: f64,
) -> PyResult<Vec<SurvfitSummaryRow>> {
    let option = match rmean_at {
        Some(value) => RmeanOption::At(value),
        None => RmeanOption::parse(rmean)?,
    };
    let fit = stacked_curves(&StackedCurves {
        time: &time,
        surv: &surv,
        n_risk: &n_risk,
        n_event: &n_event,
        lower: lower.as_deref(),
        upper: upper.as_deref(),
        strata: strata.as_deref(),
        n: &n,
        n_id: n_id.as_deref(),
        t0: start_time,
    })?;
    let table = survmean(&fit, scale, option)?;
    Ok(SurvfitSummaryRow::from_table(&table))
}

/// Python entry point: `rmst_comparison(time, status, group, tau,
/// weights=None, conf_level=0.95)`.
#[pyfunction(name = "rmst_comparison")]
#[pyo3(signature = (time, status, group, tau, weights=None, conf_level=0.95))]
pub fn rmst_comparison_py(
    time: Vec<f64>,
    status: Vec<i32>,
    group: Vec<i32>,
    tau: f64,
    weights: Option<Vec<f64>>,
    conf_level: f64,
) -> PyResult<RmstComparisonResult> {
    Ok(rmst_comparison(
        &time,
        &status,
        &group,
        weights.as_deref(),
        tau,
        conf_level,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(
        time: &[f64],
        surv: &[f64],
        n_risk: &[f64],
        n_event: &[f64],
        strata: Option<&[usize]>,
        n: &[f64],
        rmean: RmeanOption,
    ) -> Vec<SurvfitSummaryRow> {
        let fit = stacked_curves(&StackedCurves {
            time,
            surv,
            n_risk,
            n_event,
            lower: None,
            upper: None,
            strata,
            n,
            n_id: None,
            t0: 0.0,
        })
        .unwrap();
        SurvfitSummaryRow::from_table(&survmean(&fit, 1.0, rmean).unwrap())
    }

    #[test]
    fn restricted_mean_is_the_area_under_the_curve() {
        // survfit(Surv(c(1,2,3,4), c(1,1,0,1)) ~ 1): steps 0.75, 0.5, 0.25
        let time = [1.0, 2.0, 3.0, 4.0];
        let surv = [0.75, 0.5, 0.5, 0.0];
        let n_risk = [4.0, 3.0, 2.0, 1.0];
        let n_event = [1.0, 1.0, 0.0, 1.0];
        let common = rows(
            &time,
            &surv,
            &n_risk,
            &n_event,
            None,
            &[4.0],
            RmeanOption::Common,
        );
        let row = &common[0];
        // 1*1 + 0.75*1 + 0.5*1 + 0.5*1 = 2.75
        assert!((row.rmean.unwrap() - 2.75).abs() < 1e-12);
        assert_eq!(row.records, 4.0);
        assert_eq!(row.n_max, 4.0);
        assert_eq!(row.n_start, 4.0);
        assert_eq!(row.events, 3.0);
        // median: surv hits exactly 0.5 at t=2 and drops below at t=4 -> 3
        assert!((row.median - 3.0).abs() < 1e-12);
        assert_eq!(row.lower, None);
        assert_eq!(row.end_time, Some(4.0));
        let truncated = rows(
            &time,
            &surv,
            &n_risk,
            &n_event,
            None,
            &[4.0],
            RmeanOption::At(2.5),
        );
        assert!((truncated[0].rmean.unwrap() - (1.0 + 0.75 + 0.25)).abs() < 1e-12);
        let none = rows(
            &time,
            &surv,
            &n_risk,
            &n_event,
            None,
            &[4.0],
            RmeanOption::None,
        );
        assert_eq!(none[0].rmean, None);
        assert_eq!(none[0].end_time, None);
    }

    #[test]
    fn common_and_individual_truncation_differ_across_strata() {
        let time = [1.0, 2.0, 3.0, 6.0];
        let surv = [0.5, 0.0, 0.5, 0.0];
        let n_risk = [2.0, 1.0, 2.0, 1.0];
        let n_event = [1.0, 1.0, 1.0, 1.0];
        let common = rows(
            &time,
            &surv,
            &n_risk,
            &n_event,
            Some(&[2, 2]),
            &[2.0, 2.0],
            RmeanOption::Common,
        );
        assert_eq!(common[0].end_time, Some(6.0));
        let individual = rows(
            &time,
            &surv,
            &n_risk,
            &n_event,
            Some(&[2, 2]),
            &[2.0, 2.0],
            RmeanOption::Individual,
        );
        assert_eq!(individual[0].end_time, Some(2.0));
        assert_eq!(individual[1].end_time, Some(6.0));
        assert_eq!(common[0].rmean, individual[0].rmean);
    }

    #[test]
    fn comparison_handles_three_groups() {
        let time = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5, 2.0, 3.0, 5.0, 6.0];
        let status = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0];
        let group = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let result = rmst_comparison(&time, &status, &group, None, 5.0, 0.95).unwrap();
        assert_eq!(result.groups.len(), 3);
        assert_eq!(result.difference.len(), 2);
        assert_eq!(result.df, 2);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert!(
            (result.difference[0] - (result.groups[1].rmean - result.groups[0].rmean)).abs()
                < 1e-12
        );
        // summary(survfit(Surv(time, status) ~ 1, subset = group == 0), rmean = 5)$table
        assert!((result.groups[0].rmean - 2.75).abs() < 1e-12);
        assert!((result.groups[0].se_rmean - 0.649519052838329).abs() < 1e-12);
        assert!(rmst_comparison(&time, &status, &[0; 12], None, 5.0, 0.95).is_err());
    }
}
