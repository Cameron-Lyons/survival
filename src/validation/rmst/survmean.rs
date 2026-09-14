//! The `survmean` table of R's `print.survfit` / `summary.survfit`
//! (`R/print.survfit.R`): per curve the number of records, the maximum and
//! initial numbers at risk, the number of events, the restricted mean
//! survival time with its standard error, and the median with its
//! confidence limits.  A comparison of the restricted means of several
//! groups is built on top of it.

use super::{SurvfitCurve, kaplan_meier};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::{pchisq, pnorm, qnorm};
use crate::internal::matrix::{cholesky2, chsolve2};
use crate::internal::validation::validate_length;
use ndarray::Array2;
use pyo3::prelude::*;

/// R's `rmean` argument: where to truncate the mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RmeanOption {
    /// No restricted mean (R `"none"`).
    None,
    /// Truncate every curve at the largest time of any curve (R `"common"`).
    Common,
    /// Truncate each curve at its own largest time (R `"individual"`).
    Individual,
    /// Truncate at this time.
    At(f64),
}

impl RmeanOption {
    /// R's `match.arg`-style parsing of the character values.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "none" => Ok(Self::None),
            "common" => Ok(Self::Common),
            "individual" => Ok(Self::Individual),
            other => Err(SurvivalError::invalid_input(format!(
                "Invalid value for rmean option: {other:?}"
            ))),
        }
    }
}

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

/// R's `minmin`: the first time the curve drops to (or below) 0.5, with
/// the midpoint rule when it sits exactly at 0.5 for a while.
fn minmin(y: &[f64], x: &[f64]) -> f64 {
    let tolerance = f64::EPSILON.sqrt();
    let kept: Vec<(f64, f64)> = y
        .iter()
        .zip(x)
        .filter(|(y, _)| !y.is_nan() && **y < 0.5 + tolerance)
        .map(|(&y, &x)| (y, x))
        .collect();
    let Some(&(first_y, first_x)) = kept.first() else {
        return f64::NAN;
    };
    if (first_y - 0.5).abs() < tolerance
        && let Some((_, next_x)) = kept.iter().find(|(y, _)| *y < first_y)
    {
        return (first_x + next_x) / 2.0;
    }
    first_x
}

/// R's `pfun`: one table row for one curve.
fn summary_row(
    curve: &SurvfitCurve<'_>,
    nused: f64,
    n_id: Option<f64>,
    start_time: f64,
    end_time: Option<f64>,
    scale: f64,
) -> SurvfitSummaryRow {
    let time: Vec<f64> = curve.time.iter().map(|t| t / scale).collect();
    let (rmean, se_rmean) = match end_time {
        Some(end_time) => {
            let hh: Vec<f64> = curve
                .n_risk
                .iter()
                .zip(curve.n_event)
                .map(|(&n, &d)| if n - d == 0.0 { 0.0 } else { d / (n * (n - d)) })
                .collect();
            let keep = time.partition_point(|&t| t <= end_time);
            let (temptime, tempsurv, hh) = if keep == 0 {
                (vec![end_time], vec![1.0], vec![0.0])
            } else {
                let mut temptime = time[..keep].to_vec();
                temptime.push(end_time);
                let mut tempsurv = curve.surv[..keep].to_vec();
                tempsurv.push(curve.surv[keep - 1]);
                let mut hh = hh[..keep].to_vec();
                hh.push(0.0);
                (temptime, tempsurv, hh)
            };
            let n = temptime.len();
            let mut previous = start_time;
            let mut rectangles = Vec::with_capacity(n);
            for (i, &t) in temptime.iter().enumerate() {
                let height = if i == 0 { 1.0 } else { tempsurv[i - 1] };
                rectangles.push((t - previous) * height);
                previous = t;
            }
            let mean: f64 = rectangles.iter().sum();
            // sum(cumsum(rev(rectangles[-1]))^2 * rev(hh)[-1])
            let mut tail_area = 0.0;
            let mut varmean = 0.0;
            for i in (0..n - 1).rev() {
                tail_area += rectangles[i + 1];
                varmean += tail_area * tail_area * hh[i];
            }
            (Some(mean), Some(varmean.sqrt()))
        }
        None => (None, None),
    };
    let n_max = n_id.unwrap_or_else(|| curve.n_risk.iter().copied().fold(f64::MIN, f64::max));
    let median = minmin(curve.surv, &time);
    let (lower, upper) = match (curve.lower, curve.upper) {
        (Some(lower), Some(upper)) => (Some(minmin(lower, &time)), Some(minmin(upper, &time))),
        _ => (None, None),
    };
    SurvfitSummaryRow {
        records: nused,
        n_max,
        n_start: curve.n_risk.first().copied().unwrap_or(f64::NAN),
        events: curve.n_event.iter().sum(),
        rmean,
        se_rmean,
        end_time,
        median,
        lower,
        upper,
    }
}

/// The `survmean` table for a set of curves (the strata of one `survfit`
/// object).  `nused[i]` is R's `fit$n[i]` and `n_id` the optional
/// `fit$n.id`; `start_time` is `fit$t0` (or `min(0, time)`), `scale`
/// divides the times.
pub fn survmean(
    curves: &[SurvfitCurve<'_>],
    nused: &[f64],
    n_id: Option<&[f64]>,
    start_time: f64,
    rmean: RmeanOption,
    scale: f64,
) -> SurvivalResult<Vec<SurvfitSummaryRow>> {
    if curves.is_empty() {
        return Err(SurvivalError::invalid_input("no curves to summarise"));
    }
    validate_length(curves.len(), nused.len(), "nused")?;
    if let Some(n_id) = n_id {
        validate_length(curves.len(), n_id.len(), "n_id")?;
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SurvivalError::invalid_input("scale must be positive"));
    }
    for curve in curves {
        curve.validate()?;
    }
    let last_times: Vec<f64> = curves
        .iter()
        .map(|curve| curve.time.last().map_or(f64::NAN, |t| t / scale))
        .collect();
    if let RmeanOption::At(value) = rmean {
        // print.survfit: the truncation point must not precede the curve
        let smallest = curves
            .iter()
            .flat_map(|curve| curve.time.iter().copied())
            .fold(f64::INFINITY, f64::min);
        if value < smallest {
            return Err(SurvivalError::invalid_input(
                "Truncation point for the mean is < smallest survival",
            ));
        }
    }
    let common = last_times.iter().copied().fold(f64::MIN, f64::max);
    Ok(curves
        .iter()
        .enumerate()
        .map(|(i, curve)| {
            let end_time = match rmean {
                RmeanOption::None => None,
                RmeanOption::Common => Some(common),
                RmeanOption::Individual => Some(last_times[i]),
                RmeanOption::At(value) => Some(value / scale),
            };
            summary_row(
                curve,
                nused[i],
                n_id.map(|values| values[i]),
                start_time,
                end_time,
                scale,
            )
        })
        .collect())
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
/// the first is the reference.
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
        let rows_summary = survmean(
            &[SurvfitCurve::from_km(&km)],
            &[time.len() as f64],
            None,
            0.0_f64.min(time.iter().copied().fold(f64::INFINITY, f64::min)),
            RmeanOption::At(tau),
            1.0,
        )?;
        let row = &rows_summary[0];
        let rmean = row.rmean.unwrap_or(f64::NAN);
        let se = row.se_rmean.unwrap_or(f64::NAN);
        groups.push(RmstGroupResult {
            group: label,
            n: time.len(),
            events: row.events,
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

/// Python entry point for [`survmean`] on one `survfit` object: the curve
/// fields are concatenated over the strata and `strata` gives the number
/// of rows of each (`None` for a single curve).  `rmean` is `"none"`,
/// `"common"`, `"individual"` or a number.
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
    let sizes = strata.unwrap_or_else(|| vec![time.len()]);
    let curves = split_curves(
        &time,
        &surv,
        &n_risk,
        &n_event,
        lower.as_deref(),
        upper.as_deref(),
        &sizes,
    )?;
    Ok(survmean(
        &curves,
        &n,
        n_id.as_deref(),
        start_time,
        option,
        scale,
    )?)
}

/// Split concatenated curve fields into per-stratum [`SurvfitCurve`]s.
pub(super) fn split_curves<'a>(
    time: &'a [f64],
    surv: &'a [f64],
    n_risk: &'a [f64],
    n_event: &'a [f64],
    lower: Option<&'a [f64]>,
    upper: Option<&'a [f64]>,
    sizes: &[usize],
) -> SurvivalResult<Vec<SurvfitCurve<'a>>> {
    let total: usize = sizes.iter().sum();
    validate_length(total, time.len(), "time")?;
    let mut curves = Vec::with_capacity(sizes.len());
    let mut offset = 0;
    for &size in sizes {
        let range = offset..offset + size;
        let slice = |values: &'a [f64], name: &str| -> SurvivalResult<&'a [f64]> {
            values
                .get(range.clone())
                .ok_or_else(|| SurvivalError::invalid_input(format!("{name} is too short")))
        };
        curves.push(SurvfitCurve {
            time: slice(time, "time")?,
            surv: slice(surv, "surv")?,
            n_risk: slice(n_risk, "n_risk")?,
            n_event: slice(n_event, "n_event")?,
            lower: lower.map(|v| slice(v, "lower")).transpose()?,
            upper: upper.map(|v| slice(v, "upper")).transpose()?,
        });
        offset += size;
    }
    Ok(curves)
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

    fn curve<'a>(
        time: &'a [f64],
        surv: &'a [f64],
        n_risk: &'a [f64],
        n_event: &'a [f64],
    ) -> SurvfitCurve<'a> {
        SurvfitCurve {
            time,
            surv,
            n_risk,
            n_event,
            lower: None,
            upper: None,
        }
    }

    #[test]
    fn restricted_mean_is_the_area_under_the_curve() {
        // survfit(Surv(c(1,2,3,4), c(1,1,0,1)) ~ 1): steps 0.75, 0.5, 0.25
        let time = [1.0, 2.0, 3.0, 4.0];
        let surv = [0.75, 0.5, 0.5, 0.0];
        let n_risk = [4.0, 3.0, 2.0, 1.0];
        let n_event = [1.0, 1.0, 0.0, 1.0];
        let rows = survmean(
            &[curve(&time, &surv, &n_risk, &n_event)],
            &[4.0],
            None,
            0.0,
            RmeanOption::Common,
            1.0,
        )
        .unwrap();
        let row = &rows[0];
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
        let truncated = survmean(
            &[curve(&time, &surv, &n_risk, &n_event)],
            &[4.0],
            None,
            0.0,
            RmeanOption::At(2.5),
            1.0,
        )
        .unwrap();
        assert!((truncated[0].rmean.unwrap() - (1.0 + 0.75 + 0.25)).abs() < 1e-12);
        let none = survmean(
            &[curve(&time, &surv, &n_risk, &n_event)],
            &[4.0],
            None,
            0.0,
            RmeanOption::None,
            1.0,
        )
        .unwrap();
        assert_eq!(none[0].rmean, None);
    }

    #[test]
    fn variance_follows_the_greenwood_style_sum() {
        // Single event at t=1 among 2 at risk, followed by a censoring at 2.
        let time = [1.0, 2.0];
        let surv = [0.5, 0.5];
        let n_risk = [2.0, 1.0];
        let n_event = [1.0, 0.0];
        let rows = survmean(
            &[curve(&time, &surv, &n_risk, &n_event)],
            &[2.0],
            None,
            0.0,
            RmeanOption::Common,
            1.0,
        )
        .unwrap();
        // rectangles: 1, 0.5; hh = 1/(2*1) = 0.5 at t=1; varmean = 0.5^2 * 0.5
        assert!((rows[0].se_rmean.unwrap() - (0.125_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn median_midpoint_rule_matches_r() {
        let x = [1.0, 2.0, 3.0];
        assert_eq!(minmin(&[0.8, 0.5, 0.2], &x), 2.5);
        assert_eq!(minmin(&[0.8, 0.5, 0.5], &x), 2.0);
        assert_eq!(minmin(&[0.8, 0.4, 0.2], &x), 2.0);
        assert!(minmin(&[0.9, 0.8, 0.7], &x).is_nan());
    }

    #[test]
    fn common_and_individual_truncation_differ_across_strata() {
        let time_a = [1.0, 2.0];
        let surv_a = [0.5, 0.0];
        let risk_a = [2.0, 1.0];
        let event_a = [1.0, 1.0];
        let time_b = [3.0, 6.0];
        let surv_b = [0.5, 0.0];
        let risk_b = [2.0, 1.0];
        let event_b = [1.0, 1.0];
        let curves = [
            curve(&time_a, &surv_a, &risk_a, &event_a),
            curve(&time_b, &surv_b, &risk_b, &event_b),
        ];
        let common = survmean(&curves, &[2.0, 2.0], None, 0.0, RmeanOption::Common, 1.0).unwrap();
        assert_eq!(common[0].end_time, Some(6.0));
        let individual = survmean(
            &curves,
            &[2.0, 2.0],
            None,
            0.0,
            RmeanOption::Individual,
            1.0,
        )
        .unwrap();
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
        assert!(rmst_comparison(&time, &status, &[0; 12], None, 5.0, 0.95).is_err());
    }
}
