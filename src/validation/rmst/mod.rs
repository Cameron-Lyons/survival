//! Restricted mean survival time and survival quantiles from stacked
//! curve vectors, plus the summaries built on the restricted mean.
//!
//! The ports of R's `survmean` (`R/print.survfit.R`) and `quantile.survfit`
//! (`R/quantile.survfit.R`) live in `surv_analysis::survfit_summary` and
//! read a [`SurvfitKMResult`].  This module keeps the callers that do not
//! start from such a fit:
//!
//! * [`survmean_curves_py`] and [`quantile_survfit_curves_py`] (the
//!   `survmean_curves` / `quantile_survfit_curves` bindings) take the
//!   stacked `time`, `surv`, ... vectors of a `survfit` object, assemble a
//!   [`SurvfitKMResult`] with [`stacked_curves`] and call the ports.
//! * [`rmst_comparison`] (`survmean.rs`) compares the restricted means of
//!   several groups; `threshold.rs` chooses a restricted-mean horizon and
//!   `nnt.rs` computes the number needed to treat.  None of these has an R
//!   counterpart; their Kaplan-Meier curves come from
//!   `surv_analysis::survfitkm` through [`kaplan_meier`].

mod nnt;
mod quantiles;
mod survmean;
mod threshold;

pub use nnt::{NNTResult, number_needed_to_treat, number_needed_to_treat_py};
pub use quantiles::{SurvfitCurveQuantiles, quantile_survfit_curves_py};
pub use survmean::{
    RmstComparisonResult, RmstGroupResult, SurvfitSummaryRow, rmst_comparison, rmst_comparison_py,
    survmean_curves_py,
};
pub use threshold::{
    ChangepointInfo, RMSTOptimalThresholdResult, rmst_optimal_threshold, rmst_optimal_threshold_py,
};

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm};

/// The stacked curve vectors of a `survfit` object (`survmean_curves` /
/// `quantile_survfit_curves` arguments).  `strata` gives the rows of each
/// curve (`None` for a single curve), `n` R's `fit$n` and `n_id` the
/// optional `fit$n.id`; `lower` and `upper` are given together or not at
/// all.
#[derive(Clone)]
struct StackedCurves<'a> {
    time: &'a [f64],
    surv: &'a [f64],
    n_risk: &'a [f64],
    n_event: &'a [f64],
    lower: Option<&'a [f64]>,
    upper: Option<&'a [f64]>,
    strata: Option<&'a [usize]>,
    n: &'a [f64],
    n_id: Option<&'a [f64]>,
    /// R's `fit$t0`, where the area under the curve starts.
    t0: f64,
}

/// A [`SurvfitKMResult`] holding the stacked curves, so that the ports of
/// `survmean` and `quantile.survfit` can read them.  The components the
/// summaries never use (censoring counts, cumulative hazard, standard
/// errors) are left empty.
fn stacked_curves(curves: &StackedCurves<'_>) -> SurvivalResult<SurvfitKMResult> {
    let n_rows = curves.time.len();
    validate_length(n_rows, curves.surv.len(), "surv")?;
    validate_length(n_rows, curves.n_risk.len(), "n_risk")?;
    validate_length(n_rows, curves.n_event.len(), "n_event")?;
    if let Some(lower) = curves.lower {
        validate_length(n_rows, lower.len(), "lower")?;
    }
    if let Some(upper) = curves.upper {
        validate_length(n_rows, upper.len(), "upper")?;
    }
    if curves.lower.is_some() != curves.upper.is_some() {
        return Err(SurvivalError::invalid_input(
            "lower and upper limits must be given together",
        ));
    }
    validate_finite(curves.time, "time")?;
    if !curves.t0.is_finite() {
        return Err(SurvivalError::invalid_input("start time must be finite"));
    }
    let sizes: Vec<usize> = curves
        .strata
        .map_or_else(|| vec![n_rows], <[usize]>::to_vec);
    let total: usize = sizes.iter().sum();
    validate_length(total, n_rows, "time")?;
    if sizes.is_empty() {
        return Err(SurvivalError::invalid_input("no curves to summarise"));
    }
    validate_length(sizes.len(), curves.n.len(), "n")?;
    if let Some(n_id) = curves.n_id {
        validate_length(sizes.len(), n_id.len(), "n_id")?;
    }
    let mut start = 0;
    for &size in &sizes {
        if curves.time[start..start + size]
            .windows(2)
            .any(|pair| pair[1] < pair[0])
        {
            return Err(SurvivalError::invalid_input(
                "curve times must be non-decreasing",
            ));
        }
        start += size;
    }
    Ok(SurvfitKMResult {
        n: curves.n.iter().map(|&count| count as usize).collect(),
        time: curves.time.to_vec(),
        n_risk: curves.n_risk.to_vec(),
        n_event: curves.n_event.to_vec(),
        n_censor: vec![0.0; n_rows],
        n_enter: None,
        counts: None,
        surv: curves.surv.to_vec(),
        std_err: None,
        cumhaz: vec![0.0; n_rows],
        std_chaz: None,
        lower: curves.lower.map(<[f64]>::to_vec),
        upper: curves.upper.map(<[f64]>::to_vec),
        strata: curves.strata.map(<[usize]>::to_vec),
        strata_codes: curves.strata.map(|sizes| (0..sizes.len() as i32).collect()),
        n_id: curves
            .n_id
            .map(|n_id| n_id.iter().map(|&count| count as usize).collect()),
        logse: false,
        conf_int: 0.95,
        conf_type: "log".to_string(),
        conf_lower: "usual".to_string(),
        type_: "right".to_string(),
        t0: curves.t0,
        influence_surv: None,
        influence_chaz: None,
    })
}

/// Kaplan-Meier curve of one group of right-censored observations, with
/// log-scale confidence limits at `conf_level` (`survfit(Surv(time,
/// status) ~ 1, weights)`, so the variance is R's: Greenwood for integer
/// weights, the infinitesimal jackknife otherwise) and `survfit`'s default
/// `timefix`.
fn kaplan_meier(
    time: &[f64],
    status: &[i32],
    weights: Option<&[f64]>,
    conf_level: f64,
) -> SurvivalResult<SurvfitKMResult> {
    let n = time.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, status.len(), "status")?;
    validate_finite(time, "time")?;
    validate_binary_i32(status, "status")?;
    let weights = match weights {
        Some(weights) => {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            Some(weights.to_vec())
        }
        None => None,
    };
    if !(0.0..1.0).contains(&conf_level) {
        return Err(SurvivalError::invalid_input(
            "conf_level must be between 0 and 1",
        ));
    }
    let data = SurvfitKMData::try_new(
        None,
        time.to_vec(),
        status.to_vec(),
        weights,
        None,
        None,
        None,
    )?;
    let options = SurvfitKMOptions {
        conf_int: conf_level,
        ..SurvfitKMOptions::default()
    };
    survfitkm(&data, &options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stacked_curves_are_validated() {
        let time = [1.0, 2.0, 1.0, 3.0];
        let surv = [0.5, 0.25, 0.75, 0.5];
        let counts = [2.0, 1.0, 4.0, 2.0];
        let events = [1.0, 1.0, 1.0, 1.0];
        let base = StackedCurves {
            time: &time,
            surv: &surv,
            n_risk: &counts,
            n_event: &events,
            lower: None,
            upper: None,
            strata: Some(&[2, 2]),
            n: &[2.0, 4.0],
            n_id: None,
            t0: 0.0,
        };
        let fit = stacked_curves(&base).unwrap();
        assert_eq!(fit.curve_ranges(), vec![0..2, 2..4]);
        assert_eq!(fit.n, vec![2, 4]);
        assert_eq!(fit.strata_codes, Some(vec![0, 1]));

        let bad_sizes = StackedCurves {
            strata: Some(&[3, 2]),
            ..base.clone()
        };
        assert!(stacked_curves(&bad_sizes).is_err());
        let one_limit = StackedCurves {
            lower: Some(&surv),
            ..base.clone()
        };
        assert!(stacked_curves(&one_limit).is_err());
        let decreasing = StackedCurves {
            strata: None,
            n: &[4.0],
            ..base.clone()
        };
        assert!(
            stacked_curves(&decreasing)
                .unwrap_err()
                .to_string()
                .contains("non-decreasing")
        );
        let wrong_n = StackedCurves { n: &[4.0], ..base };
        assert!(stacked_curves(&wrong_n).is_err());
    }
}
