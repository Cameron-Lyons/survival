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
//!   [`SurvfitKMResult`] with [`SurvfitKMResult::from_stacked`] and call the
//!   ports.
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
