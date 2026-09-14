//! Summaries of survival curves: restricted mean survival time and the
//! survival quantiles of R's `survfit` methods.
//!
//! * [`survmean`] (`survmean.rs`) ports `survmean` from `R/print.survfit.R`,
//!   the table behind `print`/`summary(survfit)`: records, events, restricted
//!   mean, its standard error, median and its confidence limits; it also
//!   compares the restricted means of several groups.
//! * [`quantile_survfit`] (`quantiles.rs`) ports `R/quantile.survfit.R`.
//! * `threshold.rs` keeps the changepoint search for a restricted-mean
//!   horizon and `nnt.rs` the number needed to treat; neither has an R
//!   counterpart.
//!
//! Kaplan-Meier curves are always taken from `surv_analysis::survfitkm`.

mod nnt;
mod quantiles;
mod survmean;
mod threshold;

pub use nnt::{NNTResult, number_needed_to_treat, number_needed_to_treat_py};
pub use quantiles::{SurvfitCurveQuantiles, quantile_survfit, quantile_survfit_curves_py};
pub use survmean::{
    RmeanOption, RmstComparisonResult, RmstGroupResult, SurvfitSummaryRow, rmst_comparison,
    rmst_comparison_py, survmean, survmean_curves_py,
};
pub use threshold::{
    ChangepointInfo, RMSTOptimalThresholdResult, rmst_optimal_threshold, rmst_optimal_threshold_py,
};

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{validate_binary_i32, validate_finite, validate_length};
use crate::surv_analysis::{SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm};

/// One survival curve (a stratum of a `survfit` object) as the summaries
/// need it: times in increasing order with the survival, numbers at risk
/// and of events, and optionally the confidence limits.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvfitCurve<'a> {
    pub time: &'a [f64],
    pub surv: &'a [f64],
    pub n_risk: &'a [f64],
    pub n_event: &'a [f64],
    pub lower: Option<&'a [f64]>,
    pub upper: Option<&'a [f64]>,
}

impl<'a> SurvfitCurve<'a> {
    /// View a single-curve `survfit` result as a curve.  Confidence
    /// limits are taken from the fit when it has them (`conf_type !=
    /// "none"` and `se_fit`).
    pub fn from_km(fit: &'a SurvfitKMResult) -> Self {
        Self {
            time: &fit.time,
            surv: &fit.surv,
            n_risk: &fit.n_risk,
            n_event: &fit.n_event,
            lower: fit.lower.as_deref(),
            upper: fit.upper.as_deref(),
        }
    }

    /// One curve of a (possibly stratified) `survfit` result.
    pub fn from_km_curve(fit: &'a SurvfitKMResult, range: std::ops::Range<usize>) -> Self {
        Self {
            time: &fit.time[range.clone()],
            surv: &fit.surv[range.clone()],
            n_risk: &fit.n_risk[range.clone()],
            n_event: &fit.n_event[range.clone()],
            lower: fit.lower.as_deref().map(|v| &v[range.clone()]),
            upper: fit.upper.as_deref().map(|v| &v[range]),
        }
    }

    fn validate(&self) -> SurvivalResult<()> {
        let n = self.time.len();
        validate_length(n, self.surv.len(), "surv")?;
        validate_length(n, self.n_risk.len(), "n_risk")?;
        validate_length(n, self.n_event.len(), "n_event")?;
        if let Some(lower) = self.lower {
            validate_length(n, lower.len(), "lower")?;
        }
        if let Some(upper) = self.upper {
            validate_length(n, upper.len(), "upper")?;
        }
        if self.lower.is_some() != self.upper.is_some() {
            return Err(SurvivalError::invalid_input(
                "lower and upper limits must be given together",
            ));
        }
        validate_finite(self.time, "time")?;
        if self.time.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(SurvivalError::invalid_input(
                "curve times must be non-decreasing",
            ));
        }
        Ok(())
    }
}

/// Kaplan-Meier curve of one group of right-censored observations, with
/// log-scale confidence limits at `conf_level` (`survfit(Surv(time,
/// status) ~ 1, weights)`, so the variance is R's: Greenwood for integer
/// weights, the infinitesimal jackknife otherwise).
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
