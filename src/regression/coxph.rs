//! The Cox proportional hazards model: R survival's `coxph` object.
//!
//! [`CoxPHFit::fit`] is the driver behind R's `coxph()`: it dispatches to the
//! Newton-Raphson engine (`cox_optimizer`, ports of `coxfit6.c`, `agfit4.c`,
//! `coxexact.c` and `agexact.c`), then performs the post-processing of
//! `R/coxph.fit.R`, `R/agreg.fit.R`, `R/coxexact.fit.R`, `R/agexact.fit.R`
//! and `R/coxph.R`: centred linear predictors, martingale residuals, the
//! robust (cluster sandwich) variance, the Wald test and the concordance
//! of the linear predictors.
//!
//! The fitted object keeps the data it was fitted to, so the methods R
//! reconstructs from the model frame are plain method calls here:
//! `basehaz()`, `survfit()` (`R/survfit.coxph.R` on top of
//! `surv_analysis::agsurv`), `predict()` (`R/predict.coxph.R`) and the
//! residual types of `R/residuals.coxph.R` (`coxph_diagnostics`).  The
//! per-stratum baseline curves are computed once and cached.

mod bindings;
mod curves;
mod fitting;
mod prediction;
mod types;

pub use bindings::coxph_fit;
pub use types::{
    Basehaz, CoxNewData, CoxPHFit, CoxPrediction, CoxSurvfitCurve, CoxTermsPrediction, CoxphData,
    CoxphOptions, PredictReference, SurvfitOptions,
};

pub(crate) use bindings::newdata_from_python;
#[cfg(test)]
pub(crate) use fitting::linear_predictor_concordance;
pub(crate) use fitting::{
    FittedCox, add_offset_mean, centre_offset, fit_cox_engine, nocenter_columns,
};
pub(crate) use prediction::{default_assign, validate_assign};
pub(crate) use types::SortedRows;

#[cfg(test)]
mod input_tests;
#[cfg(test)]
mod tests;
