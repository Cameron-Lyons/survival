//! Residual kernels of R's `survival` package: martingale residuals for the
//! Cox model (`coxmart.c`, `agmart3.c`) and the `survreg` residuals.
//!
//! The Cox kernels share the covariate/strata conventions documented in
//! `crate::core::strata_order`: data sorted by stratum, integer stratum
//! labels, and the fitters' [`TieMethod`] in place of the C code's
//! `method == 1` flag.

pub mod agmart;
pub mod coxmart;
pub(crate) mod survreg_resid;

pub use crate::regression::TieMethod;
pub use agmart::agmart;
pub use coxmart::coxmart;
pub use survreg_resid::{SurvregResidType, SurvregResiduals, residuals_survreg};
