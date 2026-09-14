//! Residual kernels of R's `survival` package: martingale residuals for the
//! Cox model (`coxmart.c`, `agmart3.c`) and the `survreg` residuals.
//!
//! The Cox kernels share the covariate/strata conventions documented in
//! `crate::core::strata_order`: data sorted by stratum, integer stratum labels,
//! and a `TieMethod` in place of the C code's `method == 1` flag.

pub mod agmart;
pub mod coxmart;
pub(crate) mod survreg_resid;

pub use agmart::agmart;
pub use coxmart::coxmart;
pub use survreg_resid::{SurvregResidType, SurvregResiduals, residuals_survreg};

use crate::error::{SurvivalError, SurvivalResult};

/// Handling of tied event times in the partial likelihood, R's
/// `ties = "breslow"` / `"efron"` (the C kernels receive it as
/// `method = as.integer(method == "efron")`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieMethod {
    Breslow,
    Efron,
}

impl TieMethod {
    /// Parses R's `ties` argument; the exact method has no closed-form
    /// residual kernels in R either.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "breslow" => Ok(Self::Breslow),
            "efron" => Ok(Self::Efron),
            other => Err(SurvivalError::invalid_input(format!(
                "ties must be \"breslow\" or \"efron\", got {other:?}"
            ))),
        }
    }

    /// `1` for Efron, `0` otherwise: the `method` integer of the C sources.
    pub(crate) fn efron_flag(self) -> f64 {
        match self {
            Self::Breslow => 0.0,
            Self::Efron => 1.0,
        }
    }
}
