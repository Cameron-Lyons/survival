//! Ports of the small C routines and R helpers of `survival` that other
//! modules build on: risk-set expansion for `tt()` terms (`coxcount1.c`),
//! Schoenfeld residuals (`coxscho.c`), the `nsk` natural spline and the
//! `pspline` basis, plus the data conventions the kernels share
//! (`strata_order`).

pub mod bspline;
pub mod coxcount1;
pub mod coxscho;
pub mod natural_spline;
pub mod pspline;
pub mod strata_order;

pub use coxcount1::{CoxCountOutput, coxcount1, coxcount2};
pub use coxscho::{CoxschoResiduals, schoenfeld_residuals};
pub use natural_spline::{NaturalSplineKnot, SplineBasisResult, nsk, nsk_basis};
pub use pspline::{PsplineBasis, pspline_basis};
pub use strata_order::SurvResponse;
