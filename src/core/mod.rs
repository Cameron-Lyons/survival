#[path = "coxcount1.rs"]
pub(crate) mod coxcount1_module;
pub(crate) mod coxscho;
#[path = "nsk.rs"]
pub(crate) mod nsk_module;
pub(crate) mod pspline;

// Public facade exports
pub use coxcount1_module::{CoxCountOutput, coxcount1, coxcount2};
pub use coxscho::schoenfeld_residuals;
pub use nsk_module::{NaturalSplineKnot, SplineBasisResult, nsk};
pub use pspline::PSpline;
