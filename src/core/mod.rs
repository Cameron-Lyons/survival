pub(crate) mod coxcount1;
pub(crate) mod coxscho;
pub(crate) mod nsk;
pub(crate) mod pspline;

// Public facade exports
pub use coxcount1::{CoxCountOutput, coxcount1, coxcount2};
pub use coxscho::schoenfeld_residuals;
pub use nsk::{NaturalSplineKnot, SplineBasisResult, nsk};
pub use pspline::PSpline;
