#![cfg_attr(
    not(feature = "python"),
    // The non-Python build compiles PyO3-facing constructors and methods through
    // a local shim, so rustc cannot see the dynamic Python entry points.
    allow(
        dead_code,
        unused_attributes,
        unused_imports,
        unused_mut,
        unused_variables
    )
)]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(not(feature = "python"))]
extern crate self as pyo3;

#[cfg(not(feature = "python"))]
mod pyo3_shim;

#[cfg(not(feature = "python"))]
pub use pyo3_shim::{
    Bound, Py, PyAny, PyDict, PyErr, PyList, PyModule, PyRefMut, PyResult, Python, exceptions,
    types,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
mod api;
pub mod bayesian;
pub mod causal;
pub mod concordance;
pub mod constants;
pub mod core;
pub mod data_prep;
#[cfg(feature = "python")]
mod datasets;
pub mod error;
mod internal;
pub mod interpretability;
pub mod interval;
pub mod joint;
mod matrix;
pub mod missing;
#[cfg(feature = "ml")]
pub mod ml;
pub mod monitoring;
pub mod population;
#[cfg(feature = "python")]
mod pybridge;
pub mod qol;
pub mod recurrent;
pub mod regression;
pub mod relative;
#[path = "reliability/mod.rs"]
pub mod reliability;
pub mod residuals;
pub mod scoring;
pub mod simd_ops;
pub mod spatial;
pub mod surv_analysis;
mod tests;
pub mod validation;

/// Public input and data container types shared across survival analysis domains.
pub mod data_types {
    pub use crate::internal::typed_inputs::{
        AndersenGillInput, CountingProcessData, CovariateMatrix, CoxMartInput, CoxRegressionInput,
        SurvivalData, Weights,
    };
}

/// Curated imports for new Rust code.
///
/// Prefer domain modules from this module over root-level compatibility exports.
pub mod preferred {
    pub use crate::data_types::{
        AndersenGillInput, CountingProcessData, CovariateMatrix, CoxMartInput, CoxRegressionInput,
        SurvivalData, Weights,
    };
    pub use crate::error::{SurvivalError, SurvivalResult};
    #[cfg(feature = "ml")]
    pub use crate::ml;
    pub use crate::{
        bayesian, causal, concordance, core, data_prep, interpretability, interval, joint, missing,
        monitoring, population, qol, recurrent, regression, relative, reliability, residuals,
        scoring, spatial, surv_analysis, validation,
    };
}

/// PyO3-compatible prelude for internal Python-facing modules.
///
/// New Rust callers should prefer `survival::preferred` or explicit domain
/// modules over this compatibility-oriented prelude.
pub mod prelude {
    #[cfg(feature = "python")]
    pub use crate::preferred::*;
    #[cfg(not(feature = "python"))]
    pub use crate::pyo3_shim::prelude::*;
}

/// Root-level compatibility exports are retained for old callers but are no
/// longer the preferred API shape.
pub const DEPRECATED_ROOT_EXPORTS_NOTE: &str =
    "Prefer survival::preferred or explicit domain modules such as survival::regression.";

/// Backwards-compatible root-level exports.
///
/// Prefer importing through the public domain modules, for example
/// `survival::regression::coxph` or `survival::validation::brier`.
#[doc(hidden)]
pub mod compatibility {
    pub use crate::bayesian::*;
    pub use crate::causal::*;
    pub use crate::concordance::*;
    pub use crate::constants::*;
    pub use crate::core::*;
    pub use crate::data_prep::*;
    pub use crate::data_types::*;
    pub use crate::error::{SurvivalError, SurvivalResult};
    pub use crate::interpretability::*;
    pub use crate::interval::*;
    pub use crate::joint::*;
    pub use crate::missing::*;
    #[cfg(feature = "ml")]
    pub use crate::ml::*;
    pub use crate::monitoring::*;
    pub use crate::population::*;
    pub use crate::qol::*;
    pub use crate::recurrent::*;
    pub use crate::regression::*;
    pub use crate::relative::*;
    pub use crate::reliability::*;
    pub use crate::residuals::*;
    pub use crate::scoring::*;
    pub use crate::spatial::*;
    pub use crate::surv_analysis::*;
    pub use crate::validation::*;
}

#[deprecated(
    since = "1.2.17",
    note = "Prefer survival::preferred or explicit domain modules such as survival::regression"
)]
pub use compatibility::*;

#[cfg(feature = "python")]
#[pymodule]
fn _survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    api::python::register_module(&m)
}
