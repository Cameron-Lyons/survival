//! Rust port of R's `survival` package with PyO3 bindings.
//!
//! The crate root exposes the domain modules (`regression`, `surv_analysis`,
//! ...), the [`error`] types, the typed boundary inputs in [`data_types`] and a
//! [`prelude`] that gathers all three. Python bindings live under `api` and are
//! compiled only with the `python` feature; `docs/repo-layout.md` describes the
//! layout.

// In Rust-only builds the PyO3 attribute macros are no-ops, so `#[new]`
// constructors, `#[pymethods]` and `#[pyo3(get)]` fields are never reached
// (`dead_code`), and a constant that only appears in a stripped
// `#[pyo3(signature = (.. = CONST))]` default is an unused import
// (`unused_imports`). Nothing else is allowed crate-wide.
#![cfg_attr(not(feature = "python"), allow(dead_code, unused_imports))]
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(not(feature = "python"))]
extern crate self as pyo3;

#[cfg(not(feature = "python"))]
mod pyo3_shim;

// `pyo3::X` resolves to `crate::X` in Rust-only builds (see `pyo3_shim`), so
// the shim's stand-ins must sit at the crate root. They are not part of the
// supported API and are hidden from the docs.
#[cfg(not(feature = "python"))]
#[doc(hidden)]
pub use pyo3_shim::{
    Bound, Py, PyAny, PyDict, PyErr, PyErrKind, PyRefMut, PyResult, Python, exceptions, types,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

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
pub mod reliability;
pub mod residuals;
pub mod scoring;
pub mod simd_ops;
pub mod spatial;
pub mod surv_analysis;
#[cfg(test)]
mod tests;
pub mod validation;

pub use error::{SurvivalError, SurvivalResult};

/// Typed inputs accepted at the Rust and Python boundaries.
///
/// The `*Input`/`*Data` structs validate shapes once, up front; `FloatVec`,
/// `IntVec`, `BoolVec` and `FloatMatrix` are the `#[pyfunction]` argument
/// types that accept NumPy arrays, pandas/polars columns or plain sequences
/// without a `.tolist()` round trip.
pub mod data_types {
    pub use crate::internal::numpy_utils::{BoolVec, FloatMatrix, FloatVec, IntVec};
    pub use crate::internal::typed_inputs::{
        AndersenGillInput, CountingProcessData, CovariateMatrix, CoxMartInput, CoxRegressionInput,
        SurvivalData, Weights,
    };
}

/// The domain modules, the error types and the typed inputs, identical with
/// and without the `python` feature.
///
/// [`core`] is deliberately absent: a glob import of a module named `core`
/// shadows the `core` crate for the importing file, which breaks derive
/// macros that spell out `core::fmt::...` paths. Reach it as `survival::core`.
pub mod prelude {
    pub use crate::data_types::*;
    pub use crate::error::{SurvivalError, SurvivalResult};
    #[cfg(feature = "ml")]
    pub use crate::ml;
    pub use crate::{
        bayesian, causal, concordance, data_prep, interpretability, interval, joint, missing,
        monitoring, population, qol, recurrent, regression, relative, reliability, residuals,
        scoring, spatial, surv_analysis, validation,
    };
    // `use pyo3::prelude::*` inside the crate resolves here in Rust-only
    // builds, so the shim's stand-ins ride along (hidden, unsupported).
    #[cfg(not(feature = "python"))]
    #[doc(hidden)]
    pub use crate::pyo3_shim::prelude::*;
}

#[cfg(feature = "python")]
#[pymodule]
fn _survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    api::python::register_module(&m)
}
