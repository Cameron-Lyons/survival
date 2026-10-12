#![cfg_attr(not(feature = "python"), allow(dead_code, unused_imports))]

#[cfg(not(feature = "python"))]
extern crate self as pyo3;

#[cfg(not(feature = "python"))]
mod pyo3_shim;

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

pub mod data_types {
    pub use crate::internal::numpy_utils::{
        BoolVec, FloatArray3, FloatMatrix, FloatRows, FloatVec, IntVec,
    };
    pub use crate::internal::typed_inputs::{
        AndersenGillInput, CountingProcessData, CovariateMatrix, CoxMartInput, CoxRegressionInput,
        SurvivalData, Weights,
    };
}

pub mod prelude {
    pub use crate::data_types::*;
    pub use crate::error::{SurvivalError, SurvivalResult};
    #[cfg(feature = "ml")]
    pub use crate::ml;
    #[cfg(not(feature = "python"))]
    #[doc(hidden)]
    pub use crate::pyo3_shim::prelude::*;
    pub use crate::{
        bayesian, causal, concordance, data_prep, interpretability, interval, joint, missing,
        monitoring, population, qol, recurrent, regression, relative, reliability, residuals,
        scoring, spatial, surv_analysis, validation,
    };
}

#[cfg(feature = "python")]
#[pymodule]
fn _survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    api::python::register_module(&m)
}
