use pyo3::prelude::*;

// Registration files resolve every binding through its domain's `pub use`
// surface (`crate::<domain>::Name`), so a symbol only registers once its
// domain module re-exports it. The `#[pyfunction]`s that exist purely as
// Python entry points live in `pybridge` and `validation::hypothesis_tests`
// and are named explicitly; `classical::evaluation` imports the
// `concordance` and `scoring` kernels it wraps by name.
use crate::data_types::*;
use crate::interval::interval_censoring::CensorType;
#[cfg(feature = "ml")]
use crate::ml::*;
use crate::pybridge::brier::brier_py;
use crate::pybridge::cox_py_callback::{CoxPenaltyTerms, cox_callback};
use crate::validation::hypothesis_tests::{lrt_test_py, score_test_py, wald_test_py};
use crate::{
    bayesian::*, causal::*, core::*, data_prep::*, interpretability::*, interval::*, joint::*,
    missing::*, monitoring::*, population::*, qol::*, recurrent::*, regression::*, relative::*,
    reliability::*, residuals::*, spatial::*, surv_analysis::*, validation::*,
};

#[cfg(feature = "ml")]
mod applied;
mod bayesian;
mod causal;
mod classical;
mod conformal;
#[cfg(feature = "ml")]
mod functional_robustness;
mod interpretability;
#[cfg(feature = "ml")]
mod ml;
mod multistate;
mod recurrent_regression;
mod relative_spatial;

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SurvivalData>()?;
    m.add_class::<CovariateMatrix>()?;
    m.add_class::<Weights>()?;
    m.add_class::<CountingProcessData>()?;
    m.add_class::<CoxRegressionInput>()?;
    m.add_class::<CoxMartInput>()?;
    m.add_class::<AndersenGillInput>()?;
    // Declared as `#[pyclass]` outside any domain registration file; the
    // registration audit (`api::registration_audit`) requires it here until
    // its owner either registers it alongside its siblings or drops the
    // attribute.
    m.add_class::<CensorType>()?;

    classical::register(m)?;
    bayesian::register(m)?;
    causal::register(m)?;
    #[cfg(feature = "ml")]
    ml::register(m)?;
    #[cfg(feature = "ml")]
    applied::register(m)?;
    recurrent_regression::register(m)?;
    multistate::register(m)?;
    relative_spatial::register(m)?;
    interpretability::register(m)?;
    conformal::register(m)?;
    #[cfg(feature = "ml")]
    functional_robustness::register(m)?;
    Ok(())
}
