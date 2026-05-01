use pyo3::prelude::*;

use crate::pybridge::cox_py_callback::cox_callback;
use crate::pybridge::pyears3b::perform_pyears_calculation;
use crate::pybridge::pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};
use crate::validation::hypothesis_tests::{score_test_py, wald_test_py};
use crate::*;

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
