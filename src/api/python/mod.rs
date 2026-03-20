use pyo3::prelude::*;

use crate::pybridge::cox_py_callback::cox_callback;
use crate::pybridge::pyears3b::perform_pyears_calculation;
use crate::pybridge::pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};
use crate::validation::hypothesis_tests::{score_test_py, wald_test_py};
use crate::*;

mod applied;
mod bayesian;
mod causal;
mod classical;
mod conformal;
mod functional_robustness;
mod interpretability;
mod ml;
mod multistate;
mod recurrent_regression;
mod relative_spatial;

pub(crate) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    classical::register(m)?;
    bayesian::register(m)?;
    causal::register(m)?;
    ml::register(m)?;
    applied::register(m)?;
    recurrent_regression::register(m)?;
    multistate::register(m)?;
    relative_spatial::register(m)?;
    interpretability::register(m)?;
    conformal::register(m)?;
    functional_robustness::register(m)?;
    Ok(())
}
