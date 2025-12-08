use pyo3::prelude::*;

mod concordance;
mod core;
mod matrix;
mod python;
mod regression;
mod residuals;
mod scoring;
mod specialized;
mod surv_analysis;
mod tests;
mod utilities;

use concordance::concordance::concordance as concordance_fn;
use concordance::concordance1::perform_concordance1_calculation;
use concordance::concordance3::perform_concordance3_calculation;
use concordance::concordance5::perform_concordance_calculation;
use core::coxcount1::{CoxCountOutput, coxcount1, coxcount2};
use core::pspline::PSpline;
use python::cox_py_callback::cox_callback;
use python::pyears3b::perform_pyears_calculation;
use python::pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};
use regression::aareg::{AaregOptions, aareg as aareg_function};
use regression::agfit5::perform_cox_regression_frailty;
use regression::blogit::LinkFunctionParams;
use residuals::agmart::agmart;
use scoring::agscore2::perform_score_calculation;
use scoring::agscore3::perform_agscore3_calculation;
use specialized::cipoisson::{cipoisson, cipoisson_anscombe, cipoisson_exact};
use specialized::norisk::norisk;
use surv_analysis::agsurv4::agsurv4;
use surv_analysis::agsurv5::agsurv5;
use utilities::agexact::agexact;
use utilities::collapse::collapse;

#[pymodule]
fn survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perform_cox_regression_frailty, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pyears_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance1_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance3_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_score_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_agscore3_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_simple_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(aareg_function, &m)?)?;
    m.add_function(wrap_pyfunction!(collapse, &m)?)?;
    m.add_function(wrap_pyfunction!(cox_callback, &m)?)?;
    m.add_function(wrap_pyfunction!(coxcount1, &m)?)?;
    m.add_function(wrap_pyfunction!(coxcount2, &m)?)?;
    m.add_function(wrap_pyfunction!(norisk, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_exact, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_anscombe, &m)?)?;
    m.add_function(wrap_pyfunction!(concordance_fn, &m)?)?;
    m.add_function(wrap_pyfunction!(agexact, &m)?)?;
    m.add_function(wrap_pyfunction!(agsurv4, &m)?)?;
    m.add_function(wrap_pyfunction!(agsurv5, &m)?)?;
    m.add_function(wrap_pyfunction!(agmart, &m)?)?;
    m.add_class::<AaregOptions>()?;
    m.add_class::<PSpline>()?;
    m.add_class::<CoxCountOutput>()?;
    m.add_class::<LinkFunctionParams>()?;
    Ok(())
}
