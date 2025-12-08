use pyo3::prelude::*;

mod aareg;
mod agexact;
mod agfit4;
mod agfit5;
mod agmart;
mod agscore2;
mod agscore3;
mod agsurv4;
mod agsurv5;
mod cdecomp;
mod chinv2;
mod concordance1;
mod concordance3;
mod concordance5;
mod pspline;
mod pyears3b;
mod pystep;

use aareg::{AaregOptions, aareg as aareg_function};
use agfit5::perform_cox_regression_frailty;
use agscore2::perform_score_calculation;
use agscore3::perform_agscore3_calculation;
use concordance1::perform_concordance1_calculation;
use concordance3::perform_concordance3_calculation;
use concordance5::perform_concordance_calculation;
use pspline::PSpline;
use pyears3b::perform_pyears_calculation;
use pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};

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
    m.add_class::<AaregOptions>()?;
    m.add_class::<PSpline>()?;
    Ok(())
}
