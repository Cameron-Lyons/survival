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
mod pyears3b;
mod concordance1;
mod concordance5;

use agfit5::perform_cox_regression_frailty;
use pyears3b::perform_pyears_calculation;
use concordance1::perform_concordance1_calculation;
use concordance5::perform_concordance_calculation;
use agscore2::perform_score_calculation;

#[pymodule]
fn survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perform_cox_regression_frailty, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pyears_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance1_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_score_calculation, &m)?)?;
    Ok(())
}
