use pyo3::prelude::*;

mod regression {
    pub mod aareg;
    pub mod agfit4;
    pub mod agfit5;
}

mod surv_analysis {
    pub mod agsurv4;
    pub mod agsurv5;
}

mod concordance {
    pub mod concordance1;
    pub mod concordance3;
    pub mod concordance5;
}

mod matrix {
    pub mod cdecomp;
    pub mod chinv2;
}

mod scoring {
    pub mod agscore2;
    pub mod agscore3;
}

mod python {
    pub mod pyears3b;
    pub mod pystep;
}

mod core {
    pub mod pspline;
}

use concordance::concordance1::perform_concordance1_calculation;
use concordance::concordance3::perform_concordance3_calculation;
use concordance::concordance5::perform_concordance_calculation;
use core::pspline::PSpline;
use python::pyears3b::perform_pyears_calculation;
use python::pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};
use regression::aareg::{AaregOptions, aareg as aareg_function};
use regression::agfit5::perform_cox_regression_frailty;
use scoring::agscore2::perform_score_calculation;
use scoring::agscore3::perform_agscore3_calculation;

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
