use super::*;
use crate::residuals::TieMethod;

/// `coxmart`: martingale residuals of a right-censored Cox model.
#[pyfunction(name = "coxmart")]
#[pyo3(signature = (input, ties="efron"))]
fn coxmart_py(input: &CoxMartInput, ties: &str) -> PyResult<Vec<f64>> {
    Ok(crate::residuals::coxmart(input, TieMethod::parse(ties)?)?)
}

/// `agmart3`: martingale residuals of a Cox model on (start, stop] data.
#[pyfunction(name = "agmart")]
#[pyo3(signature = (input, ties="efron"))]
fn agmart_py(input: &AndersenGillInput, ties: &str) -> PyResult<Vec<f64>> {
    Ok(crate::residuals::agmart(input, TieMethod::parse(ties)?)?)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coxmart_py, m)?)?;
    m.add_function(wrap_pyfunction!(agmart_py, m)?)?;

    register_classes!(
        m,
        SurvregResiduals,
        SurvregResidType,
        SurvregPrediction,
        SurvregPredictType,
    );

    Ok(())
}
