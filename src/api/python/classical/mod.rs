use super::*;

macro_rules! register_classes {
    ($m:expr, $( $class:ty ),+ $(,)?) => {
        $( $m.add_class::<$class>()?; )+
    };
}

/// The `ties` argument of the residual kernels: the C code's
/// `method == "efron"` flag, so only `"breslow"` and `"efron"` (an exact
/// fit's residuals come from `CoxPHFit`).
fn kernel_ties(ties: &str) -> PyResult<crate::regression::TieMethod> {
    match ties {
        "breslow" => Ok(crate::regression::TieMethod::Breslow),
        "efron" => Ok(crate::regression::TieMethod::Efron),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ties must be \"breslow\" or \"efron\", got {other:?}"
        ))),
    }
}

mod core;
mod data_prep;
mod datasets;
mod diagnostics;
mod evaluation;
mod monitoring;
mod population;
mod residuals;
mod survival_models;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core::register(m)?;
    diagnostics::register(m)?;
    survival_models::register(m)?;
    evaluation::register(m)?;
    data_prep::register(m)?;
    monitoring::register(m)?;
    population::register(m)?;
    datasets::register(m)?;
    residuals::register(m)?;
    Ok(())
}
