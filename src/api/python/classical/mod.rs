use super::*;

macro_rules! register_classes {
    ($m:expr, $( $class:ty ),+ $(,)?) => {
        $( $m.add_class::<$class>()?; )+
    };
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
