use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relative_survival, m)?)?;
    m.add_function(wrap_pyfunction!(net_survival, m)?)?;
    m.add_function(wrap_pyfunction!(crude_probability_of_death, m)?)?;
    m.add_function(wrap_pyfunction!(excess_hazard_regression, m)?)?;
    m.add_class::<RelativeSurvivalResult>()?;
    m.add_class::<NetSurvivalResult>()?;
    m.add_class::<ExcessHazardModelResult>()?;
    m.add_class::<NetSurvivalMethod>()?;

    m.add_function(wrap_pyfunction!(spatial_frailty_model, m)?)?;
    m.add_function(wrap_pyfunction!(compute_spatial_smoothed_rates, m)?)?;
    m.add_function(wrap_pyfunction!(moran_i_test, m)?)?;
    m.add_class::<SpatialFrailtyResult>()?;
    m.add_class::<SpatialCorrelationStructure>()?;

    m.add_function(wrap_pyfunction!(network_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(diffusion_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(network_heterogeneity_survival, m)?)?;
    m.add_class::<CentralityType>()?;
    m.add_class::<NetworkSurvivalConfig>()?;
    m.add_class::<NetworkSurvivalResult>()?;
    m.add_class::<DiffusionSurvivalConfig>()?;
    m.add_class::<DiffusionSurvivalResult>()?;
    m.add_class::<NetworkHeterogeneityResult>()?;

    Ok(())
}
