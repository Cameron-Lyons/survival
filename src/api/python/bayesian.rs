use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bayesian_cox, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_cox_predict_survival, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric_predict, m)?)?;
    m.add_class::<BayesianCoxResult>()?;
    m.add_class::<BayesianParametricResult>()?;

    m.add_function(wrap_pyfunction!(dirichlet_process_survival, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_model_averaging_cox, m)?)?;
    m.add_function(wrap_pyfunction!(spike_slab_cox, m)?)?;
    m.add_function(wrap_pyfunction!(horseshoe_cox, m)?)?;
    m.add_class::<DirichletProcessConfig>()?;
    m.add_class::<DirichletProcessResult>()?;
    m.add_class::<BayesianModelAveragingConfig>()?;
    m.add_class::<BayesianModelAveragingResult>()?;
    m.add_class::<SpikeSlabConfig>()?;
    m.add_class::<SpikeSlabResult>()?;
    m.add_class::<HorseshoeConfig>()?;
    m.add_class::<HorseshoeResult>()?;

    Ok(())
}
