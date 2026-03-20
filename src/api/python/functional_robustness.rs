use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fpca_survival, m)?)?;
    m.add_function(wrap_pyfunction!(functional_cox, m)?)?;
    m.add_class::<BasisType>()?;
    m.add_class::<FunctionalSurvivalConfig>()?;
    m.add_class::<FunctionalPCAResult>()?;
    m.add_class::<FunctionalSurvivalResult>()?;

    m.add_function(wrap_pyfunction!(dro_survival, m)?)?;
    m.add_function(wrap_pyfunction!(robustness_analysis, m)?)?;
    m.add_class::<UncertaintySet>()?;
    m.add_class::<DROSurvivalConfig>()?;
    m.add_class::<DROSurvivalResult>()?;
    m.add_class::<RobustnessAnalysis>()?;

    m.add_function(wrap_pyfunction!(generate_adversarial_examples, m)?)?;
    m.add_function(wrap_pyfunction!(adversarial_training_survival, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_robustness, m)?)?;
    m.add_class::<AttackType>()?;
    m.add_class::<DefenseType>()?;
    m.add_class::<AdversarialAttackConfig>()?;
    m.add_class::<AdversarialDefenseConfig>()?;
    m.add_class::<AdversarialExample>()?;
    m.add_class::<AdversarialAttackResult>()?;
    m.add_class::<RobustSurvivalModel>()?;
    m.add_class::<RobustnessEvaluation>()?;

    m.add_function(wrap_pyfunction!(get_available_devices, m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_cox_regression, m)?)?;
    m.add_function(wrap_pyfunction!(batch_predict_survival, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_matrix_operations, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_compute_backend, m)?)?;
    m.add_class::<ComputeBackend>()?;
    m.add_class::<GPUConfig>()?;
    m.add_class::<DeviceInfo>()?;
    m.add_class::<ParallelCoxResult>()?;
    m.add_class::<BatchPredictionResult>()?;

    Ok(())
}
