use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distill_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(prune_survival_model, m)?)?;
    m.add_class::<DistillationConfig>()?;
    m.add_class::<DistilledSurvivalModel>()?;
    m.add_class::<DistillationResult>()?;
    m.add_class::<PruningResult>()?;

    m.add_function(wrap_pyfunction!(federated_cox, m)?)?;
    m.add_function(wrap_pyfunction!(secure_aggregate, m)?)?;
    m.add_class::<FederatedConfig>()?;
    m.add_class::<FederatedSurvivalResult>()?;
    m.add_class::<SecureAggregationResult>()?;
    m.add_class::<PrivacyAccountant>()?;

    m.add_class::<StreamingCoxConfig>()?;
    m.add_class::<StreamingCoxModel>()?;
    m.add_class::<StreamingKaplanMeier>()?;
    m.add_class::<ConceptDriftDetector>()?;

    m.add_function(wrap_pyfunction!(dp_kaplan_meier, m)?)?;
    m.add_function(wrap_pyfunction!(dp_cox_regression, m)?)?;
    m.add_function(wrap_pyfunction!(dp_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(local_dp_mean, m)?)?;
    m.add_class::<DPConfig>()?;
    m.add_class::<DPSurvivalResult>()?;
    m.add_class::<DPCoxResult>()?;
    m.add_class::<DPHistogramResult>()?;
    m.add_class::<LocalDPResult>()?;

    m.add_function(wrap_pyfunction!(super_learner_survival, m)?)?;
    m.add_function(wrap_pyfunction!(stacking_survival, m)?)?;
    m.add_function(wrap_pyfunction!(componentwise_boosting, m)?)?;
    m.add_function(wrap_pyfunction!(blending_survival, m)?)?;
    m.add_class::<SuperLearnerConfig>()?;
    m.add_class::<SuperLearnerResult>()?;
    m.add_class::<StackingConfig>()?;
    m.add_class::<StackingResult>()?;
    m.add_class::<ComponentwiseBoostingConfig>()?;
    m.add_class::<ComponentwiseBoostingResult>()?;
    m.add_class::<BlendingResult>()?;

    m.add_function(wrap_pyfunction!(active_learning_selection, m)?)?;
    m.add_function(wrap_pyfunction!(query_by_committee, m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_logrank, m)?)?;
    m.add_function(wrap_pyfunction!(power_logrank, m)?)?;
    m.add_function(wrap_pyfunction!(group_sequential_analysis, m)?)?;
    m.add_class::<ActiveLearningConfig>()?;
    m.add_class::<ActiveLearningResult>()?;
    m.add_class::<QBCResult>()?;
    m.add_class::<LogrankSampleSizeResult>()?;
    m.add_class::<LogrankPowerResult>()?;
    m.add_class::<AdaptiveDesignResult>()?;

    Ok(())
}
