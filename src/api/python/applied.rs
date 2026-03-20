use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decision_curve_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(clinical_utility_at_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(compare_decision_curves, m)?)?;
    m.add_class::<DecisionCurveResult>()?;
    m.add_class::<ClinicalUtilityResult>()?;

    m.add_function(wrap_pyfunction!(warranty_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(renewal_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_growth, m)?)?;
    m.add_class::<WarrantyConfig>()?;
    m.add_class::<WarrantyResult>()?;
    m.add_class::<RenewalResult>()?;
    m.add_class::<ReliabilityGrowthResult>()?;

    m.add_function(wrap_pyfunction!(distill_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(prune_survival_model, m)?)?;
    m.add_class::<DistillationConfig>()?;
    m.add_class::<DistilledSurvivalModel>()?;
    m.add_class::<DistillationResult>()?;
    m.add_class::<PruningResult>()?;
    m.add_class::<ModelComparisonResult>()?;

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

    m.add_function(wrap_pyfunction!(joint_longitudinal_model, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_cox_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(longitudinal_dynamic_pred, m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_cox, m)?)?;
    m.add_class::<JointModelConfig>()?;
    m.add_class::<JointLongSurvResult>()?;
    m.add_class::<LandmarkAnalysisResult>()?;
    m.add_class::<LongDynamicPredResult>()?;
    m.add_class::<TimeVaryingCoxResult>()?;

    m.add_function(wrap_pyfunction!(km_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(forest_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_survival_report, m)?)?;
    m.add_function(wrap_pyfunction!(roc_plot_data, m)?)?;
    m.add_class::<KaplanMeierPlotData>()?;
    m.add_class::<ForestPlotData>()?;
    m.add_class::<CalibrationCurveData>()?;
    m.add_class::<SurvivalReport>()?;
    m.add_class::<ROCPlotData>()?;

    m.add_function(wrap_pyfunction!(qaly_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(qaly_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(incremental_cost_effectiveness, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_sensitivity, m)?)?;
    m.add_class::<QALYResult>()?;
    m.add_class::<QTWISTResult>()?;

    Ok(())
}
