use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(g_computation, m)?)?;
    m.add_function(wrap_pyfunction!(g_computation_survival_curves, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ipcw_weights, m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_kaplan_meier, m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_treatment_effect, m)?)?;
    m.add_function(wrap_pyfunction!(marginal_structural_model, m)?)?;
    m.add_function(wrap_pyfunction!(compute_longitudinal_iptw, m)?)?;
    m.add_function(wrap_pyfunction!(target_trial_emulation, m)?)?;
    m.add_function(wrap_pyfunction!(sequential_trial_emulation, m)?)?;
    m.add_class::<GComputationResult>()?;
    m.add_class::<IPCWResult>()?;
    m.add_class::<MSMResult>()?;
    m.add_class::<TargetTrialResult>()?;

    m.add_function(wrap_pyfunction!(estimate_counterfactual_survival, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tv_survcaus, m)?)?;
    m.add_class::<CounterfactualSurvivalConfig>()?;
    m.add_class::<CounterfactualSurvivalResult>()?;
    m.add_class::<TVSurvCausConfig>()?;
    m.add_class::<TVSurvCausResult>()?;

    m.add_function(wrap_pyfunction!(tmle_ate, m)?)?;
    m.add_function(wrap_pyfunction!(tmle_survival, m)?)?;
    m.add_class::<TMLEConfig>()?;
    m.add_class::<TMLEResult>()?;
    m.add_class::<TMLESurvivalResult>()?;

    m.add_function(wrap_pyfunction!(causal_forest_survival, m)?)?;
    m.add_class::<CausalForestConfig>()?;
    m.add_class::<CausalForestSurvival>()?;
    m.add_class::<CausalForestResult>()?;

    m.add_function(wrap_pyfunction!(iv_cox, m)?)?;
    m.add_function(wrap_pyfunction!(rd_survival, m)?)?;
    m.add_function(wrap_pyfunction!(mediation_survival, m)?)?;
    m.add_function(wrap_pyfunction!(g_estimation_aft, m)?)?;
    m.add_class::<IVCoxConfig>()?;
    m.add_class::<IVCoxResult>()?;
    m.add_class::<RDSurvivalConfig>()?;
    m.add_class::<RDSurvivalResult>()?;
    m.add_class::<MediationSurvivalConfig>()?;
    m.add_class::<MediationSurvivalResult>()?;
    m.add_class::<GEstimationConfig>()?;
    m.add_class::<GEstimationResult>()?;

    m.add_function(wrap_pyfunction!(copula_censoring_model, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_bounds_survival, m)?)?;
    m.add_function(wrap_pyfunction!(mnar_sensitivity_survival, m)?)?;
    m.add_class::<CopulaType>()?;
    m.add_class::<CopulaCensoringConfig>()?;
    m.add_class::<CopulaCensoringResult>()?;
    m.add_class::<SensitivityBoundsConfig>()?;
    m.add_class::<SensitivityBoundsResult>()?;
    m.add_class::<MNARSurvivalConfig>()?;
    m.add_class::<MNARSurvivalResult>()?;

    m.add_function(wrap_pyfunction!(interval_censored_regression, m)?)?;
    m.add_function(wrap_pyfunction!(turnbull_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(npmle_interval, m)?)?;
    m.add_class::<IntervalCensoredResult>()?;
    m.add_class::<TurnbullResult>()?;
    m.add_class::<IntervalDistribution>()?;

    m.add_function(wrap_pyfunction!(joint_model, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_prediction, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_auc, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_brier_score, m)?)?;
    m.add_function(wrap_pyfunction!(landmarking_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_auc, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_c_index, m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_auc, m)?)?;
    m.add_function(wrap_pyfunction!(super_landmark_model, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_roc, m)?)?;
    m.add_class::<JointModelResult>()?;
    m.add_class::<DynamicPredictionResult>()?;
    m.add_class::<AssociationStructure>()?;
    m.add_class::<TimeVaryingAUCResult>()?;
    m.add_class::<DynamicCIndexResult>()?;
    m.add_class::<IPCWAUCResult>()?;
    m.add_class::<SuperLandmarkResult>()?;
    m.add_class::<TimeDependentROCResult>()?;

    m.add_function(wrap_pyfunction!(multiple_imputation_survival, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_missing_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(pattern_mixture_model, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(tipping_point_analysis, m)?)?;
    m.add_class::<MultipleImputationResult>()?;
    m.add_class::<PatternMixtureResult>()?;
    m.add_class::<ImputationMethod>()?;
    m.add_class::<SensitivityAnalysisType>()?;

    m.add_function(wrap_pyfunction!(double_ml_survival, m)?)?;
    m.add_function(wrap_pyfunction!(double_ml_cate, m)?)?;
    m.add_class::<DoubleMLConfig>()?;
    m.add_class::<DoubleMLResult>()?;
    m.add_class::<CATEResult>()?;

    Ok(())
}
