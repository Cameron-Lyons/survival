use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survshap, m)?)?;
    m.add_function(wrap_pyfunction!(survshap_from_model, m)?)?;
    m.add_function(wrap_pyfunction!(survshap_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survshap, m)?)?;
    m.add_function(wrap_pyfunction!(permutation_importance, m)?)?;
    m.add_function(wrap_pyfunction!(compute_shap_interactions, m)?)?;
    m.add_class::<SurvShapConfig>()?;
    m.add_class::<SurvShapResult>()?;
    m.add_class::<SurvShapExplanation>()?;
    m.add_class::<AggregationMethod>()?;
    m.add_class::<BootstrapSurvShapResult>()?;
    m.add_class::<PermutationImportanceResult>()?;
    m.add_class::<ShapInteractionResult>()?;
    m.add_class::<FeatureImportance>()?;

    m.add_function(wrap_pyfunction!(detect_time_varying_features, m)?)?;
    m.add_class::<TimeVaryingTestConfig>()?;
    m.add_class::<TimeVaryingTestResult>()?;
    m.add_class::<TimeVaryingAnalysis>()?;
    m.add_class::<TimeVaryingTestType>()?;

    m.add_function(wrap_pyfunction!(detect_changepoints, m)?)?;
    m.add_function(wrap_pyfunction!(detect_changepoints_single_series, m)?)?;
    m.add_class::<ChangepointConfig>()?;
    m.add_class::<ChangepointResult>()?;
    m.add_class::<Changepoint>()?;
    m.add_class::<AllChangepointsResult>()?;
    m.add_class::<ChangepointMethod>()?;
    m.add_class::<CostFunction>()?;

    m.add_function(wrap_pyfunction!(group_variables, m)?)?;
    m.add_class::<VariableGroupingConfig>()?;
    m.add_class::<VariableGroupingResult>()?;
    m.add_class::<FeatureGroup>()?;
    m.add_class::<GroupingMethod>()?;
    m.add_class::<LinkageType>()?;

    m.add_function(wrap_pyfunction!(analyze_local_global, m)?)?;
    m.add_class::<LocalGlobalConfig>()?;
    m.add_class::<LocalGlobalResult>()?;
    m.add_class::<LocalGlobalSummary>()?;
    m.add_class::<FeatureViewAnalysis>()?;
    m.add_class::<ViewRecommendation>()?;

    m.add_function(wrap_pyfunction!(compute_ale, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ale_2d, m)?)?;
    m.add_function(wrap_pyfunction!(compute_time_varying_ale, m)?)?;
    m.add_class::<ALEResult>()?;
    m.add_class::<ALE2DResult>()?;

    m.add_function(wrap_pyfunction!(compute_friedman_h, m)?)?;
    m.add_function(wrap_pyfunction!(compute_all_pairwise_interactions, m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_feature_importance_decomposition,
        m
    )?)?;
    m.add_class::<FriedmanHResult>()?;
    m.add_class::<FriedmanFeatureImportanceResult>()?;

    m.add_function(wrap_pyfunction!(compute_ice, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dice, m)?)?;
    m.add_function(wrap_pyfunction!(compute_survival_ice, m)?)?;
    m.add_function(wrap_pyfunction!(detect_heterogeneity, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_ice_curves, m)?)?;
    m.add_class::<ICEResult>()?;
    m.add_class::<DICEResult>()?;

    Ok(())
}
