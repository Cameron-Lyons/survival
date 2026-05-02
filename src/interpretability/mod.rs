pub(crate) mod ale_plots;
pub(crate) mod changepoints;
pub(crate) mod friedman_h;
pub(crate) mod ice_curves;
pub(crate) mod local_global;
#[path = "survshap/mod.rs"]
pub(crate) mod survshap_module;
pub(crate) mod time_varying;
pub(crate) mod variable_groups;

// Public facade exports
pub use ale_plots::{
    ALE2DResult, ALEResult, compute_ale, compute_ale_2d, compute_time_varying_ale,
};
pub use changepoints::{
    AllChangepointsResult, Changepoint, ChangepointConfig, ChangepointMethod, ChangepointResult,
    CostFunction, detect_changepoints, detect_changepoints_single_series,
};
pub use friedman_h::{
    FeatureImportanceResult, FeatureImportanceResult as FriedmanFeatureImportanceResult,
    FriedmanHResult, compute_all_pairwise_interactions, compute_feature_importance_decomposition,
    compute_friedman_h,
};
pub use ice_curves::{
    DICEResult, ICEResult, cluster_ice_curves, compute_dice, compute_ice, compute_survival_ice,
    detect_heterogeneity,
};
pub use local_global::{
    FeatureViewAnalysis, LocalGlobalConfig, LocalGlobalResult, LocalGlobalSummary,
    ViewRecommendation, analyze_local_global,
};
pub use survshap_module::{
    AggregationMethod, BootstrapSurvShapResult, FeatureImportance, PermutationImportanceResult,
    ShapInteractionResult, SurvShapConfig, SurvShapExplanation, SurvShapResult, aggregate_survshap,
    compute_shap_interactions, permutation_importance, survshap, survshap_bootstrap,
    survshap_from_model,
};
pub use time_varying::{
    TimeVaryingAnalysis, TimeVaryingTestConfig, TimeVaryingTestResult, TimeVaryingTestType,
    detect_time_varying_features,
};
pub use variable_groups::{
    FeatureGroup, GroupingMethod, LinkageType, VariableGroupingConfig, VariableGroupingResult,
    group_variables,
};
