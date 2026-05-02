pub(crate) mod anova;
pub(crate) mod bootstrap;
pub(crate) mod brier;
pub(crate) mod calibration;
pub(crate) mod cipoisson;
pub(crate) mod conformal;
pub(crate) mod crossval;
pub(crate) mod d_calibration;
pub(crate) mod decision_curve;
pub(crate) mod fairness;
pub(crate) mod hyperparameter;
pub(crate) mod hypothesis_tests;
pub(crate) mod landmark;
pub(crate) mod logrank;
pub(crate) mod meta_analysis;
pub(crate) mod model_selection;
pub(crate) mod power;
pub(crate) mod rcll;
pub(crate) mod reporting;
pub(crate) mod rmst;
pub(crate) mod royston;
pub(crate) mod survcheck;
pub(crate) mod survobrien;
pub(crate) mod time_dependent_auc;
pub(crate) mod uncertainty;
pub(crate) mod uno_c_index;
pub(crate) mod yates;

// Public facade exports
pub use anova::{AnovaCoxphResult, AnovaRow, anova_coxph, anova_coxph_single};
pub use bootstrap::{BootstrapResult, bootstrap_cox_ci, bootstrap_survreg_ci};
pub use brier::{brier, compute_brier, integrated_brier};
pub use calibration::{
    AdvancedCalibrationResult, CalibrationResult, PredictionResult, RiskStratificationResult,
    TdAUCResult, TimeDependentCalibrationResult, advanced_calibration_metrics, calibration,
    predict_cox, risk_stratification, td_auc, time_dependent_calibration,
};
pub use cipoisson::{cipoisson, cipoisson_anscombe, cipoisson_exact};
pub use conformal::{
    BootstrapConformalResult, CQRConformalResult, CVPlusCalibrationResult, CVPlusConformalResult,
    ConformalCalibrationPlot, ConformalCalibrationResult, ConformalDiagnostics,
    ConformalPredictionResult, ConformalSurvivalDistribution, ConformalWidthAnalysis,
    CovariateShiftConformalResult, CoverageSelectionResult, DoublyRobustConformalResult,
    MondrianCalibrationResult, MondrianConformalResult, MondrianDiagnostics,
    TwoSidedCalibrationResult, TwoSidedConformalResult, WeightDiagnostics,
    bootstrap_conformal_survival, conformal_calibrate, conformal_calibration_plot,
    conformal_coverage_cv, conformal_coverage_test, conformal_predict,
    conformal_survival_from_predictions, conformal_survival_parallel, conformal_width_analysis,
    conformalized_survival_distribution, covariate_shift_conformal_survival,
    cqr_conformal_survival, cvplus_conformal_calibrate, cvplus_conformal_survival,
    doubly_robust_conformal_calibrate, doubly_robust_conformal_survival,
    mondrian_conformal_calibrate, mondrian_conformal_predict, mondrian_conformal_survival,
    two_sided_conformal_calibrate, two_sided_conformal_predict, two_sided_conformal_survival,
};
pub use crossval::{CVResult, cv_cox_concordance, cv_survreg_loglik};
pub use d_calibration::{
    BrierCalibrationResult, CalibrationPlotData, DCalibrationResult, MultiTimeCalibrationResult,
    OneCalibrationResult, SmoothedCalibrationCurve, brier_calibration, calibration_plot,
    d_calibration, multi_time_calibration, one_calibration, smoothed_calibration,
};
pub use decision_curve::{
    ClinicalUtilityResult, DecisionCurveResult, ModelComparisonResult,
    clinical_utility_at_threshold, compare_decision_curves, decision_curve_analysis,
};
pub use fairness::{
    FairnessMetrics, RobustnessResult, SubgroupAnalysisResult, assess_model_robustness,
    compute_fairness_metrics, subgroup_analysis,
};
pub use hyperparameter::{
    BenchmarkResult, HyperparameterResult, HyperparameterSearchConfig, NestedCVResult,
    SearchStrategy, benchmark_models, hyperparameter_search, nested_cross_validation,
};
pub use hypothesis_tests::{
    ProportionalityTest, TestResult, lrt_test, ph_test, score_test, wald_test,
};
pub use landmark::{
    ConditionalSurvivalResult, HazardRatioResult, LandmarkResult, LifeTableResult,
    SurvivalAtTimeResult, conditional_survival, hazard_ratio, landmark_analysis,
    landmark_analysis_batch, life_table, survival_at_times,
};
pub use logrank::{
    LogRankResult, TrendTestResult, WeightType, fleming_harrington_test, logrank_test,
    logrank_trend, weighted_logrank_test,
};
pub use meta_analysis::{
    MetaAnalysisConfig, MetaAnalysisResult, MetaForestPlotData, PublicationBiasResult,
    generate_forest_plot_data, publication_bias_tests, survival_meta_analysis,
};
pub use model_selection::{
    CrossValidatedScore, ModelSelectionCriteria, SurvivalModelComparison, compare_models,
    compute_cv_score, compute_model_selection_criteria,
};
pub use power::{
    AccrualResult, SampleSizeResult, expected_events, power_survival, sample_size_survival,
    sample_size_survival_freedman,
};
pub use rcll::{RCLLResult, compute_rcll, compute_rcll_single_time, rcll, rcll_single_time};
pub use reporting::{
    CalibrationCurveData, ForestPlotData, KaplanMeierPlotData, ROCPlotData, SurvivalReport,
    calibration_plot_data, forest_plot_data, generate_survival_report, km_plot_data, roc_plot_data,
};
pub use rmst::{
    ChangepointInfo, CumulativeIncidenceResult, MedianSurvivalResult, NNTResult,
    RMSTComparisonResult, RMSTOptimalThresholdResult, RMSTResult, compute_rmst,
    cumulative_incidence, number_needed_to_treat, rmst, rmst_comparison, rmst_optimal_threshold,
    survival_quantile,
};
pub use royston::{RoystonResult, royston, royston_from_model};
pub use survcheck::{SurvCheckResult, survcheck, survcheck_simple};
pub use survobrien::{SurvObrienResult, survobrien};
pub use time_dependent_auc::{
    CumulativeDynamicAUCResult, TimeDepAUCResult, cumulative_dynamic_auc,
    cumulative_dynamic_auc_core, time_dependent_auc, time_dependent_auc_core,
};
pub use uncertainty::{
    BayesianBootstrapConfig, BayesianBootstrapResult, CalibrationUncertaintyResult,
    ConformalSurvivalConfig, ConformalSurvivalResult, EnsembleUncertaintyResult,
    JackknifePlusConfig, JackknifePlusResult, MCDropoutConfig, QuantileRegressionResult,
    UncertaintyResult, bayesian_bootstrap_survival, calibrate_prediction_intervals,
    conformal_survival, ensemble_uncertainty, jackknife_plus_survival, mc_dropout_uncertainty,
    quantile_regression_intervals,
};
pub use uno_c_index::{
    CIndexDecompositionResult, ConcordanceComparisonResult, GonenHellerResult, UnoCIndexResult,
    c_index_decomposition, compare_uno_c_indices, gonen_heller_concordance, uno_c_index,
};
pub use yates::{YatesPairwiseResult, YatesResult, yates, yates_contrast, yates_pairwise};
