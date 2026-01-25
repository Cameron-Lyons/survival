use pyo3::prelude::*;
mod bayesian;
mod causal;
mod concordance;
mod constants;
mod core;
mod datasets;
mod interpretability;
mod interval;
mod joint;
mod matrix;
mod missing;
mod ml;
mod pybridge;
mod qol;
mod recurrent;
mod regression;
mod relative;
mod residuals;
mod scoring;
pub mod simd_ops;
mod spatial;
mod specialized;
mod surv_analysis;
mod tests;
mod utilities;
mod validation;

pub use concordance::basic::concordance as compute_concordance;
pub use concordance::concordance1::{concordance1, perform_concordance1_calculation};
pub use concordance::concordance3::perform_concordance3_calculation;
pub use concordance::concordance5::perform_concordance_calculation;
pub use constants::*;
pub use core::coxcount1::{CoxCountOutput, coxcount1, coxcount2};
pub use core::coxscho::schoenfeld_residuals;
pub use core::nsk::{NaturalSplineKnot, SplineBasisResult, nsk};
pub use core::pspline::PSpline;
use pybridge::cox_py_callback::cox_callback;
use pybridge::pyears3b::perform_pyears_calculation;
use pybridge::pystep::{perform_pystep_calculation, perform_pystep_simple_calculation};
pub use regression::aareg::{AaregOptions, aareg};
pub use regression::agfit5::perform_cox_regression_frailty;
pub use regression::blogit::LinkFunctionParams;
pub use regression::clogit::{ClogitDataSet, ConditionalLogisticRegression};
pub use regression::coxph::{CoxPHModel, Subject};
pub use regression::coxph_detail::{CoxphDetail, CoxphDetailRow, coxph_detail};
pub use regression::finegray_regression::{
    CompetingRisksCIF, FineGrayResult, competing_risks_cif, finegray_regression,
};
pub use regression::ridge::{RidgePenalty, RidgeResult, ridge_cv, ridge_fit};
pub use regression::spline_hazard::{
    FlexibleParametricResult, HazardSplineResult, RestrictedCubicSplineResult, SplineConfig,
    flexible_parametric_model, predict_hazard_spline, restricted_cubic_spline,
};
pub use regression::survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, predict_survreg, predict_survreg_quantile,
};
pub use regression::survreg6::{DistributionType, SurvivalFit, SurvregConfig, survreg};
pub use residuals::agmart::agmart;
pub use residuals::coxmart::coxmart;
pub use residuals::diagnostics::{
    DfbetaResult, GofTestResult, LeverageResult, ModelInfluenceResult, OutlierDetectionResult,
    SchoenfeldSmoothResult, dfbeta_cox, goodness_of_fit_cox, leverage_cox, model_influence_cox,
    outlier_detection_cox, smooth_schoenfeld,
};
pub use residuals::survfit_resid::{SurvfitResiduals, residuals_survfit};
pub use residuals::survreg_resid::{SurvregResiduals, dfbeta_survreg, residuals_survreg};
pub use scoring::agscore2::perform_score_calculation;
pub use scoring::agscore3::perform_agscore3_calculation;
pub use scoring::coxscore2::cox_score_residuals;
pub use specialized::brier::{brier, compute_brier, integrated_brier};
pub use specialized::cch::{CchMethod, CohortData};
pub use specialized::cipoisson::{cipoisson, cipoisson_anscombe, cipoisson_exact};
pub use specialized::drift_detection::{
    DriftConfig, DriftReport, FeatureDriftResult, PerformanceDriftResult, detect_drift,
    monitor_performance,
};
pub use specialized::finegray::{FineGrayOutput, finegray};
pub use specialized::model_cards::{
    FairnessAuditResult, ModelCard, ModelPerformanceMetrics, SubgroupPerformance,
    create_model_card, fairness_audit,
};
pub use specialized::norisk::norisk;
pub use specialized::pyears_summary::{
    PyearsCell, PyearsSummary, pyears_by_cell, pyears_ci, summary_pyears,
};
pub use specialized::ratetable::{
    DimType, RateDimension, RateTable, RatetableDateResult, create_simple_ratetable, days_to_date,
    is_ratetable, ratetable_date,
};
pub use specialized::statefig::{
    StateFigData, statefig, statefig_matplotlib_code, statefig_transition_matrix, statefig_validate,
};
pub use specialized::survexp::{SurvExpResult, survexp, survexp_individual};
pub use specialized::survexp_us::{
    ExpectedSurvivalResult, compute_expected_survival, survexp_mn, survexp_us, survexp_usr,
};
pub use specialized::warranty_analysis::{
    ReliabilityGrowthResult, RenewalResult, WarrantyConfig, WarrantyResult, reliability_growth,
    renewal_analysis, warranty_analysis,
};
pub use surv_analysis::aggregate_survfit::{
    AggregateSurvfitResult, aggregate_survfit, aggregate_survfit_by_group,
};
pub use surv_analysis::agsurv4::agsurv4;
pub use surv_analysis::agsurv5::agsurv5;
pub use surv_analysis::illness_death::{
    IllnessDeathConfig, IllnessDeathPrediction, IllnessDeathResult, IllnessDeathType,
    TransitionHazard, fit_illness_death, predict_illness_death,
};
pub use surv_analysis::multi_state::{
    MarkovMSMResult, MultiStateConfig, MultiStateResult, TransitionIntensityResult,
    estimate_transition_intensities, fit_markov_msm, fit_multi_state_model,
};
pub use surv_analysis::nelson_aalen::{
    NelsonAalenResult, StratifiedKMResult, nelson_aalen, nelson_aalen_estimator,
    stratified_kaplan_meier,
};
pub use surv_analysis::pseudo::{
    GEEConfig, GEEResult, PseudoResult, pseudo, pseudo_fast, pseudo_gee_regression,
};
pub use surv_analysis::semi_markov::{
    SemiMarkovConfig, SemiMarkovPrediction, SemiMarkovResult, SojournDistribution,
    SojournTimeParams, fit_semi_markov, predict_semi_markov,
};
pub use surv_analysis::survdiff2::{SurvDiffResult, survdiff2};
pub use surv_analysis::survfit_matrix::{
    SurvfitMatrixResult, basehaz, survfit_from_cumhaz, survfit_from_hazard, survfit_from_matrix,
    survfit_multistate,
};
pub use surv_analysis::survfitaj::{SurvFitAJ, survfitaj};
pub use surv_analysis::survfitaj_extended::{
    AalenJohansenExtendedConfig, AalenJohansenExtendedResult, TransitionMatrix, TransitionType,
    VarianceEstimator, survfitaj_extended,
};
pub use surv_analysis::survfitkm::{
    KaplanMeierConfig, SurvFitKMOutput, SurvfitKMOptions, compute_survfitkm, survfitkm,
    survfitkm_with_options,
};
pub use utilities::aeq_surv::{AeqSurvResult, aeq_surv};
pub use utilities::agexact::agexact;
pub use utilities::cluster::{ClusterResult, cluster, cluster_str};
pub use utilities::collapse::collapse;
pub use utilities::neardate::{NearDateResult, neardate, neardate_str};
pub use utilities::reliability::{
    ReliabilityResult, ReliabilityScale, conditional_reliability, failure_probability,
    hazard_to_reliability, mean_residual_life, reliability, reliability_inverse,
};
pub use utilities::rttright::{RttrightResult, rttright, rttright_stratified};
pub use utilities::strata::{StrataResult, strata, strata_str};
pub use utilities::surv2data::{Surv2DataResult, surv2data};
pub use utilities::survcondense::{CondenseResult, survcondense};
pub use utilities::survsplit::{SplitResult, survsplit};
pub use utilities::tcut::{TcutResult, tcut, tcut_expand};
pub use utilities::timeline::{IntervalResult, TimelineResult, from_timeline, to_timeline};
pub use utilities::tmerge::{tmerge, tmerge2, tmerge3};
pub use validation::anova::{AnovaCoxphResult, AnovaRow, anova_coxph, anova_coxph_single};
pub use validation::bootstrap::{BootstrapResult, bootstrap_cox_ci, bootstrap_survreg_ci};
pub use validation::calibration::{
    AdvancedCalibrationResult, CalibrationResult, PredictionResult, RiskStratificationResult,
    TdAUCResult, TimeDependentCalibrationResult, advanced_calibration_metrics, calibration,
    predict_cox, risk_stratification, td_auc, time_dependent_calibration,
};
pub use validation::conformal::{
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
pub use validation::crossval::{CVResult, cv_cox_concordance, cv_survreg_loglik};
pub use validation::d_calibration::{
    BrierCalibrationResult, CalibrationPlotData, DCalibrationResult, MultiTimeCalibrationResult,
    OneCalibrationResult, SmoothedCalibrationCurve, brier_calibration, calibration_plot,
    d_calibration, multi_time_calibration, one_calibration, smoothed_calibration,
};
pub use validation::decision_curve::{
    ClinicalUtilityResult, DecisionCurveResult, ModelComparisonResult,
    clinical_utility_at_threshold, compare_decision_curves, decision_curve_analysis,
};
pub use validation::fairness::{
    FairnessMetrics, RobustnessResult, SubgroupAnalysisResult, assess_model_robustness,
    compute_fairness_metrics, subgroup_analysis,
};
pub use validation::hyperparameter::{
    BenchmarkResult, HyperparameterResult, HyperparameterSearchConfig, NestedCVResult,
    SearchStrategy, benchmark_models, hyperparameter_search, nested_cross_validation,
};
pub use validation::landmark::{
    ConditionalSurvivalResult, HazardRatioResult, LandmarkResult, LifeTableResult,
    SurvivalAtTimeResult, conditional_survival, hazard_ratio, landmark_analysis,
    landmark_analysis_batch, life_table, survival_at_times,
};
pub use validation::logrank::{
    LogRankResult, TrendTestResult, WeightType, fleming_harrington_test, logrank_test,
    logrank_trend, weighted_logrank_test,
};
pub use validation::meta_analysis::{
    MetaAnalysisConfig, MetaAnalysisResult, MetaForestPlotData, PublicationBiasResult,
    generate_forest_plot_data, publication_bias_tests, survival_meta_analysis,
};
pub use validation::model_selection::{
    CrossValidatedScore, ModelSelectionCriteria, SurvivalModelComparison, compare_models,
    compute_cv_score, compute_model_selection_criteria,
};
pub use validation::power::{
    AccrualResult, SampleSizeResult, expected_events, power_survival, sample_size_survival,
    sample_size_survival_freedman,
};
pub use validation::rcll::{
    RCLLResult, compute_rcll, compute_rcll_single_time, rcll, rcll_single_time,
};
pub use validation::reporting::{
    CalibrationCurveData, ForestPlotData, KaplanMeierPlotData, ROCPlotData, SurvivalReport,
    calibration_plot_data, forest_plot_data, generate_survival_report, km_plot_data, roc_plot_data,
};
pub use validation::rmst::{
    ChangepointInfo, CumulativeIncidenceResult, MedianSurvivalResult, NNTResult,
    RMSTComparisonResult, RMSTOptimalThresholdResult, RMSTResult, compute_rmst,
    cumulative_incidence, number_needed_to_treat, rmst, rmst_comparison, rmst_optimal_threshold,
    survival_quantile,
};
pub use validation::royston::{RoystonResult, royston, royston_from_model};
pub use validation::survcheck::{SurvCheckResult, survcheck, survcheck_simple};
pub use validation::survobrien::{SurvObrienResult, survobrien};
pub use validation::tests::{
    ProportionalityTest, TestResult, lrt_test, ph_test, score_test, wald_test,
};
use validation::tests::{score_test_py, wald_test_py};
pub use validation::time_dependent_auc::{
    CumulativeDynamicAUCResult, TimeDepAUCResult, cumulative_dynamic_auc,
    cumulative_dynamic_auc_core, time_dependent_auc, time_dependent_auc_core,
};
pub use validation::uncertainty::{
    BayesianBootstrapConfig, BayesianBootstrapResult, CalibrationUncertaintyResult,
    ConformalSurvivalConfig, ConformalSurvivalResult, EnsembleUncertaintyResult,
    JackknifePlusConfig, JackknifePlusResult, MCDropoutConfig, QuantileRegressionResult,
    UncertaintyResult, bayesian_bootstrap_survival, calibrate_prediction_intervals,
    conformal_survival, ensemble_uncertainty, jackknife_plus_survival, mc_dropout_uncertainty,
    quantile_regression_intervals,
};
pub use validation::uno_c_index::{
    CIndexDecompositionResult, ConcordanceComparisonResult, GonenHellerResult, UnoCIndexResult,
    c_index_decomposition, compare_uno_c_indices, gonen_heller_concordance, uno_c_index,
};
pub use validation::yates::{
    YatesPairwiseResult, YatesResult, yates, yates_contrast, yates_pairwise,
};

pub use bayesian::bayesian_cox::{BayesianCoxResult, bayesian_cox, bayesian_cox_predict_survival};
pub use bayesian::bayesian_extensions::{
    BayesianModelAveragingConfig, BayesianModelAveragingResult, DirichletProcessConfig,
    DirichletProcessResult, HorseshoeConfig, HorseshoeResult, SpikeSlabConfig, SpikeSlabResult,
    bayesian_model_averaging_cox, dirichlet_process_survival, horseshoe_cox, spike_slab_cox,
};
pub use bayesian::bayesian_parametric::{
    BayesianParametricResult, bayesian_parametric, bayesian_parametric_predict,
};
pub use causal::causal_forest::{
    CausalForestConfig, CausalForestResult, CausalForestSurvival, causal_forest_survival,
};
pub use causal::counterfactual_survival::{
    CounterfactualSurvivalConfig, CounterfactualSurvivalResult, TVSurvCausConfig, TVSurvCausResult,
    estimate_counterfactual_survival, estimate_tv_survcaus,
};
pub use causal::dependent_censoring::{
    CopulaCensoringConfig, CopulaCensoringResult, CopulaType, MNARSurvivalConfig,
    MNARSurvivalResult, SensitivityBoundsConfig, SensitivityBoundsResult, copula_censoring_model,
    mnar_sensitivity_survival, sensitivity_bounds_survival,
};
pub use causal::double_ml::{
    CATEResult, DoubleMLConfig, DoubleMLResult, double_ml_cate, double_ml_survival,
};
pub use causal::g_computation::{GComputationResult, g_computation, g_computation_survival_curves};
pub use causal::instrumental_variable::{
    GEstimationConfig, GEstimationResult, IVCoxConfig, IVCoxResult, MediationSurvivalConfig,
    MediationSurvivalResult, RDSurvivalConfig, RDSurvivalResult, g_estimation_aft, iv_cox,
    mediation_survival, rd_survival,
};
pub use causal::ipcw::{
    IPCWResult, compute_ipcw_weights, ipcw_kaplan_meier, ipcw_treatment_effect,
};
pub use causal::msm::{MSMResult, compute_longitudinal_iptw, marginal_structural_model};
pub use causal::target_trial::{
    TargetTrialResult, sequential_trial_emulation, target_trial_emulation,
};
pub use causal::tmle::{TMLEConfig, TMLEResult, TMLESurvivalResult, tmle_ate, tmle_survival};
pub use interpretability::ale_plots::{
    ALE2DResult, ALEResult, compute_ale, compute_ale_2d, compute_time_varying_ale,
};
pub use interpretability::changepoints::{
    AllChangepointsResult, Changepoint, ChangepointConfig, ChangepointMethod, ChangepointResult,
    CostFunction, detect_changepoints, detect_changepoints_single_series,
};
pub use interpretability::friedman_h::{
    FeatureImportanceResult as FriedmanFeatureImportanceResult, FriedmanHResult,
    compute_all_pairwise_interactions, compute_feature_importance_decomposition,
    compute_friedman_h,
};
pub use interpretability::ice_curves::{
    DICEResult, ICEResult, cluster_ice_curves, compute_dice, compute_ice, compute_survival_ice,
    detect_heterogeneity,
};
pub use interpretability::local_global::{
    FeatureViewAnalysis, LocalGlobalConfig, LocalGlobalResult, LocalGlobalSummary,
    ViewRecommendation, analyze_local_global,
};
pub use interpretability::survshap::{
    AggregationMethod, BootstrapSurvShapResult, FeatureImportance, PermutationImportanceResult,
    ShapInteractionResult, SurvShapConfig, SurvShapExplanation, SurvShapResult, aggregate_survshap,
    compute_shap_interactions, permutation_importance, survshap, survshap_bootstrap,
    survshap_from_model,
};
pub use interpretability::time_varying::{
    TimeVaryingAnalysis, TimeVaryingTestConfig, TimeVaryingTestResult, TimeVaryingTestType,
    detect_time_varying_features,
};
pub use interpretability::variable_groups::{
    FeatureGroup, GroupingMethod, LinkageType, VariableGroupingConfig, VariableGroupingResult,
    group_variables,
};
pub use interval::interval_censoring::{
    IntervalCensoredResult, IntervalDistribution, TurnbullResult, interval_censored_regression,
    npmle_interval, turnbull_estimator,
};
pub use joint::dynamic_prediction::{
    DynamicCIndexResult, DynamicPredictionResult, IPCWAUCResult, SuperLandmarkResult,
    TimeDependentROCResult, TimeVaryingAUCResult, dynamic_auc, dynamic_brier_score,
    dynamic_c_index, dynamic_prediction, ipcw_auc, landmarking_analysis, super_landmark_model,
    time_dependent_roc, time_varying_auc,
};
pub use joint::joint_model::{AssociationStructure, JointModelResult, joint_model};
pub use missing::multiple_imputation::{
    ImputationMethod, MultipleImputationResult, analyze_missing_pattern,
    multiple_imputation_survival,
};
pub use missing::pattern_mixture::{
    PatternMixtureResult, SensitivityAnalysisType, pattern_mixture_model, sensitivity_analysis,
    tipping_point_analysis,
};
pub use ml::active_learning::{
    ActiveLearningConfig, ActiveLearningResult, AdaptiveDesignResult, LogrankPowerResult,
    LogrankSampleSizeResult, QBCResult, active_learning_selection, group_sequential_analysis,
    power_logrank, query_by_committee, sample_size_logrank,
};
pub use ml::adversarial_robustness::{
    AdversarialAttackConfig, AdversarialAttackResult, AdversarialDefenseConfig, AdversarialExample,
    AttackType, DefenseType, RobustSurvivalModel, RobustnessEvaluation,
    adversarial_training_survival, evaluate_robustness, generate_adversarial_examples,
};
pub use ml::attention_cox::{AttentionCoxConfig, AttentionCoxModel, fit_attention_cox};
pub use ml::contrastive_surv::{
    ContrastiveSurv, ContrastiveSurvConfig, ContrastiveSurvResult, SurvivalLossType,
    contrastive_surv,
};
pub use ml::cox_time::{CoxTimeConfig, CoxTimeModel, fit_cox_time};
pub use ml::deep_pamm::{DeepPAMMConfig, DeepPAMMModel, fit_deep_pamm};
pub use ml::deep_surv::{Activation, DeepSurv, DeepSurvConfig, deep_surv};
pub use ml::deephit::{DeepHit, DeepHitConfig, deephit};
pub use ml::differential_privacy::{
    DPConfig, DPCoxResult, DPHistogramResult, DPSurvivalResult, LocalDPResult, dp_cox_regression,
    dp_histogram, dp_kaplan_meier, local_dp_mean,
};
pub use ml::distributionally_robust::{
    DROSurvivalConfig, DROSurvivalResult, RobustnessAnalysis, UncertaintySet, dro_survival,
    robustness_analysis,
};
pub use ml::dynamic_deephit::{
    DynamicDeepHit, DynamicDeepHitConfig, TemporalType, dynamic_deephit,
};
pub use ml::dysurv::{
    DySurvConfig, DySurvModel, DynamicRiskResult, dynamic_risk_prediction, fit_dysurv,
};
pub use ml::ensemble_surv::{
    BlendingResult, ComponentwiseBoostingConfig, ComponentwiseBoostingResult, StackingConfig,
    StackingResult, SuperLearnerConfig, SuperLearnerResult, blending_survival,
    componentwise_boosting, stacking_survival, super_learner_survival,
};
pub use ml::federated_learning::{
    FederatedConfig, FederatedSurvivalResult, PrivacyAccountant, SecureAggregationResult,
    federated_cox, secure_aggregate,
};
pub use ml::galee::{GALEE, GALEEConfig, GALEEResult, UnimodalConstraint, galee};
pub use ml::gpu_acceleration::{
    BatchPredictionResult, ComputeBackend, DeviceInfo, GPUConfig, ParallelCoxResult,
    batch_predict_survival, benchmark_compute_backend, get_available_devices, is_gpu_available,
    parallel_cox_regression, parallel_matrix_operations,
};
pub use ml::gradient_boost::{
    GBSurvLoss, GradientBoostSurvival, GradientBoostSurvivalConfig, gradient_boost_survival,
};
pub use ml::graph_surv::{GraphSurvConfig, GraphSurvModel, fit_graph_surv};
pub use ml::knowledge_distillation::{
    DistillationConfig, DistillationResult, DistilledSurvivalModel, PruningResult,
    distill_survival_model, prune_survival_model,
};
pub use ml::multimodal_surv::{
    FusionStrategy, MultimodalSurvConfig, MultimodalSurvModel, fit_multimodal_surv,
};
pub use ml::neural_mtlr::{NeuralMTLRConfig, NeuralMTLRModel, fit_neural_mtlr};
pub use ml::neural_ode_surv::{NeuralODESurvConfig, NeuralODESurvModel, fit_neural_ode_surv};
pub use ml::recurrent_surv::{
    LongitudinalSurvConfig, LongitudinalSurvModel, RecurrentSurvConfig, RecurrentSurvModel,
    fit_longitudinal_surv, fit_recurrent_surv,
};
pub use ml::state_space_surv::{MambaSurvConfig, MambaSurvModel, fit_mamba_surv};
pub use ml::streaming_survival::{
    ConceptDriftDetector, StreamingCoxConfig, StreamingCoxModel, StreamingKaplanMeier,
};
pub use ml::survival_forest::{SplitRule, SurvivalForest, SurvivalForestConfig, survival_forest};
pub use ml::survival_transformer::{
    SurvivalTransformerConfig, SurvivalTransformerModel, fit_survival_transformer,
};
pub use ml::survtrace::{SurvTrace, SurvTraceActivation, SurvTraceConfig, survtrace};
pub use ml::temporal_fusion::{
    TFTConfig, TemporalFusionTransformer, fit_temporal_fusion_transformer,
};
pub use ml::tracer::{Tracer, TracerConfig, tracer};
pub use ml::transfer_learning::{
    DomainAdaptationResult, PretrainedSurvivalModel, TransferLearningConfig, TransferStrategy,
    TransferredModel, compute_domain_distance, pretrain_survival_model, transfer_survival_model,
};
pub use qol::qaly::{
    QALYResult, incremental_cost_effectiveness, qaly_calculation, qaly_comparison,
};
pub use qol::qtwist::{QTWISTResult, qtwist_analysis, qtwist_comparison, qtwist_sensitivity};
pub use recurrent::gap_time::{GapTimeResult, gap_time_model, pwp_gap_time};
pub use recurrent::joint_frailty::{FrailtyDistribution, JointFrailtyResult, joint_frailty_model};
pub use recurrent::marginal_models::{
    MarginalMethod, MarginalModelResult, andersen_gill, marginal_recurrent_model, wei_lin_weissfeld,
};
pub use regression::cause_specific_cox::{
    CauseSpecificCoxConfig, CauseSpecificCoxResult, CensoringType, cause_specific_cox,
    cause_specific_cox_all,
};
pub use regression::cure_models::{
    BoundedCumulativeHazardConfig, BoundedCumulativeHazardResult, CureDistribution,
    CureModelComparisonResult, MixtureCureResult, NonMixtureCureConfig, NonMixtureCureResult,
    NonMixtureType, PromotionTimeCureResult, bounded_cumulative_hazard_model, compare_cure_models,
    mixture_cure_model, non_mixture_cure_model, predict_bounded_cumulative_hazard,
    predict_non_mixture_survival, promotion_time_cure_model,
};
pub use regression::elastic_net::{
    ElasticNetCoxPath, ElasticNetCoxResult, elastic_net_cox, elastic_net_cox_cv,
    elastic_net_cox_path,
};
pub use regression::fast_cox::{
    FastCoxConfig, FastCoxPath, FastCoxResult, ScreeningRule, fast_cox, fast_cox_cv, fast_cox_path,
};
pub use regression::functional_survival::{
    BasisType, FunctionalPCAResult, FunctionalSurvivalConfig, FunctionalSurvivalResult,
    fpca_survival, functional_cox,
};
pub use regression::high_dimensional::{
    GroupLassoConfig, GroupLassoResult, SISConfig, SISResult, SparseBoostingConfig,
    SparseBoostingResult, StabilitySelectionConfig, StabilitySelectionResult, group_lasso_cox,
    sis_cox, sparse_boosting_cox, stability_selection_cox,
};
pub use regression::joint_competing::{
    CauseResult, CorrelationType, JointCompetingRisksConfig, JointCompetingRisksResult,
    joint_competing_risks,
};
pub use regression::longitudinal_survival::{
    JointLongSurvResult, JointModelConfig, LandmarkAnalysisResult, LongDynamicPredResult,
    TimeVaryingCoxResult, joint_longitudinal_model, landmark_cox_analysis,
    longitudinal_dynamic_pred, time_varying_cox,
};
pub use regression::recurrent_events::{
    AndersonGillResult, NegativeBinomialFrailtyConfig, NegativeBinomialFrailtyResult, PWPConfig,
    PWPResult, PWPTimescale, WLWConfig, WLWResult, anderson_gill_model, negative_binomial_frailty,
    pwp_model, wlw_model,
};
pub use relative::net_survival::{
    NetSurvivalMethod, NetSurvivalResult, crude_probability_of_death, net_survival,
};
pub use relative::relative_survival::{
    ExcessHazardModelResult, RelativeSurvivalResult, excess_hazard_regression, relative_survival,
};
pub use spatial::network_survival::{
    CentralityType, DiffusionSurvivalConfig, DiffusionSurvivalResult, NetworkHeterogeneityResult,
    NetworkSurvivalConfig, NetworkSurvivalResult, diffusion_survival_model,
    network_heterogeneity_survival, network_survival_model,
};
pub use spatial::spatial_frailty::{
    SpatialCorrelationStructure, SpatialFrailtyResult, compute_spatial_smoothed_rates,
    moran_i_test, spatial_frailty_model,
};

#[pymodule]
fn _survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perform_cox_regression_frailty, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pyears_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance1_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance3_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_score_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_agscore3_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_simple_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(aareg, &m)?)?;
    m.add_function(wrap_pyfunction!(collapse, &m)?)?;
    m.add_function(wrap_pyfunction!(cox_callback, &m)?)?;
    m.add_function(wrap_pyfunction!(coxcount1, &m)?)?;
    m.add_function(wrap_pyfunction!(coxcount2, &m)?)?;
    m.add_function(wrap_pyfunction!(norisk, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_exact, &m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_anscombe, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_concordance, &m)?)?;
    m.add_function(wrap_pyfunction!(agexact, &m)?)?;
    m.add_function(wrap_pyfunction!(agsurv4, &m)?)?;
    m.add_function(wrap_pyfunction!(agsurv5, &m)?)?;
    m.add_function(wrap_pyfunction!(agmart, &m)?)?;
    m.add_function(wrap_pyfunction!(coxmart, &m)?)?;

    m.add_function(wrap_pyfunction!(dfbeta_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(leverage_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(smooth_schoenfeld, &m)?)?;
    m.add_function(wrap_pyfunction!(outlier_detection_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(model_influence_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(goodness_of_fit_cox, &m)?)?;
    m.add_class::<DfbetaResult>()?;
    m.add_class::<LeverageResult>()?;
    m.add_class::<SchoenfeldSmoothResult>()?;
    m.add_class::<OutlierDetectionResult>()?;
    m.add_class::<ModelInfluenceResult>()?;
    m.add_class::<GofTestResult>()?;

    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_with_options, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitaj, &m)?)?;
    m.add_function(wrap_pyfunction!(survdiff2, &m)?)?;
    m.add_function(wrap_pyfunction!(finegray, &m)?)?;
    m.add_function(wrap_pyfunction!(finegray_regression, &m)?)?;
    m.add_function(wrap_pyfunction!(competing_risks_cif, &m)?)?;
    m.add_function(wrap_pyfunction!(survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(brier, &m)?)?;
    m.add_function(wrap_pyfunction!(integrated_brier, &m)?)?;
    m.add_function(wrap_pyfunction!(tmerge, &m)?)?;
    m.add_function(wrap_pyfunction!(tmerge2, &m)?)?;
    m.add_function(wrap_pyfunction!(tmerge3, &m)?)?;
    m.add_function(wrap_pyfunction!(survsplit, &m)?)?;
    m.add_function(wrap_pyfunction!(survcondense, &m)?)?;
    m.add_function(wrap_pyfunction!(surv2data, &m)?)?;
    m.add_function(wrap_pyfunction!(to_timeline, &m)?)?;
    m.add_function(wrap_pyfunction!(from_timeline, &m)?)?;
    m.add_function(wrap_pyfunction!(survobrien, &m)?)?;
    m.add_function(wrap_pyfunction!(schoenfeld_residuals, &m)?)?;
    m.add_function(wrap_pyfunction!(cox_score_residuals, &m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_cox_ci, &m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_survreg_ci, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_cox_concordance, &m)?)?;
    m.add_function(wrap_pyfunction!(cv_survreg_loglik, &m)?)?;
    m.add_function(wrap_pyfunction!(lrt_test, &m)?)?;
    m.add_function(wrap_pyfunction!(wald_test_py, &m)?)?;
    m.add_function(wrap_pyfunction!(score_test_py, &m)?)?;
    m.add_function(wrap_pyfunction!(ph_test, &m)?)?;
    m.add_function(wrap_pyfunction!(nelson_aalen_estimator, &m)?)?;
    m.add_function(wrap_pyfunction!(stratified_kaplan_meier, &m)?)?;
    m.add_function(wrap_pyfunction!(logrank_test, &m)?)?;
    m.add_function(wrap_pyfunction!(fleming_harrington_test, &m)?)?;
    m.add_function(wrap_pyfunction!(logrank_trend, &m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival_freedman, &m)?)?;
    m.add_function(wrap_pyfunction!(power_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(expected_events, &m)?)?;
    m.add_function(wrap_pyfunction!(calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(risk_stratification, &m)?)?;
    m.add_function(wrap_pyfunction!(td_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(d_calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(one_calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot, &m)?)?;
    m.add_function(wrap_pyfunction!(brier_calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(multi_time_calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(smoothed_calibration, &m)?)?;
    m.add_function(wrap_pyfunction!(rmst, &m)?)?;
    m.add_function(wrap_pyfunction!(rmst_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(rmst_optimal_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(survival_quantile, &m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_incidence, &m)?)?;
    m.add_function(wrap_pyfunction!(number_needed_to_treat, &m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(conditional_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(hazard_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(survival_at_times, &m)?)?;
    m.add_function(wrap_pyfunction!(life_table, &m)?)?;
    m.add_function(wrap_pyfunction!(aeq_surv, &m)?)?;
    m.add_function(wrap_pyfunction!(cluster, &m)?)?;
    m.add_function(wrap_pyfunction!(cluster_str, &m)?)?;
    m.add_function(wrap_pyfunction!(strata, &m)?)?;
    m.add_function(wrap_pyfunction!(strata_str, &m)?)?;
    m.add_function(wrap_pyfunction!(neardate, &m)?)?;
    m.add_function(wrap_pyfunction!(neardate_str, &m)?)?;
    m.add_function(wrap_pyfunction!(tcut, &m)?)?;
    m.add_function(wrap_pyfunction!(tcut_expand, &m)?)?;
    m.add_function(wrap_pyfunction!(rttright, &m)?)?;
    m.add_function(wrap_pyfunction!(rttright_stratified, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_individual, &m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_ratetable, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_matplotlib_code, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_transition_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_validate, &m)?)?;

    m.add_function(wrap_pyfunction!(create_model_card, &m)?)?;
    m.add_function(wrap_pyfunction!(fairness_audit, &m)?)?;
    m.add_class::<ModelCard>()?;
    m.add_class::<ModelPerformanceMetrics>()?;
    m.add_class::<SubgroupPerformance>()?;
    m.add_class::<FairnessAuditResult>()?;

    m.add_function(wrap_pyfunction!(detect_drift, &m)?)?;
    m.add_function(wrap_pyfunction!(monitor_performance, &m)?)?;
    m.add_class::<DriftConfig>()?;
    m.add_class::<DriftReport>()?;
    m.add_class::<FeatureDriftResult>()?;
    m.add_class::<PerformanceDriftResult>()?;

    m.add_function(wrap_pyfunction!(pseudo, &m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_gee_regression, &m)?)?;
    m.add_class::<GEEConfig>()?;
    m.add_class::<GEEResult>()?;
    m.add_function(wrap_pyfunction!(aggregate_survfit, &m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit_by_group, &m)?)?;
    m.add_function(wrap_pyfunction!(survcheck, &m)?)?;
    m.add_function(wrap_pyfunction!(survcheck_simple, &m)?)?;
    m.add_function(wrap_pyfunction!(royston, &m)?)?;
    m.add_function(wrap_pyfunction!(royston_from_model, &m)?)?;
    m.add_function(wrap_pyfunction!(yates, &m)?)?;
    m.add_function(wrap_pyfunction!(yates_contrast, &m)?)?;
    m.add_function(wrap_pyfunction!(yates_pairwise, &m)?)?;
    m.add_function(wrap_pyfunction!(uno_c_index, &m)?)?;
    m.add_function(wrap_pyfunction!(compare_uno_c_indices, &m)?)?;
    m.add_function(wrap_pyfunction!(c_index_decomposition, &m)?)?;
    m.add_function(wrap_pyfunction!(gonen_heller_concordance, &m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_dynamic_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(rcll, &m)?)?;
    m.add_function(wrap_pyfunction!(rcll_single_time, &m)?)?;
    m.add_function(wrap_pyfunction!(ridge_fit, &m)?)?;
    m.add_function(wrap_pyfunction!(ridge_cv, &m)?)?;
    m.add_function(wrap_pyfunction!(nsk, &m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph, &m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph_single, &m)?)?;
    m.add_function(wrap_pyfunction!(reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(reliability_inverse, &m)?)?;
    m.add_function(wrap_pyfunction!(hazard_to_reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(failure_probability, &m)?)?;
    m.add_function(wrap_pyfunction!(conditional_reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_residual_life, &m)?)?;
    m.add_function(wrap_pyfunction!(survfit_from_hazard, &m)?)?;
    m.add_function(wrap_pyfunction!(survfit_from_cumhaz, &m)?)?;
    m.add_function(wrap_pyfunction!(survfit_from_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(survfit_multistate, &m)?)?;
    m.add_function(wrap_pyfunction!(basehaz, &m)?)?;
    m.add_class::<AaregOptions>()?;
    m.add_class::<PSpline>()?;
    m.add_class::<CoxCountOutput>()?;
    m.add_class::<LinkFunctionParams>()?;
    m.add_class::<CoxPHModel>()?;
    m.add_class::<Subject>()?;
    m.add_class::<SurvFitKMOutput>()?;
    m.add_class::<SurvfitKMOptions>()?;
    m.add_class::<KaplanMeierConfig>()?;
    m.add_class::<SurvFitAJ>()?;
    m.add_class::<FineGrayOutput>()?;
    m.add_class::<FineGrayResult>()?;
    m.add_class::<CompetingRisksCIF>()?;
    m.add_class::<SurvivalFit>()?;
    m.add_class::<SurvregConfig>()?;
    m.add_class::<DistributionType>()?;
    m.add_class::<SurvDiffResult>()?;
    m.add_class::<CchMethod>()?;
    m.add_class::<CohortData>()?;
    m.add_class::<SplitResult>()?;
    m.add_class::<CondenseResult>()?;
    m.add_class::<Surv2DataResult>()?;
    m.add_class::<TimelineResult>()?;
    m.add_class::<IntervalResult>()?;
    m.add_class::<SurvObrienResult>()?;
    m.add_class::<ClogitDataSet>()?;
    m.add_class::<ConditionalLogisticRegression>()?;
    m.add_class::<BootstrapResult>()?;
    m.add_class::<CVResult>()?;
    m.add_class::<TestResult>()?;
    m.add_class::<ProportionalityTest>()?;
    m.add_class::<NelsonAalenResult>()?;
    m.add_class::<StratifiedKMResult>()?;
    m.add_class::<LogRankResult>()?;
    m.add_class::<TrendTestResult>()?;
    m.add_class::<SampleSizeResult>()?;
    m.add_class::<AccrualResult>()?;
    m.add_class::<CalibrationResult>()?;
    m.add_class::<PredictionResult>()?;
    m.add_class::<RiskStratificationResult>()?;
    m.add_class::<TdAUCResult>()?;
    m.add_class::<DCalibrationResult>()?;
    m.add_class::<OneCalibrationResult>()?;
    m.add_class::<CalibrationPlotData>()?;
    m.add_class::<BrierCalibrationResult>()?;
    m.add_class::<MultiTimeCalibrationResult>()?;
    m.add_class::<SmoothedCalibrationCurve>()?;
    m.add_class::<RMSTResult>()?;
    m.add_class::<RMSTComparisonResult>()?;
    m.add_class::<RMSTOptimalThresholdResult>()?;
    m.add_class::<ChangepointInfo>()?;
    m.add_class::<MedianSurvivalResult>()?;
    m.add_class::<CumulativeIncidenceResult>()?;
    m.add_class::<NNTResult>()?;
    m.add_class::<LandmarkResult>()?;
    m.add_class::<ConditionalSurvivalResult>()?;
    m.add_class::<HazardRatioResult>()?;
    m.add_class::<SurvivalAtTimeResult>()?;
    m.add_class::<LifeTableResult>()?;
    m.add_class::<AeqSurvResult>()?;
    m.add_class::<ClusterResult>()?;
    m.add_class::<StrataResult>()?;
    m.add_class::<NearDateResult>()?;
    m.add_class::<TcutResult>()?;
    m.add_class::<RttrightResult>()?;
    m.add_class::<RateTable>()?;
    m.add_class::<RateDimension>()?;
    m.add_class::<DimType>()?;
    m.add_class::<SurvExpResult>()?;
    m.add_class::<StateFigData>()?;
    m.add_class::<PseudoResult>()?;
    m.add_class::<AggregateSurvfitResult>()?;
    m.add_class::<SurvCheckResult>()?;
    m.add_class::<RoystonResult>()?;
    m.add_class::<YatesResult>()?;
    m.add_class::<YatesPairwiseResult>()?;
    m.add_class::<UnoCIndexResult>()?;
    m.add_class::<ConcordanceComparisonResult>()?;
    m.add_class::<CIndexDecompositionResult>()?;
    m.add_class::<GonenHellerResult>()?;
    m.add_class::<TimeDepAUCResult>()?;
    m.add_class::<CumulativeDynamicAUCResult>()?;
    m.add_class::<RCLLResult>()?;
    m.add_class::<RidgePenalty>()?;
    m.add_class::<RidgeResult>()?;
    m.add_class::<NaturalSplineKnot>()?;
    m.add_class::<SplineBasisResult>()?;
    m.add_class::<AnovaCoxphResult>()?;
    m.add_class::<AnovaRow>()?;
    m.add_class::<ReliabilityResult>()?;
    m.add_class::<ReliabilityScale>()?;
    m.add_class::<SurvfitMatrixResult>()?;

    m.add_function(wrap_pyfunction!(datasets::load_lung, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_aml, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_veteran, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_ovarian, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_colon, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_pbc, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_cgd, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_bladder, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_heart, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_kidney, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_rats, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_stanford2, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_udca, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_myeloid, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_flchain, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_transplant, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_mgus, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_mgus2, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_diabetic, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_retinopathy, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_gbsg, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_rotterdam, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_logan, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_nwtco, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_solder, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_tobin, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_rats2, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_nafld, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_cgd0, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_pbcseq, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_hoel, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_myeloma, &m)?)?;
    m.add_function(wrap_pyfunction!(datasets::load_rhdnase, &m)?)?;

    m.add_function(wrap_pyfunction!(residuals_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(dfbeta_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(residuals_survfit, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg_quantile, &m)?)?;
    m.add_function(wrap_pyfunction!(coxph_detail, &m)?)?;

    m.add_function(wrap_pyfunction!(is_ratetable, &m)?)?;
    m.add_function(wrap_pyfunction!(ratetable_date, &m)?)?;
    m.add_function(wrap_pyfunction!(days_to_date, &m)?)?;

    m.add_function(wrap_pyfunction!(summary_pyears, &m)?)?;
    m.add_function(wrap_pyfunction!(pyears_by_cell, &m)?)?;
    m.add_function(wrap_pyfunction!(pyears_ci, &m)?)?;

    m.add_function(wrap_pyfunction!(survexp_us, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_mn, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_usr, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_survival, &m)?)?;

    m.add_class::<SurvregResiduals>()?;
    m.add_class::<SurvfitResiduals>()?;
    m.add_class::<SurvregPrediction>()?;
    m.add_class::<SurvregQuantilePrediction>()?;
    m.add_class::<CoxphDetail>()?;
    m.add_class::<CoxphDetailRow>()?;
    m.add_class::<RatetableDateResult>()?;
    m.add_class::<PyearsSummary>()?;
    m.add_class::<PyearsCell>()?;
    m.add_class::<ExpectedSurvivalResult>()?;

    m.add_function(wrap_pyfunction!(bayesian_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_cox_predict_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric_predict, &m)?)?;
    m.add_class::<BayesianCoxResult>()?;
    m.add_class::<BayesianParametricResult>()?;

    m.add_function(wrap_pyfunction!(dirichlet_process_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_model_averaging_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(spike_slab_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(horseshoe_cox, &m)?)?;
    m.add_class::<DirichletProcessConfig>()?;
    m.add_class::<DirichletProcessResult>()?;
    m.add_class::<BayesianModelAveragingConfig>()?;
    m.add_class::<BayesianModelAveragingResult>()?;
    m.add_class::<SpikeSlabConfig>()?;
    m.add_class::<SpikeSlabResult>()?;
    m.add_class::<HorseshoeConfig>()?;
    m.add_class::<HorseshoeResult>()?;

    m.add_function(wrap_pyfunction!(g_computation, &m)?)?;
    m.add_function(wrap_pyfunction!(g_computation_survival_curves, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_ipcw_weights, &m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_kaplan_meier, &m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_treatment_effect, &m)?)?;
    m.add_function(wrap_pyfunction!(marginal_structural_model, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_longitudinal_iptw, &m)?)?;
    m.add_function(wrap_pyfunction!(target_trial_emulation, &m)?)?;
    m.add_function(wrap_pyfunction!(sequential_trial_emulation, &m)?)?;
    m.add_class::<GComputationResult>()?;
    m.add_class::<IPCWResult>()?;
    m.add_class::<MSMResult>()?;
    m.add_class::<TargetTrialResult>()?;

    m.add_function(wrap_pyfunction!(estimate_counterfactual_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(estimate_tv_survcaus, &m)?)?;
    m.add_class::<CounterfactualSurvivalConfig>()?;
    m.add_class::<CounterfactualSurvivalResult>()?;
    m.add_class::<TVSurvCausConfig>()?;
    m.add_class::<TVSurvCausResult>()?;

    m.add_function(wrap_pyfunction!(tmle_ate, &m)?)?;
    m.add_function(wrap_pyfunction!(tmle_survival, &m)?)?;
    m.add_class::<TMLEConfig>()?;
    m.add_class::<TMLEResult>()?;
    m.add_class::<TMLESurvivalResult>()?;

    m.add_function(wrap_pyfunction!(causal_forest_survival, &m)?)?;
    m.add_class::<CausalForestConfig>()?;
    m.add_class::<CausalForestSurvival>()?;
    m.add_class::<CausalForestResult>()?;

    m.add_function(wrap_pyfunction!(iv_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(rd_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(mediation_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(g_estimation_aft, &m)?)?;
    m.add_class::<IVCoxConfig>()?;
    m.add_class::<IVCoxResult>()?;
    m.add_class::<RDSurvivalConfig>()?;
    m.add_class::<RDSurvivalResult>()?;
    m.add_class::<MediationSurvivalConfig>()?;
    m.add_class::<MediationSurvivalResult>()?;
    m.add_class::<GEstimationConfig>()?;
    m.add_class::<GEstimationResult>()?;

    m.add_function(wrap_pyfunction!(copula_censoring_model, &m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_bounds_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(mnar_sensitivity_survival, &m)?)?;
    m.add_class::<CopulaType>()?;
    m.add_class::<CopulaCensoringConfig>()?;
    m.add_class::<CopulaCensoringResult>()?;
    m.add_class::<SensitivityBoundsConfig>()?;
    m.add_class::<SensitivityBoundsResult>()?;
    m.add_class::<MNARSurvivalConfig>()?;
    m.add_class::<MNARSurvivalResult>()?;

    m.add_function(wrap_pyfunction!(interval_censored_regression, &m)?)?;
    m.add_function(wrap_pyfunction!(turnbull_estimator, &m)?)?;
    m.add_function(wrap_pyfunction!(npmle_interval, &m)?)?;
    m.add_class::<IntervalCensoredResult>()?;
    m.add_class::<TurnbullResult>()?;
    m.add_class::<IntervalDistribution>()?;

    m.add_function(wrap_pyfunction!(joint_model, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_prediction, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_brier_score, &m)?)?;
    m.add_function(wrap_pyfunction!(landmarking_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_c_index, &m)?)?;
    m.add_function(wrap_pyfunction!(ipcw_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(super_landmark_model, &m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_roc, &m)?)?;
    m.add_class::<JointModelResult>()?;
    m.add_class::<DynamicPredictionResult>()?;
    m.add_class::<AssociationStructure>()?;
    m.add_class::<TimeVaryingAUCResult>()?;
    m.add_class::<DynamicCIndexResult>()?;
    m.add_class::<IPCWAUCResult>()?;
    m.add_class::<SuperLandmarkResult>()?;
    m.add_class::<TimeDependentROCResult>()?;

    m.add_function(wrap_pyfunction!(multiple_imputation_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(analyze_missing_pattern, &m)?)?;
    m.add_function(wrap_pyfunction!(pattern_mixture_model, &m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(tipping_point_analysis, &m)?)?;
    m.add_class::<MultipleImputationResult>()?;
    m.add_class::<PatternMixtureResult>()?;
    m.add_class::<ImputationMethod>()?;
    m.add_class::<SensitivityAnalysisType>()?;

    m.add_function(wrap_pyfunction!(survival_forest, &m)?)?;
    m.add_function(wrap_pyfunction!(gradient_boost_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(deep_surv, &m)?)?;
    m.add_function(wrap_pyfunction!(deephit, &m)?)?;
    m.add_function(wrap_pyfunction!(survtrace, &m)?)?;
    m.add_class::<SurvivalForest>()?;
    m.add_class::<SurvivalForestConfig>()?;
    m.add_class::<SplitRule>()?;
    m.add_class::<GradientBoostSurvival>()?;
    m.add_class::<GradientBoostSurvivalConfig>()?;
    m.add_class::<GBSurvLoss>()?;
    m.add_class::<DeepSurv>()?;
    m.add_class::<DeepSurvConfig>()?;
    m.add_class::<Activation>()?;
    m.add_class::<DeepHit>()?;
    m.add_class::<DeepHitConfig>()?;
    m.add_class::<SurvTrace>()?;
    m.add_class::<SurvTraceConfig>()?;
    m.add_class::<SurvTraceActivation>()?;
    m.add_function(wrap_pyfunction!(tracer, &m)?)?;
    m.add_class::<Tracer>()?;
    m.add_class::<TracerConfig>()?;
    m.add_function(wrap_pyfunction!(dynamic_deephit, &m)?)?;
    m.add_class::<DynamicDeepHit>()?;
    m.add_class::<DynamicDeepHitConfig>()?;
    m.add_class::<TemporalType>()?;
    m.add_function(wrap_pyfunction!(contrastive_surv, &m)?)?;
    m.add_class::<ContrastiveSurv>()?;
    m.add_class::<ContrastiveSurvConfig>()?;
    m.add_class::<ContrastiveSurvResult>()?;
    m.add_class::<SurvivalLossType>()?;
    m.add_function(wrap_pyfunction!(galee, &m)?)?;
    m.add_class::<GALEE>()?;
    m.add_class::<GALEEConfig>()?;
    m.add_class::<GALEEResult>()?;
    m.add_class::<UnimodalConstraint>()?;

    m.add_function(wrap_pyfunction!(fit_cox_time, &m)?)?;
    m.add_class::<CoxTimeConfig>()?;
    m.add_class::<CoxTimeModel>()?;

    m.add_function(wrap_pyfunction!(fit_neural_mtlr, &m)?)?;
    m.add_class::<NeuralMTLRConfig>()?;
    m.add_class::<NeuralMTLRModel>()?;

    m.add_function(wrap_pyfunction!(fit_survival_transformer, &m)?)?;
    m.add_class::<SurvivalTransformerConfig>()?;
    m.add_class::<SurvivalTransformerModel>()?;

    m.add_function(wrap_pyfunction!(fit_recurrent_surv, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_longitudinal_surv, &m)?)?;
    m.add_class::<RecurrentSurvConfig>()?;
    m.add_class::<RecurrentSurvModel>()?;
    m.add_class::<LongitudinalSurvConfig>()?;
    m.add_class::<LongitudinalSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_dysurv, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_risk_prediction, &m)?)?;
    m.add_class::<DySurvConfig>()?;
    m.add_class::<DySurvModel>()?;
    m.add_class::<DynamicRiskResult>()?;

    m.add_function(wrap_pyfunction!(fit_deep_pamm, &m)?)?;
    m.add_class::<DeepPAMMConfig>()?;
    m.add_class::<DeepPAMMModel>()?;

    m.add_function(wrap_pyfunction!(fit_neural_ode_surv, &m)?)?;
    m.add_class::<NeuralODESurvConfig>()?;
    m.add_class::<NeuralODESurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_attention_cox, &m)?)?;
    m.add_class::<AttentionCoxConfig>()?;
    m.add_class::<AttentionCoxModel>()?;

    m.add_function(wrap_pyfunction!(fit_multimodal_surv, &m)?)?;
    m.add_class::<FusionStrategy>()?;
    m.add_class::<MultimodalSurvConfig>()?;
    m.add_class::<MultimodalSurvModel>()?;

    m.add_function(wrap_pyfunction!(mc_dropout_uncertainty, &m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_uncertainty, &m)?)?;
    m.add_function(wrap_pyfunction!(quantile_regression_intervals, &m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_prediction_intervals, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_bootstrap_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_plus_survival, &m)?)?;
    m.add_class::<MCDropoutConfig>()?;
    m.add_class::<UncertaintyResult>()?;
    m.add_class::<EnsembleUncertaintyResult>()?;
    m.add_class::<QuantileRegressionResult>()?;
    m.add_class::<CalibrationUncertaintyResult>()?;
    m.add_class::<ConformalSurvivalConfig>()?;
    m.add_class::<ConformalSurvivalResult>()?;
    m.add_class::<BayesianBootstrapConfig>()?;
    m.add_class::<BayesianBootstrapResult>()?;
    m.add_class::<JackknifePlusConfig>()?;
    m.add_class::<JackknifePlusResult>()?;

    m.add_function(wrap_pyfunction!(advanced_calibration_metrics, &m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_calibration, &m)?)?;
    m.add_class::<AdvancedCalibrationResult>()?;
    m.add_class::<TimeDependentCalibrationResult>()?;

    m.add_function(wrap_pyfunction!(survival_meta_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_forest_plot_data, &m)?)?;
    m.add_function(wrap_pyfunction!(publication_bias_tests, &m)?)?;
    m.add_class::<MetaAnalysisConfig>()?;
    m.add_class::<MetaAnalysisResult>()?;
    m.add_class::<MetaForestPlotData>()?;
    m.add_class::<PublicationBiasResult>()?;

    m.add_function(wrap_pyfunction!(flexible_parametric_model, &m)?)?;
    m.add_function(wrap_pyfunction!(restricted_cubic_spline, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_hazard_spline, &m)?)?;
    m.add_class::<SplineConfig>()?;
    m.add_class::<FlexibleParametricResult>()?;
    m.add_class::<RestrictedCubicSplineResult>()?;
    m.add_class::<HazardSplineResult>()?;

    m.add_function(wrap_pyfunction!(compute_fairness_metrics, &m)?)?;
    m.add_function(wrap_pyfunction!(assess_model_robustness, &m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_analysis, &m)?)?;
    m.add_class::<FairnessMetrics>()?;
    m.add_class::<RobustnessResult>()?;
    m.add_class::<SubgroupAnalysisResult>()?;

    m.add_function(wrap_pyfunction!(hyperparameter_search, &m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_models, &m)?)?;
    m.add_function(wrap_pyfunction!(nested_cross_validation, &m)?)?;
    m.add_class::<SearchStrategy>()?;
    m.add_class::<HyperparameterSearchConfig>()?;
    m.add_class::<HyperparameterResult>()?;
    m.add_class::<BenchmarkResult>()?;
    m.add_class::<NestedCVResult>()?;

    m.add_function(wrap_pyfunction!(compute_model_selection_criteria, &m)?)?;
    m.add_function(wrap_pyfunction!(compare_models, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_cv_score, &m)?)?;
    m.add_class::<ModelSelectionCriteria>()?;
    m.add_class::<SurvivalModelComparison>()?;
    m.add_class::<CrossValidatedScore>()?;

    m.add_function(wrap_pyfunction!(pretrain_survival_model, &m)?)?;
    m.add_function(wrap_pyfunction!(transfer_survival_model, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_domain_distance, &m)?)?;
    m.add_class::<TransferStrategy>()?;
    m.add_class::<TransferLearningConfig>()?;
    m.add_class::<PretrainedSurvivalModel>()?;
    m.add_class::<TransferredModel>()?;
    m.add_class::<DomainAdaptationResult>()?;

    m.add_function(wrap_pyfunction!(fit_graph_surv, &m)?)?;
    m.add_class::<GraphSurvConfig>()?;
    m.add_class::<GraphSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_mamba_surv, &m)?)?;
    m.add_class::<MambaSurvConfig>()?;
    m.add_class::<MambaSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_temporal_fusion_transformer, &m)?)?;
    m.add_class::<TFTConfig>()?;
    m.add_class::<TemporalFusionTransformer>()?;

    m.add_function(wrap_pyfunction!(double_ml_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(double_ml_cate, &m)?)?;
    m.add_class::<DoubleMLConfig>()?;
    m.add_class::<DoubleMLResult>()?;
    m.add_class::<CATEResult>()?;

    m.add_function(wrap_pyfunction!(decision_curve_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(clinical_utility_at_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(compare_decision_curves, &m)?)?;
    m.add_class::<DecisionCurveResult>()?;
    m.add_class::<ClinicalUtilityResult>()?;

    m.add_function(wrap_pyfunction!(warranty_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(renewal_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(reliability_growth, &m)?)?;
    m.add_class::<WarrantyConfig>()?;
    m.add_class::<WarrantyResult>()?;
    m.add_class::<RenewalResult>()?;
    m.add_class::<ReliabilityGrowthResult>()?;

    m.add_function(wrap_pyfunction!(distill_survival_model, &m)?)?;
    m.add_function(wrap_pyfunction!(prune_survival_model, &m)?)?;
    m.add_class::<DistillationConfig>()?;
    m.add_class::<DistilledSurvivalModel>()?;
    m.add_class::<DistillationResult>()?;
    m.add_class::<PruningResult>()?;
    m.add_class::<ModelComparisonResult>()?;

    m.add_function(wrap_pyfunction!(federated_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(secure_aggregate, &m)?)?;
    m.add_class::<FederatedConfig>()?;
    m.add_class::<FederatedSurvivalResult>()?;
    m.add_class::<SecureAggregationResult>()?;
    m.add_class::<PrivacyAccountant>()?;

    m.add_class::<StreamingCoxConfig>()?;
    m.add_class::<StreamingCoxModel>()?;
    m.add_class::<StreamingKaplanMeier>()?;
    m.add_class::<ConceptDriftDetector>()?;

    m.add_function(wrap_pyfunction!(dp_kaplan_meier, &m)?)?;
    m.add_function(wrap_pyfunction!(dp_cox_regression, &m)?)?;
    m.add_function(wrap_pyfunction!(dp_histogram, &m)?)?;
    m.add_function(wrap_pyfunction!(local_dp_mean, &m)?)?;
    m.add_class::<DPConfig>()?;
    m.add_class::<DPSurvivalResult>()?;
    m.add_class::<DPCoxResult>()?;
    m.add_class::<DPHistogramResult>()?;
    m.add_class::<LocalDPResult>()?;

    m.add_function(wrap_pyfunction!(super_learner_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(stacking_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(componentwise_boosting, &m)?)?;
    m.add_function(wrap_pyfunction!(blending_survival, &m)?)?;
    m.add_class::<SuperLearnerConfig>()?;
    m.add_class::<SuperLearnerResult>()?;
    m.add_class::<StackingConfig>()?;
    m.add_class::<StackingResult>()?;
    m.add_class::<ComponentwiseBoostingConfig>()?;
    m.add_class::<ComponentwiseBoostingResult>()?;
    m.add_class::<BlendingResult>()?;

    m.add_function(wrap_pyfunction!(active_learning_selection, &m)?)?;
    m.add_function(wrap_pyfunction!(query_by_committee, &m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_logrank, &m)?)?;
    m.add_function(wrap_pyfunction!(power_logrank, &m)?)?;
    m.add_function(wrap_pyfunction!(group_sequential_analysis, &m)?)?;
    m.add_class::<ActiveLearningConfig>()?;
    m.add_class::<ActiveLearningResult>()?;
    m.add_class::<QBCResult>()?;
    m.add_class::<LogrankSampleSizeResult>()?;
    m.add_class::<LogrankPowerResult>()?;
    m.add_class::<AdaptiveDesignResult>()?;

    m.add_function(wrap_pyfunction!(joint_longitudinal_model, &m)?)?;
    m.add_function(wrap_pyfunction!(landmark_cox_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(longitudinal_dynamic_pred, &m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_cox, &m)?)?;
    m.add_class::<JointModelConfig>()?;
    m.add_class::<JointLongSurvResult>()?;
    m.add_class::<LandmarkAnalysisResult>()?;
    m.add_class::<LongDynamicPredResult>()?;
    m.add_class::<TimeVaryingCoxResult>()?;

    m.add_function(wrap_pyfunction!(km_plot_data, &m)?)?;
    m.add_function(wrap_pyfunction!(forest_plot_data, &m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot_data, &m)?)?;
    m.add_function(wrap_pyfunction!(generate_survival_report, &m)?)?;
    m.add_function(wrap_pyfunction!(roc_plot_data, &m)?)?;
    m.add_class::<KaplanMeierPlotData>()?;
    m.add_class::<ForestPlotData>()?;
    m.add_class::<CalibrationCurveData>()?;
    m.add_class::<SurvivalReport>()?;
    m.add_class::<ROCPlotData>()?;

    m.add_function(wrap_pyfunction!(qaly_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(qaly_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(incremental_cost_effectiveness, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_sensitivity, &m)?)?;
    m.add_class::<QALYResult>()?;
    m.add_class::<QTWISTResult>()?;

    m.add_function(wrap_pyfunction!(gap_time_model, &m)?)?;
    m.add_function(wrap_pyfunction!(pwp_gap_time, &m)?)?;
    m.add_function(wrap_pyfunction!(joint_frailty_model, &m)?)?;
    m.add_function(wrap_pyfunction!(andersen_gill, &m)?)?;
    m.add_function(wrap_pyfunction!(marginal_recurrent_model, &m)?)?;
    m.add_function(wrap_pyfunction!(wei_lin_weissfeld, &m)?)?;
    m.add_class::<GapTimeResult>()?;
    m.add_class::<JointFrailtyResult>()?;
    m.add_class::<MarginalModelResult>()?;
    m.add_class::<FrailtyDistribution>()?;
    m.add_class::<MarginalMethod>()?;

    m.add_function(wrap_pyfunction!(mixture_cure_model, &m)?)?;
    m.add_function(wrap_pyfunction!(promotion_time_cure_model, &m)?)?;
    m.add_function(wrap_pyfunction!(bounded_cumulative_hazard_model, &m)?)?;
    m.add_function(wrap_pyfunction!(non_mixture_cure_model, &m)?)?;
    m.add_function(wrap_pyfunction!(compare_cure_models, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_bounded_cumulative_hazard, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_non_mixture_survival, &m)?)?;
    m.add_class::<MixtureCureResult>()?;
    m.add_class::<PromotionTimeCureResult>()?;
    m.add_class::<CureDistribution>()?;
    m.add_class::<BoundedCumulativeHazardConfig>()?;
    m.add_class::<BoundedCumulativeHazardResult>()?;
    m.add_class::<NonMixtureType>()?;
    m.add_class::<NonMixtureCureConfig>()?;
    m.add_class::<NonMixtureCureResult>()?;
    m.add_class::<CureModelComparisonResult>()?;

    m.add_function(wrap_pyfunction!(elastic_net_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_cv, &m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_path, &m)?)?;
    m.add_class::<ElasticNetCoxResult>()?;
    m.add_class::<ElasticNetCoxPath>()?;

    m.add_function(wrap_pyfunction!(fast_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(fast_cox_path, &m)?)?;
    m.add_function(wrap_pyfunction!(fast_cox_cv, &m)?)?;
    m.add_class::<FastCoxConfig>()?;
    m.add_class::<FastCoxResult>()?;
    m.add_class::<FastCoxPath>()?;
    m.add_class::<ScreeningRule>()?;

    m.add_function(wrap_pyfunction!(group_lasso_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(sparse_boosting_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(sis_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(stability_selection_cox, &m)?)?;
    m.add_class::<GroupLassoConfig>()?;
    m.add_class::<GroupLassoResult>()?;
    m.add_class::<SparseBoostingConfig>()?;
    m.add_class::<SparseBoostingResult>()?;
    m.add_class::<SISConfig>()?;
    m.add_class::<SISResult>()?;
    m.add_class::<StabilitySelectionConfig>()?;
    m.add_class::<StabilitySelectionResult>()?;

    m.add_function(wrap_pyfunction!(cause_specific_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(cause_specific_cox_all, &m)?)?;
    m.add_class::<CauseSpecificCoxConfig>()?;
    m.add_class::<CauseSpecificCoxResult>()?;
    m.add_class::<CensoringType>()?;

    m.add_function(wrap_pyfunction!(joint_competing_risks, &m)?)?;
    m.add_class::<JointCompetingRisksConfig>()?;
    m.add_class::<JointCompetingRisksResult>()?;
    m.add_class::<CauseResult>()?;
    m.add_class::<CorrelationType>()?;

    m.add_function(wrap_pyfunction!(pwp_model, &m)?)?;
    m.add_function(wrap_pyfunction!(wlw_model, &m)?)?;
    m.add_function(wrap_pyfunction!(negative_binomial_frailty, &m)?)?;
    m.add_function(wrap_pyfunction!(anderson_gill_model, &m)?)?;
    m.add_class::<PWPConfig>()?;
    m.add_class::<PWPResult>()?;
    m.add_class::<PWPTimescale>()?;
    m.add_class::<WLWConfig>()?;
    m.add_class::<WLWResult>()?;
    m.add_class::<NegativeBinomialFrailtyConfig>()?;
    m.add_class::<NegativeBinomialFrailtyResult>()?;
    m.add_class::<AndersonGillResult>()?;

    m.add_function(wrap_pyfunction!(survfitaj_extended, &m)?)?;
    m.add_class::<AalenJohansenExtendedConfig>()?;
    m.add_class::<AalenJohansenExtendedResult>()?;
    m.add_class::<TransitionMatrix>()?;
    m.add_class::<TransitionType>()?;
    m.add_class::<VarianceEstimator>()?;

    m.add_function(wrap_pyfunction!(estimate_transition_intensities, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_multi_state_model, &m)?)?;
    m.add_function(wrap_pyfunction!(fit_markov_msm, &m)?)?;
    m.add_class::<MultiStateConfig>()?;
    m.add_class::<MultiStateResult>()?;
    m.add_class::<TransitionIntensityResult>()?;
    m.add_class::<MarkovMSMResult>()?;

    m.add_function(wrap_pyfunction!(fit_semi_markov, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_semi_markov, &m)?)?;
    m.add_class::<SemiMarkovConfig>()?;
    m.add_class::<SemiMarkovResult>()?;
    m.add_class::<SemiMarkovPrediction>()?;
    m.add_class::<SojournDistribution>()?;
    m.add_class::<SojournTimeParams>()?;

    m.add_function(wrap_pyfunction!(fit_illness_death, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_illness_death, &m)?)?;
    m.add_class::<IllnessDeathConfig>()?;
    m.add_class::<IllnessDeathResult>()?;
    m.add_class::<IllnessDeathPrediction>()?;
    m.add_class::<IllnessDeathType>()?;
    m.add_class::<TransitionHazard>()?;

    m.add_function(wrap_pyfunction!(relative_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(net_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(crude_probability_of_death, &m)?)?;
    m.add_function(wrap_pyfunction!(excess_hazard_regression, &m)?)?;
    m.add_class::<RelativeSurvivalResult>()?;
    m.add_class::<NetSurvivalResult>()?;
    m.add_class::<ExcessHazardModelResult>()?;
    m.add_class::<NetSurvivalMethod>()?;

    m.add_function(wrap_pyfunction!(spatial_frailty_model, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_spatial_smoothed_rates, &m)?)?;
    m.add_function(wrap_pyfunction!(moran_i_test, &m)?)?;
    m.add_class::<SpatialFrailtyResult>()?;
    m.add_class::<SpatialCorrelationStructure>()?;

    m.add_function(wrap_pyfunction!(network_survival_model, &m)?)?;
    m.add_function(wrap_pyfunction!(diffusion_survival_model, &m)?)?;
    m.add_function(wrap_pyfunction!(network_heterogeneity_survival, &m)?)?;
    m.add_class::<CentralityType>()?;
    m.add_class::<NetworkSurvivalConfig>()?;
    m.add_class::<NetworkSurvivalResult>()?;
    m.add_class::<DiffusionSurvivalConfig>()?;
    m.add_class::<DiffusionSurvivalResult>()?;
    m.add_class::<NetworkHeterogeneityResult>()?;

    m.add_function(wrap_pyfunction!(survshap, &m)?)?;
    m.add_function(wrap_pyfunction!(survshap_from_model, &m)?)?;
    m.add_function(wrap_pyfunction!(survshap_bootstrap, &m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survshap, &m)?)?;
    m.add_function(wrap_pyfunction!(permutation_importance, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_shap_interactions, &m)?)?;
    m.add_class::<SurvShapConfig>()?;
    m.add_class::<SurvShapResult>()?;
    m.add_class::<SurvShapExplanation>()?;
    m.add_class::<AggregationMethod>()?;
    m.add_class::<BootstrapSurvShapResult>()?;
    m.add_class::<PermutationImportanceResult>()?;
    m.add_class::<ShapInteractionResult>()?;
    m.add_class::<FeatureImportance>()?;

    m.add_function(wrap_pyfunction!(detect_time_varying_features, &m)?)?;
    m.add_class::<TimeVaryingTestConfig>()?;
    m.add_class::<TimeVaryingTestResult>()?;
    m.add_class::<TimeVaryingAnalysis>()?;
    m.add_class::<TimeVaryingTestType>()?;

    m.add_function(wrap_pyfunction!(detect_changepoints, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_changepoints_single_series, &m)?)?;
    m.add_class::<ChangepointConfig>()?;
    m.add_class::<ChangepointResult>()?;
    m.add_class::<Changepoint>()?;
    m.add_class::<AllChangepointsResult>()?;
    m.add_class::<ChangepointMethod>()?;
    m.add_class::<CostFunction>()?;

    m.add_function(wrap_pyfunction!(group_variables, &m)?)?;
    m.add_class::<VariableGroupingConfig>()?;
    m.add_class::<VariableGroupingResult>()?;
    m.add_class::<FeatureGroup>()?;
    m.add_class::<GroupingMethod>()?;
    m.add_class::<LinkageType>()?;

    m.add_function(wrap_pyfunction!(analyze_local_global, &m)?)?;
    m.add_class::<LocalGlobalConfig>()?;
    m.add_class::<LocalGlobalResult>()?;
    m.add_class::<LocalGlobalSummary>()?;
    m.add_class::<FeatureViewAnalysis>()?;
    m.add_class::<ViewRecommendation>()?;

    m.add_function(wrap_pyfunction!(compute_ale, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_ale_2d, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_time_varying_ale, &m)?)?;
    m.add_class::<ALEResult>()?;
    m.add_class::<ALE2DResult>()?;

    m.add_function(wrap_pyfunction!(compute_friedman_h, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_all_pairwise_interactions, &m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_feature_importance_decomposition,
        &m
    )?)?;
    m.add_class::<FriedmanHResult>()?;
    m.add_class::<FriedmanFeatureImportanceResult>()?;

    m.add_function(wrap_pyfunction!(compute_ice, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_dice, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_survival_ice, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_heterogeneity, &m)?)?;
    m.add_function(wrap_pyfunction!(cluster_ice_curves, &m)?)?;
    m.add_class::<ICEResult>()?;
    m.add_class::<DICEResult>()?;

    m.add_function(wrap_pyfunction!(conformal_calibrate, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_predict, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival_from_predictions, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_coverage_test, &m)?)?;
    m.add_function(wrap_pyfunction!(doubly_robust_conformal_calibrate, &m)?)?;
    m.add_function(wrap_pyfunction!(doubly_robust_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_calibrate, &m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_predict, &m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(conformalized_survival_distribution, &m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(cqr_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_calibration_plot, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_width_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_coverage_cv, &m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival_parallel, &m)?)?;
    m.add_class::<ConformalCalibrationResult>()?;
    m.add_class::<ConformalPredictionResult>()?;
    m.add_class::<ConformalDiagnostics>()?;
    m.add_class::<DoublyRobustConformalResult>()?;
    m.add_class::<TwoSidedCalibrationResult>()?;
    m.add_class::<TwoSidedConformalResult>()?;
    m.add_class::<ConformalSurvivalDistribution>()?;
    m.add_class::<BootstrapConformalResult>()?;
    m.add_class::<CQRConformalResult>()?;
    m.add_class::<ConformalCalibrationPlot>()?;
    m.add_class::<ConformalWidthAnalysis>()?;
    m.add_class::<CoverageSelectionResult>()?;

    m.add_function(wrap_pyfunction!(covariate_shift_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(cvplus_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(cvplus_conformal_calibrate, &m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_calibrate, &m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_predict, &m)?)?;
    m.add_class::<CovariateShiftConformalResult>()?;
    m.add_class::<WeightDiagnostics>()?;
    m.add_class::<CVPlusConformalResult>()?;
    m.add_class::<CVPlusCalibrationResult>()?;
    m.add_class::<MondrianConformalResult>()?;
    m.add_class::<MondrianCalibrationResult>()?;
    m.add_class::<MondrianDiagnostics>()?;

    m.add_function(wrap_pyfunction!(fpca_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(functional_cox, &m)?)?;
    m.add_class::<BasisType>()?;
    m.add_class::<FunctionalSurvivalConfig>()?;
    m.add_class::<FunctionalPCAResult>()?;
    m.add_class::<FunctionalSurvivalResult>()?;

    m.add_function(wrap_pyfunction!(dro_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(robustness_analysis, &m)?)?;
    m.add_class::<UncertaintySet>()?;
    m.add_class::<DROSurvivalConfig>()?;
    m.add_class::<DROSurvivalResult>()?;
    m.add_class::<RobustnessAnalysis>()?;

    m.add_function(wrap_pyfunction!(generate_adversarial_examples, &m)?)?;
    m.add_function(wrap_pyfunction!(adversarial_training_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_robustness, &m)?)?;
    m.add_class::<AttackType>()?;
    m.add_class::<DefenseType>()?;
    m.add_class::<AdversarialAttackConfig>()?;
    m.add_class::<AdversarialDefenseConfig>()?;
    m.add_class::<AdversarialExample>()?;
    m.add_class::<AdversarialAttackResult>()?;
    m.add_class::<RobustSurvivalModel>()?;
    m.add_class::<RobustnessEvaluation>()?;

    m.add_function(wrap_pyfunction!(get_available_devices, &m)?)?;
    m.add_function(wrap_pyfunction!(is_gpu_available, &m)?)?;
    m.add_function(wrap_pyfunction!(parallel_cox_regression, &m)?)?;
    m.add_function(wrap_pyfunction!(batch_predict_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(parallel_matrix_operations, &m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_compute_backend, &m)?)?;
    m.add_class::<ComputeBackend>()?;
    m.add_class::<GPUConfig>()?;
    m.add_class::<DeviceInfo>()?;
    m.add_class::<ParallelCoxResult>()?;
    m.add_class::<BatchPredictionResult>()?;

    Ok(())
}
