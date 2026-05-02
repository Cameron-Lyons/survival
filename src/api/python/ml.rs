use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survival_forest, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_boost_survival, m)?)?;
    m.add_function(wrap_pyfunction!(deep_surv, m)?)?;
    m.add_function(wrap_pyfunction!(deephit, m)?)?;
    m.add_function(wrap_pyfunction!(survtrace, m)?)?;
    m.add_class::<SurvivalForest>()?;
    m.add_class::<SurvivalForestInput>()?;
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
    m.add_function(wrap_pyfunction!(tracer, m)?)?;
    m.add_class::<Tracer>()?;
    m.add_class::<TracerConfig>()?;
    m.add_function(wrap_pyfunction!(dynamic_deephit, m)?)?;
    m.add_class::<DynamicDeepHit>()?;
    m.add_class::<DynamicDeepHitConfig>()?;
    m.add_class::<TemporalType>()?;
    m.add_function(wrap_pyfunction!(contrastive_surv, m)?)?;
    m.add_class::<ContrastiveSurv>()?;
    m.add_class::<ContrastiveSurvConfig>()?;
    m.add_class::<ContrastiveSurvResult>()?;
    m.add_class::<SurvivalLossType>()?;
    m.add_function(wrap_pyfunction!(galee, m)?)?;
    m.add_class::<GALEE>()?;
    m.add_class::<GALEEConfig>()?;
    m.add_class::<GALEEResult>()?;
    m.add_class::<UnimodalConstraint>()?;

    m.add_function(wrap_pyfunction!(fit_cox_time, m)?)?;
    m.add_class::<CoxTimeConfig>()?;
    m.add_class::<CoxTimeModel>()?;

    m.add_function(wrap_pyfunction!(fit_neural_mtlr, m)?)?;
    m.add_class::<NeuralMTLRConfig>()?;
    m.add_class::<NeuralMTLRModel>()?;

    m.add_function(wrap_pyfunction!(fit_survival_transformer, m)?)?;
    m.add_class::<SurvivalTransformerConfig>()?;
    m.add_class::<SurvivalTransformerModel>()?;

    m.add_function(wrap_pyfunction!(fit_recurrent_surv, m)?)?;
    m.add_function(wrap_pyfunction!(fit_longitudinal_surv, m)?)?;
    m.add_class::<RecurrentSurvConfig>()?;
    m.add_class::<RecurrentSurvModel>()?;
    m.add_class::<LongitudinalSurvConfig>()?;
    m.add_class::<LongitudinalSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_dysurv, m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_risk_prediction, m)?)?;
    m.add_class::<DySurvConfig>()?;
    m.add_class::<DySurvModel>()?;
    m.add_class::<DynamicRiskResult>()?;

    m.add_function(wrap_pyfunction!(fit_deep_pamm, m)?)?;
    m.add_class::<DeepPAMMConfig>()?;
    m.add_class::<DeepPAMMModel>()?;

    m.add_function(wrap_pyfunction!(fit_neural_ode_surv, m)?)?;
    m.add_class::<NeuralODESurvConfig>()?;
    m.add_class::<NeuralODESurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_attention_cox, m)?)?;
    m.add_class::<AttentionCoxConfig>()?;
    m.add_class::<AttentionCoxModel>()?;

    m.add_function(wrap_pyfunction!(fit_multimodal_surv, m)?)?;
    m.add_class::<FusionStrategy>()?;
    m.add_class::<MultimodalSurvConfig>()?;
    m.add_class::<MultimodalSurvModel>()?;

    m.add_function(wrap_pyfunction!(mc_dropout_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_regression_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_prediction_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_bootstrap_survival, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_plus_survival, m)?)?;
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

    m.add_function(wrap_pyfunction!(advanced_calibration_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_calibration, m)?)?;
    m.add_class::<AdvancedCalibrationResult>()?;
    m.add_class::<TimeDependentCalibrationResult>()?;

    m.add_function(wrap_pyfunction!(survival_meta_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(generate_forest_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(publication_bias_tests, m)?)?;
    m.add_class::<MetaAnalysisConfig>()?;
    m.add_class::<MetaAnalysisResult>()?;
    m.add_class::<MetaForestPlotData>()?;
    m.add_class::<PublicationBiasResult>()?;

    m.add_function(wrap_pyfunction!(flexible_parametric_model, m)?)?;
    m.add_function(wrap_pyfunction!(restricted_cubic_spline, m)?)?;
    m.add_function(wrap_pyfunction!(predict_hazard_spline, m)?)?;
    m.add_class::<SplineConfig>()?;
    m.add_class::<FlexibleParametricResult>()?;
    m.add_class::<RestrictedCubicSplineResult>()?;
    m.add_class::<HazardSplineResult>()?;

    m.add_function(wrap_pyfunction!(compute_fairness_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(assess_model_robustness, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_analysis, m)?)?;
    m.add_class::<FairnessMetrics>()?;
    m.add_class::<RobustnessResult>()?;
    m.add_class::<SubgroupAnalysisResult>()?;

    m.add_function(wrap_pyfunction!(hyperparameter_search, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_models, m)?)?;
    m.add_function(wrap_pyfunction!(nested_cross_validation, m)?)?;
    m.add_class::<SearchStrategy>()?;
    m.add_class::<HyperparameterSearchConfig>()?;
    m.add_class::<HyperparameterResult>()?;
    m.add_class::<BenchmarkResult>()?;
    m.add_class::<NestedCVResult>()?;

    m.add_function(wrap_pyfunction!(compute_model_selection_criteria, m)?)?;
    m.add_function(wrap_pyfunction!(compare_models, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cv_score, m)?)?;
    m.add_class::<ModelSelectionCriteria>()?;
    m.add_class::<SurvivalModelComparison>()?;
    m.add_class::<CrossValidatedScore>()?;

    m.add_function(wrap_pyfunction!(pretrain_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_survival_model, m)?)?;
    m.add_function(wrap_pyfunction!(compute_domain_distance, m)?)?;
    m.add_class::<TransferStrategy>()?;
    m.add_class::<TransferLearningConfig>()?;
    m.add_class::<PretrainedSurvivalModel>()?;
    m.add_class::<TransferredModel>()?;
    m.add_class::<DomainAdaptationResult>()?;

    m.add_function(wrap_pyfunction!(fit_graph_surv, m)?)?;
    m.add_class::<GraphSurvConfig>()?;
    m.add_class::<GraphSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_mamba_surv, m)?)?;
    m.add_class::<MambaSurvConfig>()?;
    m.add_class::<MambaSurvModel>()?;

    m.add_function(wrap_pyfunction!(fit_temporal_fusion_transformer, m)?)?;
    m.add_class::<TFTConfig>()?;
    m.add_class::<TemporalFusionTransformer>()?;

    Ok(())
}
