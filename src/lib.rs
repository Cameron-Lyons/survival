use pyo3::prelude::*;
mod bayesian;
mod causal;
mod concordance;
mod constants;
mod core;
mod datasets;
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
mod spatial;
mod specialized;
mod surv_analysis;
mod tests;
mod utilities;
mod validation;

pub use concordance::basic::concordance as compute_concordance;
pub use concordance::concordance1::perform_concordance1_calculation;
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
pub use regression::ridge::{RidgePenalty, RidgeResult, ridge_cv, ridge_fit};
pub use regression::survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, predict_survreg, predict_survreg_quantile,
};
pub use regression::survreg6::{DistributionType, SurvivalFit, SurvregConfig, survreg};
pub use residuals::agmart::agmart;
pub use residuals::coxmart::coxmart;
pub use residuals::survfit_resid::{SurvfitResiduals, residuals_survfit};
pub use residuals::survreg_resid::{SurvregResiduals, dfbeta_survreg, residuals_survreg};
pub use scoring::agscore2::perform_score_calculation;
pub use scoring::agscore3::perform_agscore3_calculation;
pub use scoring::coxscore2::cox_score_residuals;
pub use specialized::brier::{brier, integrated_brier};
pub use specialized::cch::{CchMethod, CohortData};
pub use specialized::cipoisson::{cipoisson, cipoisson_anscombe, cipoisson_exact};
pub use specialized::finegray::{FineGrayOutput, finegray};
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
pub use surv_analysis::aggregate_survfit::{
    AggregateSurvfitResult, aggregate_survfit, aggregate_survfit_by_group,
};
pub use surv_analysis::agsurv4::agsurv4;
pub use surv_analysis::agsurv5::agsurv5;
pub use surv_analysis::nelson_aalen::{
    NelsonAalenResult, StratifiedKMResult, nelson_aalen_estimator, stratified_kaplan_meier,
};
pub use surv_analysis::pseudo::{PseudoResult, pseudo, pseudo_fast};
pub use surv_analysis::survdiff2::{SurvDiffResult, survdiff2};
pub use surv_analysis::survfit_matrix::{
    SurvfitMatrixResult, basehaz, survfit_from_cumhaz, survfit_from_hazard, survfit_from_matrix,
    survfit_multistate,
};
pub use surv_analysis::survfitaj::{SurvFitAJ, survfitaj};
pub use surv_analysis::survfitkm::{
    KaplanMeierConfig, SurvFitKMOutput, SurvfitKMOptions, survfitkm, survfitkm_with_options,
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
    CalibrationResult, PredictionResult, RiskStratificationResult, TdAUCResult, calibration,
    predict_cox, risk_stratification, td_auc,
};
pub use validation::crossval::{CVResult, cv_cox_concordance, cv_survreg_loglik};
pub use validation::landmark::{
    ConditionalSurvivalResult, HazardRatioResult, LandmarkResult, LifeTableResult,
    SurvivalAtTimeResult, conditional_survival, hazard_ratio, landmark_analysis,
    landmark_analysis_batch, life_table, survival_at_times,
};
pub use validation::logrank::{
    LogRankResult, TrendTestResult, fleming_harrington_test, logrank_test, logrank_trend,
};
pub use validation::power::{
    AccrualResult, SampleSizeResult, expected_events, power_survival, sample_size_survival,
    sample_size_survival_freedman,
};
pub use validation::rmst::{
    CumulativeIncidenceResult, MedianSurvivalResult, NNTResult, RMSTComparisonResult, RMSTResult,
    cumulative_incidence, number_needed_to_treat, rmst, rmst_comparison, survival_quantile,
};
pub use validation::royston::{RoystonResult, royston, royston_from_model};
pub use validation::survcheck::{SurvCheckResult, survcheck, survcheck_simple};
pub use validation::survobrien::{SurvObrienResult, survobrien};
pub use validation::tests::{
    ProportionalityTest, TestResult, lrt_test, ph_test, score_test, wald_test,
};
use validation::tests::{score_test_py, wald_test_py};
pub use validation::yates::{
    YatesPairwiseResult, YatesResult, yates, yates_contrast, yates_pairwise,
};

pub use bayesian::bayesian_cox::{BayesianCoxResult, bayesian_cox, bayesian_cox_predict_survival};
pub use bayesian::bayesian_parametric::{
    BayesianParametricResult, bayesian_parametric, bayesian_parametric_predict,
};
pub use causal::g_computation::{GComputationResult, g_computation, g_computation_survival_curves};
pub use causal::ipcw::{
    IPCWResult, compute_ipcw_weights, ipcw_kaplan_meier, ipcw_treatment_effect,
};
pub use causal::msm::{MSMResult, compute_longitudinal_iptw, marginal_structural_model};
pub use causal::target_trial::{
    TargetTrialResult, sequential_trial_emulation, target_trial_emulation,
};
pub use interval::interval_censoring::{
    IntervalCensoredResult, IntervalDistribution, TurnbullResult, interval_censored_regression,
    npmle_interval, turnbull_estimator,
};
pub use joint::dynamic_prediction::{
    DynamicPredictionResult, dynamic_auc, dynamic_brier_score, dynamic_prediction,
    landmarking_analysis,
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
pub use ml::gradient_boost::{GradientBoostSurvival, gradient_boost_survival};
pub use ml::survival_forest::{SurvivalForest, survival_forest};
pub use qol::qaly::{
    QALYResult, incremental_cost_effectiveness, qaly_calculation, qaly_comparison,
};
pub use qol::qtwist::{QTWISTResult, qtwist_analysis, qtwist_comparison, qtwist_sensitivity};
pub use recurrent::gap_time::{GapTimeResult, gap_time_model, pwp_gap_time};
pub use recurrent::joint_frailty::{FrailtyDistribution, JointFrailtyResult, joint_frailty_model};
pub use recurrent::marginal_models::{
    MarginalMethod, MarginalModelResult, andersen_gill, marginal_recurrent_model, wei_lin_weissfeld,
};
pub use regression::cure_models::{
    CureDistribution, MixtureCureResult, PromotionTimeCureResult, mixture_cure_model,
    promotion_time_cure_model,
};
pub use regression::elastic_net::{
    ElasticNetCoxPath, ElasticNetCoxResult, elastic_net_cox, elastic_net_cox_cv,
    elastic_net_cox_path,
};
pub use relative::net_survival::{
    NetSurvivalMethod, NetSurvivalResult, crude_probability_of_death, net_survival,
};
pub use relative::relative_survival::{
    ExcessHazardModelResult, RelativeSurvivalResult, excess_hazard_regression, relative_survival,
};
pub use spatial::spatial_frailty::{
    SpatialCorrelationStructure, SpatialFrailtyResult, compute_spatial_smoothed_rates,
    moran_i_test, spatial_frailty_model,
};

#[pymodule]
fn survival(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(survfitkm, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_with_options, &m)?)?;
    m.add_function(wrap_pyfunction!(survfitaj, &m)?)?;
    m.add_function(wrap_pyfunction!(survdiff2, &m)?)?;
    m.add_function(wrap_pyfunction!(finegray, &m)?)?;
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
    m.add_function(wrap_pyfunction!(rmst, &m)?)?;
    m.add_function(wrap_pyfunction!(rmst_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(survival_quantile, &m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_incidence, &m)?)?;
    m.add_function(wrap_pyfunction!(number_needed_to_treat, &m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(conditional_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(hazard_ratio, &m)?)?;
    m.add_function(wrap_pyfunction!(survival_at_times, &m)?)?;
    m.add_function(wrap_pyfunction!(life_table, &m)?)?;
    // New utility functions
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
    // New specialized functions
    m.add_function(wrap_pyfunction!(survexp, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_individual, &m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_ratetable, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_matplotlib_code, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_transition_matrix, &m)?)?;
    m.add_function(wrap_pyfunction!(statefig_validate, &m)?)?;
    // New survival analysis functions
    m.add_function(wrap_pyfunction!(pseudo, &m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_fast, &m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit, &m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit_by_group, &m)?)?;
    // New validation functions
    m.add_function(wrap_pyfunction!(survcheck, &m)?)?;
    m.add_function(wrap_pyfunction!(survcheck_simple, &m)?)?;
    m.add_function(wrap_pyfunction!(royston, &m)?)?;
    m.add_function(wrap_pyfunction!(royston_from_model, &m)?)?;
    m.add_function(wrap_pyfunction!(yates, &m)?)?;
    m.add_function(wrap_pyfunction!(yates_contrast, &m)?)?;
    m.add_function(wrap_pyfunction!(yates_pairwise, &m)?)?;
    // New regression/core functions
    m.add_function(wrap_pyfunction!(ridge_fit, &m)?)?;
    m.add_function(wrap_pyfunction!(ridge_cv, &m)?)?;
    m.add_function(wrap_pyfunction!(nsk, &m)?)?;
    // ANOVA functions
    m.add_function(wrap_pyfunction!(anova_coxph, &m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph_single, &m)?)?;
    // Reliability functions
    m.add_function(wrap_pyfunction!(reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(reliability_inverse, &m)?)?;
    m.add_function(wrap_pyfunction!(hazard_to_reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(failure_probability, &m)?)?;
    m.add_function(wrap_pyfunction!(conditional_reliability, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_residual_life, &m)?)?;
    // Survfit matrix functions
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
    m.add_class::<RMSTResult>()?;
    m.add_class::<RMSTComparisonResult>()?;
    m.add_class::<MedianSurvivalResult>()?;
    m.add_class::<CumulativeIncidenceResult>()?;
    m.add_class::<NNTResult>()?;
    m.add_class::<LandmarkResult>()?;
    m.add_class::<ConditionalSurvivalResult>()?;
    m.add_class::<HazardRatioResult>()?;
    m.add_class::<SurvivalAtTimeResult>()?;
    m.add_class::<LifeTableResult>()?;
    // New utility classes
    m.add_class::<AeqSurvResult>()?;
    m.add_class::<ClusterResult>()?;
    m.add_class::<StrataResult>()?;
    m.add_class::<NearDateResult>()?;
    m.add_class::<TcutResult>()?;
    m.add_class::<RttrightResult>()?;
    // New specialized classes
    m.add_class::<RateTable>()?;
    m.add_class::<RateDimension>()?;
    m.add_class::<DimType>()?;
    m.add_class::<SurvExpResult>()?;
    m.add_class::<StateFigData>()?;
    // New survival analysis classes
    m.add_class::<PseudoResult>()?;
    m.add_class::<AggregateSurvfitResult>()?;
    // New validation classes
    m.add_class::<SurvCheckResult>()?;
    m.add_class::<RoystonResult>()?;
    m.add_class::<YatesResult>()?;
    m.add_class::<YatesPairwiseResult>()?;
    // New regression/core classes
    m.add_class::<RidgePenalty>()?;
    m.add_class::<RidgeResult>()?;
    m.add_class::<NaturalSplineKnot>()?;
    m.add_class::<SplineBasisResult>()?;
    // ANOVA classes
    m.add_class::<AnovaCoxphResult>()?;
    m.add_class::<AnovaRow>()?;
    // Reliability classes
    m.add_class::<ReliabilityResult>()?;
    m.add_class::<ReliabilityScale>()?;
    // Survfit matrix classes
    m.add_class::<SurvfitMatrixResult>()?;

    // Dataset loaders
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

    // New survreg functions
    m.add_function(wrap_pyfunction!(residuals_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(dfbeta_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(residuals_survfit, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg_quantile, &m)?)?;
    m.add_function(wrap_pyfunction!(coxph_detail, &m)?)?;

    // Ratetable utilities
    m.add_function(wrap_pyfunction!(is_ratetable, &m)?)?;
    m.add_function(wrap_pyfunction!(ratetable_date, &m)?)?;
    m.add_function(wrap_pyfunction!(days_to_date, &m)?)?;

    // Pyears summary functions
    m.add_function(wrap_pyfunction!(summary_pyears, &m)?)?;
    m.add_function(wrap_pyfunction!(pyears_by_cell, &m)?)?;
    m.add_function(wrap_pyfunction!(pyears_ci, &m)?)?;

    // US mortality rate tables
    m.add_function(wrap_pyfunction!(survexp_us, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_mn, &m)?)?;
    m.add_function(wrap_pyfunction!(survexp_usr, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_survival, &m)?)?;

    // New classes
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

    // Bayesian functions
    m.add_function(wrap_pyfunction!(bayesian_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_cox_predict_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric, &m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_parametric_predict, &m)?)?;
    m.add_class::<BayesianCoxResult>()?;
    m.add_class::<BayesianParametricResult>()?;

    // Causal inference functions
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

    // Interval censoring functions
    m.add_function(wrap_pyfunction!(interval_censored_regression, &m)?)?;
    m.add_function(wrap_pyfunction!(turnbull_estimator, &m)?)?;
    m.add_function(wrap_pyfunction!(npmle_interval, &m)?)?;
    m.add_class::<IntervalCensoredResult>()?;
    m.add_class::<TurnbullResult>()?;
    m.add_class::<IntervalDistribution>()?;

    // Joint modeling functions
    m.add_function(wrap_pyfunction!(joint_model, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_prediction, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_auc, &m)?)?;
    m.add_function(wrap_pyfunction!(dynamic_brier_score, &m)?)?;
    m.add_function(wrap_pyfunction!(landmarking_analysis, &m)?)?;
    m.add_class::<JointModelResult>()?;
    m.add_class::<DynamicPredictionResult>()?;
    m.add_class::<AssociationStructure>()?;

    // Missing data functions
    m.add_function(wrap_pyfunction!(multiple_imputation_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(analyze_missing_pattern, &m)?)?;
    m.add_function(wrap_pyfunction!(pattern_mixture_model, &m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(tipping_point_analysis, &m)?)?;
    m.add_class::<MultipleImputationResult>()?;
    m.add_class::<PatternMixtureResult>()?;
    m.add_class::<ImputationMethod>()?;
    m.add_class::<SensitivityAnalysisType>()?;

    // Machine learning functions
    m.add_function(wrap_pyfunction!(survival_forest, &m)?)?;
    m.add_function(wrap_pyfunction!(gradient_boost_survival, &m)?)?;
    m.add_class::<SurvivalForest>()?;
    m.add_class::<GradientBoostSurvival>()?;

    // Quality of life functions
    m.add_function(wrap_pyfunction!(qaly_calculation, &m)?)?;
    m.add_function(wrap_pyfunction!(qaly_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(incremental_cost_effectiveness, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_analysis, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_comparison, &m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_sensitivity, &m)?)?;
    m.add_class::<QALYResult>()?;
    m.add_class::<QTWISTResult>()?;

    // Recurrent events functions
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

    // Cure model functions
    m.add_function(wrap_pyfunction!(mixture_cure_model, &m)?)?;
    m.add_function(wrap_pyfunction!(promotion_time_cure_model, &m)?)?;
    m.add_class::<MixtureCureResult>()?;
    m.add_class::<PromotionTimeCureResult>()?;
    m.add_class::<CureDistribution>()?;

    // Elastic net functions
    m.add_function(wrap_pyfunction!(elastic_net_cox, &m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_cv, &m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_path, &m)?)?;
    m.add_class::<ElasticNetCoxResult>()?;
    m.add_class::<ElasticNetCoxPath>()?;

    // Relative survival functions
    m.add_function(wrap_pyfunction!(relative_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(net_survival, &m)?)?;
    m.add_function(wrap_pyfunction!(crude_probability_of_death, &m)?)?;
    m.add_function(wrap_pyfunction!(excess_hazard_regression, &m)?)?;
    m.add_class::<RelativeSurvivalResult>()?;
    m.add_class::<NetSurvivalResult>()?;
    m.add_class::<ExcessHazardModelResult>()?;
    m.add_class::<NetSurvivalMethod>()?;

    // Spatial frailty functions
    m.add_function(wrap_pyfunction!(spatial_frailty_model, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_spatial_smoothed_rates, &m)?)?;
    m.add_function(wrap_pyfunction!(moran_i_test, &m)?)?;
    m.add_class::<SpatialFrailtyResult>()?;
    m.add_class::<SpatialCorrelationStructure>()?;

    Ok(())
}
