pub(crate) mod aareg;
pub(crate) mod agexact;
pub(crate) mod agfit5;
pub(crate) mod blogit;
pub(crate) mod cause_specific_cox;
pub(crate) mod cch;
pub(crate) mod clogit;
pub(crate) mod cox_optimizer;
pub(crate) mod coxph;
pub(crate) mod coxph_detail;
pub(crate) mod cure_models;
pub(crate) mod elastic_net;
pub(crate) mod fast_cox;
pub(crate) mod finegray_data;
pub(crate) mod finegray_regression;
pub(crate) mod functional_survival;
pub(crate) mod high_dimensional;
pub(crate) mod joint_competing;
pub(crate) mod longitudinal_survival;
pub(crate) mod parametric_survival;
pub(crate) mod recurrent_events;
pub(crate) mod ridge;
pub(crate) mod spline_hazard;
pub(crate) mod survreg_predict;
pub(crate) mod survregc1;

// Public facade exports
pub use aareg::{AaregOptions, aareg};
pub use agexact::agexact;
pub use agfit5::perform_cox_regression_frailty;
pub use blogit::LinkFunctionParams;
pub use cause_specific_cox::{
    CauseSpecificCoxConfig, CauseSpecificCoxResult, CensoringType, cause_specific_cox,
    cause_specific_cox_all,
};
pub use cch::{CchMethod, CohortData};
pub use clogit::{ClogitDataSet, ConditionalLogisticRegression};
pub use coxph::{CoxPHModel, Subject};
pub use coxph_detail::{CoxphDetail, CoxphDetailRow, coxph_detail};
pub use cure_models::{
    BoundedCumulativeHazardConfig, BoundedCumulativeHazardResult, CureDistribution,
    CureModelComparisonResult, MixtureCureResult, NonMixtureCureConfig, NonMixtureCureResult,
    NonMixtureType, PromotionTimeCureResult, bounded_cumulative_hazard_model, compare_cure_models,
    mixture_cure_model, non_mixture_cure_model, predict_bounded_cumulative_hazard,
    predict_non_mixture_survival, promotion_time_cure_model,
};
pub use elastic_net::{
    ElasticNetCVConfig, ElasticNetConfig, ElasticNetCoxPath, ElasticNetCoxResult,
    ElasticNetPathConfig, PenaltyType, elastic_net_cox, elastic_net_cox_cv, elastic_net_cox_path,
};
pub use fast_cox::{
    FastCoxCVConfig, FastCoxConfig, FastCoxPath, FastCoxPathConfig, FastCoxResult,
    FastCoxSolverConfig, ScreeningRule, fast_cox, fast_cox_cv, fast_cox_path,
};
pub use finegray_data::{FineGrayOutput, finegray};
pub use finegray_regression::{
    CompetingRisksCIF, FineGrayResult, competing_risks_cif, finegray_regression,
};
pub use functional_survival::{
    BasisType, FunctionalPCAResult, FunctionalSurvivalConfig, FunctionalSurvivalResult,
    fpca_survival, functional_cox,
};
pub use high_dimensional::{
    GroupLassoConfig, GroupLassoResult, SISConfig, SISResult, SparseBoostingConfig,
    SparseBoostingResult, StabilitySelectionConfig, StabilitySelectionResult, group_lasso_cox,
    sis_cox, sparse_boosting_cox, stability_selection_cox,
};
pub use joint_competing::{
    CauseResult, CorrelationType, JointCompetingRisksConfig, JointCompetingRisksResult,
    joint_competing_risks,
};
pub use longitudinal_survival::{
    JointLongSurvResult, JointModelConfig, LandmarkAnalysisResult, LongDynamicPredResult,
    TimeVaryingCoxResult, joint_longitudinal_model, landmark_cox_analysis,
    longitudinal_dynamic_pred, time_varying_cox,
};
pub use parametric_survival::{DistributionType, SurvivalFit, SurvregConfig, survreg};
pub use recurrent_events::{
    AndersonGillResult, NegativeBinomialFrailtyConfig, NegativeBinomialFrailtyResult, PWPConfig,
    PWPResult, PWPTimescale, WLWConfig, WLWResult, anderson_gill_model, negative_binomial_frailty,
    pwp_model, wlw_model,
};
pub use ridge::{RidgePenalty, RidgeResult, ridge_cv, ridge_fit};
pub use spline_hazard::{
    FlexibleParametricResult, HazardSplineResult, RestrictedCubicSplineResult, SplineConfig,
    flexible_parametric_model, predict_hazard_spline, restricted_cubic_spline,
};
pub use survreg_predict::{
    SurvregPrediction, SurvregQuantilePrediction, predict_survreg, predict_survreg_quantile,
};
