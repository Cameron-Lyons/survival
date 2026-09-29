//! Regression models.
//!
//! The Cox model lives in `coxph` (fit, predictions, survival curves) with
//! its residuals in `coxph_diagnostics` (`residuals.coxph` on top of the
//! `residuals`, `scoring` and `core::coxscho` kernels), `coxph.detail` in
//! `coxph_detail`, `cox.zph` in `cox_zph` and the case-cohort estimators in
//! `cch`.  [`TieMethod`] is the one tie-handling enum of the crate, shared
//! by the fitters and every kernel.
//! The multi-state Cox model (`coxphms`: the stacked data and its fit) is
//! in `coxphms`.
//! Penalised Cox models (R's `coxpenal.fit`: `ridge()`, `pspline()` and
//! `frailty()` terms) live in `coxpenal` and penalised parametric models
//! (`survpenal.fit`) in `survpenal`, both on the penalty machinery of
//! `penalized`.

#[path = "aareg_fit.rs"]
pub(crate) mod aareg_fit_module;
pub(crate) mod agexact;
pub(crate) mod blogit;
#[path = "cause_specific_cox.rs"]
pub(crate) mod cause_specific_cox_module;
pub(crate) mod cch;
pub(crate) mod cox_optimizer;
pub(crate) mod cox_zph;
pub(crate) mod coxpenal;
pub(crate) mod coxph;
pub(crate) mod coxph_detail;
pub(crate) mod coxph_diagnostics;
pub(crate) mod coxph_wtest;
pub(crate) mod coxphms;
pub(crate) mod cure_models;
pub(crate) mod elastic_net;
pub(crate) mod exact_ties;
#[path = "fast_cox/mod.rs"]
pub(crate) mod fast_cox_module;
pub(crate) mod finegray_data;
#[path = "finegray_regression.rs"]
pub(crate) mod finegray_regression_module;
pub(crate) mod functional_survival;
pub(crate) mod high_dimensional;
pub(crate) mod joint_competing;
pub(crate) mod longitudinal_survival;
pub(crate) mod parametric_survival;
pub(crate) mod penalized;
pub(crate) mod recurrent_events;
pub(crate) mod spline_hazard;
pub(crate) mod survpenal;
mod survreg_callbacks;
pub(crate) mod survreg_density;
pub use survreg_callbacks::{SurvregCallbacks, SurvregTransformCallbacks};
pub(crate) mod survreg_distributions;
pub(crate) mod survreg_predict;
pub(crate) mod survregc1;

pub use aareg_fit_module::{
    AaregData, AaregFitResult, AaregOptions, AaregTest, aareg_fit, aareg_fit_py,
};
pub use agexact::{AgexactFit, AgexactOptions, agexact_fit, agexact_py};
pub use blogit::LinkFunctionParams;
pub use cause_specific_cox_module::{
    CauseSpecificCoxConfig, CauseSpecificCoxResult, CensoringType, cause_specific_cox,
    cause_specific_cox_all,
};
pub use cch::{CchFitResult, cch, cch_borgan, cch_borgan_fit, cch_fit};
pub use cox_optimizer::TieMethod;
#[cfg(feature = "python")]
pub use cox_zph::cox_zph_py;
pub use cox_zph::{CoxZph, CoxZphTest, ZphFit, ZphTransform, cox_zph};
#[cfg(feature = "python")]
pub use coxpenal::CallbackPenalty;
pub use coxpenal::{
    COXPENAL_OUTER_MAX, CoxPenalty, CoxPenaltyTerms, CoxpenalData, CoxpenalFit, CoxpenalOptions,
    FrailtyFamily, FrailtyMethod, FrailtyPenalty, ModelTerm, PenaltyHistory, PenaltyTerm,
    PsplineMethod, PsplinePenalty, RidgePenalty, coxpenal_fit,
};
pub use coxph::{
    Basehaz, CoxNewData, CoxPHFit, CoxPrediction, CoxSurvfitCurve, CoxTermsPrediction, CoxphData,
    CoxphOptions, PredictReference, SurvfitOptions, coxph_fit,
};
pub use coxph_detail::{CoxphDetail, coxph_detail, coxph_detail_py};
pub use coxph_diagnostics::{CoxResidualType, Residuals, SchoenfeldResiduals};
pub use coxph_wtest::{CoxphWtest, coxph_wtest_py, wald_tests};
pub use coxphms::coxphms_fit;
pub use cure_models::{
    BoundedCumulativeHazardConfig, BoundedCumulativeHazardResult, CureDistribution,
    CureModelComparisonResult, LinkFunction, MixtureCureConfig, MixtureCureResult,
    NonMixtureCureConfig, NonMixtureCureResult, NonMixtureType, PromotionTimeCureResult,
    bounded_cumulative_hazard_model, compare_cure_models, mixture_cure_model,
    non_mixture_cure_model, predict_bounded_cumulative_hazard, predict_non_mixture_survival,
    promotion_time_cure_model,
};
pub use elastic_net::{
    ElasticNetCVConfig, ElasticNetConfig, ElasticNetCoxPath, ElasticNetCoxResult,
    ElasticNetPathConfig, PenaltyType, elastic_net_cox, elastic_net_cox_cv, elastic_net_cox_path,
};
pub use fast_cox_module::{
    FastCoxCVConfig, FastCoxConfig, FastCoxPath, FastCoxPathConfig, FastCoxResult,
    FastCoxSolverConfig, ScreeningRule, fast_cox, fast_cox_cv, fast_cox_path,
};
pub use finegray_data::{FineGrayOutput, finegray, finegray_py};
pub use finegray_regression_module::{
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
pub use parametric_survival::{
    SurvregControl, SurvregData, SurvregFit, survreg_fit, survreg_fit_py,
};
pub use recurrent_events::{
    AndersonGillResult, NegativeBinomialFrailtyConfig, NegativeBinomialFrailtyResult, PWPConfig,
    PWPResult, PWPTimescale, WLWConfig, WLWResult, anderson_gill_model, negative_binomial_frailty,
    pwp_model, wlw_model,
};
pub use spline_hazard::{
    FlexibleParametricResult, HazardSplineResult, RestrictedCubicSplineResult, SplineConfig,
    flexible_parametric_model, predict_hazard_spline, restricted_cubic_spline,
};
#[cfg(feature = "python")]
pub use survpenal::survpenal_fit_from_state;
pub use survpenal::{SurvpenalData, SurvpenalFit, SurvpenalOptions, survpenal_fit};
pub use survreg_distributions::{
    SurvregDistribution, SurvregFamily, SurvregTransform, dsurvreg, psurvreg, qsurvreg, rsurvreg,
    survreg_dtest,
};
pub use survreg_predict::{SurvregNewdata, SurvregPredictType, SurvregPrediction, predict_survreg};
