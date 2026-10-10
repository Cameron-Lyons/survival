//! Non-parametric survival curves and tests: Kaplan-Meier /
//! Fleming-Harrington (`survfitkm`), Aalen-Johansen (`survfitaj`), the
//! summaries built on them, the G-rho tests (`survdiff`) and the Cox
//! baseline curves.

mod aggregate_arithmetic;
pub(crate) mod aggregate_survfit;
pub(crate) mod agsurv;
mod coxsurv;
pub(crate) mod illness_death;
pub(crate) mod logrank_components;
pub(crate) mod multi_state;
pub(crate) mod nelson_aalen;
pub(crate) mod pseudo;
pub(crate) mod pseudo_gee;
pub(crate) mod semi_markov;
pub(crate) mod statefig;
pub(crate) mod survfit_aj_summary;
pub(crate) mod survfit_confint;
pub(crate) mod survfit_coxphms;
pub(crate) mod survfit_matrix;
pub(crate) mod survfit_summary;
pub(crate) mod survfitaj;
#[path = "survfitaj_extended.rs"]
pub(crate) mod survfitaj_extended_module;
pub(crate) mod survfitkm;

pub use aggregate_survfit::{
    AggregateFun, AggregateGroups, AggregateSurvfitResult, GroupingFactor, aggregate_survfit,
    aggregate_survfit_py, aggregate_survfit_with,
};
pub use agsurv::{
    AgsurvCurve, AgsurvData, CoxSurvCurve, CoxSurvType, IndividualInterval, agsurv,
    cox_survfit_baseline, expand_curve, individual_curve, step_values_at,
};
pub use illness_death::{
    IllnessDeathConfig, IllnessDeathPrediction, IllnessDeathResult, IllnessDeathType,
    TransitionHazard, fit_illness_death, predict_illness_death,
};
pub use logrank_components::{
    SurvDiffResult, SurvdiffData, survdiff, survdiff_one_sample, survdiff_one_sample_py,
    survdiff_py,
};
pub use multi_state::{
    MarkovMSMResult, MultiStateConfig, MultiStateResult, TransitionIntensityResult,
    estimate_transition_intensities, fit_markov_msm, fit_multi_state_model,
};
pub use nelson_aalen::{NelsonAalenResult, nelson_aalen, nelson_aalen_py};
pub use pseudo::{
    PseudoResidualType, SurvfitAJResid, SurvfitResid, pseudo, pseudo_aj, pseudo_aj_py, pseudo_py,
    survfitresid, survfitresid_aj, survfitresid_aj_py, survfitresid_py,
};
pub use pseudo_gee::{GEEConfig, GEEResult, pseudo_gee_regression};
pub use semi_markov::{
    SemiMarkovConfig, SemiMarkovPrediction, SemiMarkovResult, SojournDistribution,
    SojournTimeParams, fit_semi_markov, predict_semi_markov,
};
pub use statefig::{StateFigArrow, StateFigLayout, StateFigResult, statefig, statefig_py};
pub use survfit_aj_summary::{
    AJMeanTable, summary_survfit_aj, summary_survfit_aj_with_counts, survmean_aj,
};
pub use survfit_confint::{
    ConfLower, ConfType, ConfidenceBands, survfit_confint, survfit_confint_py,
};
#[cfg(feature = "python")]
pub use survfit_coxphms::coxphms_curves;
pub use survfit_matrix::{
    SurvfitMatrixMethod, SurvfitMatrixTransition, survfit_matrix, survfit_matrix_py,
};
pub use survfit_summary::{
    RmeanOption, SurvfitQuantiles, SurvmeanTable, quantile_survfit, quantile_survfit_from,
    quantile_survfit_py, summary_survfit, summary_survfit_py, summary_survfit_times,
    summary_survfit_times_with_counts, survfit0, survfit0_aj, survfit0_aj_py, survfit0_py,
    survmean, survmean_py,
};
pub use survfitaj::{
    SurvfitAJCounts, SurvfitAJData, SurvfitAJInfluence, SurvfitAJOptions, SurvfitAJResult,
    survfitaj, survfitaj_py,
};
pub use survfitaj_extended_module::{
    AalenJohansenExtendedConfig, AalenJohansenExtendedResult, TransitionMatrix, TransitionType,
    VarianceEstimator, survfitaj_extended,
};
pub use survfitkm::{
    HazardType, InfluenceRequest, StackedCurves, SurvType, SurvfitCounts, SurvfitInfluence,
    SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm, survfitkm_py,
};

pub use coxsurv::{
    CoxSurvBaselineDetails, CoxSurvData, CoxSurvNewData, CoxSurvRawResult, coxsurv_fit,
    coxsurv_fit_py,
};
