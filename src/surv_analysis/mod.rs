//! Non-parametric survival curves and tests: Kaplan-Meier /
//! Fleming-Harrington (`survfitkm`), Aalen-Johansen (`survfitaj`), the
//! summaries built on them, the G-rho tests (`survdiff`) and the Cox
//! baseline curves.

pub(crate) mod aggregate_survfit;
pub(crate) mod agsurv;
#[path = "agsurv4.rs"]
pub(crate) mod agsurv4_module;
#[path = "agsurv5.rs"]
pub(crate) mod agsurv5_module;
pub(crate) mod cox_baseline;
pub(crate) mod cox_survfit;
pub(crate) mod illness_death;
pub(crate) mod logrank_components;
pub(crate) mod multi_state;
pub(crate) mod nelson_aalen;
pub(crate) mod norisk;
pub(crate) mod pseudo;
pub(crate) mod pseudo_gee;
pub(crate) mod semi_markov;
pub(crate) mod statefig;
pub(crate) mod survfit_confint;
pub(crate) mod survfit_matrix;
pub(crate) mod survfit_summary;
pub(crate) mod survfitaj;
#[path = "survfitaj_extended.rs"]
pub(crate) mod survfitaj_extended_module;
pub(crate) mod survfitkm;

pub use aggregate_survfit::{
    AggregateFun, AggregateGroups, AggregateSurvfitResult, GroupingFactor, aggregate_survfit,
    aggregate_survfit_py,
};
pub use agsurv::{
    AgsurvCurve, AgsurvData, CoxSurvCurve, CoxSurvType, IndividualInterval, agsurv, coxsurv_fit,
    expand_curve, individual_curve,
};
pub use agsurv4_module::agsurv4;
pub use agsurv5_module::agsurv5;
pub use cox_baseline::{
    compute_baseline_survival_steps, compute_tied_baseline_summaries,
    cox_expected_baseline_by_stratum,
};
pub use cox_survfit::cox_survfit_baseline;
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
pub use norisk::{norisk_flags, norisk_py};
pub use pseudo::{
    ResidualType, SurvfitAJResid, SurvfitResid, pseudo, pseudo_aj, pseudo_aj_py, pseudo_py,
    survfitresid, survfitresid_aj, survfitresid_aj_py, survfitresid_py,
};
pub use pseudo_gee::{GEEConfig, GEEResult, pseudo_gee_regression};
pub use semi_markov::{
    SemiMarkovConfig, SemiMarkovPrediction, SemiMarkovResult, SojournDistribution,
    SojournTimeParams, fit_semi_markov, predict_semi_markov,
};
pub use statefig::{StateFigArrow, StateFigLayout, StateFigResult, statefig, statefig_py};
pub use survfit_confint::{
    ConfLower, ConfType, ConfidenceBands, survfit_confint, survfit_confint_py,
};
pub use survfit_matrix::{
    SurvfitMatrixResult, basehaz, condition_cox_survfit_curves, cox_survfit_from_baseline,
    step_matrix_values_at, step_values_at, survfit_from_cumhaz, survfit_from_hazard,
    survfit_from_matrix, survfit_multistate,
};
pub use survfit_summary::{
    RmeanOption, SurvfitQuantiles, SurvmeanTable, quantile_survfit, quantile_survfit_from,
    quantile_survfit_py, summary_survfit, summary_survfit_py, summary_survfit_times, survfit0,
    survfit0_aj, survfit0_aj_py, survfit0_py, survmean, survmean_py,
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
    HazardType, InfluenceRequest, SurvType, SurvfitCounts, SurvfitInfluence, SurvfitKMData,
    SurvfitKMOptions, SurvfitKMResult, survfitkm, survfitkm_py,
};
