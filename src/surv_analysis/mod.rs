#[path = "aggregate_survfit.rs"]
pub(crate) mod aggregate_survfit_module;
#[path = "agsurv4.rs"]
pub(crate) mod agsurv4_module;
#[path = "agsurv5.rs"]
pub(crate) mod agsurv5_module;
pub(crate) mod cox_baseline;
pub(crate) mod illness_death;
pub(crate) mod logrank_components;
pub(crate) mod multi_state;
#[path = "nelson_aalen.rs"]
pub(crate) mod nelson_aalen_module;
#[path = "norisk.rs"]
pub(crate) mod norisk_module;
#[path = "pseudo.rs"]
pub(crate) mod pseudo_module;
pub(crate) mod semi_markov;
#[path = "statefig.rs"]
pub(crate) mod statefig_module;
pub(crate) mod survfit_matrix;
#[path = "survfitaj_extended.rs"]
pub(crate) mod survfitaj_extended_module;
#[path = "survfitaj.rs"]
pub(crate) mod survfitaj_module;
#[path = "survfitkm.rs"]
pub(crate) mod survfitkm_module;

// Public facade exports
pub use aggregate_survfit_module::{
    AggregateSurvfitResult, aggregate_survfit, aggregate_survfit_by_group,
};
pub use agsurv4_module::agsurv4;
pub use agsurv5_module::agsurv5;
pub use cox_baseline::{
    compute_baseline_survival_steps, compute_tied_baseline_summaries,
    cox_expected_baseline_by_stratum,
};
pub use illness_death::{
    IllnessDeathConfig, IllnessDeathPrediction, IllnessDeathResult, IllnessDeathType,
    TransitionHazard, fit_illness_death, predict_illness_death,
};
pub use logrank_components::{
    SurvDiffResult, compute_counting_logrank_components, compute_logrank_components,
    stratified_counting_logrank_components, stratified_logrank_components, survdiff2,
};
pub use multi_state::{
    MarkovMSMResult, MultiStateConfig, MultiStateResult, TransitionIntensityResult,
    estimate_transition_intensities, fit_markov_msm, fit_multi_state_model,
};
pub use nelson_aalen_module::{
    NelsonAalenResult, StratifiedKMResult, nelson_aalen, nelson_aalen_estimator,
    stratified_kaplan_meier,
};
pub use norisk_module::norisk;
pub use pseudo_module::{
    GEEConfig, GEEResult, PseudoResult, pseudo, pseudo_fast, pseudo_gee_regression,
};
pub use semi_markov::{
    SemiMarkovConfig, SemiMarkovPrediction, SemiMarkovResult, SojournDistribution,
    SojournTimeParams, fit_semi_markov, predict_semi_markov,
};
pub use statefig_module::{
    StateFigData, statefig, statefig_matplotlib_code, statefig_transition_matrix, statefig_validate,
};
pub use survfit_matrix::{
    SurvfitMatrixResult, basehaz, condition_cox_survfit_curves, cox_survfit_from_baseline,
    step_values_at, survfit_from_cumhaz, survfit_from_hazard, survfit_from_matrix,
    survfit_multistate,
};
pub use survfitaj_extended_module::{
    AalenJohansenExtendedConfig, AalenJohansenExtendedResult, TransitionMatrix, TransitionType,
    VarianceEstimator, survfitaj_extended,
};
pub use survfitaj_module::{SurvFitAJ, survfitaj};
pub use survfitkm_module::{
    CountingSurvfitTables, KaplanMeierConfig, SurvFitKMInfluenceOutput, SurvFitKMOutput,
    SurvfitCurveResult, SurvfitKMOptions, compute_robust_survfitkm_with_timefix, compute_survfitkm,
    counting_survfit_tables, robust_counting_survfit_variance, robust_survfitkm,
    survfit_curve_from_tables, survfitkm, survfitkm_counting_influence, survfitkm_influence,
    survfitkm_with_options,
};
