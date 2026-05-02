pub(crate) mod aggregate_survfit;
pub(crate) mod agsurv4;
pub(crate) mod agsurv5;
pub(crate) mod cox_baseline;
pub(crate) mod illness_death;
pub(crate) mod logrank_components;
pub(crate) mod multi_state;
pub(crate) mod nelson_aalen;
pub(crate) mod norisk;
pub(crate) mod pseudo;
pub(crate) mod semi_markov;
pub(crate) mod statefig;
pub(crate) mod survfit_matrix;
pub(crate) mod survfitaj;
pub(crate) mod survfitaj_extended;
pub(crate) mod survfitkm;

// Public facade exports
pub use aggregate_survfit::{
    AggregateSurvfitResult, aggregate_survfit, aggregate_survfit_by_group,
};
pub use agsurv4::agsurv4;
pub use agsurv5::agsurv5;
pub use cox_baseline::{compute_baseline_survival_steps, compute_tied_baseline_summaries};
pub use illness_death::{
    IllnessDeathConfig, IllnessDeathPrediction, IllnessDeathResult, IllnessDeathType,
    TransitionHazard, fit_illness_death, predict_illness_death,
};
pub use logrank_components::{SurvDiffResult, compute_logrank_components, survdiff2};
pub use multi_state::{
    MarkovMSMResult, MultiStateConfig, MultiStateResult, TransitionIntensityResult,
    estimate_transition_intensities, fit_markov_msm, fit_multi_state_model,
};
pub use nelson_aalen::{
    NelsonAalenResult, StratifiedKMResult, nelson_aalen, nelson_aalen_estimator,
    stratified_kaplan_meier,
};
pub use norisk::norisk;
pub use pseudo::{GEEConfig, GEEResult, PseudoResult, pseudo, pseudo_fast, pseudo_gee_regression};
pub use semi_markov::{
    SemiMarkovConfig, SemiMarkovPrediction, SemiMarkovResult, SojournDistribution,
    SojournTimeParams, fit_semi_markov, predict_semi_markov,
};
pub use statefig::{
    StateFigData, statefig, statefig_matplotlib_code, statefig_transition_matrix, statefig_validate,
};
pub use survfit_matrix::{
    SurvfitMatrixResult, basehaz, survfit_from_cumhaz, survfit_from_hazard, survfit_from_matrix,
    survfit_multistate,
};
pub use survfitaj::{SurvFitAJ, survfitaj};
pub use survfitaj_extended::{
    AalenJohansenExtendedConfig, AalenJohansenExtendedResult, TransitionMatrix, TransitionType,
    VarianceEstimator, survfitaj_extended,
};
pub use survfitkm::{
    KaplanMeierConfig, SurvFitKMOutput, SurvfitKMOptions, compute_survfitkm, survfitkm,
    survfitkm_with_options,
};
