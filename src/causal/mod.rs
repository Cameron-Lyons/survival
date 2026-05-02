pub(crate) mod causal_forest;
pub(crate) mod counterfactual_survival;
pub(crate) mod dependent_censoring;
pub(crate) mod double_ml;
#[path = "g_computation.rs"]
pub(crate) mod g_computation_module;
pub(crate) mod instrumental_variable;
pub(crate) mod ipcw;
pub(crate) mod msm;
pub(crate) mod target_trial;
pub(crate) mod tmle;

// Public facade exports
pub use causal_forest::{
    CausalForestConfig, CausalForestResult, CausalForestSurvival, causal_forest_survival,
};
pub use counterfactual_survival::{
    CounterfactualSurvivalConfig, CounterfactualSurvivalResult, TVSurvCausConfig, TVSurvCausResult,
    estimate_counterfactual_survival, estimate_tv_survcaus,
};
pub use dependent_censoring::{
    CopulaCensoringConfig, CopulaCensoringResult, CopulaType, MNARSurvivalConfig,
    MNARSurvivalResult, SensitivityBoundsConfig, SensitivityBoundsResult, copula_censoring_model,
    mnar_sensitivity_survival, sensitivity_bounds_survival,
};
pub use double_ml::{
    CATEResult, DoubleMLConfig, DoubleMLResult, double_ml_cate, double_ml_survival,
};
pub use g_computation_module::{GComputationResult, g_computation, g_computation_survival_curves};
pub use instrumental_variable::{
    GEstimationConfig, GEstimationResult, IVCoxConfig, IVCoxResult, MediationSurvivalConfig,
    MediationSurvivalResult, RDSurvivalConfig, RDSurvivalResult, g_estimation_aft, iv_cox,
    mediation_survival, rd_survival,
};
pub use ipcw::{
    IPCWConfig, IPCWInput, IPCWResult, compute_ipcw_weights, ipcw_kaplan_meier,
    ipcw_treatment_effect,
};
pub use msm::{MSMResult, compute_longitudinal_iptw, marginal_structural_model};
pub use target_trial::{TargetTrialResult, sequential_trial_emulation, target_trial_emulation};
pub use tmle::{TMLEConfig, TMLEResult, TMLESurvivalResult, tmle_ate, tmle_survival};
