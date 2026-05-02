#[path = "bayesian_cox.rs"]
pub(crate) mod bayesian_cox_module;
pub(crate) mod bayesian_extensions;
#[path = "bayesian_parametric.rs"]
pub(crate) mod bayesian_parametric_module;

// Public facade exports
pub use bayesian_cox_module::{BayesianCoxResult, bayesian_cox, bayesian_cox_predict_survival};
pub use bayesian_extensions::{
    BayesianModelAveragingConfig, BayesianModelAveragingResult, DirichletProcessConfig,
    DirichletProcessResult, HorseshoeConfig, HorseshoeResult, SpikeSlabConfig, SpikeSlabResult,
    bayesian_model_averaging_cox, dirichlet_process_survival, horseshoe_cox, spike_slab_cox,
};
pub use bayesian_parametric_module::{
    BayesianParametricResult, bayesian_parametric, bayesian_parametric_predict,
};
