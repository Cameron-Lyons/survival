pub(crate) mod bayesian_cox;
pub(crate) mod bayesian_extensions;
pub(crate) mod bayesian_parametric;

// Public facade exports
pub use bayesian_cox::{BayesianCoxResult, bayesian_cox, bayesian_cox_predict_survival};
pub use bayesian_extensions::{
    BayesianModelAveragingConfig, BayesianModelAveragingResult, DirichletProcessConfig,
    DirichletProcessResult, HorseshoeConfig, HorseshoeResult, SpikeSlabConfig, SpikeSlabResult,
    bayesian_model_averaging_cox, dirichlet_process_survival, horseshoe_cox, spike_slab_cox,
};
pub use bayesian_parametric::{
    BayesianParametricResult, bayesian_parametric, bayesian_parametric_predict,
};
