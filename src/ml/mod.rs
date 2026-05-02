pub(crate) mod active_learning;
pub(crate) mod adversarial_robustness;
pub(crate) mod attention_cox;
pub(crate) mod config_validation;
pub(crate) mod contrastive_surv;
pub(crate) mod cox_time;
pub(crate) mod deep_pamm;
pub(crate) mod deep_surv;
pub(crate) mod deephit;
pub(crate) mod differential_privacy;
pub(crate) mod distributionally_robust;
pub(crate) mod dynamic_deephit;
pub(crate) mod dysurv;
pub(crate) mod ensemble_surv;
pub(crate) mod federated_learning;
pub(crate) mod galee;
pub(crate) mod gpu_acceleration;
pub(crate) mod gradient_boost;
pub(crate) mod graph_surv;
pub(crate) mod knowledge_distillation;
pub(crate) mod multimodal_surv;
pub(crate) mod neural_mtlr;
pub(crate) mod neural_ode_surv;
pub(crate) mod recurrent_surv;
pub(crate) mod state_space_surv;
pub(crate) mod streaming_survival;
pub(crate) mod survival_forest;
pub(crate) mod survival_transformer;
pub(crate) mod survtrace;
pub(crate) mod temporal_fusion;
pub(crate) mod tracer;
pub(crate) mod transfer_learning;
pub(crate) mod utils;

// Public facade exports
#[cfg(feature = "ml")]
pub use active_learning::{
    ActiveLearningConfig, ActiveLearningResult, AdaptiveDesignResult, LogrankPowerResult,
    LogrankSampleSizeResult, QBCResult, active_learning_selection, group_sequential_analysis,
    power_logrank, query_by_committee, sample_size_logrank,
};
#[cfg(feature = "ml")]
pub use adversarial_robustness::{
    AdversarialAttackConfig, AdversarialAttackResult, AdversarialDefenseConfig, AdversarialExample,
    AttackType, DefenseType, RobustSurvivalModel, RobustnessEvaluation,
    adversarial_training_survival, evaluate_robustness, generate_adversarial_examples,
};
#[cfg(feature = "ml")]
pub use attention_cox::{AttentionCoxConfig, AttentionCoxModel, fit_attention_cox};
#[cfg(feature = "ml")]
pub use contrastive_surv::{
    ContrastiveSurv, ContrastiveSurvConfig, ContrastiveSurvResult, SurvivalLossType,
    contrastive_surv,
};
#[cfg(feature = "ml")]
pub use cox_time::{CoxTimeConfig, CoxTimeModel, fit_cox_time};
#[cfg(feature = "ml")]
pub use deep_pamm::{DeepPAMMConfig, DeepPAMMModel, fit_deep_pamm};
#[cfg(feature = "ml")]
pub use deep_surv::{Activation, DeepSurv, DeepSurvConfig, deep_surv};
#[cfg(feature = "ml")]
pub use deephit::{DeepHit, DeepHitConfig, deephit};
#[cfg(feature = "ml")]
pub use differential_privacy::{
    DPConfig, DPCoxResult, DPHistogramResult, DPSurvivalResult, LocalDPResult, dp_cox_regression,
    dp_histogram, dp_kaplan_meier, local_dp_mean,
};
#[cfg(feature = "ml")]
pub use distributionally_robust::{
    DROSurvivalConfig, DROSurvivalResult, RobustnessAnalysis, UncertaintySet, dro_survival,
    robustness_analysis,
};
#[cfg(feature = "ml")]
pub use dynamic_deephit::{DynamicDeepHit, DynamicDeepHitConfig, TemporalType, dynamic_deephit};
#[cfg(feature = "ml")]
pub use dysurv::{
    DySurvConfig, DySurvModel, DynamicRiskResult, dynamic_risk_prediction, fit_dysurv,
};
#[cfg(feature = "ml")]
pub use ensemble_surv::{
    BlendingResult, ComponentwiseBoostingConfig, ComponentwiseBoostingResult, StackingConfig,
    StackingResult, SuperLearnerConfig, SuperLearnerResult, blending_survival,
    componentwise_boosting, stacking_survival, super_learner_survival,
};
#[cfg(feature = "ml")]
pub use federated_learning::{
    FederatedConfig, FederatedSurvivalResult, PrivacyAccountant, SecureAggregationResult,
    federated_cox, secure_aggregate,
};
#[cfg(feature = "ml")]
pub use galee::{GALEE, GALEEConfig, GALEEResult, UnimodalConstraint, galee};
#[cfg(feature = "ml")]
pub use gpu_acceleration::{
    BatchPredictionResult, ComputeBackend, DeviceInfo, GPUConfig, ParallelCoxResult,
    batch_predict_survival, benchmark_compute_backend, get_available_devices, is_gpu_available,
    parallel_cox_regression, parallel_matrix_operations,
};
#[cfg(feature = "ml")]
pub use gradient_boost::{
    GBSurvLoss, GradientBoostSurvival, GradientBoostSurvivalConfig, gradient_boost_survival,
};
#[cfg(feature = "ml")]
pub use graph_surv::{GraphSurvConfig, GraphSurvModel, fit_graph_surv};
#[cfg(feature = "ml")]
pub use knowledge_distillation::{
    DistillationConfig, DistillationResult, DistilledSurvivalModel, PruningResult,
    distill_survival_model, prune_survival_model,
};
#[cfg(feature = "ml")]
pub use multimodal_surv::{
    FusionStrategy, MultimodalSurvConfig, MultimodalSurvModel, fit_multimodal_surv,
};
#[cfg(feature = "ml")]
pub use neural_mtlr::{NeuralMTLRConfig, NeuralMTLRModel, fit_neural_mtlr};
#[cfg(feature = "ml")]
pub use neural_ode_surv::{NeuralODESurvConfig, NeuralODESurvModel, fit_neural_ode_surv};
#[cfg(feature = "ml")]
pub use recurrent_surv::{
    LongitudinalSurvConfig, LongitudinalSurvModel, RecurrentSurvConfig, RecurrentSurvModel,
    fit_longitudinal_surv, fit_recurrent_surv,
};
#[cfg(feature = "ml")]
pub use state_space_surv::{MambaSurvConfig, MambaSurvModel, fit_mamba_surv};
#[cfg(feature = "ml")]
pub use streaming_survival::{
    ConceptDriftDetector, StreamingCoxConfig, StreamingCoxModel, StreamingKaplanMeier,
};
#[cfg(feature = "ml")]
pub use survival_forest::{
    SplitRule, SurvivalForest, SurvivalForestConfig, SurvivalForestInput, survival_forest,
};
#[cfg(feature = "ml")]
pub use survival_transformer::{
    SurvivalTransformerConfig, SurvivalTransformerModel, fit_survival_transformer,
};
#[cfg(feature = "ml")]
pub use survtrace::{SurvTrace, SurvTraceActivation, SurvTraceConfig, survtrace};
#[cfg(feature = "ml")]
pub use temporal_fusion::{TFTConfig, TemporalFusionTransformer, fit_temporal_fusion_transformer};
#[cfg(feature = "ml")]
pub use tracer::{Tracer, TracerConfig, tracer};
#[cfg(feature = "ml")]
pub use transfer_learning::{
    DomainAdaptationResult, PretrainedSurvivalModel, TransferLearningConfig, TransferStrategy,
    TransferredModel, compute_domain_distance, pretrain_survival_model, transfer_survival_model,
};
