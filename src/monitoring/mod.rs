pub(crate) mod drift_detection;
pub(crate) mod model_cards;

// Public facade exports
pub use drift_detection::{
    DriftConfig, DriftReport, FeatureDriftResult, PerformanceDriftResult, detect_drift,
    monitor_performance,
};
pub use model_cards::{
    FairnessAuditResult, ModelCard, ModelPerformanceMetrics, SubgroupPerformance,
    create_model_card, fairness_audit,
};
