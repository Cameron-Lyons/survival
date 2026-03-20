use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        create_model_card,
        fairness_audit,
        detect_drift,
        monitor_performance,
    );

    register_classes!(
        m,
        ModelCard,
        ModelPerformanceMetrics,
        SubgroupPerformance,
        FairnessAuditResult,
        DriftConfig,
        DriftReport,
        FeatureDriftResult,
        PerformanceDriftResult,
    );

    Ok(())
}
