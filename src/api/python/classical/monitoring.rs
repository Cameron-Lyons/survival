use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_model_card, m)?)?;
    m.add_function(wrap_pyfunction!(fairness_audit, m)?)?;
    m.add_function(wrap_pyfunction!(detect_drift, m)?)?;
    m.add_function(wrap_pyfunction!(monitor_performance, m)?)?;

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
