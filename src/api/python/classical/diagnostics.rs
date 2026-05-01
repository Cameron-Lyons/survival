use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dfbeta_cox, m)?)?;
    m.add_function(wrap_pyfunction!(leverage_cox, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_schoenfeld, m)?)?;
    m.add_function(wrap_pyfunction!(outlier_detection_cox, m)?)?;
    m.add_function(wrap_pyfunction!(model_influence_cox, m)?)?;
    m.add_function(wrap_pyfunction!(goodness_of_fit_cox, m)?)?;

    register_classes!(
        m,
        DfbetaResult,
        LeverageResult,
        SchoenfeldSmoothResult,
        OutlierDetectionResult,
        ModelInfluenceResult,
        GofTestResult,
    );

    Ok(())
}
