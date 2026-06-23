use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dfbeta_cox, m)?)?;
    m.add_function(wrap_pyfunction!(clustered_crossprod, m)?)?;
    m.add_function(wrap_pyfunction!(clustered_sandwich_variance, m)?)?;
    m.add_function(wrap_pyfunction!(cox_dfbeta_from_score_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(cox_event_indices, m)?)?;
    m.add_function(wrap_pyfunction!(cox_interval_cumulative_hazard_se, m)?)?;
    m.add_function(wrap_pyfunction!(cox_zph_group_variance, m)?)?;
    m.add_function(wrap_pyfunction!(cox_zph_term_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(leverage_cox, m)?)?;
    m.add_function(wrap_pyfunction!(prediction_se_from_variance, m)?)?;
    m.add_function(wrap_pyfunction!(scale_schoenfeld_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(term_prediction_se_from_variance, m)?)?;
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
