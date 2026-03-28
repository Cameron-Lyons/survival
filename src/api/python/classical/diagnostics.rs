use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        dfbeta_cox,
        leverage_cox,
        smooth_schoenfeld,
        outlier_detection_cox,
        model_influence_cox,
        goodness_of_fit_cox,
    );

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
