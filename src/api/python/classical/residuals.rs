use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(residuals_survreg, m)?)?;
    m.add_function(wrap_pyfunction!(dfbeta_survreg, m)?)?;
    m.add_function(wrap_pyfunction!(residuals_survfit, m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg, m)?)?;
    m.add_function(wrap_pyfunction!(predict_survreg_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(coxph_detail, m)?)?;

    register_classes!(
        m,
        SurvregResiduals,
        SurvfitResiduals,
        SurvregPrediction,
        SurvregQuantilePrediction,
        CoxphDetail,
        CoxphDetailRow,
    );

    Ok(())
}
