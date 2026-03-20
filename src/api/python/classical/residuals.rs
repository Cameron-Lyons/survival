use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        residuals_survreg,
        dfbeta_survreg,
        residuals_survfit,
        predict_survreg,
        predict_survreg_quantile,
        coxph_detail,
    );

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
