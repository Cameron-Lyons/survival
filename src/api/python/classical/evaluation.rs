use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        schoenfeld_residuals,
        cox_score_residuals,
        bootstrap_cox_ci,
        bootstrap_survreg_ci,
        cv_cox_concordance,
        cv_survreg_loglik,
        lrt_test,
        wald_test_py,
        score_test_py,
        ph_test,
        sample_size_survival,
        sample_size_survival_freedman,
        power_survival,
        expected_events,
        calibration,
        predict_cox,
        risk_stratification,
        td_auc,
        d_calibration,
        one_calibration,
        calibration_plot,
        brier_calibration,
        multi_time_calibration,
        smoothed_calibration,
        landmark_analysis,
        landmark_analysis_batch,
    );

    register_classes!(
        m,
        BootstrapResult,
        CVResult,
        TestResult,
        ProportionalityTest,
        SampleSizeResult,
        AccrualResult,
        CalibrationResult,
        PredictionResult,
        RiskStratificationResult,
        TdAUCResult,
        DCalibrationResult,
        OneCalibrationResult,
        CalibrationPlotData,
        BrierCalibrationResult,
        MultiTimeCalibrationResult,
        SmoothedCalibrationCurve,
        LandmarkResult,
    );

    Ok(())
}
