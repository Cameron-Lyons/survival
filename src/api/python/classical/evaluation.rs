use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(schoenfeld_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(cox_score_residuals, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_cox_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_survreg_ci, m)?)?;
    m.add_function(wrap_pyfunction!(cv_cox_concordance, m)?)?;
    m.add_function(wrap_pyfunction!(cv_survreg_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(lrt_test, m)?)?;
    m.add_function(wrap_pyfunction!(wald_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(ph_test, m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival, m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival_freedman, m)?)?;
    m.add_function(wrap_pyfunction!(power_survival, m)?)?;
    m.add_function(wrap_pyfunction!(expected_events, m)?)?;
    m.add_function(wrap_pyfunction!(calibration, m)?)?;
    m.add_function(wrap_pyfunction!(predict_cox, m)?)?;
    m.add_function(wrap_pyfunction!(risk_stratification, m)?)?;
    m.add_function(wrap_pyfunction!(td_auc, m)?)?;
    m.add_function(wrap_pyfunction!(d_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(one_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot, m)?)?;
    m.add_function(wrap_pyfunction!(brier_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(multi_time_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(smoothed_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis_batch, m)?)?;

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
