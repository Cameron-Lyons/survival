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
    m.add_function(wrap_pyfunction!(decision_curve_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(clinical_utility_at_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(compare_decision_curves, m)?)?;
    m.add_function(wrap_pyfunction!(joint_longitudinal_model, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_cox_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(longitudinal_dynamic_pred, m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_cox, m)?)?;
    m.add_function(wrap_pyfunction!(km_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(forest_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_survival_report, m)?)?;
    m.add_function(wrap_pyfunction!(roc_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(warranty_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(renewal_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_growth, m)?)?;
    m.add_function(wrap_pyfunction!(qaly_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(qaly_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(incremental_cost_effectiveness, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_sensitivity, m)?)?;

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
        DecisionCurveResult,
        ClinicalUtilityResult,
        ModelComparisonResult,
        JointModelConfig,
        JointLongSurvResult,
        LandmarkAnalysisResult,
        LongDynamicPredResult,
        TimeVaryingCoxResult,
        KaplanMeierPlotData,
        ForestPlotData,
        CalibrationCurveData,
        SurvivalReport,
        ROCPlotData,
        WarrantyConfig,
        WarrantyResult,
        RenewalResult,
        ReliabilityGrowthResult,
        QALYResult,
        QTWISTResult,
    );

    Ok(())
}
