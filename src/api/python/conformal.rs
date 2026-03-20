use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(conformal_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_predict, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival_from_predictions, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_coverage_test, m)?)?;
    m.add_function(wrap_pyfunction!(doubly_robust_conformal_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(doubly_robust_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_predict, m)?)?;
    m.add_function(wrap_pyfunction!(two_sided_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(conformalized_survival_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(cqr_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_calibration_plot, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_width_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_coverage_cv, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival_parallel, m)?)?;
    m.add_class::<ConformalCalibrationResult>()?;
    m.add_class::<ConformalPredictionResult>()?;
    m.add_class::<ConformalDiagnostics>()?;
    m.add_class::<DoublyRobustConformalResult>()?;
    m.add_class::<TwoSidedCalibrationResult>()?;
    m.add_class::<TwoSidedConformalResult>()?;
    m.add_class::<ConformalSurvivalDistribution>()?;
    m.add_class::<BootstrapConformalResult>()?;
    m.add_class::<CQRConformalResult>()?;
    m.add_class::<ConformalCalibrationPlot>()?;
    m.add_class::<ConformalWidthAnalysis>()?;
    m.add_class::<CoverageSelectionResult>()?;

    m.add_function(wrap_pyfunction!(covariate_shift_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(cvplus_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(cvplus_conformal_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(mondrian_conformal_predict, m)?)?;
    m.add_class::<CovariateShiftConformalResult>()?;
    m.add_class::<WeightDiagnostics>()?;
    m.add_class::<CVPlusConformalResult>()?;
    m.add_class::<CVPlusCalibrationResult>()?;
    m.add_class::<MondrianConformalResult>()?;
    m.add_class::<MondrianCalibrationResult>()?;
    m.add_class::<MondrianDiagnostics>()?;

    Ok(())
}
