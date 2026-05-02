use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perform_cox_regression_frailty, m)?)?;
    m.add_function(wrap_pyfunction!(perform_pyears_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance1_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance3_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_concordance_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_score_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_agscore3_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(perform_pystep_simple_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(aareg, m)?)?;
    m.add_function(wrap_pyfunction!(collapse, m)?)?;
    m.add_function(wrap_pyfunction!(cox_callback, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount1, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount2, m)?)?;
    m.add_function(wrap_pyfunction!(norisk, m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson, m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_exact, m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_anscombe, m)?)?;
    m.add_function(wrap_pyfunction!(crate::concordance::basic::concordance, m)?)?;
    m.add_function(wrap_pyfunction!(agexact, m)?)?;
    m.add_function(wrap_pyfunction!(compute_baseline_survival_steps, m)?)?;
    m.add_function(wrap_pyfunction!(compute_tied_baseline_summaries, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::residuals::agmart_module::agmart_typed,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::residuals::coxmart_module::coxmart_typed,
        m
    )?)?;

    register_classes!(
        m,
        AaregOptions,
        PSpline,
        CoxCountOutput,
        LinkFunctionParams,
        CoxPHModel,
        Subject,
        CchMethod,
        CohortData,
        ClogitDataSet,
        ConditionalLogisticRegression,
    );

    Ok(())
}
