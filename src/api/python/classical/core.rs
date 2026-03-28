use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        perform_cox_regression_frailty,
        perform_pyears_calculation,
        perform_concordance1_calculation,
        perform_concordance3_calculation,
        perform_concordance_calculation,
        perform_score_calculation,
        perform_agscore3_calculation,
        perform_pystep_calculation,
        perform_pystep_simple_calculation,
        aareg,
        collapse,
        cox_callback,
        coxcount1,
        coxcount2,
        norisk,
        cipoisson,
        cipoisson_exact,
        cipoisson_anscombe,
        compute_concordance,
        agexact,
        compute_baseline_survival_steps,
        compute_tied_baseline_summaries,
        agmart,
        coxmart,
    );

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
