use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        survexp,
        survexp_individual,
        create_simple_ratetable,
        is_ratetable,
        ratetable_date,
        days_to_date,
        summary_pyears,
        pyears_by_cell,
        pyears_ci,
        survexp_us,
        survexp_mn,
        survexp_usr,
        compute_expected_survival,
    );

    register_classes!(
        m,
        RateTable,
        RateDimension,
        DimType,
        SurvExpResult,
        RatetableDateResult,
        PyearsSummary,
        PyearsCell,
        ExpectedSurvivalResult,
    );

    Ok(())
}
