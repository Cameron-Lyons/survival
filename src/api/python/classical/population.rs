use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(survexp, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_individual, m)?)?;
    m.add_function(wrap_pyfunction!(create_simple_ratetable, m)?)?;
    m.add_function(wrap_pyfunction!(is_ratetable, m)?)?;
    m.add_function(wrap_pyfunction!(ratetable_date, m)?)?;
    m.add_function(wrap_pyfunction!(days_to_date, m)?)?;
    m.add_function(wrap_pyfunction!(summary_pyears, m)?)?;
    m.add_function(wrap_pyfunction!(pyears_by_cell, m)?)?;
    m.add_function(wrap_pyfunction!(pyears_ci, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_us, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_mn, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_usr, m)?)?;
    m.add_function(wrap_pyfunction!(compute_expected_survival, m)?)?;

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
