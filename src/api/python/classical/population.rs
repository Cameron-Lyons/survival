use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_ratetable, m)?)?;
    m.add_function(wrap_pyfunction!(ratetable_date, m)?)?;
    m.add_function(wrap_pyfunction!(days_to_date, m)?)?;
    m.add_function(wrap_pyfunction!(match_ratetable_py, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_us, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_usr, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_mn, m)?)?;
    m.add_function(wrap_pyfunction!(pyears_py, m)?)?;
    m.add_function(wrap_pyfunction!(summary_pyears_py, m)?)?;
    m.add_function(wrap_pyfunction!(survexp_py, m)?)?;

    register_classes!(
        m,
        DimType,
        RateTable,
        RatetableCheck,
        CalendarDate,
        MatchRatetableResult,
        PyearsResult,
        PyearsSummary,
        SurvExpResult,
    );

    Ok(())
}
