use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tmerge, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge2, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge3, m)?)?;
    m.add_function(wrap_pyfunction!(survsplit, m)?)?;
    m.add_function(wrap_pyfunction!(survcondense, m)?)?;
    m.add_function(wrap_pyfunction!(surv2data, m)?)?;
    m.add_function(wrap_pyfunction!(to_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(from_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(aeq_surv, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_str, m)?)?;
    m.add_function(wrap_pyfunction!(strata, m)?)?;
    m.add_function(wrap_pyfunction!(strata_str, m)?)?;
    m.add_function(wrap_pyfunction!(neardate, m)?)?;
    m.add_function(wrap_pyfunction!(neardate_str, m)?)?;
    m.add_function(wrap_pyfunction!(tcut, m)?)?;
    m.add_function(wrap_pyfunction!(tcut_expand, m)?)?;
    m.add_function(wrap_pyfunction!(rttright, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_stratified, m)?)?;

    register_classes!(
        m,
        SplitResult,
        CondenseResult,
        Surv2DataResult,
        TimelineResult,
        IntervalResult,
        AeqSurvResult,
        ClusterResult,
        StrataResult,
        NearDateResult,
        TcutResult,
        RttrightResult,
    );

    Ok(())
}
