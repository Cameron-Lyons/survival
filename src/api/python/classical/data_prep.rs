use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aeq_surv_py, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_py, m)?)?;
    m.add_function(wrap_pyfunction!(strata_py, m)?)?;
    m.add_function(wrap_pyfunction!(lvcf_py, m)?)?;
    m.add_function(wrap_pyfunction!(neardate_py, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_py, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_py, m)?)?;
    m.add_function(wrap_pyfunction!(surv2counting_py, m)?)?;
    m.add_function(wrap_pyfunction!(totimeline_py, m)?)?;
    m.add_function(wrap_pyfunction!(survcondense_py, m)?)?;
    m.add_function(wrap_pyfunction!(survsplit_py, m)?)?;
    m.add_function(wrap_pyfunction!(tcut_py, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge_step_py, m)?)?;

    register_classes!(
        m,
        AeqSurvResult,
        ClusterResult,
        StrataResult,
        RttrightResult,
        Surv2CountingResult,
        TotimelineResult,
        SurvcondenseResult,
        SurvSplitResult,
        TcutResult,
        TmergeStep,
    );

    Ok(())
}
