use super::*;

#[pyfunction(name = "rttright_time_matrix")]
#[pyo3(signature = (time, status, times, weights=None, strata=None, timefix=true, renorm=true))]
#[allow(clippy::too_many_arguments)]
fn rttright_time_matrix_py(
    py: Python<'_>,
    time: Vec<f64>,
    status: Vec<i32>,
    times: Vec<f64>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    timefix: bool,
    renorm: bool,
) -> PyResult<Vec<Vec<f64>>> {
    py.detach(move || rttright_time_matrix(time, status, times, weights, strata, timefix, renorm))
}

#[pyfunction(name = "rttright_counting_matrix")]
#[pyo3(signature = (start, stop, status, weights, last, strata, delta, times=None))]
#[allow(clippy::too_many_arguments)]
fn rttright_counting_matrix_py(
    py: Python<'_>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weights: Vec<f64>,
    last: Vec<bool>,
    strata: Vec<i32>,
    delta: f64,
    times: Option<Vec<f64>>,
) -> PyResult<Vec<Vec<f64>>> {
    py.detach(move || {
        rttright_counting_matrix(start, stop, status, weights, last, strata, delta, times)
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tmerge, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge_plan, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge2, m)?)?;
    m.add_function(wrap_pyfunction!(tmerge3, m)?)?;
    m.add_function(wrap_pyfunction!(survsplit, m)?)?;
    m.add_function(wrap_pyfunction!(survcondense, m)?)?;
    m.add_function(wrap_pyfunction!(survcondense_plan, m)?)?;
    m.add_function(wrap_pyfunction!(surv2data, m)?)?;
    m.add_function(wrap_pyfunction!(surv2data_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(from_timeline_rows, m)?)?;
    m.add_function(wrap_pyfunction!(to_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(from_timeline, m)?)?;
    m.add_function(wrap_pyfunction!(lvcf_indices, m)?)?;
    m.add_function(wrap_pyfunction!(lvcf_numeric_indices, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_replacements, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_numeric_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_numeric_str, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_str_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(nostutter_str_str, m)?)?;
    m.add_function(wrap_pyfunction!(aeq_surv, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_str, m)?)?;
    m.add_function(wrap_pyfunction!(strata, m)?)?;
    m.add_function(wrap_pyfunction!(strata_compact, m)?)?;
    m.add_function(wrap_pyfunction!(strata_str, m)?)?;
    m.add_function(wrap_pyfunction!(neardate, m)?)?;
    m.add_function(wrap_pyfunction!(neardate_str, m)?)?;
    m.add_function(wrap_pyfunction!(tcut, m)?)?;
    m.add_function(wrap_pyfunction!(tcut_expand, m)?)?;
    m.add_function(wrap_pyfunction!(rttright, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_counting_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_stratified, m)?)?;
    m.add_function(wrap_pyfunction!(rttright_time_matrix_py, m)?)?;

    register_classes!(
        m,
        SplitResult,
        CondensePlanResult,
        CondenseResult,
        Surv2DataResult,
        Surv2TimelineResult,
        FromTimelineRowsResult,
        TimelineResult,
        IntervalResult,
        TmergePlanResult,
        AeqSurvResult,
        ClusterResult,
        StrataResult,
        NearDateResult,
        TcutResult,
        RttrightResult,
    );

    Ok(())
}
