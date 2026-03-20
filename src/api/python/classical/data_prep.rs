use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_functions!(
        m,
        tmerge,
        tmerge2,
        tmerge3,
        survsplit,
        survcondense,
        surv2data,
        to_timeline,
        from_timeline,
        aeq_surv,
        cluster,
        cluster_str,
        strata,
        strata_str,
        neardate,
        neardate_str,
        tcut,
        tcut_expand,
        rttright,
        rttright_stratified,
    );

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
