use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(agsurv4, m)?)?;
    m.add_function(wrap_pyfunction!(agsurv5, m)?)?;
    m.add_function(wrap_pyfunction!(cox_survfit_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfit_confint_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfit0_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfit0_aj_py, m)?)?;
    m.add_function(wrap_pyfunction!(survmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(summary_survfit_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_survfit_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfitaj_py, m)?)?;
    m.add_function(wrap_pyfunction!(survdiff_py, m)?)?;
    m.add_function(wrap_pyfunction!(survdiff_one_sample_py, m)?)?;
    m.add_function(wrap_pyfunction!(finegray, m)?)?;
    m.add_function(wrap_pyfunction!(finegray_regression, m)?)?;
    m.add_function(wrap_pyfunction!(competing_risks_cif, m)?)?;
    m.add_function(wrap_pyfunction!(survreg, m)?)?;
    m.add_function(wrap_pyfunction!(survreg_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(survreg_dtest, m)?)?;
    m.add_function(wrap_pyfunction!(dsurvreg, m)?)?;
    m.add_function(wrap_pyfunction!(psurvreg, m)?)?;
    m.add_function(wrap_pyfunction!(qsurvreg, m)?)?;
    m.add_function(wrap_pyfunction!(rsurvreg, m)?)?;
    m.add_function(wrap_pyfunction!(flexible_parametric_model, m)?)?;
    m.add_function(wrap_pyfunction!(restricted_cubic_spline, m)?)?;
    m.add_function(wrap_pyfunction!(predict_hazard_spline, m)?)?;
    m.add_function(wrap_pyfunction!(brier_py, m)?)?;
    m.add_function(wrap_pyfunction!(survobrien_py, m)?)?;
    m.add_function(wrap_pyfunction!(nelson_aalen_py, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(survmean_curves_py, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_survfit_curves_py, m)?)?;
    m.add_function(wrap_pyfunction!(rmst_comparison_py, m)?)?;
    m.add_function(wrap_pyfunction!(rmst_optimal_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(number_needed_to_treat_py, m)?)?;
    m.add_function(wrap_pyfunction!(turnbull_py, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_survival_py, m)?)?;
    m.add_function(wrap_pyfunction!(hazard_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(survival_at_times_py, m)?)?;
    m.add_function(wrap_pyfunction!(life_table_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfitresid_py, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_py, m)?)?;
    m.add_function(wrap_pyfunction!(survfitresid_aj_py, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_aj_py, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_gee_regression, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit_py, m)?)?;
    m.add_function(wrap_pyfunction!(survcheck_py, m)?)?;
    m.add_function(wrap_pyfunction!(royston_py, m)?)?;
    m.add_function(wrap_pyfunction!(yates_py, m)?)?;
    m.add_function(wrap_pyfunction!(yates_risk_py, m)?)?;
    m.add_function(wrap_pyfunction!(population_means_py, m)?)?;
    m.add_function(wrap_pyfunction!(uno_c_index, m)?)?;
    m.add_function(wrap_pyfunction!(compare_uno_c_indices, m)?)?;
    m.add_function(wrap_pyfunction!(c_index_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(gonen_heller_concordance, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_auc, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_dynamic_auc, m)?)?;
    m.add_function(wrap_pyfunction!(rcll, m)?)?;
    m.add_function(wrap_pyfunction!(rcll_single_time, m)?)?;
    m.add_function(wrap_pyfunction!(nsk, m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::reliability::core::reliability, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::reliability::core::reliability_inverse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::reliability::core::hazard_to_reliability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::reliability::core::failure_probability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::reliability::core::conditional_reliability,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::reliability::core::mean_residual_life,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(survfit_from_hazard, m)?)?;
    m.add_function(wrap_pyfunction!(survfit_from_cumhaz, m)?)?;
    m.add_function(wrap_pyfunction!(survfit_from_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(survfit_multistate, m)?)?;
    m.add_function(wrap_pyfunction!(condition_cox_survfit_curves, m)?)?;
    m.add_function(wrap_pyfunction!(step_values_at, m)?)?;
    m.add_function(wrap_pyfunction!(step_matrix_values_at, m)?)?;
    m.add_function(wrap_pyfunction!(cox_survfit_from_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(basehaz, m)?)?;
    m.add_function(wrap_pyfunction!(statefig_py, m)?)?;

    register_classes!(
        m,
        GEEConfig,
        GEEResult,
        SurvfitKMResult,
        SurvfitCounts,
        SurvfitInfluence,
        ConfidenceBands,
        SurvmeanTable,
        SurvfitQuantiles,
        SurvfitResid,
        SurvfitAJResid,
        SurvfitAJResult,
        SurvfitAJCounts,
        SurvfitAJInfluence,
        FineGrayOutput,
        FineGrayResult,
        CompetingRisksCIF,
        SurvregFit,
        SurvregData,
        SurvregControl,
        SurvregDistribution,
        SurvregFamily,
        SurvregTransform,
        SplineConfig,
        FlexibleParametricResult,
        RestrictedCubicSplineResult,
        HazardSplineResult,
        SurvDiffResult,
        SurvObrienExpansion,
        BrierResult,
        NelsonAalenResult,
        LogRankResult,
        SurvfitSummaryRow,
        SurvfitCurveQuantiles,
        RmstGroupResult,
        RmstComparisonResult,
        RMSTOptimalThresholdResult,
        ChangepointInfo,
        NNTResult,
        TurnbullCurve,
        TurnbullResult,
        ConditionalSurvivalResult,
        HazardRatioResult,
        SurvivalAtTimeResult,
        LifeTableResult,
        StateFigResult,
        StateFigArrow,
        AggregateSurvfitResult,
        AggregateGroups,
        GroupingFactor,
        SurvCheckResult,
        SurvCheckTransitions,
        SurvCheckEvents,
        SurvCheckFlags,
        SurvCheckProblem,
        RoystonResult,
        YatesResult,
        YatesEstimate,
        YatesContrast,
        UnoCIndexResult,
        ConcordanceComparisonResult,
        CIndexDecompositionResult,
        GonenHellerResult,
        TimeDepAUCResult,
        CumulativeDynamicAUCResult,
        RCLLResult,
        NaturalSplineKnot,
        SplineBasisResult,
        AnovaCoxphResult,
        AnovaRow,
        CipoissonResult,
        ReliabilityResult,
        ReliabilityScale,
        SurvfitMatrixResult,
    );

    Ok(())
}
