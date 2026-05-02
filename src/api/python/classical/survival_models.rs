use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(agsurv4, m)?)?;
    m.add_function(wrap_pyfunction!(agsurv5, m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm, m)?)?;
    m.add_function(wrap_pyfunction!(survfitkm_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(survfitaj, m)?)?;
    m.add_function(wrap_pyfunction!(compute_logrank_components, m)?)?;
    m.add_function(wrap_pyfunction!(survdiff2, m)?)?;
    m.add_function(wrap_pyfunction!(finegray, m)?)?;
    m.add_function(wrap_pyfunction!(finegray_regression, m)?)?;
    m.add_function(wrap_pyfunction!(competing_risks_cif, m)?)?;
    m.add_function(wrap_pyfunction!(survreg, m)?)?;
    m.add_function(wrap_pyfunction!(brier, m)?)?;
    m.add_function(wrap_pyfunction!(integrated_brier, m)?)?;
    m.add_function(wrap_pyfunction!(survobrien, m)?)?;
    m.add_function(wrap_pyfunction!(nelson_aalen_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(stratified_kaplan_meier, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_test, m)?)?;
    m.add_function(wrap_pyfunction!(fleming_harrington_test, m)?)?;
    m.add_function(wrap_pyfunction!(logrank_trend, m)?)?;
    m.add_function(wrap_pyfunction!(rmst, m)?)?;
    m.add_function(wrap_pyfunction!(rmst_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(rmst_optimal_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(survival_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_incidence, m)?)?;
    m.add_function(wrap_pyfunction!(number_needed_to_treat, m)?)?;
    m.add_function(wrap_pyfunction!(conditional_survival, m)?)?;
    m.add_function(wrap_pyfunction!(hazard_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(survival_at_times, m)?)?;
    m.add_function(wrap_pyfunction!(life_table, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_fast, m)?)?;
    m.add_function(wrap_pyfunction!(pseudo_gee_regression, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_survfit_by_group, m)?)?;
    m.add_function(wrap_pyfunction!(survcheck, m)?)?;
    m.add_function(wrap_pyfunction!(survcheck_simple, m)?)?;
    m.add_function(wrap_pyfunction!(royston, m)?)?;
    m.add_function(wrap_pyfunction!(royston_from_model, m)?)?;
    m.add_function(wrap_pyfunction!(yates, m)?)?;
    m.add_function(wrap_pyfunction!(yates_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(yates_pairwise, m)?)?;
    m.add_function(wrap_pyfunction!(uno_c_index, m)?)?;
    m.add_function(wrap_pyfunction!(compare_uno_c_indices, m)?)?;
    m.add_function(wrap_pyfunction!(c_index_decomposition, m)?)?;
    m.add_function(wrap_pyfunction!(gonen_heller_concordance, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_auc, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_dynamic_auc, m)?)?;
    m.add_function(wrap_pyfunction!(rcll, m)?)?;
    m.add_function(wrap_pyfunction!(rcll_single_time, m)?)?;
    m.add_function(wrap_pyfunction!(ridge_fit, m)?)?;
    m.add_function(wrap_pyfunction!(ridge_cv, m)?)?;
    m.add_function(wrap_pyfunction!(nsk, m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph, m)?)?;
    m.add_function(wrap_pyfunction!(anova_coxph_single, m)?)?;
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
    m.add_function(wrap_pyfunction!(basehaz, m)?)?;
    m.add_function(wrap_pyfunction!(statefig, m)?)?;
    m.add_function(wrap_pyfunction!(statefig_matplotlib_code, m)?)?;
    m.add_function(wrap_pyfunction!(statefig_transition_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(statefig_validate, m)?)?;

    register_classes!(
        m,
        GEEConfig,
        GEEResult,
        SurvFitKMOutput,
        SurvfitKMOptions,
        KaplanMeierConfig,
        SurvFitAJ,
        FineGrayOutput,
        FineGrayResult,
        CompetingRisksCIF,
        SurvivalFit,
        SurvregConfig,
        DistributionType,
        SurvDiffResult,
        SurvObrienResult,
        NelsonAalenResult,
        StratifiedKMResult,
        LogRankResult,
        TrendTestResult,
        RMSTResult,
        RMSTComparisonResult,
        RMSTOptimalThresholdResult,
        ChangepointInfo,
        MedianSurvivalResult,
        CumulativeIncidenceResult,
        NNTResult,
        ConditionalSurvivalResult,
        HazardRatioResult,
        SurvivalAtTimeResult,
        LifeTableResult,
        StateFigData,
        PseudoResult,
        AggregateSurvfitResult,
        SurvCheckResult,
        RoystonResult,
        YatesResult,
        YatesPairwiseResult,
        UnoCIndexResult,
        ConcordanceComparisonResult,
        CIndexDecompositionResult,
        GonenHellerResult,
        TimeDepAUCResult,
        CumulativeDynamicAUCResult,
        RCLLResult,
        RidgePenalty,
        RidgeResult,
        NaturalSplineKnot,
        SplineBasisResult,
        AnovaCoxphResult,
        AnovaRow,
        ReliabilityResult,
        ReliabilityScale,
        SurvfitMatrixResult,
    );

    Ok(())
}
