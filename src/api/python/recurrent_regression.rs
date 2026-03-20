use super::*;

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gap_time_model, m)?)?;
    m.add_function(wrap_pyfunction!(pwp_gap_time, m)?)?;
    m.add_function(wrap_pyfunction!(joint_frailty_model, m)?)?;
    m.add_function(wrap_pyfunction!(andersen_gill, m)?)?;
    m.add_function(wrap_pyfunction!(marginal_recurrent_model, m)?)?;
    m.add_function(wrap_pyfunction!(wei_lin_weissfeld, m)?)?;
    m.add_class::<GapTimeResult>()?;
    m.add_class::<JointFrailtyResult>()?;
    m.add_class::<MarginalModelResult>()?;
    m.add_class::<FrailtyDistribution>()?;
    m.add_class::<MarginalMethod>()?;

    m.add_function(wrap_pyfunction!(mixture_cure_model, m)?)?;
    m.add_function(wrap_pyfunction!(promotion_time_cure_model, m)?)?;
    m.add_function(wrap_pyfunction!(bounded_cumulative_hazard_model, m)?)?;
    m.add_function(wrap_pyfunction!(non_mixture_cure_model, m)?)?;
    m.add_function(wrap_pyfunction!(compare_cure_models, m)?)?;
    m.add_function(wrap_pyfunction!(predict_bounded_cumulative_hazard, m)?)?;
    m.add_function(wrap_pyfunction!(predict_non_mixture_survival, m)?)?;
    m.add_class::<MixtureCureResult>()?;
    m.add_class::<PromotionTimeCureResult>()?;
    m.add_class::<CureDistribution>()?;
    m.add_class::<BoundedCumulativeHazardConfig>()?;
    m.add_class::<BoundedCumulativeHazardResult>()?;
    m.add_class::<NonMixtureType>()?;
    m.add_class::<NonMixtureCureConfig>()?;
    m.add_class::<NonMixtureCureResult>()?;
    m.add_class::<CureModelComparisonResult>()?;

    m.add_function(wrap_pyfunction!(elastic_net_cox, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_cv, m)?)?;
    m.add_function(wrap_pyfunction!(elastic_net_cox_path, m)?)?;
    m.add_class::<ElasticNetCoxResult>()?;
    m.add_class::<ElasticNetCoxPath>()?;

    m.add_function(wrap_pyfunction!(fast_cox, m)?)?;
    m.add_function(wrap_pyfunction!(fast_cox_path, m)?)?;
    m.add_function(wrap_pyfunction!(fast_cox_cv, m)?)?;
    m.add_class::<FastCoxConfig>()?;
    m.add_class::<FastCoxResult>()?;
    m.add_class::<FastCoxPath>()?;
    m.add_class::<ScreeningRule>()?;

    m.add_function(wrap_pyfunction!(group_lasso_cox, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_boosting_cox, m)?)?;
    m.add_function(wrap_pyfunction!(sis_cox, m)?)?;
    m.add_function(wrap_pyfunction!(stability_selection_cox, m)?)?;
    m.add_class::<GroupLassoConfig>()?;
    m.add_class::<GroupLassoResult>()?;
    m.add_class::<SparseBoostingConfig>()?;
    m.add_class::<SparseBoostingResult>()?;
    m.add_class::<SISConfig>()?;
    m.add_class::<SISResult>()?;
    m.add_class::<StabilitySelectionConfig>()?;
    m.add_class::<StabilitySelectionResult>()?;

    m.add_function(wrap_pyfunction!(cause_specific_cox, m)?)?;
    m.add_function(wrap_pyfunction!(cause_specific_cox_all, m)?)?;
    m.add_class::<CauseSpecificCoxConfig>()?;
    m.add_class::<CauseSpecificCoxResult>()?;
    m.add_class::<CensoringType>()?;

    m.add_function(wrap_pyfunction!(joint_competing_risks, m)?)?;
    m.add_class::<JointCompetingRisksConfig>()?;
    m.add_class::<JointCompetingRisksResult>()?;
    m.add_class::<CauseResult>()?;
    m.add_class::<CorrelationType>()?;

    m.add_function(wrap_pyfunction!(pwp_model, m)?)?;
    m.add_function(wrap_pyfunction!(wlw_model, m)?)?;
    m.add_function(wrap_pyfunction!(negative_binomial_frailty, m)?)?;
    m.add_function(wrap_pyfunction!(anderson_gill_model, m)?)?;
    m.add_class::<PWPConfig>()?;
    m.add_class::<PWPResult>()?;
    m.add_class::<PWPTimescale>()?;
    m.add_class::<WLWConfig>()?;
    m.add_class::<WLWResult>()?;
    m.add_class::<NegativeBinomialFrailtyConfig>()?;
    m.add_class::<NegativeBinomialFrailtyResult>()?;
    m.add_class::<AndersonGillResult>()?;

    Ok(())
}
