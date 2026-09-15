use super::*;
use crate::concordance::{
    ConcordanceCounts, ConcordanceFit, ConcordanceOptions, ConcordanceRanks, TimeWeight,
    concordancefit,
};
use crate::core::{CoxschoResiduals, SurvResponse};
use ndarray::ArrayView2;

fn covariate_view(covariates: &CovariateMatrix) -> PyResult<ArrayView2<'_, f64>> {
    ArrayView2::from_shape((covariates.n_obs, covariates.n_vars), &covariates.values)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn concordance_options(
    timewt: &str,
    ymin: Option<f64>,
    ymax: Option<f64>,
    influence: u8,
    ranks: bool,
    reverse: bool,
    timefix: bool,
    keepstrata: usize,
    std_err: bool,
) -> PyResult<ConcordanceOptions> {
    Ok(ConcordanceOptions {
        timewt: TimeWeight::parse(timewt)?,
        ymin,
        ymax,
        influence,
        ranks,
        reverse,
        timefix,
        keepstrata,
        std_err,
    })
}

/// R's `concordancefit` for right-censored data.
#[pyfunction(name = "concordancefit")]
#[pyo3(signature = (survival, x, weights=None, strata=None, cluster=None, timewt="n", ymin=None, ymax=None, influence=0, ranks=false, reverse=false, timefix=true, keepstrata=10, std_err=true))]
#[allow(clippy::too_many_arguments)]
fn concordancefit_py(
    survival: &SurvivalData,
    x: &CovariateMatrix,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    cluster: Option<Vec<i32>>,
    timewt: &str,
    ymin: Option<f64>,
    ymax: Option<f64>,
    influence: u8,
    ranks: bool,
    reverse: bool,
    timefix: bool,
    keepstrata: usize,
    std_err: bool,
) -> PyResult<ConcordanceFit> {
    let options = concordance_options(
        timewt, ymin, ymax, influence, ranks, reverse, timefix, keepstrata, std_err,
    )?;
    Ok(concordancefit(
        SurvResponse::Right(survival),
        covariate_view(x)?,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        cluster.as_deref(),
        &options,
    )?)
}

/// R's `concordancefit` for (start, stop] data.
#[pyfunction(name = "concordancefit_counting")]
#[pyo3(signature = (counting, x, weights=None, strata=None, cluster=None, timewt="n", ymin=None, ymax=None, influence=0, ranks=false, reverse=false, timefix=true, keepstrata=10, std_err=true))]
#[allow(clippy::too_many_arguments)]
fn concordancefit_counting_py(
    counting: &CountingProcessData,
    x: &CovariateMatrix,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    cluster: Option<Vec<i32>>,
    timewt: &str,
    ymin: Option<f64>,
    ymax: Option<f64>,
    influence: u8,
    ranks: bool,
    reverse: bool,
    timefix: bool,
    keepstrata: usize,
    std_err: bool,
) -> PyResult<ConcordanceFit> {
    let options = concordance_options(
        timewt, ymin, ymax, influence, ranks, reverse, timefix, keepstrata, std_err,
    )?;
    Ok(concordancefit(
        SurvResponse::Counting(counting),
        covariate_view(x)?,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        cluster.as_deref(),
        &options,
    )?)
}

/// `coxscore2`: score residuals (`n x p`) of a right-censored Cox model.
#[pyfunction(name = "coxscore2")]
#[pyo3(signature = (survival, covariates, score, weights=None, strata=None, ties="efron"))]
fn coxscore2_py(
    survival: &SurvivalData,
    covariates: &CovariateMatrix,
    score: Vec<f64>,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    ties: &str,
) -> PyResult<Vec<Vec<f64>>> {
    let resid = crate::scoring::coxscore2(
        survival,
        covariate_view(covariates)?,
        &score,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        kernel_ties(ties)?,
    )?;
    Ok(resid.outer_iter().map(|row| row.to_vec()).collect())
}

/// `agscore3`: score residuals (`n x p`) of a Cox model on (start, stop] data.
#[pyfunction(name = "agscore3")]
#[pyo3(signature = (counting, covariates, score, weights=None, strata=None, ties="efron"))]
fn agscore3_py(
    counting: &CountingProcessData,
    covariates: &CovariateMatrix,
    score: Vec<f64>,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    ties: &str,
) -> PyResult<Vec<Vec<f64>>> {
    let resid = crate::scoring::agscore3(
        counting,
        covariate_view(covariates)?,
        &score,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        kernel_ties(ties)?,
    )?;
    Ok(resid.outer_iter().map(|row| row.to_vec()).collect())
}

/// `coxscho`: Schoenfeld residuals of a right-censored Cox model.
#[pyfunction(name = "schoenfeld_residuals")]
#[pyo3(signature = (survival, covariates, score, weights=None, strata=None, ties="efron"))]
fn schoenfeld_residuals_py(
    survival: &SurvivalData,
    covariates: &CovariateMatrix,
    score: Vec<f64>,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    ties: &str,
) -> PyResult<CoxschoResiduals> {
    Ok(crate::core::schoenfeld_residuals(
        SurvResponse::Right(survival),
        covariate_view(covariates)?,
        &score,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        kernel_ties(ties)?,
    )?)
}

/// `coxscho`: Schoenfeld residuals of a Cox model on (start, stop] data.
#[pyfunction(name = "schoenfeld_residuals_counting")]
#[pyo3(signature = (counting, covariates, score, weights=None, strata=None, ties="efron"))]
fn schoenfeld_residuals_counting_py(
    counting: &CountingProcessData,
    covariates: &CovariateMatrix,
    score: Vec<f64>,
    weights: Option<&Weights>,
    strata: Option<Vec<i32>>,
    ties: &str,
) -> PyResult<CoxschoResiduals> {
    Ok(crate::core::schoenfeld_residuals(
        SurvResponse::Counting(counting),
        covariate_view(covariates)?,
        &score,
        weights.map(|w| w.values.as_slice()),
        strata.as_deref(),
        kernel_ties(ties)?,
    )?)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(concordancefit_py, m)?)?;
    m.add_function(wrap_pyfunction!(concordancefit_counting_py, m)?)?;
    m.add_function(wrap_pyfunction!(coxscore2_py, m)?)?;
    m.add_function(wrap_pyfunction!(agscore3_py, m)?)?;
    m.add_function(wrap_pyfunction!(schoenfeld_residuals_py, m)?)?;
    m.add_function(wrap_pyfunction!(schoenfeld_residuals_counting_py, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_cox_ci, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_survreg_ci, m)?)?;
    m.add_function(wrap_pyfunction!(cv_cox_concordance, m)?)?;
    m.add_function(wrap_pyfunction!(cv_survreg_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(lrt_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(wald_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(score_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival, m)?)?;
    m.add_function(wrap_pyfunction!(sample_size_survival_freedman, m)?)?;
    m.add_function(wrap_pyfunction!(power_survival, m)?)?;
    m.add_function(wrap_pyfunction!(expected_events, m)?)?;
    m.add_function(wrap_pyfunction!(calibration, m)?)?;
    m.add_function(wrap_pyfunction!(predict_cox, m)?)?;
    m.add_function(wrap_pyfunction!(risk_stratification, m)?)?;
    m.add_function(wrap_pyfunction!(td_auc, m)?)?;
    m.add_function(wrap_pyfunction!(d_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(one_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot, m)?)?;
    m.add_function(wrap_pyfunction!(brier_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(multi_time_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(smoothed_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(advanced_calibration_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(time_dependent_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(mc_dropout_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_regression_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_prediction_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(conformal_survival, m)?)?;
    m.add_function(wrap_pyfunction!(bayesian_bootstrap_survival, m)?)?;
    m.add_function(wrap_pyfunction!(jackknife_plus_survival, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_analysis_batch, m)?)?;
    m.add_function(wrap_pyfunction!(decision_curve_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(clinical_utility_at_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(compare_decision_curves, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fairness_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(assess_model_robustness, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(hyperparameter_search, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_models, m)?)?;
    m.add_function(wrap_pyfunction!(nested_cross_validation, m)?)?;
    m.add_function(wrap_pyfunction!(compute_model_selection_criteria, m)?)?;
    m.add_function(wrap_pyfunction!(compare_models, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cv_score, m)?)?;
    m.add_function(wrap_pyfunction!(survival_meta_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(generate_forest_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(publication_bias_tests, m)?)?;
    m.add_function(wrap_pyfunction!(joint_longitudinal_model, m)?)?;
    m.add_function(wrap_pyfunction!(landmark_cox_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(longitudinal_dynamic_pred, m)?)?;
    m.add_function(wrap_pyfunction!(time_varying_cox, m)?)?;
    m.add_function(wrap_pyfunction!(km_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(forest_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(calibration_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(generate_survival_report, m)?)?;
    m.add_function(wrap_pyfunction!(roc_plot_data, m)?)?;
    m.add_function(wrap_pyfunction!(warranty_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(renewal_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_growth, m)?)?;
    m.add_function(wrap_pyfunction!(qaly_calculation, m)?)?;
    m.add_function(wrap_pyfunction!(qaly_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(incremental_cost_effectiveness, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(qtwist_sensitivity, m)?)?;

    register_classes!(
        m,
        ConcordanceFit,
        ConcordanceCounts,
        ConcordanceRanks,
        CoxschoResiduals,
        BootstrapResult,
        CVResult,
        TestResult,
        SampleSizeResult,
        AccrualResult,
        CalibrationResult,
        PredictionResult,
        RiskStratificationResult,
        TdAUCResult,
        DCalibrationResult,
        OneCalibrationResult,
        CalibrationPlotData,
        BrierCalibrationResult,
        MultiTimeCalibrationResult,
        SmoothedCalibrationCurve,
        AdvancedCalibrationResult,
        TimeDependentCalibrationResult,
        MCDropoutConfig,
        UncertaintyResult,
        EnsembleUncertaintyResult,
        QuantileRegressionResult,
        CalibrationUncertaintyResult,
        ConformalSurvivalConfig,
        ConformalSurvivalResult,
        BayesianBootstrapConfig,
        BayesianBootstrapResult,
        JackknifePlusConfig,
        JackknifePlusResult,
        LandmarkResult,
        DecisionCurveResult,
        ClinicalUtilityResult,
        ModelComparisonResult,
        FairnessMetrics,
        RobustnessResult,
        SubgroupAnalysisResult,
        SearchStrategy,
        HyperparameterSearchConfig,
        HyperparameterResult,
        BenchmarkResult,
        NestedCVResult,
        ModelSelectionCriteria,
        SurvivalModelComparison,
        CrossValidatedScore,
        MetaAnalysisConfig,
        MetaAnalysisResult,
        MetaForestPlotData,
        PublicationBiasResult,
        JointModelConfig,
        JointLongSurvResult,
        LandmarkAnalysisResult,
        LongDynamicPredResult,
        TimeVaryingCoxResult,
        KaplanMeierPlotData,
        ForestPlotData,
        CalibrationCurveData,
        SurvivalReport,
        ROCPlotData,
        WarrantyConfig,
        WarrantyResult,
        RenewalResult,
        ReliabilityGrowthResult,
        QALYResult,
        QTWISTResult,
    );

    Ok(())
}
