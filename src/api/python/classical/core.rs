use super::*;

/// `coxcount1`: risk sets of right-censored data for `tt()` terms.
#[pyfunction(name = "coxcount1")]
#[pyo3(signature = (survival, strata=None))]
fn coxcount1_py(survival: &SurvivalData, strata: Option<Vec<i32>>) -> PyResult<CoxCountOutput> {
    Ok(crate::core::coxcount1(survival, strata.as_deref())?)
}

/// `coxcount2`: risk sets of (start, stop] data for `tt()` terms.
#[pyfunction(name = "coxcount2")]
#[pyo3(signature = (counting, strata=None))]
fn coxcount2_py(
    counting: &CountingProcessData,
    strata: Option<Vec<i32>>,
) -> PyResult<CoxCountOutput> {
    Ok(crate::core::coxcount2(counting, strata.as_deref())?)
}

/// The B-spline basis of a `pspline()` term.
#[pyfunction(name = "pspline_basis")]
fn pspline_basis_py(
    x: Vec<f64>,
    nterm: usize,
    degree: usize,
    boundary_knots: (f64, f64),
) -> PyResult<PsplineBasis> {
    Ok(crate::core::pspline_basis(
        &x,
        nterm,
        degree,
        boundary_knots,
    )?)
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aareg, m)?)?;
    m.add_function(wrap_pyfunction!(aareg_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cox_callback, m)?)?;
    m.add_class::<CoxPenaltyTerms>()?;
    m.add_function(wrap_pyfunction!(coxph_fit, m)?)?;
    m.add_function(wrap_pyfunction!(coxpenal_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cch_borgan_fit, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount1_py, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount2_py, m)?)?;
    m.add_function(wrap_pyfunction!(norisk_py, m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_py, m)?)?;
    m.add_function(wrap_pyfunction!(pspline_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(agexact_py, m)?)?;
    m.add_function(wrap_pyfunction!(cox_zph_py, m)?)?;
    m.add_function(wrap_pyfunction!(coxph_detail_py, m)?)?;
    m.add_function(wrap_pyfunction!(compute_baseline_survival_steps, m)?)?;
    m.add_function(wrap_pyfunction!(compute_tied_baseline_summaries, m)?)?;
    m.add_function(wrap_pyfunction!(cox_expected_baseline_by_stratum, m)?)?;

    register_classes!(
        m,
        AaregConfidenceInterval,
        AaregDiagnostics,
        AaregFitDetails,
        AaregFitResult,
        AaregOptions,
        AaregResult,
        PsplineBasis,
        CoxCountOutput,
        CoxPHFit,
        CoxPenalty,
        CoxpenalFit,
        PenaltyHistory,
        crate::regression::TieMethod,
        CoxPrediction,
        CoxTermsPrediction,
        Basehaz,
        CoxSurvfitCurve,
        SchoenfeldResiduals,
        CoxphDetail,
        CoxZph,
        CoxZphTest,
        AgexactFit,
        LinkFunctionParams,
        CchFitResult,
    );

    Ok(())
}
