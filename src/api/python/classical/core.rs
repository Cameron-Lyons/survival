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

/// Rebuilds an object pickled by its class's `__reduce__` from the class
/// and the state `internal::pickle` encoded.
#[pyfunction(name = "_unpickle")]
fn unpickle(cls: &Bound<'_, pyo3::types::PyType>, state: &[u8]) -> PyResult<Py<PyAny>> {
    use crate::internal::pickle::decode;
    let py = cls.py();
    macro_rules! restore {
        ($($class:ty),+ $(,)?) => {$(
            if cls.is(py.get_type::<$class>()) {
                return Ok(Py::new(py, decode::<$class>(py, state)?)?.into_any());
            }
        )+};
    }
    restore!(
        CoxPHFit,
        crate::regression::CoxphFitResult,
        crate::regression::TieMethod,
        CoxpenalFit,
        CoxPenalty,
        CoxPenaltyTerms,
        PenaltyHistory,
        crate::concordance::ConcordanceFit,
        crate::concordance::ConcordanceCounts,
        crate::concordance::ConcordanceRanks,
        SurvregFit,
        crate::regression::SurvregFitResult,
        SurvregControl,
        SurvregDistribution,
        SurvregFamily,
        SurvregTransform,
        SurvfitKMResult,
        SurvfitCounts,
        SurvfitInfluence,
        SurvfitAJResult,
        SurvfitAJCounts,
        SurvfitAJInfluence,
        crate::validation::AnovaRow,
        crate::validation::AnovaCoxphResult,
        crate::validation::YatesContrast,
        crate::validation::SurvCheckFlags,
        crate::validation::SurvCheckTransitions,
        crate::validation::SurvCheckEvents,
        crate::data_prep::TcutResult,
        crate::core::SplineBasisResult,
    );
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "_unpickle cannot restore a {}",
        cls.name()?
    )))
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aareg_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(cox_callback, m)?)?;
    m.add_class::<CoxPenaltyTerms>()?;
    m.add_function(wrap_pyfunction!(coxph_fit, m)?)?;
    m.add_function(wrap_pyfunction!(crate::regression::coxph_fit_raw, m)?)?;
    m.add_function(wrap_pyfunction!(coxphms_fit, m)?)?;
    m.add_function(wrap_pyfunction!(coxpenal_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(cch_borgan_fit, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount1_py, m)?)?;
    m.add_function(wrap_pyfunction!(coxcount2_py, m)?)?;
    m.add_function(wrap_pyfunction!(cipoisson_py, m)?)?;
    m.add_function(wrap_pyfunction!(pchisq_py, m)?)?;
    m.add_function(wrap_pyfunction!(pspline_basis_py, m)?)?;
    m.add_function(wrap_pyfunction!(agexact_py, m)?)?;
    m.add_function(wrap_pyfunction!(cox_zph_py, m)?)?;
    m.add_function(wrap_pyfunction!(cox_zph_smooth_py, m)?)?;
    m.add_function(wrap_pyfunction!(coxph_detail_py, m)?)?;
    m.add_function(wrap_pyfunction!(unpickle, m)?)?;

    register_classes!(
        m,
        AaregFitResult,
        PsplineBasis,
        CoxCountOutput,
        CoxPHFit,
        crate::regression::CoxphFitResult,
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
        CoxZphSmooth,
        AgexactFit,
        LinkFunctionParams,
        CchFitResult,
    );

    Ok(())
}
