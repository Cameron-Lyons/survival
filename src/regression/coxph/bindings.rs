//! Python input/output adapters; numerical work stays in the Rust methods.

use super::{
    Basehaz, CoxNewData, CoxPHFit, CoxPrediction, CoxSurvfitCurve, CoxTermsPrediction, CoxphData,
    CoxphOptions, PredictReference, SurvfitOptions, default_assign,
};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::matrix_rows;
use crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec};
#[cfg(feature = "python")]
use crate::internal::validation::validate_length;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph_diagnostics::SchoenfeldResiduals;
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymethods]
impl CoxPrediction {
    /// Independent writable float64 snapshots; absent errors remain None.
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::IntoPyArray;
        use pyo3::types::PyDict;
        let (fit, se) = py.detach(|| -> SurvivalResult<_> {
            if let Some(se) = &self.se_fit {
                validate_length(self.fit.len(), se.len(), "prediction errors")?;
            }
            Ok((self.fit.clone(), self.se_fit.clone()))
        })?;
        let result = PyDict::new(py);
        result.set_item("fit", fit.into_pyarray(py))?;
        result.set_item("se_fit", se.map(|se| se.into_pyarray(py)))?;
        Ok(result)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl CoxTermsPrediction {
    /// Independent writable float64 matrices, including empty dimensions.
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let result = crate::internal::numpy_utils::prediction_matrix_arrays(
            py,
            &self.fit,
            self.se_fit.as_deref(),
            self.n_columns,
        )?;
        result.set_item("constant", self.constant)?;
        Ok(result)
    }
}

/// Package owned Python buffers without scanning them under the GIL. The
/// receiving Rust method validates the rows, including its missing-value policy.
/// Extra row arguments without a design matrix are always an error.
pub(crate) fn newdata_from_python(
    x: Option<FloatMatrix>,
    strata: Option<IntVec>,
    offset: Option<FloatVec>,
    time: Option<FloatVec>,
    entry: Option<FloatVec>,
) -> SurvivalResult<Option<CoxNewData>> {
    let Some(x) = x else {
        if strata.is_some() || offset.is_some() || time.is_some() || entry.is_some() {
            return Err(SurvivalError::invalid_input(
                "new_strata, new_offset, new_time and new_entry require newdata",
            ));
        }
        return Ok(None);
    };
    Ok(Some(CoxNewData {
        x: x.into_inner(),
        strata: strata.map(IntVec::into_inner),
        offset: offset.map(FloatVec::into_inner),
        time: time.map(FloatVec::into_inner),
        entry: entry.map(FloatVec::into_inner),
    }))
}

#[pymethods]
impl CoxPHFit {
    /// Pickle and copy support (see `internal::pickle`).
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    #[getter]
    fn var(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var)
    }

    #[getter]
    fn naive_var(&self) -> Option<Vec<Vec<f64>>> {
        self.naive_var.as_ref().map(matrix_rows)
    }

    #[getter]
    fn x(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.x)
    }

    #[getter(nvar)]
    fn nvar_getter(&self) -> usize {
        self.nvar()
    }

    #[pyo3(name = "hazard_ratios")]
    fn hazard_ratios_py(&self) -> Vec<f64> {
        self.hazard_ratios()
    }

    /// `basehaz(fit, centered)`.
    #[pyo3(name = "basehaz", signature = (centered = true))]
    fn basehaz_py(&self, centered: bool) -> PyResult<Basehaz> {
        Ok(self.basehaz(centered)?)
    }

    /// `predict(fit, newdata, type, se.fit, reference)` for the vector-valued
    /// types `lp`, `risk`, `expected` and `survival`.
    #[pyo3(signature = (r#type = "lp", newdata = None, new_strata = None, new_offset = None, new_time = None, new_entry = None, se_fit = false, reference = "strata", *, collapse = None))]
    #[allow(clippy::too_many_arguments)]
    fn predict(
        &self,
        py: Python<'_>,
        r#type: &str,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        new_time: Option<FloatVec>,
        new_entry: Option<FloatVec>,
        se_fit: bool,
        reference: &str,
        collapse: Option<IntVec>,
    ) -> PyResult<CoxPrediction> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, new_time, new_entry)?;
        let reference = PredictReference::parse(reference)?;
        let newdata = newdata.as_ref();
        Ok(py.detach(|| {
            let prediction = match r#type {
                "lp" => self.predict_lp(newdata, se_fit, reference),
                "risk" => self.predict_risk(newdata, se_fit, reference),
                "expected" => self.predict_expected(newdata, se_fit),
                "survival" => self.predict_survival(newdata, se_fit),
                other => {
                    return Err(SurvivalError::invalid_input(format!(
                        "type must be 'lp', 'risk', 'expected' or 'survival', got '{other}'; use predict_terms for 'terms'"
                    )));
                }
            }?;
            match collapse.as_deref() {
                Some(group) => prediction.collapse(group),
                None => Ok(prediction),
            }
        })?)
    }

    /// `predict(fit, type = "terms")`; `assign` lists the columns of each
    /// term (default: one term per column).
    #[pyo3(name = "predict_terms", signature = (newdata = None, new_strata = None, new_offset = None, se_fit = false, reference = "sample", assign = None, *, collapse = None))]
    #[allow(clippy::too_many_arguments)]
    fn predict_terms_py(
        &self,
        py: Python<'_>,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        se_fit: bool,
        reference: &str,
        assign: Option<Vec<Vec<usize>>>,
        collapse: Option<IntVec>,
    ) -> PyResult<CoxTermsPrediction> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, None, None)?;
        let reference = PredictReference::parse(reference)?;
        let assign = assign.unwrap_or_else(|| default_assign(self.nvar()));
        Ok(py.detach(|| match collapse.as_deref() {
            Some(group) => {
                self.predict_terms_grouped(newdata.as_ref(), se_fit, reference, &assign, group)
            }
            None => self.predict_terms(newdata.as_ref(), se_fit, reference, &assign),
        })?)
    }

    /// `survfit(fit, newdata, stype, ctype, se.fit, censor, start.time)`.
    #[pyo3(name = "survfit", signature = (newdata = None, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_py(
        &self,
        py: Python<'_>,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, None, None)?;
        let options = SurvfitOptions {
            stype,
            ctype,
            se_fit,
            censor,
            start_time,
        };
        Ok(py.detach(|| self.survfit(newdata.as_ref(), options))?)
    }

    /// Survival at requested times: a NumPy matrix of shape (n_times, n_rows).
    /// Omit newdata to predict for the original training rows.
    #[pyo3(name = "predict_survival_at", signature = (times, newdata = None, new_strata = None, new_offset = None))]
    fn predict_survival_at_py(
        &self,
        py: Python<'_>,
        times: FloatVec,
        newdata: Option<FloatMatrix>,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
    ) -> PyResult<FloatMatrix> {
        let newdata = newdata_from_python(newdata, new_strata, new_offset, None, None)?;
        Ok(FloatMatrix::new(py.detach(|| {
            self.predict_survival_at(&times, newdata.as_ref())
        })?))
    }

    #[pyo3(name = "expected_survival", signature = (newdata, group, weights, new_strata=None, new_offset=None, y=None, times=None, method="ederer"))]
    #[allow(clippy::too_many_arguments)]
    fn expected_survival_py(
        &self,
        py: Python<'_>,
        newdata: FloatMatrix,
        group: IntVec,
        weights: FloatVec,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        y: Option<FloatVec>,
        times: Option<FloatVec>,
        method: &str,
    ) -> PyResult<crate::population::SurvExpResult> {
        let new = newdata_from_python(Some(newdata), new_strata, new_offset, None, None)?
            .expect("newdata supplied");
        let group = group
            .iter()
            .map(|&v| {
                usize::try_from(v)
                    .map_err(|_| SurvivalError::invalid_input("group codes must be nonnegative"))
            })
            .collect::<SurvivalResult<Vec<_>>>()?;
        Ok(py.detach(|| {
            self.expected_survival(
                &new,
                &group,
                &weights,
                y.as_deref(),
                times.as_deref(),
                method,
            )
        })?)
    }

    /// `residuals(fit, type = "martingale", weighted, collapse)`.
    #[pyo3(name = "martingale_residuals", signature = (weighted = false, collapse = None))]
    fn martingale_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.martingale_residuals(weighted, collapse.as_deref())?)
    }

    /// `residuals(fit, type = "deviance", weighted, collapse)`.
    #[pyo3(name = "deviance_residuals", signature = (weighted = false, collapse = None))]
    fn deviance_residuals_py(
        &self,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<f64>> {
        Ok(self.deviance_residuals(weighted, collapse.as_deref())?)
    }

    /// `residuals(fit, type = "score", weighted, collapse)`.
    #[pyo3(name = "score_residuals", signature = (weighted = false, collapse = None))]
    fn score_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let residuals = py.detach(|| self.score_residuals(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&residuals))
    }

    /// `residuals(fit, type = "dfbeta", weighted, collapse)`.
    #[pyo3(name = "dfbeta", signature = (weighted = true, collapse = None))]
    fn dfbeta_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let dfbeta = py.detach(|| self.dfbeta(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&dfbeta))
    }

    /// `residuals(fit, type = "dfbetas", weighted, collapse)`.
    #[pyo3(name = "dfbetas", signature = (weighted = true, collapse = None))]
    fn dfbetas_py(
        &self,
        py: Python<'_>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let dfbetas = py.detach(|| self.dfbetas(weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&dfbetas))
    }

    /// `residuals(fit, type = "schoenfeld", weighted)`.
    #[pyo3(name = "schoenfeld_residuals", signature = (weighted = false))]
    fn schoenfeld_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
    ) -> PyResult<SchoenfeldResiduals> {
        Ok(py.detach(|| self.schoenfeld_residuals(weighted))?)
    }

    /// `residuals(fit, type = "scaledsch", weighted)`.
    #[pyo3(name = "scaled_schoenfeld_residuals", signature = (weighted = false))]
    fn scaled_schoenfeld_residuals_py(
        &self,
        py: Python<'_>,
        weighted: bool,
    ) -> PyResult<SchoenfeldResiduals> {
        Ok(py.detach(|| self.scaled_schoenfeld_residuals(weighted))?)
    }

    /// `residuals(fit, type = "partial", weighted, collapse)`; `assign`
    /// lists the columns of each term (default: one term per column).
    #[pyo3(name = "partial_residuals", signature = (assign = None, weighted = false, collapse = None))]
    fn partial_residuals_py(
        &self,
        py: Python<'_>,
        assign: Option<Vec<Vec<usize>>>,
        weighted: bool,
        collapse: Option<IntVec>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let assign = assign.unwrap_or_else(|| default_assign(self.nvar()));
        let residuals =
            py.detach(|| self.partial_residuals(&assign, weighted, collapse.as_deref()))?;
        Ok(matrix_rows(&residuals))
    }

    /// `survfit(fit, newdata, id)` for time-dependent new data.
    #[pyo3(name = "survfit_individual", signature = (newdata, new_entry, new_time, id, new_strata = None, new_offset = None, stype = 2, ctype = None, se_fit = true, censor = true, start_time = None))]
    #[allow(clippy::too_many_arguments)]
    fn survfit_individual_py(
        &self,
        py: Python<'_>,
        newdata: FloatMatrix,
        new_entry: FloatVec,
        new_time: FloatVec,
        id: IntVec,
        new_strata: Option<IntVec>,
        new_offset: Option<FloatVec>,
        stype: u8,
        ctype: Option<u8>,
        se_fit: bool,
        censor: bool,
        start_time: Option<f64>,
    ) -> PyResult<Vec<CoxSurvfitCurve>> {
        let newdata = newdata_from_python(
            Some(newdata),
            new_strata,
            new_offset,
            Some(new_time),
            Some(new_entry),
        )?
        .expect("newdata was supplied");
        let options = SurvfitOptions {
            stype,
            ctype,
            se_fit,
            censor,
            start_time,
        };
        Ok(py.detach(|| self.survfit_individual(&newdata, &id, options))?)
    }
}

/// `coxph()` on explicit data: fits `Surv(time, status) ~ x` (or
/// `Surv(entry, time, status) ~ x` when `entry` is given).
///
/// `nocenter` lists the values of a column that exempt it from centring
/// (R's default `c(-1, 0, 1)` when omitted; an empty list centres every
/// column, R's `nocenter = NULL`).  `cluster` requests the robust variance.
#[pyfunction]
#[pyo3(signature = (time, status, x, entry=None, strata=None, weights=None, offset=None, method="efron", init=None, iter_max=None, eps=None, toler_chol=None, nocenter=None, cluster=None, robust=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_fit(
    py: Python<'_>,
    time: FloatVec,
    status: IntVec,
    x: FloatMatrix,
    entry: Option<FloatVec>,
    strata: Option<IntVec>,
    weights: Option<FloatVec>,
    offset: Option<FloatVec>,
    method: &str,
    init: Option<Vec<f64>>,
    iter_max: Option<usize>,
    eps: Option<f64>,
    toler_chol: Option<f64>,
    nocenter: Option<Vec<f64>>,
    cluster: Option<IntVec>,
    robust: Option<bool>,
) -> PyResult<CoxPHFit> {
    if x.nrow() != time.len() {
        return Err(SurvivalError::invalid_input(format!(
            "x has {} rows but time has {}",
            x.nrow(),
            time.len()
        ))
        .into());
    }
    let data = CoxphData {
        time: time.into_inner(),
        entry: entry.map(FloatVec::into_inner),
        status: status.into_inner(),
        x: x.into_inner(),
        weights: weights.map(FloatVec::into_inner),
        strata: strata.map(IntVec::into_inner),
        offset: offset.map(FloatVec::into_inner),
    };
    let defaults = CoxphOptions::default();
    let options = CoxphOptions {
        method: TieMethod::parse(Some(method))?,
        init,
        iter_max: iter_max.unwrap_or(defaults.iter_max),
        eps: eps.unwrap_or(defaults.eps),
        toler_chol: toler_chol.unwrap_or(defaults.toler_chol),
        nocenter: nocenter.or(defaults.nocenter),
        cluster: cluster.map(IntVec::into_inner),
        robust,
    };
    Ok(py.detach(move || CoxPHFit::fit(data, options))?)
}
