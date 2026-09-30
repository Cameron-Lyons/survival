//! Bare Cox fitters: preserve offsets and return the optimizer's results without
//! full-model concordance, robust variance, baseline caches or retained input data.

use super::TieMethod;
use super::coxph::{CoxphData, CoxphOptions, fit_cox_engine};
use super::coxph_diagnostics::risk_scores;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::matrix_rows;
use crate::internal::numpy_utils::{FloatMatrix, FloatVec, IntVec};
use crate::residuals::{agmart::agmart_rows, coxmart::coxmart_rows};
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Numerical result of a bare Cox fit. Null models retain empty coefficient
/// arrays; their likelihood still has initial and final entries. Residuals are
/// absent when they were not requested. Input rows and matrices are not retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", skip_from_py_object)]
pub struct CoxphFitResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    pub var: Array2<f64>,
    #[pyo3(get)]
    pub loglik: [f64; 2],
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub iter: usize,
    #[pyo3(get)]
    pub flag: i32,
    #[pyo3(get)]
    pub info: Option<[i32; 4]>,
    #[pyo3(get)]
    pub first: Vec<f64>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub residuals: Option<Vec<f64>>,
    #[pyo3(get)]
    pub method: TieMethod,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl CoxphFitResult {
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    #[getter(var)]
    fn var_py(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var)
    }
}

impl CoxphFitResult {
    /// Fit at the original offsets. `options.cluster` and robust variance belong
    /// to `CoxPHFit` and are refused here. `resid=false` skips martingale work.
    pub fn fit(data: CoxphData, mut options: CoxphOptions, resid: bool) -> SurvivalResult<Self> {
        data.validate()?;
        if options.cluster.is_some() || options.robust == Some(true) {
            return Err(SurvivalError::invalid_input(
                "bare Cox fits do not compute robust variance",
            ));
        }
        let n = data.n();
        let nvar = data.x.ncols();
        let exact = options.method == TieMethod::Exact;
        if data.entry.is_some() && !exact && !data.status.contains(&1) {
            return Err(SurvivalError::invalid_input(
                "Can't fit a Cox model with 0 failures",
            ));
        }
        if nvar == 0 && !exact {
            options.iter_max = 0;
            if data.entry.is_none() {
                options.init = None;
            }
        }
        let (result, _) = fit_cox_engine(
            &data,
            &options,
            data.offset.as_deref(),
            exact && data.entry.is_some(),
        )?;
        let mut coefficients = result.coefficients;
        let center: f64 = coefficients
            .iter()
            .zip(&result.means)
            .map(|(b, m)| b * m)
            .sum();
        let linear_predictors: Vec<f64> = data
            .x
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .zip(&coefficients)
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    + data.offset.as_ref().map_or(0.0, |values| values[i])
                    - center
            })
            .collect();
        let residuals = if resid {
            let risk = if !exact && (nvar > 0 || data.entry.is_some()) {
                risk_scores(&linear_predictors)
            } else {
                linear_predictors.iter().map(|value| value.exp()).collect()
            };
            let weights = data.weights.unwrap_or_else(|| vec![1.0; n]);
            let strata = data.strata.unwrap_or_else(|| vec![0; n]);
            Some(match data.entry {
                Some(entry) => agmart_rows(
                    &entry,
                    &data.time,
                    &data.status,
                    &risk,
                    &weights,
                    &strata,
                    options.method,
                ),
                None => coxmart_rows(
                    &result.order,
                    &data.time,
                    &data.status,
                    &risk,
                    &weights,
                    &strata,
                    options.method,
                ),
            })
        } else {
            None
        };
        let rank = result.info.map_or(result.flag, |info| info[0]);
        if (exact || options.iter_max > 0) && i64::from(rank) < nvar as i64 {
            for (j, coefficient) in coefficients.iter_mut().enumerate() {
                if result.var[(j, j)] == 0.0 {
                    *coefficient = f64::NAN;
                }
            }
        }
        Ok(Self {
            coefficients,
            var: result.var,
            loglik: result.loglik,
            score: result.sctest,
            iter: result.iter,
            flag: result.flag,
            info: result.info,
            first: result.score,
            means: result.means,
            linear_predictors,
            residuals,
            method: options.method,
            n,
        })
    }
}

/// Bare Cox fitting on explicit arrays, without full-model post-processing.
#[pyfunction]
#[pyo3(signature = (time, status, x, entry=None, strata=None, weights=None, offset=None, method="efron", init=None, iter_max=None, eps=None, toler_chol=None, nocenter=None, resid=true))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_fit_raw(
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
    resid: bool,
) -> PyResult<CoxphFitResult> {
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
        nocenter,
        ..defaults
    };
    Ok(py.detach(move || CoxphFitResult::fit(data, options, resid))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(counting: bool, columns: usize) -> CoxphData {
        CoxphData::try_new(
            vec![1.0, 2.0, 3.0, 4.0],
            counting.then(|| vec![0.0; 4]),
            vec![1, 1, 0, 1],
            Array2::from_shape_fn((4, columns), |(i, _)| (i % 2) as f64),
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn optional_residuals_and_null_likelihood() {
        let options = CoxphOptions {
            nocenter: None,
            ..Default::default()
        };
        let full = CoxphFitResult::fit(data(false, 0), options.clone(), true).unwrap();
        let lean = CoxphFitResult::fit(data(false, 0), options, false).unwrap();
        assert_eq!(full.loglik, lean.loglik);
        assert!((full.loglik[0] + 12.0_f64.ln()).abs() < 1e-12);
        for (actual, expected) in
            full.residuals
                .unwrap()
                .iter()
                .zip([0.75, 5.0 / 12.0, -7.0 / 12.0, -7.0 / 12.0])
        {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(lean.residuals.is_none());
        assert!(lean.coefficients.is_empty());
    }

    #[test]
    fn zero_events_and_exact_empty_design_follow_bare_fitters() {
        let mut censored = data(false, 1);
        censored.status.fill(0);
        let fitted = CoxphFitResult::fit(censored, CoxphOptions::default(), false).unwrap();
        assert_eq!(fitted.coefficients, vec![0.0]);
        assert_eq!(fitted.flag, 1000);
        assert_eq!(fitted.iter, 21);
        let mut censored = data(true, 1);
        censored.status.fill(0);
        assert!(CoxphFitResult::fit(censored, CoxphOptions::default(), false).is_err());
        let options = CoxphOptions {
            method: TieMethod::Exact,
            nocenter: None,
            ..Default::default()
        };
        let fitted = CoxphFitResult::fit(data(true, 0), options, false).unwrap();
        assert_eq!(fitted.iter, 1);
        assert!(fitted.coefficients.is_empty());
    }
}
