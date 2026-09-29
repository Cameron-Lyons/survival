//! Compact penalized AFT results on a prepared response.

use super::{SurvpenalData, SurvpenalFit, SurvpenalOptions};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::matrix_rows;
use crate::regression::coxpenal::{CoxPenalty, ModelTerm, PenaltyHistory};
use crate::regression::parametric_survival::{SurvregControl, SurvregData};
use crate::regression::penalized::terms::model_terms;
use crate::regression::survreg_distributions::SurvregDistribution;
use crate::regression::survreg_lowlevel::fitting_distribution;
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Bare `survpenal.fit` components, with estimated log-scales in coefficients.
/// No response, design, per-observation frailty indices or callbacks are retained.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", skip_from_py_object)]
pub struct SurvpenalFitResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub icoef: Vec<f64>,
    pub var: Array2<f64>,
    pub var2: Array2<f64>,
    #[pyo3(get)]
    pub loglik: [f64; 2],
    #[pyo3(get)]
    pub iter: [usize; 2],
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub df: Vec<f64>,
    #[pyo3(get)]
    pub penalty: [f64; 2],
    #[pyo3(get)]
    pub pterms: Vec<u8>,
    #[pyo3(get)]
    pub assign2: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub history: Vec<PenaltyHistory>,
    #[pyo3(get)]
    pub frail: Option<Vec<f64>>,
    #[pyo3(get)]
    pub fvar: Option<Vec<f64>>,
    #[pyo3(get)]
    pub n_eff: f64,
    #[pyo3(get)]
    pub score: Vec<f64>,
    #[pyo3(get)]
    pub n: usize,
    /// Number of dense design columns, excluding estimated scales.
    #[pyo3(get)]
    pub nvar: usize,
    #[pyo3(get)]
    pub scale: Vec<f64>,
}

#[pymethods]
impl SurvpenalFitResult {
    #[cfg(feature = "python")]
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::internal::pickle::Reduced<'py>> {
        crate::internal::pickle::reduce(py, self)
    }

    #[getter(var)]
    fn var_py(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var)
    }

    #[getter(var2)]
    fn var2_py(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.var2)
    }
}

impl SurvpenalFitResult {
    /// Fit without response transforms, Jacobian corrections or robust variance.
    pub fn fit(
        data: &SurvpenalData,
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
        nstrata: Option<usize>,
    ) -> SurvivalResult<Self> {
        Self::fit_inputs(&data.survreg, &data.terms, distribution, options, nstrata)
    }

    fn fit_inputs(
        data: &SurvregData,
        terms: &[ModelTerm],
        distribution: &SurvregDistribution,
        options: &SurvpenalOptions,
        nstrata: Option<usize>,
    ) -> SurvivalResult<Self> {
        if options.robust || data.cluster.is_some() {
            return Err(SurvivalError::invalid_input(
                "bare AFT fits do not compute robust variance",
            ));
        }
        let distribution = fitting_distribution(distribution);
        Ok(SurvpenalFit::fit_engine(data, terms, &distribution, options, nstrata)?.fit)
    }
}

/// Bare penalized AFT fit; columns and term assignments are zero-based.
#[pyfunction]
#[pyo3(signature = (data, distribution, penalties, pcols, assign=None, init=None, scale=0.0, control=None, nstrat=None))]
#[allow(clippy::too_many_arguments)]
pub fn survpenal_fit_raw(
    py: Python<'_>,
    data: &SurvregData,
    distribution: &SurvregDistribution,
    penalties: Vec<CoxPenalty>,
    pcols: Vec<Vec<usize>>,
    assign: Option<Vec<Vec<usize>>>,
    init: Option<Vec<f64>>,
    scale: f64,
    control: Option<SurvregControl>,
    nstrat: Option<usize>,
) -> PyResult<SurvpenalFitResult> {
    let options = SurvpenalOptions {
        init,
        scale,
        control: control.unwrap_or_default(),
        robust: false,
    };
    Ok(py.detach(|| {
        let terms = model_terms(
            data.nvar(),
            penalties.into_iter().map(|p| p.term).collect(),
            pcols,
            assign,
        )?;
        SurvpenalFitResult::fit_inputs(data, &terms, distribution, &options, nstrat)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxpenal::{PenaltyTerm, RidgePenalty};

    #[test]
    fn compact_fit_agrees_with_full_solver_on_prepared_data() {
        let x = Array2::from_shape_fn((12, 2), |(i, j)| if j == 0 { 1.0 } else { (i % 3) as f64 });
        let data = SurvregData::try_new(
            vec![1.2, 2.5, 0.9, 3.0, 1.8, 2.7, 3.9, 1.1, 2.2, 3.6, 1.4, 2.9],
            vec![1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            x,
            None,
            None,
            Some(vec![0.2; 12]),
            None,
            None,
        )
        .unwrap();
        let terms = model_terms(
            2,
            vec![PenaltyTerm::Ridge(RidgePenalty {
                theta: Some(1.0),
                df: None,
                eps: 0.1,
                scale: true,
                scale_values: None,
            })],
            vec![vec![1]],
            None,
        )
        .unwrap();
        let data = SurvpenalData::try_new(data, terms).unwrap();
        let distribution = SurvregDistribution::from_name("gaussian", None).unwrap();
        let options = SurvpenalOptions::default();
        let raw = SurvpenalFitResult::fit(&data, &distribution, &options, None).unwrap();
        let full = SurvpenalFit::fit(&data, &distribution, &options).unwrap();
        assert_eq!(raw.coefficients, full.survreg.coefficients);
        assert_eq!(raw.linear_predictors, full.survreg.linear_predictors);
        assert_eq!(raw.df, full.df);
        assert_eq!(matrix_rows(&raw.var), full.survreg.variance_matrix);
        assert!(
            SurvpenalFitResult::fit(
                &data,
                &distribution,
                &SurvpenalOptions {
                    robust: true,
                    ..options
                },
                None
            )
            .is_err()
        );
    }
}
