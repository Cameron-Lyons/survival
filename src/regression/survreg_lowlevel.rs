//! AFT fitting on an already transformed response, without full-model retention.

use super::parametric_survival::{SurvregControl, SurvregData, fit_survreg_engine};
use super::survreg_distributions::{SurvregDistribution, SurvregTransform};
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Bare `survreg.fit` components. Estimated log-scales remain in coefficients;
/// redundant coefficients retain the optimizer's values. Training data and
/// callback distributions are not retained, so the result is always serializable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "survival._survival", skip_from_py_object)]
pub struct SurvregFitResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub icoef: Vec<f64>,
    #[pyo3(get)]
    pub var: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub loglik: [f64; 2],
    #[pyo3(get)]
    pub iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub score: Vec<f64>,
    /// R drops variance dimension names when undoing covariate rescaling.
    #[pyo3(get)]
    pub rescaled: bool,
}

crate::internal::pickle::picklable!(SurvregFitResult);

/// The density on the fitting scale. Bare fitters ignore response transforms
/// and distribution-level fixed scales; the explicit scale argument controls fitting.
pub(crate) fn fitting_distribution(distribution: &SurvregDistribution) -> SurvregDistribution {
    let mut result = distribution.clone();
    result.transform = SurvregTransform::Identity;
    result.transform_callbacks = None;
    result.scale = None;
    result
}

impl SurvregFitResult {
    /// Fit prepared response data; `nstrata` can include unobserved scale strata.
    /// Cluster variance belongs to the full-model interface and is refused here.
    pub fn fit(
        data: &SurvregData,
        distribution: &SurvregDistribution,
        init: Option<&[f64]>,
        scale: f64,
        control: &SurvregControl,
        nstrata: Option<usize>,
    ) -> SurvivalResult<Self> {
        if data.cluster.is_some() {
            return Err(SurvivalError::invalid_input(
                "bare AFT fits do not compute robust variance",
            ));
        }
        let distribution = fitting_distribution(distribution);
        Ok(fit_survreg_engine(data, &distribution, init, scale, control, nstrata)?.fit)
    }
}

/// Bare AFT fit, preserving the response scale and dropping training data.
#[pyfunction]
#[pyo3(signature = (data, distribution, init=None, scale=0.0, control=None, nstrat=None))]
pub fn survreg_fit_raw(
    py: Python<'_>,
    data: &SurvregData,
    distribution: &SurvregDistribution,
    init: Option<Vec<f64>>,
    scale: f64,
    control: Option<SurvregControl>,
    nstrat: Option<usize>,
) -> PyResult<SurvregFitResult> {
    let control = control.unwrap_or_default();
    Ok(py.detach(|| {
        SurvregFitResult::fit(data, distribution, init.as_deref(), scale, &control, nstrat)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn bare_distribution_ignores_transform_and_fixed_scale() {
        let data = SurvregData::try_new(
            vec![-1.0, 0.5, 1.0, 2.0],
            vec![1; 4],
            Array2::ones((4, 1)),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let distribution = SurvregDistribution::from_name("exponential", None).unwrap();
        let base = SurvregDistribution::from_name("extreme", None).unwrap();
        let control = SurvregControl::default();
        let bare = SurvregFitResult::fit(&data, &distribution, None, 0.0, &control, None).unwrap();
        let expected = SurvregFitResult::fit(&data, &base, None, 0.0, &control, None).unwrap();
        assert_eq!(bare.coefficients, expected.coefficients);
        assert_eq!(bare.loglik, expected.loglik);
        assert_eq!(bare.df, 2);
    }
}
