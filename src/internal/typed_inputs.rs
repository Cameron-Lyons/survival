use crate::internal::validation::{
    validate_length, validate_no_nan, validate_non_empty, validate_non_negative,
};
use crate::{SurvivalError, SurvivalResult};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_finite(slice: &[f64], field: &'static str) -> SurvivalResult<()> {
    for (index, value) in slice.iter().enumerate() {
        if !value.is_finite() {
            return Err(SurvivalError::invalid_input(format!(
                "{} contains non-finite value {} at index {}",
                field, value, index
            )));
        }
    }
    Ok(())
}

fn validate_status_values(status: &[i32], field: &'static str) -> SurvivalResult<()> {
    for (index, value) in status.iter().enumerate() {
        if *value < 0 {
            return Err(SurvivalError::invalid_input(format!(
                "{} contains negative status {} at index {}",
                field, value, index
            )));
        }
    }
    Ok(())
}

fn validate_strata_len(strata: &[i32], expected: usize) -> SurvivalResult<()> {
    validate_length(expected, strata.len(), "strata")?;
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvivalData {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
}

#[pymethods]
impl SurvivalData {
    #[new]
    #[pyo3(signature = (time, status))]
    pub fn new(time: Vec<f64>, status: Vec<i32>) -> PyResult<Self> {
        Ok(Self::try_new(time, status)?)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

impl SurvivalData {
    pub fn try_new(time: Vec<f64>, status: Vec<i32>) -> SurvivalResult<Self> {
        Self::validate_parts(&time, &status)?;
        Ok(Self { time, status })
    }

    pub(crate) fn validate_parts(time: &[f64], status: &[i32]) -> SurvivalResult<()> {
        validate_non_empty(time, "time")?;
        validate_length(time.len(), status.len(), "status")?;
        validate_no_nan(time, "time")?;
        validate_finite(time, "time")?;
        validate_non_negative(time, "time")?;
        validate_status_values(status, "status")?;
        Ok(())
    }

    pub(crate) fn len(&self) -> usize {
        self.time.len()
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CovariateMatrix {
    #[pyo3(get)]
    pub values: Vec<f64>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pymethods]
impl CovariateMatrix {
    #[new]
    #[pyo3(signature = (values, n_obs, n_vars))]
    pub fn new(values: Vec<f64>, n_obs: usize, n_vars: usize) -> PyResult<Self> {
        Ok(Self::try_new(values, n_obs, n_vars)?)
    }

    pub fn __len__(&self) -> usize {
        self.n_obs
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }
}

impl CovariateMatrix {
    pub fn try_new(values: Vec<f64>, n_obs: usize, n_vars: usize) -> SurvivalResult<Self> {
        Self::validate_parts(&values, n_obs, n_vars)?;

        Ok(Self {
            values,
            n_obs,
            n_vars,
        })
    }

    pub(crate) fn validate_parts(
        values: &[f64],
        n_obs: usize,
        n_vars: usize,
    ) -> SurvivalResult<()> {
        if n_obs == 0 {
            return Err(SurvivalError::invalid_input("n_obs must be positive"));
        }
        if n_vars == 0 {
            return Err(SurvivalError::invalid_input("n_vars must be positive"));
        }

        let expected = n_obs
            .checked_mul(n_vars)
            .ok_or_else(|| SurvivalError::invalid_input("n_obs * n_vars overflows usize"))?;
        validate_length(expected, values.len(), "values")?;
        validate_no_nan(values, "values")?;
        validate_finite(values, "values")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct Weights {
    #[pyo3(get)]
    pub values: Vec<f64>,
}

#[pymethods]
impl Weights {
    #[new]
    #[pyo3(signature = (values))]
    pub fn new(values: Vec<f64>) -> PyResult<Self> {
        Ok(Self::try_new(values)?)
    }

    #[staticmethod]
    pub fn unit(n_obs: usize) -> PyResult<Self> {
        if n_obs == 0 {
            return Err(PyValueError::new_err("n_obs must be positive"));
        }
        Ok(Self {
            values: vec![1.0; n_obs],
        })
    }

    pub fn __len__(&self) -> usize {
        self.values.len()
    }
}

impl Weights {
    pub fn try_new(values: Vec<f64>) -> SurvivalResult<Self> {
        Self::validate_values(&values)?;
        Ok(Self { values })
    }

    pub(crate) fn validate_values(values: &[f64]) -> SurvivalResult<()> {
        validate_non_empty(values, "weights")?;
        validate_no_nan(values, "weights")?;
        validate_finite(values, "weights")?;
        validate_non_negative(values, "weights")?;
        Ok(())
    }

    pub(crate) fn validate_len(&self, expected: usize) -> SurvivalResult<()> {
        validate_length(expected, self.values.len(), "weights")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CountingProcessData {
    #[pyo3(get)]
    pub start: Vec<f64>,
    #[pyo3(get)]
    pub stop: Vec<f64>,
    #[pyo3(get)]
    pub event: Vec<i32>,
}

#[pymethods]
impl CountingProcessData {
    #[new]
    #[pyo3(signature = (start, stop, event))]
    pub fn new(start: Vec<f64>, stop: Vec<f64>, event: Vec<i32>) -> PyResult<Self> {
        Ok(Self::try_new(start, stop, event)?)
    }

    pub fn __len__(&self) -> usize {
        self.len()
    }
}

impl CountingProcessData {
    pub fn try_new(start: Vec<f64>, stop: Vec<f64>, event: Vec<i32>) -> SurvivalResult<Self> {
        validate_non_empty(&start, "start")?;
        validate_length(start.len(), stop.len(), "stop")?;
        validate_length(start.len(), event.len(), "event")?;
        validate_no_nan(&start, "start")?;
        validate_no_nan(&stop, "stop")?;
        validate_finite(&start, "start")?;
        validate_finite(&stop, "stop")?;
        validate_non_negative(&start, "start")?;
        validate_non_negative(&stop, "stop")?;
        validate_status_values(&event, "event")?;

        for (index, (&start, &stop)) in start.iter().zip(stop.iter()).enumerate() {
            if stop < start {
                return Err(SurvivalError::invalid_input(format!(
                    "stop {} is before start {} at index {}",
                    stop, start, index
                )));
            }
        }

        Ok(Self { start, stop, event })
    }

    pub(crate) fn len(&self) -> usize {
        self.start.len()
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxRegressionInput {
    pub covariates: CovariateMatrix,
    pub survival: SurvivalData,
    pub weights: Option<Weights>,
    pub offset: Option<Vec<f64>>,
}

#[pymethods]
impl CoxRegressionInput {
    #[new]
    #[pyo3(signature = (covariates, survival, weights=None, offset=None))]
    pub fn new(
        covariates: &CovariateMatrix,
        survival: &SurvivalData,
        weights: Option<&Weights>,
        offset: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::try_new(
            covariates.clone(),
            survival.clone(),
            weights.cloned(),
            offset,
        )
    }

    #[getter]
    pub fn n_obs(&self) -> usize {
        self.covariates.n_obs
    }

    #[getter]
    pub fn n_vars(&self) -> usize {
        self.covariates.n_vars
    }
}

impl CoxRegressionInput {
    pub fn try_new(
        covariates: CovariateMatrix,
        survival: SurvivalData,
        weights: Option<Weights>,
        offset: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        Self::validate_parts(&covariates, &survival, weights.as_ref(), offset.as_deref())?;

        Ok(Self {
            covariates,
            survival,
            weights,
            offset,
        })
    }

    pub(crate) fn validate_slices(
        x: &[f64],
        n_obs: usize,
        n_vars: usize,
        time: &[f64],
        status: &[i32],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
    ) -> SurvivalResult<()> {
        CovariateMatrix::validate_parts(x, n_obs, n_vars)?;
        SurvivalData::validate_parts(time, status)?;
        validate_length(n_obs, time.len(), "survival")?;

        if let Some(weights) = weights {
            Weights::validate_values(weights)?;
            validate_length(n_obs, weights.len(), "weights")?;
        }

        if let Some(offset) = offset {
            validate_length(n_obs, offset.len(), "offset")?;
            validate_no_nan(offset, "offset")?;
            validate_finite(offset, "offset")?;
        }

        Ok(())
    }

    pub(crate) fn validate_parts(
        covariates: &CovariateMatrix,
        survival: &SurvivalData,
        weights: Option<&Weights>,
        offset: Option<&[f64]>,
    ) -> SurvivalResult<()> {
        validate_length(covariates.n_obs, survival.len(), "survival")?;

        if let Some(weights) = weights {
            weights.validate_len(covariates.n_obs)?;
        }

        if let Some(offset) = offset {
            validate_length(covariates.n_obs, offset.len(), "offset")?;
            validate_no_nan(offset, "offset")?;
            validate_finite(offset, "offset")?;
        }

        Ok(())
    }

    pub(crate) fn weights_or_unit(&self) -> Vec<f64> {
        self.weights
            .as_ref()
            .map(|weights| weights.values.clone())
            .unwrap_or_else(|| vec![1.0; self.covariates.n_obs])
    }

    pub(crate) fn offset_or_zero(&self) -> Vec<f64> {
        self.offset
            .clone()
            .unwrap_or_else(|| vec![0.0; self.covariates.n_obs])
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoxMartInput {
    pub survival: SurvivalData,
    pub score: Vec<f64>,
    pub weights: Option<Weights>,
    pub strata: Option<Vec<i32>>,
}

#[pymethods]
impl CoxMartInput {
    #[new]
    #[pyo3(signature = (survival, score, weights=None, strata=None))]
    pub fn new(
        survival: &SurvivalData,
        score: Vec<f64>,
        weights: Option<&Weights>,
        strata: Option<Vec<i32>>,
    ) -> SurvivalResult<Self> {
        Self::try_new(survival.clone(), score, weights.cloned(), strata)
    }

    #[getter]
    pub fn n_obs(&self) -> usize {
        self.survival.len()
    }
}

impl CoxMartInput {
    pub fn try_new(
        survival: SurvivalData,
        score: Vec<f64>,
        weights: Option<Weights>,
        strata: Option<Vec<i32>>,
    ) -> SurvivalResult<Self> {
        let n_obs = survival.len();
        validate_length(n_obs, score.len(), "score")?;
        validate_no_nan(&score, "score")?;
        validate_finite(&score, "score")?;

        if let Some(weights) = &weights {
            weights.validate_len(n_obs)?;
        }
        if let Some(strata) = &strata {
            validate_strata_len(strata, n_obs)?;
        }

        Ok(Self {
            survival,
            score,
            weights,
            strata,
        })
    }

    pub(crate) fn weights_or_unit(&self) -> Vec<f64> {
        self.weights
            .as_ref()
            .map(|weights| weights.values.clone())
            .unwrap_or_else(|| vec![1.0; self.survival.len()])
    }

    pub(crate) fn strata_or_default(&self) -> Vec<i32> {
        self.strata
            .clone()
            .unwrap_or_else(|| vec![0; self.survival.len()])
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AndersenGillInput {
    pub counting: CountingProcessData,
    pub score: Vec<f64>,
    pub weights: Option<Weights>,
    pub strata: Option<Vec<i32>>,
}

#[pymethods]
impl AndersenGillInput {
    #[new]
    #[pyo3(signature = (counting, score, weights=None, strata=None))]
    pub fn new(
        counting: &CountingProcessData,
        score: Vec<f64>,
        weights: Option<&Weights>,
        strata: Option<Vec<i32>>,
    ) -> SurvivalResult<Self> {
        Self::try_new(counting.clone(), score, weights.cloned(), strata)
    }

    #[getter]
    pub fn n_obs(&self) -> usize {
        self.counting.len()
    }
}

impl AndersenGillInput {
    pub fn try_new(
        counting: CountingProcessData,
        score: Vec<f64>,
        weights: Option<Weights>,
        strata: Option<Vec<i32>>,
    ) -> SurvivalResult<Self> {
        let n_obs = counting.len();
        validate_length(n_obs, score.len(), "score")?;
        validate_no_nan(&score, "score")?;
        validate_finite(&score, "score")?;

        if let Some(weights) = &weights {
            weights.validate_len(n_obs)?;
        }
        if let Some(strata) = &strata {
            validate_strata_len(strata, n_obs)?;
        }

        Ok(Self {
            counting,
            score,
            weights,
            strata,
        })
    }

    pub(crate) fn weights_or_unit(&self) -> Vec<f64> {
        self.weights
            .as_ref()
            .map(|weights| weights.values.clone())
            .unwrap_or_else(|| vec![1.0; self.counting.len()])
    }

    pub(crate) fn strata_or_default(&self) -> Vec<i32> {
        self.strata
            .clone()
            .unwrap_or_else(|| vec![0; self.counting.len()])
    }
}
