use std::collections::BTreeMap;

use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN};
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[inline]
fn exp_clamped(value: f64) -> f64 {
    value.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp()
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ClogitDataSet {
    case_control_status: Vec<u8>,
    strata: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}
impl Default for ClogitDataSet {
    fn default() -> Self {
        Self::new()
    }
}
#[pymethods]
impl ClogitDataSet {
    #[new]
    pub fn new() -> ClogitDataSet {
        ClogitDataSet {
            case_control_status: Vec::new(),
            strata: Vec::new(),
            covariates: Vec::new(),
        }
    }
    pub fn add_observation(
        &mut self,
        case_control_status: u8,
        stratum: u8,
        covariates: Vec<f64>,
    ) -> PyResult<()> {
        if case_control_status > 1 {
            return Err(value_error("case_control_status values must be 0 or 1"));
        }

        if let Some(expected_covariates) = self.covariates.first().map(Vec::len)
            && covariates.len() != expected_covariates
        {
            return Err(value_error(
                "all observations must have the same number of covariates",
            ));
        }

        if covariates.iter().any(|value| !value.is_finite()) {
            return Err(value_error("covariates must contain only finite values"));
        }

        self.case_control_status.push(case_control_status);
        self.strata.push(stratum);
        self.covariates.push(covariates);
        Ok(())
    }
    pub fn get_num_observations(&self) -> usize {
        self.case_control_status.len()
    }
    pub fn get_num_covariates(&self) -> usize {
        self.covariates.first().map_or(0, Vec::len)
    }
    pub fn __len__(&self) -> usize {
        self.get_num_observations()
    }
    pub fn is_empty(&self) -> bool {
        self.case_control_status.is_empty()
    }
}
impl ClogitDataSet {
    pub(crate) fn get_case_control_status(&self, id: usize) -> u8 {
        self.case_control_status[id]
    }

    pub(crate) fn get_stratum(&self, id: usize) -> u8 {
        self.strata[id]
    }

    pub(crate) fn get_covariates(&self, id: usize) -> &Vec<f64> {
        &self.covariates[id]
    }

    fn validate(&self) -> PyResult<BTreeMap<u8, Vec<usize>>> {
        if self.case_control_status.is_empty() {
            return Err(value_error("at least one observation is required"));
        }

        if self.case_control_status.len() != self.strata.len()
            || self.case_control_status.len() != self.covariates.len()
        {
            return Err(value_error(
                "case_control_status, strata, and covariates must have the same length",
            ));
        }

        let num_covariates = self.get_num_covariates();
        let mut strata_groups: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for observation in 0..self.get_num_observations() {
            let case_status = self.get_case_control_status(observation);
            if case_status > 1 {
                return Err(value_error("case_control_status values must be 0 or 1"));
            }

            let covariates = self.get_covariates(observation);
            if covariates.len() != num_covariates {
                return Err(value_error(
                    "all observations must have the same number of covariates",
                ));
            }
            if covariates.iter().any(|value| !value.is_finite()) {
                return Err(value_error("covariates must contain only finite values"));
            }

            strata_groups
                .entry(self.get_stratum(observation))
                .or_default()
                .push(observation);
        }

        for indices in strata_groups.values() {
            if indices.len() < 2 {
                return Err(value_error(
                    "each stratum must contain at least two observations",
                ));
            }

            let case_count = indices
                .iter()
                .map(|&idx| self.get_case_control_status(idx) as usize)
                .sum::<usize>();
            if case_count == 0 || case_count == indices.len() {
                return Err(value_error(
                    "each stratum must contain at least one case and one control",
                ));
            }
        }

        Ok(strata_groups)
    }
}
#[pyclass]
pub struct ConditionalLogisticRegression {
    data: ClogitDataSet,
    #[pyo3(get)]
    coefficients: Vec<f64>,
    #[pyo3(get, set)]
    max_iter: u32,
    #[pyo3(get, set)]
    tol: f64,
    #[pyo3(get)]
    iterations: u32,
    #[pyo3(get)]
    converged: bool,
}
#[pymethods]
impl ConditionalLogisticRegression {
    #[new]
    #[pyo3(signature = (data, max_iter=100, tol=crate::constants::CLOGIT_TOLERANCE))]
    pub fn new(data: ClogitDataSet, max_iter: u32, tol: f64) -> PyResult<Self> {
        validate_solver_controls(max_iter, tol)?;
        Ok(ConditionalLogisticRegression {
            data,
            coefficients: Vec::new(),
            max_iter,
            tol,
            iterations: 0,
            converged: false,
        })
    }
    pub fn fit(&mut self) -> PyResult<()> {
        validate_solver_controls(self.max_iter, self.tol)?;
        let num_covariates = self.data.get_num_covariates();
        let strata_groups = self.data.validate()?;
        if num_covariates == 0 {
            self.coefficients.clear();
            self.iterations = 0;
            self.converged = true;
            return Ok(());
        }

        let n_strata = strata_groups.len() as f64;

        self.coefficients = vec![0.0; num_covariates];
        self.iterations = 0;
        self.converged = false;

        // Use a stable gradient ascent step on the conditional softmax objective
        // within each stratum so matched-set membership actually affects the fit.
        let learning_rate = 0.1;
        while self.iterations < self.max_iter {
            let mut gradient = vec![0.0; num_covariates];

            for indices in strata_groups.values() {
                let mut linear_predictors = Vec::with_capacity(indices.len());
                let mut max_linear_predictor = f64::NEG_INFINITY;
                let mut case_count = 0.0;
                let mut case_sums = vec![0.0; num_covariates];

                for &observation in indices {
                    let covariates = self.data.get_covariates(observation);
                    let case_status = self.data.get_case_control_status(observation) as f64;
                    case_count += case_status;

                    for (sum, value) in case_sums.iter_mut().zip(covariates.iter()) {
                        *sum += case_status * value;
                    }

                    let linear_predictor: f64 = self
                        .coefficients
                        .iter()
                        .zip(covariates.iter())
                        .map(|(coef, cov)| coef * cov)
                        .sum();
                    max_linear_predictor = max_linear_predictor.max(linear_predictor);
                    linear_predictors.push(linear_predictor);
                }

                let weights: Vec<f64> = linear_predictors
                    .iter()
                    .map(|value| (value - max_linear_predictor).exp())
                    .collect();
                let total_weight = weights.iter().sum::<f64>();
                if total_weight <= crate::constants::DIVISION_FLOOR {
                    continue;
                }

                for covariate_idx in 0..num_covariates {
                    let expected = indices
                        .iter()
                        .zip(weights.iter())
                        .map(|(&observation, weight)| {
                            self.data.get_covariates(observation)[covariate_idx] * *weight
                        })
                        .sum::<f64>()
                        / total_weight;
                    gradient[covariate_idx] += case_sums[covariate_idx] - case_count * expected;
                }
            }

            let mut max_change = 0.0_f64;
            for (coefficient, gradient_component) in self.coefficients.iter_mut().zip(gradient) {
                let delta = (learning_rate * gradient_component / n_strata).clamp(-0.5, 0.5);
                *coefficient += delta;
                max_change = max_change.max(delta.abs());
            }

            self.iterations += 1;
            if max_change < self.tol {
                self.converged = true;
                break;
            }
        }

        Ok(())
    }
    pub fn predict(&self, covariates: Vec<f64>) -> PyResult<f64> {
        self.validate_prediction_row(&covariates)?;
        let exp_sum: f64 = self
            .coefficients
            .iter()
            .zip(covariates.iter())
            .map(|(coef, cov)| coef * cov)
            .sum();
        Ok(exp_clamped(exp_sum))
    }
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|coefficient| exp_clamped(*coefficient))
            .collect()
    }
}

impl ConditionalLogisticRegression {
    fn validate_prediction_row(&self, covariates: &[f64]) -> PyResult<()> {
        if self.coefficients.is_empty() && self.iterations == 0 && !self.converged {
            return Err(value_error("model must be fit before prediction"));
        }
        if covariates.len() != self.coefficients.len() {
            return Err(value_error(format!(
                "prediction row has {} covariates, expected {}",
                covariates.len(),
                self.coefficients.len()
            )));
        }
        if covariates.iter().any(|value| !value.is_finite()) {
            return Err(value_error("prediction covariates must be finite"));
        }
        Ok(())
    }
}

fn validate_solver_controls(max_iter: u32, tol: f64) -> PyResult<()> {
    if max_iter == 0 {
        return Err(value_error("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(value_error("tol must be a positive finite value"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matched_dataset() -> ClogitDataSet {
        let mut dataset = ClogitDataSet::new();
        for (case_status, stratum, covariates) in [
            (1, 0, vec![2.0]),
            (0, 0, vec![1.0]),
            (1, 1, vec![3.0]),
            (0, 1, vec![1.0]),
        ] {
            dataset
                .add_observation(case_status, stratum, covariates)
                .expect("valid matched observation");
        }
        dataset
    }

    #[test]
    fn dataset_rejects_invalid_rows() {
        let mut dataset = ClogitDataSet::new();

        assert!(dataset.add_observation(2, 0, vec![1.0]).is_err());
        assert!(dataset.add_observation(1, 0, vec![f64::NAN]).is_err());

        dataset.add_observation(1, 0, vec![1.0]).unwrap();
        assert!(dataset.add_observation(0, 0, vec![1.0, 2.0]).is_err());
        assert_eq!(dataset.get_num_observations(), 1);
    }

    #[test]
    fn solver_controls_are_validated() {
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 0, 1e-6).is_err());
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 10, 0.0).is_err());
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 10, f64::INFINITY).is_err());
    }

    #[test]
    fn fit_validates_data_before_null_model_short_circuit() {
        let mut dataset = ClogitDataSet::new();
        dataset.add_observation(1, 0, Vec::new()).unwrap();

        let mut model = ConditionalLogisticRegression::new(dataset, 10, 1e-6).unwrap();
        assert!(model.fit().is_err());
    }

    #[test]
    fn prediction_requires_fitted_model_and_matching_finite_row() {
        let mut model = ConditionalLogisticRegression::new(matched_dataset(), 10, 1e-9).unwrap();
        assert!(model.predict(vec![1.0]).is_err());

        model.fit().unwrap();

        assert!(model.predict(Vec::new()).is_err());
        assert!(model.predict(vec![f64::NAN]).is_err());
        assert!(model.predict(vec![1.0]).unwrap().is_finite());
    }

    #[test]
    fn exponentials_are_clamped_for_public_outputs() {
        let mut model = ConditionalLogisticRegression::new(matched_dataset(), 1, 1e-9).unwrap();
        model.coefficients = vec![1_000.0, -1_000.0];
        model.iterations = 1;

        let ratios = model.odds_ratios();

        assert_eq!(ratios.len(), 2);
        assert!(ratios.iter().all(|ratio| ratio.is_finite()));
        assert!(model.predict(vec![1.0, 1.0]).unwrap().is_finite());
    }
}
