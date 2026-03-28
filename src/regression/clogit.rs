use std::collections::BTreeMap;

use pyo3::prelude::*;
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
    pub fn add_observation(&mut self, case_control_status: u8, stratum: u8, covariates: Vec<f64>) {
        self.case_control_status.push(case_control_status);
        self.strata.push(stratum);
        self.covariates.push(covariates);
    }
    pub fn get_num_observations(&self) -> usize {
        self.case_control_status.len()
    }
    pub fn get_num_covariates(&self) -> usize {
        if self.covariates.is_empty() {
            0
        } else {
            self.covariates[0].len()
        }
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
        if self.case_control_status.len() != self.strata.len()
            || self.case_control_status.len() != self.covariates.len()
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "case_control_status, strata, and covariates must have the same length",
            ));
        }

        let num_covariates = self.get_num_covariates();
        let mut strata_groups: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for observation in 0..self.get_num_observations() {
            let case_status = self.get_case_control_status(observation);
            if case_status > 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "case_control_status values must be 0 or 1",
                ));
            }

            if self.get_covariates(observation).len() != num_covariates {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "all observations must have the same number of covariates",
                ));
            }

            strata_groups
                .entry(self.get_stratum(observation))
                .or_default()
                .push(observation);
        }

        for indices in strata_groups.values() {
            if indices.len() < 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each stratum must contain at least two observations",
                ));
            }

            let case_count = indices
                .iter()
                .map(|&idx| self.get_case_control_status(idx) as usize)
                .sum::<usize>();
            if case_count == 0 || case_count == indices.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
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
    pub fn new(data: ClogitDataSet, max_iter: u32, tol: f64) -> ConditionalLogisticRegression {
        ConditionalLogisticRegression {
            data,
            coefficients: Vec::new(),
            max_iter,
            tol,
            iterations: 0,
            converged: false,
        }
    }
    pub fn fit(&mut self) -> PyResult<()> {
        let num_covariates = self.data.get_num_covariates();
        if num_covariates == 0 {
            self.coefficients.clear();
            self.iterations = 0;
            self.converged = true;
            return Ok(());
        }

        let strata_groups = self.data.validate()?;
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
    pub fn predict(&self, covariates: Vec<f64>) -> f64 {
        let exp_sum: f64 = self
            .coefficients
            .iter()
            .zip(covariates.iter())
            .map(|(coef, cov)| coef * cov)
            .sum();
        exp_sum.exp()
    }
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|c| c.exp()).collect()
    }
}
