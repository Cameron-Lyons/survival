#![allow(dead_code)]

use pyo3::prelude::*;

#[pyclass]
struct DataSet {
    case_control_status: Vec<u8>,
    strata: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}

impl DataSet {
    pub fn new() -> DataSet {
        DataSet {
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
    pub fn get_case_control_status(&self, id: usize) -> u8 {
        self.case_control_status[id]
    }
    pub fn get_stratum(&self, id: usize) -> u8 {
        self.strata[id]
    }
    pub fn get_covariates(&self, id: usize) -> &Vec<f64> {
        &self.covariates[id]
    }
    pub fn get_num_observations(&self) -> usize {
        self.case_control_status.len()
    }
    pub fn get_num_covariates(&self) -> usize {
        self.covariates[0].len()
    }
}

#[pyclass]
struct ConditionalLogisticRegression {
    data: DataSet,
    coefficients: Vec<f64>,
    max_iter: u32,
    tol: f64,
}

impl ConditionalLogisticRegression {
    pub fn new(data: DataSet) -> ConditionalLogisticRegression {
        ConditionalLogisticRegression {
            data,
            coefficients: Vec::new(),
            max_iter: 100,
            tol: 1e-6,
        }
    }
    pub fn set_max_iter(&mut self, max_iter: u32) {
        self.max_iter = max_iter;
    }
    pub fn set_tol(&mut self, tol: f64) {
        self.tol = tol;
    }
    pub fn fit(&mut self) {
        let num_covariates = self.data.get_num_covariates();
        self.coefficients = vec![0.0; num_covariates];
        let mut old_coefficients = vec![0.0; num_covariates];
        let mut iter = 0;
        while iter < self.max_iter {
            for covariate_idx in 0..num_covariates {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for observation in 0..self.data.get_num_observations() {
                    let case_control_status = self.data.get_case_control_status(observation);
                    let _stratum = self.data.get_stratum(observation);
                    let covariates = self.data.get_covariates(observation);
                    let exp_sum: f64 = self
                        .coefficients
                        .iter()
                        .zip(covariates.iter())
                        .map(|(coef, cov)| coef * cov)
                        .sum();
                    let exp = exp_sum.exp();
                    numerator += case_control_status as f64 * covariates[covariate_idx] * exp;
                    denominator += covariates[covariate_idx] * exp;
                }
                old_coefficients[covariate_idx] = self.coefficients[covariate_idx];
                self.coefficients[covariate_idx] += numerator / denominator;
            }
            let mut diff = 0.0;
            for (coef, old_coef) in self.coefficients.iter().zip(old_coefficients.iter()) {
                diff += (coef - old_coef).abs();
            }
            if diff < self.tol {
                break;
            }
            iter += 1;
        }
    }
    pub fn get_coefficients(&self) -> &Vec<f64> {
        &self.coefficients
    }
    pub fn predict(&self, covariates: &[f64]) -> f64 {
        let exp_sum: f64 = self
            .coefficients
            .iter()
            .zip(covariates.iter())
            .map(|(coef, cov)| coef * cov)
            .sum();
        exp_sum.exp()
    }
}

#[pymodule]
fn my_python_module(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConditionalLogisticRegression>()?;
    Ok(())
}
