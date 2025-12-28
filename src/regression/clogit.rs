use pyo3::prelude::*;
#[pyclass]
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
    pub(crate) fn get_covariates(&self, id: usize) -> &Vec<f64> {
        &self.covariates[id]
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
    #[pyo3(signature = (data, max_iter=100, tol=1e-6))]
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
    pub fn fit(&mut self) {
        let num_covariates = self.data.get_num_covariates();
        if num_covariates == 0 {
            return;
        }
        self.coefficients = vec![0.0; num_covariates];
        let mut old_coefficients = vec![0.0; num_covariates];
        self.iterations = 0;
        self.converged = false;
        while self.iterations < self.max_iter {
            for covariate_idx in 0..num_covariates {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for observation in 0..self.data.get_num_observations() {
                    let case_control_status = self.data.get_case_control_status(observation);
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
                if denominator.abs() > 1e-10 {
                    self.coefficients[covariate_idx] += numerator / denominator;
                }
            }
            let diff: f64 = self
                .coefficients
                .iter()
                .zip(old_coefficients.iter())
                .map(|(coef, old_coef)| (coef - old_coef).abs())
                .sum();
            self.iterations += 1;
            if diff < self.tol {
                self.converged = true;
                break;
            }
        }
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
