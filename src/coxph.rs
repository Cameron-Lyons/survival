//Fit Proportional Hazards Regression Model

use ndarray::Array2;
use pyo3::prelude::*;


#[pyclass]
struct CoxPHModel {
    coefficients: Array2<f64>,
    baseline_hazard: Vec<f64>,
    risk_scores: Vec<f64>,
    event_times: Vec<f64>,
    censoring: Vec<u8>,
    covariates: Array2<f64>,
}

impl CoxPHModel {
    pub fn new(covariates: Array2<f64>, event_times: Vec<f64>, censoring: Vec<u8>) -> Self {
        Self {
            coefficients: Array2::<f64>::zeros((covariates.ncols(), 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times,
            censoring,
            covariates,
        }
    }
    pub fn fit(&mut self, n_iters: u16{
        for (i, row) in self.covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.dot(&row);
            self.risk_scores.push(risk_score);
        }
        for _ in 0..n_iters {
            self.update_baseline_hazard();
            self.update_coefficients();
        }
        for t in &self.event_times {
            let hazard_at_t = 0.0;
            self.baseline_hazard.push(hazard_at_t);
        }
    }
    fn update_baseline_hazard(&mut self) {
        let mut baseline_hazard = Vec::new();
        for t in &self.event_times {
            let hazard_at_t = 0.0;
            baseline_hazard.push(hazard_at_t);
        }
        self.baseline_hazard = baseline_hazard;
    }
    fn update_coefficients(&mut self) {
        let mut coefficients = Array2::<f64>::zeros((self.covariates.ncols(), 1));
        for (i, row) in self.covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.dot(&row);
            self.risk_scores.push(risk_score);
        }
        self.coefficients = coefficients;
    }
    pub fn predict(&self, covariates: Array2<f64>) -> Vec<f64> {
        let mut risk_scores = Vec::new();
        for (i, row) in covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.dot(&row);
            risk_scores.push(risk_score);
        }
        risk_scores
    }
}

#[pymodule]
fn pyCoxPHModel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CoxPHModel>()?;
    Ok(())
}
