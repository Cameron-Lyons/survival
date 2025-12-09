#![allow(clippy::new_without_default)]
#![allow(clippy::unused_enumerate_index)]
use ndarray::Array2;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct Subject {
    pub id: usize,
    pub covariates: Vec<f64>,
    pub is_case: bool,
    pub is_subcohort: bool,
    pub stratum: usize,
}

#[pyclass]
pub struct CoxPHModel {
    coefficients: Array2<f64>,
    baseline_hazard: Vec<f64>,
    risk_scores: Vec<f64>,
    #[pyo3(get, set)]
    pub event_times: Vec<f64>,
    #[pyo3(get, set)]
    pub censoring: Vec<u8>,
    covariates: Array2<f64>,
}

impl CoxPHModel {
#[allow(clippy::too_many_arguments)]
    pub fn new() -> Self {
        Self {
            coefficients: Array2::<f64>::zeros((1, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times: Vec::new(),
            censoring: Vec::new(),
            covariates: Array2::<f64>::zeros((1, 1)),
        }
    }

    pub fn new_with_data(
        covariates: Array2<f64>,
        event_times: Vec<f64>,
        censoring: Vec<u8>,
    ) -> Self {
        Self {
            coefficients: Array2::<f64>::zeros((covariates.ncols(), 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times,
            censoring,
            covariates,
        }
    }

    pub fn add_subject(&mut self, _subject: &Subject) {}
    pub fn fit(&mut self, n_iters: u16) {
        for (_i, row) in self.covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.column(0).dot(&row);
            self.risk_scores.push(risk_score);
        }
        for _ in 0..n_iters {
            self.update_baseline_hazard();
            self.update_coefficients();
        }
        for _t in &self.event_times {
            let hazard_at_t = 0.0;
            self.baseline_hazard.push(hazard_at_t);
        }
    }
    fn update_baseline_hazard(&mut self) {
        let mut baseline_hazard = Vec::new();
        for _t in &self.event_times {
            let hazard_at_t = 0.0;
            baseline_hazard.push(hazard_at_t);
        }
        self.baseline_hazard = baseline_hazard;
    }
    fn update_coefficients(&mut self) {
        let coefficients = Array2::<f64>::zeros((self.covariates.ncols(), 1));
        for (_i, row) in self.covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.column(0).dot(&row);
            self.risk_scores.push(risk_score);
        }
        self.coefficients = coefficients;
    }
    pub fn predict(&self, covariates: Array2<f64>) -> Vec<f64> {
        let mut risk_scores = Vec::new();
        for (_i, row) in covariates.outer_iter().enumerate() {
            let risk_score = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk_score);
        }
        risk_scores
    }
}

#[pymodule]
#[pyo3(name = "pyCoxPHModel")]
fn py_cox_ph_model(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CoxPHModel>()?;
    Ok(())
}
