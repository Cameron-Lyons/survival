#![allow(clippy::new_without_default)]
#![allow(clippy::unused_enumerate_index)]
use ndarray::Array2;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct Subject {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub covariates: Vec<f64>,
    #[pyo3(get, set)]
    pub is_case: bool,
    #[pyo3(get, set)]
    pub is_subcohort: bool,
    #[pyo3(get, set)]
    pub stratum: usize,
}

#[pymethods]
impl Subject {
    #[new]
    pub fn new(
        id: usize,
        covariates: Vec<f64>,
        is_case: bool,
        is_subcohort: bool,
        stratum: usize,
    ) -> Self {
        Self {
            id,
            covariates,
            is_case,
            is_subcohort,
            stratum,
        }
    }
}

#[pyclass]
pub struct CoxPHModel {
    coefficients: Array2<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub risk_scores: Vec<f64>,
    #[pyo3(get, set)]
    pub event_times: Vec<f64>,
    #[pyo3(get, set)]
    pub censoring: Vec<u8>,
    covariates: Array2<f64>,
}

#[pymethods]
impl CoxPHModel {
    #[new]
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

    #[pyo3(signature = (covariates, event_times, censoring))]
    #[staticmethod]
    pub fn new_with_data(
        covariates: Vec<Vec<f64>>,
        event_times: Vec<f64>,
        censoring: Vec<u8>,
    ) -> Self {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (i, row) in covariates.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                cov_array[[i, j]] = val;
            }
        }
        Self {
            coefficients: Array2::<f64>::zeros((ncols, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times,
            censoring,
            covariates: cov_array,
        }
    }

    pub fn add_subject(&mut self, _subject: &Subject) {}

    #[pyo3(signature = (n_iters = 10))]
    pub fn fit(&mut self, n_iters: u16) {
        self.risk_scores.clear();
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

    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (i, row) in covariates.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                cov_array[[i, j]] = val;
            }
        }
        let mut risk_scores = Vec::new();
        for (_i, row) in cov_array.outer_iter().enumerate() {
            let risk_score = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk_score);
        }
        risk_scores
    }

    #[getter]
    pub fn get_coefficients(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for col in self.coefficients.columns() {
            result.push(col.iter().cloned().collect());
        }
        result
    }

    pub fn brier_score(&self) -> f64 {
        let mut score = 0.0;
        let mut count = 0.0;
        for i in 0..self.event_times.len() {
            let time = self.event_times[i];
            let status = self.censoring[i] as f64;
            let pred = self.predict_survival(time);
            score += (pred - status).powi(2);
            count += 1.0;
        }
        if count > 0.0 { score / count } else { 0.0 }
    }

    fn predict_survival(&self, _time: f64) -> f64 {
        0.5
    }

    pub fn survival_curve(
        &self,
        covariates: Vec<Vec<f64>>,
        time_points: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (i, row) in covariates.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                cov_array[[i, j]] = val;
            }
        }

        let times = time_points.unwrap_or_else(|| {
            let mut t = self.event_times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t.dedup();
            t
        });

        let mut risk_scores = Vec::new();
        for row in cov_array.outer_iter() {
            let risk = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk.exp());
        }

        let mut cum_baseline_haz = Vec::new();
        let mut cum = 0.0;
        for &haz in &self.baseline_hazard {
            cum += haz;
            cum_baseline_haz.push(cum);
        }

        let mut survival_curves = Vec::new();
        for risk_exp in &risk_scores {
            let mut surv = Vec::new();
            for &t in &times {
                let baseline_haz = self
                    .baseline_hazard
                    .iter()
                    .zip(&self.event_times)
                    .filter(|&(_, et)| *et <= t)
                    .map(|(h, _)| *h)
                    .sum::<f64>();

                let s = (-baseline_haz * risk_exp).exp();
                surv.push(s);
            }
            survival_curves.push(surv);
        }

        Ok((times, survival_curves))
    }
}
