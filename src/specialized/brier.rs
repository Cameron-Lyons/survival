use crate::regression::coxph::CoxPHModel;
use pyo3::prelude::*;

#[pymethods]
impl CoxPHModel {
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
        score / count
    }

    fn predict_survival(&self, _time: f64) -> f64 {
        0.5
    }
}
