use coxph::CoxPHModel;
use pyo3::prelude::*;

#[pymethods]
impl CoxPHModel {
    pub fn brier_score(&self) -> f64 {
        let mut score = 0.0;
        let mut count = 0.0;
        for (time, status) in self.data.iter() {
            let pred = self.predict(time);
            score += (pred - status).powi(2);
            count += 1.0;
        }
        score / count
    }
}
