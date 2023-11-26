// Compute the Brier score for a Cox model

use coxph::CoxPHModel;

impl CoxPHModel {
    /// Compute the Brier score for the model
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
