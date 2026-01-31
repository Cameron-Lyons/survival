#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct StreamingCoxConfig {
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub regularization: f64,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub window_size: usize,
    #[pyo3(get, set)]
    pub decay_factor: f64,
    #[pyo3(get, set)]
    pub min_events_to_update: usize,
}

#[pymethods]
impl StreamingCoxConfig {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, regularization=0.0, batch_size=32, window_size=1000, decay_factor=0.99, min_events_to_update=10))]
    pub fn new(
        learning_rate: f64,
        regularization: f64,
        batch_size: usize,
        window_size: usize,
        decay_factor: f64,
        min_events_to_update: usize,
    ) -> Self {
        Self {
            learning_rate,
            regularization,
            batch_size,
            window_size,
            decay_factor,
            min_events_to_update,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct StreamingCoxModel {
    coefficients: Vec<f64>,
    n_features: usize,
    n_samples_seen: usize,
    n_events_seen: usize,
    config: StreamingCoxConfig,
    gradient_accumulator: Vec<f64>,
    hessian_accumulator: Vec<f64>,
    baseline_hazard: Vec<(f64, f64)>,
}

#[pymethods]
impl StreamingCoxModel {
    #[new]
    pub fn new(n_features: usize, config: Option<StreamingCoxConfig>) -> Self {
        let config =
            config.unwrap_or_else(|| StreamingCoxConfig::new(0.01, 0.0, 32, 1000, 0.99, 10));

        Self {
            coefficients: vec![0.0; n_features],
            n_features,
            n_samples_seen: 0,
            n_events_seen: 0,
            config,
            gradient_accumulator: vec![0.0; n_features],
            hessian_accumulator: vec![0.0; n_features],
            baseline_hazard: Vec::new(),
        }
    }

    fn partial_fit(
        &mut self,
        covariates: Vec<Vec<f64>>,
        time: Vec<f64>,
        event: Vec<i32>,
    ) -> PyResult<()> {
        let n = covariates.len();
        if n == 0 || time.len() != n || event.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid input dimensions",
            ));
        }

        if covariates[0].len() != self.n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Number of features doesn't match model",
            ));
        }

        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap());

        let mut gradient = vec![0.0; self.n_features];
        let mut hessian = vec![0.0; self.n_features];

        let mut risk_set_sum = 0.0;
        let mut risk_set_x_sum = vec![0.0; self.n_features];
        let mut risk_set_x2_sum = vec![0.0; self.n_features];

        for &i in &sorted_indices {
            let eta: f64 = self
                .coefficients
                .iter()
                .zip(covariates[i].iter())
                .map(|(&c, &x)| c * x)
                .sum();
            let exp_eta = eta.exp().min(1e10);

            risk_set_sum += exp_eta;
            for j in 0..self.n_features {
                risk_set_x_sum[j] += exp_eta * covariates[i][j];
                risk_set_x2_sum[j] += exp_eta * covariates[i][j].powi(2);
            }

            if event[i] == 1 {
                for j in 0..self.n_features {
                    if risk_set_sum > 1e-10 {
                        let mean_x = risk_set_x_sum[j] / risk_set_sum;
                        gradient[j] += covariates[i][j] - mean_x;

                        let var_x = risk_set_x2_sum[j] / risk_set_sum - mean_x.powi(2);
                        hessian[j] += var_x.max(0.0);
                    }
                }
            }
        }

        let decay = self.config.decay_factor;
        for j in 0..self.n_features {
            self.gradient_accumulator[j] = decay * self.gradient_accumulator[j] + gradient[j];
            self.hessian_accumulator[j] = decay * self.hessian_accumulator[j] + hessian[j];
        }

        let n_events: usize = event.iter().filter(|&&e| e == 1).count();
        self.n_events_seen += n_events;
        self.n_samples_seen += n;

        if self.n_events_seen >= self.config.min_events_to_update {
            for j in 0..self.n_features {
                let h = self.hessian_accumulator[j] + self.config.regularization;
                if h > 1e-10 {
                    let update = self.config.learning_rate * self.gradient_accumulator[j] / h;
                    self.coefficients[j] += update.clamp(-1.0, 1.0);
                }
            }
        }

        for &i in &sorted_indices {
            if event[i] == 1 {
                let eta: f64 = self
                    .coefficients
                    .iter()
                    .zip(covariates[i].iter())
                    .map(|(&c, &x)| c * x)
                    .sum();
                let hazard = eta.exp() / risk_set_sum.max(1e-10);
                self.baseline_hazard.push((time[i], hazard));
            }
        }

        if self.baseline_hazard.len() > self.config.window_size {
            self.baseline_hazard
                .drain(0..self.baseline_hazard.len() - self.config.window_size);
        }

        Ok(())
    }

    fn predict_risk(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let risks: Vec<f64> = covariates
            .iter()
            .map(|cov| {
                let eta: f64 = self
                    .coefficients
                    .iter()
                    .zip(cov.iter())
                    .map(|(&c, &x)| c * x)
                    .sum();
                eta.exp()
            })
            .collect();

        Ok(risks)
    }

    fn predict_survival(&self, covariates: Vec<Vec<f64>>, time: f64) -> PyResult<Vec<f64>> {
        let cumulative_baseline: f64 = self
            .baseline_hazard
            .iter()
            .filter(|&&(t, _)| t <= time)
            .map(|&(_, h)| h)
            .sum();

        let survival: Vec<f64> = covariates
            .iter()
            .map(|cov| {
                let eta: f64 = self
                    .coefficients
                    .iter()
                    .zip(cov.iter())
                    .map(|(&c, &x)| c * x)
                    .sum();
                (-cumulative_baseline * eta.exp()).exp().clamp(0.0, 1.0)
            })
            .collect();

        Ok(survival)
    }

    fn get_coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    fn get_n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    fn get_n_events_seen(&self) -> usize {
        self.n_events_seen
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingCoxModel(n_features={}, samples_seen={}, events_seen={})",
            self.n_features, self.n_samples_seen, self.n_events_seen
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct StreamingKaplanMeier {
    time_points: Vec<f64>,
    at_risk: Vec<f64>,
    events: Vec<f64>,
    survival_probs: Vec<f64>,
    n_samples_seen: usize,
    window_size: usize,
}

#[pymethods]
impl StreamingKaplanMeier {
    #[new]
    #[pyo3(signature = (window_size=1000))]
    pub fn new(window_size: usize) -> Self {
        Self {
            time_points: Vec::new(),
            at_risk: Vec::new(),
            events: Vec::new(),
            survival_probs: Vec::new(),
            n_samples_seen: 0,
            window_size,
        }
    }

    fn partial_fit(&mut self, time: Vec<f64>, event: Vec<i32>) -> PyResult<()> {
        let n = time.len();
        if event.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and event must have same length",
            ));
        }

        for (&t, &e) in time.iter().zip(event.iter()) {
            let e = e as f64;

            let pos = self.time_points.iter().position(|&x| (x - t).abs() < 1e-10);

            if let Some(idx) = pos {
                self.at_risk[idx] += 1.0;
                self.events[idx] += e;
            } else {
                let insert_pos = self
                    .time_points
                    .iter()
                    .position(|&x| x > t)
                    .unwrap_or(self.time_points.len());

                self.time_points.insert(insert_pos, t);
                self.at_risk.insert(insert_pos, 1.0);
                self.events.insert(insert_pos, e);
            }
        }

        self.n_samples_seen += n;

        self.update_survival();

        if self.time_points.len() > self.window_size {
            let excess = self.time_points.len() - self.window_size;
            self.time_points.drain(0..excess);
            self.at_risk.drain(0..excess);
            self.events.drain(0..excess);
            self.survival_probs.drain(0..excess);
        }

        Ok(())
    }

    fn update_survival(&mut self) {
        self.survival_probs.clear();
        let mut surv = 1.0;

        let mut cumulative_at_risk = self.at_risk.iter().sum::<f64>();

        for i in 0..self.time_points.len() {
            if cumulative_at_risk > 0.0 {
                surv *= 1.0 - self.events[i] / cumulative_at_risk;
            }
            self.survival_probs.push(surv.max(0.0));
            cumulative_at_risk -= self.at_risk[i];
        }
    }

    fn predict_survival(&self, time: f64) -> f64 {
        if self.time_points.is_empty() {
            return 1.0;
        }

        let idx = self
            .time_points
            .iter()
            .position(|&t| t > time)
            .unwrap_or(self.time_points.len());

        if idx == 0 {
            1.0
        } else {
            self.survival_probs[idx - 1]
        }
    }

    fn get_survival_curve(&self) -> (Vec<f64>, Vec<f64>) {
        (self.time_points.clone(), self.survival_probs.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingKaplanMeier(n_time_points={}, samples_seen={})",
            self.time_points.len(),
            self.n_samples_seen
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ConceptDriftDetector {
    #[pyo3(get)]
    pub window_size: usize,
    #[pyo3(get)]
    pub threshold: f64,
    recent_log_likelihoods: Vec<f64>,
    reference_mean: f64,
    reference_std: f64,
    drift_detected: bool,
    n_samples_seen: usize,
}

#[pymethods]
impl ConceptDriftDetector {
    #[new]
    #[pyo3(signature = (window_size=100, threshold=3.0))]
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            recent_log_likelihoods: Vec::new(),
            reference_mean: 0.0,
            reference_std: 1.0,
            drift_detected: false,
            n_samples_seen: 0,
        }
    }

    fn update(&mut self, log_likelihood: f64) -> bool {
        self.recent_log_likelihoods.push(log_likelihood);
        self.n_samples_seen += 1;

        if self.recent_log_likelihoods.len() > self.window_size {
            self.recent_log_likelihoods.remove(0);
        }

        if self.n_samples_seen == self.window_size {
            let n = self.recent_log_likelihoods.len() as f64;
            self.reference_mean = self.recent_log_likelihoods.iter().sum::<f64>() / n;
            self.reference_std = (self
                .recent_log_likelihoods
                .iter()
                .map(|&x| (x - self.reference_mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
                .max(0.01);
        }

        if self.n_samples_seen > self.window_size {
            let n = self.recent_log_likelihoods.len() as f64;
            let current_mean = self.recent_log_likelihoods.iter().sum::<f64>() / n;
            let z_score = (current_mean - self.reference_mean) / (self.reference_std / n.sqrt());

            if z_score.abs() > self.threshold {
                self.drift_detected = true;
                self.reference_mean = current_mean;
                self.reference_std = (self
                    .recent_log_likelihoods
                    .iter()
                    .map(|&x| (x - current_mean).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt()
                    .max(0.01);
                return true;
            }
        }

        false
    }

    fn is_drift_detected(&self) -> bool {
        self.drift_detected
    }

    fn reset(&mut self) {
        self.drift_detected = false;
        self.recent_log_likelihoods.clear();
        self.n_samples_seen = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "ConceptDriftDetector(window={}, threshold={:.2}, drift={})",
            self.window_size, self.threshold, self.drift_detected
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_cox_model() {
        let mut model = StreamingCoxModel::new(2, None);

        let covariates = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let event = vec![1, 0, 1, 1];

        model.partial_fit(covariates.clone(), time, event).unwrap();

        assert_eq!(model.get_n_samples_seen(), 4);
        assert_eq!(model.get_coefficients().len(), 2);
    }

    #[test]
    fn test_streaming_km() {
        let mut km = StreamingKaplanMeier::new(100);

        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];

        km.partial_fit(time, event).unwrap();

        let surv = km.predict_survival(2.5);
        assert!((0.0..=1.0).contains(&surv));
    }

    #[test]
    fn test_concept_drift_detector() {
        let mut detector = ConceptDriftDetector::new(10, 2.0);

        for i in 0..20 {
            let ll = if i < 15 { -10.0 + i as f64 * 0.1 } else { -5.0 };
            detector.update(ll);
        }

        assert!(detector.n_samples_seen >= 10);
    }

    #[test]
    fn test_streaming_cox_predict() {
        let mut model = StreamingCoxModel::new(2, None);

        let covariates = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.2, 0.8],
            vec![0.8, 0.2],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 1, 0, 1, 1];

        for _ in 0..5 {
            model
                .partial_fit(covariates.clone(), time.clone(), event.clone())
                .unwrap();
        }

        let test_cov = vec![vec![0.5, 0.5]];
        let risks = model.predict_risk(test_cov.clone()).unwrap();
        assert!(!risks.is_empty());
        assert!(risks[0] > 0.0);

        let survival = model.predict_survival(test_cov, 3.0).unwrap();
        assert!(survival[0] >= 0.0 && survival[0] <= 1.0);
    }

    #[test]
    fn test_streaming_cox_feature_mismatch() {
        let mut model = StreamingCoxModel::new(2, None);
        let covariates = vec![vec![1.0, 0.5, 0.3]];
        let time = vec![1.0];
        let event = vec![1];

        let result = model.partial_fit(covariates, time, event);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_cox_invalid_input() {
        let mut model = StreamingCoxModel::new(2, None);
        let result = model.partial_fit(vec![], vec![], vec![]);
        assert!(result.is_err());

        let result = model.partial_fit(vec![vec![1.0, 0.5]], vec![1.0, 2.0], vec![1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_cox_incremental_updates() {
        let mut model = StreamingCoxModel::new(2, None);

        let cov1 = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let time1 = vec![1.0, 2.0, 3.0];
        let event1 = vec![1, 0, 1];
        model.partial_fit(cov1, time1, event1).unwrap();
        assert_eq!(model.get_n_samples_seen(), 3);

        let cov2 = vec![vec![0.5, 0.5], vec![0.2, 0.8]];
        let time2 = vec![4.0, 5.0];
        let event2 = vec![1, 0];
        model.partial_fit(cov2, time2, event2).unwrap();
        assert_eq!(model.get_n_samples_seen(), 5);
    }

    #[test]
    fn test_streaming_km_decreasing_survival() {
        let mut km = StreamingKaplanMeier::new(100);

        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 1, 1, 1];
        km.partial_fit(time, event).unwrap();

        let s1 = km.predict_survival(1.5);
        let s2 = km.predict_survival(3.5);
        let s3 = km.predict_survival(5.5);
        assert!(s1 >= s2);
        assert!(s2 >= s3);
    }

    #[test]
    fn test_streaming_km_empty_predict() {
        let km = StreamingKaplanMeier::new(100);
        let surv = km.predict_survival(5.0);
        assert!((surv - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_km_get_curve() {
        let mut km = StreamingKaplanMeier::new(100);
        let time = vec![1.0, 2.0, 3.0];
        let event = vec![1, 0, 1];
        km.partial_fit(time, event).unwrap();

        let (times, probs) = km.get_survival_curve();
        assert_eq!(times.len(), probs.len());
        assert!(!times.is_empty());
    }

    #[test]
    fn test_streaming_km_input_mismatch() {
        let mut km = StreamingKaplanMeier::new(100);
        let result = km.partial_fit(vec![1.0, 2.0], vec![1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_concept_drift_reset() {
        let mut detector = ConceptDriftDetector::new(10, 2.0);

        for i in 0..20 {
            detector.update(-10.0 + i as f64 * 0.1);
        }

        detector.reset();
        assert!(!detector.is_drift_detected());
        assert_eq!(detector.n_samples_seen, 0);
    }

    #[test]
    fn test_streaming_cox_with_custom_config() {
        let config = StreamingCoxConfig::new(0.05, 0.01, 16, 500, 0.95, 5);
        let mut model = StreamingCoxModel::new(3, Some(config));

        let covariates = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.3],
            vec![0.5, 0.5, 0.8],
            vec![0.3, 0.7, 0.2],
            vec![0.8, 0.2, 0.6],
            vec![0.6, 0.4, 0.4],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 1, 0, 1, 1, 0];

        model.partial_fit(covariates.clone(), time, event).unwrap();

        let coeffs = model.get_coefficients();
        assert_eq!(coeffs.len(), 3);
        assert_eq!(model.get_n_events_seen(), 4);
    }
}
