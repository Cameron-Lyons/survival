//Fit Proportional Hazards Regression Model

use ndarray::Array2;

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
}
