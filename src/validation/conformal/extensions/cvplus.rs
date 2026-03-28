use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CVPlusCalibrationResult {
    #[pyo3(get)]
    pub conformity_scores: Vec<f64>,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_calibration: usize,
    #[pyo3(get)]
    pub adjustment_factor: f64,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CVPlusConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub loo_scores: Vec<f64>,
    #[pyo3(get)]
    pub n_calibration: usize,
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_loo, coverage_level=None))]
pub fn cvplus_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_loo: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<CVPlusCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted_loo.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted_loo must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted_loo[i];
            scores.push(score);
        }
    }

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let adjustment_factor = (n_uncensored as f64 + 1.0) / n_uncensored as f64;
    let quantile_level = (1.0 - coverage) * adjustment_factor;
    let quantile_level = quantile_level.min(1.0);

    let weights: Vec<f64> = vec![1.0; n_uncensored];
    let quantile_threshold = weighted_quantile(&scores, &weights, quantile_level);

    Ok(CVPlusCalibrationResult {
        conformity_scores: scores,
        quantile_threshold,
        coverage_level: coverage,
        n_calibration: n_uncensored,
        adjustment_factor,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_loo, predicted_new, coverage_level=None))]
pub fn cvplus_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_loo: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<CVPlusConformalResult> {
    let calib_result = cvplus_conformal_calibrate(time, status, predicted_loo, coverage_level)?;

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - calib_result.quantile_threshold).max(0.0))
        .collect();

    Ok(CVPlusConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: calib_result.coverage_level,
        quantile_threshold: calib_result.quantile_threshold,
        loo_scores: calib_result.conformity_scores,
        n_calibration: calib_result.n_calibration,
    })
}
