use super::*;

#[pyfunction]
#[pyo3(signature = (time, status, predicted, coverage_level=None, use_ipcw=None))]
pub fn conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
) -> PyResult<ConformalCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let use_ipcw_flag = use_ipcw.unwrap_or(true);
    let (scores, weights) =
        compute_conformity_scores(&time, &status, &predicted, use_ipcw_flag, DEFAULT_IPCW_TRIM);

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let quantile_level = (1.0 - coverage) * (1.0 + 1.0 / n_uncensored as f64);
    let quantile_level = quantile_level.min(1.0);

    let quantile_threshold = weighted_quantile(&scores, &weights, quantile_level);

    let n_effective = if use_ipcw_flag {
        let sum_weights: f64 = weights.iter().sum();
        let sum_sq_weights: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq_weights > 0.0 {
            sum_weights * sum_weights / sum_sq_weights
        } else {
            n_uncensored as f64
        }
    } else {
        n_uncensored as f64
    };

    Ok(ConformalCalibrationResult {
        conformity_scores: scores,
        ipcw_weights: if use_ipcw_flag { Some(weights) } else { None },
        quantile_threshold,
        coverage_level: coverage,
        n_calibration: n_uncensored,
        n_effective,
    })
}

#[pyfunction]
#[pyo3(signature = (quantile_threshold, predicted_new, coverage_level=None))]
pub fn conformal_predict(
    quantile_threshold: f64,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalPredictionResult> {
    if predicted_new.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - quantile_threshold).max(0.0))
        .collect();

    Ok(ConformalPredictionResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
    })
}

#[pyfunction]
#[pyo3(signature = (time_calib, status_calib, predicted_calib, predicted_new, coverage_level=None, use_ipcw=None))]
pub fn conformal_survival_from_predictions(
    time_calib: Vec<f64>,
    status_calib: Vec<i32>,
    predicted_calib: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
) -> PyResult<ConformalPredictionResult> {
    let calib_result = conformal_calibrate(
        time_calib,
        status_calib,
        predicted_calib,
        coverage_level,
        use_ipcw,
    )?;

    conformal_predict(
        calib_result.quantile_threshold,
        predicted_new,
        Some(calib_result.coverage_level),
    )
}

#[pyfunction]
#[pyo3(signature = (time_test, status_test, lpb, coverage_level=None))]
pub fn conformal_coverage_test(
    time_test: Vec<f64>,
    status_test: Vec<i32>,
    lpb: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalDiagnostics> {
    let n = time_test.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status_test.len() != n || lpb.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time_test, status_test, and lpb must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let mut covered_count = 0usize;
    let mut total_count = 0usize;

    for i in 0..n {
        if status_test[i] == 1 {
            total_count += 1;
            if time_test[i] >= lpb[i] {
                covered_count += 1;
            }
        }
    }

    if total_count == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in test set",
        ));
    }

    let empirical_coverage = covered_count as f64 / total_count as f64;
    let expected_coverage = coverage;

    let se = (empirical_coverage * (1.0 - empirical_coverage) / total_count as f64).sqrt();
    let z = 1.96;
    let coverage_ci_lower = (empirical_coverage - z * se).max(0.0);
    let coverage_ci_upper = (empirical_coverage + z * se).min(1.0);

    let mut sorted_lpb: Vec<f64> = lpb.clone();
    sorted_lpb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean_lpb = lpb.iter().sum::<f64>() / n as f64;
    let median_lpb = if n.is_multiple_of(2) {
        (sorted_lpb[n / 2 - 1] + sorted_lpb[n / 2]) / 2.0
    } else {
        sorted_lpb[n / 2]
    };

    Ok(ConformalDiagnostics {
        empirical_coverage,
        expected_coverage,
        coverage_ci_lower,
        coverage_ci_upper,
        mean_lpb,
        median_lpb,
    })
}
