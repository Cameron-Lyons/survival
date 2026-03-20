use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct WeightDiagnostics {
    #[pyo3(get)]
    pub effective_sample_size: f64,
    #[pyo3(get)]
    pub min_weight: f64,
    #[pyo3(get)]
    pub max_weight: f64,
    #[pyo3(get)]
    pub weight_variance: f64,
    #[pyo3(get)]
    pub n_trimmed: usize,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CovariateShiftConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_threshold: f64,
    #[pyo3(get)]
    pub combined_weights: Vec<f64>,
    #[pyo3(get)]
    pub weight_diagnostics: WeightDiagnostics,
    #[pyo3(get)]
    pub n_calibration: usize,
    #[pyo3(get)]
    pub n_effective: f64,
}

fn compute_weight_diagnostics(weights: &[f64], n_trimmed: usize) -> WeightDiagnostics {
    if weights.is_empty() {
        return WeightDiagnostics {
            effective_sample_size: 0.0,
            min_weight: 0.0,
            max_weight: 0.0,
            weight_variance: 0.0,
            n_trimmed,
        };
    }

    let sum_weights: f64 = weights.iter().sum();
    let sum_sq_weights: f64 = weights.iter().map(|w| w * w).sum();
    let effective_sample_size = if sum_sq_weights > 0.0 {
        sum_weights * sum_weights / sum_sq_weights
    } else {
        weights.len() as f64
    };

    let min_weight = weights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mean_weight = sum_weights / weights.len() as f64;
    let weight_variance = weights
        .iter()
        .map(|&w| (w - mean_weight).powi(2))
        .sum::<f64>()
        / weights.len() as f64;

    WeightDiagnostics {
        effective_sample_size,
        min_weight,
        max_weight,
        weight_variance,
        n_trimmed,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, importance_weights, predicted_new, coverage_level=None, use_ipcw=None, weight_trim=None))]
#[allow(clippy::too_many_arguments)]
pub fn covariate_shift_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    importance_weights: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    use_ipcw: Option<bool>,
    weight_trim: Option<f64>,
) -> PyResult<CovariateShiftConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n || importance_weights.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, predicted, and importance_weights must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let use_ipcw_flag = use_ipcw.unwrap_or(true);
    let trim = weight_trim.unwrap_or(DEFAULT_WEIGHT_TRIM);

    let censoring_surv = if use_ipcw_flag {
        compute_km_censoring_survival(&time, &status)
    } else {
        vec![1.0; n]
    };

    let mut scores = Vec::with_capacity(n);
    let mut combined_weights = Vec::with_capacity(n);
    let mut n_trimmed = 0usize;

    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted[i];
            scores.push(score);

            let ipcw_weight = if use_ipcw_flag {
                1.0 / censoring_surv[i].max(trim)
            } else {
                1.0
            };

            let mut combined = importance_weights[i] * ipcw_weight;

            let max_combined =
                importance_weights.iter().cloned().fold(0.0_f64, f64::max) * MAX_WEIGHT_RATIO;
            if combined > max_combined && max_combined > 0.0 {
                combined = max_combined;
                n_trimmed += 1;
            }
            if combined < trim {
                combined = trim;
                n_trimmed += 1;
            }

            combined_weights.push(combined);
        }
    }

    let n_uncensored = scores.len();
    if n_uncensored == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let quantile_level = (1.0 - coverage) * (1.0 + 1.0 / n_uncensored as f64);
    let quantile_level = quantile_level.min(1.0);

    let quantile_threshold = weighted_quantile(&scores, &combined_weights, quantile_level);

    let weight_diagnostics = compute_weight_diagnostics(&combined_weights, n_trimmed);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - quantile_threshold).max(0.0))
        .collect();

    Ok(CovariateShiftConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
        quantile_threshold,
        combined_weights,
        weight_diagnostics: weight_diagnostics.clone(),
        n_calibration: n_uncensored,
        n_effective: weight_diagnostics.effective_sample_size,
    })
}
