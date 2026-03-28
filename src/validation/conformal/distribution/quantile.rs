use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CQRConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_quantile_lower: Vec<f64>,
    #[pyo3(get)]
    pub predicted_quantile_upper: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub quantile_lower: f64,
    #[pyo3(get)]
    pub quantile_upper: f64,
}

fn estimate_conditional_quantile(
    time: &[f64],
    status: &[i32],
    predicted: &[f64],
    target_pred: f64,
    quantile: f64,
    bandwidth: f64,
) -> f64 {
    let mut weighted_times = Vec::new();
    let mut weights = Vec::new();

    for i in 0..time.len() {
        if status[i] == 1 {
            let dist = ((predicted[i] - target_pred) / bandwidth).abs();
            let weight = (-0.5 * dist * dist).exp();
            if weight > 1e-10 {
                weighted_times.push(time[i]);
                weights.push(weight);
            }
        }
    }

    if weighted_times.is_empty() {
        return target_pred;
    }

    weighted_quantile(&weighted_times, &weights, quantile)
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None, bandwidth=None))]
#[allow(clippy::too_many_arguments)]
pub fn cqr_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    bandwidth: Option<f64>,
) -> PyResult<CQRConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    let alpha = 1.0 - coverage;

    let pred_std = {
        let mean: f64 = predicted.iter().sum::<f64>() / n as f64;
        let variance: f64 = predicted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt()
    };
    let bw = bandwidth.unwrap_or(pred_std * 0.5);

    let quantile_lower = alpha / 2.0;
    let quantile_upper = 1.0 - alpha / 2.0;

    let results: Vec<(f64, f64)> = predicted_new
        .par_iter()
        .map(|&p| {
            let q_low =
                estimate_conditional_quantile(&time, &status, &predicted, p, quantile_lower, bw);
            let q_high =
                estimate_conditional_quantile(&time, &status, &predicted, p, quantile_upper, bw);
            (q_low, q_high)
        })
        .collect();

    let predicted_quantile_lower: Vec<f64> = results.iter().map(|(l, _)| *l).collect();
    let predicted_quantile_upper: Vec<f64> = results.iter().map(|(_, u)| *u).collect();

    let mut conformity_scores = Vec::new();
    for i in 0..n {
        if status[i] == 1 {
            let q_low = estimate_conditional_quantile(
                &time,
                &status,
                &predicted,
                predicted[i],
                quantile_lower,
                bw,
            );
            let q_high = estimate_conditional_quantile(
                &time,
                &status,
                &predicted,
                predicted[i],
                quantile_upper,
                bw,
            );
            let score = (q_low - time[i]).max(time[i] - q_high).max(0.0);
            conformity_scores.push(score);
        }
    }

    let n_scores = conformity_scores.len();
    if n_scores == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations",
        ));
    }

    let weights: Vec<f64> = vec![1.0; n_scores];
    let q_level = (1.0 - alpha) * (n_scores as f64 + 1.0) / n_scores as f64;
    let q_level = q_level.min(1.0);
    let threshold = weighted_quantile(&conformity_scores, &weights, q_level);

    let lower_bound: Vec<f64> = predicted_quantile_lower
        .iter()
        .map(|&q| (q - threshold).max(0.0))
        .collect();

    let upper_bound: Vec<f64> = predicted_quantile_upper
        .iter()
        .map(|&q| q + threshold)
        .collect();

    Ok(CQRConformalResult {
        lower_bound,
        upper_bound,
        predicted_quantile_lower,
        predicted_quantile_upper,
        coverage_level: coverage,
        quantile_lower,
        quantile_upper,
    })
}
