use super::super::*;

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None))]
pub fn conformal_survival_parallel(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
) -> PyResult<ConformalPredictionResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);

    let scores: Vec<f64> = (0..n)
        .into_par_iter()
        .filter_map(|i| {
            if status[i] == 1 {
                Some(time[i] - predicted[i])
            } else {
                None
            }
        })
        .collect();

    let n_scores = scores.len();
    if n_scores == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations",
        ));
    }

    let weights: Vec<f64> = vec![1.0; n_scores];
    let q_level = (1.0 - coverage) * (n_scores as f64 + 1.0) / n_scores as f64;
    let q_level = q_level.min(1.0);
    let threshold = weighted_quantile(&scores, &weights, q_level);

    let lower_predictive_bound: Vec<f64> = predicted_new
        .par_iter()
        .map(|&p| (p - threshold).max(0.0))
        .collect();

    Ok(ConformalPredictionResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
    })
}
