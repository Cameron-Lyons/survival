fn aggregate_shap_values(
    shap_values: &[Vec<Vec<f64>>],
    time_points: &[f64],
    method: AggregationMethod,
    n_features: usize,
    n_times: usize,
) -> Vec<f64> {
    let n_samples = shap_values.len();
    if n_samples == 0 || n_times == 0 {
        return vec![0.0; n_features];
    }

    let time_diffs: Vec<f64> = if n_times >= 2 {
        time_points.windows(2).map(|w| w[1] - w[0]).collect()
    } else {
        Vec::new()
    };

    let total_time = time_points.last().unwrap_or(&1.0) - time_points.first().unwrap_or(&0.0);
    let time_weights: Vec<f64> = if total_time > 0.0 {
        time_points
            .iter()
            .map(|&t| 1.0 - (t - time_points[0]) / total_time)
            .collect()
    } else {
        Vec::new()
    };

    let mut importance = vec![0.0; n_features];

    for f in 0..n_features {
        let mut feature_agg = 0.0;

        for sample in shap_values.iter() {
            let shap_t = &sample[f];

            let sample_agg = match method {
                AggregationMethod::Mean => {
                    shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                }
                AggregationMethod::MaxAbsolute => {
                    shap_t.iter().map(|v| v.abs()).fold(0.0, f64::max)
                }
                AggregationMethod::Integral => {
                    if n_times < 2 {
                        shap_t.first().copied().unwrap_or(0.0).abs()
                    } else {
                        let mut integral = 0.0;
                        for i in 0..time_diffs.len() {
                            let avg = (shap_t[i + 1].abs() + shap_t[i].abs()) / 2.0;
                            integral += avg * time_diffs[i];
                        }
                        integral
                    }
                }
                AggregationMethod::TimeWeighted => {
                    if total_time <= 0.0 {
                        shap_t.iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                    } else {
                        let mut weighted_sum = 0.0;
                        for (i, &weight) in time_weights.iter().enumerate() {
                            weighted_sum += shap_t[i].abs() * weight;
                        }
                        weighted_sum / n_times as f64
                    }
                }
            };

            feature_agg += sample_agg;
        }

        importance[f] = feature_agg / n_samples as f64;
    }

    importance
}

#[pyfunction]
#[pyo3(signature = (shap_values, time_points, method))]
pub fn aggregate_survshap(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    method: AggregationMethod,
) -> PyResult<Vec<f64>> {
    let n_samples = shap_values.len();
    if n_samples == 0 {
        return Ok(Vec::new());
    }

    let n_features = shap_values[0].len();
    let n_times = time_points.len();

    if n_features == 0 {
        return Ok(Vec::new());
    }

    for sample in &shap_values {
        if sample.len() != n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All samples must have the same number of features",
            ));
        }
        for feature_shap in sample {
            if feature_shap.len() != n_times {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "SHAP values time dimension must match time_points length",
                ));
            }
        }
    }

    Ok(aggregate_shap_values(
        &shap_values,
        &time_points,
        method,
        n_features,
        n_times,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    x_explain,
    x_background,
    time_points,
    n_explain,
    n_background,
    n_features,
    predict_fn,
    config=None,
    aggregation_method=None
))]
#[allow(clippy::too_many_arguments)]
pub fn survshap_from_model(
    py: Python<'_>,
    x_explain: Vec<f64>,
    x_background: Vec<f64>,
    time_points: Vec<f64>,
    n_explain: usize,
    n_background: usize,
    n_features: usize,
    predict_fn: Py<PyAny>,
    config: Option<&SurvShapConfig>,
    aggregation_method: Option<AggregationMethod>,
) -> PyResult<SurvShapResult> {
    if x_explain.len() != n_explain * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_explain length must equal n_explain * n_features",
        ));
    }
    if x_background.len() != n_background * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x_background length must equal n_background * n_features",
        ));
    }

    let predictions_explain: Vec<f64> = predict_fn
        .call(py, (x_explain.clone(), n_explain), None)?
        .extract(py)?;

    let predictions_background: Vec<f64> = predict_fn
        .call(py, (x_background.clone(), n_background), None)?
        .extract(py)?;

    survshap(
        x_explain,
        x_background,
        predictions_explain,
        predictions_background,
        time_points,
        n_explain,
        n_background,
        n_features,
        config,
        aggregation_method,
    )
}

