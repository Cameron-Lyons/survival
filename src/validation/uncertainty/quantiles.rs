#[pyfunction]
#[pyo3(signature = (
    model_predictions,
    confidence_level=0.95
))]
pub fn ensemble_uncertainty(
    model_predictions: Vec<Vec<Vec<f64>>>,
    confidence_level: f64,
) -> PyResult<EnsembleUncertaintyResult> {
    if model_predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "model_predictions must not be empty",
        ));
    }

    let n_models = model_predictions.len();
    let n_obs = model_predictions[0].len();
    let n_times = model_predictions[0].first().map(|p| p.len()).unwrap_or(0);

    let mean_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| model_predictions.iter().map(|p| p[i][t]).sum::<f64>() / n_models as f64)
                .collect()
        })
        .collect();

    let std_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mean = mean_prediction[i][t];
                    let var: f64 = model_predictions
                        .iter()
                        .map(|p| (p[i][t] - mean).powi(2))
                        .sum::<f64>()
                        / n_models as f64;
                    var.sqrt()
                })
                .collect()
        })
        .collect();
    let model_disagreement: Vec<f64> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut total_disagreement = 0.0;
            for (t_idx, _) in model_predictions[0][i].iter().enumerate().take(n_times) {
                for m1 in 0..n_models {
                    for m2 in (m1 + 1)..n_models {
                        total_disagreement += (model_predictions[m1][i][t_idx]
                            - model_predictions[m2][i][t_idx])
                            .abs();
                    }
                }
            }
            let n_pairs = (n_models * (n_models - 1) / 2) as f64;
            total_disagreement / (n_pairs * n_times as f64)
        })
        .collect();

    let z = 1.96 * (1.0 + (1.0 - confidence_level).ln().abs()).sqrt();

    let prediction_intervals: Vec<Vec<(f64, f64)>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| ((mi - z * si).clamp(0.0, 1.0), (mi + z * si).clamp(0.0, 1.0)))
                .collect()
        })
        .collect();

    Ok(EnsembleUncertaintyResult {
        mean_prediction,
        std_prediction,
        model_disagreement,
        prediction_intervals,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct QuantileRegressionResult {
    #[pyo3(get)]
    pub median: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub lower_quantile: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub upper_quantile: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub quantiles: Vec<f64>,
}

#[pymethods]
impl QuantileRegressionResult {
    fn __repr__(&self) -> String {
        format!("QuantileRegressionResult(quantiles={:?})", self.quantiles)
    }

    fn prediction_interval_width(&self) -> Vec<Vec<f64>> {
        self.upper_quantile
            .iter()
            .zip(self.lower_quantile.iter())
            .map(|(u, l)| u.iter().zip(l.iter()).map(|(&ui, &li)| ui - li).collect())
            .collect()
    }
}

#[cfg(test)]
fn compute_quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (q * (values.len() - 1) as f64).round() as usize;
    values[idx.min(values.len() - 1)]
}

#[inline]
fn quantile_index(len: usize, q: f64) -> usize {
    if len == 0 {
        0
    } else {
        ((q * (len - 1) as f64).round() as usize).min(len - 1)
    }
}

#[pyfunction]
#[pyo3(signature = (
    bootstrap_predictions,
    quantiles=None
))]
pub fn quantile_regression_intervals(
    bootstrap_predictions: Vec<Vec<Vec<f64>>>,
    quantiles: Option<Vec<f64>>,
) -> PyResult<QuantileRegressionResult> {
    let quantiles = quantiles.unwrap_or_else(|| vec![0.025, 0.5, 0.975]);

    if bootstrap_predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bootstrap_predictions must not be empty",
        ));
    }

    let n_obs = bootstrap_predictions[0].len();
    let n_times = bootstrap_predictions[0]
        .first()
        .map(|p| p.len())
        .unwrap_or(0);

    let lower_q = quantiles.first().copied().unwrap_or(0.025);
    let median_q = quantiles.get(1).copied().unwrap_or(0.5);
    let upper_q = quantiles.last().copied().unwrap_or(0.975);

    let quantile_bands: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            let mut lower = vec![0.0; n_times];
            let mut med = vec![0.0; n_times];
            let mut upper = vec![0.0; n_times];

            for t in 0..n_times {
                let mut values: Vec<f64> = bootstrap_predictions.iter().map(|p| p[i][t]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lower_idx = quantile_index(values.len(), lower_q);
                let median_idx = quantile_index(values.len(), median_q);
                let upper_idx = quantile_index(values.len(), upper_q);
                lower[t] = values[lower_idx];
                med[t] = values[median_idx];
                upper[t] = values[upper_idx];
            }

            (lower, med, upper)
        })
        .collect();

    let (lower_quantile, median, upper_quantile): QuantilePredictionBands =
        quantile_bands.into_iter().fold(
            (
                Vec::with_capacity(n_obs),
                Vec::with_capacity(n_obs),
                Vec::with_capacity(n_obs),
            ),
            |mut acc, (l, m, u)| {
                acc.0.push(l);
                acc.1.push(m);
                acc.2.push(u);
                acc
            },
        );

    Ok(QuantileRegressionResult {
        median,
        lower_quantile,
        upper_quantile,
        quantiles,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CalibrationUncertaintyResult {
    #[pyo3(get)]
    pub expected_coverage: f64,
    #[pyo3(get)]
    pub observed_coverage: f64,
    #[pyo3(get)]
    pub calibration_error: f64,
    #[pyo3(get)]
    pub sharpness: f64,
}

#[pymethods]
impl CalibrationUncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "CalibrationUncertaintyResult(expected={:.3}, observed={:.3}, error={:.3})",
            self.expected_coverage, self.observed_coverage, self.calibration_error
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    true_times,
    true_events,
    lower_bounds,
    upper_bounds,
    expected_coverage=0.95
))]
pub fn calibrate_prediction_intervals(
    true_times: Vec<f64>,
    true_events: Vec<i32>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    expected_coverage: f64,
) -> PyResult<CalibrationUncertaintyResult> {
    let n = true_times.len();
    if n == 0 || true_events.len() != n || lower_bounds.len() != n || upper_bounds.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut covered = 0;
    let mut total_width = 0.0;

    for i in 0..n {
        if true_events[i] == 1
            && true_times[i] >= lower_bounds[i]
            && true_times[i] <= upper_bounds[i]
        {
            covered += 1;
        }
        total_width += upper_bounds[i] - lower_bounds[i];
    }

    let n_events = true_events.iter().filter(|&&e| e == 1).count();
    let observed_coverage = if n_events > 0 {
        covered as f64 / n_events as f64
    } else {
        0.0
    };

    let calibration_error = (observed_coverage - expected_coverage).abs();
    let sharpness = total_width / n as f64;

    Ok(CalibrationUncertaintyResult {
        expected_coverage,
        observed_coverage,
        calibration_error,
        sharpness,
    })
}
