fn uncertainty_value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

fn validate_probability_open(value: f64, field: &str) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 || value >= 1.0 {
        return Err(uncertainty_value_error(format!(
            "{field} must be a finite value between 0 and 1"
        )));
    }
    Ok(())
}

fn validate_prediction_cube(
    predictions: &[Vec<Vec<f64>>],
    field: &str,
) -> PyResult<(usize, usize, usize)> {
    if predictions.is_empty() {
        return Err(uncertainty_value_error(format!("{field} must not be empty")));
    }

    let n_models = predictions.len();
    let n_obs = predictions[0].len();
    if n_obs == 0 {
        return Err(uncertainty_value_error(format!(
            "{field} must contain at least one observation"
        )));
    }

    let n_times = predictions[0].first().map(|p| p.len()).unwrap_or(0);
    if n_times == 0 {
        return Err(uncertainty_value_error(format!(
            "{field} must contain at least one time point"
        )));
    }

    for (model_idx, model) in predictions.iter().enumerate() {
        if model.len() != n_obs {
            return Err(uncertainty_value_error(format!(
                "{field} must be rectangular; model {model_idx} has {} observations, expected {n_obs}",
                model.len()
            )));
        }
        for (obs_idx, row) in model.iter().enumerate() {
            if row.len() != n_times {
                return Err(uncertainty_value_error(format!(
                    "{field} must be rectangular; model {model_idx}, observation {obs_idx} has {} time points, expected {n_times}",
                    row.len()
                )));
            }
            for (time_idx, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(uncertainty_value_error(format!(
                        "{field} contains non-finite value {value} at model {model_idx}, observation {obs_idx}, time {time_idx}"
                    )));
                }
            }
        }
    }

    Ok((n_models, n_obs, n_times))
}

fn validate_quantiles(quantiles: &[f64]) -> PyResult<()> {
    if quantiles.len() != 3 {
        return Err(uncertainty_value_error(
            "quantiles must contain exactly three values: lower, median, and upper",
        ));
    }
    for (idx, &value) in quantiles.iter().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(uncertainty_value_error(format!(
                "quantiles must contain finite values between 0 and 1; got {value} at index {idx}"
            )));
        }
    }
    for (idx, pair) in quantiles.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(uncertainty_value_error(format!(
                "quantiles must be nondecreasing; got {} then {} at positions {} and {}",
                pair[0],
                pair[1],
                idx,
                idx + 1
            )));
        }
    }
    Ok(())
}

fn validate_finite_vector(values: &[f64], field: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(uncertainty_value_error(format!(
                "{field} contains non-finite value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_binary_events(values: &[i32], field: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(uncertainty_value_error(format!(
                "{field} values must be 0 or 1; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (
    model_predictions,
    confidence_level=0.95
))]
pub fn ensemble_uncertainty(
    model_predictions: Vec<Vec<Vec<f64>>>,
    confidence_level: f64,
) -> PyResult<EnsembleUncertaintyResult> {
    validate_probability_open(confidence_level, "confidence_level")?;
    let (n_models, n_obs, n_times) =
        validate_prediction_cube(&model_predictions, "model_predictions")?;

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
            if n_pairs > 0.0 {
                total_disagreement / (n_pairs * n_times as f64)
            } else {
                0.0
            }
        })
        .collect();

    let z = Z_SCORE_95 * (1.0 + (1.0 - confidence_level).ln().abs()).sqrt();

    let prediction_intervals: Vec<Vec<(f64, f64)>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            let (lower, upper) = clamped_normal_ci_bounds(m, s, z, 0.0, 1.0);
            lower.into_iter().zip(upper).collect()
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
    values.sort_by(f64::total_cmp);
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
    validate_quantiles(&quantiles)?;

    let (_, n_obs, n_times) =
        validate_prediction_cube(&bootstrap_predictions, "bootstrap_predictions")?;

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
                values.sort_by(f64::total_cmp);
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
        return Err(uncertainty_value_error(
            "All inputs must have the same non-zero length",
        ));
    }
    validate_finite_vector(&true_times, "true_times")?;
    validate_binary_events(&true_events, "true_events")?;
    validate_finite_vector(&lower_bounds, "lower_bounds")?;
    validate_finite_vector(&upper_bounds, "upper_bounds")?;
    validate_probability_open(expected_coverage, "expected_coverage")?;

    let mut covered = 0;
    let mut total_width = 0.0;

    for i in 0..n {
        if lower_bounds[i] > upper_bounds[i] {
            return Err(uncertainty_value_error(format!(
                "lower_bounds must be less than or equal to upper_bounds; got {} > {} at index {i}",
                lower_bounds[i], upper_bounds[i]
            )));
        }
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
