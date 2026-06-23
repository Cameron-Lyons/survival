const NORMAL_CI_Z_SCORE: f64 = crate::constants::Z_SCORE_95;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MCDropoutConfig {
    #[pyo3(get, set)]
    pub n_samples: usize,
    #[pyo3(get, set)]
    pub dropout_rate: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl MCDropoutConfig {
    #[new]
    #[pyo3(signature = (n_samples=100, dropout_rate=0.1, seed=None))]
    pub fn new(n_samples: usize, dropout_rate: f64, seed: Option<u64>) -> PyResult<Self> {
        if n_samples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_samples must be positive",
            ));
        }
        if !dropout_rate.is_finite() || !(0.0..1.0).contains(&dropout_rate) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout_rate must be in [0, 1)",
            ));
        }
        Ok(Self {
            n_samples,
            dropout_rate,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct UncertaintyResult {
    #[pyo3(get)]
    pub mean_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub lower_ci: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub upper_ci: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub epistemic_uncertainty: Vec<f64>,
    #[pyo3(get)]
    pub aleatoric_uncertainty: Vec<f64>,
}

#[pymethods]
impl UncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "UncertaintyResult(n_samples={}, n_times={})",
            self.mean_prediction.len(),
            self.mean_prediction.first().map(|p| p.len()).unwrap_or(0)
        )
    }

    fn total_uncertainty(&self) -> Vec<f64> {
        self.epistemic_uncertainty
            .iter()
            .zip(self.aleatoric_uncertainty.iter())
            .map(|(&e, &a)| (e.powi(2) + a.powi(2)).sqrt())
            .collect()
    }
}

fn _apply_dropout(values: &[f64], dropout_rate: f64, rng: &mut fastrand::Rng) -> Vec<f64> {
    debug_assert!(dropout_rate.is_finite() && (0.0..1.0).contains(&dropout_rate));
    let scale = 1.0 / (1.0 - dropout_rate);
    values
        .iter()
        .map(|&v| {
            if rng.f64() < dropout_rate {
                0.0
            } else {
                v * scale
            }
        })
        .collect()
}

fn validate_probability_prediction_cube(
    predictions: &[Vec<Vec<f64>>],
) -> PyResult<(usize, usize, usize)> {
    let shape = validate_prediction_cube(predictions, "predictions")?;
    for (model_idx, model) in predictions.iter().enumerate() {
        for (obs_idx, row) in model.iter().enumerate() {
            for (time_idx, &value) in row.iter().enumerate() {
                if !(0.0..=1.0).contains(&value) {
                    return Err(uncertainty_value_error(format!(
                        "predictions must contain probabilities between 0 and 1; got {value} at model {model_idx}, observation {obs_idx}, time {time_idx}"
                    )));
                }
            }
        }
    }
    Ok(shape)
}

#[pyfunction]
#[pyo3(signature = (predictions,))]
pub fn mc_dropout_uncertainty(predictions: Vec<Vec<Vec<f64>>>) -> PyResult<UncertaintyResult> {
    let (n_samples, n_obs, n_times) = validate_probability_prediction_cube(&predictions)?;

    let mean_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| predictions.iter().map(|p| p[i][t]).sum::<f64>() / n_samples as f64)
                .collect()
        })
        .collect();

    let std_prediction: Vec<Vec<f64>> = (0..n_obs)
        .into_par_iter()
        .map(|i| {
            (0..n_times)
                .map(|t| {
                    let mean = mean_prediction[i][t];
                    let var: f64 = predictions
                        .iter()
                        .map(|p| (p[i][t] - mean).powi(2))
                        .sum::<f64>()
                        / n_samples as f64;
                    var.sqrt()
                })
                .collect()
        })
        .collect();

    let lower_ci: Vec<Vec<f64>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| (mi - NORMAL_CI_Z_SCORE * si).clamp(0.0, 1.0))
                .collect()
        })
        .collect();

    let upper_ci: Vec<Vec<f64>> = mean_prediction
        .iter()
        .zip(std_prediction.iter())
        .map(|(m, s)| {
            m.iter()
                .zip(s.iter())
                .map(|(&mi, &si)| (mi + NORMAL_CI_Z_SCORE * si).clamp(0.0, 1.0))
                .collect()
        })
        .collect();

    let epistemic_uncertainty: Vec<f64> = std_prediction
        .iter()
        .map(|s| s.iter().sum::<f64>() / s.len() as f64)
        .collect();

    let aleatoric_uncertainty: Vec<f64> = mean_prediction
        .iter()
        .map(|m| {
            let var: f64 = m.iter().map(|&mi| mi * (1.0 - mi)).sum::<f64>() / m.len() as f64;
            var.sqrt()
        })
        .collect();

    Ok(UncertaintyResult {
        mean_prediction,
        std_prediction,
        lower_ci,
        upper_ci,
        epistemic_uncertainty,
        aleatoric_uncertainty,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct EnsembleUncertaintyResult {
    #[pyo3(get)]
    pub mean_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_prediction: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub model_disagreement: Vec<f64>,
    #[pyo3(get)]
    pub prediction_intervals: Vec<Vec<(f64, f64)>>,
}

#[pymethods]
impl EnsembleUncertaintyResult {
    fn __repr__(&self) -> String {
        format!(
            "EnsembleUncertaintyResult(n_samples={})",
            self.mean_prediction.len()
        )
    }
}
