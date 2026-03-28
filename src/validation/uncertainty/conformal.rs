
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ConformalSurvivalConfig {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl ConformalSurvivalConfig {
    #[new]
    #[pyo3(signature = (alpha=0.1, method="cqr".to_string(), n_bootstrap=100, seed=None))]
    pub fn new(alpha: f64, method: String, n_bootstrap: usize, seed: Option<u64>) -> Self {
        Self {
            alpha,
            method,
            n_bootstrap,
            seed,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ConformalSurvivalResult {
    #[pyo3(get)]
    pub lower_bounds: Vec<f64>,
    #[pyo3(get)]
    pub upper_bounds: Vec<f64>,
    #[pyo3(get)]
    pub point_predictions: Vec<f64>,
    #[pyo3(get)]
    pub coverage: f64,
    #[pyo3(get)]
    pub interval_widths: Vec<f64>,
    #[pyo3(get)]
    pub calibration_scores: Vec<f64>,
}

#[pymethods]
impl ConformalSurvivalResult {
    #[new]
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        point_predictions: Vec<f64>,
        coverage: f64,
        interval_widths: Vec<f64>,
        calibration_scores: Vec<f64>,
    ) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            point_predictions,
            coverage,
            interval_widths,
            calibration_scores,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct BayesianBootstrapConfig {
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub confidence_level: f64,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl BayesianBootstrapConfig {
    #[new]
    #[pyo3(signature = (n_bootstrap=1000, confidence_level=0.95, seed=None))]
    pub fn new(n_bootstrap: usize, confidence_level: f64, seed: Option<u64>) -> Self {
        Self {
            n_bootstrap,
            confidence_level,
            seed,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct BayesianBootstrapResult {
    #[pyo3(get)]
    pub mean_survival: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub posterior_samples: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub credible_bands: HashMap<String, Vec<f64>>,
}

#[pymethods]
impl BayesianBootstrapResult {
    #[new]
    pub fn new(
        mean_survival: Vec<f64>,
        lower_ci: Vec<f64>,
        upper_ci: Vec<f64>,
        time_points: Vec<f64>,
        posterior_samples: Vec<Vec<f64>>,
        credible_bands: HashMap<String, Vec<f64>>,
    ) -> Self {
        Self {
            mean_survival,
            lower_ci,
            upper_ci,
            time_points,
            posterior_samples,
            credible_bands,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct JackknifePlusConfig {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub plus_variant: bool,
    #[pyo3(get, set)]
    pub cv_folds: usize,
}

#[pymethods]
impl JackknifePlusConfig {
    #[new]
    #[pyo3(signature = (alpha=0.1, plus_variant=true, cv_folds=5))]
    pub fn new(alpha: f64, plus_variant: bool, cv_folds: usize) -> Self {
        Self {
            alpha,
            plus_variant,
            cv_folds,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct JackknifePlusResult {
    #[pyo3(get)]
    pub lower_bounds: Vec<f64>,
    #[pyo3(get)]
    pub upper_bounds: Vec<f64>,
    #[pyo3(get)]
    pub point_predictions: Vec<f64>,
    #[pyo3(get)]
    pub coverage: f64,
    #[pyo3(get)]
    pub residuals: Vec<f64>,
}

#[pymethods]
impl JackknifePlusResult {
    #[new]
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        point_predictions: Vec<f64>,
        coverage: f64,
        residuals: Vec<f64>,
    ) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            point_predictions,
            coverage,
            residuals,
        }
    }
}

fn compute_conformity_scores(
    time: &[f64],
    event: &[i32],
    predictions: &[f64],
    method: &str,
) -> Vec<f64> {
    match method {
        "cqr" => time
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).abs())
            .collect(),
        "weighted" => time
            .iter()
            .zip(event.iter())
            .zip(predictions.iter())
            .map(|((t, e), p)| {
                let weight = if *e == 1 { 1.0 } else { 0.5 };
                weight * (t - p).abs()
            })
            .collect(),
        "censoring_adjusted" => {
            let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            time.iter()
                .zip(event.iter())
                .zip(predictions.iter())
                .map(|((t, e), p)| {
                    if *e == 1 {
                        (t - p).abs()
                    } else {
                        ((max_time - t) / max_time) * (t - p).abs()
                    }
                })
                .collect()
        }
        _ => time
            .iter()
            .zip(predictions.iter())
            .map(|(t, p)| (t - p).abs())
            .collect(),
    }
}

fn compute_coverage(lower: &[f64], upper: &[f64], actual: &[f64]) -> f64 {
    if actual.is_empty() || lower.len() != upper.len() {
        return 0.0;
    }

    let n = lower.len().min(actual.len());
    let covered = (0..n)
        .filter(|&i| actual[i] >= lower[i] && actual[i] <= upper[i])
        .count();

    covered as f64 / n as f64
}
