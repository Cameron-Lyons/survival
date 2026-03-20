
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct StackingConfig {
    #[pyo3(get, set)]
    pub n_folds: usize,
    #[pyo3(get, set)]
    pub meta_model: String,
    #[pyo3(get, set)]
    pub use_probabilities: bool,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl StackingConfig {
    #[new]
    #[pyo3(signature = (
        n_folds=5,
        meta_model="cox",
        use_probabilities=true,
        seed=None
    ))]
    pub fn new(
        n_folds: usize,
        meta_model: &str,
        use_probabilities: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_folds < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_folds must be at least 2",
            ));
        }
        Ok(Self {
            n_folds,
            meta_model: meta_model.to_string(),
            use_probabilities,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct StackingResult {
    #[pyo3(get)]
    pub meta_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub stacked_predictions: Vec<f64>,
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub base_model_importance: Vec<f64>,
}

#[pymethods]
impl StackingResult {
    fn __repr__(&self) -> String {
        format!(
            "StackingResult(n_base_models={}, C-index={:.4})",
            self.meta_coefficients.len(),
            self.c_index
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    base_predictions,
    config
))]
pub fn stacking_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    base_predictions: Vec<Vec<f64>>,
    config: StackingConfig,
) -> PyResult<StackingResult> {
    let n = time.len();
    let n_models = base_predictions.len();

    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }
    if n_models == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must provide at least one base model",
        ));
    }

    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
    let folds = create_cv_folds(n, config.n_folds, seed);

    let mut oof_predictions: Vec<Vec<f64>> = vec![vec![0.0; n]; n_models];

    for test_indices in &folds {
        let train_indices: Vec<usize> = (0..n).filter(|i| !test_indices.contains(i)).collect();
        oof_predictions
            .par_iter_mut()
            .enumerate()
            .for_each(|(m, model_oof_predictions)| {
                let train_sum: f64 = train_indices.iter().map(|&i| base_predictions[m][i]).sum();
                let train_mean = if train_indices.is_empty() {
                    1.0
                } else {
                    train_sum / train_indices.len() as f64
                };

                for &test_i in test_indices {
                    model_oof_predictions[test_i] =
                        base_predictions[m][test_i] / train_mean.max(1e-10);
                }
            });
    }

    let meta_features: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| (0..n_models).map(|m| oof_predictions[m][i]).collect())
        .collect();

    let train_indices: Vec<usize> = (0..n).collect();
    let meta_coefficients = fit_base_cox(&time, &event, &meta_features, &train_indices, 0.01, 100);

    let stacked_predictions: Vec<f64> = meta_features
        .iter()
        .map(|x| {
            x.iter()
                .zip(meta_coefficients.iter())
                .map(|(&xi, &bi)| xi * bi)
                .sum::<f64>()
                .exp()
        })
        .collect();

    let c_index = compute_c_index(&time, &event, &stacked_predictions);

    let total_abs: f64 = meta_coefficients.iter().map(|&c| c.abs()).sum();
    let base_model_importance: Vec<f64> = if total_abs > 0.0 {
        meta_coefficients
            .iter()
            .map(|&c| c.abs() / total_abs)
            .collect()
    } else {
        vec![1.0 / n_models as f64; n_models]
    };

    Ok(StackingResult {
        meta_coefficients,
        stacked_predictions,
        c_index,
        base_model_importance,
    })
}
