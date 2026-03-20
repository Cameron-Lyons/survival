
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BlendingResult {
    #[pyo3(get)]
    pub blend_weights: Vec<f64>,
    #[pyo3(get)]
    pub blended_predictions: Vec<f64>,
    #[pyo3(get)]
    pub validation_c_index: f64,
}

#[pymethods]
impl BlendingResult {
    fn __repr__(&self) -> String {
        format!(
            "BlendingResult(n_models={}, val_C={:.4})",
            self.blend_weights.len(),
            self.validation_c_index
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    val_time,
    val_event,
    val_predictions,
    test_predictions
))]
pub fn blending_survival(
    val_time: Vec<f64>,
    val_event: Vec<i32>,
    val_predictions: Vec<Vec<f64>>,
    test_predictions: Vec<Vec<f64>>,
) -> PyResult<BlendingResult> {
    let n_val = val_time.len();
    let n_models = val_predictions.len();

    if n_val == 0 || val_event.len() != n_val {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Validation arrays must have the same non-zero length",
        ));
    }
    if n_models == 0 || test_predictions.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Must have same number of models for validation and test",
        ));
    }

    let outcomes: Vec<f64> = val_event.iter().map(|&e| e as f64).collect();
    let blend_weights = nnls_weights(&val_predictions, &outcomes, n_models);

    let n_test = test_predictions[0].len();
    let blended_predictions: Vec<f64> = (0..n_test)
        .into_par_iter()
        .map(|i| {
            (0..n_models)
                .map(|m| blend_weights[m] * test_predictions[m][i])
                .sum()
        })
        .collect();

    let val_blended: Vec<f64> = (0..n_val)
        .into_par_iter()
        .map(|i| {
            (0..n_models)
                .map(|m| blend_weights[m] * val_predictions[m][i])
                .sum()
        })
        .collect();

    let validation_c_index = compute_c_index(&val_time, &val_event, &val_blended);

    Ok(BlendingResult {
        blend_weights,
        blended_predictions,
        validation_c_index,
    })
}

