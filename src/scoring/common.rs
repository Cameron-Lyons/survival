use crate::utilities::validation::{validate_length, ValidationError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn validation_err_to_pyresult<T>(result: Result<T, ValidationError>) -> PyResult<T> {
    result.map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub fn validate_scoring_inputs(
    n: usize,
    time_data_len: usize,
    covariates_len: usize,
    strata_len: usize,
    score_len: usize,
    weights_len: usize,
) -> PyResult<()> {
    if n == 0 {
        return Err(PyRuntimeError::new_err("No observations provided"));
    }
    validation_err_to_pyresult(validate_length(3 * n, time_data_len, "time_data"))?;
    if !covariates_len.is_multiple_of(n) {
        return Err(PyRuntimeError::new_err(
            "Covariates length should be divisible by number of observations",
        ));
    }
    validation_err_to_pyresult(validate_length(n, strata_len, "strata"))?;
    validation_err_to_pyresult(validate_length(n, score_len, "score"))?;
    validation_err_to_pyresult(validate_length(n, weights_len, "weights"))?;
    Ok(())
}
pub fn compute_summary_stats(residuals: &[f64], n: usize, nvar: usize) -> Vec<f64> {
    let mut summary_stats = Vec::with_capacity(nvar * 2);
    for i in 0..nvar {
        let start_idx = i * n;
        let end_idx = (i + 1) * n;
        let var_residuals = &residuals[start_idx..end_idx];
        let mean = var_residuals.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            var_residuals
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            0.0
        };
        summary_stats.push(mean);
        summary_stats.push(variance);
    }
    summary_stats
}
pub fn build_score_result(
    py: Python<'_>,
    residuals: Vec<f64>,
    n: usize,
    nvar: usize,
    method: i32,
) -> PyResult<Py<PyDict>> {
    let summary_stats = compute_summary_stats(&residuals, n, nvar);
    let dict = PyDict::new(py);
    dict.set_item("residuals", residuals)?;
    dict.set_item("n_observations", n)?;
    dict.set_item("n_variables", nvar)?;
    dict.set_item("method", if method == 0 { "breslow" } else { "efron" })?;
    dict.set_item("summary_stats", summary_stats)?;
    Ok(dict.into())
}
