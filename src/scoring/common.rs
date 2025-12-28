use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Validates common scoring input parameters for agscore functions
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

    if time_data_len != 3 * n {
        return Err(PyRuntimeError::new_err(
            "Time data should have 3*n elements (start, stop, event)",
        ));
    }

    if !covariates_len.is_multiple_of(n) {
        return Err(PyRuntimeError::new_err(
            "Covariates length should be divisible by number of observations",
        ));
    }

    if strata_len != n {
        return Err(PyRuntimeError::new_err(
            "Strata length does not match observations",
        ));
    }

    if score_len != n {
        return Err(PyRuntimeError::new_err(
            "Score length does not match observations",
        ));
    }

    if weights_len != n {
        return Err(PyRuntimeError::new_err(
            "Weights length does not match observations",
        ));
    }

    Ok(())
}

/// Computes summary statistics (mean and variance) for each variable's residuals
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

/// Builds a score result PyDict with residuals and summary statistics
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
