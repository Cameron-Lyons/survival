use crate::constants::PARALLEL_THRESHOLD_MEDIUM;
use crate::utilities::validation::{ValidationError, validate_length};
use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[inline]
pub fn apply_deltas_add<F>(
    indices: &[usize],
    nvar: usize,
    matrix: &mut Array2<f64>,
    compute_deltas: F,
) where
    F: Fn(usize) -> Vec<f64> + Sync + Send,
{
    if indices.len() > PARALLEL_THRESHOLD_MEDIUM {
        let updates: Vec<(usize, Vec<f64>)> = indices
            .par_iter()
            .map(|&idx| (idx, compute_deltas(idx)))
            .collect();
        for (idx, deltas) in updates {
            for j in 0..nvar {
                matrix[[j, idx]] += deltas[j];
            }
        }
    } else {
        for &idx in indices {
            let deltas = compute_deltas(idx);
            for j in 0..nvar {
                matrix[[j, idx]] += deltas[j];
            }
        }
    }
}

#[inline]
pub fn apply_deltas_set<F>(
    indices: &[usize],
    nvar: usize,
    matrix: &mut Array2<f64>,
    compute_deltas: F,
) where
    F: Fn(usize) -> Vec<f64> + Sync + Send,
{
    if indices.len() > PARALLEL_THRESHOLD_MEDIUM {
        let updates: Vec<(usize, Vec<f64>)> = indices
            .par_iter()
            .map(|&idx| (idx, compute_deltas(idx)))
            .collect();
        for (idx, deltas) in updates {
            for j in 0..nvar {
                matrix[[j, idx]] = deltas[j];
            }
        }
    } else {
        for &idx in indices {
            let deltas = compute_deltas(idx);
            for j in 0..nvar {
                matrix[[j, idx]] = deltas[j];
            }
        }
    }
}

fn validation_err_to_pyresult<T>(result: Result<T, ValidationError>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(e.to_string()))
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
        return Err(PyValueError::new_err("No observations provided"));
    }
    validation_err_to_pyresult(validate_length(3 * n, time_data_len, "time_data"))?;
    if !covariates_len.is_multiple_of(n) {
        return Err(PyValueError::new_err(
            "Covariates length should be divisible by number of observations",
        ));
    }
    validation_err_to_pyresult(validate_length(n, strata_len, "strata"))?;
    validation_err_to_pyresult(validate_length(n, score_len, "score"))?;
    validation_err_to_pyresult(validate_length(n, weights_len, "weights"))?;
    Ok(())
}
pub fn compute_summary_stats(residuals: &[f64], n: usize, nvar: usize) -> Vec<f64> {
    if n > PARALLEL_THRESHOLD_MEDIUM && nvar > 1 {
        (0..nvar)
            .into_par_iter()
            .flat_map(|i| {
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
                vec![mean, variance]
            })
            .collect()
    } else {
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
