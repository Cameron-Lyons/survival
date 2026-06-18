use super::cox_baseline::compute_tied_baseline_summaries;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Backward-compatible aggregate survival helper kept for the public crate API.
#[pyfunction]
pub fn agsurv5(
    n: usize,
    nvar: usize,
    dd: Vec<i32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    xsum: Vec<f64>,
    xsum2: Vec<f64>,
) -> PyResult<Py<PyDict>> {
    compute_tied_baseline_summaries(n, nvar, dd, x1, x2, xsum, xsum2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agsurv5_rejects_malformed_inputs_without_panicking() {
        let err = agsurv5(1, 1, vec![0], vec![1.0], vec![0.0], vec![1.0], vec![0.0])
            .expect_err("zero event count should fail");

        assert!(err.to_string().contains("positive event counts"));
    }
}
