use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::{DIVISION_FLOOR, exp_ci_bounds_95};
use crate::internal::statistical::{chi2_cdf, ln_gamma, normal_cdf};

fn covariate_matrix_or_intercept(covariates: Vec<f64>, n: usize) -> (usize, Vec<f64>) {
    if covariates.is_empty() {
        (1, vec![1.0; n])
    } else {
        (covariates.len() / n, covariates)
    }
}

fn sorted_unique_i32(values: &[i32]) -> Vec<i32> {
    let mut unique_values = values.to_vec();
    unique_values.sort_unstable();
    unique_values.dedup();
    unique_values
}

fn index_by_i32(values: &[i32]) -> std::collections::HashMap<i32, usize> {
    values
        .iter()
        .enumerate()
        .map(|(idx, &value)| (value, idx))
        .collect()
}

fn validate_recurrent_lengths(n: usize, lengths: &[usize]) -> PyResult<()> {
    if lengths.iter().any(|&len| len != n) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }
    Ok(())
}

include!("pwp.rs");
include!("wlw.rs");
include!("negative_binomial.rs");
include!("anderson_gill.rs");
include!("tests.rs");
