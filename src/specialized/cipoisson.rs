use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, Gamma, Normal};
#[pyfunction]
pub fn cipoisson_exact(k: u32, time: f64, p: f64) -> PyResult<(f64, f64)> {
    if time <= 0.0 || p <= 0.0 || p >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid input values",
        ));
    }
    let alpha_low = p / 2.0;
    let alpha_high = 1.0 - alpha_low;
    let gamma_low = Gamma::new(k as f64, 1.0).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Error creating Gamma distribution")
    })?;
    let gamma_high = Gamma::new((k + 1) as f64, 1.0).map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Error creating Gamma distribution")
    })?;
    let lower_bound = gamma_low.inverse_cdf(alpha_low);
    let upper_bound = gamma_high.inverse_cdf(alpha_high);
    Ok((lower_bound / time, upper_bound / time))
}
#[pyfunction]
pub fn cipoisson_anscombe(k: u32, time: f64, p: f64) -> PyResult<(f64, f64)> {
    if time <= 0.0 || p <= 0.0 || p >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid input values",
        ));
    }
    let transformed_k = (k as f64 + 3.0 / 8.0).sqrt();
    let z = Normal::new(0.0, 1.0)
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Error creating Normal distribution")
        })?
        .inverse_cdf(p / 2.0);
    let variance: f64 = 1.0 / 4.0;
    let lower_bound = transformed_k - z * (variance.sqrt());
    let upper_bound = transformed_k + z * (variance.sqrt());
    let lower_bound_poisson = (lower_bound.powi(2) - 3.0 / 8.0).max(0.0) / time;
    let upper_bound_poisson = (upper_bound.powi(2) - 3.0 / 8.0) / time;
    Ok((lower_bound_poisson, upper_bound_poisson))
}
#[pyfunction]
pub fn cipoisson(k: u32, time: f64, p: f64, method: String) -> PyResult<(f64, f64)> {
    match method.as_str() {
        "exact" => cipoisson_exact(k, time, p),
        "anscombe" => cipoisson_anscombe(k, time, p),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid method",
        )),
    }
}
