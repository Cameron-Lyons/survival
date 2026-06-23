use crate::internal::statistical::{gamma_inverse_cdf, normal_inverse_cdf};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_cipoisson_inputs(time: f64, p: f64) -> PyResult<()> {
    if !time.is_finite() || time <= 0.0 {
        return Err(PyValueError::new_err("time must be positive and finite"));
    }
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(PyValueError::new_err(
            "p must be a confidence level between 0 and 1",
        ));
    }
    Ok(())
}

fn parse_cipoisson_method(method: &str) -> PyResult<&'static str> {
    match method {
        value if "exact".starts_with(value) && !value.is_empty() => Ok("exact"),
        value if "anscombe".starts_with(value) && !value.is_empty() => Ok("anscombe"),
        _ => Err(PyValueError::new_err(
            "method must uniquely match 'exact' or 'anscombe'",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (k, time=1.0, p=0.95))]
pub fn cipoisson_exact(k: u32, time: f64, p: f64) -> PyResult<(f64, f64)> {
    validate_cipoisson_inputs(time, p)?;
    let alpha = (1.0 - p) / 2.0;

    let lower_bound = if k == 0 {
        0.0
    } else {
        gamma_inverse_cdf(alpha, k as f64)
    };

    let upper_bound = gamma_inverse_cdf(1.0 - alpha, (k + 1) as f64);

    Ok((lower_bound / time, upper_bound / time))
}

#[pyfunction]
#[pyo3(signature = (k, time=1.0, p=0.95))]
pub fn cipoisson_anscombe(k: u32, time: f64, p: f64) -> PyResult<(f64, f64)> {
    validate_cipoisson_inputs(time, p)?;
    let alpha = (1.0 - p) / 2.0;
    let z = normal_inverse_cdf(alpha);
    let count = k as f64;
    let lower_bound = ((count - 1.0 / 8.0).sqrt() + z / 2.0).powi(2);
    let upper_bound = ((count + 7.0 / 8.0).sqrt() - z / 2.0).powi(2);
    Ok((lower_bound / time, upper_bound / time))
}

#[pyfunction]
#[pyo3(
    signature = (k, time=1.0, p=0.95, method="exact".to_string()),
    text_signature = "(k, time=1.0, p=0.95, method='exact')"
)]
pub fn cipoisson(k: u32, time: f64, p: f64, method: String) -> PyResult<(f64, f64)> {
    match parse_cipoisson_method(&method)? {
        "exact" => cipoisson_exact(k, time, p),
        "anscombe" => cipoisson_anscombe(k, time, p),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    #[test]
    fn cipoisson_exact_matches_r_survival_reference() {
        let (lower, upper) = cipoisson_exact(5, 10.0, 0.95).unwrap();
        assert_close(lower, 0.1623486, 1e-6);
        assert_close(upper, 1.1668332, 1e-6);

        let (lower, upper) = cipoisson_exact(20, 4.0, 0.90).unwrap();
        assert_close(lower, 3.313663, 1e-6);
        assert_close(upper, 7.265505, 1e-6);
    }

    #[test]
    fn cipoisson_anscombe_matches_r_survival_reference() {
        let (lower, upper) = cipoisson_anscombe(5, 10.0, 0.95).unwrap();
        assert_close(lower, 0.1507881, 1e-6);
        assert_close(upper, 1.1586004, 1e-6);

        let (lower, upper) = cipoisson_anscombe(20, 4.0, 0.90).unwrap();
        assert_close(lower, 3.304600, 1e-6);
        assert_close(upper, 7.266646, 1e-6);
    }

    #[test]
    fn cipoisson_uses_r_style_method_prefixes_and_defaults() {
        let exact = cipoisson(5, 10.0, 0.95, "e".to_string()).unwrap();
        assert_eq!(exact, cipoisson_exact(5, 10.0, 0.95).unwrap());

        let anscombe = cipoisson(5, 10.0, 0.95, "a".to_string()).unwrap();
        assert_eq!(anscombe, cipoisson_anscombe(5, 10.0, 0.95).unwrap());

        let default = cipoisson(5, 1.0, 0.95, "exact".to_string()).unwrap();
        assert_eq!(default, cipoisson_exact(5, 1.0, 0.95).unwrap());
    }

    #[test]
    fn cipoisson_rejects_malformed_inputs() {
        initialize_python();

        let err = cipoisson_exact(1, f64::NAN, 0.95).unwrap_err();
        assert!(err.to_string().contains("time must be positive and finite"));

        let err = cipoisson_anscombe(1, 1.0, 1.0).unwrap_err();
        assert!(
            err.to_string()
                .contains("p must be a confidence level between 0 and 1")
        );

        let err = cipoisson(1, 1.0, 0.95, "".to_string()).unwrap_err();
        assert!(
            err.to_string()
                .contains("method must uniquely match 'exact' or 'anscombe'")
        );
    }
}
