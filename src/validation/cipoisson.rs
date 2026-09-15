//! Confidence limits for a Poisson rate.
//!
//! Port of R survival `R/cipoisson.R`: the exact (gamma quantile) and
//! Anscombe limits, vectorised over `k`, `time` and `p` with R's recycling
//! rule.  A non-positive `time` yields `NaN` limits (R returns `NA`; the
//! `summary.pyears` code calls this with `time = 0`).

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::{qgamma, qnorm};
use pyo3::prelude::*;

/// Which approximation to use for the limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipoissonMethod {
    /// Gamma-quantile limits (`qgamma(p, k)`, `qgamma(1 - p, k + 1)`).
    Exact,
    /// Anscombe's square-root transformation.
    Anscombe,
}

impl CipoissonMethod {
    /// R's `match.arg`: any unambiguous prefix of `"exact"` or `"anscombe"`.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            value if !value.is_empty() && "exact".starts_with(value) => Ok(Self::Exact),
            value if !value.is_empty() && "anscombe".starts_with(value) => Ok(Self::Anscombe),
            _ => Err(SurvivalError::invalid_input(
                "method must uniquely match 'exact' or 'anscombe'",
            )),
        }
    }
}

/// Lower and upper limits, one pair per recycled input.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct CipoissonResult {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

fn recycled(values: &[f64], length: usize, name: &str) -> SurvivalResult<Vec<f64>> {
    if values.is_empty() {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must not be empty"
        )));
    }
    Ok((0..length).map(|i| values[i % values.len()]).collect())
}

/// Confidence limits for Poisson counts `k` observed over `time`, at
/// confidence level `p` (R `cipoisson(k, time, p, method)`).
pub fn cipoisson(
    k: &[f64],
    time: &[f64],
    p: &[f64],
    method: CipoissonMethod,
) -> SurvivalResult<CipoissonResult> {
    let n = k.len().max(time.len()).max(p.len());
    let k = recycled(k, n, "k")?;
    let time = recycled(time, n, "time")?;
    let p = recycled(p, n, "p")?;
    for (index, &count) in k.iter().enumerate() {
        if count.is_nan() || count < 0.0 {
            return Err(SurvivalError::invalid_input(format!(
                "k[{index}] must be a non-negative count"
            )));
        }
    }
    // R lets a missing p through (the quantiles come back NA); anything else
    // outside [0, 1] is rejected up front rather than by qgamma's NaN.
    for (index, &level) in p.iter().enumerate() {
        if !level.is_nan() && (!level.is_finite() || !(0.0..=1.0).contains(&level)) {
            return Err(SurvivalError::invalid_input(format!(
                "p[{index}] must be a confidence level between 0 and 1 inclusive"
            )));
        }
    }
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for i in 0..n {
        let alpha = (1.0 - p[i]) / 2.0;
        let (low, high) = match method {
            CipoissonMethod::Exact => {
                let low = if k[i] == 0.0 {
                    0.0
                } else {
                    qgamma(alpha, k[i], 1.0, true, false)
                };
                (low, qgamma(1.0 - alpha, k[i] + 1.0, 1.0, true, false))
            }
            CipoissonMethod::Anscombe => {
                let z = qnorm(alpha, true, false);
                (
                    ((k[i] - 1.0 / 8.0).sqrt() + z / 2.0).powi(2),
                    ((k[i] + 7.0 / 8.0).sqrt() - z / 2.0).powi(2),
                )
            }
        };
        if time[i].is_nan() || time[i] <= 0.0 {
            lower.push(f64::NAN);
            upper.push(f64::NAN);
        } else {
            lower.push(low / time[i]);
            upper.push(high / time[i]);
        }
    }
    Ok(CipoissonResult { lower, upper })
}

/// Python entry point: `cipoisson(k, time=None, p=None, method="exact")`;
/// `time` defaults to 1 and `p` to 0.95, and `k`, `time`, `p` recycle like
/// R vectors.
#[pyfunction(name = "cipoisson")]
#[pyo3(signature = (k, time=None, p=None, method="exact"))]
pub fn cipoisson_py(
    k: Vec<f64>,
    time: Option<Vec<f64>>,
    p: Option<Vec<f64>>,
    method: &str,
) -> PyResult<CipoissonResult> {
    let time = time.unwrap_or_else(|| vec![1.0]);
    let p = p.unwrap_or_else(|| vec![0.95]);
    Ok(cipoisson(&k, &time, &p, CipoissonMethod::parse(method)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    #[test]
    fn exact_limits_match_r() {
        let result = cipoisson(&[5.0], &[10.0], &[0.95], CipoissonMethod::Exact).unwrap();
        assert_close(result.lower[0], 0.1623486, 1e-6);
        assert_close(result.upper[0], 1.1668332, 1e-6);
        let result = cipoisson(&[20.0], &[4.0], &[0.90], CipoissonMethod::Exact).unwrap();
        assert_close(result.lower[0], 3.313663, 1e-6);
        assert_close(result.upper[0], 7.265505, 1e-6);
    }

    #[test]
    fn anscombe_limits_match_r() {
        let result = cipoisson(&[5.0], &[10.0], &[0.95], CipoissonMethod::Anscombe).unwrap();
        assert_close(result.lower[0], 0.1507881, 1e-6);
        assert_close(result.upper[0], 1.1586004, 1e-6);
        let result = cipoisson(&[20.0], &[4.0], &[0.90], CipoissonMethod::Anscombe).unwrap();
        assert_close(result.lower[0], 3.304600, 1e-6);
        assert_close(result.upper[0], 7.266646, 1e-6);
    }

    #[test]
    fn recycles_arguments_and_flags_non_positive_time() {
        let result = cipoisson(
            &[0.0, 5.0, 20.0],
            &[1.0, 0.0],
            &[0.95],
            CipoissonMethod::Exact,
        )
        .unwrap();
        assert_eq!(result.lower.len(), 3);
        assert_eq!(result.lower[0], 0.0);
        assert!(result.lower[1].is_nan() && result.upper[1].is_nan());
        assert_close(result.lower[2], 12.21652, 1e-4);
    }

    #[test]
    fn method_prefixes_follow_match_arg() {
        assert_eq!(CipoissonMethod::parse("e").unwrap(), CipoissonMethod::Exact);
        assert_eq!(
            CipoissonMethod::parse("ans").unwrap(),
            CipoissonMethod::Anscombe
        );
        assert!(CipoissonMethod::parse("").is_err());
        assert!(CipoissonMethod::parse("bogus").is_err());
    }

    #[test]
    fn rejects_bad_counts_and_levels() {
        assert!(cipoisson(&[-1.0], &[1.0], &[0.95], CipoissonMethod::Exact).is_err());
        assert!(cipoisson(&[1.0], &[1.0], &[1.5], CipoissonMethod::Exact).is_err());
        assert!(cipoisson(&[], &[1.0], &[0.95], CipoissonMethod::Exact).is_err());
    }

    #[test]
    fn boundary_levels_give_degenerate_limits() {
        let result = cipoisson(&[1.2], &[1.0], &[1.0], CipoissonMethod::Exact).unwrap();
        assert_eq!(result.lower[0], 0.0);
        assert_eq!(result.upper[0], f64::INFINITY);
        let result = cipoisson(&[1.2], &[2.0], &[0.0], CipoissonMethod::Exact).unwrap();
        assert_close(result.lower[0], 0.443968106737396, 1e-10);
        assert_close(result.upper[0], 0.938570591679505, 1e-10);
    }
}
