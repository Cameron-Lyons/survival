//! Natural-spline curves and standard errors from R's `plot.cox.zph`.
//! Knot placement includes prediction times as well as observed event times.
//! Terms sharing an observed-row mask reuse their QR factorization and leverage.

use std::collections::HashMap;

use crate::core::natural_spline::ns;
use crate::error::{SurvivalError, SurvivalResult};
#[cfg(feature = "python")]
use crate::internal::numpy_utils::{FloatMatrix, FloatVec};
use crate::internal::qr::LinpackLeastSquares;
use ndarray::{ArrayView2, s};
use pyo3::prelude::*;

/// Numerical diagnostic curves in transformed time and coefficient units.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxZphSmooth {
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// Prediction-time by term matrix. Singular terms contain NaNs.
    #[pyo3(get)]
    pub y: Vec<Vec<f64>>,
    /// One standard error, before R's factor of two for plotted bands.
    #[pyo3(get)]
    pub std_err: Option<Vec<Vec<f64>>>,
    /// Zero-based columns skipped because the spline design is singular.
    #[pyo3(get)]
    pub skipped: Vec<usize>,
}

/// Smooth scaled Schoenfeld residuals with R's natural cubic spline.
///
/// `x` contains transformed death times, `y` is death-by-term, and `variance`
/// contains the diagonal of the `cox.zph` residual variance. Missing residuals
/// are NaN and are excluded separately for each term. No residual variance is
/// estimated from the spline fit. `df >= 2` and `nsmo >= 2` are required.
pub fn cox_zph_smooth(
    x: &[f64],
    y: ArrayView2<'_, f64>,
    variance: &[f64],
    df: usize,
    nsmo: usize,
    se: bool,
) -> SurvivalResult<CoxZphSmooth> {
    let invalid = SurvivalError::invalid_input;
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(invalid("x must contain finite transformed event times"));
    }
    let (n, p) = y.dim();
    if n != x.len() || p == 0 || variance.len() != p {
        return Err(invalid(
            "y rows must match x and its columns must match variance",
        ));
    }
    if y.iter().any(|v| v.is_infinite()) {
        return Err(invalid("residuals must be finite or NaN"));
    }
    if se && variance.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(invalid("variance must be finite and nonnegative"));
    }
    if df < 2 || nsmo < 2 {
        return Err(invalid("df and nsmo must both be at least 2"));
    }
    let low = x.iter().copied().fold(f64::INFINITY, f64::min);
    let high = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if low >= high {
        return Err(invalid(
            "smoothing requires distinct transformed event times",
        ));
    }
    let mut grid: Vec<f64> = (0..nsmo)
        .map(|i| low + (high - low) * (i as f64 / (nsmo - 1) as f64))
        .collect();
    grid[nsmo - 1] = high;
    let all: Vec<f64> = grid.iter().chain(x).copied().collect();
    let basis = ns(&all, Some(df), None, true, (low, high))?.values;
    let predicted = basis.slice(s![..nsmo, ..]);
    let observed = basis.slice(s![nsmo.., ..]);
    let mut fits = HashMap::new();
    let mut result = CoxZphSmooth {
        x: grid,
        y: vec![vec![f64::NAN; p]; nsmo],
        std_err: se.then(|| vec![vec![f64::NAN; p]; nsmo]),
        skipped: Vec::new(),
    };
    for column in 0..p {
        let rows: Vec<usize> = (0..n).filter(|&row| !y[[row, column]].is_nan()).collect();
        let (qr, leverage) = fits.entry(rows.clone()).or_insert_with(|| {
            let columns = (0..df)
                .map(|col| rows.iter().map(|&row| observed[[row, col]]).collect())
                .collect();
            let qr = LinpackLeastSquares::new(columns, rows.len());
            let leverage: Vec<f64> = if se && qr.rank() == df {
                predicted
                    .rows()
                    .into_iter()
                    .map(|row| {
                        qr.prediction_variance(row.as_slice().expect("contiguous basis row"))
                            .expect("full-rank spline")
                    })
                    .collect()
            } else {
                Vec::new()
            };
            (qr, leverage)
        });
        if qr.rank() < df {
            result.skipped.push(column);
            continue;
        }
        let coef = qr.coefficients(&rows.iter().map(|&row| y[[row, column]]).collect::<Vec<_>>());
        for row in 0..nsmo {
            result.y[row][column] = predicted
                .row(row)
                .iter()
                .zip(&coef)
                .map(|(a, b)| a * b)
                .sum();
            if let Some(errors) = &mut result.std_err {
                errors[row][column] = (variance[column] * leverage[row]).sqrt();
            }
        }
    }
    Ok(result)
}

#[cfg(feature = "python")]
#[pyfunction(name = "cox_zph_smooth")]
#[pyo3(signature = (x, y, variance, df=4, nsmo=40, se=true))]
pub fn cox_zph_smooth_py(
    py: Python<'_>,
    x: FloatVec,
    y: FloatMatrix,
    variance: FloatVec,
    df: usize,
    nsmo: usize,
    se: bool,
) -> PyResult<CoxZphSmooth> {
    Ok(py.detach(|| cox_zph_smooth(&x, y.view(), &variance, df, nsmo, se))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_spline_recovers_a_line_and_its_analytic_variance() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = array![[1.0, 2.0], [3.0, 6.0], [5.0, 10.0], [7.0, 14.0]];
        let result = cox_zph_smooth(&x, y.view(), &[1.0, 4.0], 2, 9, true).unwrap();
        for (i, &value) in result.x.iter().enumerate() {
            assert!((result.y[i][0] - (1.0 + 2.0 * value)).abs() < 1e-12);
            assert!((result.y[i][1] - 2.0 * result.y[i][0]).abs() < 1e-12);
            let expected = (0.25 + (value - 1.5).powi(2) / 5.0).sqrt();
            let error = &result.std_err.as_ref().unwrap()[i];
            assert!((error[0] - expected).abs() < 1e-12);
            assert!((error[1] - 2.0 * expected).abs() < 1e-12);
        }
    }

    #[test]
    fn missing_residuals_use_their_own_design_and_singular_terms_are_skipped() {
        let y = array![
            [1.0, f64::NAN, f64::NAN],
            [2.0, 4.0, 2.0],
            [3.0, 6.0, f64::NAN],
            [4.0, 8.0, f64::NAN]
        ];
        let result =
            cox_zph_smooth(&[0.0, 1.0, 2.0, 3.0], y.view(), &[1.0; 3], 2, 4, true).unwrap();
        assert_eq!(result.skipped, vec![2]);
        assert!(result.y.iter().all(|row| row[2].is_nan()));
        assert!((result.y[0][1] - 2.0).abs() < 1e-12);
        assert!(result.std_err.unwrap()[0][1] > 0.7_f64.sqrt());
        assert!(cox_zph_smooth(&[0.0; 4], y.view(), &[1.0; 3], 2, 4, false).is_err());
        assert!(cox_zph_smooth(&[0.0, 1.0, 2.0, 3.0], y.view(), &[1.0; 3], 1, 4, true).is_err());
    }
}
