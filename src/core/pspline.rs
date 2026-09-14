//! The P-spline basis of R's `pspline()` term (survival 3.8-12,
//! `pspline.R`): `nterm + degree` B-splines on equally spaced knots that
//! extend `degree` steps beyond each boundary knot, evaluated with
//! `spline.des`; beyond the boundary the basis is continued linearly,
//! `f(edge) + (x - edge) f'(edge)`.  The difference penalty, the smoothing
//! parameter search and the `combine` argument live at the R level
//! (`pspline.R`, `coxpenal.fit`) and are not ported here.

use crate::core::bspline::spline_design;
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

/// The full (intercept-included) P-spline basis and its knots.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct PsplineBasis {
    /// `n x (nterm + degree)` basis rows; a missing `x` gives a `NaN` row.
    #[pyo3(get)]
    pub basis: Vec<Vec<f64>>,
    /// The `nterm + 2 * degree + 1` equally spaced knots.
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub nterm: usize,
    #[pyo3(get)]
    pub degree: usize,
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
}

/// `pspline(x, nterm, degree, Boundary.knots)` without the penalty
/// attributes.  `boundary_knots` may be equal only when every observed `x`
/// equals it (a constant covariate), in which case the middle basis
/// function is 1 as in R.
pub fn pspline_basis(
    x: &[f64],
    nterm: usize,
    degree: usize,
    boundary_knots: (f64, f64),
) -> SurvivalResult<PsplineBasis> {
    if nterm < 3 {
        return Err(SurvivalError::invalid_input("Too few basis functions"));
    }
    if degree == 0 {
        return Err(SurvivalError::invalid_input("degree must be positive"));
    }
    let (lower, upper) = boundary_knots;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(SurvivalError::invalid_input(
            "Invalid values for Boundary.knots",
        ));
    }
    if let Some(value) = x.iter().find(|value| value.is_infinite()) {
        return Err(SurvivalError::invalid_input(format!(
            "x contains infinite value {value}"
        )));
    }
    let n_basis = nterm + degree;
    let dx = (upper - lower) / nterm as f64;
    let knots: Vec<f64> = (0..nterm + degree)
        .map(|idx| lower + dx * (idx as f64 - degree as f64))
        .chain((0..=degree).map(|idx| upper + dx * idx as f64))
        .collect();

    let nan_row = vec![f64::NAN; n_basis];
    if lower == upper {
        if x.iter().any(|value| !value.is_nan() && *value != lower) {
            return Err(SurvivalError::invalid_input(
                "zero-width boundary_knots require all observed x values to match the boundary",
            ));
        }
        let mut constant = vec![0.0; n_basis];
        constant[nterm - 1] = 1.0;
        let basis = x
            .iter()
            .map(|value| {
                if value.is_nan() {
                    nan_row.clone()
                } else {
                    constant.clone()
                }
            })
            .collect();
        return Ok(PsplineBasis {
            basis,
            knots,
            nterm,
            degree,
            boundary_knots,
        });
    }

    let order = degree + 1;
    let edge = |pivot: f64| spline_design(&knots, &[pivot, pivot], order, &[0, 1]);
    let left = edge(lower)?;
    let right = edge(upper)?;
    let inside: Vec<f64> = x
        .iter()
        .copied()
        .filter(|value| *value >= lower && *value <= upper)
        .collect();
    let inside_rows = spline_design(&knots, &inside, order, &[0])?;
    let mut inside_cursor = 0;
    let basis = x
        .iter()
        .map(|&value| {
            if value.is_nan() {
                nan_row.clone()
            } else if value < lower || value > upper {
                let (tt, pivot) = if value < lower {
                    (&left, lower)
                } else {
                    (&right, upper)
                };
                (0..n_basis)
                    .map(|col| tt[[0, col]] + (value - pivot) * tt[[1, col]])
                    .collect()
            } else {
                let row = inside_rows.row(inside_cursor).to_vec();
                inside_cursor += 1;
                row
            }
        })
        .collect();
    Ok(PsplineBasis {
        basis,
        knots,
        nterm,
        degree,
        boundary_knots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_at_boundary_and_extrapolation_rows() {
        let out = pspline_basis(&[0.0, 1.0, 5.0, 6.0], 8, 3, (1.0, 5.0)).unwrap();
        assert_eq!(out.basis.len(), 4);
        assert_eq!(out.basis[0].len(), 11);
        assert_eq!(out.knots.len(), 15);
        let expected_left = [7.0 / 6.0, 2.0 / 3.0, -5.0 / 6.0];
        for (actual, expected) in out.basis[0][..3].iter().zip(expected_left) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let expected_right = [-5.0 / 6.0, 2.0 / 3.0, 7.0 / 6.0];
        for (actual, expected) in out.basis[3][8..].iter().zip(expected_right) {
            assert!((actual - expected).abs() < 1e-12);
        }
        // At the boundary knots the basis is the B-spline value itself.
        assert!((out.basis[1][0] - 1.0 / 6.0).abs() < 1e-12);
        assert!((out.basis[1][1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((out.basis[1][2] - 1.0 / 6.0).abs() < 1e-12);
        for row in &out.basis[1..3] {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn preserves_missing_and_constant_rows() {
        let out = pspline_basis(&[2.0, f64::NAN, 2.0], 5, 3, (2.0, 2.0)).unwrap();
        assert_eq!(out.knots, vec![2.0; 12]);
        assert_eq!(out.basis[0], vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(out.basis[1].iter().all(|value| value.is_nan()));
        assert_eq!(out.basis[2], out.basis[0]);

        let out = pspline_basis(&[1.5, f64::NAN, 4.0], 4, 2, (1.0, 5.0)).unwrap();
        assert!(out.basis[1].iter().all(|value| value.is_nan()));
        assert!((out.basis[0].iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((out.basis[2].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_inputs() {
        assert!(pspline_basis(&[1.0], 2, 3, (0.0, 1.0)).is_err());
        assert!(pspline_basis(&[1.0], 3, 0, (0.0, 1.0)).is_err());
        assert!(pspline_basis(&[f64::INFINITY], 3, 1, (0.0, 1.0)).is_err());
        assert!(pspline_basis(&[1.0], 3, 1, (1.0, 0.0)).is_err());
        assert!(pspline_basis(&[2.0], 3, 1, (1.0, 1.0)).is_err());
    }
}
