use crate::core::bspline::spline_design;
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

#[doc = " The full (intercept-included) P-spline basis and its knots."]
#[derive(Debug, Clone)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct PsplineBasis {
    #[doc = " `n x (nterm + degree)` basis rows; a missing `x` gives a `NaN` row."]
    #[pyo3(get)]
    pub basis: Vec<Vec<f64>>,
    #[doc = " The `nterm + 2 * degree + 1` equally spaced knots."]
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub nterm: usize,
    #[pyo3(get)]
    pub degree: usize,
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
}

#[cfg(feature = "python")]
#[pymethods]
impl PsplineBasis {
    #[doc = " Independent writable NumPy snapshots: the `n x (nterm + degree)` basis"]
    #[doc = " matrix and the knots."]
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use numpy::IntoPyArray;
        let width = self.nterm + self.degree;
        let flat: Vec<f64> = self.basis.iter().flatten().copied().collect();
        let basis = ndarray::Array2::from_shape_vec((self.basis.len(), width), flat)
            .expect("every basis row has nterm + degree columns");
        let result = pyo3::types::PyDict::new(py);
        result.set_item("basis", basis.into_pyarray(py))?;
        result.set_item("knots", self.knots.clone().into_pyarray(py))?;
        Ok(result)
    }
}

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
    fn every_row_has_one_value_per_basis_function() {
        for (x, nterm, degree, boundary) in [
            (vec![0.0, 1.0, f64::NAN, 5.0, 6.0], 8, 3, (1.0, 5.0)),
            (vec![2.0, f64::NAN], 5, 3, (2.0, 2.0)),
            (vec![1.5, 4.0], 4, 1, (1.0, 5.0)),
        ] {
            let out = pspline_basis(&x, nterm, degree, boundary).unwrap();
            assert_eq!(out.basis.len(), x.len());
            assert!(out.basis.iter().all(|row| row.len() == nterm + degree));
        }
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
