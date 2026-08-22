use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

type PolyBasis = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_degree(degree: usize) -> PyResult<()> {
    if degree == 0 {
        return Err(value_error("degree must be at least 1"));
    }
    Ok(())
}

fn validate_orthogonal_coefficients(degree: usize, alpha: &[f64], norm2: &[f64]) -> PyResult<()> {
    if alpha.len() != degree {
        return Err(value_error(format!(
            "alpha length ({}) must match degree ({degree})",
            alpha.len()
        )));
    }
    if norm2.len() != degree + 2 {
        return Err(value_error(format!(
            "norm2 length ({}) must equal degree + 2 ({})",
            norm2.len(),
            degree + 2
        )));
    }
    if alpha.iter().any(|value| !value.is_finite()) {
        return Err(value_error("alpha must contain only finite values"));
    }
    if norm2
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(value_error(
            "norm2 must contain only finite positive values",
        ));
    }
    Ok(())
}

fn raw_polynomial_rows(x: &[f64], degree: usize) -> PyResult<Vec<Vec<f64>>> {
    x.len()
        .checked_mul(degree)
        .ok_or_else(|| value_error("polynomial basis dimensions are too large"))?;
    Ok(x.par_iter()
        .map(|&value| {
            let mut row = Vec::with_capacity(degree);
            let mut power = value;
            for _ in 0..degree {
                row.push(power);
                power *= value;
            }
            row
        })
        .collect())
}

fn unique_value_count(x: &[f64]) -> usize {
    let mut values = x.to_vec();
    values.sort_by(f64::total_cmp);
    values.dedup_by(|left, right| *left == *right);
    values.len()
}

fn fit_orthogonal_coefficients(x: &[f64], degree: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if x.iter().any(|value| !value.is_finite()) {
        return Err(value_error(
            "missing or infinite values are not allowed in orthogonal poly",
        ));
    }
    if degree >= unique_value_count(x) {
        return Err(value_error(
            "degree must be less than the number of unique points",
        ));
    }

    let mut alpha = Vec::with_capacity(degree);
    let mut norm2 = Vec::with_capacity(degree + 2);
    norm2.push(1.0);
    norm2.push(x.len() as f64);

    let mut previous_previous = vec![0.0; x.len()];
    let mut previous = vec![1.0; x.len()];
    for column in 0..degree {
        let previous_norm = norm2[column + 1];
        let weighted_sum = x
            .iter()
            .zip(&previous)
            .map(|(&value, &basis)| value * basis * basis)
            .sum::<f64>();
        let center = weighted_sum / previous_norm;
        if !center.is_finite() {
            return Err(value_error(
                "orthogonal polynomial coefficients are not finite",
            ));
        }
        alpha.push(center);

        let ratio = if column == 0 {
            0.0
        } else {
            norm2[column + 1] / norm2[column]
        };
        let current: Vec<f64> = x
            .iter()
            .zip(&previous)
            .zip(&previous_previous)
            .map(|((&value, &prior), &older)| (value - center) * prior - ratio * older)
            .collect();
        let current_norm = current.iter().map(|value| value * value).sum::<f64>();
        if !current_norm.is_finite() || current_norm <= 0.0 {
            return Err(value_error(
                "degree must be less than the numerical rank of the input",
            ));
        }
        norm2.push(current_norm);
        previous_previous = previous;
        previous = current;
    }
    Ok((alpha, norm2))
}

fn orthogonal_polynomial_rows(
    x: &[f64],
    degree: usize,
    alpha: &[f64],
    norm2: &[f64],
) -> PyResult<Vec<Vec<f64>>> {
    validate_orthogonal_coefficients(degree, alpha, norm2)?;
    x.len()
        .checked_mul(degree)
        .ok_or_else(|| value_error("polynomial basis dimensions are too large"))?;

    Ok(x.par_iter()
        .map(|&value| {
            if value.is_nan() {
                return vec![f64::NAN; degree];
            }
            let mut row = Vec::with_capacity(degree);
            let mut previous_previous = 0.0;
            let mut previous = 1.0;
            for column in 0..degree {
                let ratio = if column == 0 {
                    0.0
                } else {
                    norm2[column + 1] / norm2[column]
                };
                let current = (value - alpha[column]) * previous - ratio * previous_previous;
                row.push(current / norm2[column + 2].sqrt());
                previous_previous = previous;
                previous = current;
            }
            row
        })
        .collect())
}

pub(crate) fn poly_basis_core(
    x: &[f64],
    degree: usize,
    raw: bool,
    alpha: Option<&[f64]>,
    norm2: Option<&[f64]>,
) -> PyResult<PolyBasis> {
    validate_degree(degree)?;
    if raw {
        return Ok((raw_polynomial_rows(x, degree)?, Vec::new(), Vec::new()));
    }

    let (fitted_alpha, fitted_norm2) = match (alpha, norm2) {
        (None, None) => fit_orthogonal_coefficients(x, degree)?,
        (Some(alpha), Some(norm2)) => {
            validate_orthogonal_coefficients(degree, alpha, norm2)?;
            (alpha.to_vec(), norm2.to_vec())
        }
        _ => {
            return Err(value_error(
                "alpha and norm2 must either both be supplied or both be omitted",
            ));
        }
    };
    let rows = orthogonal_polynomial_rows(x, degree, &fitted_alpha, &fitted_norm2)?;
    Ok((rows, fitted_alpha, fitted_norm2))
}

#[pyfunction]
#[pyo3(signature = (x, degree, raw=false, alpha=None, norm2=None))]
pub fn poly_basis(
    x: Vec<f64>,
    degree: usize,
    raw: bool,
    alpha: Option<Vec<f64>>,
    norm2: Option<Vec<f64>>,
) -> PyResult<PolyBasis> {
    poly_basis_core(&x, degree, raw, alpha.as_deref(), norm2.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_basis_matches_r_stats_poly_fixture() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (rows, alpha, norm2) = poly_basis_core(&x, 3, false, None, None).unwrap();
        let expected = [
            [-0.5400617248673217, 0.5400617248673216, -0.4308202184276644],
            [-0.3857583749052297, 0.0771516749810460, 0.3077287274483316],
            [-0.2314550249431379, -0.2314550249431379, 0.4308202184276648],
            [-0.0771516749810459, -0.3857583749052297, 0.1846372364689991],
            [0.0771516749810459, -0.3857583749052297, -0.1846372364689989],
            [
                0.231_455_024_943_138,
                -0.2314550249431379,
                -0.4308202184276646,
            ],
            [0.3857583749052298, 0.0771516749810459, -0.3077287274483319],
            [0.5400617248673216, 0.5400617248673217, 0.4308202184276646],
        ];

        assert_eq!(alpha.len(), 3);
        for value in alpha {
            assert!((value - 1.5).abs() < 1e-14);
        }
        assert_eq!(norm2, vec![1.0, 8.0, 42.0, 168.0, 594.0]);
        for (actual_row, expected_row) in rows.iter().zip(expected) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn stored_coefficients_rebuild_new_rows() {
        let training = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (_rows, alpha, norm2) = poly_basis_core(&training, 3, false, None, None).unwrap();
        let (predicted, predicted_alpha, predicted_norm2) =
            poly_basis_core(&[10.0, 11.0, 15.0], 3, false, Some(&alpha), Some(&norm2)).unwrap();
        let expected = [
            [1.31157847467778, 5.16916222373008, 21.9718311398109],
            [1.46588182463987, 6.55789237338891, 31.5729674361988],
            [2.08309522448824, 13.6558464716451, 95.8267257274105],
        ];

        assert_eq!(predicted_alpha, alpha);
        assert_eq!(predicted_norm2, norm2);
        for (actual_row, expected_row) in predicted.iter().zip(expected) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn raw_basis_preserves_missing_rows() {
        let (rows, alpha, norm2) =
            poly_basis_core(&[-2.0, f64::NAN, 3.0], 3, true, None, None).unwrap();
        assert_eq!(rows[0], vec![-2.0, 4.0, -8.0]);
        assert!(rows[1].iter().all(|value| value.is_nan()));
        assert_eq!(rows[2], vec![3.0, 9.0, 27.0]);
        assert!(alpha.is_empty());
        assert!(norm2.is_empty());
    }

    #[test]
    fn orthogonal_basis_validates_state_and_rank() {
        assert!(poly_basis_core(&[1.0, 2.0], 0, false, None, None).is_err());
        assert!(poly_basis_core(&[1.0, 1.0, 2.0], 2, false, None, None).is_err());
        assert!(poly_basis_core(&[1.0, f64::NAN, 3.0], 1, false, None, None).is_err());
        assert!(poly_basis_core(&[1.0], 1, false, Some(&[0.0]), None).is_err());
        assert!(poly_basis_core(&[1.0], 1, false, Some(&[0.0]), Some(&[1.0, 1.0])).is_err());
    }
}
