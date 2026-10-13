use crate::core::bspline::spline_design;
use crate::core::natural_spline::{SplineBasisResult, ns, quantile_type7};
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::{Array2, s};

fn observed_values(x: &[f64]) -> SurvivalResult<Vec<f64>> {
    if let Some(value) = x.iter().find(|value| value.is_infinite()) {
        return Err(SurvivalError::invalid_input(format!(
            "x contains non-finite value {value}"
        )));
    }
    let observed: Vec<f64> = x.iter().copied().filter(|value| !value.is_nan()).collect();
    if observed.is_empty() {
        return Err(SurvivalError::invalid_input(
            "x must contain at least one non-missing value",
        ));
    }
    Ok(observed)
}

fn default_boundary(observed: &[f64]) -> (f64, f64) {
    observed
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
            (low.min(value), high.max(value))
        })
}

fn sorted_boundary(boundary: (f64, f64)) -> SurvivalResult<(f64, f64)> {
    if !(boundary.0.is_finite() && boundary.1.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "Boundary.knots must be finite",
        ));
    }
    Ok(if boundary.0 <= boundary.1 {
        boundary
    } else {
        (boundary.1, boundary.0)
    })
}

fn expand_missing(x: &[f64], values: &Array2<f64>) -> Vec<f64> {
    let n_cols = values.ncols();
    let mut basis = Vec::with_capacity(x.len() * n_cols);
    let mut observed_row = 0;
    for value in x {
        if value.is_nan() {
            basis.extend(std::iter::repeat_n(f64::NAN, n_cols));
        } else {
            basis.extend(values.row(observed_row).iter().copied());
            observed_row += 1;
        }
    }
    basis
}

#[doc = " R's `splines::ns(x, df, knots, intercept, Boundary.knots)`: `NaN` rows for"]
#[doc = " missing `x`, and the range of the observed values when no boundary is given."]
pub fn ns_basis(
    x: &[f64],
    df: Option<usize>,
    knots: Option<&[f64]>,
    intercept: bool,
    boundary_knots: Option<(f64, f64)>,
) -> SurvivalResult<SplineBasisResult> {
    let observed = observed_values(x)?;
    if let Some(knots) = knots
        && knots.iter().any(|knot| !knot.is_finite())
    {
        return Err(SurvivalError::invalid_input("non-finite knots"));
    }
    let boundary = match boundary_knots {
        Some(pair) => sorted_boundary(pair)?,
        None if observed.len() == 1 => {
            sorted_boundary((observed[0] * 7.0 / 8.0, observed[0] * 9.0 / 8.0))?
        }
        None => default_boundary(&observed),
    };
    let basis = ns(&observed, df, knots, intercept, boundary)?;
    Ok(SplineBasisResult {
        basis: expand_missing(x, &basis.values),
        n_rows: x.len(),
        n_cols: basis.values.ncols(),
        knots: basis.knots,
        boundary_knots: boundary,
    })
}

fn shove_bs_knots(knots: &mut [f64], low: f64, high: f64) {
    if knots.is_empty() {
        return;
    }
    let minimum = knots.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if minimum == low && knots.iter().any(|&knot| knot != low) {
        let next = knots
            .iter()
            .copied()
            .filter(|&knot| knot > low)
            .fold(f64::INFINITY, f64::min);
        for knot in knots.iter_mut().filter(|knot| **knot == low) {
            *knot += (next - low) / 8.0;
        }
    }
    if maximum == high && knots.iter().any(|&knot| knot != high) {
        let previous = knots
            .iter()
            .copied()
            .filter(|&knot| knot < high)
            .fold(f64::NEG_INFINITY, f64::max);
        for knot in knots.iter_mut().filter(|knot| **knot == high) {
            *knot -= (high - previous) / 8.0;
        }
    }
}

#[doc = " R's `splines::bs(x, df, knots, degree, intercept, Boundary.knots)`, including"]
#[doc = " its Taylor extrapolation for `x` beyond explicitly given boundary knots."]
pub fn bs_basis(
    x: &[f64],
    df: Option<usize>,
    knots: Option<&[f64]>,
    degree: usize,
    intercept: bool,
    boundary_knots: Option<(f64, f64)>,
) -> SurvivalResult<SplineBasisResult> {
    if degree < 1 {
        return Err(SurvivalError::invalid_input(
            "'degree' must be integer >= 1",
        ));
    }
    let observed = observed_values(x)?;
    let ord = degree + 1;
    let explicit = boundary_knots.is_some();
    let (low, high) = match boundary_knots {
        Some(pair) => sorted_boundary(pair)?,
        None => default_boundary(&observed),
    };
    let outside = |value: f64| explicit && (value < low || value > high);
    let knots: Vec<f64> = match (knots, df) {
        (Some(knots), _) => {
            if knots.iter().any(|knot| !knot.is_finite()) {
                return Err(SurvivalError::invalid_input("non-finite knots"));
            }
            knots.to_vec()
        }
        (None, Some(df)) => {
            let n_interior = (df + 1).saturating_sub(ord + usize::from(intercept));
            let mut inside: Vec<f64> = observed.iter().copied().filter(|&v| !outside(v)).collect();
            inside.sort_by(f64::total_cmp);
            if inside.is_empty() && n_interior > 0 {
                return Err(SurvivalError::invalid_input(
                    "no x values inside the boundary knots to place interior knots",
                ));
            }
            let mut knots: Vec<f64> = (1..=n_interior)
                .map(|i| quantile_type7(&inside, i as f64 / (n_interior + 1) as f64))
                .collect();
            shove_bs_knots(&mut knots, low, high);
            knots
        }
        (None, None) => Vec::new(),
    };
    let mut aknots = vec![low; ord];
    aknots.extend(std::iter::repeat_n(high, ord));
    aknots.extend(&knots);
    aknots.sort_by(f64::total_cmp);
    let ncoef = aknots.len() - ord;
    let mut values = Array2::zeros((observed.len(), ncoef));
    let inside: Vec<usize> = (0..observed.len())
        .filter(|&i| !outside(observed[i]))
        .collect();
    if inside.len() < observed.len() {
        let derivs: Vec<usize> = (0..ord).collect();
        let mut factorial = 1.0;
        let scale: Vec<f64> = (0..ord)
            .map(|d| {
                if d > 0 {
                    factorial *= d as f64;
                }
                factorial
            })
            .collect();
        let pivots = [
            (0.75 * low + 0.25 * aknots[ord], true),
            (0.75 * high + 0.25 * aknots[aknots.len() - ord - 1], false),
        ];
        for (pivot, left) in pivots {
            let rows: Vec<usize> = (0..observed.len())
                .filter(|&i| {
                    if left {
                        observed[i] < low
                    } else {
                        observed[i] > high
                    }
                })
                .collect();
            if rows.is_empty() {
                continue;
            }
            let tt = spline_design(&aknots, &vec![pivot; ord], ord, &derivs)?;
            for row in rows {
                let delta = observed[row] - pivot;
                for col in 0..ncoef {
                    values[[row, col]] = (0..ord)
                        .map(|d| delta.powi(d as i32) * (tt[[d, col]] / scale[d]))
                        .sum();
                }
            }
        }
        let inside_x: Vec<f64> = inside.iter().map(|&i| observed[i]).collect();
        if !inside_x.is_empty() {
            let design = spline_design(&aknots, &inside_x, ord, &[0])?;
            for (k, &row) in inside.iter().enumerate() {
                values.row_mut(row).assign(&design.row(k));
            }
        }
    } else {
        values = spline_design(&aknots, &observed, ord, &[0])?;
    }
    if !intercept {
        values = values.slice(s![.., 1..]).to_owned();
    }
    Ok(SplineBasisResult {
        basis: expand_missing(x, &values),
        n_rows: x.len(),
        n_cols: values.ncols(),
        knots,
        boundary_knots: (low, high),
    })
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12 * expected.abs().max(1.0),
            "expected {expected:.17e}, got {actual:.17e}"
        );
    }

    #[test]
    fn ns_matches_r_with_quantile_knots_and_missing_rows() {
        let x = [1.0, 2.0, f64::NAN, 4.0, 7.0, 8.0, 10.0];
        let basis = ns_basis(&x, Some(3), None, false, None).unwrap();
        assert_eq!((basis.n_rows, basis.n_cols), (7, 3));
        assert_close(basis.knots[0], 3.333333333333333);
        assert_close(basis.knots[1], 7.333333333333333);
        assert_eq!(basis.boundary_knots, (1.0, 10.0));
        assert!(basis.basis[6..9].iter().all(|value| value.is_nan()));
        let expected = [
            -0.065208376439665067,
            0.25454510701251193,
            -0.18181793358036566,
        ];
        for (actual, expected) in basis.basis[3..6].iter().zip(expected) {
            assert_close(*actual, expected);
        }
    }

    #[test]
    fn bs_matches_r_inside_and_beyond_given_boundaries() {
        let x = [0.0, 1.0, 2.5, 4.0, 6.0];
        let basis = bs_basis(&x, None, Some(&[2.0]), 3, false, Some((1.0, 4.0))).unwrap();
        assert_eq!((basis.n_rows, basis.n_cols), (5, 4));
        let expected = [
            [
                -8.4444444444444429,
                1.5555555555555554,
                -0.1111111111111111,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
            [0.1875, 0.46875, 0.328125, 0.015625],
            [0.0, 0.0, 0.0, 1.0],
            [
                -0.44444444444444442,
                3.5555555555555554,
                -10.111111111111111,
                8.0,
            ],
        ];
        for (row, expected) in expected.iter().enumerate() {
            for (col, &value) in expected.iter().enumerate() {
                assert_close(basis.basis[row * 4 + col], value);
            }
        }
    }

    #[test]
    fn bs_df_places_knots_and_keeps_an_intercept_column() {
        let x: Vec<f64> = (1..=20).map(f64::from).collect();
        let basis = bs_basis(&x, Some(6), None, 3, true, None).unwrap();
        assert_eq!(basis.n_cols, 6);
        assert_close(basis.knots[0], 7.333333333333333);
        assert_close(basis.knots[1], 13.666666666666666);
        for row in basis.basis.chunks(6) {
            assert_close(row.iter().sum::<f64>(), 1.0);
        }
    }
}
