use super::bspline::{basis_row, derivative_row};
use super::nsk_module::SplineBasisResult;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const ORDER: usize = 4;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn quantile_type7(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

fn normalize_boundaries(
    observed: &[f64],
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<(f64, f64)> {
    let (mut lower, mut upper) = match boundary_knots {
        Some(bounds) => bounds,
        None if observed.len() == 1 => (observed[0] * 7.0 / 8.0, observed[0] * 9.0 / 8.0),
        None => observed.iter().copied().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(lower, upper), value| (lower.min(value), upper.max(value)),
        ),
    };
    if lower > upper {
        std::mem::swap(&mut lower, &mut upper);
    }
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(value_error(
            "boundary_knots must contain two distinct finite values",
        ));
    }
    Ok((lower, upper))
}

fn computed_knots(
    observed: &[f64],
    n_interior: usize,
    boundaries: (f64, f64),
) -> PyResult<Vec<f64>> {
    if n_interior == 0 {
        return Ok(Vec::new());
    }
    let mut inside: Vec<f64> = observed
        .iter()
        .copied()
        .filter(|value| *value >= boundaries.0 && *value <= boundaries.1)
        .collect();
    inside.sort_by(f64::total_cmp);
    if inside.is_empty() {
        return Err(value_error(
            "x must contain values within boundary_knots to compute knots",
        ));
    }
    let mut knots: Vec<f64> = (1..=n_interior)
        .map(|index| quantile_type7(&inside, index as f64 / (n_interior + 1) as f64))
        .collect();

    if knots.contains(&boundaries.0) {
        let next = knots
            .iter()
            .copied()
            .filter(|value| *value > boundaries.0)
            .reduce(f64::min)
            .ok_or_else(|| value_error("all interior knots match left boundary knot"))?;
        let replacement = boundaries.0 + (next - boundaries.0) / 8.0;
        for knot in &mut knots {
            if *knot == boundaries.0 {
                *knot = replacement;
            }
        }
    }
    if knots.contains(&boundaries.1) {
        let previous = knots
            .iter()
            .copied()
            .filter(|value| *value < boundaries.1)
            .reduce(f64::max)
            .ok_or_else(|| value_error("all interior knots match right boundary knot"))?;
        let replacement = boundaries.1 - (boundaries.1 - previous) / 8.0;
        for knot in &mut knots {
            if *knot == boundaries.1 {
                *knot = replacement;
            }
        }
    }
    Ok(knots)
}

struct HouseholderReflector {
    pivot: usize,
    values: Vec<f64>,
    beta: f64,
}

fn householder_reflectors(matrix: &[Vec<f64>]) -> PyResult<Vec<HouseholderReflector>> {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    if rows < columns {
        return Err(value_error("natural spline constraint matrix is malformed"));
    }
    let mut factors = matrix.to_vec();
    let mut reflectors = Vec::with_capacity(columns);

    for pivot in 0..columns {
        let norm = factors[pivot..]
            .iter()
            .map(|row| row[pivot] * row[pivot])
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(value_error("natural spline constraints are rank deficient"));
        }
        let alpha = -norm.copysign(factors[pivot][pivot]);
        let mut reflector: Vec<f64> = factors[pivot..].iter().map(|row| row[pivot]).collect();
        reflector[0] -= alpha;
        let squared_norm = reflector.iter().map(|value| value * value).sum::<f64>();
        if squared_norm == 0.0 || !squared_norm.is_finite() {
            return Err(value_error("natural spline constraints are rank deficient"));
        }
        let beta = 2.0 / squared_norm;

        let mut column = pivot;
        while column < columns {
            let projection = reflector
                .iter()
                .enumerate()
                .map(|(offset, value)| value * factors[pivot + offset][column])
                .sum::<f64>();
            for (offset, value) in reflector.iter().enumerate() {
                factors[pivot + offset][column] -= beta * value * projection;
            }
            column += 1;
        }
        reflectors.push(HouseholderReflector {
            pivot,
            values: reflector,
            beta,
        });
    }
    Ok(reflectors)
}

fn apply_householder_constraints(
    mut values: Vec<f64>,
    reflectors: &[HouseholderReflector],
) -> Vec<f64> {
    for reflector in reflectors {
        let projection = reflector
            .values
            .iter()
            .enumerate()
            .map(|(offset, value)| values[reflector.pivot + offset] * value)
            .sum::<f64>();
        for (offset, value) in reflector.values.iter().enumerate() {
            values[reflector.pivot + offset] -= reflector.beta * projection * value;
        }
    }
    values
}

fn raw_basis_row(x: f64, knots: &[f64], boundaries: (f64, f64)) -> Vec<f64> {
    let pivot = if x < boundaries.0 {
        Some(boundaries.0)
    } else if x > boundaries.1 {
        Some(boundaries.1)
    } else {
        None
    };
    match pivot {
        Some(pivot) => basis_row(knots, pivot, ORDER)
            .into_iter()
            .zip(derivative_row(knots, pivot, ORDER, 1))
            .map(|(basis, derivative)| (x - pivot).mul_add(derivative, basis))
            .collect(),
        None => basis_row(knots, x, ORDER),
    }
}

pub(crate) fn ns_basis_core(
    x: &[f64],
    df: Option<usize>,
    knots: Option<&[f64]>,
    boundary_knots: Option<(f64, f64)>,
    intercept: bool,
) -> PyResult<SplineBasisResult> {
    if x.iter().any(|value| value.is_infinite()) {
        return Err(value_error("x must contain only finite or missing values"));
    }
    let observed: Vec<f64> = x.iter().copied().filter(|value| !value.is_nan()).collect();
    if observed.is_empty() {
        return Err(value_error("x must contain at least one non-missing value"));
    }
    let boundaries = normalize_boundaries(&observed, boundary_knots)?;
    let minimum_df = 1 + usize::from(intercept);
    let interior = match knots {
        Some(values) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(value_error("knots must contain only finite values"));
            }
            values.to_vec()
        }
        None => computed_knots(
            &observed,
            df.unwrap_or(minimum_df).saturating_sub(minimum_df),
            boundaries,
        )?,
    };

    let mut augmented = Vec::with_capacity(interior.len() + 8);
    augmented.extend([boundaries.0; 4]);
    augmented.extend(interior.iter().copied());
    augmented.extend([boundaries.1; 4]);
    augmented.sort_by(f64::total_cmp);
    let first_column = usize::from(!intercept);
    let raw_width = augmented.len() - ORDER - first_column;
    let output_width = raw_width
        .checked_sub(2)
        .ok_or_else(|| value_error("natural spline basis has too few columns"))?;

    let constraints: Vec<Vec<f64>> = [boundaries.0, boundaries.1]
        .iter()
        .map(|value| derivative_row(&augmented, *value, ORDER, 2)[first_column..].to_vec())
        .collect();
    let constraint_transpose: Vec<Vec<f64>> = (0..raw_width)
        .map(|column| vec![constraints[0][column], constraints[1][column]])
        .collect();
    let reflectors = householder_reflectors(&constraint_transpose)?;

    let basis: Vec<f64> = x
        .par_iter()
        .flat_map_iter(|value| {
            if value.is_nan() {
                return vec![f64::NAN; output_width];
            }
            let raw = raw_basis_row(*value, &augmented, boundaries);
            apply_householder_constraints(raw[first_column..].to_vec(), &reflectors)[2..].to_vec()
        })
        .collect();

    Ok(SplineBasisResult {
        basis,
        n_rows: x.len(),
        n_cols: output_width,
        knots: interior,
        boundary_knots: boundaries,
    })
}

#[pyfunction]
#[pyo3(signature = (x, df=None, knots=None, boundary_knots=None, intercept=false))]
pub fn ns_basis(
    x: Vec<f64>,
    df: Option<usize>,
    knots: Option<Vec<f64>>,
    boundary_knots: Option<(f64, f64)>,
    intercept: bool,
) -> PyResult<SplineBasisResult> {
    ns_basis_core(&x, df, knots.as_deref(), boundary_knots, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(result: &SplineBasisResult) -> Vec<Vec<f64>> {
        result
            .basis
            .chunks(result.n_cols)
            .map(<[f64]>::to_vec)
            .collect()
    }

    #[test]
    fn explicit_basis_matches_r_splines_fixture_including_extrapolation() {
        let result = ns_basis_core(
            &[-2.0, 0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 13.0],
            None,
            Some(&[1.0, 4.0, 7.0]),
            Some((0.0, 8.0)),
            false,
        )
        .unwrap();
        let expected = [
            [
                0.0,
                0.181568259800641,
                -0.907841299003203,
                0.726273039202563,
            ],
            [0.0, 0.0, 0.0, 0.0],
            [
                0.00446428571428571,
                -0.0437709197733687,
                0.218854598866844,
                -0.175083679093475,
            ],
            [
                0.112599206349206,
                -0.0937484650816598,
                0.473702642868617,
                -0.378962114294893,
            ],
            [
                0.456349206349206,
                -0.00814919180508497,
                0.358206276485742,
                -0.286565021188594,
            ],
            [
                0.456349206349206,
                0.441262450350684,
                0.111148065706897,
                -0.0722517858988506,
            ],
            [
                0.0,
                -0.0952380952380952,
                0.476190476190476,
                0.619047619047619,
            ],
            [0.0, -3.30952380952381, 1.54761904761905, 2.76190476190476],
        ];
        assert_eq!(result.n_cols, 4);
        for (actual, expected) in rows(&result).iter().zip(expected) {
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 2e-14);
            }
        }
    }

    #[test]
    fn computed_knots_and_missing_rows_match_r() {
        let result =
            ns_basis_core(&[1.0, f64::NAN, 2.0, 5.0, 9.0], Some(3), None, None, false).unwrap();
        assert_eq!(result.knots, vec![2.0, 5.0]);
        assert_eq!(result.boundary_knots, (1.0, 9.0));
        assert_eq!(result.n_cols, 3);
        assert!(rows(&result)[1].iter().all(|value| value.is_nan()));
        let expected_last = [-0.150537634408602, 0.413978494623656, 0.736559139784946];
        for (actual, expected) in rows(&result)[4].iter().zip(expected_last) {
            assert!((actual - expected).abs() < 2e-14);
        }
    }

    #[test]
    fn intercept_and_duplicate_knots_match_r_dimensions() {
        let intercept =
            ns_basis_core(&[1.0, 2.0, 3.0, 4.0, 5.0], Some(4), None, None, true).unwrap();
        assert_eq!(intercept.n_cols, 4);

        let duplicate = ns_basis_core(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            None,
            Some(&[3.0, 3.0, 6.0]),
            Some((1.0, 8.0)),
            false,
        )
        .unwrap();
        assert_eq!(duplicate.n_cols, 4);
        assert_eq!(duplicate.knots, vec![3.0, 3.0, 6.0]);
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        assert!(ns_basis_core(&[f64::NAN], Some(3), None, None, false).is_err());
        assert!(ns_basis_core(&[1.0, f64::INFINITY], Some(3), None, None, false).is_err());
        assert!(ns_basis_core(&[1.0, 2.0], Some(3), None, Some((1.0, 1.0)), false).is_err());
        assert!(
            ns_basis_core(
                &[1.0, 2.0],
                None,
                Some(&[f64::NAN]),
                Some((0.0, 3.0)),
                false,
            )
            .is_err()
        );
    }
}
