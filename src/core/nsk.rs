use crate::internal::matrix::invert_matrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn minimum_df(intercept: bool) -> usize {
    if intercept { 2 } else { 1 }
}

fn validate_df(df: usize, intercept: bool) -> PyResult<()> {
    let minimum = minimum_df(intercept);
    if df < minimum {
        return Err(value_error(format!(
            "df must be at least {minimum} when intercept is {intercept}"
        )));
    }
    Ok(())
}

fn validate_finite_slice(values: &[f64], field: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{field} contains non-finite value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_boundary_knots(boundary_knots: (f64, f64)) -> PyResult<()> {
    let (lower, upper) = boundary_knots;
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(value_error(
            "boundary_knots must be finite and strictly increasing",
        ));
    }
    Ok(())
}

fn uses_data_boundary(boundary_knots: (f64, f64)) -> bool {
    boundary_knots.0 == f64::NEG_INFINITY && boundary_knots.1 == f64::INFINITY
}

fn sorted_unique_values(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| *a == *b);
    values
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NaturalSplineKnot {
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
    #[pyo3(get)]
    pub intercept: bool,
    #[pyo3(get)]
    pub df: usize,
}

#[pymethods]
impl NaturalSplineKnot {
    #[new]
    #[pyo3(signature = (knots=None, boundary_knots=None, df=None, intercept=None))]
    pub fn new(
        knots: Option<Vec<f64>>,
        boundary_knots: Option<(f64, f64)>,
        df: Option<usize>,
        intercept: Option<bool>,
    ) -> PyResult<Self> {
        let intercept_val = intercept.unwrap_or(false);
        let bounds = boundary_knots.unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        if !uses_data_boundary(bounds) {
            validate_boundary_knots(bounds)?;
        }

        let (interior_knots, computed_df) = match knots {
            Some(k) => {
                validate_finite_slice(&k, "knots")?;
                let normalized = sorted_unique_values(k);
                let d = normalized.len() + 1 + if intercept_val { 1 } else { 0 };
                (normalized, d)
            }
            None => {
                let d = df.unwrap_or_else(|| minimum_df(intercept_val));
                validate_df(d, intercept_val)?;
                (vec![], d)
            }
        };

        Ok(NaturalSplineKnot {
            knots: interior_knots,
            boundary_knots: bounds,
            intercept: intercept_val,
            df: computed_df,
        })
    }

    pub fn basis(&self, x: Vec<f64>) -> PyResult<SplineBasisResult> {
        let n = x.len();
        validate_df(self.df, self.intercept)?;
        if let Some((idx, value)) = x.iter().enumerate().find(|(_, value)| value.is_infinite()) {
            return Err(value_error(format!(
                "x contains non-finite value {value} at index {idx}"
            )));
        }
        validate_finite_slice(&self.knots, "knots")?;
        if !uses_data_boundary(self.boundary_knots) {
            validate_boundary_knots(self.boundary_knots)?;
        }

        if n == 0 {
            return Ok(SplineBasisResult {
                basis: vec![],
                n_rows: 0,
                n_cols: self.df,
                knots: self.knots.clone(),
                boundary_knots: self.boundary_knots,
            });
        }

        let observed_x: Vec<f64> = x.iter().copied().filter(|value| !value.is_nan()).collect();
        if observed_x.is_empty() {
            return Err(value_error("x must contain at least one non-missing value"));
        }

        let mut all_knots = resolve_all_knots(
            &observed_x,
            &self.knots,
            self.boundary_knots,
            self.df,
            self.intercept,
        )?;
        all_knots = sorted_unique_values(all_knots);
        validate_knot_span(&all_knots)?;

        let n_raw_basis = all_knots.len();
        let n_cols = if self.intercept {
            n_raw_basis
        } else {
            n_raw_basis - 1
        };

        let basis: Vec<f64> = x
            .par_iter()
            .flat_map(|&xi| {
                if xi.is_nan() {
                    vec![f64::NAN; n_raw_basis]
                } else {
                    natural_spline_raw_basis_at_point(xi, &all_knots)
                }
            })
            .collect();

        let transformed_basis =
            transform_to_knot_heights(&basis, n, n_raw_basis, &all_knots, self.intercept)?;

        let bk_low = all_knots[0];
        let bk_high = all_knots[all_knots.len() - 1];
        let interior_knots = all_knots[1..all_knots.len() - 1].to_vec();

        Ok(SplineBasisResult {
            basis: transformed_basis,
            n_rows: n,
            n_cols,
            knots: interior_knots,
            boundary_knots: (bk_low, bk_high),
        })
    }

    pub fn predict(&self, x: Vec<f64>, coef: Vec<f64>) -> PyResult<Vec<f64>> {
        validate_finite_slice(&coef, "coef")?;
        let basis_result = self.basis(x)?;

        if coef.len() != basis_result.n_cols {
            return Err(value_error(format!(
                "coef length ({}) must match number of basis functions ({})",
                coef.len(),
                basis_result.n_cols
            )));
        }

        let mut predictions = Vec::with_capacity(basis_result.n_rows);

        for i in 0..basis_result.n_rows {
            let mut pred = 0.0;
            for (j, &c) in coef.iter().enumerate().take(basis_result.n_cols) {
                pred += basis_result.basis[i * basis_result.n_cols + j] * c;
            }
            predictions.push(pred);
        }

        Ok(predictions)
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SplineBasisResult {
    #[pyo3(get)]
    pub basis: Vec<f64>,
    #[pyo3(get)]
    pub n_rows: usize,
    #[pyo3(get)]
    pub n_cols: usize,
    #[pyo3(get)]
    pub knots: Vec<f64>,
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
}

#[pyfunction]
#[pyo3(signature = (x, df=None, knots=None, boundary_knots=None))]
pub fn nsk(
    x: Vec<f64>,
    df: Option<usize>,
    knots: Option<Vec<f64>>,
    boundary_knots: Option<(f64, f64)>,
) -> PyResult<SplineBasisResult> {
    let spline = NaturalSplineKnot::new(knots, boundary_knots, df, Some(false))?;
    spline.basis(x)
}

fn validate_knot_span(knots: &[f64]) -> PyResult<()> {
    if knots.len() < 2 {
        return Err(value_error(
            "at least two distinct finite knots are required for nsk",
        ));
    }
    validate_finite_slice(knots, "knots")?;
    for (idx, pair) in knots.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(value_error(format!(
                "knots must be strictly increasing after duplicate collapse; got {} then {} at positions {} and {}",
                pair[0],
                pair[1],
                idx,
                idx + 1
            )));
        }
    }
    Ok(())
}

fn quantile_type7(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !probability.is_finite() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let p = probability.clamp(0.0, 1.0);
    let pos = p * (sorted.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let weight = pos - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

fn default_boundary_knots(x: &[f64]) -> PyResult<(f64, f64)> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let lower = quantile_type7(&sorted, 0.05)
        .ok_or_else(|| value_error("x must not be empty when boundary_knots are not provided"))?;
    let upper = quantile_type7(&sorted, 0.95)
        .ok_or_else(|| value_error("x must not be empty when boundary_knots are not provided"))?;
    validate_boundary_knots((lower, upper)).map_err(|_| {
        value_error("x must span a non-zero finite range when boundary_knots are not provided")
    })?;
    Ok((lower, upper))
}

fn compute_quantile_knots(x: &[f64], n_knots: usize, low: f64, high: f64) -> PyResult<Vec<f64>> {
    if n_knots == 0 {
        return Ok(vec![]);
    }

    let mut sorted: Vec<f64> = x.iter().copied().filter(|&v| v > low && v < high).collect();
    sorted.sort_by(f64::total_cmp);

    if sorted.is_empty() {
        return Err(value_error(format!(
            "not enough x values inside boundary_knots to compute {n_knots} interior knots"
        )));
    }

    let mut knots = Vec::with_capacity(n_knots);
    for i in 1..=n_knots {
        let p = i as f64 / (n_knots + 1) as f64;
        let knot = quantile_type7(&sorted, p).ok_or_else(|| {
            value_error(format!(
                "not enough x values inside boundary_knots to compute {n_knots} interior knots"
            ))
        })?;
        knots.push(knot);
    }

    Ok(knots)
}

fn resolve_all_knots(
    x: &[f64],
    knots: &[f64],
    boundary_knots: (f64, f64),
    df: usize,
    intercept: bool,
) -> PyResult<Vec<f64>> {
    let (bk_low, bk_high) = if uses_data_boundary(boundary_knots) {
        default_boundary_knots(x)?
    } else {
        validate_boundary_knots(boundary_knots)?;
        boundary_knots
    };

    if knots.is_empty() {
        let n_interior = df.checked_sub(minimum_df(intercept)).ok_or_else(|| {
            value_error(format!(
                "df must be at least {} when intercept is {}",
                minimum_df(intercept),
                intercept
            ))
        })?;
        let mut all_knots = vec![bk_low];
        all_knots.extend(compute_quantile_knots(x, n_interior, bk_low, bk_high)?);
        all_knots.push(bk_high);
        return Ok(all_knots);
    }

    let min_knot = knots
        .iter()
        .fold(f64::INFINITY, |acc, &value| acc.min(value));
    let max_knot = knots
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
    let mut all_knots = knots.to_vec();
    if bk_low < min_knot {
        all_knots.push(bk_low);
    }
    if bk_high > max_knot {
        all_knots.push(bk_high);
    }
    Ok(all_knots)
}

fn natural_spline_raw_basis_at_point(x: f64, knots: &[f64]) -> Vec<f64> {
    let k = knots.len();
    if k < 2 {
        return vec![1.0];
    }

    let mut basis = Vec::with_capacity(k);

    basis.push(1.0);
    basis.push(x);

    let upper = knots[k - 1];
    let penultimate = knots[k - 2];

    for &knot in knots.iter().take(k - 2) {
        basis.push(natural_spline_d(x, knot, upper) - natural_spline_d(x, penultimate, upper));
    }

    basis
}

fn natural_spline_d(x: f64, knot: f64, upper: f64) -> f64 {
    (truncated_power(x, knot, 3) - truncated_power(x, upper, 3)) / (upper - knot)
}

fn truncated_power(x: f64, knot: f64, degree: i32) -> f64 {
    if x > knot {
        (x - knot).powi(degree)
    } else {
        0.0
    }
}

fn transform_to_knot_heights(
    basis: &[f64],
    n: usize,
    n_basis: usize,
    knots: &[f64],
    intercept: bool,
) -> PyResult<Vec<f64>> {
    let k = knots.len();
    if k == 0 || k != n_basis {
        return Ok(basis.to_vec());
    }
    if basis.len() != n * n_basis {
        return Err(value_error(format!(
            "basis length ({}) must equal n * n_basis ({})",
            basis.len(),
            n * n_basis
        )));
    }

    let mut b_matrix = vec![vec![0.0; n_basis]; k];
    for (i, &knot) in knots.iter().enumerate() {
        let basis_at_knot = natural_spline_raw_basis_at_point(knot, knots);
        for (j, &val) in basis_at_knot.iter().enumerate() {
            b_matrix[i][j] = val;
        }
    }

    let inverse = invert_matrix(&b_matrix).ok_or_else(|| {
        value_error("knot-height transform is singular; knots must be distinct and well-spaced")
    })?;

    let transformed: Vec<f64> = basis
        .par_chunks(n_basis)
        .flat_map_iter(|row| {
            let first_col = if intercept { 0 } else { 1 };
            (first_col..n_basis).map(|col| {
                row.iter()
                    .zip(inverse.iter())
                    .map(|(&basis_value, inverse_row)| basis_value * inverse_row[col])
                    .sum::<f64>()
            })
        })
        .collect();

    let n_output = n_basis - usize::from(!intercept);
    for (row_idx, row) in transformed.chunks(n_output).enumerate() {
        if row.iter().all(|value| value.is_nan()) {
            continue;
        }
        if let Some((col_idx, value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite())
        {
            let idx = row_idx * n_output + col_idx;
            return Err(value_error(format!(
                "knot-height transform produced non-finite value {value} at index {idx}"
            )));
        }
    }

    Ok(transformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    #[test]
    fn test_nsk_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = nsk(x, Some(3), None, None).unwrap();

        assert_eq!(result.n_rows, 5);
        assert_eq!(result.n_cols, 3);
        assert!((result.boundary_knots.0 - 1.2).abs() < 1e-12);
        assert!((result.boundary_knots.1 - 4.8).abs() < 1e-12);
        assert!((result.knots[0] - 2.6666666666666665).abs() < 1e-12);
        assert!((result.knots[1] - 3.333333333333333).abs() < 1e-12);
        assert_eq!(result.basis.len(), result.n_rows * result.n_cols);
        assert!((result.basis[0] - -0.306_633_906_633_906_8).abs() < 1e-12);
        assert!((result.basis[1] - 0.12972972972972977).abs() < 1e-12);
        assert!((result.basis[2] - -0.007507507507507517).abs() < 1e-12);
    }

    #[test]
    fn test_nsk_with_knots() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let knots = vec![3.0, 5.0, 7.0];
        let boundary = (1.0, 10.0);

        let result = nsk(x, None, Some(knots.clone()), Some(boundary)).unwrap();

        assert_eq!(result.knots, knots);
        assert_eq!(result.boundary_knots, boundary);
    }

    #[test]
    fn test_nsk_basis_is_knot_height_parameterized() {
        let x = vec![1.0, 3.0, 5.0, 7.0, 10.0];
        let knots = vec![3.0, 5.0, 7.0];
        let boundary = (1.0, 10.0);

        let result = nsk(x, None, Some(knots), Some(boundary)).unwrap();

        assert_eq!(result.n_rows, 5);
        assert_eq!(result.n_cols, 4);
        for row in 0..result.n_rows {
            for col in 0..result.n_cols {
                let expected = if row > 0 && row - 1 == col { 1.0 } else { 0.0 };
                let actual = result.basis[row * result.n_cols + col];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "basis[{row}, {col}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_natural_spline_knot_predict() {
        let spline =
            NaturalSplineKnot::new(Some(vec![3.0, 5.0, 7.0]), Some((1.0, 10.0)), None, None)
                .unwrap();

        let x = vec![1.0, 3.0, 5.0, 7.0, 10.0];
        let basis_result = spline.basis(x.clone()).unwrap();
        assert_eq!(basis_result.n_cols, 4);

        let coef = vec![30.0, 50.0, 70.0, 100.0];
        let predictions = spline.predict(x, coef.clone()).unwrap();
        let expected = [0.0, 30.0, 50.0, 70.0, 100.0];

        for (actual, expected) in predictions.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_nsk_rejects_malformed_inputs() {
        initialize_python();

        assert!(nsk(vec![1.0, f64::INFINITY], Some(3), None, None).is_err());
        assert!(nsk(vec![f64::NAN, f64::NAN], Some(3), None, None).is_err());
        assert!(nsk(vec![1.0, 1.0], Some(3), None, None).is_err());
        assert!(nsk(vec![1.0, 2.0], Some(0), None, None).is_err());
        assert!(nsk(vec![1.0, 2.0], Some(3), None, Some((2.0, 2.0))).is_err());
        assert!(nsk(vec![1.0, 2.0], None, Some(vec![f64::NAN]), Some((0.0, 3.0))).is_err());
        assert!(NaturalSplineKnot::new(None, Some((0.0, 10.0)), Some(1), Some(true)).is_err());

        let tied = nsk(vec![0.0, 1.0, 1.0, 1.0, 2.0], Some(4), None, None)
            .expect("duplicate computed quantile knots collapse like R survival::nsk");
        assert_eq!(tied.n_cols, 2);
        assert_eq!(tied.knots, vec![1.0]);
    }

    #[test]
    fn test_nsk_preserves_missing_rows_and_uses_observed_values_for_knots() {
        let result = nsk(vec![1.0, f64::NAN, 2.0, 3.0, 4.0], Some(3), None, None).unwrap();
        let observed = nsk(vec![1.0, 2.0, 3.0, 4.0], Some(3), None, None).unwrap();

        assert_eq!(result.n_rows, 5);
        assert_eq!(result.n_cols, observed.n_cols);
        assert_eq!(result.knots, observed.knots);
        assert_eq!(result.boundary_knots, observed.boundary_knots);
        assert!(
            result.basis[result.n_cols..2 * result.n_cols]
                .iter()
                .all(|value| value.is_nan())
        );

        for (result_row, observed_row) in [0, 2, 3, 4].into_iter().zip(0..4) {
            let result_start = result_row * result.n_cols;
            let observed_start = observed_row * observed.n_cols;
            for col_idx in 0..result.n_cols {
                assert!(
                    (result.basis[result_start + col_idx]
                        - observed.basis[observed_start + col_idx])
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn test_natural_spline_knot_predict_rejects_non_finite_coef() {
        initialize_python();

        let spline = NaturalSplineKnot::new(None, Some((0.0, 2.0)), Some(1), None).unwrap();
        let err = spline
            .predict(vec![0.0, 1.0], vec![1.0, f64::INFINITY])
            .expect_err("non-finite coefficient should be rejected");

        assert!(err.to_string().contains("coef contains non-finite"));
    }

    #[test]
    fn test_truncated_power() {
        assert_eq!(truncated_power(5.0, 3.0, 2), 4.0);
        assert_eq!(truncated_power(2.0, 3.0, 2), 0.0);
        assert_eq!(truncated_power(3.0, 3.0, 2), 0.0);
    }
}
