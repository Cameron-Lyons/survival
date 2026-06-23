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

fn validate_strictly_increasing(values: &[f64], field: &str) -> PyResult<()> {
    for (idx, pair) in values.windows(2).enumerate() {
        if pair[0] >= pair[1] {
            return Err(value_error(format!(
                "{field} must be strictly increasing; got {} then {} at positions {} and {}",
                pair[0],
                pair[1],
                idx,
                idx + 1
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

fn validate_knots_inside_boundary(knots: &[f64], boundary_knots: (f64, f64)) -> PyResult<()> {
    let (lower, upper) = boundary_knots;
    for (idx, &knot) in knots.iter().enumerate() {
        if knot <= lower || knot >= upper {
            return Err(value_error(format!(
                "knots must lie strictly inside boundary_knots; got {knot} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn uses_data_boundary(boundary_knots: (f64, f64)) -> bool {
    boundary_knots.0 == f64::NEG_INFINITY && boundary_knots.1 == f64::INFINITY
}

/// Natural spline with knot heights as basis coefficients.
///
/// This creates a natural cubic spline basis where the coefficients
/// directly represent the function values at the knots, making them
/// easily interpretable.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NaturalSplineKnot {
    /// Interior knot locations
    #[pyo3(get)]
    pub knots: Vec<f64>,
    /// Boundary knots (extrapolation becomes linear beyond these)
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
    /// Whether to include an intercept column
    #[pyo3(get)]
    pub intercept: bool,
    /// Degrees of freedom
    #[pyo3(get)]
    pub df: usize,
}

#[pymethods]
impl NaturalSplineKnot {
    /// Create a natural spline basis specification.
    ///
    /// # Arguments
    /// * `knots` - Interior knot locations (or None to compute from data)
    /// * `boundary_knots` - Boundary knot locations (or None to use data range)
    /// * `df` - Degrees of freedom (used if knots not specified)
    /// * `intercept` - Whether to include intercept (default: false)
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
                validate_strictly_increasing(&k, "knots")?;
                if !uses_data_boundary(bounds) {
                    validate_knots_inside_boundary(&k, bounds)?;
                }
                let d = k.len() + 1 + if intercept_val { 1 } else { 0 };
                (k, d)
            }
            None => {
                let d = df.unwrap_or(3);
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

    /// Compute the spline basis matrix for given data.
    ///
    /// # Arguments
    /// * `x` - Data values at which to evaluate the basis
    ///
    /// # Returns
    /// Matrix of basis function values (n x df), flattened row-major
    pub fn basis(&self, x: Vec<f64>) -> PyResult<SplineBasisResult> {
        let n = x.len();
        validate_df(self.df, self.intercept)?;
        validate_finite_slice(&x, "x")?;
        validate_finite_slice(&self.knots, "knots")?;
        validate_strictly_increasing(&self.knots, "knots")?;
        if !uses_data_boundary(self.boundary_knots) {
            validate_boundary_knots(self.boundary_knots)?;
            validate_knots_inside_boundary(&self.knots, self.boundary_knots)?;
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

        let (bk_low, bk_high) = if uses_data_boundary(self.boundary_knots) {
            let min_x = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_x = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if min_x >= max_x {
                return Err(value_error(
                    "x must span a non-zero finite range when boundary_knots are not provided",
                ));
            }
            (min_x, max_x)
        } else {
            self.boundary_knots
        };
        validate_boundary_knots((bk_low, bk_high))?;

        let interior_knots = if self.knots.is_empty() {
            let n_interior = self
                .df
                .checked_sub(minimum_df(self.intercept))
                .ok_or_else(|| {
                    value_error(format!(
                        "df must be at least {} when intercept is {}",
                        minimum_df(self.intercept),
                        self.intercept
                    ))
                })?;
            let computed = compute_quantile_knots(&x, n_interior, bk_low, bk_high);
            if computed.len() != n_interior {
                return Err(value_error(format!(
                    "not enough x values inside boundary_knots to compute {n_interior} interior knots"
                )));
            }
            validate_strictly_increasing(&computed, "computed knots")?;
            computed
        } else {
            self.knots.clone()
        };
        validate_knots_inside_boundary(&interior_knots, (bk_low, bk_high))?;

        let mut all_knots = vec![bk_low];
        all_knots.extend(interior_knots.iter().copied());
        all_knots.push(bk_high);

        let n_basis = all_knots.len();

        let basis: Vec<f64> = x
            .par_iter()
            .flat_map(|&xi| natural_spline_basis_at_point(xi, &all_knots))
            .collect();

        let transformed_basis = transform_to_knot_heights(&basis, n, n_basis, &all_knots)?;

        Ok(SplineBasisResult {
            basis: transformed_basis,
            n_rows: n,
            n_cols: n_basis,
            knots: interior_knots,
            boundary_knots: (bk_low, bk_high),
        })
    }

    /// Predict values given coefficients (which are function values at knots).
    ///
    /// # Arguments
    /// * `x` - Points at which to predict
    /// * `coef` - Coefficients (function values at knots)
    ///
    /// # Returns
    /// Predicted values at each x
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

/// Result of computing spline basis
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SplineBasisResult {
    /// Basis matrix (flattened row-major)
    #[pyo3(get)]
    pub basis: Vec<f64>,
    /// Number of observations
    #[pyo3(get)]
    pub n_rows: usize,
    /// Number of basis functions
    #[pyo3(get)]
    pub n_cols: usize,
    /// Interior knots used
    #[pyo3(get)]
    pub knots: Vec<f64>,
    /// Boundary knots used
    #[pyo3(get)]
    pub boundary_knots: (f64, f64),
}

/// Create natural spline basis for given data.
///
/// # Arguments
/// * `x` - Data values
/// * `df` - Degrees of freedom (default: 3)
/// * `knots` - Optional interior knot locations
/// * `boundary_knots` - Optional boundary knot locations
///
/// # Returns
/// `SplineBasisResult` with basis matrix
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

/// Compute quantile knots from data
fn compute_quantile_knots(x: &[f64], n_knots: usize, low: f64, high: f64) -> Vec<f64> {
    if n_knots == 0 {
        return vec![];
    }

    let mut sorted: Vec<f64> = x.iter().copied().filter(|&v| v > low && v < high).collect();
    sorted.sort_by(f64::total_cmp);

    if sorted.is_empty() {
        return vec![];
    }

    let mut knots = Vec::with_capacity(n_knots);
    for i in 1..=n_knots {
        let p = i as f64 / (n_knots + 1) as f64;
        let pos = (p * (sorted.len() as f64 + 1.0) - 1.0).clamp(0.0, (sorted.len() - 1) as f64);
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let weight = pos - lower as f64;
        knots.push(sorted[lower] * (1.0 - weight) + sorted[upper] * weight);
    }

    knots
}

/// Compute natural spline basis functions at a single point
fn natural_spline_basis_at_point(x: f64, knots: &[f64]) -> Vec<f64> {
    let k = knots.len();
    if k < 2 {
        return vec![1.0];
    }

    let mut basis = Vec::with_capacity(k);

    basis.push(1.0);
    basis.push(x);

    let bk_low = knots[0];
    let bk_high = knots[k - 1];
    let h = bk_high - bk_low;

    for i in 0..(k - 2) {
        let knot = knots[i + 1];
        let d_k = truncated_power(x, knot, 3) / h.powi(2);
        let d_k_upper = truncated_power(x, bk_high, 3) / h.powi(2);
        let d_k1_upper = truncated_power(x, knots[k - 2], 3) / h.powi(2);

        let ratio = (knots[i + 1] - bk_low) / (bk_high - knots[k - 2]).max(1e-10);
        let val = d_k - d_k_upper - ratio * (d_k1_upper - d_k_upper);
        basis.push(val);
    }

    basis
}

/// Truncated power function
fn truncated_power(x: f64, knot: f64, degree: i32) -> f64 {
    if x > knot {
        (x - knot).powi(degree)
    } else {
        0.0
    }
}

/// Transform basis to knot-height parameterization
fn transform_to_knot_heights(
    basis: &[f64],
    n: usize,
    n_basis: usize,
    knots: &[f64],
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
        let basis_at_knot = natural_spline_basis_at_point(knot, knots);
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
            (0..n_basis).map(|col| {
                row.iter()
                    .zip(inverse.iter())
                    .map(|(&basis_value, inverse_row)| basis_value * inverse_row[col])
                    .sum::<f64>()
            })
        })
        .collect();

    if let Some((idx, value)) = transformed
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(value_error(format!(
            "knot-height transform produced non-finite value {value} at index {idx}"
        )));
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
        assert!(result.n_cols > 0);
        assert_eq!(result.basis.len(), result.n_rows * result.n_cols);
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
        assert_eq!(result.n_cols, 5);
        for row in 0..result.n_rows {
            for col in 0..result.n_cols {
                let expected = if row == col { 1.0 } else { 0.0 };
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
        assert_eq!(basis_result.n_cols, 5);

        let coef = vec![10.0, 30.0, 50.0, 70.0, 100.0];
        let predictions = spline.predict(x, coef.clone()).unwrap();

        for (actual, expected) in predictions.iter().zip(coef.iter()) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn test_nsk_rejects_malformed_inputs() {
        initialize_python();

        assert!(nsk(vec![1.0, f64::NAN], Some(3), None, None).is_err());
        assert!(nsk(vec![1.0, 1.0], Some(3), None, None).is_err());
        assert!(nsk(vec![1.0, 2.0], Some(0), None, None).is_err());
        assert!(nsk(vec![1.0, 2.0], Some(3), None, Some((2.0, 2.0))).is_err());
        assert!(nsk(vec![1.0, 2.0], None, Some(vec![f64::NAN]), Some((0.0, 3.0))).is_err());
        assert!(nsk(vec![1.0, 2.0], None, Some(vec![1.5, 1.5]), Some((0.0, 3.0))).is_err());
        assert!(nsk(vec![1.0, 2.0], None, Some(vec![2.5]), Some((0.0, 2.0))).is_err());
        assert!(NaturalSplineKnot::new(None, Some((0.0, 10.0)), Some(1), Some(true)).is_err());

        let err = nsk(vec![0.0, 1.0, 1.0, 1.0, 2.0], Some(4), None, None)
            .expect_err("duplicate computed quantile knots should be rejected");
        assert!(
            err.to_string()
                .contains("computed knots must be strictly increasing")
        );
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
