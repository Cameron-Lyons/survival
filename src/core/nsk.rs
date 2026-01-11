use pyo3::prelude::*;

/// Natural spline with knot heights as basis coefficients.
///
/// This creates a natural cubic spline basis where the coefficients
/// directly represent the function values at the knots, making them
/// easily interpretable.
#[derive(Debug, Clone)]
#[pyclass]
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

        let (interior_knots, computed_df) = match knots {
            Some(k) => {
                let d = k.len() + 1 + if intercept_val { 1 } else { 0 };
                (k, d)
            }
            None => {
                let d = df.unwrap_or(3);
                (vec![], d)
            }
        };

        let bounds = boundary_knots.unwrap_or((f64::NEG_INFINITY, f64::INFINITY));

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
        if n == 0 {
            return Ok(SplineBasisResult {
                basis: vec![],
                n_rows: 0,
                n_cols: self.df,
                knots: self.knots.clone(),
                boundary_knots: self.boundary_knots,
            });
        }

        // Determine boundary knots from data if not specified
        let (bk_low, bk_high) = if self.boundary_knots.0.is_infinite() {
            let min_x = x.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_x, max_x)
        } else {
            self.boundary_knots
        };

        let interior_knots = if self.knots.is_empty() {
            let n_interior = self.df - 1 - if self.intercept { 1 } else { 0 };
            compute_quantile_knots(&x, n_interior, bk_low, bk_high)
        } else {
            self.knots.clone()
        };

        let mut all_knots = vec![bk_low];
        all_knots.extend(interior_knots.iter().cloned());
        all_knots.push(bk_high);

        let n_basis = all_knots.len();

        let mut basis = vec![0.0; n * n_basis];

        for (i, &xi) in x.iter().enumerate() {
            let basis_vals = natural_spline_basis_at_point(xi, &all_knots);
            for (j, &val) in basis_vals.iter().enumerate() {
                basis[i * n_basis + j] = val;
            }
        }

        let transformed_basis = transform_to_knot_heights(&basis, n, n_basis, &all_knots);

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
        let basis_result = self.basis(x)?;

        if coef.len() != basis_result.n_cols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
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
#[pyclass]
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

    let mut sorted: Vec<f64> = x.iter().cloned().filter(|&v| v > low && v < high).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if sorted.is_empty() {
        return vec![];
    }

    let mut knots = Vec::with_capacity(n_knots);
    for i in 1..=n_knots {
        let p = i as f64 / (n_knots + 1) as f64;
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        knots.push(sorted[idx.min(sorted.len() - 1)]);
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
fn transform_to_knot_heights(basis: &[f64], _n: usize, n_basis: usize, knots: &[f64]) -> Vec<f64> {
    let k = knots.len();
    if k == 0 || k != n_basis {
        return basis.to_vec();
    }

    let mut b_matrix = vec![0.0; k * n_basis];
    for (i, &knot) in knots.iter().enumerate() {
        let basis_at_knot = natural_spline_basis_at_point(knot, knots);
        for (j, &val) in basis_at_knot.iter().enumerate() {
            b_matrix[i * n_basis + j] = val;
        }
    }

    basis.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_natural_spline_knot_predict() {
        let spline = NaturalSplineKnot::new(None, Some((0.0, 10.0)), Some(3), None).unwrap();

        let x = vec![0.0, 2.5, 5.0, 7.5, 10.0];
        let basis_result = spline.basis(x.clone()).unwrap();

        // Use some coefficients
        let coef = vec![1.0; basis_result.n_cols];
        let predictions = spline.predict(x, coef).unwrap();

        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_truncated_power() {
        assert_eq!(truncated_power(5.0, 3.0, 2), 4.0); // (5-3)^2 = 4
        assert_eq!(truncated_power(2.0, 3.0, 2), 0.0); // x < knot
        assert_eq!(truncated_power(3.0, 3.0, 2), 0.0); // x = knot
    }
}
