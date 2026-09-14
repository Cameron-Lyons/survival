//! Dense linear algebra shared across the crate.
//!
//! Two families live here:
//!
//! * Faithful ports of R survival's generalised Cholesky routines
//!   (`src/cholesky2.c`, `src/chsolve2.c`, `src/chinv2.c`), which are what
//!   `coxph`, `survreg` and `coxph.wtest` use on information matrices.
//!   They never fail: redundant columns are reported through a zero diagonal
//!   and a reduced rank, exactly as in R.
//! * A partial-pivot LU factorisation for general square systems, with
//!   explicit singularity detection (`Err(SurvivalError::Singular)` or `None`).
//!
//! Matrices are `ndarray::Array2<f64>` indexed `m[[row, col]]`; the C sources
//! index `matrix[i][j]`, which maps to `m[[i, j]]` below.

use crate::constants::{GAUSSIAN_ELIMINATION_TOL, NEAR_ZERO_MATRIX, RIDGE_REGULARIZATION};
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::{Array1, Array2};
use std::borrow::Cow;

pub(crate) fn standardize_row_major_matrix(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    debug_assert_eq!(x.len(), n_rows * n_cols);

    let mut means = vec![0.0; n_cols];
    let mut scales = vec![1.0; n_cols];
    let mut standardized = vec![0.0; n_rows * n_cols];

    for col in 0..n_cols {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for row in 0..n_rows {
            let value = x[row * n_cols + col];
            sum += value;
            sum_sq += value * value;
        }

        means[col] = sum / n_rows as f64;
        let variance = sum_sq / n_rows as f64 - means[col] * means[col];
        scales[col] = variance.sqrt().max(crate::constants::DIVISION_FLOOR);
        let inv_scale = 1.0 / scales[col];

        for row in 0..n_rows {
            standardized[row * n_cols + col] = (x[row * n_cols + col] - means[col]) * inv_scale;
        }
    }

    (standardized, means, scales)
}

pub(crate) fn standardize_or_borrow_row_major_matrix(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    standardize: bool,
) -> (Cow<'_, [f64]>, Vec<f64>, Vec<f64>) {
    if standardize {
        let (standardized, means, scales) = standardize_row_major_matrix(x, n_rows, n_cols);
        (Cow::Owned(standardized), means, scales)
    } else {
        (Cow::Borrowed(x), vec![0.0; n_cols], vec![1.0; n_cols])
    }
}

fn require_square(matrix: &Array2<f64>, context: &str) -> SurvivalResult<usize> {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return Err(SurvivalError::invalid_input(format!(
            "{context}: matrix must be square, got {rows} x {cols}"
        )));
    }
    Ok(rows)
}

fn require_finite(matrix: &Array2<f64>, context: &str) -> SurvivalResult<()> {
    if let Some(((row, col), value)) = matrix.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "{context}: matrix contains non-finite value {value} at [{row}, {col}]"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Generalised Cholesky (R survival: cholesky2.c / chsolve2.c / chinv2.c)
// ---------------------------------------------------------------------------

/// Port of R survival's `cholesky2` (`src/cholesky2.c`): the generalised
/// Cholesky decomposition `C = F D F'` of a symmetric matrix, where `F` is
/// lower triangular with unit diagonal and `D` is diagonal.
///
/// Reads the diagonal and the upper triangle (`m[[i, j]]`, `i < j`); on return
/// `D` occupies the diagonal, `F` (without its unit diagonal) the lower
/// triangle `m[[j, i]]`, `j > i`, and the upper triangle is left undisturbed.
///
/// `toler` is R's `toler.chol` (`coxph.control` default `eps^0.75`,
/// `coxph.wtest` default `1e-9`): a pivot below `toler * max(diag)` marks the
/// column redundant and zeroes its diagonal. A non-finite diagonal counts as
/// zero. Returns the rank when the matrix is non-negative definite, or minus
/// the rank when a pivot was more negative than `-8 * eps`.
///
/// Panics if `matrix` is not square.
pub(crate) fn cholesky2(matrix: &mut Array2<f64>, toler: f64) -> i32 {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "cholesky2 requires a square matrix");

    let mut eps = 0.0_f64;
    for i in 0..n {
        if matrix[[i, i]] > eps {
            eps = matrix[[i, i]];
        }
        for j in (i + 1)..n {
            matrix[[j, i]] = matrix[[i, j]];
        }
    }
    eps = if eps == 0.0 { toler } else { eps * toler };

    let mut rank = 0_i32;
    let mut nonneg = 1_i32;
    for i in 0..n {
        let pivot = matrix[[i, i]];
        if !pivot.is_finite() || pivot < eps {
            matrix[[i, i]] = 0.0;
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in (i + 1)..n {
                let temp = matrix[[j, i]] / pivot;
                matrix[[j, i]] = temp;
                matrix[[j, j]] -= temp * temp * pivot;
                for k in (j + 1)..n {
                    matrix[[k, j]] -= temp * matrix[[k, i]];
                }
            }
        }
    }
    rank * nonneg
}

/// Port of R survival's `chsolve2` (`src/chsolve2.c`): solves `A b = y`
/// given the [`cholesky2`] factorisation of `A` in `chol`, overwriting `y`
/// with `b`. Components belonging to redundant columns (zero diagonal) are
/// set to zero, which is how `coxph` keeps iterating past a singular
/// information matrix.
///
/// Panics if `y.len()` differs from the matrix order.
// Canonical helper; `regression/cox_optimizer.rs` and `regression/coxph_wtest.rs`
// still carry private copies and are expected to migrate to this one.
#[allow(dead_code)]
pub(crate) fn chsolve2(chol: &Array2<f64>, y: &mut [f64]) {
    let n = chol.nrows();
    assert_eq!(n, chol.ncols(), "chsolve2 requires a square matrix");
    assert_eq!(y.len(), n, "chsolve2 right-hand side length must match");

    for i in 0..n {
        let mut temp = y[i];
        for (j, &known) in y.iter().enumerate().take(i) {
            temp -= known * chol[[i, j]];
        }
        y[i] = temp;
    }

    for i in (0..n).rev() {
        if chol[[i, i]] == 0.0 {
            y[i] = 0.0;
        } else {
            let mut temp = y[i] / chol[[i, i]];
            for (j, &known) in y.iter().enumerate().skip(i + 1) {
                temp -= known * chol[[j, i]];
            }
            y[i] = temp;
        }
    }
}

/// Port of R survival's `chinv2` (`src/chinv2.c`): inverts a matrix given its
/// [`cholesky2`] factorisation. On return the upper triangle and diagonal
/// (`m[[i, j]]`, `i <= j`) hold `(F D F')^{-1}`; below the diagonal is
/// `F^{-1}`. Rows and columns of redundant variables are zeroed (R's `coxph`
/// reports those coefficients as `NA`). Callers wanting the full symmetric
/// inverse copy the upper triangle into the lower one, as `coxfit6.c` does;
/// [`symmetric_inverse_via_cholesky`] performs that step.
///
/// Panics if `matrix` is not square.
pub(crate) fn chinv2(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    assert_eq!(n, matrix.ncols(), "chinv2 requires a square matrix");

    for i in 0..n {
        if matrix[[i, i]] > 0.0 {
            matrix[[i, i]] = 1.0 / matrix[[i, i]];
            for j in (i + 1)..n {
                matrix[[j, i]] = -matrix[[j, i]];
                for k in 0..i {
                    let update = matrix[[j, i]] * matrix[[i, k]];
                    matrix[[j, k]] += update;
                }
            }
        }
    }

    for i in 0..n {
        if matrix[[i, i]] == 0.0 {
            for j in 0..i {
                matrix[[j, i]] = 0.0;
            }
            for j in i..n {
                matrix[[i, j]] = 0.0;
            }
        } else {
            for j in (i + 1)..n {
                let temp = matrix[[j, i]] * matrix[[j, j]];
                matrix[[i, j]] = temp;
                for k in i..j {
                    let update = temp * matrix[[j, k]];
                    matrix[[i, k]] += update;
                }
            }
        }
    }
}

/// Result of [`symmetric_inverse_via_cholesky`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SymmetricInverse {
    /// Full symmetric (generalised) inverse; rows and columns listed in
    /// `singular_columns` are zero.
    pub inverse: Array2<f64>,
    /// Number of non-redundant columns (absolute value of the `cholesky2`
    /// return).
    pub rank: usize,
    /// `false` when `cholesky2` found a pivot more negative than `-8 * eps`,
    /// i.e. the matrix is not non-negative definite.
    pub non_negative_definite: bool,
    /// Zero-based columns whose diagonal of the inverse is zero — R's
    /// `which.sing <- diag(var) == 0` in `coxph.fit`.
    pub singular_columns: Vec<usize>,
}

/// Inverts a symmetric matrix the way `coxfit6.c` inverts the information
/// matrix: `cholesky2`, `chinv2`, then copy the upper triangle into the lower
/// one. Redundant columns are zeroed and reported rather than turned into an
/// error, matching R. Fails only for malformed input (non-square or
/// non-finite), mirroring `coxph.wtest`'s `"infinite argument"` check.
pub(crate) fn symmetric_inverse_via_cholesky(
    matrix: &Array2<f64>,
    toler: f64,
) -> SurvivalResult<SymmetricInverse> {
    const CONTEXT: &str = "symmetric inverse";
    let n = require_square(matrix, CONTEXT)?;
    require_finite(matrix, CONTEXT)?;

    let mut work = matrix.clone();
    let flag = cholesky2(&mut work, toler);
    chinv2(&mut work);
    for i in 0..n {
        for j in 0..i {
            work[[i, j]] = work[[j, i]];
        }
    }
    let singular_columns = (0..n).filter(|&i| work[[i, i]] == 0.0).collect();

    Ok(SymmetricInverse {
        inverse: work,
        rank: flag.unsigned_abs() as usize,
        non_negative_definite: flag >= 0,
        singular_columns,
    })
}

// ---------------------------------------------------------------------------
// Partial-pivot LU for general square systems
// ---------------------------------------------------------------------------

/// `P A = L U` with partial (row) pivoting, stored row-major. Unlike the
/// Cholesky routines above this applies to any square matrix, and singular
/// systems are reported rather than patched: a pivot with absolute value at
/// most `max|a_ij| * GAUSSIAN_ELIMINATION_TOL` is an error naming the column
/// (this is the analogue of R's `solve()` "system is computationally
/// singular").
#[derive(Debug, Clone)]
pub(crate) struct LuDecomposition {
    factors: Vec<f64>,
    swaps: Vec<usize>,
    n: usize,
}

impl LuDecomposition {
    const CONTEXT: &'static str = "LU factorisation";

    pub(crate) fn decompose(matrix: &Array2<f64>) -> SurvivalResult<Self> {
        let n = require_square(matrix, Self::CONTEXT)?;
        require_finite(matrix, Self::CONTEXT)?;
        if n == 0 {
            return Ok(Self {
                factors: Vec::new(),
                swaps: Vec::new(),
                n: 0,
            });
        }

        let mut factors: Vec<f64> = matrix.iter().copied().collect();
        let scale = factors.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if scale == 0.0 {
            return Err(SurvivalError::singular_columns(
                Self::CONTEXT,
                (0..n).collect(),
            ));
        }
        let pivot_tolerance = scale * GAUSSIAN_ELIMINATION_TOL;
        let mut swaps = Vec::with_capacity(n);

        for pivot_col in 0..n {
            let mut pivot_row = pivot_col;
            let mut pivot_abs = factors[pivot_col * n + pivot_col].abs();
            for row in (pivot_col + 1)..n {
                let candidate = factors[row * n + pivot_col].abs();
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = row;
                }
            }
            if !pivot_abs.is_finite() || pivot_abs <= pivot_tolerance {
                return Err(SurvivalError::singular_columns(
                    Self::CONTEXT,
                    vec![pivot_col],
                ));
            }

            swaps.push(pivot_row);
            if pivot_row != pivot_col {
                for col in 0..n {
                    factors.swap(pivot_col * n + col, pivot_row * n + col);
                }
            }

            let pivot = factors[pivot_col * n + pivot_col];
            for row in (pivot_col + 1)..n {
                let multiplier_index = row * n + pivot_col;
                let multiplier = factors[multiplier_index] / pivot;
                factors[multiplier_index] = multiplier;

                let row_start = row * n;
                let pivot_start = pivot_col * n;
                for col in (pivot_col + 1)..n {
                    factors[row_start + col] =
                        (-multiplier).mul_add(factors[pivot_start + col], factors[row_start + col]);
                }
            }
        }

        Ok(Self { factors, swaps, n })
    }

    /// Solves `A x = rhs`.
    pub(crate) fn solve(&self, rhs: &[f64]) -> SurvivalResult<Vec<f64>> {
        if rhs.len() != self.n {
            return Err(SurvivalError::invalid_input(format!(
                "{}: right-hand side has length {}, expected {}",
                Self::CONTEXT,
                rhs.len(),
                self.n
            )));
        }
        if let Some((index, value)) = rhs.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(SurvivalError::invalid_input(format!(
                "{}: right-hand side contains non-finite value {value} at index {index}",
                Self::CONTEXT
            )));
        }
        if self.n == 0 {
            return Ok(Vec::new());
        }

        let mut solution = rhs.to_vec();
        for (row, &swap_row) in self.swaps.iter().enumerate() {
            if row != swap_row {
                solution.swap(row, swap_row);
            }
        }

        for row in 0..self.n {
            let row_start = row * self.n;
            let mut value = solution[row];
            for (col, &known_value) in solution.iter().take(row).enumerate() {
                value = (-self.factors[row_start + col]).mul_add(known_value, value);
            }
            solution[row] = value;
        }

        for row in (0..self.n).rev() {
            let row_start = row * self.n;
            let mut value = solution[row];
            for (col, &known_value) in solution.iter().enumerate().skip(row + 1) {
                value = (-self.factors[row_start + col]).mul_add(known_value, value);
            }
            solution[row] = value / self.factors[row_start + row];
        }

        if solution.iter().any(|value| !value.is_finite()) {
            return Err(SurvivalError::computation(format!(
                "{}: solution overflowed to a non-finite value",
                Self::CONTEXT
            )));
        }
        Ok(solution)
    }

    /// Dense inverse, one solve per unit vector.
    pub(crate) fn inverse(&self) -> SurvivalResult<Array2<f64>> {
        let mut inverse = Array2::zeros((self.n, self.n));
        let mut rhs = vec![0.0; self.n];
        for col in 0..self.n {
            rhs[col] = 1.0;
            let solution = self.solve(&rhs)?;
            rhs[col] = 0.0;
            for (row, value) in solution.into_iter().enumerate() {
                inverse[[row, col]] = value;
            }
        }
        Ok(inverse)
    }
}

/// Solves `A x = b` by partial-pivot LU. `None` when `A` is not square,
/// contains non-finite values, is singular, or `b` has the wrong length —
/// callers map that to their own error; use [`LuDecomposition`] directly for
/// the structured [`SurvivalError`].
pub(crate) fn lu_solve(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    let factorization = LuDecomposition::decompose(matrix).ok()?;
    factorization
        .solve(vector.as_slice()?)
        .ok()
        .map(Array1::from_vec)
}

/// Dense inverse by partial-pivot LU; `Err(SurvivalError::Singular)` names the
/// first column at which elimination broke down.
pub(crate) fn lu_inverse(matrix: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    LuDecomposition::decompose(matrix)?.inverse()
}

/// `Option` form of [`lu_inverse`] for callers that only need success/failure.
pub(crate) fn matrix_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    lu_inverse(matrix).ok()
}

/// `Vec<Vec<f64>>` adapter over [`lu_inverse`]. `None` for an empty, ragged,
/// non-square or singular matrix.
pub(crate) fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|row| row.len() != n) {
        return None;
    }
    let flat: Vec<f64> = mat.iter().flatten().copied().collect();
    let matrix = Array2::from_shape_vec((n, n), flat).ok()?;
    let inverse = matrix_inverse(&matrix)?;
    Some(inverse.outer_iter().map(|row| row.to_vec()).collect())
}

/// Always-successful inverse of a flattened row-major `n x n` information or
/// covariance matrix, kept for residual and frailty code that reports rather
/// than fails on a singular fit.
///
/// A non-singular matrix gets its exact LU inverse. A singular one gets R's
/// `chinv2` generalised inverse of the symmetrised matrix (see
/// [`symmetric_inverse_via_cholesky`]): the redundant rows and columns are
/// zero, so downstream quantities for those coefficients are zero rather than
/// garbage, exactly as `residuals.coxph` behaves for an `NA` coefficient.
///
/// Panics if `a.len() != n * n`.
pub(crate) fn invert_flat_square_matrix_with_fallback(a: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n, "flattened matrix must hold n * n entries");
    if n == 0 {
        return Vec::new();
    }
    let matrix =
        Array2::from_shape_vec((n, n), a.to_vec()).expect("length was checked against n * n above");
    if let Ok(inverse) = lu_inverse(&matrix) {
        return inverse.into_raw_vec_and_offset().0;
    }

    let mut symmetric = matrix.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            let average = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            symmetric[[i, j]] = average;
            symmetric[[j, i]] = average;
        }
    }
    match symmetric_inverse_via_cholesky(&symmetric, GAUSSIAN_ELIMINATION_TOL) {
        Ok(result) => result.inverse.into_raw_vec_and_offset().0,
        // Only non-finite entries reach here; every column is then redundant.
        Err(_) => vec![0.0; n * n],
    }
}

/// LU solve that retries with a ridge (`max|a_ij| * RIDGE_REGULARIZATION` on
/// the diagonal) when the system is singular. This is not an R algorithm:
/// `survreg` steps through a singular Hessian with `cholesky2`/`chsolve2`
/// instead. Kept only for `parametric_survival`, whose Newton loop still
/// relies on the damped step; new callers should use [`lu_solve`] or
/// [`chsolve2`] and handle singularity explicitly.
pub(crate) fn regularized_lu_solve(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> SurvivalResult<Array1<f64>> {
    const CONTEXT: &str = "regularized LU solve";
    let n = require_square(matrix, CONTEXT)?;
    if n == 0 {
        if vector.is_empty() {
            return Ok(Array1::zeros(0));
        }
        return Err(SurvivalError::invalid_input(format!(
            "{CONTEXT}: empty matrix with a right-hand side of length {}",
            vector.len()
        )));
    }

    let max_val = matrix.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_val < NEAR_ZERO_MATRIX {
        return Err(SurvivalError::singular(CONTEXT));
    }

    match LuDecomposition::decompose(matrix) {
        Ok(factorization) => {
            let rhs = vector.as_slice().ok_or_else(|| {
                SurvivalError::invalid_input(format!("{CONTEXT}: non-contiguous right-hand side"))
            })?;
            factorization.solve(rhs).map(Array1::from_vec)
        }
        Err(SurvivalError::Singular { .. }) => {
            let ridge = max_val * RIDGE_REGULARIZATION;
            let mut reg_matrix = matrix.clone();
            for i in 0..n {
                reg_matrix[[i, i]] += ridge;
            }
            lu_solve(&reg_matrix, vector).ok_or_else(|| SurvivalError::singular(CONTEXT))
        }
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, col), value) in actual.indexed_iter() {
            assert!(
                (value - expected[[row, col]]).abs() <= tol,
                "mismatch at [{row}, {col}]: expected {}, got {value}",
                expected[[row, col]]
            );
        }
    }

    // Reference values from R survival 3.8-11:
    //   A <- matrix(c(4,2,2, 2,5,3, 2,3,6), 3, 3); b <- 1:3
    //   coxph.wtest(A, b)$solve  -> -0.046875  0.15625  0.4375
    //   coxph.wtest(A, b)$test   -> 1.578125, df = 3
    //   solve(A) -> [0.328125 -0.09375 -0.0625; -0.09375 0.3125 -0.125; -0.0625 -0.125 0.25]
    fn spd() -> Array2<f64> {
        arr2(&[[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
    }

    fn spd_inverse() -> Array2<f64> {
        arr2(&[
            [0.328125, -0.09375, -0.0625],
            [-0.09375, 0.3125, -0.125],
            [-0.0625, -0.125, 0.25],
        ])
    }

    //   B <- matrix(c(4,2,6, 2,5,7, 6,7,13), 3, 3)   # column 3 = column 1 + column 2
    //   coxph.wtest(B, 1:3)$df -> 2 ; $solve -> 0.0625 0.375 0 ; $test -> 0.8125
    //   solve(B[1:2, 1:2]) -> [0.3125 -0.125; -0.125 0.25]
    fn rank_deficient() -> Array2<f64> {
        arr2(&[[4.0, 2.0, 6.0], [2.0, 5.0, 7.0], [6.0, 7.0, 13.0]])
    }

    #[test]
    fn cholesky2_factors_spd_matrix_as_fdft() {
        let mut work = spd();
        assert_eq!(cholesky2(&mut work, 1e-9), 3);

        // D = diag(4, 4, 4); F below the diagonal.
        for i in 0..3 {
            assert_close(work[[i, i]], 4.0, 1e-12);
        }
        assert_close(work[[1, 0]], 0.5, 1e-12);
        assert_close(work[[2, 0]], 0.5, 1e-12);
        assert_close(work[[2, 1]], 0.5, 1e-12);
        // Upper triangle is untouched.
        assert_eq!(work[[0, 1]], 2.0);
        assert_eq!(work[[0, 2]], 2.0);
        assert_eq!(work[[1, 2]], 3.0);

        // Reassemble F D F' and compare with the input.
        let mut f = Array2::eye(3);
        let mut d = Array2::zeros((3, 3));
        for i in 0..3 {
            d[[i, i]] = work[[i, i]];
            for j in 0..i {
                f[[i, j]] = work[[i, j]];
            }
        }
        let reassembled = f.dot(&d).dot(&f.t());
        assert_matrix_close(&reassembled, &spd(), 1e-12);
    }

    #[test]
    fn cholesky2_reads_only_the_upper_triangle() {
        let mut work = spd();
        work[[1, 0]] = 99.0;
        work[[2, 0]] = -99.0;
        work[[2, 1]] = 42.0;
        let mut reference = spd();
        assert_eq!(cholesky2(&mut work, 1e-9), cholesky2(&mut reference, 1e-9));
        assert_matrix_close(&work, &reference, 0.0);
    }

    #[test]
    fn cholesky2_reports_rank_and_zeroes_redundant_column() {
        let mut work = rank_deficient();
        assert_eq!(cholesky2(&mut work, 1e-9), 2);
        assert_eq!(work[[2, 2]], 0.0);
        assert_close(work[[0, 0]], 4.0, 1e-12);
        assert_close(work[[1, 1]], 4.0, 1e-12);
        assert_close(work[[2, 1]], 1.0, 1e-12);
        assert_close(work[[2, 0]], 1.5, 1e-12);
    }

    #[test]
    fn cholesky2_flags_indefinite_matrix_with_negative_rank() {
        // C <- matrix(c(1,2, 2,1), 2, 2): coxph.wtest(C, c(1,1))$df == 1, test == 1
        let mut work = arr2(&[[1.0, 2.0], [2.0, 1.0]]);
        assert_eq!(cholesky2(&mut work, 1e-9), -1);
        assert_eq!(work[[1, 1]], 0.0);

        let mut y = vec![1.0, 1.0];
        chsolve2(&work, &mut y);
        assert_eq!(y, vec![1.0, 0.0]);
    }

    #[test]
    fn cholesky2_treats_non_finite_and_zero_diagonals_like_r() {
        let mut work = arr2(&[[f64::NAN, 0.0], [0.0, 2.0]]);
        assert_eq!(cholesky2(&mut work, 1e-9), 1);
        assert_eq!(work[[0, 0]], 0.0);
        assert_close(work[[1, 1]], 2.0, 0.0);

        // No positive diagonals: eps falls back to toler itself.
        let mut zeros = Array2::zeros((2, 2));
        assert_eq!(cholesky2(&mut zeros, 1e-9), 0);

        let mut empty = Array2::zeros((0, 0));
        assert_eq!(cholesky2(&mut empty, 1e-9), 0);
    }

    #[test]
    fn chsolve2_matches_coxph_wtest_solve() {
        let mut work = spd();
        cholesky2(&mut work, 1e-9);
        let mut y = vec![1.0, 2.0, 3.0];
        chsolve2(&work, &mut y);
        assert_close(y[0], -0.046875, 1e-12);
        assert_close(y[1], 0.15625, 1e-12);
        assert_close(y[2], 0.4375, 1e-12);
        let wald: f64 = y.iter().zip([1.0, 2.0, 3.0]).map(|(s, b)| s * b).sum();
        assert_close(wald, 1.578125, 1e-12);

        let mut work = rank_deficient();
        cholesky2(&mut work, 1e-9);
        let mut y = vec![1.0, 2.0, 3.0];
        chsolve2(&work, &mut y);
        assert_close(y[0], 0.0625, 1e-12);
        assert_close(y[1], 0.375, 1e-12);
        assert_eq!(y[2], 0.0);
        let wald: f64 = y.iter().zip([1.0, 2.0, 3.0]).map(|(s, b)| s * b).sum();
        assert_close(wald, 0.8125, 1e-12);
    }

    #[test]
    fn chinv2_upper_triangle_holds_inverse() {
        let mut work = spd();
        cholesky2(&mut work, 1e-9);
        chinv2(&mut work);
        let expected = spd_inverse();
        for i in 0..3 {
            for j in i..3 {
                assert_close(work[[i, j]], expected[[i, j]], 1e-12);
            }
        }
    }

    #[test]
    fn chinv2_zeroes_redundant_rows_and_columns() {
        let mut work = rank_deficient();
        cholesky2(&mut work, 1e-9);
        chinv2(&mut work);
        assert_close(work[[0, 0]], 0.3125, 1e-12);
        assert_close(work[[0, 1]], -0.125, 1e-12);
        assert_close(work[[1, 1]], 0.25, 1e-12);
        assert_eq!(work[[0, 2]], 0.0);
        assert_eq!(work[[1, 2]], 0.0);
        assert_eq!(work[[2, 2]], 0.0);
    }

    #[test]
    fn symmetric_inverse_via_cholesky_returns_full_inverse_and_singular_report() {
        let result = symmetric_inverse_via_cholesky(&spd(), 1e-9).unwrap();
        assert_eq!(result.rank, 3);
        assert!(result.non_negative_definite);
        assert!(result.singular_columns.is_empty());
        assert_matrix_close(&result.inverse, &spd_inverse(), 1e-12);
        assert_matrix_close(&result.inverse, &result.inverse.t().to_owned(), 0.0);

        let result = symmetric_inverse_via_cholesky(&rank_deficient(), 1e-9).unwrap();
        assert_eq!(result.rank, 2);
        assert!(result.non_negative_definite);
        assert_eq!(result.singular_columns, vec![2]);
        let expected = arr2(&[[0.3125, -0.125, 0.0], [-0.125, 0.25, 0.0], [0.0, 0.0, 0.0]]);
        assert_matrix_close(&result.inverse, &expected, 1e-12);

        let result =
            symmetric_inverse_via_cholesky(&arr2(&[[1.0, 2.0], [2.0, 1.0]]), 1e-9).unwrap();
        assert!(!result.non_negative_definite);
        assert_eq!(result.rank, 1);
        assert_eq!(result.singular_columns, vec![1]);
    }

    #[test]
    fn symmetric_inverse_via_cholesky_rejects_malformed_input() {
        let nonsquare = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(matches!(
            symmetric_inverse_via_cholesky(&nonsquare, 1e-9),
            Err(SurvivalError::InvalidInput(_))
        ));
        let err = symmetric_inverse_via_cholesky(&arr2(&[[1.0, f64::NAN], [0.0, 1.0]]), 1e-9)
            .unwrap_err();
        assert!(err.to_string().contains("non-finite value"));
    }

    #[test]
    fn lu_decomposition_solves_and_inverts() {
        let matrix = arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        let lu = LuDecomposition::decompose(&matrix).unwrap();
        let solution = lu.solve(&[3.0, 4.0]).unwrap();
        assert_close(2.0 * solution[0] + solution[1], 3.0, 1e-12);
        assert_close(solution[0] + 3.0 * solution[1], 4.0, 1e-12);

        let inverse = lu.inverse().unwrap();
        assert_matrix_close(&matrix.dot(&inverse), &Array2::eye(2), 1e-12);

        let empty = LuDecomposition::decompose(&Array2::zeros((0, 0))).unwrap();
        assert!(empty.solve(&[]).unwrap().is_empty());
        assert_eq!(empty.inverse().unwrap().dim(), (0, 0));
    }

    #[test]
    fn lu_decomposition_reports_singular_column() {
        let singular = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        match LuDecomposition::decompose(&singular) {
            Err(SurvivalError::Singular { columns, .. }) => assert_eq!(columns, vec![1]),
            other => panic!("expected singular error, got {other:?}"),
        }
        match LuDecomposition::decompose(&Array2::zeros((2, 2))) {
            Err(SurvivalError::Singular { columns, .. }) => assert_eq!(columns, vec![0, 1]),
            other => panic!("expected singular error, got {other:?}"),
        }
        assert!(matches!(
            lu_inverse(&singular),
            Err(SurvivalError::Singular { .. })
        ));
        assert!(matrix_inverse(&singular).is_none());
    }

    #[test]
    fn lu_decomposition_rejects_malformed_input() {
        let nonsquare = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(matches!(
            LuDecomposition::decompose(&nonsquare),
            Err(SurvivalError::InvalidInput(_))
        ));
        assert!(matches!(
            LuDecomposition::decompose(&arr2(&[[1.0, f64::INFINITY], [0.0, 1.0]])),
            Err(SurvivalError::InvalidInput(_))
        ));
        let lu = LuDecomposition::decompose(&Array2::eye(2)).unwrap();
        assert!(matches!(
            lu.solve(&[1.0]),
            Err(SurvivalError::InvalidInput(_))
        ));
        assert!(matches!(
            lu.solve(&[1.0, f64::NAN]),
            Err(SurvivalError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_lu_solve() {
        let matrix = arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        let vector = Array1::from_vec(vec![3.0, 4.0]);
        let result = lu_solve(&matrix, &vector).unwrap();
        assert_close(2.0 * result[0] + result[1], 3.0, 1e-10);
        assert_close(result[0] + 3.0 * result[1], 4.0, 1e-10);
    }

    #[test]
    fn test_lu_solve_uses_partial_pivoting() {
        let matrix = arr2(&[[0.0, 2.0], [1.0, 3.0]]);
        let vector = Array1::from_vec(vec![4.0, 5.0]);
        let result = lu_solve(&matrix, &vector).unwrap();
        assert_close(result[0], -1.0, 1e-12);
        assert_close(result[1], 2.0, 1e-12);
    }

    #[test]
    fn test_lu_solve_rejects_singular_and_malformed_systems() {
        let singular = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        let rhs = Array1::from_vec(vec![1.0, 2.0]);
        assert!(lu_solve(&singular, &rhs).is_none());

        let nonsquare = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();
        assert!(lu_solve(&nonsquare, &rhs).is_none());
        assert!(lu_solve(&arr2(&[[1.0, 0.0], [0.0, 1.0]]), &Array1::zeros(1)).is_none());
    }

    #[test]
    fn test_matrix_inverse_nontrivial_product_is_identity() {
        let matrix = arr2(&[[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]]);
        let inverse = matrix_inverse(&matrix).unwrap();
        assert_matrix_close(&matrix.dot(&inverse), &Array2::eye(3), 1e-10);
        assert_eq!(
            matrix_inverse(&Array2::zeros((0, 0))).unwrap().dim(),
            (0, 0)
        );
    }

    #[test]
    fn invert_matrix_vec_adapter_matches_lu_inverse() {
        let rows = vec![
            vec![4.0, 2.0, 2.0],
            vec![2.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let inverse = invert_matrix(&rows).unwrap();
        let expected = spd_inverse();
        for (i, row) in inverse.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_close(*value, expected[[i, j]], 1e-12);
            }
        }

        assert!(invert_matrix(&[]).is_none());
        assert!(invert_matrix(&[vec![1.0, 2.0], vec![3.0]]).is_none());
        assert!(invert_matrix(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
    }

    #[test]
    fn flat_inverse_uses_lu_then_chinv2_generalised_inverse() {
        let flat: Vec<f64> = spd().iter().copied().collect();
        let inverse = invert_flat_square_matrix_with_fallback(&flat, 3);
        let expected: Vec<f64> = spd_inverse().iter().copied().collect();
        for (actual, expected) in inverse.iter().zip(expected) {
            assert_close(*actual, expected, 1e-12);
        }

        let flat: Vec<f64> = rank_deficient().iter().copied().collect();
        let inverse = invert_flat_square_matrix_with_fallback(&flat, 3);
        let expected = [0.3125, -0.125, 0.0, -0.125, 0.25, 0.0, 0.0, 0.0, 0.0];
        for (actual, expected) in inverse.iter().zip(expected) {
            assert_close(*actual, expected, 1e-12);
        }

        assert!(invert_flat_square_matrix_with_fallback(&[], 0).is_empty());
        assert_eq!(
            invert_flat_square_matrix_with_fallback(&[4.0], 1),
            vec![0.25]
        );
        assert_eq!(
            invert_flat_square_matrix_with_fallback(&[0.0], 1),
            vec![0.0]
        );
    }

    #[test]
    fn test_regularized_lu_solve_identity() {
        let matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = regularized_lu_solve(&matrix, &vector).unwrap();
        assert_close(result[0], 1.0, 1e-10);
        assert_close(result[1], 2.0, 1e-10);
    }

    #[test]
    fn test_regularized_lu_solve_empty() {
        let matrix: Array2<f64> = Array2::zeros((0, 0));
        let vector: Array1<f64> = Array1::zeros(0);
        let result = regularized_lu_solve(&matrix, &vector).unwrap();
        assert_eq!(result.len(), 0);
        assert!(regularized_lu_solve(&matrix, &Array1::zeros(1)).is_err());
    }

    #[test]
    fn test_regularized_lu_solve_near_zero_matrix() {
        let matrix = arr2(&[[1e-15, 0.0], [0.0, 1e-15]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        assert!(matches!(
            regularized_lu_solve(&matrix, &vector),
            Err(SurvivalError::Singular { .. })
        ));
    }

    #[test]
    fn test_regularized_lu_solve_ridges_singular_system() {
        let singular = arr2(&[[1.0, 2.0], [2.0, 4.0]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = regularized_lu_solve(&singular, &vector).unwrap();
        // The ridged system (A + ridge I) x = b is consistent with A x ~ b.
        assert_close(result[0] + 2.0 * result[1], 1.0, 1e-4);
    }

    #[test]
    fn standardize_row_major_matrix_centers_and_scales_columns() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (standardized, means, scales) = standardize_row_major_matrix(&x, 3, 2);

        assert_eq!(standardized.len(), x.len());
        assert_eq!(means, vec![3.0, 4.0]);
        assert!((scales[0] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((scales[1] - (8.0_f64 / 3.0).sqrt()).abs() < 1e-12);

        for col in 0..2 {
            let column_sum: f64 = (0..3).map(|row| standardized[row * 2 + col]).sum();
            assert!(column_sum.abs() < 1e-12);
        }
    }

    #[test]
    fn standardize_or_borrow_row_major_matrix_borrows_when_disabled() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (matrix, means, scales) = standardize_or_borrow_row_major_matrix(&x, 2, 2, false);

        assert!(matches!(matrix, Cow::Borrowed(_)));
        assert_eq!(matrix.as_ref(), x.as_slice());
        assert_eq!(means, vec![0.0, 0.0]);
        assert_eq!(scales, vec![1.0, 1.0]);
    }

    #[test]
    fn standardize_or_borrow_row_major_matrix_owns_when_enabled() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (matrix, means, scales) = standardize_or_borrow_row_major_matrix(&x, 2, 2, true);

        assert!(matches!(matrix, Cow::Owned(_)));
        assert_eq!(means, vec![2.0, 3.0]);
        assert_eq!(scales, vec![1.0, 1.0]);
    }
}
