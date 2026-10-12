use crate::constants::GAUSSIAN_ELIMINATION_TOL;
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::Array2;
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

pub(crate) fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.outer_iter().map(|row| row.to_vec()).collect()
}

pub(crate) fn matrix_from_rows(rows: &[Vec<f64>], name: &str) -> SurvivalResult<Array2<f64>> {
    let ncols = rows.first().map_or(0, Vec::len);
    matrix_from_rows_with_columns(rows, ncols, name)
}

pub(crate) fn matrix_from_rows_with_columns(
    rows: &[Vec<f64>],
    ncols: usize,
    name: &str,
) -> SurvivalResult<Array2<f64>> {
    if rows.iter().any(|row| row.len() != ncols) {
        return Err(SurvivalError::invalid_input(format!(
            "{name} must be rectangular"
        )));
    }
    Array2::from_shape_vec(
        (rows.len(), ncols),
        rows.iter().flatten().copied().collect(),
    )
    .map_err(|err| SurvivalError::invalid_input(err.to_string()))
}

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

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SymmetricInverse {
    pub inverse: Array2<f64>,
    pub rank: usize,
    pub non_negative_definite: bool,
    pub singular_columns: Vec<usize>,
}

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
        let anorm = (0..n)
            .map(|col| (0..n).map(|row| factors[row * n + col].abs()).sum::<f64>())
            .fold(0.0, f64::max);
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
            if pivot_abs == 0.0 || !pivot_abs.is_finite() {
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
            let (head, tail) = factors.split_at_mut((pivot_col + 1) * n);
            let pivot_row = &head[pivot_col * n + pivot_col + 1..(pivot_col + 1) * n];
            for row in tail.chunks_exact_mut(n) {
                let multiplier = row[pivot_col] / pivot;
                row[pivot_col] = multiplier;
                for (value, &above) in row[pivot_col + 1..].iter_mut().zip(pivot_row) {
                    *value -= multiplier * above;
                }
            }
        }

        let lu = Self { factors, swaps, n };
        let rcond = lu.reciprocal_condition(anorm);
        if rcond < f64::EPSILON {
            return Err(SurvivalError::singular(format!(
                "{} (reciprocal condition number = {rcond:.5e})",
                Self::CONTEXT
            )));
        }
        Ok(lu)
    }

    fn reciprocal_condition(&self, anorm: f64) -> f64 {
        let inverse_norm = self.inverse_norm_estimate();
        if inverse_norm.is_nan() {
            return 0.0;
        }
        1.0 / (anorm * inverse_norm)
    }

    fn inverse_norm_estimate(&self) -> f64 {
        let n = self.n;
        let l1 = |values: &[f64]| values.iter().map(|value| value.abs()).sum::<f64>();
        let signs = |values: &[f64]| -> Vec<f64> {
            values
                .iter()
                .map(|&value| if value >= 0.0 { 1.0 } else { -1.0 })
                .collect()
        };
        let argmax = |values: &[f64]| -> usize {
            let mut best = 0;
            for (index, value) in values.iter().enumerate() {
                if value.abs() > values[best].abs() {
                    best = index;
                }
            }
            best
        };
        let mut x = vec![1.0 / n as f64; n];
        self.substitute(&mut x);
        let mut estimate = l1(&x);
        if n == 1 || estimate.is_nan() {
            return estimate;
        }
        let mut sign = signs(&x);
        let mut z = sign.clone();
        self.substitute_transpose(&mut z);
        let mut j = argmax(&z);
        for _ in 1..5 {
            let mut y = vec![0.0; n];
            y[j] = 1.0;
            self.substitute(&mut y);
            let previous = estimate;
            estimate = l1(&y);
            let next_sign = signs(&y);
            if next_sign == sign || estimate <= previous {
                estimate = estimate.max(previous);
                break;
            }
            sign = next_sign;
            z.copy_from_slice(&sign);
            self.substitute_transpose(&mut z);
            let last = j;
            j = argmax(&z);
            if z[last].abs() == z[j].abs() {
                break;
            }
        }
        let mut alternative: Vec<f64> = (0..n)
            .map(|i| {
                let magnitude = 1.0 + i as f64 / (n - 1) as f64;
                if i % 2 == 0 { magnitude } else { -magnitude }
            })
            .collect();
        self.substitute(&mut alternative);
        estimate.max(2.0 * l1(&alternative) / (3.0 * n as f64))
    }

    fn substitute_transpose(&self, rhs: &mut [f64]) {
        for row in 0..self.n {
            let mut value = rhs[row];
            for (col, &known_value) in rhs.iter().take(row).enumerate() {
                value -= self.factors[col * self.n + row] * known_value;
            }
            rhs[row] = value / self.factors[row * self.n + row];
        }
        for row in (0..self.n).rev() {
            let mut value = rhs[row];
            for (col, &known_value) in rhs.iter().enumerate().skip(row + 1) {
                value -= self.factors[col * self.n + row] * known_value;
            }
            rhs[row] = value;
        }
        for (row, &swap_row) in self.swaps.iter().enumerate().rev() {
            if row != swap_row {
                rhs.swap(row, swap_row);
            }
        }
    }

    fn substitute(&self, rhs: &mut [f64]) {
        for (row, &swap_row) in self.swaps.iter().enumerate() {
            if row != swap_row {
                rhs.swap(row, swap_row);
            }
        }
        for row in 0..self.n {
            let row_start = row * self.n;
            let mut value = rhs[row];
            for (col, &known_value) in rhs.iter().take(row).enumerate() {
                value -= self.factors[row_start + col] * known_value;
            }
            rhs[row] = value;
        }
        for row in (0..self.n).rev() {
            let row_start = row * self.n;
            let mut value = rhs[row];
            for (col, &known_value) in rhs.iter().enumerate().skip(row + 1) {
                value -= self.factors[row_start + col] * known_value;
            }
            rhs[row] = value / self.factors[row_start + row];
        }
    }

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
        self.substitute(&mut solution);
        if solution.iter().any(|value| !value.is_finite()) {
            return Err(SurvivalError::computation(format!(
                "{}: solution overflowed to a non-finite value",
                Self::CONTEXT
            )));
        }
        Ok(solution)
    }

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

pub(crate) fn lu_inverse(matrix: &Array2<f64>) -> SurvivalResult<Array2<f64>> {
    LuDecomposition::decompose(matrix)?.inverse()
}

pub(crate) fn matrix_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    lu_inverse(matrix).ok()
}

pub(crate) fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|row| row.len() != n) {
        return None;
    }
    let flat: Vec<f64> = mat.iter().flatten().copied().collect();
    let matrix = Array2::from_shape_vec((n, n), flat).ok()?;
    Some(matrix_rows(&matrix_inverse(&matrix)?))
}

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
        Err(_) => vec![0.0; n * n],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn exact_inverse_norm(lu: &LuDecomposition) -> f64 {
        let inverse = lu.inverse().unwrap();
        (0..lu.n)
            .map(|col| (0..lu.n).map(|row| inverse[[row, col]].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    fn test_matrix(n: usize, seed: u64) -> Array2<f64> {
        let mut state = seed;
        Array2::from_shape_fn((n, n), |(row, col)| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (state >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
            if row == col {
                n as f64 * 0.3 + noise
            } else {
                noise
            }
        })
    }

    #[test]
    fn transpose_substitution_solves_the_transposed_system() {
        let matrix = test_matrix(7, 3);
        let lu = LuDecomposition::decompose(&matrix).unwrap();
        let rhs: Vec<f64> = (0..7).map(|i| i as f64 - 2.5).collect();
        let mut solution = rhs.clone();
        lu.substitute_transpose(&mut solution);
        for col in 0..7 {
            let value: f64 = (0..7).map(|row| matrix[[row, col]] * solution[row]).sum();
            assert!((value - rhs[col]).abs() < 1e-10);
        }
    }

    #[test]
    fn inverse_norm_estimate_bounds_the_exact_norm() {
        for (n, seed) in [(1, 1), (2, 5), (5, 9), (12, 17), (40, 23)] {
            let lu = LuDecomposition::decompose(&test_matrix(n, seed)).unwrap();
            let exact = exact_inverse_norm(&lu);
            let estimate = lu.inverse_norm_estimate();
            assert!(estimate <= exact * (1.0 + 1e-12), "{estimate} > {exact}");
            assert!(estimate >= exact / 3.0, "{estimate} < {exact} / 3");
        }
        let lu = LuDecomposition::decompose(&arr2(&[[4.0, 1.0], [1.0, 3.0]])).unwrap();
        assert!((lu.inverse_norm_estimate() - exact_inverse_norm(&lu)).abs() < 1e-14);
    }

    #[test]
    fn known_matrix_columns_survive_empty_rows_and_columns() {
        assert_eq!(
            matrix_from_rows_with_columns(&[], 3, "x").unwrap().dim(),
            (0, 3)
        );
        assert_eq!(
            matrix_from_rows_with_columns(&[vec![], vec![]], 0, "x")
                .unwrap()
                .dim(),
            (2, 0)
        );
    }

    #[test]
    fn known_matrix_columns_validate_every_row() {
        assert!(matrix_from_rows_with_columns(&[vec![1.0, 2.0]], 1, "x").is_err());
        assert!(matrix_from_rows_with_columns(&[vec![1.0], vec![]], 1, "x").is_err());
        assert_eq!(
            matrix_from_rows_with_columns(&[vec![1.0, 2.0], vec![3.0, 4.0]], 2, "x").unwrap(),
            arr2(&[[1.0, 2.0], [3.0, 4.0]])
        );
    }

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

    fn rank_deficient() -> Array2<f64> {
        arr2(&[[4.0, 2.0, 6.0], [2.0, 5.0, 7.0], [6.0, 7.0, 13.0]])
    }

    #[test]
    fn cholesky2_factors_spd_matrix_as_fdft() {
        let mut work = spd();
        assert_eq!(cholesky2(&mut work, 1e-9), 3);

        for i in 0..3 {
            assert_close(work[[i, i]], 4.0, 1e-12);
        }
        assert_close(work[[1, 0]], 0.5, 1e-12);
        assert_close(work[[2, 0]], 0.5, 1e-12);
        assert_close(work[[2, 1]], 0.5, 1e-12);
        assert_eq!(work[[0, 1]], 2.0);
        assert_eq!(work[[0, 2]], 2.0);
        assert_eq!(work[[1, 2]], 3.0);

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
            Err(SurvivalError::Singular { columns, .. }) => assert_eq!(columns, vec![0]),
            other => panic!("expected singular error, got {other:?}"),
        }
        assert!(matches!(
            lu_inverse(&singular),
            Err(SurvivalError::Singular { .. })
        ));
        assert!(matrix_inverse(&singular).is_none());
    }

    #[test]
    fn lu_decomposition_uses_r_condition_number_rule() {
        let inverse = lu_inverse(&arr2(&[[1.0, 0.0], [0.0, 1e-13]])).unwrap();
        assert_matrix_close(&inverse, &arr2(&[[1.0, 0.0], [0.0, 1e13]]), 1.0);
        let inverse = lu_inverse(&arr2(&[[1.0, 1.0], [1.0, 1.0 + 1e-15]])).unwrap();
        let expected = arr2(&[
            [900719925474100.2, -900719925474099.2],
            [-900719925474099.2, 900719925474099.2],
        ]);
        assert_matrix_close(&inverse, &expected, 0.5);
        for (singular, rcond) in [
            (arr2(&[[1.0, 1.0], [1.0, 1.0 + 4e-16]]), "1.11022e-16"),
            (arr2(&[[1.0, 0.0], [0.0, 1e-17]]), "1.00000e-17"),
        ] {
            match LuDecomposition::decompose(&singular) {
                Err(SurvivalError::Singular { context, columns }) => {
                    assert!(columns.is_empty());
                    assert_eq!(
                        context,
                        format!("LU factorisation (reciprocal condition number = {rcond})")
                    );
                }
                other => panic!("expected singular error, got {other:?}"),
            }
        }
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
    fn row_conversions_round_trip_and_reject_ragged_rows() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let matrix = matrix_from_rows(&rows, "x").unwrap();
        assert_eq!(matrix, arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
        assert_eq!(matrix_rows(&matrix), rows);
        assert_eq!(matrix_from_rows(&[], "x").unwrap().dim(), (0, 0));
        let err = matrix_from_rows(&[vec![1.0, 2.0], vec![3.0]], "x").unwrap_err();
        assert!(err.to_string().contains("x must be rectangular"));
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
