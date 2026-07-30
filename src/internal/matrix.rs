use crate::constants::{GAUSSIAN_ELIMINATION_TOL, NEAR_ZERO_MATRIX, RIDGE_REGULARIZATION};
use crate::internal::validation::MatrixError;
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

struct PartialPivotLu {
    factors: Vec<f64>,
    swaps: Vec<usize>,
    n: usize,
}

impl PartialPivotLu {
    fn decompose(matrix: &Array2<f64>) -> Option<Self> {
        let (rows, cols) = matrix.dim();
        if rows != cols {
            return None;
        }
        if rows == 0 {
            return Some(Self {
                factors: Vec::new(),
                swaps: Vec::new(),
                n: 0,
            });
        }

        let n = rows;
        let mut factors = Vec::with_capacity(n * n);
        let mut scale = 0.0_f64;
        for row in 0..n {
            for col in 0..n {
                let value = matrix[[row, col]];
                if !value.is_finite() {
                    return None;
                }
                scale = scale.max(value.abs());
                factors.push(value);
            }
        }
        if scale == 0.0 {
            return None;
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
                return None;
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

        Some(Self { factors, swaps, n })
    }

    fn solve_slice(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        if rhs.len() != self.n || rhs.iter().any(|value| !value.is_finite()) {
            return None;
        }
        if self.n == 0 {
            return Some(Vec::new());
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
            let diagonal = self.factors[row_start + row];
            if diagonal == 0.0 || !diagonal.is_finite() {
                return None;
            }
            solution[row] = value / diagonal;
        }

        solution
            .iter()
            .all(|value| value.is_finite())
            .then_some(solution)
    }

    fn inverse(&self) -> Option<Array2<f64>> {
        let mut inverse = vec![0.0; self.n * self.n];
        let mut rhs = vec![0.0; self.n];

        for col in 0..self.n {
            rhs[col] = 1.0;
            let solution = self.solve_slice(&rhs)?;
            rhs[col] = 0.0;
            for row in 0..self.n {
                inverse[row * self.n + col] = solution[row];
            }
        }

        Array2::from_shape_vec((self.n, self.n), inverse).ok()
    }
}

pub(crate) fn regularized_lu_solve(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
) -> Result<Array1<f64>, MatrixError> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        if vector.is_empty() {
            return Ok(Array1::zeros(0));
        }
        return Err(MatrixError::EmptyMatrix);
    }

    let max_val = matrix.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_val < NEAR_ZERO_MATRIX {
        return Err(MatrixError::SingularMatrix);
    }

    match lu_solve_internal(matrix, vector) {
        Some(result) => Ok(result),
        None => {
            let n = matrix.nrows();
            let ridge = max_val * RIDGE_REGULARIZATION;
            let mut reg_matrix = matrix.clone();
            for i in 0..n {
                reg_matrix[[i, i]] += ridge;
            }
            match lu_solve_internal(&reg_matrix, vector) {
                Some(result) => Ok(result),
                None => Err(MatrixError::SingularMatrix),
            }
        }
    }
}

fn lu_solve_internal(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return vector.is_empty().then(|| Array1::zeros(0));
    }

    let factorization = PartialPivotLu::decompose(matrix)?;
    factorization
        .solve_slice(vector.as_slice()?)
        .map(Array1::from_vec)
}

pub(crate) fn lu_solve(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    lu_solve_internal(matrix, vector)
}

pub(crate) fn matrix_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Some(matrix.clone());
    }

    PartialPivotLu::decompose(matrix)?.inverse()
}

pub(crate) fn invert_flat_square_matrix_with_fallback(a: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if a.len() != n * n {
        return vec![0.0; n * n];
    }
    if n == 1 {
        return vec![if a[0].abs() > GAUSSIAN_ELIMINATION_TOL {
            1.0 / a[0]
        } else {
            0.0
        }];
    }

    if let Ok(arr) = Array2::from_shape_vec((n, n), a.to_vec())
        && let Some(inv) = matrix_inverse(&arr)
    {
        return inv.iter().copied().collect();
    }

    let mut aug = vec![0.0; n * 2 * n];
    let width = 2 * n;

    for i in 0..n {
        let row_offset = i * width;
        for j in 0..n {
            aug[row_offset + j] = a[i * n + j];
        }
        aug[row_offset + n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k * width + i].abs() > aug[max_row * width + i].abs() {
                max_row = k;
            }
        }

        if max_row != i {
            for j in 0..width {
                aug.swap(i * width + j, max_row * width + j);
            }
        }

        let pivot = aug[i * width + i];
        if pivot.abs() < GAUSSIAN_ELIMINATION_TOL {
            continue;
        }

        for j in 0..width {
            aug[i * width + j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k * width + i];
                for j in 0..width {
                    let pivot_val = aug[i * width + j];
                    aug[k * width + j] -= factor * pivot_val;
                }
            }
        }
    }

    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * width + n + j];
        }
    }

    inv
}

pub(crate) fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 {
        return None;
    }
    for row in mat {
        if row.len() != n {
            return None;
        }
    }

    let mut aug: Vec<Vec<f64>> = mat
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut new_row = row.clone();
            new_row.extend(vec![0.0; n]);
            new_row[n + i] = 1.0;
            new_row
        })
        .collect();

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < GAUSSIAN_ELIMINATION_TOL {
            return None;
        }

        let pivot = aug[i][i];
        for val in aug[i].iter_mut().take(2 * n) {
            *val /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                let (pivot_row, target_row) = if k < i {
                    let (left, right) = aug.split_at_mut(i);
                    (&right[0], &mut left[k])
                } else {
                    let (left, right) = aug.split_at_mut(k);
                    (&left[i], &mut right[0])
                };

                for j in 0..(2 * n) {
                    target_row[j] -= factor * pivot_row[j];
                }
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_regularized_lu_solve_identity() {
        let matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = regularized_lu_solve(&matrix, &vector).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
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

    #[test]
    fn test_regularized_lu_solve_empty() {
        let matrix: Array2<f64> = Array2::zeros((0, 0));
        let vector: Array1<f64> = Array1::zeros(0);
        let result = regularized_lu_solve(&matrix, &vector).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_regularized_lu_solve_near_zero_matrix() {
        let matrix = arr2(&[[1e-15, 0.0], [0.0, 1e-15]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = regularized_lu_solve(&matrix, &vector);
        assert!(matches!(result, Err(MatrixError::SingularMatrix)));
    }

    #[test]
    fn test_lu_solve() {
        let matrix = arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        let vector = Array1::from_vec(vec![3.0, 4.0]);
        let result = lu_solve(&matrix, &vector).unwrap();
        let ax0 = 2.0 * result[0] + 1.0 * result[1];
        let ax1 = 1.0 * result[0] + 3.0 * result[1];
        assert!((ax0 - 3.0).abs() < 1e-10);
        assert!((ax1 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_solve_uses_partial_pivoting() {
        let matrix = arr2(&[[0.0, 2.0], [1.0, 3.0]]);
        let vector = Array1::from_vec(vec![4.0, 5.0]);
        let result = lu_solve(&matrix, &vector).unwrap();
        assert!((result[0] + 1.0).abs() < 1e-12);
        assert!((result[1] - 2.0).abs() < 1e-12);
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
    fn test_matrix_inverse() {
        let matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let inv = matrix_inverse(&matrix).unwrap();
        assert!((inv[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((inv[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(inv[[0, 1]].abs() < 1e-10);
        assert!(inv[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse_nontrivial_product_is_identity() {
        let matrix = arr2(&[[4.0, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]]);
        let inverse = matrix_inverse(&matrix).unwrap();
        let product = matrix.dot(&inverse);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((product[[row, col]] - expected).abs() < 1e-10);
            }
        }
    }
}
