use crate::constants::{
    GAUSSIAN_ELIMINATION_TOL, NEAR_ZERO_MATRIX, PARALLEL_THRESHOLD_LARGE, RIDGE_REGULARIZATION,
};
use crate::internal::validation::MatrixError;
use faer::{linalg::solvers::DenseSolveCore, prelude::*};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

#[inline]
fn ndarray_to_faer(arr: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = arr.dim();
    if arr.is_standard_layout() && rows * cols > PARALLEL_THRESHOLD_LARGE {
        Mat::from_fn(rows, cols, |i, j| {
            // SAFETY: `Mat::from_fn` calls this closure with `i < rows` and
            // `j < cols`, where `(rows, cols)` came directly from `arr.dim()`.
            // The standard-layout check is only a fast-path precondition; the
            // bounds invariant is what makes unchecked ndarray access valid.
            unsafe { *arr.uget((i, j)) }
        })
    } else {
        Mat::from_fn(rows, cols, |i, j| arr[[i, j]])
    }
}

#[inline]
fn faer_col_to_ndarray(col: faer::ColRef<f64>) -> Array1<f64> {
    let n = col.nrows();
    let mut result = Array1::uninit(n);
    for i in 0..n {
        result[i].write(col[i]);
    }
    // SAFETY: `result` has length `n`, and the loop above writes exactly once
    // to every index in `0..n` before initialization is assumed.
    unsafe { result.assume_init() }
}

#[inline]
fn faer_mat_to_ndarray(mat: faer::MatRef<f64>) -> Array2<f64> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let mut data = vec![0.0; rows * cols];

    if rows * cols > PARALLEL_THRESHOLD_LARGE {
        data.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
            for j in 0..cols {
                row[j] = mat[(i, j)];
            }
        });
    } else {
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = mat[(i, j)];
            }
        }
    }

    Array2::from_shape_vec((rows, cols), data)
        .expect("shape and buffer length are consistent for Faer matrix conversion")
}

pub(crate) fn cholesky_solve(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
    _tol: f64,
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
        return Some(Array1::zeros(vector.len()));
    }

    let mat = ndarray_to_faer(matrix);
    let b: Col<f64> = Col::from_fn(vector.len(), |i| vector[i]);

    let lu = mat.partial_piv_lu();
    let x: Col<f64> = lu.solve(&b);
    Some(faer_col_to_ndarray(x.as_ref()))
}

pub(crate) fn lu_solve(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    lu_solve_internal(matrix, vector)
}

pub(crate) fn matrix_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Some(matrix.clone());
    }

    let mat = ndarray_to_faer(matrix);
    let lu = mat.partial_piv_lu();
    let inv: Mat<f64> = lu.inverse();

    Some(faer_mat_to_ndarray(inv.as_ref()))
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

pub(crate) fn cholesky_check(matrix: &Array2<f64>) -> bool {
    let n = matrix.nrows();
    if n == 0 {
        return true;
    }

    for i in 0..n {
        if matrix[[i, i]] <= 0.0 {
            return false;
        }
        for j in 0..i {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > crate::constants::SYMMETRY_TOL {
                return false;
            }
        }
    }

    let b = Array1::from_elem(n, 1.0);
    lu_solve(matrix, &b).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_cholesky_solve_identity() {
        let matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = cholesky_solve(&matrix, &vector, 1e-9).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_solve_empty() {
        let matrix: Array2<f64> = Array2::zeros((0, 0));
        let vector: Array1<f64> = Array1::zeros(0);
        let result = cholesky_solve(&matrix, &vector, 1e-9).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cholesky_solve_near_zero_matrix() {
        let matrix = arr2(&[[1e-15, 0.0], [0.0, 1e-15]]);
        let vector = Array1::from_vec(vec![1.0, 2.0]);
        let result = cholesky_solve(&matrix, &vector, 1e-9);
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
    fn test_matrix_inverse() {
        let matrix = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let inv = matrix_inverse(&matrix).unwrap();
        assert!((inv[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((inv[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(inv[[0, 1]].abs() < 1e-10);
        assert!(inv[[1, 0]].abs() < 1e-10);
    }
}
