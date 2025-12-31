use faer::{Side, prelude::*};
use ndarray::{Array1, Array2};

/// Convert ndarray Array2 to faer Mat
fn ndarray_to_faer(arr: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = arr.dim();
    Mat::from_fn(rows, cols, |i, j| arr[[i, j]])
}

/// Convert faer Col to ndarray Array1
fn faer_col_to_ndarray(col: faer::ColRef<f64>) -> Array1<f64> {
    Array1::from_iter((0..col.nrows()).map(|i| col.read(i)))
}

pub fn cholesky_solve(
    matrix: &Array2<f64>,
    vector: &Array1<f64>,
    _tol: f64,
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Ok(Array1::zeros(vector.len()));
    }

    let max_val = matrix.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_val < 1e-10 {
        return Ok(Array1::zeros(vector.len()));
    }

    let mat = ndarray_to_faer(matrix);
    let b: Col<f64> = Col::from_fn(vector.len(), |i| vector[i]);

    // Try Cholesky decomposition
    match mat.cholesky(Side::Lower) {
        Ok(chol) => {
            let x = chol.solve(&b);
            Ok(faer_col_to_ndarray(x.as_ref()))
        }
        Err(_) => {
            // Add ridge regularization and retry
            let n = matrix.nrows();
            let ridge = max_val * 1e-6;
            let reg_mat = Mat::from_fn(n, n, |i, j| {
                if i == j {
                    mat.read(i, j) + ridge
                } else {
                    mat.read(i, j)
                }
            });

            match reg_mat.cholesky(Side::Lower) {
                Ok(chol) => {
                    let x = chol.solve(&b);
                    Ok(faer_col_to_ndarray(x.as_ref()))
                }
                Err(_) => Ok(Array1::zeros(vector.len())),
            }
        }
    }
}

/// Solve a linear system Ax = b using LU decomposition
pub fn lu_solve(matrix: &Array2<f64>, vector: &Array1<f64>) -> Option<Array1<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Some(Array1::zeros(vector.len()));
    }

    let mat = ndarray_to_faer(matrix);
    let b: Col<f64> = Col::from_fn(vector.len(), |i| vector[i]);

    let lu = mat.partial_piv_lu();
    let x = lu.solve(&b);
    Some(faer_col_to_ndarray(x.as_ref()))
}

/// Compute the inverse of a matrix using LU decomposition
pub fn matrix_inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Some(matrix.clone());
    }

    let mat = ndarray_to_faer(matrix);
    let lu = mat.partial_piv_lu();
    let inv = lu.inverse();

    let (rows, cols) = (inv.nrows(), inv.ncols());
    let mut result = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = inv.read(i, j);
        }
    }
    Some(result)
}

/// Check if Cholesky decomposition is possible (matrix is positive definite)
pub fn cholesky_check(matrix: &Array2<f64>) -> bool {
    let mat = ndarray_to_faer(matrix);
    mat.cholesky(Side::Lower).is_ok()
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
    fn test_lu_solve() {
        let matrix = arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        let vector = Array1::from_vec(vec![3.0, 4.0]);
        let result = lu_solve(&matrix, &vector).unwrap();
        // Verify Ax = b
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
