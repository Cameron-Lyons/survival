use ndarray::{Array1, Array2};
use ndarray_linalg::{Cholesky, Solve, UPLO};

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

    match matrix.cholesky(UPLO::Lower) {
        Ok(chol) => chol
            .solve(vector)
            .map_err(|e| format!("Cholesky solve failed: {}", e).into()),
        Err(_) => {
            let n = matrix.nrows();
            let mut reg_matrix = matrix.clone();
            let ridge = max_val * 1e-6;
            for i in 0..n {
                reg_matrix[[i, i]] += ridge;
            }
            match reg_matrix.cholesky(UPLO::Lower) {
                Ok(chol) => chol.solve(vector).map_err(|e| {
                    format!("Cholesky solve failed after regularization: {}", e).into()
                }),
                Err(_) => Ok(Array1::zeros(vector.len())),
            }
        }
    }
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
}
