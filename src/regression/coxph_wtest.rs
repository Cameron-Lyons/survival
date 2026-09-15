//! Cholesky-based Wald test, a port of R survival's `coxph.wtest()`
//! (`R/coxph.wtest.R`, `src/coxph_wtest.c`).
//!
//! `coxph` uses it for the overall Wald statistic `b' V^{-1} b` because a
//! plain `solve()` would choke on a singular variance matrix: the
//! generalised Cholesky (`cholesky2`/`chsolve2`, `internal::matrix`) zeroes
//! redundant directions and reports the rank as the degrees of freedom.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::{cholesky2, chsolve2};
use ndarray::Array2;
use pyo3::prelude::*;

/// `coxph.wtest(var, b)`: one test per column of `b`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxphWtest {
    /// `b' V^{-1} b` for each column of `b`.
    #[pyo3(get)]
    pub test: Vec<f64>,
    /// Rank of `var`, the degrees of freedom of each test.
    #[pyo3(get)]
    pub df: usize,
    /// `V^{-1} b` (one row per variable, one column per test).
    #[pyo3(get)]
    pub solve: Vec<Vec<f64>>,
}

/// Port of `coxph_wtest.c`.  `b` is `nvar x ntest`.
pub fn wald_tests(
    var: &Array2<f64>,
    b: &Array2<f64>,
    toler_chol: f64,
) -> SurvivalResult<CoxphWtest> {
    let nvar = b.nrows();
    let ntest = b.ncols();
    if var.nrows() != var.ncols() {
        return Err(SurvivalError::invalid_input(
            "First argument must be a square matrix",
        ));
    }
    if var.nrows() != nvar {
        return Err(SurvivalError::invalid_input(
            "Argument lengths do not match",
        ));
    }
    if b.iter().chain(var.iter()).any(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "infinite argument in coxph.wtest",
        ));
    }
    if !(toler_chol.is_finite() && toler_chol >= 0.0) {
        return Err(SurvivalError::invalid_input(
            "toler.chol must be non-negative",
        ));
    }
    // dmatrix() hands cholesky2 the columns of R's matrix as rows, so the
    // C routine reads R's lower triangle; the transpose reproduces that.
    let mut factor = var.t().to_owned();
    cholesky2(&mut factor, toler_chol);
    let df = (0..nvar).filter(|&i| factor[(i, i)] > 0.0).count();
    let mut test = Vec::with_capacity(ntest);
    let mut solve = vec![vec![0.0; ntest]; nvar];
    let mut column = vec![0.0; nvar];
    for j in 0..ntest {
        for i in 0..nvar {
            column[i] = b[(i, j)];
        }
        chsolve2(&factor, &mut column);
        test.push((0..nvar).map(|i| b[(i, j)] * column[i]).sum());
        for i in 0..nvar {
            solve[i][j] = column[i];
        }
    }
    Ok(CoxphWtest { test, df, solve })
}

/// The scalar case `coxph.wtest(var, b)$test` for a coefficient vector.
pub(crate) fn wald_statistic(var: &Array2<f64>, b: &[f64], toler_chol: f64) -> SurvivalResult<f64> {
    if b.is_empty() {
        return Ok(0.0);
    }
    let column = Array2::from_shape_vec((b.len(), 1), b.to_vec())
        .map_err(|err| SurvivalError::invalid_input(err.to_string()))?;
    Ok(wald_tests(var, &column, toler_chol)?.test[0])
}

/// `coxph.wtest(var, b, toler.chol)`; `b` is given as one row per test.
#[pyfunction(name = "coxph_wtest")]
#[pyo3(signature = (var, b, toler_chol=1e-9))]
pub fn coxph_wtest_py(
    var: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    toler_chol: f64,
) -> PyResult<CoxphWtest> {
    let nvar = var.len();
    if var.iter().any(|row| row.len() != nvar) {
        return Err(SurvivalError::invalid_input("First argument must be a square matrix").into());
    }
    if b.iter().any(|row| row.len() != nvar) {
        return Err(SurvivalError::invalid_input("Argument lengths do not match").into());
    }
    let var = Array2::from_shape_vec((nvar, nvar), var.into_iter().flatten().collect())
        .map_err(|err| SurvivalError::invalid_input(err.to_string()))?;
    let ntest = b.len();
    let mut columns = Array2::zeros((nvar, ntest));
    for (j, row) in b.iter().enumerate() {
        for (i, &value) in row.iter().enumerate() {
            columns[(i, j)] = value;
        }
    }
    Ok(wald_tests(&var, &columns, toler_chol)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
    }

    #[test]
    fn matches_reference_full_rank_and_multiple_rhs() {
        let result = coxph_wtest_py(
            vec![vec![2.0, 0.5], vec![0.5, 1.0]],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            1e-9,
        )
        .expect("factorization should succeed");

        assert_eq!(result.df, 2);
        assert_close(&result.test, &[4.0, 16.571428571428573]);
        assert_close(&result.solve[0], &[0.0, 0.5714285714285714]);
        assert_close(&result.solve[1], &[2.0, 3.7142857142857144]);
    }

    #[test]
    fn matches_reference_singular_and_indefinite_semantics() {
        let cases = [
            (
                vec![
                    vec![1.0, 2.0, 3.0],
                    vec![2.0, 4.0, 6.0],
                    vec![3.0, 6.0, 9.0],
                ],
                vec![1.0, 2.0, 3.0],
                1,
                vec![1.0, 0.0, 0.0],
            ),
            (
                vec![
                    vec![0.0, 0.0, 0.0],
                    vec![0.0, 2.0, 0.0],
                    vec![0.0, 0.0, 3.0],
                ],
                vec![1.0, 2.0, 3.0],
                2,
                vec![0.0, 1.0, 1.0],
            ),
            (
                vec![vec![1.0, 2.0], vec![2.0, 1.0]],
                vec![1.0, 2.0],
                1,
                vec![1.0, 0.0],
            ),
        ];

        for (matrix, rhs, expected_df, expected_solve) in cases {
            let result =
                coxph_wtest_py(matrix, vec![rhs], 1e-9).expect("factorization should succeed");
            assert_eq!(result.df, expected_df);
            assert_close(
                &result.solve.iter().map(|row| row[0]).collect::<Vec<_>>(),
                &expected_solve,
            );
        }
    }

    #[test]
    fn matches_r_column_major_lower_triangle_semantics() {
        let result = coxph_wtest_py(
            vec![vec![2.0, 0.25], vec![7.0, 1.0]],
            vec![vec![1.0, 2.0]],
            1e-9,
        )
        .expect("factorization should succeed");

        assert_eq!(result.df, 1);
        assert_close(&result.test, &[0.5]);
        assert_close(
            &result.solve.iter().map(|row| row[0]).collect::<Vec<_>>(),
            &[0.5, 0.0],
        );
    }

    #[test]
    fn wald_statistic_matches_r_reference() {
        // A <- matrix(c(4,2,2, 2,5,3, 2,3,6), 3, 3); coxph.wtest(A, 1:3)$test = 1.578125
        let var = arr2(&[[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]]);
        let test = wald_statistic(&var, &[1.0, 2.0, 3.0], 1e-9).unwrap();
        assert!((test - 1.578125).abs() < 1e-12);
    }

    #[test]
    fn rejects_malformed_inputs() {
        assert!(coxph_wtest_py(vec![vec![1.0, 2.0]], vec![vec![1.0]], 1e-9).is_err());
        assert!(coxph_wtest_py(vec![vec![1.0]], vec![vec![1.0, 2.0]], 1e-9).is_err());
        assert!(coxph_wtest_py(vec![vec![f64::INFINITY]], vec![vec![1.0]], 1e-9).is_err());
        assert!(coxph_wtest_py(vec![vec![1.0]], vec![vec![1.0]], -1.0).is_err());
    }
}
