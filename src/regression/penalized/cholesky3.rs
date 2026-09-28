//! R survival's sparse-plus-dense Cholesky routines, `src/cholesky3.c`,
//! `src/chsolve3.c` and `src/chinv3.c`, used by the penalised Newton
//! kernels (`coxfit5.c`, `agfit5.c`, `survreg7.c`).
//!
//! The matrix has the block structure `[D  B'; B  C]`: `D` (the sparse
//! frailty groups, `m` of them) is diagonal and stored in `diag`, and only
//! the dense slice `[B C]` is kept (`n2` rows, `m + n2` columns, the C code's
//! `matrix[i][j]` with row `i` a dense coefficient).  The factorisation
//! `F D F'` keeps that shape.

use ndarray::Array2;

/// `cholesky3`: the generalised Cholesky `C = F D F'` of the sparse-plus-dense
/// matrix (`diag`, `matrix`).  `D` overwrites the diagonals, `F` the lower
/// triangle of the dense slice; a pivot below the tolerance (`toler` times
/// the smallest diagonal, or `toler` itself when none is negative — as in
/// the C code, which scales by the minimum) zeroes its column.  Returns the
/// rank, negated when a pivot was more negative than `-8 * eps`.
pub(crate) fn cholesky3(matrix: &mut Array2<f64>, m: usize, diag: &mut [f64], toler: f64) -> i32 {
    let n2 = matrix.nrows();
    let mut nonneg = 1;
    let mut eps = 0.0f64;
    for &d in diag.iter().take(m) {
        if d < eps {
            eps = d;
        }
    }
    for i in 0..n2 {
        if matrix[(i, i + m)] < eps {
            eps = matrix[(i, i + m)];
        }
    }
    eps = if eps == 0.0 { toler } else { eps * toler };

    let mut rank = 0;
    // Pivot out the diagonal elements.
    for i in 0..m {
        let pivot = diag[i];
        if !pivot.is_finite() || pivot < eps {
            for j in 0..n2 {
                matrix[(j, i)] = 0.0;
            }
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in 0..n2 {
                let temp = matrix[(j, i)] / pivot;
                matrix[(j, i)] = temp;
                matrix[(j, j + m)] -= temp * temp * pivot;
                for k in j + 1..n2 {
                    matrix[(k, j + m)] -= temp * matrix[(k, i)];
                }
            }
        }
    }
    // Now the dense part.
    for i in 0..n2 {
        let pivot = matrix[(i, i + m)];
        if !pivot.is_finite() || pivot < eps {
            for j in i..n2 {
                matrix[(j, i + m)] = 0.0;
            }
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in i + 1..n2 {
                let temp = matrix[(j, i + m)] / pivot;
                matrix[(j, i + m)] = temp;
                matrix[(j, j + m)] -= temp * temp * pivot;
                for k in j + 1..n2 {
                    matrix[(k, j + m)] -= temp * matrix[(k, i + m)];
                }
            }
        }
    }
    rank * nonneg
}

/// `chsolve3`: solves `A b = y` from the [`cholesky3`] factors, overwriting
/// `y`; components of redundant columns are set to zero.
pub(crate) fn chsolve3(matrix: &Array2<f64>, m: usize, diag: &[f64], y: &mut [f64]) {
    let n2 = matrix.nrows();
    // Solve F b = y (the diagonal portion is unchanged).
    for i in 0..n2 {
        let mut temp = y[i + m];
        for j in 0..m {
            temp -= y[j] * matrix[(i, j)];
        }
        for j in 0..i {
            temp -= y[j + m] * matrix[(i, j + m)];
        }
        y[i + m] = temp;
    }
    // Solve D F' z = b: the dense portion, then the diagonal one.
    for i in (0..n2).rev() {
        if matrix[(i, i + m)] == 0.0 {
            y[i + m] = 0.0;
        } else {
            let mut temp = y[i + m] / matrix[(i, i + m)];
            for j in i + 1..n2 {
                temp -= y[j + m] * matrix[(j, i + m)];
            }
            y[i + m] = temp;
        }
    }
    for i in (0..m).rev() {
        if diag[i] == 0.0 {
            y[i] = 0.0;
        } else {
            let mut temp = y[i] / diag[i];
            for j in 0..n2 {
                temp -= y[j + m] * matrix[(j, i)];
            }
            y[i] = temp;
        }
    }
}

/// `chinv3`: inverts the Cholesky factor in place — `D^{-1}` on the
/// diagonals (only positive entries are inverted), `F^{-1}` in the lower
/// triangle.
pub(crate) fn chinv3(matrix: &mut Array2<f64>, m: usize, fdiag: &mut [f64]) {
    let n2 = matrix.nrows();
    for i in 0..m {
        if fdiag[i] > 0.0 {
            fdiag[i] = 1.0 / fdiag[i];
            for j in 0..n2 {
                matrix[(j, i)] = -matrix[(j, i)];
            }
        }
    }
    for i in 0..n2 {
        let ii = i + m;
        if matrix[(i, ii)] > 0.0 {
            matrix[(i, ii)] = 1.0 / matrix[(i, ii)];
            for j in i + 1..n2 {
                matrix[(j, ii)] = -matrix[(j, ii)];
                for k in 0..ii {
                    let update = matrix[(j, ii)] * matrix[(i, k)];
                    matrix[(j, k)] += update;
                }
            }
        }
    }
}

/// The "nicer output for the S user" block that follows `chinv3` in
/// `coxfit5.c`, `agfit5.c` and `survreg7.c` (survreg7.c:466-474): the dense
/// diagonal of `D^{-1}` moves from the inverse factor `hinv` into `fdiag`,
/// and both slices get a unit diagonal and zeros above it.
pub(crate) fn finish_factors(
    hmat: &mut Array2<f64>,
    hinv: &mut Array2<f64>,
    nf: usize,
    fdiag: &mut [f64],
) {
    let nvar3 = nf + hinv.nrows();
    for i in nf..nvar3 {
        let ii = i - nf;
        fdiag[i] = hinv[(ii, i)];
        hinv[(ii, i)] = 1.0;
        hmat[(ii, i)] = 1.0;
        for j in i + 1..nvar3 {
            hinv[(ii, j)] = 0.0;
            hmat[(ii, j)] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dense matrix `[D B'; B C]` of a factorisation.
    fn full_matrix(matrix: &Array2<f64>, m: usize, diag: &[f64]) -> Array2<f64> {
        let n2 = matrix.nrows();
        let n = n2 + m;
        let mut full = Array2::zeros((n, n));
        for i in 0..m {
            full[(i, i)] = diag[i];
        }
        for i in 0..n2 {
            for j in 0..=i + m {
                full[(i + m, j)] = matrix[(i, j)];
                full[(j, i + m)] = matrix[(i, j)];
            }
        }
        full
    }

    #[test]
    fn cholesky3_solves_and_inverts_the_sparse_plus_dense_system() {
        let mut diag = vec![4.0, 5.0];
        let mut matrix =
            Array2::from_shape_vec((2, 4), vec![1.0, 0.5, 6.0, 0.0, 0.2, 1.0, 1.5, 7.0]).unwrap();
        let full = full_matrix(&matrix, 2, &diag);
        let rank = cholesky3(&mut matrix, 2, &mut diag, 1e-12);
        assert_eq!(rank, 4);
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        let rhs = y.clone();
        chsolve3(&matrix, 2, &diag, &mut y);
        for i in 0..4 {
            let value: f64 = (0..4).map(|j| full[(i, j)] * y[j]).sum();
            assert!((value - rhs[i]).abs() < 1e-12, "row {i}");
        }
        // The inverse factor: F^{-1}' D^{-1} F^{-1} is the inverse.
        chinv3(&mut matrix, 2, &mut diag);
        let mut finv = Array2::<f64>::eye(4);
        for i in 0..2 {
            for j in 0..=i + 2 {
                if j != i + 2 {
                    finv[(i + 2, j)] = matrix[(i, j)];
                }
            }
        }
        let mut dinv = Array2::<f64>::zeros((4, 4));
        for i in 0..2 {
            dinv[(i, i)] = diag[i];
            dinv[(i + 2, i + 2)] =
                1.0 / matrix[(i, i + 2)] * matrix[(i, i + 2)] * matrix[(i, i + 2)];
        }
        // D^{-1} of the dense rows still sits on the matrix diagonal.
        for i in 0..2 {
            dinv[(i + 2, i + 2)] = matrix[(i, i + 2)];
        }
        let inverse = finv.t().dot(&dinv).dot(&finv);
        let identity = inverse.dot(&full);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity[(i, j)] - expected).abs() < 1e-10, "({i}, {j})");
            }
        }
    }

    #[test]
    fn cholesky3_zeroes_redundant_columns() {
        let mut diag = vec![0.0];
        let mut matrix =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.0, 1.0, 2.0, 2.0]).unwrap();
        let rank = cholesky3(&mut matrix, 1, &mut diag, 1e-12);
        assert_eq!(rank, 1);
        assert_eq!(matrix.column(0).to_vec(), vec![0.0, 0.0]);
        assert_eq!(matrix[(1, 2)], 0.0);
        let mut y = vec![1.0, 1.0, 1.0];
        chsolve3(&matrix, 1, &diag, &mut y);
        assert_eq!(y[0], 0.0);
        assert_eq!(y[2], 0.0);
    }
}
