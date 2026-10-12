use ndarray::Array2;

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

pub(crate) fn chsolve3(matrix: &Array2<f64>, m: usize, diag: &[f64], y: &mut [f64]) {
    let n2 = matrix.nrows();
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
