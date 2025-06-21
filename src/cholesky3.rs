fn cholesky3(matrix: &mut [&mut [f64]], n: usize, m: usize, diag: &[f64], toler: f64) -> i32 {
    let n2 = n - m;
    let mut nonneg = 1;
    let mut eps = 0.0;

    for &val in diag.iter().take(m) {
        if val < eps {
            eps = val;
        }
    }
    for i in 0..n2 {
        let val = matrix[i][i + m];
        if val < eps {
            eps = val;
        }
    }

    if eps == 0.0 {
        eps = toler;
    } else {
        eps *= toler;
    }

    let mut rank = 0;

    for i in 0..m {
        let pivot = diag[i];
        if !pivot.is_finite() || pivot < eps {
            for j in 0..n2 {
                matrix[j][i] = 0.0;
            }
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in 0..n2 {
                let temp = matrix[j][i] / pivot;
                matrix[j][i] = temp;
                let update = temp * temp * pivot;
                matrix[j][j + m] -= update;
                for k in (j + 1)..n2 {
                    matrix[k][j + m] -= temp * matrix[k][i];
                }
            }
        }
    }

    for i in 0..n2 {
        let pivot = matrix[i][i + m];
        if !pivot.is_finite() || pivot < eps {
            for j in i..n2 {
                matrix[j][i + m] = 0.0;
            }
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in (i + 1)..n2 {
                let temp = matrix[j][i + m] / pivot;
                matrix[j][i + m] = temp;
                let update = temp * temp * pivot;
                matrix[j][j + m] -= update;
                for k in (j + 1)..n2 {
                    matrix[k][j + m] -= temp * matrix[k][i + m];
                }
            }
        }
    }

    rank * nonneg
}
