#![allow(dead_code)]

fn cholesky5(matrix: &mut [&mut [f64]], n: usize, toler: f64) -> i32 {
    let mut eps = 0.0;

    for (i, row) in matrix.iter().enumerate() {
        let diag = row[i].abs();
        if diag > eps {
            eps = diag;
        }
    }

    if eps == 0.0 {
        eps = toler;
    } else {
        eps *= toler;
    }

    let mut rank = 0;

    for i in 0..n {
        let pivot = matrix[i][i];

        if !pivot.is_finite() || pivot.abs() < eps {
            for row in matrix.iter_mut().skip(i) {
                row[i] = 0.0;
            }
        } else {
            rank += 1;
            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..n {
                let temp = matrix[j][i] / pivot;
                matrix[j][i] = temp;
                matrix[j][j] -= temp * temp * pivot;

                for k_row in matrix.iter_mut().skip(j + 1) {
                    k_row[j] -= temp * k_row[i];
                }
            }
        }
    }

    rank
}

fn chinv5(matrix: &mut [&mut [f64]], n: usize, flag: i32) {
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if matrix[i][i] != 0.0 {
            matrix[i][i] = 1.0 / matrix[i][i];

            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..n {
                matrix[j][i] = -matrix[j][i];
                #[allow(clippy::needless_range_loop)]
                for k_idx in 0..i {
                    matrix[j][k_idx] += matrix[j][i] * matrix[i][k_idx];
                }
            }
        } else {
            for row in matrix.iter_mut().skip(i + 1) {
                row[i] = 0.0;
            }
        }
    }

    if flag == 1 {
        return;
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if matrix[i][i] == 0.0 {
            for row in matrix.iter_mut().take(i) {
                row[i] = 0.0;
            }
            for elem in matrix[i].iter_mut().skip(i) {
                *elem = 0.0;
            }
        } else {
            #[allow(clippy::needless_range_loop)]
            for j in (i + 1)..n {
                let temp = matrix[j][i] * matrix[j][j];
                if j != i {
                    matrix[i][j] = temp;
                }
                #[allow(clippy::needless_range_loop)]
                for k in i..j {
                    matrix[i][k] += temp * matrix[j][k];
                }
            }
        }
    }
}
