#![allow(dead_code)]
fn cholesky5(matrix: &mut [&mut [f64]], n: usize, toler: f64) -> i32 {
    let mut eps = 0.0;

    for i in 0..n {
        let diag = matrix[i][i].abs();
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
            for j in i..n {
                matrix[j][i] = 0.0;
            }
        } else {
            rank += 1;
            for j in (i + 1)..n {
                let temp = matrix[j][i] / pivot;
                matrix[j][i] = temp;
                matrix[j][j] -= temp * temp * pivot;

                for k in (j + 1)..n {
                    matrix[k][j] -= temp * matrix[k][i];
                }
            }
        }
    }

    rank
}

fn chinv5(matrix: &mut [&mut [f64]], n: usize, flag: i32) {
    for i in 0..n {
        if matrix[i][i] != 0.0 {
            matrix[i][i] = 1.0 / matrix[i][i];

            for j in (i + 1)..n {
                matrix[j][i] = -matrix[j][i];
                for k in 0..i {
                    matrix[j][k] += matrix[j][i] * matrix[i][k];
                }
            }
        } else {
            for j in (i + 1)..n {
                matrix[j][i] = 0.0;
            }
        }
    }

    if flag == 1 {
        return;
    }

    for i in 0..n {
        if matrix[i][i] == 0.0 {
            for j in 0..i {
                matrix[j][i] = 0.0;
            }
            for j in i..n {
                matrix[i][j] = 0.0;
            }
        } else {
            for j in (i + 1)..n {
                let temp = matrix[j][i] * matrix[j][j];
                if j != i {
                    matrix[i][j] = temp;
                }
                for k in i..j {
                    matrix[i][k] += temp * matrix[j][k];
                }
            }
        }
    }
}
