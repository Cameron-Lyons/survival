#![allow(dead_code)]
pub fn gchol(matrix: &[f64], n: usize, toler: f64) -> Vec<f64> {
    let mut gc = matrix.to_vec();
    let _status = cholesky5(&mut gc, n, toler);

    for i in 0..n {
        for j in (i + 1)..n {
            let idx = j + i * n;
            gc[idx] = 0.0;
        }
    }
    gc
}

pub fn gchol_solve(x: &[f64], y: &mut [f64], n: usize, flag: i32) -> Vec<f64> {
    let mut new_x = x.to_vec();
    chsolve5(&mut new_x, n, y, flag);
    new_x
}

pub fn gchol_inv(matrix: &[f64], n: usize, flag: i32) -> Vec<f64> {
    let mut new_matrix = matrix.to_vec();
    chinv5(&mut new_matrix, n, flag);

    if flag == 1 {
        for i in 0..n {
            let diag_idx = i + i * n;
            new_matrix[diag_idx] = 1.0;
            for j in (i + 1)..n {
                let idx = j + i * n;
                new_matrix[idx] = 0.0;
            }
        }
    } else {
        for i in 0..n {
            for j in (i + 1)..n {
                let lower_idx = j + i * n;
                let upper_idx = i + j * n;
                new_matrix[upper_idx] = new_matrix[lower_idx];
            }
        }
    }
    new_matrix
}

pub fn cholesky5(mat: &mut [f64], n: usize, toler: f64) -> i32 {
    let mut rank = 0;

    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i {
            let idx = i + k * n;
            sum += mat[idx].powi(2);
        }
        let diag_idx = i + i * n;
        mat[diag_idx] -= sum;

        if mat[diag_idx] <= toler {
            mat[diag_idx] = 0.0;
        } else {
            rank += 1;
            mat[diag_idx] = mat[diag_idx].sqrt();

            for j in (i + 1)..n {
                let mut col_sum = 0.0;
                for k in 0..i {
                    col_sum += mat[j + k * n] * mat[i + k * n];
                }
                let ji_idx = j + i * n;
                mat[ji_idx] = (mat[ji_idx] - col_sum) / mat[diag_idx];
            }
        }

        for j in (i + 1)..n {
            let ij_idx = i + j * n;
            mat[ij_idx] = 0.0;
        }
    }
    rank
}

pub fn chsolve5(mat: &mut [f64], n: usize, y: &mut [f64], flag: i32) {
    if flag == 1 {
        for i in 0..n {
            let diag_idx = i + i * n;
            if mat[diag_idx] == 0.0 {
                y[i] = 0.0;
                continue;
            }

            let mut temp = y[i];
            for j in 0..i {
                temp -= y[j] * mat[i + j * n];
            }
            y[i] = temp / mat[diag_idx];
        }
    } else {
        for i in (0..n).rev() {
            let diag_idx = i + i * n;
            if mat[diag_idx] == 0.0 {
                y[i] = 0.0;
                continue;
            }

            let mut temp = y[i];
            for j in (i + 1)..n {
                temp -= y[j] * mat[j + i * n];
            }
            y[i] = temp / mat[diag_idx];
        }
    }
}

pub fn chinv5(mat: &mut [f64], n: usize, flag: i32) {
    for i in 0..n {
        let diag_idx = i + i * n;
        if mat[diag_idx] > 0.0 {
            mat[diag_idx] = 1.0 / mat[diag_idx];

            for j in (i + 1)..n {
                let ji_idx = j + i * n;
                mat[ji_idx] = -mat[ji_idx];

                for k in 0..i {
                    mat[j + k * n] += mat[ji_idx] * mat[i + k * n];
                }
            }
        }
    }

    if flag == 1 {
        for i in 0..n {
            mat[i + i * n] = 1.0;
            for j in (i + 1)..n {
                mat[i + j * n] = 0.0;
            }
        }
        return;
    }

    for i in 0..n {
        let diag_idx = i + i * n;
        if mat[diag_idx] == 0.0 {
            for j in 0..=i {
                mat[j + i * n] = 0.0;
            }
            for j in i..n {
                mat[i + j * n] = 0.0;
            }
        } else {
            for j in (i + 1)..n {
                let ji_idx = j + i * n;
                let temp = mat[ji_idx] * mat[j + j * n];
                mat[i + j * n] = temp;

                for k in i..j {
                    mat[i + k * n] += temp * mat[j + k * n];
                }
            }
        }
    }
}
