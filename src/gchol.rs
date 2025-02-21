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

fn cholesky5(_mat: &mut [f64], _n: usize, _toler: f64) -> i32 {
    0
}

fn chsolve5(_mat: &mut [f64], _n: usize, _y: &mut [f64], _flag: i32) {}

fn chinv5(_mat: &mut [f64], _n: usize, _flag: i32) {}
