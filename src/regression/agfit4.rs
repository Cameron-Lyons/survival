#[allow(dead_code)]
fn cholesky(mat: &mut [f64], n: usize, tol: f64) -> i32 {
    let mut rank = 0;
    for col in 0..n {
        let diag = col * n + col;
        let mut sum = mat[diag];
        for k in 0..col {
            sum -= mat[col * n + k].powi(2);
        }
        if sum > tol {
            mat[diag] = sum.sqrt();
            rank += 1;
        } else {
            mat[diag] = 0.0;
        }
        for row in (col + 1)..n {
            let idx = row * n + col;
            let mut sum = mat[idx];
            for k in 0..col {
                sum -= mat[row * n + k] * mat[col * n + k];
            }
            if mat[diag] > 0.0 {
                mat[idx] = sum / mat[diag];
            } else {
                mat[idx] = 0.0;
            }
        }
    }
    rank
}

#[allow(dead_code)]
fn chsolve2(mat: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let diag = i * n + i;
        if mat[diag] == 0.0 {
            b[i] = 0.0;
            continue;
        }
        let mut temp = b[i];
        for j in 0..i {
            temp -= mat[i * n + j] * b[j];
        }
        b[i] = temp / mat[diag];
    }
    for i in (0..n).rev() {
        let diag = i * n + i;
        if mat[diag] == 0.0 {
            b[i] = 0.0;
            continue;
        }
        let mut temp = b[i];
        for j in (i + 1)..n {
            temp -= mat[j * n + i] * b[j];
        }
        b[i] = temp / mat[diag];
    }
}
