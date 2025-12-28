#[cfg(test)]
pub fn cholesky2(matrix: &mut [f64], n: usize, toler: f64) -> i32 {
    let mut eps = 0.0;
    let mut nonneg = 1;
    let mut rank = 0;
    for i in 0..n {
        let diag_idx = i * n + i;
        let diag = matrix[diag_idx];
        if diag > eps {
            eps = diag;
        }
        for j in (i + 1)..n {
            let upper_idx = i * n + j;
            let lower_idx = j * n + i;
            matrix[lower_idx] = matrix[upper_idx];
        }
    }
    if eps == 0.0 {
        eps = toler;
    } else {
        eps *= toler;
    }
    for i in 0..n {
        let diag_idx = i * n + i;
        let pivot = matrix[diag_idx];
        if !pivot.is_finite() || pivot < eps {
            matrix[diag_idx] = 0.0;
            if pivot < -8.0 * eps {
                nonneg = -1;
            }
        } else {
            rank += 1;
            for j in (i + 1)..n {
                let ji_idx = j * n + i;
                let temp = matrix[ji_idx] / pivot;
                matrix[ji_idx] = temp;
                let jj_idx = j * n + j;
                matrix[jj_idx] -= temp * temp * pivot;
                for k in (j + 1)..n {
                    let kj_idx = k * n + j;
                    let ki_idx = k * n + i;
                    matrix[kj_idx] -= temp * matrix[ki_idx];
                }
            }
        }
    }
    rank * nonneg
}
