#[cfg(test)]
pub fn chinv2(matrix: &mut [f64], n: usize) {
    assert_eq!(matrix.len(), n * n, "Matrix must be of size n x n");
    for i in 0..n {
        let diag_idx = i * n + i;
        if matrix[diag_idx] > 0.0 {
            matrix[diag_idx] = 1.0 / matrix[diag_idx];
            for j in (i + 1)..n {
                let j_i = j * n + i;
                matrix[j_i] = -matrix[j_i];
                for k in 0..i {
                    let j_k = j * n + k;
                    let i_k = i * n + k;
                    matrix[j_k] += matrix[j_i] * matrix[i_k];
                }
            }
        }
    }
    for i in 0..n {
        let diag_idx = i * n + i;
        if matrix[diag_idx] == 0.0 {
            for j in 0..i {
                matrix[j * n + i] = 0.0;
            }
            for j in i..n {
                matrix[i * n + j] = 0.0;
            }
        } else {
            for j in (i + 1)..n {
                let j_i = j * n + i;
                let j_j = j * n + j;
                let temp = matrix[j_i] * matrix[j_j];
                matrix[i * n + j] = temp;
                for k in i..j {
                    let i_k = i * n + k;
                    let j_k = j * n + k;
                    matrix[i_k] += temp * matrix[j_k];
                }
            }
        }
    }
}
