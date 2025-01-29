/// Performs Cholesky inversion for matrices with diagonal upper portion and dense lower portion
///
/// # Arguments
/// * `matrix` - Mutable slice representing the dense lower portion (n-m rows x n columns)
/// * `n` - Total matrix dimension
/// * `m` - Size of diagonal upper portion
/// * `fdiag` - Mutable slice containing diagonal elements of upper portion
pub fn chinv3(matrix: &mut [f64], n: usize, m: usize, fdiag: &mut [f64]) {
    let n2 = n - m;
    assert_eq!(
        matrix.len(),
        n2 * n,
        "Matrix must have n-m rows of n elements"
    );
    assert_eq!(fdiag.len(), m, "fdiag must contain m elements");

    // Invert diagonal upper portion
    for i in 0..m {
        if fdiag[i] > 0.0 {
            fdiag[i] = 1.0 / fdiag[i];
            // Negate corresponding column in dense portion
            for row in 0..n2 {
                matrix[row * n + i] = -matrix[row * n + i];
            }
        }
    }

    // Invert dense lower portion
    for i in 0..n2 {
        let original_row = i + m;
        let diag_idx = i * n + original_row;

        if matrix[diag_idx] > 0.0 {
            matrix[diag_idx] = 1.0 / matrix[diag_idx];

            for j in (i + 1)..n2 {
                let j_row = j * n + original_row;
                matrix[j_row] = -matrix[j_row];

                for k in 0..original_row {
                    let j_col = j * n + k;
                    let i_col = i * n + k;
                    matrix[j_col] += matrix[j_row] * matrix[i_col];
                }
            }
        }
    }
}

/// Computes matrix product for the specialized Cholesky inversion
pub fn chprod3(matrix: &mut [f64], n: usize, m: usize, fdiag: &[f64]) {
    let n2 = n - m;
    assert_eq!(
        matrix.len(),
        n2 * n,
        "Matrix must have n-m rows of n elements"
    );
    assert_eq!(fdiag.len(), m, "fdiag must contain m elements");

    for i in 0..n2 {
        let original_row = i + m;
        let diag_idx = i * n + original_row;

        if matrix[diag_idx] == 0.0 {
            // Handle singular row
            for j in 0..i {
                matrix[j * n + original_row] = 0.0;
            }
            for j in original_row..n {
                matrix[i * n + j] = 0.0;
            }
        } else {
            for j in (i + 1)..n2 {
                let j_row = j * n + original_row;
                let j_diag = j * n + (j + m);
                let temp = matrix[j_row] * matrix[j_diag];

                if j != i {
                    matrix[i * n + (j + m)] = temp;
                }

                for k in i..j {
                    let k_col = k + m;
                    matrix[i * n + k_col] += temp * matrix[j * n + k_col];
                }
            }
        }
    }
}
