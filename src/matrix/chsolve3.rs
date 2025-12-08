#![allow(dead_code)]
fn chsolve3(matrix: &mut [&mut [f64]], n: usize, m: usize, diag: &[f64], y: &mut [f64]) {
    let n2 = n - m;

    for i in 0..n2 {
        let mut temp = y[i + m];
        for j in 0..m {
            temp -= y[j] * matrix[i][j];
        }
        for j in 0..i {
            temp -= y[j + m] * matrix[i][j + m];
        }
        y[i + m] = temp;
    }
    for i in (0..n2).rev() {
        let pivot = matrix[i][i + m];
        if pivot == 0.0 {
            y[i + m] = 0.0;
        } else {
            let mut temp = y[i + m] / pivot;
            for j in (i + 1)..n2 {
                temp -= y[j + m] * matrix[j][i + m];
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
                temp -= y[j + m] * matrix[j][i];
            }
            y[i] = temp;
        }
    }
}
