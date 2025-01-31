fn chsolve5(matrix: &mut [&mut [f64]], n: usize, y: &mut [f64], flag: i32) {
    if flag < 2 {
        for i in 0..n {
            let mut temp = y[i];
            for j in 0..i {
                temp -= y[j] * matrix[i][j];
            }
            y[i] = temp;
        }
    }
    if flag > 0 {
        for i in 0..n {
            y[i] = if matrix[i][i] > 0.0 {
                y[i] / matrix[i][i].sqrt()
            } else {
                0.0
            };
        }
    } else {
        for i in 0..n {
            y[i] = if matrix[i][i] != 0.0 {
                y[i] / matrix[i][i]
            } else {
                0.0
            };
        }
    }
    if flag != 1 {
        for i in (0..n).rev() {
            let mut temp = y[i];
            for j in (i + 1)..n {
                temp -= y[j] * matrix[j][i];
            }
            y[i] = temp;
        }
    }
}
