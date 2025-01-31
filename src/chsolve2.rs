fn chsolve2(matrix: &mut [&mut [f64]], n: usize, y: &mut [f64]) {
    for i in 0..n {
        let mut temp = y[i];
        for j in 0..i {
            temp -= y[j] * matrix[i][j];
        }
        y[i] = temp;
    }

    for i in (0..n).rev() {
        if matrix[i][i].abs() < f64::EPSILON {
            y[i] = 0.0;
        } else {
            let denominator = matrix[i][i];
            let mut temp = y[i] / denominator;
            for j in (i + 1)..n {
                temp -= y[j] * matrix[j][i];
            }
            y[i] = temp;
        }
    }
}
