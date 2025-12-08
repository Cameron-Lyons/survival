#![allow(dead_code)]
pub fn coxph_wtest(
    nvar2: &mut i32,
    ntest: &i32,
    var: &[f64],
    b: &mut [f64],
    solve: &mut [f64],
    tolerch: f64,
) {
    let nvar = *nvar2 as usize;
    let ntest = *ntest as usize;

    let mut var2 = vec![vec![0.0; nvar]; nvar];
    for j in 0..nvar {
        for i in 0..nvar {
            var2[i][j] = var[j * nvar + i];
        }
    }

    cholesky2(&mut var2, tolerch);

    let df = var2
        .iter()
        .enumerate()
        .filter(|(i, row)| row[*i] > 0.0)
        .count();
    *nvar2 = df as i32;

    for i in 0..ntest {
        let b_start = i * nvar;
        let b_end = b_start + nvar;
        let solve_start = i * nvar;
        let solve_end = solve_start + nvar;

        solve[solve_start..solve_end].copy_from_slice(&b[b_start..b_end]);

        chsolve2(&var2, &mut solve[solve_start..solve_end]);

        let sum = b[b_start..b_end]
            .iter()
            .zip(&solve[solve_start..solve_end])
            .map(|(b_val, solve_val)| b_val * solve_val)
            .sum();

        b[i] = sum;
    }
}

fn cholesky2(matrix: &mut [Vec<f64>], tolerch: f64) {
    let n = matrix.len();
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= matrix[i][k] * matrix[j][k];
            }
            if j == i {
                matrix[i][j] = if sum > tolerch { sum.sqrt() } else { 0.0 };
            } else {
                matrix[i][j] = if matrix[j][j] != 0.0 {
                    sum / matrix[j][j]
                } else {
                    0.0
                };
            }
        }
    }
}

fn chsolve2(matrix: &[Vec<f64>], x: &mut [f64]) {
    let n = matrix.len();
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= matrix[i][k] * x[k];
        }
        x[i] = if matrix[i][i] != 0.0 {
            sum / matrix[i][i]
        } else {
            0.0
        };
    }

    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= matrix[k][i] * x[k];
        }
        x[i] = if matrix[i][i] != 0.0 {
            sum / matrix[i][i]
        } else {
            0.0
        };
    }
}
