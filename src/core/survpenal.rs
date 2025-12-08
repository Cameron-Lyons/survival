#![allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PenaltyResult {
    pub new_coef: Vec<f64>,
    pub first_deriv: Vec<f64>,
    pub second_deriv: Vec<f64>,
    pub loglik_penalty: f64,
    pub flags: Vec<i32>,
}

pub fn survpenal(
    whichcase: i32,
    nfrail: usize,
    nvar: usize,
    hmat: &mut [f64],
    #[allow(non_snake_case)] JJ: &mut [f64],
    hdiag: &mut [f64],
    jdiag: &mut [f64],
    u: &mut [f64],
    beta: &mut [f64],
    penalty: &mut f64,
    ptype: i32,
    pdiag: i32,
    sparse_penalty: impl Fn(&[f64]) -> PenaltyResult,
    dense_penalty: impl Fn(&[f64]) -> PenaltyResult,
) {
    let matrix_cols = nvar + nfrail;

    if ptype == 1 || ptype == 3 {
        let sparse_coef = &beta[..nfrail];
        let result = sparse_penalty(sparse_coef);
        *penalty += result.loglik_penalty;

        if whichcase == 0 {
            beta[..nfrail].copy_from_slice(&result.new_coef);

            if result.flags.iter().any(|&f| f > 0) {
                for i in 0..nfrail {
                    hdiag[i] = 1.0;
                    jdiag[i] = 1.0;
                    u[i] = 0.0;
                    for j in 0..nvar {
                        let idx = j * matrix_cols + i;
                        hmat[idx] = 0.0;
                    }
                }
            } else {
                for i in 0..nfrail {
                    u[i] += result.first_deriv[i];
                    hdiag[i] += result.second_deriv[i];
                    jdiag[i] += result.second_deriv[i];
                }
            }
        }
    }

    if ptype > 1 {
        let dense_coef = &beta[nfrail..(nfrail + nvar)];
        let result = dense_penalty(dense_coef);
        *penalty += result.loglik_penalty;

        if whichcase == 0 {
            beta[nfrail..(nfrail + nvar)].copy_from_slice(&result.new_coef);

            for (i, val) in result.first_deriv.iter().enumerate() {
                u[nfrail + i] += val;
            }

            if pdiag == 0 {
                for i in 0..nvar {
                    let idx = i * matrix_cols + (nfrail + i);
                    JJ[idx] += result.second_deriv[i];
                    hmat[idx] += result.second_deriv[i];
                }
            } else {
                let mut k = 0;
                for i in 0..nvar {
                    for j in 0..nvar {
                        let idx = i * matrix_cols + (nfrail + j);
                        JJ[idx] += result.second_deriv[k];
                        hmat[idx] += result.second_deriv[k];
                        k += 1;
                    }
                }
            }

            for i in 0..nvar {
                if result.flags[i] == 1 {
                    u[nfrail + i] = 0.0;
                    let diag_idx = i * matrix_cols + (nfrail + i);
                    hmat[diag_idx] = 1.0;

                    for j in 0..i {
                        let off_idx = i * matrix_cols + (nfrail + j);
                        hmat[off_idx] = 0.0;
                    }
                }
            }
        }
    }
}
