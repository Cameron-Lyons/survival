#[derive(Debug, Clone)]
pub(crate) struct PenaltyResult {
    pub new_coef: Vec<f64>,
    pub first_deriv: Vec<f64>,
    pub second_deriv: Vec<f64>,
    pub loglik_penalty: f64,
    pub flags: Vec<i32>,
}

#[allow(non_snake_case)]
pub(crate) struct MatrixBuffers<'a> {
    pub hmat: &'a mut [f64],
    pub JJ: &'a mut [f64],
    pub hdiag: &'a mut [f64],
    pub jdiag: &'a mut [f64],
    pub u: &'a mut [f64],
    pub beta: &'a mut [f64],
}

pub(crate) struct PenaltyParams {
    pub whichcase: i32,
    pub nfrail: usize,
    pub nvar: usize,
    pub ptype: i32,
    pub pdiag: i32,
}

pub(crate) fn survpenal(
    params: PenaltyParams,
    matrices: MatrixBuffers,
    penalty: &mut f64,
    sparse_penalty: impl Fn(&[f64]) -> PenaltyResult,
    dense_penalty: impl Fn(&[f64]) -> PenaltyResult,
) {
    let matrix_cols = params.nvar + params.nfrail;

    if params.ptype == 1 || params.ptype == 3 {
        let sparse_coef = &matrices.beta[..params.nfrail];
        let result = sparse_penalty(sparse_coef);
        *penalty += result.loglik_penalty;

        if params.whichcase == 0 {
            matrices.beta[..params.nfrail].copy_from_slice(&result.new_coef);

            if result.flags.iter().any(|&f| f > 0) {
                for i in 0..params.nfrail {
                    matrices.hdiag[i] = 1.0;
                    matrices.jdiag[i] = 1.0;
                    matrices.u[i] = 0.0;
                    for j in 0..params.nvar {
                        let idx = j * matrix_cols + i;
                        matrices.hmat[idx] = 0.0;
                    }
                }
            } else {
                for i in 0..params.nfrail {
                    matrices.u[i] += result.first_deriv[i];
                    matrices.hdiag[i] += result.second_deriv[i];
                    matrices.jdiag[i] += result.second_deriv[i];
                }
            }
        }
    }

    if params.ptype > 1 {
        let dense_coef = &matrices.beta[params.nfrail..(params.nfrail + params.nvar)];
        let result = dense_penalty(dense_coef);
        *penalty += result.loglik_penalty;

        if params.whichcase == 0 {
            matrices.beta[params.nfrail..(params.nfrail + params.nvar)]
                .copy_from_slice(&result.new_coef);

            for (i, val) in result.first_deriv.iter().enumerate() {
                matrices.u[params.nfrail + i] += val;
            }

            if params.pdiag == 0 {
                for i in 0..params.nvar {
                    let idx = i * matrix_cols + (params.nfrail + i);
                    matrices.JJ[idx] += result.second_deriv[i];
                    matrices.hmat[idx] += result.second_deriv[i];
                }
            } else {
                let mut k = 0;
                for i in 0..params.nvar {
                    for j in 0..params.nvar {
                        let idx = i * matrix_cols + (params.nfrail + j);
                        matrices.JJ[idx] += result.second_deriv[k];
                        matrices.hmat[idx] += result.second_deriv[k];
                        k += 1;
                    }
                }
            }

            for i in 0..params.nvar {
                if result.flags[i] == 1 {
                    matrices.u[params.nfrail + i] = 0.0;
                    let diag_idx = i * matrix_cols + (params.nfrail + i);
                    matrices.hmat[diag_idx] = 1.0;

                    for j in 0..i {
                        let off_idx = i * matrix_cols + (params.nfrail + j);
                        matrices.hmat[off_idx] = 0.0;
                    }
                }
            }
        }
    }
}
