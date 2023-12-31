use pyo3::prelude::*;

#[pyclass]
struct CoxModel {
    maxiter: usize,
    nused: usize,
    nvar: usize,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<u8>,
    covar: Vec<Vec<f64>>,
    strata: Vec<u8>,
    offset: Vec<f64>,
    eps: f64,
    tol_chol: f64,
    // returned parameters
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Vec<Vec<f64>>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter_used: usize,
    work: Vec<f64>,
}

impl CoxModel {
    pub fn new(
        maxiter: usize,
        nused: usize,
        nvar: usize,
        start: Vec<f64>,
        stop: Vec<f64>,
        event: Vec<u8>,
        covar: Vec<Vec<f64>>,
        strata: Vec<u8>,
        offset: Vec<f64>,
        eps: f64,
        tol_chol: f64,
        initial_beta: Vec<f64>,
    ) -> CoxModel {
        let means = vec![0.0; nvar];
        let imat = vec![vec![0.0; nvar]; nvar];
        let u = vec![0.0; nvar];

        let loglik = [0.0, 0.0];
        let sctest = 0.0;
        let flag = 0;

        // Work arrays
        let work_len = nvar * nvar + nvar + nvar;
        let work = vec![0.0; work_len];

        CoxModel {
            maxiter,
            nused,
            nvar,
            start,
            stop,
            event,
            covar,
            strata,
            offset,
            eps,
            tol_chol,
            means,
            beta: initial_beta,
            u,
            imat,
            loglik,
            sctest,
            flag,
            iter_used: 0,
            work,
        }
    }

    pub fn compute(&mut self) {
        for iter in 0..self.maxiter {
            self.score_and_info();

            let delta_beta = self.solve_system();

            for i in 0..self.nvar {
                self.beta[i] += delta_beta[i];
            }

            if self.has_converged(&delta_beta) {
                self.iter_used = iter + 1;
                self.flag = 0; // Indicating successful convergence
                break;
            }
        }

        if self.flag != 0 {
            self.flag = 1000;
        }
        self.finalize_statistics();
    }

    fn has_converged(&self, delta_beta: &[f64]) -> bool {
        // Check if the maximum absolute change in beta coefficients is below the threshold
        delta_beta.iter().all(|&change| change.abs() <= self.eps)
    }

    fn score_and_info(&mut self) {
        let mut score = vec![0.0; self.nvar];
        let mut info_matrix = vec![vec![0.0; self.nvar]; self.nvar];

        let mut strata_start = 0;
        while strata_start < self.nused {
            let strata_end = self.strata[strata_start..]
                .iter()
                .position(|&x| x == 1)
                .map(|x| x + strata_start)
                .unwrap_or(self.nused);

            let mut risk_set_sum = vec![0.0; self.nvar];
            let mut weighted_risk_set_sum = vec![0.0; self.nvar];
            let mut exp_lin_pred = Vec::with_capacity(self.nused);

            for i in strata_start..strata_end {
                let lin_pred = self.offset[i]
                    + self.covar[i]
                        .iter()
                        .zip(&self.beta)
                        .map(|(&x, &b)| x * b)
                        .sum::<f64>();

                let exp_lin = lin_pred.exp();
                exp_lin_pred.push(exp_lin);

                if self.event[i] == 1 {
                    for (j, &cov_ij) in self.covar[i].iter().enumerate() {
                        score[j] += cov_ij;
                        risk_set_sum[j] += cov_ij * exp_lin;
                    }
                }
            }

            let mut denominator = 0.0;
            for i in (strata_start..strata_end).rev() {
                denominator += exp_lin_pred[i - strata_start];
                if self.event[i] == 1 {
                    for j in 0..self.nvar {
                        score[j] -= self.covar[i][j] * weighted_risk_set_sum[j] / denominator;
                        for k in 0..=j {
                            info_matrix[j][k] += (self.covar[i][j]
                                * self.covar[i][k]
                                * exp_lin_pred[i - strata_start])
                                / denominator;
                            if k != j {
                                info_matrix[k][j] = info_matrix[j][k];
                            }
                        }
                    }
                }

                for j in 0..self.nvar {
                    weighted_risk_set_sum[j] += self.covar[i][j] * exp_lin_pred[i - strata_start];
                }
            }

            strata_start = strata_end + 1;
        }

        self.u = score;
        self.imat = info_matrix;
    }

    fn solve_system(&self) -> Vec<f64> {
        let mut cholesky = self.imat.clone();
        if !self.cholesky2(&mut cholesky) {
            panic!("Cholesky decomposition failed, matrix might not be positive definite!");
        }

        let delta_beta = self.chsolve2(&cholesky, &self.u);

        delta_beta
    }

    fn cholesky2(&mut self) -> Result<(), &'static str> {
        for i in 0..self.nvar {
            for j in 0..(i + 1) {
                let mut sum = self.imat[i][j];
                for k in 0..j {
                    sum -= self.imat[i][k] * self.imat[j][k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    self.imat[i][i] = sum.sqrt();
                } else {
                    self.imat[j][i] = sum / self.imat[j][j];
                }
            }
        }
        Ok(())
    }

    fn chsolve2(&self, b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut x = b.to_vec();

        // Solve L * y = b
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..i {
                sum -= self.imat[i][j] * x[j];
            }
            x[i] = sum / self.imat[i][i];
        }

        // Solve L^T * x = y
        for i in (0..n).rev() {
            let mut sum = x[i];
            for j in (i + 1)..n {
                sum -= self.imat[j][i] * x[j];
            }
            x[i] = sum / self.imat[i][i];
        }

        x
    }

    fn chinv2(&mut self) {
        let n = self.nvar;

        // Inverting the lower triangular matrix L
        for i in 0..n {
            self.imat[i][i] = 1.0 / self.imat[i][i];
            for j in (i + 1)..n {
                let mut sum = 0.0;
                for k in (i..j).rev() {
                    sum -= self.imat[j][k] * self.imat[k][i];
                }
                self.imat[j][i] = sum / self.imat[j][j];
            }
        }

        // Computing the inverse of the original matrix from the inverse of L
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                let end = if i < j { i } else { j };
                for k in j..=end {
                    sum += self.imat[k][i] * self.imat[k][j];
                }
                self.imat[i][j] = sum;
                self.imat[j][i] = sum; // because the matrix is symmetric
            }
        }
    }

    fn finalize_statistics(&mut self) {
        let mut loglik = 0.0;
        let mut score_test_statistic = 0.0;

        let mut strata_start = 0;
        while strata_start < self.nused {
            let strata_end = self.strata[strata_start..]
                .iter()
                .position(|&x| x == 1)
                .map(|x| x + strata_start)
                .unwrap_or(self.nused);

            let mut risk_sum = 0.0;

            for i in strata_start..strata_end {
                let lin_pred = self.offset[i]
                    + self.covar[i]
                        .iter()
                        .zip(&self.beta)
                        .map(|(&cov_ij, &beta_j)| cov_ij * beta_j)
                        .sum::<f64>();

                risk_sum += lin_pred.exp();

                if self.event[i] == 1 {
                    loglik += lin_pred - risk_sum.ln();
                }
            }

            strata_start = strata_end + 1;
        }

        if let Some(imat_inv) = self.invert_information_matrix() {
            score_test_statistic = self
                .u
                .iter()
                .zip(imat_inv.iter())
                .map(|(&ui, inv_row)| {
                    ui * inv_row
                        .iter()
                        .zip(&self.u)
                        .map(|(&inv_ij, &uj)| inv_ij * uj)
                        .sum::<f64>()
                })
                .sum();
        }

        self.loglik[1] = loglik;
        self.sctest = score_test_statistic;
    }
    fn invert_information_matrix(&self) -> Result<Vec<Vec<f64>>, &'static str> {
        let n = self.imat.len();
        let mut inv_matrix = vec![vec![0.0; n]; n];
        let mut l = vec![vec![0.0; n]; n];

        // Performing Cholesky decomposition
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                // Summation for diagonals
                if j == i {
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    let diag = self.imat[j][j] - sum;
                    if diag <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    l[j][j] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j] == 0.0 {
                        return Err("Divide by zero encountered");
                    }
                    l[i][j] = (self.imat[i][j] - sum) / l[j][j];
                }
            }
        }

        // Inverting the lower triangular matrix L
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    inv_matrix[i][i] = 1.0 / l[i][i];
                } else {
                    let mut sum = 0.0;
                    for k in j..i {
                        sum += l[i][k] * inv_matrix[k][j];
                    }
                    inv_matrix[i][j] = -sum / l[i][i];
                }
            }
        }

        // Transposing and multiplying by the inverse to get the inverse of the original matrix
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in i..n {
                    sum += inv_matrix[k][i] * inv_matrix[k][j];
                }
                inv_matrix[i][j] = sum;
                if i != j {
                    inv_matrix[j][i] = sum;
                }
            }
        }

        Ok(inv_matrix)
    }
}

#[pymodule]
fn pyCoxModel(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CoxModel>()?;
    Ok(())
}
