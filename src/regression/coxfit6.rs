use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::cholesky::CholeskyInto;
use ndarray_linalg::{error::Result as LinalgResult, Cholesky, Inverse, Solve};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoxError {
    #[error("Cholesky decomposition failed")]
    CholeskyDecomposition,
    #[error("Matrix inversion failed")]
    MatrixInversion,
    #[error("Non-finite values encountered during iteration")]
    NonFinite,
}

#[derive(Debug, Clone, Copy)]
pub enum Method {
    Breslow,
    Efron,
}

pub struct CoxFit {
    time: Array1<f64>,
    status: Array1<i32>,
    covar: Array2<f64>,
    strata: Array1<i32>,
    offset: Array1<f64>,
    weights: Array1<f64>,
    method: Method,
    max_iter: usize,
    eps: f64,
    toler: f64,
    scale: Vec<f64>,
    means: Vec<f64>,
    beta: Vec<f64>,
    u: Vec<f64>,
    imat: Array2<f64>,
    loglik: [f64; 2],
    sctest: f64,
    flag: i32,
    iter: usize,
}

impl CoxFit {
    pub fn new(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        method: Method,
        max_iter: usize,
        eps: f64,
        toler: f64,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        let nvar = covar.ncols();
        let nused = covar.nrows();

        let mut cox = Self {
            time,
            status,
            covar,
            strata,
            offset,
            weights,
            method,
            max_iter,
            eps,
            toler,
            scale: vec![1.0; nvar],
            means: vec![0.0; nvar],
            beta: initial_beta,
            u: vec![0.0; nvar],
            imat: Array2::zeros((nvar, nvar)),
            loglik: [0.0; 2],
            sctest: 0.0,
            flag: 0,
            iter: 0,
        };

        cox.scale_center(doscale)?;
        Ok(cox)
    }

    fn scale_center(&mut self, doscale: Vec<bool>) -> Result<(), CoxError> {
        let nused = self.weights.len();
        let nvar = self.covar.ncols();
        let total_weight: f64 = self.weights.sum();

        for i in 0..nvar {
            if !doscale[i] {
                self.scale[i] = 1.0;
                self.means[i] = 0.0;
                continue;
            }

            let mut mean = 0.0;
            for (person, &w) in self.weights.iter().enumerate() {
                mean += w * self.covar[(person, i)];
            }
            mean /= total_weight;
            self.means[i] = mean;

            for person in 0..nused {
                self.covar[(person, i)] -= mean;
            }

            let mut abs_sum = 0.0;
            for (person, &w) in self.weights.iter().enumerate() {
                abs_sum += w * self.covar[(person, i)].abs();
            }

            self.scale[i] = if abs_sum > 0.0 {
                total_weight / abs_sum
            } else {
                1.0
            };

            for person in 0..nused {
                self.covar[(person, i)] *= self.scale[i];
            }
        }

        for i in 0..nvar {
            self.beta[i] /= self.scale[i];
        }

        Ok(())
    }

    fn iterate(&mut self, beta: &[f64]) -> Result<f64, CoxError> {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let method = self.method;

        self.u.fill(0.0);
        self.imat.fill(0.0);
        let mut a = vec![0.0; nvar];
        let mut a2 = vec![0.0; nvar];
        let mut cmat = Array2::zeros((nvar, nvar));
        let mut cmat2 = Array2::zeros((nvar, nvar));

        let mut loglik = 0.0;
        let mut person = nused as isize - 1;

        while person >= 0 {
            let person_idx = person as usize;
            if self.strata[person_idx] == 1 {
                a.fill(0.0);
                cmat.fill(0.0);
            }

            let dtime = self.time[person_idx];
            let mut ndead = 0;
            let mut deadwt = 0.0;
            let mut denom2 = 0.0;
            let mut nrisk = 0;
            let mut denom = 0.0;

            while person >= 0 && self.time[person as usize] == dtime {
                let person_i = person as usize;
                nrisk += 1;
                let zbeta = self.offset[person_i]
                    + beta
                        .iter()
                        .enumerate()
                        .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(person_i, i)]);
                let risk = zbeta.exp() * self.weights[person_i];

                if self.status[person_i] == 0 {
                    denom += risk;
                    for i in 0..nvar {
                        a[i] += risk * self.covar[(person_i, i)];
                        for j in 0..=i {
                            cmat[(i, j)] +=
                                risk * self.covar[(person_i, i)] * self.covar[(person_i, j)];
                        }
                    }
                } else {
                    ndead += 1;
                    deadwt += self.weights[person_i];
                    denom2 += risk;
                    loglik += self.weights[person_i] * zbeta;

                    for i in 0..nvar {
                        self.u[i] += self.weights[person_i] * self.covar[(person_i, i)];
                        a2[i] += risk * self.covar[(person_i, i)];
                        for j in 0..=i {
                            cmat2[(i, j)] +=
                                risk * self.covar[(person_i, i)] * self.covar[(person_i, j)];
                        }
                    }
                }

                person -= 1;
                if person >= 0 && self.strata[person as usize] == 1 {
                    break;
                }
            }

            if ndead > 0 {
                match method {
                    Method::Breslow => {
                        denom += denom2;
                        loglik -= deadwt * denom.ln();

                        for i in 0..nvar {
                            a[i] += a2[i];
                            let temp = a[i] / denom;
                            self.u[i] -= deadwt * temp;

                            for j in 0..=i {
                                cmat[(i, j)] += cmat2[(i, j)];
                                let val = deadwt * (cmat[(i, j)] - temp * a[j]) / denom;
                                self.imat[(j, i)] += val;
                                if i != j {
                                    self.imat[(i, j)] += val;
                                }
                            }
                        }
                    }
                    Method::Efron => {
                        let wtave = deadwt / ndead as f64;
                        for k in 0..ndead {
                            let kf = k as f64;
                            denom += denom2 / ndead as f64;
                            loglik -= wtave * denom.ln();

                            for i in 0..nvar {
                                a[i] += a2[i] / ndead as f64;
                                let temp = a[i] / denom;
                                self.u[i] -= wtave * temp;

                                for j in 0..=i {
                                    cmat[(i, j)] += cmat2[(i, j)] / ndead as f64;
                                    let val = wtave * (cmat[(i, j)] - temp * a[j]) / denom;
                                    self.imat[(j, i)] += val;
                                    if i != j {
                                        self.imat[(i, j)] += val;
                                    }
                                }
                            }
                        }
                    }
                }

                a2.fill(0.0);
                cmat2.fill(0.0);
            }
        }

        Ok(loglik)
    }

    pub fn fit(&mut self) -> Result<(), CoxError> {
        let nvar = self.beta.len();
        let mut newbeta = vec![0.0; nvar];
        let mut a = vec![0.0; nvar];
        let mut halving = 0;
        let mut notfinite;

        self.loglik[0] = self.iterate(&self.beta)?;
        self.loglik[1] = self.loglik[0];

        a.copy_from_slice(&self.u);
        self.flag = Self::cholesky(&mut self.imat, self.toler)?;
        Self::chsolve(&self.imat, &mut a)?;

        self.sctest = a.iter().zip(&self.u).map(|(ai, ui)| ai * ui).sum();

        if self.max_iter == 0 || !self.loglik[0].is_finite() {
            Self::chinv(&mut self.imat)?;
            self.rescale_params();
            return Ok(());
        }

        newbeta.copy_from_slice(&self.beta);
        for i in 0..nvar {
            newbeta[i] += a[i];
        }

        self.loglik[1] = self.loglik[0];
        for iter in 1..=self.max_iter {
            self.iter = iter;
            let newlk = match self.iterate(&newbeta) {
                Ok(lk) if lk.is_finite() => lk,
                _ => {
                    notfinite = true;
                    f64::NAN
                }
            };

            notfinite = !newlk.is_finite();
            if !notfinite {
                for i in 0..nvar {
                    if !self.u[i].is_finite() {
                        notfinite = true;
                        break;
                    }
                    for j in 0..nvar {
                        if !self.imat[(i, j)].is_finite() {
                            notfinite = true;
                            break;
                        }
                    }
                }
            }

            if !notfinite && ((self.loglik[1] - newlk).abs() / newlk.abs() <= self.eps) {
                self.loglik[1] = newlk;
                Self::chinv(&mut self.imat)?;
                self.rescale_params();
                if halving > 0 {
                    self.flag = -2;
                }
                return Ok(());
            }

            if notfinite || newlk < self.loglik[1] {
                halving += 1;
                for i in 0..nvar {
                    newbeta[i] =
                        (newbeta[i] + (halving as f64) * self.beta[i]) / (halving as f64 + 1.0);
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                a.copy_from_slice(&self.u);
                Self::chsolve(&self.imat, &mut a)?;
                for i in 0..nvar {
                    newbeta[i] = self.beta[i] + a[i];
                }
            }
        }

        self.loglik[1] = self.iterate(&self.beta)?;
        Self::chinv(&mut self.imat)?;
        self.rescale_params();
        self.flag = 1000;

        Ok(())
    }

    fn rescale_params(&mut self) {
        let nvar = self.beta.len();
        for i in 0..nvar {
            self.beta[i] *= self.scale[i];
            self.u[i] /= self.scale[i];
            for j in 0..nvar {
                self.imat[(i, j)] *= self.scale[i] * self.scale[j];
            }
        }
    }

    fn cholesky(mat: &mut Array2<f64>, toler: f64) -> Result<i32, CoxError> {
        let n = mat.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                mat[(i, j)] = mat[(j, i)];
            }
        }

        match mat.cholesky_into() {
            Ok(_) => Ok(n as i32),
            Err(_) => {
                for i in 0..n {
                    if mat[(i, i)] < toler {
                        return Ok(i as i32);
                    }
                }
                Err(CoxError::CholeskyDecomposition)
            }
        }
    }

    fn chsolve(chol: &Array2<f64>, a: &mut [f64]) -> Result<(), CoxError> {
        let n = chol.nrows();
        let mut b = Array1::from_vec(a.to_vec());
        if let Err(_) = chol.solve_mut(&mut b) {
            return Err(CoxError::CholeskyDecomposition);
        }
        a.copy_from_slice(&b.to_vec());
        Ok(())
    }

    fn chinv(mat: &mut Array2<f64>) -> Result<(), CoxError> {
        let n = mat.nrows();
        let chol = match mat.cholesky_into() {
            Ok(chol) => chol,
            Err(_) => return Err(CoxError::MatrixInversion),
        };
        let inv = match chol.inv() {
            Ok(inv) => inv,
            Err(_) => return Err(CoxError::MatrixInversion),
        };
        *mat = inv;
        Ok(())
    }

    pub fn results(
        self,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Array2<f64>,
        [f64; 2],
        f64,
        i32,
        usize,
    ) {
        (
            self.beta,
            self.means,
            self.u,
            self.imat,
            self.loglik,
            self.sctest,
            self.flag,
            self.iter,
        )
    }
}
