use crate::constants::{
    CHOLESKY_TOL, CONVERGENCE_EPSILON, CONVERGENCE_FLAG, COX_MAX_ITER, DEFAULT_MAX_ITER,
    PARALLEL_THRESHOLD_MEDIUM, RIDGE_REGULARIZATION,
};
use crate::utilities::matrix::{cholesky_check, lu_solve, matrix_inverse};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct CoxFitConfig {
    pub method: Method,

    pub max_iter: usize,

    pub eps: f64,

    pub toler: f64,
}

impl Default for CoxFitConfig {
    fn default() -> Self {
        Self {
            method: Method::Breslow,
            max_iter: DEFAULT_MAX_ITER,
            eps: CONVERGENCE_EPSILON,
            toler: CHOLESKY_TOL,
        }
    }
}

#[derive(Debug)]
pub enum CoxError {
    CholeskyDecomposition,
    MatrixInversion,
}

impl fmt::Display for CoxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoxError::CholeskyDecomposition => write!(f, "Cholesky decomposition failed"),
            CoxError::MatrixInversion => write!(f, "Matrix inversion failed"),
        }
    }
}

impl std::error::Error for CoxError {}
#[derive(Debug, Clone, Copy)]
pub enum Method {
    Breslow,
    #[allow(dead_code)]
    Efron,
}
pub type CoxFitResults = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Array2<f64>,
    [f64; 2],
    f64,
    i32,
    usize,
);
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

pub struct CoxFitBuilder {
    time: Array1<f64>,
    status: Array1<i32>,
    covar: Array2<f64>,
    strata: Option<Array1<i32>>,
    offset: Option<Array1<f64>>,
    weights: Option<Array1<f64>>,
    method: Method,
    max_iter: usize,
    eps: f64,
    toler: f64,
    doscale: Option<Vec<bool>>,
    initial_beta: Option<Vec<f64>>,
}

impl CoxFitBuilder {
    pub fn new(time: Array1<f64>, status: Array1<i32>, covar: Array2<f64>) -> Self {
        Self {
            time,
            status,
            covar,
            strata: None,
            offset: None,
            weights: None,
            method: Method::Breslow,
            max_iter: COX_MAX_ITER,
            eps: CONVERGENCE_EPSILON,
            toler: CONVERGENCE_EPSILON,
            doscale: None,
            initial_beta: None,
        }
    }

    pub fn strata(mut self, strata: Array1<i32>) -> Self {
        self.strata = Some(strata);
        self
    }

    #[allow(dead_code)]
    pub fn offset(mut self, offset: Array1<f64>) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    pub fn method(mut self, method: Method) -> Self {
        self.method = method;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn toler(mut self, toler: f64) -> Self {
        self.toler = toler;
        self
    }

    #[allow(dead_code)]
    pub fn doscale(mut self, doscale: Vec<bool>) -> Self {
        self.doscale = Some(doscale);
        self
    }

    pub fn initial_beta(mut self, initial_beta: Vec<f64>) -> Self {
        self.initial_beta = Some(initial_beta);
        self
    }

    pub fn build(self) -> Result<CoxFit, CoxError> {
        let nused = self.covar.nrows();
        let nvar = self.covar.ncols();

        let strata = self.strata.unwrap_or_else(|| Array1::from_elem(nused, 0));
        let offset = self.offset.unwrap_or_else(|| Array1::from_elem(nused, 0.0));
        let weights = self
            .weights
            .unwrap_or_else(|| Array1::from_elem(nused, 1.0));
        let doscale = self.doscale.unwrap_or_else(|| vec![true; nvar]);
        let initial_beta = self.initial_beta.unwrap_or_else(|| vec![0.0; nvar]);

        CoxFit::new(
            self.time,
            self.status,
            self.covar,
            strata,
            offset,
            weights,
            self.method,
            self.max_iter,
            self.eps,
            self.toler,
            doscale,
            initial_beta,
        )
    }
}
impl CoxFit {
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        time: Array1<f64>,
        status: Array1<i32>,
        covar: Array2<f64>,
        strata: Array1<i32>,
        offset: Array1<f64>,
        weights: Array1<f64>,
        config: CoxFitConfig,
        doscale: Vec<bool>,
        initial_beta: Vec<f64>,
    ) -> Result<Self, CoxError> {
        let nvar = covar.ncols();
        let _nused = covar.nrows();
        let mut cox = Self {
            time,
            status,
            covar,
            strata,
            offset,
            weights,
            method: config.method,
            max_iter: config.max_iter,
            eps: config.eps,
            toler: config.toler,
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

    #[allow(clippy::too_many_arguments)]
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
        let config = CoxFitConfig {
            method,
            max_iter,
            eps,
            toler,
        };
        Self::with_config(
            time,
            status,
            covar,
            strata,
            offset,
            weights,
            config,
            doscale,
            initial_beta,
        )
    }
    fn scale_center(&mut self, doscale: Vec<bool>) -> Result<(), CoxError> {
        let nvar = self.covar.ncols();
        let nused = self.covar.nrows();
        let total_weight: f64 = self.weights.sum();
        let means: Vec<f64> = (0..nvar)
            .into_par_iter()
            .map(|i| {
                if !doscale[i] {
                    0.0
                } else {
                    let mut mean = 0.0;
                    for (person, &w) in self.weights.iter().enumerate() {
                        mean += w * self.covar[(person, i)];
                    }
                    mean / total_weight
                }
            })
            .collect();
        let scales: Vec<f64> = (0..nvar)
            .into_par_iter()
            .map(|i| {
                if !doscale[i] {
                    1.0
                } else {
                    let mean = means[i];
                    let abs_sum: f64 = (0..nused)
                        .map(|person| self.weights[person] * (self.covar[(person, i)] - mean).abs())
                        .sum();
                    if abs_sum > 0.0 {
                        total_weight / abs_sum
                    } else {
                        1.0
                    }
                }
            })
            .collect();
        if nused > PARALLEL_THRESHOLD_MEDIUM && nvar > 1 {
            use std::sync::atomic::{AtomicPtr, Ordering};
            let covar_ptr = AtomicPtr::new(self.covar.as_mut_ptr());
            let covar_stride = self.covar.strides();
            let row_stride = covar_stride[0];
            let col_stride = covar_stride[1];

            (0..nvar).into_par_iter().for_each(|i| {
                if doscale[i] {
                    let mean = means[i];
                    let scale_val = scales[i];
                    let base_ptr = covar_ptr.load(Ordering::Relaxed);
                    for person in 0..nused {
                        unsafe {
                            let offset = person as isize * row_stride + i as isize * col_stride;
                            let ptr = base_ptr.offset(offset);
                            *ptr = (*ptr - mean) * scale_val;
                        }
                    }
                }
            });
        } else {
            for i in 0..nvar {
                if doscale[i] {
                    let mean = means[i];
                    let scale_val = scales[i];
                    for person in 0..nused {
                        self.covar[(person, i)] = (self.covar[(person, i)] - mean) * scale_val;
                    }
                }
            }
        }
        self.means = means;
        self.scale = scales;
        let new_beta: Vec<f64> = self
            .beta
            .par_iter()
            .zip(self.scale.par_iter())
            .map(|(&b, &s)| b / s)
            .collect();
        self.beta = new_beta;
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
        let mut denom = 0.0;

        let (zbeta_vals, risk_vals): (Vec<f64>, Vec<f64>) = if nused > PARALLEL_THRESHOLD_MEDIUM {
            (0..nused)
                .into_par_iter()
                .map(|p| {
                    let zb = self.offset[p]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(p, i)]);
                    (zb, zb.exp() * self.weights[p])
                })
                .unzip()
        } else {
            (0..nused)
                .map(|p| {
                    let zb = self.offset[p]
                        + beta
                            .iter()
                            .enumerate()
                            .fold(0.0, |acc, (i, &b)| acc + b * self.covar[(p, i)]);
                    (zb, zb.exp() * self.weights[p])
                })
                .unzip()
        };

        let mut person = nused as isize - 1;
        while person >= 0 {
            let person_idx = person as usize;
            if self.strata[person_idx] == 1 {
                a.fill(0.0);
                cmat.fill(0.0);
                denom = 0.0;
            }
            let dtime = self.time[person_idx];
            let mut ndead = 0;
            let mut deadwt = 0.0;
            let mut denom2 = 0.0;
            let mut _nrisk = 0;
            while person >= 0 && self.time[person as usize] == dtime {
                let person_i = person as usize;
                _nrisk += 1;
                let zbeta = zbeta_vals[person_i];
                let risk = risk_vals[person_i];
                if self.status[person_i] == 0 {
                    denom += risk;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..nvar {
                        let covar_i = self.covar[(person_i, i)];
                        let risk_covar_i = risk * covar_i;
                        a[i] += risk_covar_i;
                        for j in 0..=i {
                            cmat[(i, j)] += risk_covar_i * self.covar[(person_i, j)];
                        }
                    }
                } else {
                    ndead += 1;
                    deadwt += self.weights[person_i];
                    denom2 += risk;
                    loglik += self.weights[person_i] * zbeta;
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..nvar {
                        let covar_i = self.covar[(person_i, i)];
                        self.u[i] += self.weights[person_i] * covar_i;
                        let risk_covar_i = risk * covar_i;
                        a2[i] += risk_covar_i;
                        for j in 0..=i {
                            cmat2[(i, j)] += risk_covar_i * self.covar[(person_i, j)];
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
                        #[allow(clippy::needless_range_loop)]
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
                            let _kf = k as f64;
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
        let mut _notfinite;
        let beta_copy = self.beta.clone();
        self.loglik[0] = self.iterate(&beta_copy)?;
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
                    _notfinite = true;
                    f64::NAN
                }
            };
            _notfinite = !newlk.is_finite();
            if !_notfinite {
                for i in 0..nvar {
                    if !self.u[i].is_finite() {
                        _notfinite = true;
                        break;
                    }
                    for j in 0..nvar {
                        if !self.imat[(i, j)].is_finite() {
                            _notfinite = true;
                            break;
                        }
                    }
                }
            }
            if !_notfinite && ((self.loglik[1] - newlk).abs() / newlk.abs() <= self.eps) {
                self.loglik[1] = newlk;
                Self::chinv(&mut self.imat)?;
                self.rescale_params();
                if halving > 0 {
                    self.flag = -2;
                }
                return Ok(());
            }
            if _notfinite || newlk < self.loglik[1] {
                halving += 1;
                for (newbeta_elem, beta_elem) in newbeta.iter_mut().zip(self.beta.iter()).take(nvar)
                {
                    *newbeta_elem =
                        (*newbeta_elem + (halving as f64) * beta_elem) / (halving as f64 + 1.0);
                }
            } else {
                halving = 0;
                self.loglik[1] = newlk;
                self.beta.copy_from_slice(&newbeta);
                a.copy_from_slice(&self.u);
                Self::chsolve(&self.imat, &mut a)?;
                for (newbeta_elem, (beta_elem, a_elem)) in newbeta
                    .iter_mut()
                    .zip(self.beta.iter().zip(a.iter()))
                    .take(nvar)
                {
                    *newbeta_elem = beta_elem + a_elem;
                }
            }
        }
        let beta_final = self.beta.clone();
        self.loglik[1] = self.iterate(&beta_final)?;
        Self::chinv(&mut self.imat)?;
        self.rescale_params();
        self.flag = CONVERGENCE_FLAG;
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
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in (i + 1)..n {
                mat[(i, j)] = mat[(j, i)];
            }
        }
        if cholesky_check(mat) {
            Ok(n as i32)
        } else {
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                if mat[(i, i)] < toler {
                    return Ok(i as i32);
                }
            }
            Err(CoxError::CholeskyDecomposition)
        }
    }
    fn chsolve(chol: &Array2<f64>, a: &mut [f64]) -> Result<(), CoxError> {
        let b = Array1::from_iter(a.iter().copied());
        let result = lu_solve(chol, &b).ok_or(CoxError::CholeskyDecomposition)?;
        a.copy_from_slice(result.as_slice().unwrap_or(&[]));
        Ok(())
    }
    fn chinv(mat: &mut Array2<f64>) -> Result<(), CoxError> {
        let n = mat.nrows();
        let mut mat_reg = mat.clone();
        for i in 0..n {
            mat_reg[(i, i)] += RIDGE_REGULARIZATION;
        }
        let inv = matrix_inverse(&mat_reg).ok_or(CoxError::MatrixInversion)?;
        *mat = inv;
        Ok(())
    }
    pub fn results(self) -> CoxFitResults {
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
