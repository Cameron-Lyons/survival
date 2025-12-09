#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_memcpy)]
use crate::core::coxsafe::coxsafe;

pub struct CoxParams {
    pub beta: Vec<f64>,
    pub fbeta: Vec<f64>,
    pub fdiag: Vec<f64>,
    pub nfrail: usize,
    pub method: i32,
    pub ptype: i32,
    pub pdiag: i32,
    pub strata: Vec<i32>,
    pub maxiter: usize,
    pub eps: f64,
}

pub struct CoxData {
    pub nused: usize,
    pub nvar: usize,
    pub y: Vec<f64>,
    pub covar2: Vec<f64>,
    pub weights2: Vec<f64>,
    pub offset2: Vec<f64>,
    pub sorted: Vec<i32>,
    pub strata: Vec<i32>,
    pub docenter: Vec<i32>,
    pub fmat: Vec<Vec<f64>>,
}

pub struct CoxResult {
    pub means: Vec<f64>,
    pub beta: Vec<f64>,
    pub u: Vec<f64>,
    pub imat: Vec<Vec<f64>>,
    pub loglik: f64,
    pub flag: i32,
    pub maxiter: usize,
    pub fbeta: Vec<f64>,
    pub fdiag: Vec<f64>,
    pub jmat: Vec<Vec<f64>>,
    pub expect: Vec<f64>,
}

pub struct CoxFit5 {
    covar: Vec<Vec<f64>>,
    cmat: Vec<Vec<f64>>,
    cmat2: Vec<Vec<f64>>,
    mark: Vec<f64>,
    wtave: Vec<f64>,
    a: Vec<f64>,
    oldbeta: Vec<f64>,
    a2: Vec<f64>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    status: Vec<i32>,
    sort: Vec<i32>,
    ttime: Vec<f64>,
    tmean: Vec<f64>,
    ptype: i32,
    pdiag: i32,
    ipen: Vec<f64>,
    upen: Vec<f64>,
    logpen: f64,
    zflag: Vec<i32>,
    frail: Option<Vec<i32>>,
    score: Vec<f64>,
}

impl CoxFit5 {
    pub fn new() -> Self {
        CoxFit5 {
            covar: Vec::new(),
            cmat: Vec::new(),
            cmat2: Vec::new(),
            mark: Vec::new(),
            wtave: Vec::new(),
            a: Vec::new(),
            oldbeta: Vec::new(),
            a2: Vec::new(),
            offset: Vec::new(),
            weights: Vec::new(),
            status: Vec::new(),
            sort: Vec::new(),
            ttime: Vec::new(),
            tmean: Vec::new(),
            ptype: 0,
            pdiag: 0,
            ipen: Vec::new(),
            upen: Vec::new(),
            logpen: 0.0,
            zflag: Vec::new(),
            frail: None,
            score: Vec::new(),
        }
    }

    pub fn coxfit5_a(&mut self, params: &CoxParams, data: &CoxData) -> CoxResult {
        let nused = data.nused;
        let nvar = data.nvar;
        let nf = params.nfrail;
        let nvar2 = nvar + nf;
        let method = params.method as f64;
        self.ptype = params.ptype;
        self.pdiag = params.pdiag;

        if nvar > 0 {
            self.covar = vec![vec![0.0; nused]; nvar];
            self.cmat = vec![vec![0.0; nvar + 1]; nvar2];
            self.cmat2 = vec![vec![0.0; nvar + 1]; nvar2];
        }

        self.a = vec![0.0; 4 * nvar2 + 6 * nused];
        let _oldbeta = vec![0.0; nvar2];
        let _a2 = vec![0.0; nvar2];
        self.mark = vec![0.0; nused];
        self.wtave = vec![0.0; nused];
        self.weights = data.weights2.clone();
        self.offset = data.offset2.clone();
        self.status = data.y.chunks_exact(2).map(|c| c[1] as i32).collect();
        self.sort = data.sorted.clone();
        self.ttime = data.y.chunks_exact(2).map(|c| c[0]).collect();

        let mut istrat = 0;
        for i in 0..nused {
            let p = self.sort[i] as usize;
            if self.status[p] == 1 {
                let mut temp = 0.0;
                let mut ndead = 0.0;
                let mut j = i;
                while j < nused
                    && (j < params.strata[istrat] as usize
                        || self.ttime[self.sort[j] as usize] == self.ttime[p])
                {
                    let k = self.sort[j] as usize;
                    ndead += self.status[k] as f64;
                    temp += self.weights[k];
                    j += 1;
                }
                let k = self.sort[j - 1] as usize;
                self.mark[k] = ndead;
                self.wtave[k] = temp / ndead.max(1.0);
                istrat += (j >= params.strata[istrat] as usize) as usize;
            }
        }

        let mut means = vec![0.0; nvar];
        for i in 0..nvar {
            if data.docenter[i] != 0 {
                means[i] =
                    data.covar2[i * nused..(i + 1) * nused].iter().sum::<f64>() / nused as f64;
                for p in 0..nused {
                    self.covar[i][p] = data.covar2[i * nused + p] - means[i];
                }
            }
        }

        let mut loglik = 0.0;
        let u = vec![0.0; nvar];
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        istrat = 0;

        for ii in 0..nused {
            if ii == params.strata[istrat] as usize {
                denom = 0.0;
                istrat += 1;
            }

            let p = self.sort[ii] as usize;
            let mut zbeta = self.offset[p];
            for i in 0..nvar {
                zbeta += params.beta[i] * self.covar[i][p];
            }
            zbeta = coxsafe(zbeta);
            let risk = zbeta.exp() * self.weights[p];
            denom += risk;

            if self.status[p] == 1 {
                efron_wt += risk;
                loglik += self.weights[p] * zbeta;
            }

            if self.mark[p] > 0.0 {
                let ndead = self.mark[p];
                for k in 0..ndead as usize {
                    let temp = k as f64 * method / ndead;
                    let d2 = denom - temp * efron_wt;
                    loglik -= self.wtave[p] * d2.ln();
                }
                efron_wt = 0.0;
            }
        }

        CoxResult {
            means,
            beta: params.beta.clone(),
            u,
            imat: vec![vec![0.0; nvar]; nvar],
            loglik,
            flag: 0,
            maxiter: 0,
            fbeta: vec![0.0; nf],
            fdiag: vec![0.0; nvar2],
            jmat: vec![vec![0.0; nvar2]; nvar2],
            expect: vec![0.0; nused],
        }
    }
    pub fn coxfit5_b(&mut self, params: &mut CoxParams, data: &CoxData) -> CoxResult {
        let nvar = data.nvar;
        let nf = params.nfrail;
        let nvar2 = nvar + nf;
        let mut result = CoxResult::new(nvar, nf, nvar2, data.nused);
        let mut oldbeta = vec![0.0; nvar2];
        let mut newbeta = vec![0.0; nvar2];
        let mut u = vec![0.0; nvar2];
        let mut imat = vec![vec![0.0; nvar2]; nvar2];
        let mut cholesky_mat = vec![vec![0.0; nvar2]; nvar2];
        let mut work = vec![0.0; nvar2];
        let mut loglik = 0.0;

        newbeta[..nvar].copy_from_slice(&params.beta[..nvar]);
        newbeta[nvar..(nf + nvar)].copy_from_slice(&params.fbeta[..nf]);

        for iter in 0..params.maxiter {
            loglik = 0.0;

            u.fill(0.0);
            for row in &mut imat {
                row.fill(0.0);
            }

            let mut istrat = 0;
            let mut denom = 0.0;
            let mut efron_wt = 0.0;
            let mut risk_sum = vec![0.0; nvar2];
            let mut risk_sum2 = vec![vec![0.0; nvar2]; nvar2];

            for ii in 0..data.nused {
                if ii == data.strata[istrat] as usize {
                    denom = 0.0;
                    efron_wt = 0.0;
                    risk_sum.fill(0.0);
                    for row in &mut risk_sum2 {
                        row.fill(0.0);
                    }
                    istrat += 1;
                }

                let p = self.sort[ii] as usize;
                let mut zbeta = self.offset[p];

                for i in 0..nvar {
                    zbeta += newbeta[i] * self.covar[i][p];
                }
                for i in 0..nf {
                    zbeta += newbeta[nvar + i] * data.fmat[i][p];
                }
                zbeta = coxsafe(zbeta);
                let risk = zbeta.exp() * self.weights[p];
                denom += risk;

                for i in 0..nvar {
                    risk_sum[i] += risk * self.covar[i][p];
                }
                for i in 0..nf {
                    risk_sum[nvar + i] += risk * data.fmat[i][p];
                }

                for i in 0..nvar2 {
                    for j in 0..nvar2 {
                        let x_i = if i < nvar {
                            self.covar[i][p]
                        } else {
                            data.fmat[i - nvar][p]
                        };
                        let x_j = if j < nvar {
                            self.covar[j][p]
                        } else {
                            data.fmat[j - nvar][p]
                        };
                        risk_sum2[i][j] += risk * x_i * x_j;
                    }
                }

                if self.status[p] == 1 {
                    efron_wt += risk;
                    loglik += self.weights[p] * zbeta;
                }

                if self.mark[p] > 0.0 {
                    let ndead = self.mark[p] as usize;
                    for k in 0..ndead {
                        let temp = k as f64 * params.method as f64 / ndead as f64;
                        let d2 = denom - temp * efron_wt;
                        loglik -= self.wtave[p] * d2.ln();

                        let wt = self.wtave[p] / d2;
                        for i in 0..nvar2 {
                            let x_i = if i < nvar {
                                self.covar[i][p]
                            } else {
                                data.fmat[i - nvar][p]
                            };
                            u[i] += wt * (risk_sum[i] - temp * efron_wt * x_i);
                        }

                        for i in 0..nvar2 {
                            for j in 0..nvar2 {
                                let x_i = if i < nvar {
                                    self.covar[i][p]
                                } else {
                                    data.fmat[i - nvar][p]
                                };
                                let x_j = if j < nvar {
                                    self.covar[j][p]
                                } else {
                                    data.fmat[j - nvar][p]
                                };
                                imat[i][j] += wt * (risk_sum2[i][j] - temp * efron_wt * x_i * x_j);
                            }
                        }
                    }
                    efron_wt = 0.0;
                }
            }

            if nf > 0 {
                for i in 0..nf {
                    u[nvar + i] -= newbeta[nvar + i] / params.fdiag[i];
                    imat[nvar + i][nvar + i] += 1.0 / params.fdiag[i];
                    loglik -= 0.5 * newbeta[nvar + i] * newbeta[nvar + i] / params.fdiag[i];
                }
            }

            oldbeta.copy_from_slice(&newbeta);

            for i in 0..nvar2 {
                for j in 0..nvar2 {
                    cholesky_mat[i][j] = imat[i][j];
                }
            }
            if cholesky(&mut cholesky_mat, 1e-10) != 0 {
                result.flag = 1000;
                return result;
            }

            work.copy_from_slice(&u);
            cholesky_solve(&cholesky_mat, &mut work);

            for i in 0..nvar2 {
                newbeta[i] += work[i];
            }

            let mut max_change = 0.0;
            for i in 0..nvar2 {
                let change = (newbeta[i] - oldbeta[i]).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            if max_change < params.eps {
                result.flag = 0;
                result.maxiter = iter + 1;
                result.loglik = loglik;
                result.beta = newbeta[..nvar].to_vec();
                result.fbeta = newbeta[nvar..].to_vec();
                result.u = u[..nvar].to_vec();
                result.imat = imat[..nvar]
                    .iter()
                    .map(|row| row[..nvar].to_vec())
                    .collect();
                result.fdiag = params.fdiag.clone();
                return result;
            }
        }

        result.flag = 1000;
        result.maxiter = params.maxiter;
        result.loglik = loglik;
        result.beta = newbeta[..nvar].to_vec();
        result.fbeta = newbeta[nvar..].to_vec();
        result.u = u[..nvar].to_vec();
        result.imat = imat[..nvar]
            .iter()
            .map(|row| row[..nvar].to_vec())
            .collect();
        result.fdiag = params.fdiag.clone();
        result.flag = 1000;
        result
    }

    pub fn coxfit5_c(self, data: &CoxData) -> Vec<f64> {
        let mut expect = vec![0.0; data.nused];
        let mut hazard = 0.0;
        let mut istrat = 0;

        for ip in (0..data.nused).rev() {
            let p = self.sort[ip] as usize;
            if self.status[p] == 1 {
                let ndead = self.mark[p] as usize;
                let temp = self.wtave[p];
                for j in 0..ndead {
                    let i = self.sort[ip - j] as usize;
                    expect[i] = self.score[i] * (hazard + temp);
                }
                hazard += temp;
            } else {
                expect[p] = self.score[p] * hazard;
            }

            if ip == self.sort[istrat] as usize {
                hazard = 0.0;
                istrat += 1;
            }
        }

        expect
    }
}

impl CoxResult {
    fn new(nvar: usize, nf: usize, nvar2: usize, nused: usize) -> Self {
        CoxResult {
            means: vec![0.0; nvar],
            beta: vec![0.0; nvar],
            u: vec![0.0; nvar],
            imat: vec![vec![0.0; nvar]; nvar],
            loglik: 0.0,
            flag: 0,
            maxiter: 0,
            fbeta: vec![0.0; nf],
            fdiag: vec![0.0; nvar2],
            jmat: vec![vec![0.0; nvar2]; nvar2],
            expect: vec![0.0; nused],
        }
    }
}

fn cholesky(mat: &mut [Vec<f64>], tolerch: f64) -> i32 {
    let n = mat.len();
    for i in 0..n {
        for j in i..n {
            let mut sum = mat[i][j];
            for k in 0..i {
                sum -= mat[i][k] * mat[j][k];
            }
            if i == j {
                if sum <= tolerch {
                    return (i + 1) as i32;
                }
                mat[i][i] = sum.sqrt();
            } else {
                mat[j][i] = sum / mat[i][i];
            }
        }
    }
    0
}

fn cholesky_solve(mat: &[Vec<f64>], b: &mut [f64]) {
    let n = mat.len();
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= mat[i][j] * b[j];
        }
        b[i] = sum / mat[i][i];
    }
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= mat[j][i] * b[j];
        }
        b[i] = sum / mat[i][i];
    }
}
