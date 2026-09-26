//! The penalised Newton-Raphson kernels of R survival: `src/coxfit5.c`
//! (`coxfit5_a`/`coxfit5_b`/`coxfit5_c`, right-censored data) and
//! `src/agfit5.c` (`agfit5a`/`agfit5b`, (start, stop] data), on the sparse
//! Cholesky routines of [`crate::regression::penalized::cholesky3`].
//!
//! The information matrix of a model with a sparse frailty term has the
//! block structure `[D  B'; B  C]`: `D` (the frailty groups) is diagonal,
//! stored as `fdiag`, and only the dense slice `[B C]` (`nvar` rows,
//! `nfrail + nvar` columns, R's `jmat`) is kept, so an iteration costs
//! `O(n * nvar + ndeaths * nvar * (nfrail + nvar))` in time and
//! `O(nvar * (nfrail + nvar))` in memory, as in the C code, instead of
//! `O((nfrail + nvar)^2)`.
//!
//! The C routines keep their state in static arrays between the `_a`
//! (set-up and initial log likelihood), `_b` (iterate) and `_c` (expected
//! events, right-censored only) calls; [`Kernel`] is that state.  The
//! penalty callbacks (`cox_callback` in `cox_Rcallback.c`) are the
//! [`PenaltyCallback`] trait.

use crate::error::SurvivalResult;
use crate::regression::penalized::cholesky3::{chinv3, cholesky3, chsolve3, finish_factors};
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use ndarray::Array2;

/// `coxsafe` (`src/coxsafe.c`): clamps a linear predictor so that neither
/// the exponential nor the running risk-set sums lose all their digits.
fn coxsafe(x: f64) -> f64 {
    x.clamp(-200.0, 22.0)
}

/// What `coxfit5_b`/`agfit5b` return.
#[derive(Debug, Clone)]
pub(super) struct InnerFit {
    /// The penalised partial likelihood at the returned coefficients.
    pub loglik: f64,
    /// Score vector (frailty entries first).
    pub u: Vec<f64>,
    /// R's `hmat`: the dense slice of the Cholesky factor of the penalised
    /// information (unit diagonal, zeros above).
    pub hmat: Array2<f64>,
    /// R's `hinv`: the dense slice of the inverse Cholesky factor.
    pub hinv: Array2<f64>,
    /// `D^{-1}` of the factorisation, frailty entries first (0 for a
    /// redundant column).
    pub fdiag: Vec<f64>,
    /// Rank of the information (`-rank` if not non-negative definite),
    /// 1000 when the iteration budget ran out.
    pub flag: i32,
    /// Iterations used.
    pub iter: usize,
}

/// Sorted, centred data plus the work arrays of the C kernels.
pub(super) struct Kernel {
    n: usize,
    nvar: usize,
    nf: usize,
    /// `n x nvar` centred dense covariates in the caller's row order.
    covar: Array2<f64>,
    means: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    status: Vec<i32>,
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    /// 0-based frailty group of each row.
    frail: Option<Vec<usize>>,
    /// Rows by (stratum, decreasing stop time, status), R's `sorted[, 1]`.
    sort1: Vec<usize>,
    /// Rows by (stratum, decreasing start time), R's `sorted[, 2]`.
    sort2: Vec<usize>,
    /// Cumulative stratum sizes in sort order (R's `newstrat`).
    strata_end: Vec<usize>,
    /// Right-censored data: number of tied deaths, stored on the last of
    /// them in sort order (0 elsewhere), and their mean weight.
    mark: Vec<usize>,
    wtave: Vec<f64>,
    /// 1 for Efron, 0 for Breslow.
    method: f64,
    /// `exp(eta)` (right-censored) or `eta` ((start, stop]) of the last
    /// evaluation, R's `score`.
    score: Vec<f64>,
    a: Vec<f64>,
    a2: Vec<f64>,
    tmean: Vec<f64>,
    cmat: Array2<f64>,
    cmat2: Array2<f64>,
}

/// The inputs of [`Kernel::new`].
pub(super) struct KernelData<'a> {
    pub stop: &'a [f64],
    pub start: Option<&'a [f64]>,
    pub status: &'a [i32],
    /// Dense covariates (the sparse frailty column removed).
    pub x: &'a Array2<f64>,
    pub weights: &'a [f64],
    pub offset: &'a [f64],
    pub strata: Option<&'a [i32]>,
    /// 0-based frailty group per row.
    pub frail: Option<&'a [usize]>,
    pub nfrail: usize,
    pub efron: bool,
    /// Per dense column: centre it (R's `docenter`).
    pub docenter: &'a [bool],
}

impl Kernel {
    /// `coxfit5_a`/`agfit5a` without the log likelihood: sort, mark the
    /// tied deaths and centre the covariates.
    pub(super) fn new(data: KernelData<'_>) -> Self {
        let n = data.stop.len();
        let nvar = data.x.ncols();
        let nf = data.nfrail;
        let nvar2 = nvar + nf;
        let stratum = |i: usize| data.strata.map_or(0, |s| s[i]);

        // order(strata, -stop, status) and order(strata, -start).
        let mut sort1: Vec<usize> = (0..n).collect();
        sort1.sort_by(|&l, &r| {
            stratum(l)
                .cmp(&stratum(r))
                .then_with(|| data.stop[r].total_cmp(&data.stop[l]))
                .then_with(|| data.status[l].cmp(&data.status[r]))
        });
        let sort2 = match data.start {
            Some(start) => {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&l, &r| {
                    stratum(l)
                        .cmp(&stratum(r))
                        .then_with(|| start[r].total_cmp(&start[l]))
                });
                order
            }
            None => Vec::new(),
        };
        let mut strata_end = Vec::new();
        for position in 0..n {
            if position + 1 == n || stratum(sort1[position]) != stratum(sort1[position + 1]) {
                strata_end.push(position + 1);
            }
        }

        // mark: the number of tied deaths on the last of them; wtave: their
        // mean weight.  Within a stratum the sort puts the censored rows of
        // a time before its deaths.
        let mut mark = vec![0usize; n];
        let mut wtave = vec![0.0; n];
        if data.start.is_none() {
            let mut istrat = 0;
            let mut i = 0;
            while i < n {
                let p = sort1[i];
                if data.status[p] == 1 {
                    let mut ndead = 0;
                    let mut total = 0.0;
                    let mut j = i;
                    while j < n {
                        let k = sort1[j];
                        if data.stop[k] != data.stop[p] || j == strata_end[istrat] {
                            break;
                        }
                        ndead += 1;
                        total += data.weights[k];
                        j += 1;
                    }
                    let k = sort1[j - 1];
                    mark[k] = ndead;
                    wtave[k] = total / ndead as f64;
                    i = j;
                } else {
                    i += 1;
                }
                if i == strata_end[istrat] {
                    istrat += 1;
                }
            }
        }

        let mut covar = data.x.clone();
        let mut means = vec![0.0; nvar];
        for (i, mean) in means.iter_mut().enumerate() {
            if !data.docenter[i] {
                continue;
            }
            *mean = covar.column(i).sum() / n as f64;
            let center = *mean;
            covar.column_mut(i).mapv_inplace(|value| value - center);
        }

        Self {
            n,
            nvar,
            nf,
            covar,
            means,
            weights: data.weights.to_vec(),
            offset: data.offset.to_vec(),
            status: data.status.to_vec(),
            stop: data.stop.to_vec(),
            start: data.start.map(<[f64]>::to_vec),
            frail: data.frail.map(<[usize]>::to_vec),
            sort1,
            sort2,
            strata_end,
            mark,
            wtave,
            method: f64::from(u8::from(data.efron)),
            score: vec![0.0; n],
            a: vec![0.0; nvar2],
            a2: vec![0.0; nvar2],
            tmean: vec![0.0; nvar2],
            cmat: Array2::zeros((nvar, nvar2)),
            cmat2: Array2::zeros((nvar, nvar2)),
        }
    }

    pub(super) fn means(&self) -> &[f64] {
        &self.means
    }

    fn nvar2(&self) -> usize {
        self.nvar + self.nf
    }

    /// `offset + fbeta[group] + beta' x`, clamped.
    fn eta(&self, p: usize, beta: &[f64], fbeta: &[f64]) -> f64 {
        let mut zbeta = self.offset[p];
        if let Some(frail) = &self.frail {
            zbeta += fbeta[frail[p]];
        }
        for (i, b) in beta.iter().enumerate() {
            zbeta += b * self.covar[(p, i)];
        }
        coxsafe(zbeta)
    }

    /// The partial likelihood at `beta` with no frailty (`coxfit5_a`,
    /// `agfit5a`); the dense penalty is added by the caller.
    pub(super) fn initial_loglik(&self, beta: &[f64]) -> f64 {
        let no_frailty = vec![0.0; self.nf];
        if self.start.is_some() {
            self.initial_loglik_counting(beta, &no_frailty)
        } else {
            self.initial_loglik_right(beta, &no_frailty)
        }
    }

    fn initial_loglik_right(&self, beta: &[f64], fbeta: &[f64]) -> f64 {
        let mut loglik = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        let mut istrat = 0;
        for ii in 0..self.n {
            if ii == self.strata_end[istrat] {
                denom = 0.0;
                istrat += 1;
            }
            let p = self.sort1[ii];
            let zbeta = self.eta(p, beta, fbeta);
            let risk = zbeta.exp() * self.weights[p];
            denom += risk;
            if self.status[p] == 1 {
                efron_wt += risk;
                loglik += self.weights[p] * zbeta;
            }
            if self.mark[p] > 0 {
                let ndead = self.mark[p] as f64;
                for k in 0..self.mark[p] {
                    let temp = k as f64 * self.method / ndead;
                    loglik -= self.wtave[p] * (denom - temp * efron_wt).ln();
                }
                efron_wt = 0.0;
            }
        }
        loglik
    }

    fn initial_loglik_counting(&self, beta: &[f64], fbeta: &[f64]) -> f64 {
        let start = self.start.as_deref().expect("counting-process data");
        let score: Vec<f64> = (0..self.n).map(|p| self.eta(p, beta, fbeta)).collect();
        let mut loglik = 0.0;
        let mut istrat = 0;
        let mut indx2 = 0;
        let mut denom = 0.0;
        let mut person = 0;
        while person < self.n {
            let p = self.sort1[person];
            if self.status[p] == 0 {
                denom += score[p].exp() * self.weights[p];
                person += 1;
            } else {
                let time = self.stop[p];
                // Subtract the subjects whose start time is to the right.
                while indx2 < self.strata_end[istrat] {
                    let q = self.sort2[indx2];
                    if start[q] < time {
                        break;
                    }
                    denom -= score[q].exp() * self.weights[q];
                    indx2 += 1;
                }
                let mut efron_wt = 0.0;
                let mut meanwt = 0.0;
                let mut deaths = 0.0;
                let mut k = person;
                while k < self.strata_end[istrat] {
                    let q = self.sort1[k];
                    if self.stop[q] < time {
                        break;
                    }
                    let risk = score[q].exp() * self.weights[q];
                    denom += risk;
                    if self.status[q] == 1 {
                        deaths += 1.0;
                        efron_wt += risk;
                        meanwt += self.weights[q];
                    }
                    k += 1;
                }
                meanwt /= deaths;
                let mut itemp = -1.0;
                while person < k {
                    let q = self.sort1[person];
                    if self.status[q] == 1 {
                        itemp += 1.0;
                        let temp = itemp * self.method / deaths;
                        loglik +=
                            self.weights[q] * score[q] - meanwt * (denom - temp * efron_wt).ln();
                    }
                    person += 1;
                }
            }
            if person == self.strata_end[istrat] {
                istrat += 1;
                denom = 0.0;
                indx2 = person;
            }
        }
        loglik
    }

    /// Adds one row's contributions to the risk-set sums (`denom`, `a`,
    /// `cmat`) with the sign `sign`.
    fn accumulate(&mut self, p: usize, risk: f64, sign: f64, second: bool) {
        let nf = self.nf;
        let fgrp = self.frail.as_ref().map(|f| f[p]);
        let target = if second { &mut self.a2 } else { &mut self.a };
        if let Some(g) = fgrp {
            target[g] += sign * risk;
        }
        for i in 0..self.nvar {
            let xi = self.covar[(p, i)];
            target[i + nf] += sign * risk * xi;
        }
        let cmat = if second {
            &mut self.cmat2
        } else {
            &mut self.cmat
        };
        for i in 0..self.nvar {
            let xi = self.covar[(p, i)];
            if let Some(g) = fgrp {
                cmat[(i, g)] += sign * risk * xi;
            }
            for j in 0..=i {
                cmat[(i, j + nf)] += sign * risk * xi * self.covar[(p, j)];
            }
        }
    }

    /// The death-time update of the score and information: one Efron step
    /// `k` of `ndead` for the mean weight `meanwt` and the tied-death sums.
    #[allow(clippy::too_many_arguments)]
    fn death_step(
        &mut self,
        denom: f64,
        efron_wt: f64,
        k: usize,
        ndead: f64,
        meanwt: f64,
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> f64 {
        let nf = self.nf;
        let temp = k as f64 * self.method / ndead;
        let d2 = denom - temp * efron_wt;
        for i in 0..self.nvar2() {
            let temp2 = (self.a[i] - temp * self.a2[i]) / d2;
            self.tmean[i] = temp2;
            u[i] -= meanwt * temp2;
            if i < nf {
                // The C code adds temp2 * (1 - temp2) without the weight,
                // which breaks weighted sparse fits (R returns NA
                // coefficients); the weighted form is the diagonal of the
                // dense update below for an indicator column.
                fdiag[i] += meanwt * temp2 * (1.0 - temp2);
            } else {
                let ii = i - nf;
                for j in 0..=i {
                    jmat[(ii, j)] += meanwt
                        * ((self.cmat[(ii, j)] - temp * self.cmat2[(ii, j)]) / d2
                            - temp2 * self.tmean[j]);
                }
            }
        }
        -meanwt * d2.ln()
    }

    fn reset_sums(&mut self, second: bool) {
        if second {
            self.a2.fill(0.0);
            self.cmat2.fill(0.0);
        } else {
            self.a.fill(0.0);
            self.cmat.fill(0.0);
        }
    }

    /// One evaluation of `coxfit5_b`: the penalised-model score `u`, the
    /// dense information slice `jmat`, the frailty diagonal `fdiag` and the
    /// partial likelihood at (`beta`, `fbeta`).
    fn evaluate_right(
        &mut self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> f64 {
        let nf = self.nf;
        let mut newlk = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        let mut istrat = 0;
        for ip in 0..self.n {
            if ip == 0 || ip == self.strata_end[istrat] {
                efron_wt = 0.0;
                denom = 0.0;
                self.reset_sums(false);
                self.reset_sums(true);
            }
            if ip == self.strata_end[istrat] {
                istrat += 1;
            }
            let p = self.sort1[ip];
            let zbeta = self.eta(p, beta, fbeta);
            self.score[p] = zbeta.exp();
            let risk = self.score[p] * self.weights[p];
            denom += risk;
            self.accumulate(p, risk, 1.0, false);
            if self.status[p] == 1 {
                efron_wt += risk;
                newlk += self.weights[p] * zbeta;
                if let Some(frail) = &self.frail {
                    u[frail[p]] += self.weights[p];
                }
                for i in 0..self.nvar {
                    u[i + nf] += self.weights[p] * self.covar[(p, i)];
                }
                self.accumulate(p, risk, 1.0, true);
            }
            if self.mark[p] > 0 {
                let ndead = self.mark[p] as f64;
                let wtave = self.wtave[p];
                for k in 0..self.mark[p] {
                    newlk += self.death_step(denom, efron_wt, k, ndead, wtave, u, jmat, fdiag);
                }
                efron_wt = 0.0;
                self.reset_sums(true);
            }
        }
        newlk
    }

    /// One evaluation of `agfit5b` for (start, stop] data.
    fn evaluate_counting(
        &mut self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> f64 {
        let nf = self.nf;
        for p in 0..self.n {
            self.score[p] = self.eta(p, beta, fbeta);
        }
        let mut newlk = 0.0;
        let mut istrat = 0;
        let mut indx2 = 0;
        let mut denom = 0.0;
        self.reset_sums(false);
        let mut person = 0;
        while person < self.n {
            let p = self.sort1[person];
            if self.status[p] == 0 {
                let risk = self.score[p].exp() * self.weights[p];
                denom += risk;
                self.accumulate(p, risk, 1.0, false);
                person += 1;
            } else {
                let time = self.stop[p];
                // Subtract the subjects whose start time is to the right.
                while indx2 < self.strata_end[istrat] {
                    let q = self.sort2[indx2];
                    if self.start.as_ref().expect("counting-process data")[q] < time {
                        break;
                    }
                    let risk = self.score[q].exp() * self.weights[q];
                    denom -= risk;
                    self.accumulate(q, risk, -1.0, false);
                    indx2 += 1;
                }
                // The sums over this death time (a2 and cmat2).
                let mut efron_wt = 0.0;
                let mut meanwt = 0.0;
                let mut deaths = 0.0;
                self.reset_sums(true);
                let mut k = person;
                while k < self.strata_end[istrat] {
                    let q = self.sort1[k];
                    if self.stop[q] < time {
                        break;
                    }
                    let risk = self.score[q].exp() * self.weights[q];
                    denom += risk;
                    self.accumulate(q, risk, 1.0, false);
                    if self.status[q] == 1 {
                        deaths += 1.0;
                        // agfit5b.c adds risk * weights[p] here, weighting
                        // the Efron sum twice (agfit5a, agfit4 and coxfit5
                        // add the weighted risk once); the bug is not
                        // copied, so weighted Efron fits agree with agreg.fit.
                        efron_wt += risk;
                        meanwt += self.weights[q];
                        if let Some(frail) = &self.frail {
                            u[frail[q]] += self.weights[q];
                        }
                        for i in 0..self.nvar {
                            u[i + nf] += self.weights[q] * self.covar[(q, i)];
                        }
                        self.accumulate(q, risk, 1.0, true);
                    }
                    k += 1;
                }
                meanwt /= deaths;
                let mut itemp = 0;
                while person < k {
                    let q = self.sort1[person];
                    if self.status[q] == 1 {
                        newlk += self.weights[q] * self.score[q];
                        newlk +=
                            self.death_step(denom, efron_wt, itemp, deaths, meanwt, u, jmat, fdiag);
                        itemp += 1;
                    }
                    person += 1;
                }
            }
            if person == self.strata_end[istrat] {
                istrat += 1;
                denom = 0.0;
                indx2 = person;
                self.reset_sums(false);
            }
        }
        newlk
    }

    fn evaluate(
        &mut self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> f64 {
        u.fill(0.0);
        jmat.fill(0.0);
        fdiag[..self.nf].fill(0.0);
        if self.start.is_some() {
            self.evaluate_counting(beta, fbeta, u, jmat, fdiag)
        } else {
            self.evaluate_right(beta, fbeta, u, jmat, fdiag)
        }
    }

    /// `coxfit5_b`/`agfit5b`: iterate to convergence from (`beta`, `fbeta`),
    /// which hold the solution on return.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn newton(
        &mut self,
        maxiter: usize,
        beta: &mut [f64],
        fbeta: &mut [f64],
        eps: f64,
        toler_chol: f64,
        shape: PenaltyShape,
        callback: &mut impl PenaltyCallback,
    ) -> SurvivalResult<InnerFit> {
        let nvar = self.nvar;
        let nf = self.nf;
        let nvar2 = nvar + nf;
        let mut u = vec![0.0; nvar2];
        let mut jmat = Array2::zeros((nvar, nvar2));
        let mut fdiag = vec![0.0; nvar2];
        let mut oldbeta = [fbeta.to_vec(), beta.to_vec()].concat();
        let mut loglik = 0.0;
        let mut newlk;
        let mut halving = false;
        let mut iter = 0;
        let mut flag;
        loop {
            newlk = self.evaluate(beta, fbeta, &mut u, &mut jmat, &mut fdiag);

            if shape.sparse {
                let terms = callback.call(1, fbeta)?;
                if terms.flag[0] {
                    // Force the frailties to zero.
                    for i in 0..nf {
                        u[i] = 0.0;
                        fdiag[i] = 1.0;
                        jmat.column_mut(i).fill(0.0);
                    }
                } else {
                    for i in 0..nf {
                        u[i] += terms.first[i];
                        fdiag[i] += terms.second[i];
                    }
                    newlk += terms.penalty;
                }
            }
            if shape.dense {
                let terms = callback.call(2, beta)?;
                newlk += terms.penalty;
                if shape.full_imat {
                    // `second` is R's c(matrix), column-major.
                    let mut k = 0;
                    for i in 0..nvar {
                        u[i + nf] += terms.first[i];
                        for j in nf..nvar2 {
                            jmat[(i, j)] += terms.second[k];
                            k += 1;
                        }
                    }
                } else {
                    for i in 0..nvar {
                        u[i + nf] += terms.first[i];
                        jmat[(i, i + nf)] += terms.second[i];
                    }
                }
                for i in 0..nvar {
                    if terms.flag[i] {
                        u[i + nf] = 0.0;
                        for j in 0..i {
                            jmat[(i, j + nf)] = 0.0;
                        }
                        // coxfit5.c writes jmat[i+nf][i], a slip that is
                        // only in bounds without a sparse term; the unit
                        // diagonal is what it means.
                        jmat[(i, i + nf)] = 1.0;
                    }
                }
            }

            flag = cholesky3(&mut jmat, nf, &mut fdiag, toler_chol);
            if newlk.abs() < eps || ((1.0 - loglik / newlk).abs() <= eps && !halving) {
                loglik = newlk;
                break;
            }
            if iter == maxiter {
                flag = 1000;
                loglik = newlk;
                break;
            }
            if iter > 0 && newlk < loglik {
                halving = true;
                for i in 0..nvar {
                    beta[i] = (oldbeta[i + nf] + beta[i]) / 2.0;
                }
                for i in 0..nf {
                    fbeta[i] = (oldbeta[i] + fbeta[i]) / 2.0;
                }
            } else {
                halving = false;
                loglik = newlk;
                chsolve3(&jmat, nf, &fdiag, &mut u);
                oldbeta[nf..].copy_from_slice(beta);
                oldbeta[..nf].copy_from_slice(fbeta);
                for (b, step) in beta.iter_mut().zip(&u[nf..]) {
                    *b += step;
                }
                for (b, step) in fbeta.iter_mut().zip(&u[..nf]) {
                    *b += step;
                }
            }
            iter += 1;
        }

        let mut hmat = jmat.clone();
        chinv3(&mut jmat, nf, &mut fdiag);
        finish_factors(&mut hmat, &mut jmat, nf, &mut fdiag);
        Ok(InnerFit {
            loglik,
            u,
            hmat,
            hinv: jmat,
            fdiag,
            flag,
            iter,
        })
    }

    /// `coxfit5_c`: the expected number of events of each subject at the
    /// scores of the last evaluation (right-censored data).  The C code
    /// never resets its cumulative hazard between strata (it compares the
    /// position with the end of the last stratum), so R's martingale
    /// residuals of a stratified penalised fit carry the hazard of one
    /// stratum into the next; this port resets it, as `coxmart.c` does.
    pub(super) fn expected_events(&self) -> Vec<f64> {
        let n = self.n;
        let mut expect = vec![0.0; n];
        let mut hazard_jump = vec![0.0; n];
        let mut istrat = 0;
        let mut denom = 0.0;
        for ip in 0..n {
            let p = self.sort1[ip];
            if ip == self.strata_end[istrat] {
                denom = 0.0;
                istrat += 1;
            }
            denom += self.score[p] * self.weights[p];
            if self.mark[p] > 0 {
                let ndead = self.mark[p];
                let mut wtsum = 0.0;
                let mut efron_wt = 0.0;
                for j in 0..ndead {
                    let i = self.sort1[ip - j];
                    efron_wt += self.score[i] * self.weights[i];
                    wtsum += self.weights[i];
                }
                if ndead < 2 || self.method == 0.0 {
                    expect[p] = wtsum / denom;
                    hazard_jump[p] = wtsum / denom;
                } else {
                    let mut hazard = 0.0;
                    let mut hazard2 = 0.0;
                    let wtsum = wtsum / ndead as f64;
                    for j in 0..ndead {
                        let temp = j as f64 / ndead as f64;
                        hazard += wtsum / (denom - efron_wt * temp);
                        hazard2 += wtsum * (1.0 - temp) / (denom - efron_wt * temp);
                    }
                    expect[p] = hazard;
                    hazard_jump[p] = hazard2;
                }
            }
        }
        // The cumulative hazard, walking back up the sort order.
        let mut hazard = 0.0;
        let mut istrat = self.strata_end.len() - 1;
        let mut ip = n as isize - 1;
        while ip >= 0 {
            let p = self.sort1[ip as usize];
            if self.status[p] > 0 {
                let ndead = self.mark[p];
                let jump = expect[p];
                let hazard2 = hazard_jump[p];
                for j in 0..ndead {
                    let i = self.sort1[ip as usize - j];
                    expect[i] = self.score[i] * (hazard + hazard2);
                }
                ip -= ndead as isize;
                hazard += jump;
            } else {
                expect[p] = hazard * self.score[p];
                ip -= 1;
            }
            if istrat > 0 && self.strata_end[istrat - 1] as isize == ip + 1 {
                hazard = 0.0;
                istrat -= 1;
            }
        }
        expect
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coxsafe_clamps_the_linear_predictor() {
        assert_eq!(coxsafe(30.0), 22.0);
        assert_eq!(coxsafe(-300.0), -200.0);
        assert_eq!(coxsafe(1.5), 1.5);
    }
}
