use crate::error::SurvivalResult;
use crate::regression::penalized::cholesky3::{chinv3, cholesky3, chsolve3, finish_factors};
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use ndarray::Array2;

fn coxsafe(x: f64) -> f64 {
    x.clamp(-200.0, 22.0)
}

#[derive(Debug, Clone, Copy, Default)]
struct RunningSum {
    hi: f64,
    lo: f64,
}

impl RunningSum {
    fn add(&mut self, value: f64) {
        let sum = self.hi + value;
        let part = sum - self.hi;
        self.lo += (self.hi - (sum - part)) + (value - part);
        self.hi = sum;
    }

    fn since(self, earlier: Self) -> f64 {
        (self.hi - earlier.hi) + (self.lo - earlier.lo)
    }
}

#[derive(Debug, Clone)]
pub(super) struct InnerFit {
    pub loglik: f64,
    pub u: Vec<f64>,
    pub hmat: Array2<f64>,
    pub hinv: Array2<f64>,
    pub fdiag: Vec<f64>,
    pub flag: i32,
    pub iter: usize,
}

pub(super) struct Kernel {
    n: usize,
    nvar: usize,
    nf: usize,
    covar: Array2<f64>,
    means: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    status: Vec<i32>,
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    frail: Option<Vec<usize>>,
    sort1: Vec<usize>,
    sort2: Vec<usize>,
    strata_end: Vec<usize>,
    mark: Vec<usize>,
    wtave: Vec<f64>,
    method: f64,
    score: Vec<f64>,
    a: Vec<f64>,
    a2: Vec<f64>,
    tmean: Vec<f64>,
    cmat: Array2<f64>,
    cmat2: Array2<f64>,
    running: Vec<RunningSum>,
    flushed: Vec<RunningSum>,
    tied: Vec<usize>,
    tied_mark: Vec<bool>,
}

pub(super) struct KernelData<'a> {
    pub stop: &'a [f64],
    pub start: Option<&'a [f64]>,
    pub status: &'a [i32],
    pub x: &'a Array2<f64>,
    pub weights: &'a [f64],
    pub offset: &'a [f64],
    pub strata: Option<&'a [i32]>,
    pub frail: Option<&'a [usize]>,
    pub nfrail: usize,
    pub efron: bool,
    pub docenter: &'a [bool],
}

impl Kernel {
    pub(super) fn new(data: KernelData<'_>) -> Self {
        let n = data.stop.len();
        let nvar = data.x.ncols();
        let nf = data.nfrail;
        let nvar2 = nvar + nf;
        let stratum = |i: usize| data.strata.map_or(0, |s| s[i]);

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
            running: vec![RunningSum::default(); nvar + 2],
            flushed: vec![RunningSum::default(); nf * (nvar + 2)],
            tied: Vec::new(),
            tied_mark: vec![false; nf],
        }
    }

    pub(super) fn means(&self) -> &[f64] {
        &self.means
    }

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

    fn accumulate(&mut self, p: usize, risk: f64, sign: f64, second: bool) {
        let nf = self.nf;
        let fgrp = self.frail.as_ref().map(|f| f[p]);
        if second
            && let Some(g) = fgrp
            && !self.tied_mark[g]
        {
            self.tied_mark[g] = true;
            self.tied.push(g);
        }
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

    fn flush_group(&mut self, g: usize, u: &mut [f64], jmat: &mut Array2<f64>, fdiag: &mut [f64]) {
        let width = self.nvar + 2;
        let flushed = &mut self.flushed[g * width..(g + 1) * width];
        let a = self.a[g];
        if a != 0.0 {
            let ds1 = self.running[0].since(flushed[0]);
            let ds2 = self.running[1].since(flushed[1]);
            u[g] -= a * ds1;
            fdiag[g] += a * ds1 - a * a * ds2;
            for ii in 0..self.nvar {
                let dt = self.running[ii + 2].since(flushed[ii + 2]);
                jmat[(ii, g)] += self.cmat[(ii, g)] * ds1 - a * dt;
            }
        }
        flushed.copy_from_slice(&self.running);
    }

    fn flush_person(&mut self, p: usize, u: &mut [f64], jmat: &mut Array2<f64>, fdiag: &mut [f64]) {
        if let Some(g) = self.frail.as_ref().map(|frail| frail[p]) {
            self.flush_group(g, u, jmat, fdiag);
        }
    }

    fn flush_all(&mut self, u: &mut [f64], jmat: &mut Array2<f64>, fdiag: &mut [f64]) {
        for g in 0..self.nf {
            self.flush_group(g, u, jmat, fdiag);
        }
    }

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
        let nvar = self.nvar;
        let temp = k as f64 * self.method / ndead;
        let d2 = denom - temp * efron_wt;
        for ii in 0..nvar {
            let i = ii + nf;
            let temp2 = (self.a[i] - temp * self.a2[i]) / d2;
            self.tmean[i] = temp2;
            u[i] -= meanwt * temp2;
        }
        for ii in 0..nvar {
            let i = ii + nf;
            for j in nf..=i {
                jmat[(ii, j)] += meanwt
                    * ((self.cmat[(ii, j)] - temp * self.cmat2[(ii, j)]) / d2
                        - self.tmean[i] * self.tmean[j]);
            }
        }
        if nf > 0 {
            let weight = meanwt / d2;
            self.running[0].add(weight);
            self.running[1].add(weight / d2);
            for ii in 0..nvar {
                self.running[ii + 2].add(weight * self.tmean[ii + nf]);
            }
            if temp != 0.0 {
                for &g in &self.tied {
                    let plain = self.a[g] / d2;
                    let tied = (self.a[g] - temp * self.a2[g]) / d2;
                    u[g] -= meanwt * (tied - plain);
                    fdiag[g] += meanwt * (tied * (1.0 - tied) - plain * (1.0 - plain));
                    for ii in 0..nvar {
                        jmat[(ii, g)] += meanwt
                            * (-temp * self.cmat2[(ii, g)] / d2
                                - self.tmean[ii + nf] * (tied - plain));
                    }
                }
            }
        }
        -meanwt * d2.ln()
    }

    fn reset_sums(&mut self, second: bool) {
        if second {
            for &g in &self.tied {
                self.a2[g] = 0.0;
                self.cmat2.column_mut(g).fill(0.0);
                self.tied_mark[g] = false;
            }
            self.tied.clear();
            self.a2[self.nf..].fill(0.0);
            self.cmat2.slice_mut(ndarray::s![.., self.nf..]).fill(0.0);
        } else {
            self.a.fill(0.0);
            self.cmat.fill(0.0);
        }
    }

    fn reset_running(&mut self) {
        self.running.fill(RunningSum::default());
        self.flushed.fill(RunningSum::default());
    }

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
        self.reset_running();
        for ip in 0..self.n {
            if ip == 0 || ip == self.strata_end[istrat] {
                efron_wt = 0.0;
                denom = 0.0;
                self.flush_all(u, jmat, fdiag);
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
            self.flush_person(p, u, jmat, fdiag);
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
        self.flush_all(u, jmat, fdiag);
        newlk
    }

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
        self.reset_running();
        self.reset_sums(false);
        self.reset_sums(true);
        let mut person = 0;
        while person < self.n {
            let p = self.sort1[person];
            if self.status[p] == 0 {
                let risk = self.score[p].exp() * self.weights[p];
                denom += risk;
                self.flush_person(p, u, jmat, fdiag);
                self.accumulate(p, risk, 1.0, false);
                person += 1;
            } else {
                let time = self.stop[p];
                while indx2 < self.strata_end[istrat] {
                    let q = self.sort2[indx2];
                    if self.start.as_ref().expect("counting-process data")[q] < time {
                        break;
                    }
                    let risk = self.score[q].exp() * self.weights[q];
                    denom -= risk;
                    self.flush_person(q, u, jmat, fdiag);
                    self.accumulate(q, risk, -1.0, false);
                    indx2 += 1;
                }
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
                    self.flush_person(q, u, jmat, fdiag);
                    self.accumulate(q, risk, 1.0, false);
                    if self.status[q] == 1 {
                        deaths += 1.0;
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
                self.flush_all(u, jmat, fdiag);
                self.reset_sums(false);
            }
        }
        self.flush_all(u, jmat, fdiag);
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
