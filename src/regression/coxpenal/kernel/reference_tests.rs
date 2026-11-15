//! The penalized Cox kernel as it stood before its loops were restructured for
//! speed, kept verbatim as an oracle. Every case asserts the production kernel
//! reproduces it bit for bit.

use super::*;
use crate::internal::rng::Rng;

struct Reference {
    n: usize,
    nvar: usize,
    nf: usize,
    covar: Array2<f64>,
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

impl Reference {
    fn new(data: &KernelData<'_>) -> Self {
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
        for i in 0..nvar {
            if !data.docenter[i] {
                continue;
            }
            let center = covar.column(i).sum() / n as f64;
            covar.column_mut(i).mapv_inplace(|value| value - center);
        }

        Self {
            n,
            nvar,
            nf,
            covar,
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

    fn initial_loglik(&self, beta: &[f64]) -> f64 {
        let fbeta = vec![0.0; self.nf];
        if self.start.is_some() {
            self.initial_loglik_counting(beta, &fbeta)
        } else {
            self.initial_loglik_right(beta, &fbeta)
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

    fn expected_events(&self) -> Vec<f64> {
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

/// The layout of a randomized data set.
#[derive(Clone, Copy)]
struct Shape {
    n: usize,
    nvar: usize,
    /// The number of distinct event times, or `None` for all distinct.
    distinct_times: Option<usize>,
    nstrata: usize,
    nfrail: usize,
    counting: bool,
}

/// Randomized data: tied or untied times, entries that coincide with event
/// times, strata, weights spanning two orders of magnitude, offsets, an
/// uncentred 0/1 column and frailty groups, some of them empty.
struct Case {
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    status: Vec<i32>,
    x: Array2<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    strata: Option<Vec<i32>>,
    frail: Option<Vec<usize>>,
    nfrail: usize,
    docenter: Vec<bool>,
    beta: Vec<f64>,
    fbeta: Vec<f64>,
}

impl Case {
    fn random(rng: &mut Rng, shape: Shape) -> Self {
        let n = shape.n;
        let mut slots: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut slots);
        let slots: Vec<usize> = match shape.distinct_times {
            Some(distinct) => (0..n).map(|_| rng.usize(0..distinct)).collect(),
            None => slots,
        };
        let stop = slots.iter().map(|&k| 1.0 + k as f64 * 0.5).collect();
        let start = shape.counting.then(|| {
            slots
                .iter()
                .map(|&k| {
                    if k == 0 || rng.bool() {
                        0.0
                    } else {
                        1.0 + rng.usize(0..k) as f64 * 0.5
                    }
                })
                .collect()
        });
        let status = (0..n).map(|_| i32::from(rng.f64() < 0.7)).collect();
        let mut x = Array2::zeros((n, shape.nvar));
        for ((_, j), value) in x.indexed_iter_mut() {
            *value = if j == 0 {
                f64::from(u8::from(rng.bool()))
            } else {
                3.0 * rng.f64() - 1.0 + j as f64
            };
        }
        let weights = (0..n).map(|_| 10f64.powf(2.0 * rng.f64() - 1.0)).collect();
        let offset = (0..n).map(|_| 0.5 * rng.f64() - 0.25).collect();
        let strata = (shape.nstrata > 1)
            .then(|| (0..n).map(|_| rng.usize(0..shape.nstrata) as i32).collect());
        let frail =
            (shape.nfrail > 0).then(|| (0..n).map(|_| rng.usize(0..shape.nfrail)).collect());
        Self {
            stop,
            start,
            status,
            x,
            weights,
            offset,
            strata,
            frail,
            nfrail: shape.nfrail,
            docenter: (0..shape.nvar).map(|j| j != 0).collect(),
            beta: (0..shape.nvar).map(|_| 0.6 * rng.f64() - 0.3).collect(),
            fbeta: (0..shape.nfrail).map(|_| 0.8 * rng.f64() - 0.4).collect(),
        }
    }

    fn data(&self, efron: bool) -> KernelData<'_> {
        KernelData {
            stop: &self.stop,
            start: self.start.as_deref(),
            status: &self.status,
            x: &self.x,
            weights: &self.weights,
            offset: &self.offset,
            strata: self.strata.as_deref(),
            frail: self.frail.as_deref(),
            nfrail: self.nfrail,
            efron,
            docenter: &self.docenter,
            unsafe_offsets: false,
        }
    }
}

struct Evaluation {
    loglik: f64,
    u: Vec<f64>,
    jmat: Array2<f64>,
    fdiag: Vec<f64>,
}

impl Evaluation {
    /// NaN-filled, so that evaluation must overwrite every entry it reports.
    fn new(nvar: usize, nf: usize) -> Self {
        Self {
            loglik: f64::NAN,
            u: vec![f64::NAN; nf + nvar],
            jmat: Array2::from_elem((nvar, nf + nvar), f64::NAN),
            fdiag: vec![f64::NAN; nf + nvar],
        }
    }
}

fn assert_bits(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "{label}[{i}]: {a:e} != {e:e}");
    }
}

/// Compares the two kernels at two coefficient vectors and returns the
/// reference's last evaluation.
fn compare(case: &Case, efron: bool, label: &str) -> Evaluation {
    let nvar = case.x.ncols();
    let nf = case.nfrail;
    let mut kernel = Kernel::new(case.data(efron));
    let mut reference = Reference::new(&case.data(efron));
    let means: Vec<f64> = (0..nvar)
        .map(|j| {
            if case.docenter[j] {
                case.x.column(j).sum() / case.x.nrows() as f64
            } else {
                0.0
            }
        })
        .collect();
    assert_bits(&format!("{label} means"), kernel.means(), &means);
    assert_eq!(
        kernel.initial_loglik(&case.beta).unwrap().to_bits(),
        reference.initial_loglik(&case.beta).to_bits(),
        "{label} initial loglik"
    );
    let mut new = Evaluation::new(nvar, nf);
    let mut old = Evaluation::new(nvar, nf);
    for trial in 0..2 {
        let label = format!("{label} trial {trial}");
        let beta: Vec<f64> = case.beta.iter().map(|b| b * f64::from(trial + 1)).collect();
        new.loglik = kernel
            .evaluate(
                &beta,
                &case.fbeta,
                &mut new.u,
                &mut new.jmat,
                &mut new.fdiag,
            )
            .unwrap();
        old.loglik = reference.evaluate(
            &beta,
            &case.fbeta,
            &mut old.u,
            &mut old.jmat,
            &mut old.fdiag,
        );
        assert_bits(&format!("{label} loglik"), &[new.loglik], &[old.loglik]);
        assert_bits(&format!("{label} u"), &new.u, &old.u);
        assert_bits(
            &format!("{label} fdiag"),
            &new.fdiag[..nf],
            &old.fdiag[..nf],
        );
        for i in 0..nvar {
            let (new_row, old_row) = (new.jmat.row(i), old.jmat.row(i));
            assert_bits(
                &format!("{label} jmat row {i}"),
                &new_row.as_slice().unwrap()[..=nf + i],
                &old_row.as_slice().unwrap()[..=nf + i],
            );
        }
        if case.start.is_none() {
            assert_bits(
                &format!("{label} expected events"),
                &kernel.expected_events().unwrap(),
                &reference.expected_events(),
            );
        }
    }
    old
}

fn shapes() -> Vec<Shape> {
    let mut shapes = Vec::new();
    for counting in [false, true] {
        for (distinct_times, nstrata) in [
            (None, 1),
            (None, 3),
            (Some(12), 1),
            (Some(6), 3),
            (Some(1), 2),
        ] {
            for (nvar, nfrail) in [(3, 0), (1, 0), (2, 7), (0, 4), (4, 60)] {
                shapes.push(Shape {
                    n: 160,
                    nvar,
                    distinct_times,
                    nstrata,
                    nfrail,
                    counting,
                });
            }
        }
    }
    shapes
}

#[test]
fn kernel_matches_the_reference_on_randomized_tied_stratified_weighted_data() {
    let mut rng = Rng::with_seed(0x5eed_c0de);
    for (index, shape) in shapes().into_iter().enumerate() {
        let case = Case::random(&mut rng, shape);
        for efron in [false, true] {
            compare(&case, efron, &format!("case {index} efron {efron}"));
        }
    }
}

#[test]
fn kernel_matches_the_reference_when_a_stratum_has_no_deaths() {
    let mut rng = Rng::with_seed(17);
    for counting in [false, true] {
        let mut case = Case::random(
            &mut rng,
            Shape {
                n: 90,
                nvar: 2,
                distinct_times: Some(5),
                nstrata: 3,
                nfrail: 9,
                counting,
            },
        );
        let strata = case.strata.clone().unwrap();
        for (status, stratum) in case.status.iter_mut().zip(strata) {
            if stratum == 1 {
                *status = 0;
            }
        }
        for efron in [false, true] {
            compare(&case, efron, &format!("counting {counting} efron {efron}"));
        }
    }
}

#[test]
fn kernel_matches_the_reference_with_zero_and_extreme_weights_and_empty_groups() {
    let mut rng = Rng::with_seed(99);
    for counting in [false, true] {
        let mut case = Case::random(
            &mut rng,
            Shape {
                n: 120,
                nvar: 3,
                distinct_times: Some(8),
                nstrata: 2,
                nfrail: 200,
                counting,
            },
        );
        for (i, weight) in case.weights.iter_mut().enumerate() {
            *weight *= [1e-150, 1.0, 1e150, 1.0, 0.0][i % 5];
        }
        for efron in [false, true] {
            compare(&case, efron, &format!("counting {counting} efron {efron}"));
        }
    }
}

#[test]
fn the_stable_path_reads_the_input_rows_unchanged() {
    let mut rng = Rng::with_seed(11);
    for counting in [false, true] {
        let case = Case::random(
            &mut rng,
            Shape {
                n: 70,
                nvar: 3,
                distinct_times: Some(7),
                nstrata: 3,
                nfrail: 8,
                counting,
            },
        );
        let kernel = Kernel::new(case.data(true));
        let reference = Reference::new(&case.data(true));
        let data = kernel.original();
        assert_bits(
            "covariates",
            data.covar.as_slice().unwrap(),
            reference.covar.as_slice().unwrap(),
        );
        assert_bits("weights", &data.weights, &reference.weights);
        assert_bits("offset", &data.offset, &reference.offset);
        assert_bits("stop", &data.stop, &reference.stop);
        assert_eq!(data.status, reference.status);
        assert_eq!(data.frail, reference.frail);
        assert_eq!(data.start.is_some(), counting);
        if let (Some(start), Some(expected)) = (&data.start, &reference.start) {
            assert_bits("start", start, expected);
        }
        assert_eq!(kernel.sort1, reference.sort1);
        assert_eq!(kernel.strata_end, reference.strata_end);
    }
}

#[test]
fn kernel_matches_the_reference_for_column_major_covariates() {
    let mut rng = Rng::with_seed(5);
    for counting in [false, true] {
        let mut case = Case::random(
            &mut rng,
            Shape {
                n: 100,
                nvar: 4,
                distinct_times: Some(9),
                nstrata: 2,
                nfrail: 6,
                counting,
            },
        );
        case.x = case.x.t().as_standard_layout().into_owned().reversed_axes();
        assert!(!case.x.is_standard_layout());
        for efron in [false, true] {
            compare(&case, efron, &format!("counting {counting} efron {efron}"));
        }
    }
}

#[test]
fn kernel_matches_the_reference_when_second_moments_overflow() {
    // The censored row cancels the first death in the risk-set sum, which
    // stays finite, while the two tied deaths' second moment overflows: the
    // reference's first death step then takes 0 * inf.
    for counting in [false, true] {
        let case = Case {
            stop: vec![2., 1., 1.],
            start: counting.then(|| vec![0.; 3]),
            status: vec![0, 1, 1],
            x: ndarray::array![[0., -1e308], [1., 1e308], [0., 1e308]],
            weights: vec![1.; 3],
            offset: vec![0.; 3],
            strata: None,
            frail: Some(vec![0, 1, 1]),
            nfrail: 2,
            docenter: vec![false; 2],
            beta: vec![0.; 2],
            fbeta: vec![0.1, -0.1],
        };
        for efron in [false, true] {
            let old = compare(&case, efron, &format!("counting {counting} efron {efron}"));
            assert!(old.u[2..].iter().any(|value| value.is_nan()));
        }
    }
}
