use crate::error::SurvivalResult;
use crate::regression::cox_gram::GRAM_MIN_VARIABLES;
use crate::regression::efron_steps::{EfronStepSums, scaled_first};
use crate::regression::log_scale::LogScale;
use crate::regression::penalized::cholesky3::{chinv3, cholesky3, chsolve3, finish_factors};
use crate::regression::penalized::terms::{PenaltyCallback, PenaltyShape};
use ndarray::Array2;
use std::sync::OnceLock;

mod gram;
#[cfg(test)]
mod reference_tests;
mod stable;
#[cfg(test)]
mod stable_tests;
mod tied_frailty;

use self::stable::OriginalOrder;
use self::tied_frailty::TieSums;

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

/// The penalized Cox likelihood sweeps (R's coxfit5 and agfit5).
///
/// Every per-observation array is stored in `sort1` order (stratum, then
/// decreasing stop time, censored before deaths), so a sweep streams memory
/// instead of gathering rows; `sort1` maps a position back to its input row.
/// The arithmetic and its order are those of a sweep through the input rows
/// by `sort1`.
pub(super) struct Kernel {
    n: usize,
    nvar: usize,
    nf: usize,
    covar: Array2<f64>,
    covar_bounds: Vec<f64>,
    predictor_anchors: OnceLock<Vec<usize>>,
    unsafe_offsets: bool,
    stable_score: Option<Vec<LogScale>>,
    means: Vec<f64>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    status: Vec<i32>,
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    frail: Option<Vec<usize>>,
    /// The input row at each position.
    sort1: Vec<usize>,
    /// Positions by stratum and decreasing entry time (counting data).
    exits: Vec<usize>,
    original: OnceLock<OriginalOrder>,
    strata_end: Vec<usize>,
    mark: Vec<usize>,
    wtave: Vec<f64>,
    /// The deaths whose Efron second moments can reach a result; see
    /// [`second_moment_deaths`].
    second_moments: Vec<bool>,
    method: f64,
    score: Vec<f64>,
    a: Vec<f64>,
    a2: Vec<f64>,
    tmean: Vec<f64>,
    cmat: Array2<f64>,
    cmat2: Array2<f64>,
    /// Whether `a2` or `cmat2` holds anything since it was last cleared.
    second_pending: bool,
    running: Vec<RunningSum>,
    flushed: Vec<RunningSum>,
    tied: Vec<usize>,
    tied_mark: Vec<bool>,
    /// The current tie set's corrections of the `tied` frailty groups.
    tie_sums: TieSums,
    /// The current tie set's death steps, whose dense information
    /// [`Self::finish_information`] adds when there is no sparse frailty.
    steps: EfronStepSums,
    /// The Gram evaluation's storage, for fits without a sparse frailty and
    /// with at least [`GRAM_MIN_VARIABLES`] covariates.
    gram: Option<Box<gram::PenalizedGram>>,
}

/// Efron's second moments (`a2`, `cmat2`) enter death step `k` of a tie set
/// of `d` only through `k / d` times them, which is zero for a lone death and
/// for Breslow ties; while they stay finite, `x - 0 * a2` is `x` exactly (no
/// running sum is ever -0), so those deaths need not accumulate them. They
/// stay finite when `n` terms of the largest possible magnitude,
/// `e^22 * weight * max(1, |x|)^2`, sum to a finite value with room to spare;
/// otherwise every death accumulates them, keeping the NaN that
/// `0 * inf` gives.
fn second_moment_deaths(
    status: &[i32],
    stop: &[f64],
    strata_end: &[usize],
    efron: bool,
    largest_term: f64,
) -> Vec<bool> {
    let n = status.len();
    let bounded = (4.0 * n as f64 * largest_term).is_finite();
    let mut deaths = vec![false; n];
    let mut begin = 0;
    for &end in strata_end {
        let mut i = begin;
        while i < end {
            let mut j = i + 1;
            if status[i] == 1 {
                while j < end && status[j] == 1 && stop[j] == stop[i] {
                    j += 1;
                }
                deaths[i..j].fill(!bounded || (efron && j - i > 1));
            }
            i = j;
        }
        begin = end;
    }
    deaths
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
    pub unsafe_offsets: bool,
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
        let mut position = vec![0; n];
        for (ip, &row) in sort1.iter().enumerate() {
            position[row] = ip;
        }
        // A stable sort of the input rows: tied entries leave the risk set in
        // input order.
        let exits = match data.start {
            Some(start) => {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&l, &r| {
                    stratum(l)
                        .cmp(&stratum(r))
                        .then_with(|| start[r].total_cmp(&start[l]))
                });
                order.into_iter().map(|row| position[row]).collect()
            }
            None => Vec::new(),
        };
        let mut strata_end = Vec::new();
        for position in 0..n {
            if position + 1 == n || stratum(sort1[position]) != stratum(sort1[position + 1]) {
                strata_end.push(position + 1);
            }
        }

        let sorted =
            |values: &[f64]| -> Vec<f64> { sort1.iter().map(|&row| values[row]).collect() };
        let weights = sorted(data.weights);
        let stop = sorted(data.stop);
        let status: Vec<i32> = sort1.iter().map(|&row| data.status[row]).collect();

        let mut mark = vec![0usize; n];
        let mut wtave = vec![0.0; n];
        if data.start.is_none() {
            let mut istrat = 0;
            let mut i = 0;
            while i < n {
                if status[i] == 1 {
                    let mut ndead = 0;
                    let mut total = 0.0;
                    let mut j = i;
                    while j < n {
                        if stop[j] != stop[i] || j == strata_end[istrat] {
                            break;
                        }
                        ndead += 1;
                        total += weights[j];
                        j += 1;
                    }
                    mark[j - 1] = ndead;
                    wtave[j - 1] = total / ndead as f64;
                    i = j;
                } else {
                    i += 1;
                }
                if i == strata_end[istrat] {
                    istrat += 1;
                }
            }
        }

        // Each mean sums its column in input order and layout, which fixes
        // its last bits; only then are the centred rows laid out by position.
        let means: Vec<f64> = (0..nvar)
            .map(|i| {
                if data.docenter[i] {
                    data.x.column(i).sum() / n as f64
                } else {
                    0.0
                }
            })
            .collect();
        let mut covar = Array2::zeros((n, nvar));
        for (mut target, &row) in covar.rows_mut().into_iter().zip(&sort1) {
            for (((value, &x), &center), &docenter) in target
                .iter_mut()
                .zip(data.x.row(row))
                .zip(&means)
                .zip(data.docenter)
            {
                *value = if docenter { x - center } else { x };
            }
        }
        let covar_bounds: Vec<f64> = (0..nvar)
            .map(|j| covar.column(j).iter().map(|x| x.abs()).fold(0.0, f64::max))
            .collect();
        let largest_risk =
            coxsafe(f64::INFINITY).exp() * weights.iter().fold(0.0, |m: f64, w| m.max(w.abs()));
        let largest_x = covar_bounds.iter().fold(1.0, |m: f64, &b| m.max(b));
        let second_moments = second_moment_deaths(
            &status,
            &stop,
            &strata_end,
            data.efron,
            largest_risk * largest_x * largest_x,
        );

        Self {
            n,
            nvar,
            nf,
            covar_bounds,
            predictor_anchors: OnceLock::new(),
            unsafe_offsets: data.unsafe_offsets,
            stable_score: None,
            covar,
            means,
            weights,
            offset: sorted(data.offset),
            status,
            stop,
            start: data.start.map(sorted),
            frail: data
                .frail
                .map(|frail| sort1.iter().map(|&row| frail[row]).collect()),
            sort1,
            exits,
            original: OnceLock::new(),
            strata_end,
            mark,
            wtave,
            second_moments,
            method: f64::from(u8::from(data.efron)),
            score: vec![0.0; n],
            a: vec![0.0; nvar2],
            a2: vec![0.0; nvar2],
            tmean: vec![0.0; nvar2],
            cmat: Array2::zeros((nvar, nvar2)),
            cmat2: Array2::zeros((nvar, nvar2)),
            second_pending: false,
            running: vec![RunningSum::default(); nvar + 2],
            flushed: vec![RunningSum::default(); nf * (nvar + 2)],
            tied: Vec::new(),
            tied_mark: vec![false; nf],
            tie_sums: TieSums::new(nvar),
            steps: EfronStepSums::default(),
            gram: (nf == 0 && nvar >= GRAM_MIN_VARIABLES)
                .then(|| Box::new(gram::PenalizedGram::new(n, nvar))),
        }
    }

    pub(super) fn means(&self) -> &[f64] {
        &self.means
    }

    /// Observation `p`'s centred covariates.
    fn row(&self, p: usize) -> &[f64] {
        self.covar
            .row(p)
            .to_slice()
            .expect("covariates are row-major")
    }

    fn eta(&self, p: usize, beta: &[f64], fbeta: &[f64]) -> f64 {
        debug_assert_eq!(beta.len(), self.nvar);
        let mut zbeta = self.offset[p];
        if let Some(frail) = &self.frail {
            zbeta += fbeta[frail[p]];
        }
        for (b, x) in beta.iter().zip(self.row(p)) {
            zbeta += b * x;
        }
        coxsafe(zbeta)
    }

    pub(super) fn initial_loglik(&self, beta: &[f64]) -> SurvivalResult<f64> {
        let no_frailty = vec![0.0; self.nf];
        if self.needs_stable(beta, &no_frailty) {
            let mut u = vec![0.0; self.nf + self.nvar];
            let mut jmat = Array2::zeros((self.nvar, self.nf + self.nvar));
            let mut fdiag = vec![0.0; self.nf + self.nvar];
            return self
                .evaluate_stable(beta, &no_frailty, &mut u, &mut jmat, &mut fdiag)
                .map(|(loglik, _)| loglik);
        }
        Ok(if self.start.is_some() {
            self.initial_loglik_counting(beta, &no_frailty)
        } else {
            self.initial_loglik_right(beta, &no_frailty)
        })
    }

    fn initial_loglik_right(&self, beta: &[f64], fbeta: &[f64]) -> f64 {
        let mut loglik = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        let mut istrat = 0;
        for p in 0..self.n {
            if p == self.strata_end[istrat] {
                denom = 0.0;
                istrat += 1;
            }
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
            if self.status[person] == 0 {
                denom += score[person].exp() * self.weights[person];
                person += 1;
            } else {
                let time = self.stop[person];
                while indx2 < self.strata_end[istrat] {
                    let q = self.exits[indx2];
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
                    if self.stop[k] < time {
                        break;
                    }
                    let risk = score[k].exp() * self.weights[k];
                    denom += risk;
                    if self.status[k] == 1 {
                        deaths += 1.0;
                        efron_wt += risk;
                        meanwt += self.weights[k];
                    }
                    k += 1;
                }
                meanwt /= deaths;
                let mut itemp = -1.0;
                while person < k {
                    if self.status[person] == 1 {
                        itemp += 1.0;
                        let temp = itemp * self.method / deaths;
                        loglik += self.weights[person] * score[person]
                            - meanwt * (denom - temp * efron_wt).ln();
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
        self.second_pending |= second;
        if second
            && let Some(g) = fgrp
            && !self.tied_mark[g]
        {
            self.tied_mark[g] = true;
            self.tied.push(g);
        }
        let row = self.covar.row(p);
        let x = row.to_slice().expect("covariates are row-major");
        let (target, cmat) = if second {
            (&mut self.a2, &mut self.cmat2)
        } else {
            (&mut self.a, &mut self.cmat)
        };
        // Each product keeps the association ((sign * risk) * x_i) * x_j.
        let risk = sign * risk;
        if let Some(g) = fgrp {
            target[g] += risk;
        }
        for (sum, &xi) in target[nf..].iter_mut().zip(x) {
            *sum += risk * xi;
        }
        for (i, &xi) in x.iter().enumerate() {
            let weighted = risk * xi;
            let cross = cmat
                .row_mut(i)
                .into_slice()
                .expect("moment matrices are row-major");
            if let Some(g) = fgrp {
                cross[g] += weighted;
            }
            for (sum, &xj) in cross[nf..=nf + i].iter_mut().zip(&x[..=i]) {
                *sum += weighted * xj;
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

    /// Death step `k` of a tie set of `ndead`: its log-likelihood term, with
    /// its score and information added in. The Efron corrections of the tied
    /// frailty groups are only summed here; [`Self::finish_tie_set`] applies
    /// them once the set's steps are done. Without a sparse frailty, a set of
    /// several deaths only records the step, and [`Self::finish_information`]
    /// adds the information of all its steps.
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
    ) -> f64 {
        let nf = self.nf;
        let nvar = self.nvar;
        let temp = k as f64 * self.method / ndead;
        let d2 = denom - temp * efron_wt;
        for (((mean, score), &a), &a2) in self.tmean[nf..]
            .iter_mut()
            .zip(&mut u[nf..])
            .zip(&self.a[nf..])
            .zip(&self.a2[nf..])
        {
            let temp2 = (a - temp * a2) / d2;
            *mean = temp2;
            *score -= meanwt * temp2;
        }
        if nf == 0 && ndead > 1.0 {
            self.steps.add(1.0 - temp, d2, denom);
        } else {
            self.add_step_information(temp, d2, meanwt, jmat);
        }
        if nf > 0 {
            let weight = meanwt / d2;
            self.running[0].add(weight);
            self.running[1].add(weight / d2);
            for ii in 0..nvar {
                self.running[ii + 2].add(weight * self.tmean[ii + nf]);
            }
            if temp != 0.0 {
                self.tie_sums
                    .add(meanwt, temp, denom, d2, &self.tmean[nf..]);
            }
        }
        -meanwt * d2.ln()
    }

    /// R's dense information update of one death step, from the step means
    /// in `tmean`.
    fn add_step_information(&self, temp: f64, d2: f64, meanwt: f64, jmat: &mut Array2<f64>) {
        let nf = self.nf;
        let means = &self.tmean[nf..];
        for (ii, &mean_i) in means.iter().enumerate() {
            let span = nf..=nf + ii;
            let cmat = self.cmat.row(ii);
            let cmat2 = self.cmat2.row(ii);
            let mut info = jmat.row_mut(ii);
            let info = &mut info.as_slice_mut().expect("information is row-major")[span.clone()];
            let cmat = &cmat.to_slice().expect("moment matrices are row-major")[span.clone()];
            let cmat2 = &cmat2.to_slice().expect("moment matrices are row-major")[span];
            for (((value, &c), &c2), &mean_j) in info.iter_mut().zip(cmat).zip(cmat2).zip(means) {
                *value += meanwt * ((c - temp * c2) / d2 - mean_i * mean_j);
            }
        }
    }

    /// The dense information of a tie set's recorded death steps: collapsed
    /// over the steps (see [`crate::regression::efron_steps`]) when its sums
    /// are finite, and otherwise R's update of each step in turn, which is
    /// what [`Self::death_step`] would have added.
    fn finish_information(
        &mut self,
        denom: f64,
        efron_wt: f64,
        ndead: usize,
        meanwt: f64,
        jmat: &mut Array2<f64>,
    ) {
        if self.nf > 0 || ndead < 2 {
            return;
        }
        let steps = std::mem::take(&mut self.steps);
        let nvar = self.nvar;
        let finite = self.a.iter().chain(&self.a2).all(|v| v.is_finite())
            && (0..nvar).all(|ii| {
                (0..=ii)
                    .all(|jj| self.cmat[(ii, jj)].is_finite() && self.cmat2[(ii, jj)].is_finite())
            });
        if steps.is_usable() && finite {
            let scale = 1.0 / denom;
            for ii in 0..nvar {
                let moments_i = scaled_first(self.a[ii], self.a2[ii], scale);
                for jj in 0..=ii {
                    let moments_j = scaled_first(self.a[jj], self.a2[jj], scale);
                    jmat[(ii, jj)] += meanwt
                        * steps.entry(
                            moments_i,
                            moments_j,
                            self.cmat[(ii, jj)],
                            self.cmat2[(ii, jj)],
                            scale,
                        );
                }
            }
            return;
        }
        let deaths = ndead as f64;
        for k in 0..ndead {
            let temp = k as f64 * self.method / deaths;
            let d2 = denom - temp * efron_wt;
            for ((mean, &a), &a2) in self.tmean.iter_mut().zip(&self.a).zip(&self.a2) {
                *mean = (a - temp * a2) / d2;
            }
            self.add_step_information(temp, d2, meanwt, jmat);
        }
    }

    fn reset_sums(&mut self, second: bool) {
        if second {
            if !std::mem::take(&mut self.second_pending) {
                return;
            }
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
        for p in 0..self.n {
            if p == 0 || p == self.strata_end[istrat] {
                efron_wt = 0.0;
                denom = 0.0;
                self.flush_all(u, jmat, fdiag);
                self.reset_sums(false);
                self.reset_sums(true);
            }
            if p == self.strata_end[istrat] {
                istrat += 1;
            }
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
                for (score, &x) in u[nf..].iter_mut().zip(self.row(p)) {
                    *score += self.weights[p] * x;
                }
                if self.second_moments[p] {
                    self.accumulate(p, risk, 1.0, true);
                }
            }
            if self.mark[p] > 0 {
                let ndead = self.mark[p] as f64;
                let wtave = self.wtave[p];
                for k in 0..self.mark[p] {
                    newlk += self.death_step(denom, efron_wt, k, ndead, wtave, u, jmat);
                }
                self.finish_information(denom, efron_wt, self.mark[p], wtave, jmat);
                self.finish_tie_set(denom, u, jmat, fdiag);
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
            if self.status[person] == 0 {
                let risk = self.score[person].exp() * self.weights[person];
                denom += risk;
                self.flush_person(person, u, jmat, fdiag);
                self.accumulate(person, risk, 1.0, false);
                person += 1;
            } else {
                let time = self.stop[person];
                while indx2 < self.strata_end[istrat] {
                    let q = self.exits[indx2];
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
                    if self.stop[k] < time {
                        break;
                    }
                    let risk = self.score[k].exp() * self.weights[k];
                    denom += risk;
                    self.flush_person(k, u, jmat, fdiag);
                    self.accumulate(k, risk, 1.0, false);
                    if self.status[k] == 1 {
                        deaths += 1.0;
                        efron_wt += risk;
                        meanwt += self.weights[k];
                        if let Some(frail) = &self.frail {
                            u[frail[k]] += self.weights[k];
                        }
                        for (score, &x) in u[nf..].iter_mut().zip(self.row(k)) {
                            *score += self.weights[k] * x;
                        }
                        if self.second_moments[k] {
                            self.accumulate(k, risk, 1.0, true);
                        }
                    }
                    k += 1;
                }
                meanwt /= deaths;
                let mut itemp = 0;
                while person < k {
                    if self.status[person] == 1 {
                        newlk += self.weights[person] * self.score[person];
                        newlk += self.death_step(denom, efron_wt, itemp, deaths, meanwt, u, jmat);
                        itemp += 1;
                    }
                    person += 1;
                }
                self.finish_information(denom, efron_wt, itemp, meanwt, jmat);
                self.finish_tie_set(denom, u, jmat, fdiag);
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
    ) -> SurvivalResult<f64> {
        u.fill(0.0);
        jmat.fill(0.0);
        fdiag[..self.nf].fill(0.0);
        if self.needs_stable(beta, fbeta) {
            let (loglik, scores) = self.evaluate_stable(beta, fbeta, u, jmat, fdiag)?;
            self.stable_score = Some(scores);
            return Ok(loglik);
        }
        self.stable_score = None;
        Ok(if self.start.is_some() {
            if let Some(loglik) = self.evaluate_counting_gram(beta, fbeta, u, jmat) {
                loglik
            } else {
                u.fill(0.0);
                jmat.fill(0.0);
                self.evaluate_counting(beta, fbeta, u, jmat, fdiag)
            }
        } else if let Some(loglik) = self.evaluate_right_gram(beta, fbeta, u, jmat) {
            loglik
        } else {
            u.fill(0.0);
            jmat.fill(0.0);
            self.evaluate_right(beta, fbeta, u, jmat, fdiag)
        })
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
            newlk = self.evaluate(beta, fbeta, &mut u, &mut jmat, &mut fdiag)?;

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

    pub(super) fn expected_events(&self) -> SurvivalResult<Vec<f64>> {
        if let Some(scores) = &self.stable_score {
            return self.stable_expected_events(scores);
        }
        let n = self.n;
        let mut expect = vec![0.0; n];
        let mut hazard_jump = vec![0.0; n];
        let mut istrat = 0;
        let mut denom = 0.0;
        for p in 0..n {
            if p == self.strata_end[istrat] {
                denom = 0.0;
                istrat += 1;
            }
            denom += self.score[p] * self.weights[p];
            if self.mark[p] > 0 {
                let ndead = self.mark[p];
                let mut wtsum = 0.0;
                let mut efron_wt = 0.0;
                for i in (p + 1 - ndead..=p).rev() {
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
            let p = ip as usize;
            if self.status[p] > 0 {
                let ndead = self.mark[p];
                let jump = expect[p];
                let hazard2 = hazard_jump[p];
                let tie = p + 1 - ndead..=p;
                for (expected, &score) in expect[tie.clone()].iter_mut().zip(&self.score[tie]) {
                    *expected = score * (hazard + hazard2);
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
        Ok(self.unsorted(&expect))
    }

    /// `values`, one per position, in input row order.
    fn unsorted<T: Copy + Default>(&self, values: &[T]) -> Vec<T> {
        let mut rows = vec![T::default(); self.n];
        for (&row, &value) in self.sort1.iter().zip(values) {
            rows[row] = value;
        }
        rows
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

    #[test]
    fn only_tied_efron_deaths_accumulate_second_moments_while_they_stay_finite() {
        // Positions in risk-set order: a tie of two deaths at 5, a lone death
        // at 4 after a censoring, and a tie at 3 split by the stratum boundary.
        let status = [1, 1, 0, 1, 1, 1, 0];
        let stop = [5., 5., 4., 4., 3., 3., 3.];
        let strata_end = [5, 7];
        let flags =
            |efron, largest| second_moment_deaths(&status, &stop, &strata_end, efron, largest);
        assert_eq!(
            flags(true, 1e9),
            [true, true, false, false, false, false, false]
        );
        assert_eq!(flags(false, 1e9), [false; 7]);
        let deaths = status.map(|s| s == 1);
        assert_eq!(flags(true, f64::MAX / 8.0), deaths);
        assert_eq!(flags(false, f64::INFINITY), deaths);
    }
}
