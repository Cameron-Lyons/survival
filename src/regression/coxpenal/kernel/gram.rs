//! The right-censored penalized sweep without a sparse frailty, with its dense
//! information built from two weighted Gram matrices
//! ([`crate::regression::cox_gram`]).
//!
//! The log-likelihood, score and risk scores are those of
//! [`Kernel::evaluate_right`], bit for bit. A sweep that meets a non-finite
//! weight, factor or information entry returns `None`, and the caller runs
//! `evaluate_right` instead.

use super::*;
use crate::regression::cox_gram::{CoxGram, IntervalWeights};

/// Scratch storage of [`Kernel::evaluate_right_gram`].
#[derive(Debug)]
pub(super) struct PenalizedGram {
    /// Each position's risk, then its weight `omega`.
    weight: Vec<f64>,
    /// Each tie set's last position and its `(alpha, gamma)`, in sweep order.
    sets: Vec<(usize, f64, f64)>,
    /// Counting data: each death time's stratum and `(time, alpha, gamma)`.
    death_times: Vec<(usize, (f64, f64, f64))>,
    intervals: IntervalWeights,
    gram: CoxGram,
}

impl PenalizedGram {
    pub(super) fn new(n: usize, nvar: usize) -> Self {
        Self {
            weight: vec![0.0; n],
            sets: Vec::new(),
            death_times: Vec::new(),
            intervals: IntervalWeights::default(),
            gram: CoxGram::new(n, nvar),
        }
    }
}

impl Kernel {
    /// [`Self::evaluate_right`] with Gram-matrix information, for kernels that
    /// allocated a [`PenalizedGram`]; `None` sends the caller to
    /// `evaluate_right` with zeroed `u` and `jmat`.
    pub(super) fn evaluate_right_gram(
        &mut self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
    ) -> Option<f64> {
        let mut work = self.gram.take()?;
        let result = self.gram_sweep(&mut work, beta, fbeta, u, jmat);
        self.gram = Some(work);
        result
    }

    /// [`Self::evaluate_counting`] with Gram-matrix information, for kernels
    /// that allocated a [`PenalizedGram`]; `None` sends the caller to
    /// `evaluate_counting` with zeroed `u` and `jmat`.
    pub(super) fn evaluate_counting_gram(
        &mut self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
    ) -> Option<f64> {
        let mut work = self.gram.take()?;
        let result = self.counting_gram_sweep(&mut work, beta, fbeta, u, jmat);
        self.gram = Some(work);
        result
    }

    /// Adds `sign` times row `p`'s risk-weighted covariates to `a` (or `a2`),
    /// as [`Self::accumulate`] adds them, without the second moments.
    fn accumulate_first(&mut self, p: usize, risk: f64, sign: f64, second: bool) {
        let row = self.covar.row(p);
        let x = row.to_slice().expect("covariates are row-major");
        let target = if second { &mut self.a2 } else { &mut self.a };
        let risk = sign * risk;
        for (sum, &xi) in target.iter_mut().zip(x) {
            *sum += risk * xi;
        }
    }

    /// R's steps at one death time (`deaths` deaths of mean weight `meanwt`):
    /// the score terms and step means of [`Self::death_step`], recorded in
    /// `gram`; returns the time's `(alpha, gamma)` and adds each step's
    /// log-likelihood term to `newlk` as the sweeps do.
    #[allow(clippy::too_many_arguments)]
    fn gram_death_steps(
        &self,
        gram: &mut CoxGram,
        denom: f64,
        efron_wt: f64,
        deaths: usize,
        meanwt: f64,
        u: &mut [f64],
        newlk: &mut f64,
        before_step: impl Fn(usize, &mut f64),
    ) -> Option<(f64, f64)> {
        let ndead = deaths as f64;
        let (mut alpha, mut gamma) = (0.0, 0.0);
        let mut steps = EfronStepSums::default();
        for k in 0..deaths {
            before_step(k, newlk);
            let temp = k as f64 * self.method / ndead;
            let d2 = denom - temp * efron_wt;
            if deaths == 1 {
                let mut mean_row = gram.mean_row();
                for (((mean, score), &a), &a2) in mean_row
                    .iter_mut()
                    .zip(u.iter_mut())
                    .zip(&self.a)
                    .zip(&self.a2)
                {
                    let temp2 = (a - temp * a2) / d2;
                    *mean = temp2;
                    *score -= meanwt * temp2;
                }
            } else {
                for ((score, &a), &a2) in u.iter_mut().zip(&self.a).zip(&self.a2) {
                    *score -= meanwt * ((a - temp * a2) / d2);
                }
                steps.add(1.0 - temp, d2, denom);
            }
            *newlk += -meanwt * d2.ln();
            alpha += meanwt / d2;
            gamma += meanwt * (1.0 - temp) / d2;
        }
        if deaths == 1 {
            gram.push_mean(meanwt);
        } else {
            gram.push_tie_set(&steps, &self.a, &self.a2, denom, meanwt)?;
        }
        Some((alpha, gamma))
    }

    fn counting_gram_sweep(
        &mut self,
        work: &mut PenalizedGram,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
    ) -> Option<f64> {
        debug_assert_eq!(self.nf, 0);
        let nvar = self.nvar;
        work.gram.clear();
        work.death_times.clear();
        for p in 0..self.n {
            self.score[p] = self.eta(p, beta, fbeta);
        }
        let mut newlk = 0.0;
        let mut istrat = 0;
        let mut indx2 = 0;
        let mut denom = 0.0;
        self.a.fill(0.0);
        self.a2.fill(0.0);
        let mut person = 0;
        while person < self.n {
            if self.status[person] == 0 {
                let risk = self.score[person].exp() * self.weights[person];
                denom += risk;
                self.accumulate_first(person, risk, 1.0, false);
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
                    self.accumulate_first(q, risk, -1.0, false);
                    indx2 += 1;
                }
                let mut efron_wt = 0.0;
                let mut meanwt = 0.0;
                let mut deaths = 0.0;
                self.a2.fill(0.0);
                let mut k = person;
                while k < self.strata_end[istrat] {
                    if self.stop[k] < time {
                        break;
                    }
                    let risk = self.score[k].exp() * self.weights[k];
                    denom += risk;
                    self.accumulate_first(k, risk, 1.0, false);
                    if self.status[k] == 1 {
                        deaths += 1.0;
                        efron_wt += risk;
                        meanwt += self.weights[k];
                        let row = self.covar.row(k);
                        let x = row.to_slice().expect("covariates are row-major");
                        for (score, &xi) in u.iter_mut().zip(x) {
                            *score += self.weights[k] * xi;
                        }
                        if self.second_moments[k] {
                            self.accumulate_first(k, risk, 1.0, true);
                        }
                    }
                    k += 1;
                }
                meanwt /= deaths;
                // The deaths' own terms interleave with the steps, as in
                // `evaluate_counting`.
                let death_rows: Vec<usize> =
                    (person..k).filter(|&row| self.status[row] == 1).collect();
                let (weights, scores) = (&self.weights, &self.score);
                let (alpha, gamma) = self.gram_death_steps(
                    &mut work.gram,
                    denom,
                    efron_wt,
                    death_rows.len(),
                    meanwt,
                    u,
                    &mut newlk,
                    |step, newlk| {
                        let row = death_rows[step];
                        *newlk += weights[row] * scores[row];
                    },
                )?;
                work.death_times.push((istrat, (time, alpha, gamma)));
                person = k;
            }
            if person == self.strata_end[istrat] {
                istrat += 1;
                denom = 0.0;
                indx2 = person;
                self.a.fill(0.0);
            }
        }

        // Each row's weight: the alphas of the death times in (start, stop],
        // with its own death time's gamma in place of that time's alpha.
        let start = self.start.as_ref().expect("counting-process data");
        let mut taken = 0;
        let mut begin = 0;
        for (stratum, &end) in self.strata_end.iter().enumerate() {
            let count = work.death_times[taken..]
                .iter()
                .take_while(|time| time.0 == stratum)
                .count();
            let times = &work.death_times[taken..taken + count];
            taken += count;
            work.intervals.reset(times.iter().map(|time| &time.1));
            let rows = start[begin..end]
                .iter()
                .zip(&self.stop[begin..end])
                .zip(&self.status[begin..end])
                .zip(self.score[begin..end].iter().zip(&self.weights[begin..end]));
            for ((((&entry, &stop), &status), (&score, &weight)), target) in
                rows.zip(&mut work.weight[begin..end])
            {
                let share = work.intervals.share(entry, stop, status == 1);
                let omega = score.exp() * weight * share;
                if !omega.is_finite() {
                    return None;
                }
                *target = omega;
            }
            begin = end;
        }
        let gram = work.gram.finish(self.covar.view(), &work.weight);
        for ii in 0..nvar {
            for jj in 0..=ii {
                jmat[(ii, jj)] = gram[(ii, jj)];
            }
        }
        let finite = newlk.is_finite()
            && u.iter().all(|value| value.is_finite())
            && (0..nvar).all(|ii| (0..=ii).all(|jj| jmat[(ii, jj)].is_finite()));
        finite.then_some(newlk)
    }

    fn gram_sweep(
        &mut self,
        work: &mut PenalizedGram,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
    ) -> Option<f64> {
        debug_assert_eq!(self.nf, 0);
        let nvar = self.nvar;
        work.gram.clear();
        work.sets.clear();
        let mut newlk = 0.0;
        let mut denom = 0.0;
        let mut efron_wt = 0.0;
        let mut istrat = 0;
        for p in 0..self.n {
            if p == 0 || p == self.strata_end[istrat] {
                efron_wt = 0.0;
                denom = 0.0;
                self.a.fill(0.0);
                self.a2.fill(0.0);
            }
            if p == self.strata_end[istrat] {
                istrat += 1;
            }
            let zbeta = self.eta(p, beta, fbeta);
            self.score[p] = zbeta.exp();
            let risk = self.score[p] * self.weights[p];
            denom += risk;
            work.weight[p] = risk;
            let row = self.covar.row(p);
            let x = row.to_slice().expect("covariates are row-major");
            for (sum, &xi) in self.a.iter_mut().zip(x) {
                *sum += risk * xi;
            }
            if self.status[p] == 1 {
                efron_wt += risk;
                newlk += self.weights[p] * zbeta;
                for (score, &xi) in u.iter_mut().zip(x) {
                    *score += self.weights[p] * xi;
                }
                if self.second_moments[p] {
                    for (sum, &xi) in self.a2.iter_mut().zip(x) {
                        *sum += risk * xi;
                    }
                }
            }
            if self.mark[p] > 0 {
                let deaths = self.mark[p];
                let ndead = deaths as f64;
                let wtave = self.wtave[p];
                let (mut alpha, mut gamma) = (0.0, 0.0);
                let mut steps = EfronStepSums::default();
                for k in 0..deaths {
                    let temp = k as f64 * self.method / ndead;
                    let d2 = denom - temp * efron_wt;
                    if deaths == 1 {
                        let mut mean_row = work.gram.mean_row();
                        for (((mean, score), &a), &a2) in mean_row
                            .iter_mut()
                            .zip(u.iter_mut())
                            .zip(&self.a)
                            .zip(&self.a2)
                        {
                            let temp2 = (a - temp * a2) / d2;
                            *mean = temp2;
                            *score -= wtave * temp2;
                        }
                    } else {
                        for ((score, &a), &a2) in u.iter_mut().zip(&self.a).zip(&self.a2) {
                            *score -= wtave * ((a - temp * a2) / d2);
                        }
                        steps.add(1.0 - temp, d2, denom);
                    }
                    newlk += -wtave * d2.ln();
                    alpha += wtave / d2;
                    gamma += wtave * (1.0 - temp) / d2;
                }
                if deaths == 1 {
                    work.gram.push_mean(wtave);
                } else {
                    work.gram
                        .push_tie_set(&steps, &self.a, &self.a2, denom, wtave)?;
                }
                work.sets.push((p, alpha, gamma));
                efron_wt = 0.0;
                self.a2.fill(0.0);
            }
        }

        // Each position's weight: the alphas of the tie sets at or after it in
        // its stratum, with its own set's gamma in place of that set's alpha.
        // Strata are walked last to first, as the tie sets are popped.
        let mut next = work.sets.len();
        for stratum in (0..self.strata_end.len()).rev() {
            let begin = if stratum == 0 {
                0
            } else {
                self.strata_end[stratum - 1]
            };
            let end = self.strata_end[stratum];
            let mut cumulative = 0.0;
            let mut own = (0.0, 0);
            for p in (begin..end).rev() {
                if next > 0 && work.sets[next - 1].0 == p {
                    let (_, alpha, gamma) = work.sets[next - 1];
                    next -= 1;
                    own = (cumulative + gamma, self.mark[p]);
                    cumulative += alpha;
                }
                let share = if self.status[p] == 1 && own.1 > 0 {
                    own.1 -= 1;
                    own.0
                } else {
                    cumulative
                };
                let omega = work.weight[p] * share;
                if !omega.is_finite() {
                    return None;
                }
                work.weight[p] = omega;
            }
        }
        debug_assert_eq!(next, 0);
        let gram = work.gram.finish(self.covar.view(), &work.weight);
        for ii in 0..nvar {
            for jj in 0..=ii {
                jmat[(ii, jj)] = gram[(ii, jj)];
            }
        }
        let finite = newlk.is_finite()
            && u.iter().all(|value| value.is_finite())
            && (0..nvar).all(|ii| (0..=ii).all(|jj| jmat[(ii, jj)].is_finite()));
        finite.then_some(newlk)
    }
}
