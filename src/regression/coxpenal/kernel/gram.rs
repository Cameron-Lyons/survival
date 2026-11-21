//! The right-censored penalized sweep without a sparse frailty, with its dense
//! information built from two weighted Gram matrices
//! ([`crate::regression::cox_gram`]).
//!
//! The log-likelihood, score and risk scores are those of
//! [`Kernel::evaluate_right`], bit for bit. A sweep that meets a non-finite
//! weight, factor or information entry returns `None`, and the caller runs
//! `evaluate_right` instead.

use super::*;
use crate::regression::cox_gram::CoxGram;

/// Scratch storage of [`Kernel::evaluate_right_gram`].
#[derive(Debug)]
pub(super) struct PenalizedGram {
    /// Each position's risk, then its weight `omega`.
    weight: Vec<f64>,
    /// Each tie set's last position and its `(alpha, gamma)`, in sweep order.
    sets: Vec<(usize, f64, f64)>,
    gram: CoxGram,
}

impl PenalizedGram {
    pub(super) fn new(n: usize, nvar: usize) -> Self {
        Self {
            weight: vec![0.0; n],
            sets: Vec::new(),
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
