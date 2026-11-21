//! The right-censored Breslow and Efron evaluation with its information built
//! from two weighted Gram matrices ([`crate::regression::cox_gram`]).
//!
//! The log-likelihood and score are those of R's sweep, bit for bit. A sweep
//! that meets an extreme risk or a non-finite weight or information entry
//! returns `None`, and the caller evaluates R's sweep instead, so overflow and
//! missing values behave as before.

use super::*;
use crate::regression::cox_gram::CoxGram;
pub(super) use crate::regression::cox_gram::GRAM_MIN_VARIABLES;

/// Scratch storage of [`CoxData::coxfit6_gram`].
#[derive(Debug)]
pub(crate) struct GramWorkspace {
    risk_set: RiskSetSums,
    tied: RiskSetSums,
    /// Each subject's risk, then its weight `omega`.
    weight: Vec<f64>,
    /// Each time group's `(start, end, stratum, alpha, gamma)`, in sweep order.
    groups: Vec<(usize, usize, usize, f64, f64)>,
    gram: CoxGram,
}

impl GramWorkspace {
    pub(super) fn new(n: usize, nvar: usize) -> Self {
        Self {
            risk_set: RiskSetSums::zeros(nvar, false),
            tied: RiskSetSums::zeros(nvar, false),
            weight: vec![0.0; n],
            groups: Vec::new(),
            gram: CoxGram::new(n, nvar),
        }
    }
}

impl CoxData {
    /// The evaluation of [`Self::coxfit6`] with Gram-matrix information, or
    /// `None` when it meets an extreme risk or a non-finite value; the caller
    /// then runs R's sweep on zeroed `u` and `imat`.
    pub(super) fn coxfit6_gram(
        &self,
        beta: &[f64],
        eta: &mut [f64],
        workspace: &mut GramWorkspace,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> Option<f64> {
        let nvar = u.len();
        let row = self.rows();
        self.fill_linear_predictors(beta, eta);
        workspace.groups.clear();
        workspace.gram.clear();
        workspace.risk_set.clear();
        let mut loglik = 0.0;
        let mut stratum = 0;
        let mut end = self.time.len();
        while end > 0 {
            if self.stratum_end[end - 1] == 1 {
                workspace.risk_set.clear();
                stratum += 1;
            }
            let time = self.time[end - 1];
            let mut start = end - 1;
            while start > 0 && self.stratum_end[start - 1] != 1 && self.time[start - 1] == time {
                start -= 1;
            }
            let ndead = (start..end)
                .filter(|&person| self.status[person] != 0)
                .count();
            let tied_sums = self.efron && ndead > 1;
            if tied_sums {
                workspace.tied.clear();
            }
            let mut death_weight = 0.0;
            for person in (start..end).rev() {
                let (weight, x) = (self.weights[person], row(person));
                let risk = eta[person].exp() * weight;
                if !(1e-100..=1e100).contains(&risk) {
                    return None;
                }
                workspace.weight[person] = risk;
                workspace.risk_set.add(weight, risk, x);
                if self.status[person] != 0 {
                    if tied_sums {
                        workspace.tied.add(weight, risk, x);
                    }
                    death_weight += weight;
                    loglik += weight * eta[person];
                    add_scaled(u, weight, x);
                }
            }
            let (mut alpha, mut gamma) = (0.0, 0.0);
            if ndead > 0 {
                let steps = if self.efron { ndead } else { 1 };
                let wtave = death_weight / steps as f64;
                // The set's terms are summed before they join the total, as
                // R's sweep adds them.
                let mut set_loglik = 0.0;
                let mut step_sums = EfronStepSums::default();
                for j in 0..steps {
                    let fraction = j as f64 / ndead as f64;
                    let sum =
                        |all: f64, deaths: f64| if j == 0 { all } else { all - fraction * deaths };
                    let (risk_set, tied) = (&workspace.risk_set, &workspace.tied);
                    let denom = sum(risk_set.denom, tied.denom);
                    set_loglik -= wtave * denom.ln();
                    if steps == 1 {
                        let mut mean_row = workspace.gram.mean_row();
                        for (i, (score, mean)) in u.iter_mut().zip(mean_row.iter_mut()).enumerate()
                        {
                            *mean = sum(risk_set.a[i], tied.a[i]) / denom;
                            *score -= wtave * *mean;
                        }
                    } else {
                        for (i, score) in u.iter_mut().enumerate() {
                            *score -= wtave * (sum(risk_set.a[i], tied.a[i]) / denom);
                        }
                        step_sums.add(1.0 - fraction, denom, risk_set.denom);
                    }
                    alpha += wtave / denom;
                    gamma += wtave * (1.0 - fraction) / denom;
                }
                loglik += set_loglik;
                if steps == 1 {
                    workspace.gram.push_mean(wtave);
                } else {
                    let (risk_set, tied) = (&workspace.risk_set, &workspace.tied);
                    workspace.gram.push_tie_set(
                        &step_sums,
                        &risk_set.a,
                        &tied.a,
                        risk_set.denom,
                        wtave,
                    )?;
                }
            }
            workspace.groups.push((start, end, stratum, alpha, gamma));
            end = start;
        }

        // Each subject's weight: the alphas of the death times it is at risk
        // at, with its own death time's gamma in place of that time's alpha.
        let mut cumulative = 0.0;
        let mut current = usize::MAX;
        for &(start, end, stratum, alpha, gamma) in workspace.groups.iter().rev() {
            if stratum != current {
                cumulative = 0.0;
                current = stratum;
            }
            let before = cumulative;
            cumulative += alpha;
            for person in start..end {
                let share = if self.status[person] != 0 {
                    before + gamma
                } else {
                    cumulative
                };
                let omega = workspace.weight[person] * share;
                if !omega.is_finite() {
                    return None;
                }
                workspace.weight[person] = omega;
            }
        }
        let gram = workspace.gram.finish(self.covar.view(), &workspace.weight);
        for k in 0..nvar {
            for i in k..nvar {
                imat[(k, i)] = gram[(k, i)];
            }
        }
        let finite = loglik.is_finite()
            && u.iter().all(|value| value.is_finite())
            && (0..nvar).all(|k| (k..nvar).all(|i| imat[(k, i)].is_finite()));
        finite.then_some(loglik)
    }
}
