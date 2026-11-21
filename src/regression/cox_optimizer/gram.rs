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
    /// Counting data: each row's entry time, read from the walk once.
    entry: Vec<f64>,
    /// Counting data: each death time's `(stratum, time, alpha, gamma)`, in
    /// walk order.
    death_times: Vec<(usize, f64, f64, f64)>,
    /// Counting data: one stratum's death times, increasing, with the
    /// double-double prefix sums of their alphas and their gammas.
    ascending: Vec<f64>,
    prefix: Vec<(f64, f64)>,
    gammas: Vec<f64>,
    gram: CoxGram,
}

impl GramWorkspace {
    pub(super) fn new(n: usize, nvar: usize) -> Self {
        Self {
            risk_set: RiskSetSums::zeros(nvar, false),
            tied: RiskSetSums::zeros(nvar, false),
            weight: vec![0.0; n],
            groups: Vec::new(),
            entry: Vec::new(),
            death_times: Vec::new(),
            ascending: Vec::new(),
            prefix: Vec::new(),
            gammas: Vec::new(),
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
                let set_loglik;
                (set_loglik, alpha, gamma) = self.gram_death_time(
                    &workspace.risk_set,
                    &workspace.tied,
                    ndead,
                    death_weight,
                    u,
                    &mut workspace.gram,
                )?;
                loglik += set_loglik;
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

    /// R's Breslow or Efron steps at one death time of `ndead` deaths, whose
    /// case weights total `death_weight`: subtracts the steps' means from the
    /// score as `death_time_update` does, records their mean rows, and
    /// returns the set's log-likelihood term with its `alpha` and `gamma`.
    fn gram_death_time(
        &self,
        risk_set: &RiskSetSums,
        tied: &RiskSetSums,
        ndead: usize,
        death_weight: f64,
        u: &mut [f64],
        gram: &mut CoxGram,
    ) -> Option<(f64, f64, f64)> {
        let steps = if self.efron { ndead } else { 1 };
        let wtave = death_weight / steps as f64;
        // The set's terms are summed before they join the total, as R's
        // sweep adds them.
        let (mut set_loglik, mut alpha, mut gamma) = (0.0, 0.0, 0.0);
        let mut step_sums = EfronStepSums::default();
        for j in 0..steps {
            let fraction = j as f64 / ndead as f64;
            let sum = |all: f64, deaths: f64| if j == 0 { all } else { all - fraction * deaths };
            let denom = sum(risk_set.denom, tied.denom);
            set_loglik -= wtave * denom.ln();
            if steps == 1 {
                let mut mean_row = gram.mean_row();
                for (i, (score, mean)) in u.iter_mut().zip(mean_row.iter_mut()).enumerate() {
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
        if steps == 1 {
            gram.push_mean(wtave);
        } else {
            gram.push_tie_set(&step_sums, &risk_set.a, &tied.a, risk_set.denom, wtave)?;
        }
        Some((set_loglik, alpha, gamma))
    }

    /// [`Self::agfit4`]'s walk on risks centred at zero, with Gram-matrix
    /// information; `None` sends the caller to the walk on zeroed `u` and
    /// `imat`. A row at risk over `(entry, stop]` is weighted by the alphas of
    /// the death times in that interval, a difference of double-double prefix
    /// sums, so short intervals late in follow-up keep their precision.
    pub(super) fn agfit4_gram(
        &self,
        walk: &AgWalk,
        eta: &[f64],
        risk: &[f64],
        workspace: &mut GramWorkspace,
        u: &mut [f64],
        imat: &mut Array2<f64>,
    ) -> Option<f64> {
        let nvar = u.len();
        let row = self.rows();
        if workspace.entry.is_empty() {
            workspace.entry = vec![0.0; self.time.len()];
            for &(entry, person) in &walk.by_entry {
                workspace.entry[person] = entry;
            }
        }
        workspace.gram.clear();
        workspace.death_times.clear();
        workspace.weight.fill(0.0);
        let mut loglik = 0.0;
        for (stratum, &(start, end)) in walk.bounds.iter().enumerate() {
            workspace.risk_set.clear();
            let (by_stop, by_entry) = (&walk.by_stop[start..end], &walk.by_entry[start..end]);
            let (mut joined, mut left) = (0, 0);
            while let Some(next) = by_stop[joined..].iter().position(|j| j.death) {
                let first_death = joined + next;
                let time = by_stop[first_death].time;
                while left < by_entry.len() && by_entry[left].0 >= time {
                    let person = by_entry[left].1;
                    workspace
                        .risk_set
                        .remove(self.weights[person], risk[person], row(person));
                    left += 1;
                }
                while joined < by_stop.len() && by_stop[joined].time >= time {
                    let person = by_stop[joined].row;
                    workspace
                        .risk_set
                        .add(self.weights[person], risk[person], row(person));
                    joined += 1;
                }
                let deaths = || by_stop[first_death..joined].iter().filter(|j| j.death);
                let ndead = deaths().count();
                let tied_sums = self.efron && ndead > 1;
                if tied_sums {
                    workspace.tied.clear();
                }
                let mut death_weight = 0.0;
                for joining in deaths() {
                    let person = joining.row;
                    let (weight, x) = (self.weights[person], row(person));
                    if tied_sums {
                        workspace.tied.add(weight, risk[person], x);
                    }
                    death_weight += weight;
                    loglik += weight * (eta[person] - 0.0);
                    add_scaled(u, weight, x);
                }
                let (set_loglik, alpha, gamma) = self.gram_death_time(
                    &workspace.risk_set,
                    &workspace.tied,
                    ndead,
                    death_weight,
                    u,
                    &mut workspace.gram,
                )?;
                loglik += set_loglik;
                workspace.death_times.push((stratum, time, alpha, gamma));
            }
        }

        // Each row's weight: the alphas of the death times in (entry, stop],
        // with its own death time's gamma in place of that time's alpha.
        let mut taken = 0;
        for (stratum, &(start, end)) in walk.bounds.iter().enumerate() {
            let count = workspace.death_times[taken..]
                .iter()
                .take_while(|time| time.0 == stratum)
                .count();
            let times = &workspace.death_times[taken..taken + count];
            taken += count;
            workspace.ascending.clear();
            workspace.gammas.clear();
            workspace.prefix.clear();
            workspace.prefix.push((0.0, 0.0));
            for &(_, time, alpha, gamma) in times.iter().rev() {
                workspace.ascending.push(time);
                workspace.gammas.push(gamma);
                let &(high, low) = workspace.prefix.last().expect("starts with zero");
                workspace.prefix.push(add_double_double(high, low, alpha));
            }
            let interval = |from: usize, to: usize| {
                let (high_to, low_to) = workspace.prefix[to];
                let (high_from, low_from) = workspace.prefix[from];
                (high_to - high_from) + (low_to - low_from)
            };
            for joining in &walk.by_stop[start..end] {
                let person = joining.row;
                let through = workspace.ascending.partition_point(|&t| t <= joining.time);
                let before = workspace
                    .ascending
                    .partition_point(|&t| t <= workspace.entry[person]);
                let share = if joining.death {
                    interval(before, through - 1) + workspace.gammas[through - 1]
                } else {
                    interval(before, through)
                };
                let omega = risk[person] * share;
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

/// `high + low + value` as a double-double sum (Knuth's two-sum, then a fast
/// renormalization), so prefix sums keep twice the precision of a double.
fn add_double_double(high: f64, low: f64, value: f64) -> (f64, f64) {
    let sum = high + value;
    let virtual_value = sum - high;
    let error = (high - (sum - virtual_value)) + (value - virtual_value);
    let low = low + error;
    let total = sum + low;
    (total, low - (total - sum))
}
