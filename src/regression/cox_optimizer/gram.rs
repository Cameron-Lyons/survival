//! The right-censored Breslow and Efron evaluation with its information built
//! from two weighted Gram matrices.
//!
//! R's `coxfit6` adds each subject's `r x x'` to running risk-set sums and
//! reads them at every death time, `O((n + t) p^2)` scalar work for `n`
//! subjects, `t` death times and `p` covariates. The information needs those
//! sums only through `sum_t sum_j wtave (C_t - f_j C2_t) / d_tj`, which is
//! `sum_i omega_i x_i x_i'`: every death time adds `alpha = sum_j wtave / d_j`
//! to the weight of each subject still at risk, and `gamma = sum_j wtave s_j / d_j`
//! (`s_j = 1 - f_j`) instead to each of its deaths. The mean part is
//! `sum_tj wtave m_tj m_tj'` for the step means `m_tj` that the score already
//! computes. Both are matrix products, which run far faster than the running
//! sums once there are more than a few covariates, and the weights are sums of
//! positive terms. A tie set of several Efron steps contributes two mean rows,
//! the Cholesky factor of its steps' sums ([`EfronStepSums::mean_factor`]),
//! instead of one per step. Rows go through the products in blocks of [`BLOCK`], so the
//! scratch storage does not grow with the data.
//!
//! The log-likelihood and score are those of R's sweep, bit for bit. A sweep
//! that meets an extreme risk or a non-finite weight or information entry
//! returns `None`, and the caller evaluates R's sweep instead, so overflow and
//! missing values behave as before.

use super::*;
use crate::regression::efron_steps::{EfronStepSums, scaled_first};
use ndarray::linalg::general_mat_mul;
use ndarray::s;

/// The fewest covariates for which the Gram evaluation replaces R's sweep.
pub(super) const GRAM_MIN_VARIABLES: usize = 8;

/// Rows per matrix product.
const BLOCK: usize = 4096;

/// Scratch storage of [`CoxData::coxfit6_gram`].
#[derive(Debug)]
pub(crate) struct GramWorkspace {
    risk_set: RiskSetSums,
    tied: RiskSetSums,
    /// Each subject's risk, then its weight `omega`.
    weight: Vec<f64>,
    /// Each time group's `(start, end, stratum, alpha, gamma)`, in sweep order.
    groups: Vec<(usize, usize, usize, f64, f64)>,
    /// Up to [`BLOCK`] step means, row by row, and their weights.
    means: Array2<f64>,
    mean_weights: Vec<f64>,
    scratch: Array2<f64>,
    gram: Array2<f64>,
}

impl GramWorkspace {
    pub(super) fn new(n: usize, nvar: usize) -> Self {
        let block = BLOCK.min(n.max(1));
        Self {
            risk_set: RiskSetSums::zeros(nvar, false),
            tied: RiskSetSums::zeros(nvar, false),
            weight: vec![0.0; n],
            groups: Vec::new(),
            means: Array2::zeros((block, nvar)),
            mean_weights: Vec::with_capacity(block),
            scratch: Array2::zeros((block, nvar)),
            gram: Array2::zeros((nvar, nvar)),
        }
    }

    /// Records the weight of the mean row just written, subtracting the
    /// buffer once it is full.
    fn push_mean(&mut self, weight: f64) {
        self.mean_weights.push(weight);
        if self.mean_weights.len() == self.means.nrows() {
            self.flush_means();
        }
    }

    /// Subtracts the buffered step means' `sum wtave m m'` from the Gram matrix.
    fn flush_means(&mut self) {
        let rows = self.mean_weights.len();
        if rows == 0 {
            return;
        }
        let means = self.means.slice(s![..rows, ..]);
        let mut weighted = self.scratch.slice_mut(s![..rows, ..]);
        for ((mut target, source), &weight) in weighted
            .rows_mut()
            .into_iter()
            .zip(means.rows())
            .zip(&self.mean_weights)
        {
            for (value, &mean) in target.iter_mut().zip(source) {
                *value = weight * mean;
            }
        }
        general_mat_mul(-1.0, &means.t(), &weighted, 1.0, &mut self.gram);
        self.mean_weights.clear();
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
        workspace.mean_weights.clear();
        workspace.gram.fill(0.0);
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
                        let mut mean_row = workspace.means.row_mut(workspace.mean_weights.len());
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
                    workspace.push_mean(wtave);
                } else {
                    // The steps' means span the survivors' and deaths' sums, so
                    // two rows carry their outer products.
                    let ((first, cross), (_, second)) = step_sums.mean_factor()?;
                    let scale = 1.0 / workspace.risk_set.denom;
                    for factor in [(first, cross), (0.0, second)] {
                        let slot = workspace.mean_weights.len();
                        let (risk_set, tied) = (&workspace.risk_set, &workspace.tied);
                        let mut mean_row = workspace.means.row_mut(slot);
                        for (i, mean) in mean_row.iter_mut().enumerate() {
                            let (survivors, deaths) = scaled_first(risk_set.a[i], tied.a[i], scale);
                            *mean = factor.0 * survivors + factor.1 * deaths;
                        }
                        workspace.push_mean(wtave);
                    }
                }
            }
            workspace.groups.push((start, end, stratum, alpha, gamma));
            end = start;
        }
        workspace.flush_means();

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
        let n = self.time.len();
        let mut first = 0;
        while first < n {
            let last = (first + workspace.scratch.nrows()).min(n);
            let rows = self.covar.slice(s![first..last, ..]);
            let mut weighted = workspace.scratch.slice_mut(s![..last - first, ..]);
            for ((mut target, source), &omega) in weighted
                .rows_mut()
                .into_iter()
                .zip(rows.rows())
                .zip(&workspace.weight[first..last])
            {
                for (value, &x) in target.iter_mut().zip(source) {
                    *value = omega * x;
                }
            }
            general_mat_mul(1.0, &rows.t(), &weighted, 1.0, &mut workspace.gram);
            first = last;
        }
        for k in 0..nvar {
            for i in k..nvar {
                imat[(k, i)] = workspace.gram[(k, i)];
            }
        }
        let finite = loglik.is_finite()
            && u.iter().all(|value| value.is_finite())
            && (0..nvar).all(|k| (k..nvar).all(|i| imat[(k, i)].is_finite()));
        finite.then_some(loglik)
    }
}
