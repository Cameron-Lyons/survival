//! Stable penalized likelihoods without a dense frailty indicator matrix.
//!
//! Dense moments use the shared risk sweep. A frailty group's risk and mean
//! stay fixed between its entries/exits; interval queries over the accumulated
//! hazard and variance integrate its score and cross information lazily.

use super::Kernel;
use crate::error::SurvivalResult;
use crate::regression::cox_optimizer::{
    TieMethod, add_predictor_parts, bounds_need_anchor, predictor_bound_needs_anchor,
    relative_covariate_scale,
};
use crate::regression::log_risk_sweep::{LogMomentTree, LogMoments, LogRiskSweep};
use crate::regression::log_scale::LogScale;
use crate::residuals::martingale_log::martingale_log_scales;
use ndarray::Array2;
use std::collections::BTreeMap;

impl Kernel {
    pub(super) fn needs_stable(&self, beta: &[f64], fbeta: &[f64]) -> bool {
        self.unsafe_offsets
            || bounds_need_anchor(&self.covar_bounds, beta)
            || fbeta
                .iter()
                .any(|value| predictor_bound_needs_anchor(value.abs()))
    }

    fn paired_scores(&self, beta: &[f64], fbeta: &[f64]) -> Vec<LogScale> {
        let anchors = self.predictor_anchors.get_or_init(|| {
            let mut anchors = Vec::with_capacity(self.strata_end.len());
            let mut start = 0;
            for &end in &self.strata_end {
                anchors.push(
                    *self.sort1[start..end]
                        .iter()
                        .max_by(|&&a, &&b| self.stop[a].total_cmp(&self.stop[b]).then(a.cmp(&b)))
                        .expect("nonempty stratum"),
                );
                start = end;
            }
            anchors
        });
        let mut scores = vec![LogScale::EMPTY; self.n];
        let mut start = 0;
        for (&end, &anchor) in self.strata_end.iter().zip(anchors) {
            let anchor = self.covar.row(anchor).to_vec();
            for &row in &self.sort1[start..end] {
                let covariate = relative_covariate_scale(self.covar.row(row), &anchor, beta);
                let mut score =
                    add_predictor_parts(LogScale::from_log(self.offset[row]), covariate);
                if let Some(frail) = &self.frail {
                    score = add_predictor_parts(score, LogScale::from_log(fbeta[frail[row]]));
                }
                scores[row] = score;
            }
            start = end;
        }
        scores
    }

    pub(super) fn evaluate_stable(
        &self,
        beta: &[f64],
        fbeta: &[f64],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> SurvivalResult<(f64, Vec<LogScale>)> {
        let scores = self.paired_scores(beta, fbeta);
        let mut loglik = 0.0;
        let mut mean = vec![0.0; self.nvar];
        let mut covariance = Array2::zeros((self.nvar, self.nvar));
        let mut positions = vec![(0, 0); if self.nf > 0 { self.n } else { 0 }];
        let mut start = 0;
        for &end in &self.strata_end {
            let rows = &self.sort1[start..end];
            let mut sparse =
                (self.nf > 0).then(|| SparseRisk::new(self, rows, &scores, &mut positions));
            let sweep = LogRiskSweep {
                stop: &self.stop,
                entry: self.start.as_deref(),
                status: &self.status,
                x: self.covar.view(),
                weights: &self.weights,
                log_risk: &[],
                rows,
            };
            sweep.for_each_time_with_covariance_scales(&scores, |event| {
                if let Some(sparse) = &mut sparse {
                    sparse.update(event.time, u, jmat, fdiag);
                }
                let Some(&first) = event.deaths.first() else {
                    return Ok(());
                };
                let reference = event.deaths.iter().fold(scores[first], |reference, &row| {
                    if scores[row].ratio_log(reference) > 0.0 {
                        scores[row]
                    } else {
                        reference
                    }
                });
                for &row in event.deaths {
                    let weight = self.weights[row];
                    loglik += weight * scores[row].ratio_log(reference);
                    if let Some(frail) = &self.frail {
                        u[frail[row]] += weight;
                    }
                    for (i, value) in u[self.nf..].iter_mut().enumerate() {
                        *value += weight * self.covar[(row, i)];
                    }
                }
                let death_groups = sparse
                    .as_mut()
                    .map(|sparse| sparse.remove_deaths(event.deaths, u, jmat, fdiag));
                let steps = if self.method == 1.0 {
                    event.deaths.len()
                } else {
                    1
                };
                let average_weight = event.death_weight / steps as f64;
                let log_average_weight = event.log_death_weight - (steps as f64).ln();
                let mut block = sparse.as_ref().map(|_| TieBlock::new(self.nvar));
                for step in 0..steps {
                    let denom = event.efron_denom(step);
                    loglik += average_weight * reference.ratio_log(denom);
                    event.efron_mean(step, &mut mean);
                    event.efron_covariance(step, &mut covariance);
                    for (i, value) in u[self.nf..].iter_mut().enumerate() {
                        *value -= average_weight * mean[i];
                        for j in 0..=i {
                            jmat[(i, j + self.nf)] += average_weight * covariance[(i, j)];
                        }
                    }
                    if let (Some(sparse), Some(block)) = (&mut sparse, &mut block) {
                        let remaining =
                            (event.deaths.len() - step) as f64 / event.deaths.len() as f64;
                        block.add(denom, log_average_weight, remaining.ln(), &mean);
                        sparse.push_increment(denom, log_average_weight, &mean);
                    }
                }
                if let (Some(sparse), Some(deaths), Some(block)) =
                    (&mut sparse, &death_groups, &block)
                {
                    sparse.death_block(deaths, block, u, jmat, fdiag);
                    for &group in deaths.keys() {
                        sparse.groups[group].last = sparse.index;
                    }
                }
                Ok(())
            })?;
            if let Some(sparse) = &mut sparse {
                for group in 0..sparse.groups.len() {
                    sparse.flush(group, u, jmat, fdiag);
                }
            }
            start = end;
        }
        Ok((loglik, scores))
    }

    pub(super) fn stable_expected_events(&self, scores: &[LogScale]) -> SurvivalResult<Vec<f64>> {
        let mut strata = vec![0; self.n];
        let mut start = 0;
        for (group, &end) in self.strata_end.iter().enumerate() {
            for &row in &self.sort1[start..end] {
                strata[row] = group as i32;
            }
            start = end;
        }
        let method = if self.method == 1.0 {
            TieMethod::Efron
        } else {
            TieMethod::Breslow
        };
        let residuals = martingale_log_scales(
            self.start.as_deref(),
            &self.stop,
            &self.status,
            scores,
            &self.weights,
            &strata,
            method,
        )?;
        Ok(self
            .status
            .iter()
            .zip(residuals)
            .map(|(&event, residual)| f64::from(event) - residual)
            .collect())
    }
}

/// H, rH and V, rV, r²V integrate an Efron block once. Every touched group
/// then combines its nondeath and death risks in O(p), independent of the
/// number of tied deaths.
struct TieBlock {
    hazards: [LogMoments; 2],
    variances: [LogMoments; 3],
}

impl TieBlock {
    fn new(nvar: usize) -> Self {
        Self {
            hazards: std::array::from_fn(|_| LogMoments::new(nvar)),
            variances: std::array::from_fn(|_| LogMoments::new_with_degree(0, 2.0)),
        }
    }

    fn add(&mut self, denom: LogScale, log_average: f64, log_remaining: f64, mean: &[f64]) {
        let hazard = denom.reciprocal(log_average);
        let variance = LogScale::new_parts(
            -denom.origin,
            -denom.origin_correction,
            log_average - 2.0 * denom.log_mass,
        );
        for (degree, moment) in self.hazards.iter_mut().enumerate() {
            moment.add(hazard.shifted(degree as f64 * log_remaining), mean);
        }
        for (degree, moment) in self.variances.iter_mut().enumerate() {
            moment.add(variance.shifted(degree as f64 * log_remaining), &[]);
        }
    }
}

/// log(A V / H), with degree-two V. Cancel denominator origins before
/// combining their logarithms, including when a tiny group's risk vanishes.
fn variance_hazard_ratio(variance: LogScale, hazard: LogScale, risk: LogScale) -> f64 {
    if risk.is_zero() {
        return f64::NEG_INFINITY;
    }
    ((variance.origin - hazard.origin) + (variance.origin + risk.origin))
        + ((variance.origin_correction - hazard.origin_correction)
            + (variance.origin_correction + risk.origin_correction))
        + ((variance.log_mass - hazard.log_mass) + risk.log_mass)
}

fn probability_complement(left: f64, right: f64) -> f64 {
    let probability = LogScale::from_log(left)
        .mix(LogScale::from_log(right))
        .0
        .log_value();
    -probability.min(0.0).exp_m1()
}

struct Group {
    label: usize,
    risk: LogMomentTree,
    last: usize,
}

struct SparseRisk<'a> {
    kernel: &'a Kernel,
    scores: &'a [LogScale],
    positions: &'a [(usize, usize)],
    groups: Vec<Group>,
    by_stop: Vec<usize>,
    by_start: Vec<usize>,
    entered: usize,
    departed: usize,
    index: usize,
    hazards: LogMomentTree,
    variances: LogMomentTree,
    scratch: Vec<f64>,
}

impl<'a> SparseRisk<'a> {
    fn new(
        kernel: &'a Kernel,
        rows: &[usize],
        scores: &'a [LogScale],
        positions: &'a mut [(usize, usize)],
    ) -> Self {
        let frail = kernel.frail.as_ref().expect("sparse groups");
        let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
        for &row in rows {
            grouped.entry(frail[row]).or_default().push(row);
        }
        let mut groups = Vec::with_capacity(grouped.len());
        for (label, members) in grouped {
            for (position, &row) in members.iter().enumerate() {
                positions[row] = (groups.len(), position);
            }
            groups.push(Group {
                label,
                risk: LogMomentTree::new(members.len(), kernel.nvar),
                last: 0,
            });
        }
        let mut by_stop = rows.to_vec();
        by_stop.sort_by(|&a, &b| kernel.stop[a].total_cmp(&kernel.stop[b]));
        let mut by_start = rows.to_vec();
        if let Some(start) = &kernel.start {
            by_start.sort_by(|&a, &b| start[a].total_cmp(&start[b]));
        }
        let mut state = Self {
            kernel,
            scores,
            positions,
            groups,
            by_stop,
            by_start,
            entered: 0,
            departed: 0,
            index: 0,
            hazards: LogMomentTree::new(rows.len(), kernel.nvar),
            variances: LogMomentTree::new_with_degree(rows.len(), 0, 2.0),
            scratch: vec![0.0; kernel.nvar],
        };
        if kernel.start.is_none() {
            for &row in rows {
                state.set(row, scores[row].shifted(kernel.weights[row].ln()));
            }
        }
        state
    }

    fn set(&mut self, row: usize, weight: LogScale) {
        let (group, position) = self.positions[row];
        for (target, &x) in self.scratch.iter_mut().zip(self.kernel.covar.row(row)) {
            *target = x;
        }
        self.groups[group].risk.set(position, weight, &self.scratch);
    }

    fn flush(&mut self, group: usize, u: &mut [f64], jmat: &mut Array2<f64>, fdiag: &mut [f64]) {
        let group = &mut self.groups[group];
        if group.last == self.index {
            return;
        }
        let risk = group.risk.weight();
        if !risk.is_zero() {
            let hazard = self.hazards.range(group.last, self.index);
            let variance = self.variances.range(group.last, self.index);
            let expected = hazard.weight.exp_at_scale(risk);
            if expected == 0.0 {
                group.last = self.index;
                return;
            }
            u[group.label] -= expected;
            // A²V/(AH) is at most one. Cancel the risk origins before
            // subtracting the moments, retaining a tiny positive 1-P when
            // the group accounts for almost all of a risk set.
            let log_ratio = variance_hazard_ratio(variance.weight, hazard.weight, risk);
            fdiag[group.label] += expected * (-log_ratio.min(0.0).exp_m1());
            for (i, (&mean, &overall)) in group.risk.mean().iter().zip(&hazard.mean).enumerate() {
                jmat[(i, group.label)] += expected * (mean - overall);
            }
        }
        group.last = self.index;
    }

    fn update(&mut self, time: f64, u: &mut [f64], jmat: &mut Array2<f64>, fdiag: &mut [f64]) {
        if let Some(start) = &self.kernel.start {
            while self.entered < self.by_start.len() && start[self.by_start[self.entered]] < time {
                let row = self.by_start[self.entered];
                self.flush(self.positions[row].0, u, jmat, fdiag);
                self.set(row, self.scores[row].shifted(self.kernel.weights[row].ln()));
                self.entered += 1;
            }
        }
        while self.departed < self.by_stop.len()
            && self.kernel.stop[self.by_stop[self.departed]] < time
        {
            let row = self.by_stop[self.departed];
            self.flush(self.positions[row].0, u, jmat, fdiag);
            self.set(row, LogScale::EMPTY);
            self.departed += 1;
        }
    }

    fn remove_deaths(
        &mut self,
        deaths: &[usize],
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) -> BTreeMap<usize, LogMoments> {
        let mut groups = BTreeMap::<usize, LogMoments>::new();
        for &row in deaths {
            let group = self.positions[row].0;
            self.flush(group, u, jmat, fdiag);
            self.set(row, LogScale::EMPTY);
            groups
                .entry(group)
                .or_insert_with(|| LogMoments::new(self.kernel.nvar))
                .add(
                    self.scores[row].shifted(self.kernel.weights[row].ln()),
                    &self.scratch,
                );
        }
        groups
    }

    fn death_block(
        &mut self,
        deaths: &BTreeMap<usize, LogMoments>,
        block: &TieBlock,
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) {
        for (&index, death) in deaths {
            let group = &self.groups[index];
            let nondeath = group.risk.weight();
            let risks = [nondeath, death.weight];
            let means = [group.risk.mean(), death.mean.as_slice()];
            for part in 0..2 {
                let hazard = &block.hazards[part];
                let expected = hazard.weight.exp_at_scale(risks[part]);
                if expected == 0.0 {
                    continue;
                }
                u[group.label] -= expected;
                let nondeath_probability =
                    variance_hazard_ratio(block.variances[part].weight, hazard.weight, nondeath);
                let death_probability = variance_hazard_ratio(
                    block.variances[part + 1].weight,
                    hazard.weight,
                    death.weight,
                );
                fdiag[group.label] +=
                    expected * probability_complement(nondeath_probability, death_probability);
                for (i, (&mean, &overall)) in means[part].iter().zip(&hazard.mean).enumerate() {
                    jmat[(i, group.label)] += expected * (mean - overall);
                }
            }
        }
    }

    fn push_increment(&mut self, denom: LogScale, log_average: f64, mean: &[f64]) {
        self.hazards
            .set(self.index, denom.reciprocal(log_average), mean);
        self.variances.set(
            self.index,
            LogScale::new_parts(
                -denom.origin,
                -denom.origin_correction,
                log_average - 2.0 * denom.log_mass,
            ),
            &[],
        );
        self.index += 1;
    }
}
