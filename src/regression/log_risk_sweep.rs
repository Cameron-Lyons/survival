//! Active risk sets and interval integrals without exponentiating log predictors.
//!
//! Tree nodes retain their own scale. Removing a dominant risk therefore exposes
//! the smaller risks still present, even when a single stratum spans thousands
//! of log units. This path is reserved for extreme inputs; ordinary kernels keep
//! their established arithmetic.

use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::exact_ties::{ExactRiskAccumulator, ExactRiskTree};
use crate::regression::log_scale::LogScale;
use ndarray::{Array2, ArrayView2};

pub(crate) fn needs_log_risk(lp: &[f64], weights: &[f64]) -> bool {
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_weight, mut max_weight) = (f64::INFINITY, 0.0_f64);
    for (&value, &weight) in lp.iter().zip(weights) {
        if !(-200.0..=200.0).contains(&value) || !(1e-100..=1e100).contains(&weight) {
            return true;
        }
        low = low.min(value);
        high = high.max(value);
        min_weight = min_weight.min(weight);
        max_weight = max_weight.max(weight);
    }
    if lp.is_empty() {
        return false;
    }
    let lower = low + min_weight.ln();
    let upper = high + max_weight.ln();
    // Many large rows can erase a small active set when future entrants are
    // subtracted. Include their possible aggregate mass, not only the row span.
    lower < -200.0 || upper > 200.0 || upper - lower + (lp.len() as f64).ln() > 30.0
}

fn weighted_mean(left: f64, right: f64, a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return right;
    }
    if b == 0.0 {
        return left;
    }
    let difference = right - left;
    if !difference.is_finite() {
        return a * left + b * right;
    }
    if a >= b {
        left + b * difference
    } else {
        right - a * difference
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LogMoments {
    pub weight: LogScale,
    pub mean: Vec<f64>,
    degree: f64,
}

impl LogMoments {
    pub fn new(nvar: usize) -> Self {
        Self::new_with_degree(nvar, 1.0)
    }

    pub fn new_with_degree(nvar: usize, degree: f64) -> Self {
        Self {
            weight: LogScale::EMPTY,
            mean: vec![0.0; nvar],
            degree,
        }
    }

    pub fn add(&mut self, weight: LogScale, mean: &[f64]) {
        if weight.is_zero() {
            return;
        }
        if self.weight.is_zero() {
            self.weight = weight;
            self.mean.copy_from_slice(mean);
            return;
        }
        let (total, a, b) = self.weight.mix_degree(weight, self.degree);
        for (left, &right) in self.mean.iter_mut().zip(mean) {
            *left = weighted_mean(*left, right, a, b);
        }
        self.weight = total;
    }
}

/// Log-weighted means, with replacement and interval queries in O(p log n).
#[derive(Debug, Clone)]
pub(crate) struct LogMomentTree {
    base: usize,
    nvar: usize,
    weight: Vec<LogScale>,
    mean: Vec<f64>,
    degree: f64,
}

impl LogMomentTree {
    pub fn new(nrows: usize, nvar: usize) -> Self {
        Self::new_with_degree(nrows, nvar, 1.0)
    }

    pub fn new_with_degree(nrows: usize, nvar: usize, degree: f64) -> Self {
        let base = nrows.max(1).next_power_of_two();
        Self {
            base,
            nvar,
            weight: vec![LogScale::EMPTY; 2 * base],
            mean: vec![0.0; 2 * base * nvar],
            degree,
        }
    }

    pub fn set(&mut self, row: usize, weight: LogScale, mean: &[f64]) {
        let mut node = self.base + row;
        self.weight[node] = weight;
        self.mean[node * self.nvar..(node + 1) * self.nvar].copy_from_slice(mean);
        while node > 1 {
            node /= 2;
            let (left, right) = (2 * node, 2 * node + 1);
            if self.weight[left].is_zero() || self.weight[right].is_zero() {
                let source = if self.weight[left].is_zero() {
                    right
                } else {
                    left
                };
                self.weight[node] = self.weight[source];
                self.mean.copy_within(
                    source * self.nvar..(source + 1) * self.nvar,
                    node * self.nvar,
                );
                continue;
            }
            let (total, a, b) = self.weight[left].mix_degree(self.weight[right], self.degree);
            self.weight[node] = total;
            for j in 0..self.nvar {
                let l = self.mean[left * self.nvar + j];
                let r = self.mean[right * self.nvar + j];
                self.mean[node * self.nvar + j] = weighted_mean(l, r, a, b);
            }
        }
    }

    pub fn weight(&self) -> LogScale {
        self.weight[1]
    }

    pub fn mean(&self) -> &[f64] {
        &self.mean[self.nvar..2 * self.nvar]
    }

    pub fn range(&self, first: usize, end: usize) -> LogMoments {
        let (mut left, mut right) = (self.base + first, self.base + end);
        let mut result = LogMoments::new_with_degree(self.nvar, self.degree);
        while left < right {
            if left % 2 == 1 {
                result.add(
                    self.weight[left],
                    &self.mean[left * self.nvar..(left + 1) * self.nvar],
                );
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                result.add(
                    self.weight[right],
                    &self.mean[right * self.nvar..(right + 1) * self.nvar],
                );
            }
            left /= 2;
            right /= 2;
        }
        result
    }
}

pub(crate) struct LogRiskTime<'a> {
    pub time: f64,
    pub rows_at_time: &'a [usize],
    pub deaths: &'a [usize],
    pub nrisk: usize,
    pub risk_weight: f64,
    pub denom: LogScale,
    pub mean: &'a [f64],
    pub death_risk: LogScale,
    pub nondeath_risk: LogScale,
    pub nondeath_mean: &'a [f64],
    pub death_mean: &'a [f64],
    pub death_weight: f64,
    pub log_death_weight: f64,
    pub covariance: Option<ArrayView2<'a, f64>>,
    pub nondeath_covariance: Option<ArrayView2<'a, f64>>,
    pub death_covariance: Option<ArrayView2<'a, f64>>,
}

impl LogRiskTime<'_> {
    pub fn efron_denom(&self, step: usize) -> LogScale {
        if step == 0 {
            return self.denom;
        }
        self.efron_parts(step).0
    }

    pub fn efron_mean(&self, step: usize, out: &mut [f64]) {
        if step == 0 {
            out.copy_from_slice(self.mean);
            return;
        }
        let (_, a, b) = self.efron_parts(step);
        for ((out, &survivor), &death) in
            out.iter_mut().zip(self.nondeath_mean).zip(self.death_mean)
        {
            *out = weighted_mean(survivor, death, a, b);
        }
    }

    fn efron_parts(&self, step: usize) -> (LogScale, f64, f64) {
        let fraction = step as f64 / self.deaths.len() as f64;
        let death = self.death_risk.shifted((-fraction).ln_1p());
        self.nondeath_risk.mix(death)
    }

    /// Covariance of the positive mixture of survivors and remaining tied deaths.
    pub fn efron_covariance(&self, step: usize, out: &mut Array2<f64>) {
        if step == 0 {
            out.assign(&self.covariance.expect("covariance sweep"));
            return;
        }
        let (_, a, b) = self.efron_parts(step);
        let survivor = self.nondeath_covariance.expect("covariance sweep");
        let death = self.death_covariance.expect("covariance sweep");
        if a == 0.0 {
            out.assign(&death);
            return;
        }
        if b == 0.0 {
            out.assign(&survivor);
            return;
        }
        for i in 0..self.mean.len() {
            for j in 0..self.mean.len() {
                out[(i, j)] = a * survivor[(i, j)]
                    + b * death[(i, j)]
                    + a * b
                        * (self.nondeath_mean[i] - self.death_mean[i])
                        * (self.nondeath_mean[j] - self.death_mean[j]);
            }
        }
    }
}

pub(crate) struct LogRiskSweep<'a> {
    pub stop: &'a [f64],
    pub entry: Option<&'a [f64]>,
    pub status: &'a [i32],
    pub x: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    /// Unweighted log risk, in original row order.
    pub log_risk: &'a [f64],
    /// Original row indices belonging to one stratum, in any order.
    pub rows: &'a [usize],
}

impl LogRiskSweep<'_> {
    pub fn for_each_time(
        &self,
        callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        self.walk(None, false, callback)
    }

    pub fn for_each_time_scales(
        &self,
        log_risk: &[LogScale],
        callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        self.walk(Some(log_risk), false, callback)
    }

    pub fn for_each_time_with_covariance(
        &self,
        callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        self.walk(None, true, callback)
    }

    pub fn for_each_time_with_covariance_scales(
        &self,
        log_risk: &[LogScale],
        callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        self.walk(Some(log_risk), true, callback)
    }

    fn walk(
        &self,
        scales: Option<&[LogScale]>,
        covariance: bool,
        mut callback: impl FnMut(LogRiskTime<'_>) -> SurvivalResult<()>,
    ) -> SurvivalResult<()> {
        let n = self.rows.len();
        let p = self.x.ncols();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| self.stop[self.rows[a]].total_cmp(&self.stop[self.rows[b]]));
        let by_start = self.entry.map(|entry| {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| entry[self.rows[a]].total_cmp(&entry[self.rows[b]]));
            order
        });
        let mut risk = LogMomentTree::new(n, p);
        let mut weight = LogMomentTree::new(n, 0);
        let x: Vec<Vec<f64>> = self
            .rows
            .iter()
            .map(|&row| self.x.row(row).to_vec())
            .collect();
        let covariance_x = covariance.then(|| Array2::from_shape_fn((n, p), |(i, j)| x[i][j]));
        let mut covariance_tree = covariance.then(|| ExactRiskTree::new(0, n, p));
        let mut log_risk = Vec::with_capacity(n);
        for (position, &row) in self.rows.iter().enumerate() {
            let predictor = scales.map_or_else(
                || LogScale::from_log(self.log_risk[row]),
                |values| values[row],
            );
            if !predictor.origin.is_finite()
                || !predictor.origin_correction.is_finite()
                || !predictor.log_mass.is_finite()
                || !self.weights[row].is_finite()
                || self.weights[row] < 0.0
            {
                return Err(SurvivalError::invalid_input(
                    "Cox log risks and weights must be finite, with nonnegative weights",
                ));
            }
            log_risk.push(predictor.shifted(self.weights[row].ln()));
            if self.entry.is_none() {
                risk.set(position, log_risk[position], &x[position]);
                weight.set(position, LogScale::from_log(self.weights[row].ln()), &[]);
                if let Some(tree) = &mut covariance_tree {
                    tree.set_active(position, true);
                }
            }
        }
        let mut nrisk = if self.entry.is_none() { n } else { 0 };
        let (mut entered, mut departed, mut first) = (0, 0, 0);
        while first < n {
            let time = self.stop[self.rows[order[first]]];
            let mut end = first + 1;
            while end < n && self.stop[self.rows[order[end]]] == time {
                end += 1;
            }
            if let (Some(entry), Some(by_start)) = (self.entry, &by_start) {
                while entered < n && entry[self.rows[by_start[entered]]] < time {
                    let position = by_start[entered];
                    risk.set(position, log_risk[position], &x[position]);
                    weight.set(
                        position,
                        LogScale::from_log(self.weights[self.rows[position]].ln()),
                        &[],
                    );
                    if let Some(tree) = &mut covariance_tree {
                        tree.set_active(position, true);
                    }
                    nrisk += 1;
                    entered += 1;
                }
            }
            while departed < n && self.stop[self.rows[order[departed]]] < time {
                let position = order[departed];
                risk.set(position, LogScale::EMPTY, &x[position]);
                weight.set(position, LogScale::EMPTY, &[]);
                if let Some(tree) = &mut covariance_tree {
                    tree.set_active(position, false);
                }
                nrisk -= 1;
                departed += 1;
            }
            let rows_at_time: Vec<usize> = order[first..end]
                .iter()
                .map(|&position| self.rows[position])
                .collect();
            let mut deaths = Vec::new();
            let mut death_risk = LogMoments::new(p);
            let mut death_weight = LogMoments::new(0);
            let mut death_covariance = covariance.then(|| ExactRiskAccumulator::new(p));
            for &position in &order[first..end] {
                let row = self.rows[position];
                if self.status[row] == 1 {
                    deaths.push(row);
                    death_risk.add(log_risk[position], &x[position]);
                    death_weight.add(LogScale::from_log(self.weights[row].ln()), &[]);
                    if let (Some(moment), Some(x)) = (&mut death_covariance, &covariance_x) {
                        moment.add_scale(position, log_risk[position], x);
                    }
                }
            }
            let live_covariance = match (&mut covariance_tree, &covariance_x) {
                (Some(tree), Some(x)) => {
                    tree.refresh_scales(&log_risk, x);
                    Some(tree.covariance().to_owned())
                }
                _ => None,
            };
            // K-P survival increments need the surviving risk even when the
            // deaths dominate the total beyond floating-point precision.
            for &position in &order[first..end] {
                if self.status[self.rows[position]] == 1 {
                    risk.set(position, LogScale::EMPTY, &x[position]);
                    if let Some(tree) = &mut covariance_tree {
                        tree.set_active(position, false);
                    }
                }
            }
            let nondeath_risk = risk.weight();
            let nondeath_mean = risk.mean().to_vec();
            let nondeath_covariance = match (&mut covariance_tree, &covariance_x) {
                (Some(tree), Some(x)) => {
                    tree.refresh_scales(&log_risk, x);
                    Some(tree.covariance().to_owned())
                }
                _ => None,
            };
            for &position in &order[first..end] {
                if self.status[self.rows[position]] == 1 {
                    risk.set(position, log_risk[position], &x[position]);
                    if let Some(tree) = &mut covariance_tree {
                        tree.set_active(position, true);
                    }
                }
            }
            if risk.weight().is_zero() && !death_weight.weight.is_zero() {
                return Err(SurvivalError::invalid_input(
                    "positive-weight events require a positive risk-set sum",
                ));
            }
            callback(LogRiskTime {
                time,
                rows_at_time: &rows_at_time,
                deaths: &deaths,
                nrisk,
                risk_weight: weight.weight().exp_at(0.0),
                denom: risk.weight(),
                mean: risk.mean(),
                death_risk: death_risk.weight,
                nondeath_risk,
                nondeath_mean: &nondeath_mean,
                death_mean: &death_risk.mean,
                death_weight: death_weight.weight.exp_at(0.0),
                log_death_weight: death_weight.weight.log_value(),
                covariance: live_covariance.as_ref().map(|value| value.view()),
                nondeath_covariance: nondeath_covariance.as_ref().map(|value| value.view()),
                death_covariance: death_covariance
                    .as_ref()
                    .map(|value| value.covariance.view()),
            })?;
            first = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removal_and_interval_queries_recover_tiny_remaining_moments() {
        let mut tree = LogMomentTree::new(4, 1);
        for (row, (weight, mean)) in [(1000., 2.), (-1000., -1.), (999., 3.), (-999., 1.)]
            .into_iter()
            .enumerate()
        {
            tree.set(row, LogScale::from_log(weight), &[mean]);
        }
        let alone = tree.range(1, 2);
        assert_eq!(alone.weight.log_value(), -1000.);
        assert_eq!(alone.mean, [-1.]);
        tree.set(0, LogScale::EMPTY, &[2.]);
        tree.set(2, LogScale::EMPTY, &[3.]);
        let expected = (-1_f64).exp();
        assert!((tree.weight().log_value() - (-999. + expected.ln_1p())).abs() < 1e-12);
        assert!((tree.mean()[0] - (1. - expected) / (1. + expected)).abs() < 1e-15);
    }

    #[test]
    fn efron_moments_mix_survivors_positively_when_deaths_dominate() {
        let deaths = vec![0; 1000];
        let moment = LogRiskTime {
            time: 1.,
            rows_at_time: &deaths,
            deaths: &deaths,
            nrisk: 1001,
            risk_weight: 1001.,
            denom: LogScale::from_log((-30_f64).exp().ln_1p()),
            mean: &[1.],
            death_risk: LogScale::from_log(0.),
            nondeath_risk: LogScale::from_log(-30.),
            nondeath_mean: &[-1.],
            death_mean: &[1.],
            death_weight: 1000.,
            log_death_weight: 1000_f64.ln(),
            covariance: None,
            nondeath_covariance: None,
            death_covariance: None,
        };
        let mut mean = [0.];
        moment.efron_mean(999, &mut mean);
        let survivor = (-30_f64).exp();
        let death = 1. - 999. / 1000.;
        assert!((mean[0] - (death - survivor) / (death + survivor)).abs() < 2e-16);
        assert!((moment.efron_denom(999).log_value() - (death + survivor).ln()).abs() < 1e-15);
    }

    #[test]
    fn ordinary_preflight_uses_bounds_without_exponentiating_rows() {
        assert!(!needs_log_risk(&[-2., 0., 3.], &[1., 0.5, 2.]));
        assert!(needs_log_risk(&[-100., 100.], &[1., 1.]));
        assert!(needs_log_risk(&[1000., 1000.], &[1., 1.]));
        assert!(needs_log_risk(&[0., 0.], &[1e-100, 1e100]));
        assert!(!needs_log_risk(&[], &[]));
    }

    #[test]
    fn finite_opposite_covariate_means_do_not_overflow_their_difference() {
        let mut moments = LogMoments::new(1);
        moments.add(LogScale::from_log(1000.), &[1e308]);
        moments.add(LogScale::from_log(1000.), &[-1e308]);
        assert_eq!(moments.mean, [0.]);
    }

    #[test]
    fn dynamic_origins_preserve_multiplicity_weights_and_squared_integrals() {
        for origin in [1e16, -1e16, 1e308, -1e308] {
            let mut tree = LogMomentTree::new(3, 1);
            for (row, (weight, mean)) in [(1_f64, 0.), (3., 2.), (2., 5.)].into_iter().enumerate() {
                tree.set(row, LogScale::new(origin, weight.ln()), &[mean]);
            }
            assert!((tree.weight().exp_at(-origin) - 6.).abs() < 1e-14);
            assert!((tree.mean()[0] - 16. / 6.).abs() < 1e-14);
            assert!((tree.range(0, 2).mean[0] - 1.5).abs() < 1e-14);
            tree.set(2, LogScale::EMPTY, &[5.]);
            assert!((tree.weight().exp_at(-origin) - 4.).abs() < 1e-14);

            let mut integrals = LogMomentTree::new_with_degree(2, 0, 2.);
            integrals.set(0, LogScale::new(-origin, 0.5_f64.ln()), &[]);
            integrals.set(1, LogScale::new(-origin, 0.25_f64.ln()), &[]);
            assert!((integrals.range(0, 2).weight.exp_at_degree(origin, 2.) - 0.75).abs() < 1e-15);
        }
    }

    #[test]
    fn compensated_origins_do_not_absorb_small_risk_masses() {
        let predictor = LogScale::from_predictor_parts(1e308, 1e292);
        let inverse = LogScale::from_predictor_parts(-1e308, -1e292);
        let mut tree = LogMomentTree::new(3, 1);
        for (row, (weight, mean)) in [(1_f64, 0.), (3., 2.), (2., 5.)].into_iter().enumerate() {
            tree.set(row, predictor.shifted(weight.ln()), &[mean]);
        }
        assert!((tree.weight().exp_at_scale(inverse) - 6.).abs() < 1e-14);
        assert!((tree.mean()[0] - 16. / 6.).abs() < 1e-14);
        let increment = tree.weight().reciprocal(0.);
        let mut variance = LogMomentTree::new_with_degree(2, 0, 2.);
        let square = LogScale::new_parts(
            increment.origin,
            increment.origin_correction,
            2. * increment.log_mass,
        );
        variance.set(0, square, &[]);
        variance.set(1, square, &[]);
        assert!((variance.weight().exp_at_scale_degree(predictor, 2.) - 2. / 36.).abs() < 1e-15);
    }
}
