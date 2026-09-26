//! The backward risk-set sweep shared by the Cox kernels that work one
//! death time at a time.
//!
//! R survival's (start, stop] kernels (`agfit4.c`, `zph2.c`) keep one
//! running risk set per stratum: walking the death times from the largest
//! down, the rows whose entry time is at or after the death time leave the
//! set and then the rows whose stop time is at or after it join.
//! [`RiskSetSums`] holds those running sums and resets them to exactly zero
//! whenever the set empties (at every cut point of `survSplit` data), so
//! rounding error never outlives a risk set.  [`RecenteredRiskSet`] adds
//! `agfit4.c`'s rescaling of the risk scores for the Newton iteration of
//! `coxph`.
//!
//! [`StratumSweep`] walks one stratum for `coxscho.c`, `coxdetail.c` and
//! `zph1.c`/`zph2.c` (the C behind `residuals(type = "schoenfeld")`,
//! `coxph.detail` and `cox.zph`), for right-censored and (start, stop] data
//! alike, and hands each death time's sums to the kernel, which applies its
//! own Breslow or Efron arithmetic.  Death times are visited in decreasing
//! order; kernels needing ascending output reverse afterwards.

use crate::error::{SurvivalError, SurvivalResult};
use ndarray::ArrayView2;

/// Weighted sums over a set of rows.
#[derive(Debug, Clone)]
pub(crate) struct RiskSetSums {
    /// Number of rows.
    pub count: usize,
    /// `sum w`.
    pub weight: f64,
    /// `sum w r` (risk `r = exp(lp)`).
    pub denom: f64,
    /// `sum w r x`.
    pub a: Vec<f64>,
    /// `sum w r x x'`, row-major `nvar x nvar` (lower triangle); empty
    /// unless second moments were requested.
    pub cmat: Vec<f64>,
}

impl RiskSetSums {
    pub(crate) fn zeros(nvar: usize, second_moments: bool) -> Self {
        Self {
            count: 0,
            weight: 0.0,
            denom: 0.0,
            a: vec![0.0; nvar],
            cmat: if second_moments {
                vec![0.0; nvar * nvar]
            } else {
                Vec::new()
            },
        }
    }

    /// Empties the set, every sum exactly zero.
    pub(crate) fn clear(&mut self) {
        self.count = 0;
        self.weight = 0.0;
        self.denom = 0.0;
        self.a.fill(0.0);
        self.cmat.fill(0.0);
    }

    /// Adds a row with case weight `weight`, weighted risk `risk = w r` and
    /// covariates `x`.
    #[inline]
    pub(crate) fn add(&mut self, weight: f64, risk: f64, x: &[f64]) {
        self.count += 1;
        self.weight += weight;
        self.accumulate(risk, x);
    }

    /// Removes a row added with the same `risk`.  Removing the last row
    /// resets the sums to exactly zero instead (`agfit4.c`), so no
    /// cancellation error carries over to the next risk set.
    #[inline]
    pub(crate) fn remove(&mut self, weight: f64, risk: f64, x: &[f64]) {
        self.count -= 1;
        if self.count == 0 {
            self.clear();
        } else {
            self.weight -= weight;
            self.accumulate(-risk, x);
        }
    }

    #[inline]
    fn accumulate(&mut self, risk: f64, x: &[f64]) {
        self.denom += risk;
        let nvar = x.len();
        let second_moments = !self.cmat.is_empty();
        for (i, (a, &xi)) in self.a.iter_mut().zip(x).enumerate() {
            let risk_xi = risk * xi;
            *a += risk_xi;
            if second_moments {
                for (c, &xj) in self.cmat[i * nvar..=i * nvar + i].iter_mut().zip(x) {
                    *c += risk_xi * xj;
                }
            }
        }
    }

    /// Multiplies the risk-weighted sums by `factor`.
    fn rescale(&mut self, factor: f64) {
        self.denom *= factor;
        for sum in self.a.iter_mut().chain(&mut self.cmat) {
            *sum *= factor;
        }
    }
}

/// A row of a [`RecenteredRiskSet`]: its linear predictor and case weight,
/// and its risk score about the centre of `epoch`.
#[derive(Debug, Clone, Copy, Default)]
struct Member {
    eta: f64,
    weight: f64,
    risk: f64,
    epoch: u64,
}

/// The running risk set of `agfit4.c`'s likelihood evaluation, with second
/// moments: sums of the weighted risk scores `w exp(eta - recenter)`.  The
/// centre follows the mean linear predictor of the set, so `exp` neither
/// overflows nor underflows the whole set away when the linear predictors
/// are large (an offset of -750, or near-infinite coefficients).  It
/// cancels from the likelihood, score and information as long as the
/// deaths' own `eta` terms are shifted by it too.
///
/// Rows are identified by an index below the `nrow` given to
/// [`RecenteredRiskSet::restart`]; the set remembers what each row joined
/// with, so its risk score is computed once unless the centre moves.
#[derive(Debug)]
pub(crate) struct RecenteredRiskSet {
    pub sums: RiskSetSums,
    /// `sum eta` over the set.
    etasum: f64,
    /// The constant subtracted from every linear predictor.
    pub recenter: f64,
    /// Number of times the centre moved, over every evaluation
    /// (`agreg.fit`'s `info["rescale"]`).
    pub rescales: i32,
    /// Advances whenever the centre is set.
    epoch: u64,
    members: Vec<Member>,
}

impl RecenteredRiskSet {
    pub(crate) fn new(nvar: usize) -> Self {
        Self {
            sums: RiskSetSums::zeros(nvar, true),
            etasum: 0.0,
            recenter: 0.0,
            rescales: 0,
            epoch: 0,
            members: Vec::new(),
        }
    }

    /// Starts an evaluation of the likelihood over rows `0..nrow`: an empty
    /// set centred at 0.
    pub(crate) fn restart(&mut self, nrow: usize) {
        self.clear();
        self.recenter = 0.0;
        self.epoch += 1;
        self.members.resize(nrow, Member::default());
    }

    /// Empties the set for a new stratum; as in `agfit4.c` the centre
    /// carries over.
    pub(crate) fn clear(&mut self) {
        self.sums.clear();
        self.etasum = 0.0;
    }

    /// The weighted risk score `w exp(eta - recenter)` of a row in the set.
    #[inline]
    pub(crate) fn risk(&mut self, row: usize) -> f64 {
        let (recenter, epoch) = (self.recenter, self.epoch);
        let member = &mut self.members[row];
        if member.epoch != epoch {
            member.risk = (member.eta - recenter).exp() * member.weight;
            member.epoch = epoch;
        }
        member.risk
    }

    /// Adds a row.  When the mean linear predictor of the set, the row
    /// included, lies more than 200 from the centre, the centre moves to it
    /// and the sums are rescaled first; a move beyond `exp`'s range while
    /// rows are at risk is `agfit4.c`'s overflow error.
    #[inline]
    pub(crate) fn add(
        &mut self,
        row: usize,
        eta: f64,
        weight: f64,
        x: &[f64],
    ) -> SurvivalResult<()> {
        self.etasum += eta;
        let mean = self.etasum / (self.sums.count + 1) as f64;
        if (mean - self.recenter).abs() > 200.0 {
            let shift = mean - self.recenter;
            self.recenter = mean;
            self.rescales += 1;
            self.epoch += 1;
            if self.sums.denom > 0.0 {
                if shift.abs() > 709.0 {
                    return Err(SurvivalError::computation("exp overflow due to covariates"));
                }
                self.sums.rescale((-shift).exp());
            }
        }
        let risk = (eta - self.recenter).exp() * weight;
        self.members[row] = Member {
            eta,
            weight,
            risk,
            epoch: self.epoch,
        };
        self.sums.add(weight, risk, x);
        Ok(())
    }

    /// Removes a row of the set (see [`RiskSetSums::remove`]).
    #[inline]
    pub(crate) fn remove(&mut self, row: usize, x: &[f64]) {
        let risk = self.risk(row);
        let Member { eta, weight, .. } = self.members[row];
        self.sums.remove(weight, risk, x);
        self.etasum = if self.sums.count == 0 {
            0.0
        } else {
            self.etasum - eta
        };
    }
}

/// Whether each row of a stratum (`rows`, sorted by ascending stop time)
/// has one of the stratum's death times in its interval `(entry, stop]`,
/// `agreg.fit`'s `!ignore`.  Only those rows ever belong to a death time's
/// risk set.  A sweep that walks them alone removes only rows it has added:
/// a row that entered at or after a death time spans a later death time, so
/// it joined the set before the sweep reached this one.
pub(crate) fn spans_a_death(
    rows: &[usize],
    stop: &[f64],
    entry: &[f64],
    status: &[i32],
) -> Vec<bool> {
    let mut spans = vec![false; rows.len()];
    // The largest death time at or before the current stop time.
    let mut last_death = f64::NEG_INFINITY;
    let mut end = 0;
    while end < rows.len() {
        let start = end;
        let time = stop[rows[start]];
        while end < rows.len() && stop[rows[end]] == time {
            end += 1;
        }
        if rows[start..end].iter().any(|&row| status[row] == 1) {
            last_death = time;
        }
        for (spans, &row) in spans[start..end].iter_mut().zip(&rows[start..end]) {
            *spans = entry[row] < last_death;
        }
    }
    spans
}

/// One death time's risk set and its tied deaths.
#[derive(Debug)]
pub(crate) struct DeathTime<'a> {
    pub time: f64,
    /// Original row indices of the deaths at this time, in sorted order.
    pub deaths: &'a [usize],
    /// Sums over the whole risk set (`entry < time <= stop`), deaths included.
    pub risk_set: &'a RiskSetSums,
    /// Sums over the deaths only.
    pub tied: &'a RiskSetSums,
}

impl DeathTime<'_> {
    pub fn ndead(&self) -> usize {
        self.deaths.len()
    }

    /// Efron's `j`-th denominator `denom - j/d * denom2`.
    pub fn efron_denom(&self, j: usize) -> f64 {
        self.risk_set.denom - self.tied.denom * j as f64 / self.ndead() as f64
    }

    /// Efron's `j`-th first-moment sum `a - j/d * a2`.
    pub fn efron_a(&self, j: usize, i: usize) -> f64 {
        self.risk_set.a[i] - self.tied.a[i] * j as f64 / self.ndead() as f64
    }

    /// Efron's `j`-th second-moment sum `cmat - j/d * cmat2` (lower triangle).
    pub fn efron_cmat(&self, j: usize, i: usize, k: usize) -> f64 {
        let index = i * self.risk_set.a.len() + k;
        self.risk_set.cmat[index] - self.tied.cmat[index] * j as f64 / self.ndead() as f64
    }
}

/// The data of one stratum (in the caller's row order) and its rows sorted
/// by ascending stop time.
pub(crate) struct StratumSweep<'a> {
    pub stop: &'a [f64],
    pub entry: Option<&'a [f64]>,
    pub status: &'a [i32],
    pub x: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    /// `exp(linear predictor)` (weights are applied inside the sweep).
    pub risk: &'a [f64],
    /// Row indices of the stratum sorted by (stop time, index).
    pub rows: &'a [usize],
    pub second_moments: bool,
}

impl StratumSweep<'_> {
    /// Visits every death time of the stratum from the largest downwards.
    pub fn for_each_death_time(&self, mut visit: impl FnMut(&DeathTime<'_>)) {
        let nvar = self.x.ncols();
        let x = self.x.as_standard_layout();
        let x = x.as_slice().expect("standard layout");
        let row_x = |row: usize| &x[row * nvar..(row + 1) * nvar];
        let rows = self.rows;
        // Rows join the risk set by decreasing stop time and, for (start,
        // stop] data, leave it by decreasing entry time.
        let (joining, leaving) = match self.entry {
            Some(entry) => {
                let spans = spans_a_death(rows, self.stop, entry, self.status);
                let joining: Vec<usize> = rows
                    .iter()
                    .zip(&spans)
                    .rev()
                    .filter_map(|(&row, &spans)| spans.then_some(row))
                    .collect();
                let mut leaving = joining.clone();
                leaving.sort_by(|&l, &r| entry[r].total_cmp(&entry[l]));
                (joining, leaving)
            }
            None => (rows.iter().rev().copied().collect(), Vec::new()),
        };
        let mut risk_set = RiskSetSums::zeros(nvar, self.second_moments);
        let mut tied = risk_set.clone();
        let (mut joined, mut left) = (0, 0);
        let mut deaths = Vec::new();
        let mut end = rows.len();
        while end > 0 {
            let time = self.stop[rows[end - 1]];
            let mut start = end - 1;
            while start > 0 && self.stop[rows[start - 1]] == time {
                start -= 1;
            }
            let tied_rows = &rows[start..end];
            end = start;
            deaths.clear();
            deaths.extend(
                tied_rows
                    .iter()
                    .copied()
                    .filter(|&row| self.status[row] == 1),
            );
            if deaths.is_empty() {
                continue;
            }
            if let Some(entry) = self.entry {
                while left < leaving.len() && entry[leaving[left]] >= time {
                    let row = leaving[left];
                    risk_set.remove(
                        self.weights[row],
                        self.weights[row] * self.risk[row],
                        row_x(row),
                    );
                    left += 1;
                }
            }
            while joined < joining.len() && self.stop[joining[joined]] >= time {
                let row = joining[joined];
                risk_set.add(
                    self.weights[row],
                    self.weights[row] * self.risk[row],
                    row_x(row),
                );
                joined += 1;
            }
            tied.clear();
            for &row in &deaths {
                tied.add(
                    self.weights[row],
                    self.weights[row] * self.risk[row],
                    row_x(row),
                );
            }
            visit(&DeathTime {
                time,
                deaths: &deaths,
                risk_set: &risk_set,
                tied: &tied,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn sweep_visits_death_times_with_the_correct_risk_sets() {
        let stop = [1.0, 2.0, 2.0, 3.0, 4.0];
        let entry = [0.0, 0.0, 1.5, 0.0, 2.5];
        let status = [1, 1, 1, 0, 1];
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let weights = [1.0, 2.0, 1.0, 1.0, 1.0];
        let risk = [1.0; 5];
        let rows = [0usize, 1, 2, 3, 4];
        let sweep = StratumSweep {
            stop: &stop,
            entry: Some(&entry),
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
            rows: &rows,
            second_moments: true,
        };
        let mut visited = Vec::new();
        sweep.for_each_death_time(|death| {
            visited.push((
                death.time,
                death.deaths.to_vec(),
                death.risk_set.count,
                death.risk_set.denom,
                death.risk_set.a[0],
                death.tied.denom,
            ));
        });
        // t = 4: row 4 only (row 3 stopped at 3).
        assert_eq!(visited[0], (4.0, vec![4], 1, 1.0, 5.0, 1.0));
        // t = 2: rows 1, 2, 3 (row 4 enters at 2.5).
        assert_eq!(visited[1], (2.0, vec![1, 2], 3, 4.0, 11.0, 3.0));
        // t = 1: rows 0, 1, 3 (row 2 enters at 1.5).
        assert_eq!(visited[2], (1.0, vec![0], 3, 4.0, 9.0, 1.0));
    }

    #[test]
    fn right_censored_sweep_keeps_everyone_with_a_later_stop() {
        let stop = [1.0, 2.0, 2.0];
        let status = [1, 1, 0];
        let x = arr2(&[[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]);
        let weights = [1.0; 3];
        let risk = [1.0, 2.0, 3.0];
        let rows = [0usize, 1, 2];
        let sweep = StratumSweep {
            stop: &stop,
            entry: None,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
            rows: &rows,
            second_moments: true,
        };
        let mut seen = 0;
        sweep.for_each_death_time(|death| {
            seen += 1;
            if death.time == 2.0 {
                assert_eq!(death.risk_set.denom, 5.0);
                assert_eq!(death.tied.denom, 2.0);
                assert_eq!(death.risk_set.cmat[2], 12.0);
                assert!((death.efron_denom(0) - 5.0).abs() < 1e-12);
            } else {
                assert_eq!(death.risk_set.denom, 6.0);
            }
        });
        assert_eq!(seen, 2);
    }

    #[test]
    fn rows_spanning_no_death_never_join_and_emptied_sets_restart_from_zero() {
        // survSplit-like: (0, 1], (1, 2], (2, 3] pieces; (1.2, 1.8] spans no
        // death time and the huge risk of (2, 3] leaves before t = 2.
        let stop = [1.0, 2.0, 3.0, 1.8, 2.0];
        let entry = [0.0, 1.0, 2.0, 1.2, 0.0];
        let status = [1, 1, 1, 0, 0];
        let x = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let weights = [1.0; 5];
        let risk = [1.0, 0.5, 1e300, 7.0, 0.25];
        let rows = [0usize, 3, 1, 4, 2];
        assert_eq!(
            spans_a_death(&rows, &stop, &entry, &status),
            vec![true, false, true, true, true]
        );
        let sweep = StratumSweep {
            stop: &stop,
            entry: Some(&entry),
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
            rows: &rows,
            second_moments: true,
        };
        let mut visited = Vec::new();
        sweep.for_each_death_time(|death| {
            let cmat = death.risk_set.cmat[0];
            visited.push((death.time, death.risk_set.count, death.risk_set.denom, cmat));
        });
        assert_eq!(visited[0], (3.0, 1, 1e300, 1e300 * 3.0 * 3.0));
        // Row 2 left the emptied set: the sums restart from exact zeros.
        assert_eq!(visited[1], (2.0, 2, 0.75, 2.0 + 6.25));
        assert_eq!(visited[2], (1.0, 2, 1.25, 1.0 + 6.25));
    }

    #[test]
    fn recentering_keeps_huge_linear_predictors_in_range() {
        let mut set = RecenteredRiskSet::new(1);
        set.restart(2);
        set.add(0, -750.0, 1.0, &[1.0]).unwrap();
        set.add(1, -749.0, 2.0, &[2.0]).unwrap();
        assert_eq!(set.rescales, 1);
        assert_eq!(set.recenter, -750.0);
        let (e, e2) = (1.0, 2.0 * 1f64.exp());
        assert!((set.sums.denom - (e + e2)).abs() < 1e-12);
        assert!((set.sums.a[0] - (e + 2.0 * e2)).abs() < 1e-12);
        set.remove(0, &[1.0]);
        assert!((set.sums.denom - e2).abs() < 1e-12);
        set.remove(1, &[2.0]);
        assert_eq!((set.sums.count, set.sums.denom), (0, 0.0));

        // A centre move beyond exp's range while rows are at risk.
        let mut set = RecenteredRiskSet::new(1);
        set.restart(2);
        set.add(0, 0.0, 1.0, &[0.0]).unwrap();
        let error = set.add(1, 2000.0, 1.0, &[0.0]).unwrap_err();
        assert!(error.to_string().contains("exp overflow due to covariates"));
    }
}
