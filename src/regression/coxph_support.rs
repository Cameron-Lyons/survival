//! The backward risk-set sweep shared by the Cox residual kernels.
//!
//! Every C routine behind `residuals.coxph`, `coxph.detail` and `cox.zph`
//! (`coxscho.c`, `coxscore2.c`/`agscore3.c`, `agmart3.c`, `coxdetail.c`,
//! `zph1.c`/`zph2.c`) walks one stratum from its largest time downwards,
//! keeping the weighted risk-set sums `sum w r`, `sum w r x` (and, for the
//! information-type kernels, `sum w r x x'`) plus the same sums over the
//! deaths tied at the current time.  [`StratumSweep`] does that walk once
//! per stratum, for right-censored and (start, stop] data alike, and hands
//! each death time's sums to the kernel, which applies its own Breslow or
//! Efron arithmetic.  Death times are visited in decreasing order; kernels
//! needing cumulative quantities reverse afterwards.

use ndarray::{Array2, ArrayView2};

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
    /// `sum w r x x'`, when second moments were requested.
    pub cmat: Option<Array2<f64>>,
}

impl RiskSetSums {
    fn zeros(nvar: usize, second_moments: bool) -> Self {
        Self {
            count: 0,
            weight: 0.0,
            denom: 0.0,
            a: vec![0.0; nvar],
            cmat: second_moments.then(|| Array2::zeros((nvar, nvar))),
        }
    }

    fn add(&mut self, weight: f64, risk: f64, x: ndarray::ArrayView1<'_, f64>) {
        self.count += 1;
        self.weight += weight;
        self.denom += risk;
        for (i, &xi) in x.iter().enumerate() {
            self.a[i] += risk * xi;
            if let Some(cmat) = self.cmat.as_mut() {
                for j in 0..=i {
                    cmat[(i, j)] += risk * xi * x[j];
                }
            }
        }
    }

    /// `self - other`, elementwise (lower triangle of `cmat`).
    fn minus(&self, other: &Self) -> Self {
        Self {
            count: self.count - other.count,
            weight: self.weight - other.weight,
            denom: self.denom - other.denom,
            a: self.a.iter().zip(&other.a).map(|(l, r)| l - r).collect(),
            cmat: self
                .cmat
                .as_ref()
                .zip(other.cmat.as_ref())
                .map(|(l, r)| l - r),
        }
    }
}

/// One death time's risk set and its tied deaths.
#[derive(Debug)]
pub(crate) struct DeathTime<'a> {
    pub time: f64,
    /// Original row indices of the deaths at this time, in sorted order.
    pub deaths: &'a [usize],
    /// Sums over the whole risk set (`entry < time <= stop`), deaths included.
    pub risk_set: RiskSetSums,
    /// Sums over the deaths only.
    pub tied: RiskSetSums,
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
        let (cmat, cmat2) = (
            self.risk_set
                .cmat
                .as_ref()
                .expect("second moments requested"),
            self.tied.cmat.as_ref().expect("second moments requested"),
        );
        cmat[(i, k)] - cmat2[(i, k)] * j as f64 / self.ndead() as f64
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
        let rows = self.rows;
        let n = rows.len();
        if n == 0 {
            return;
        }
        // Not-yet-entered rows are subtracted as in agfit4.c: walk the rows
        // by decreasing entry time alongside the stop-time walk.
        let entry_order: Vec<usize> = match self.entry {
            Some(entry) => {
                let mut order = rows.to_vec();
                order.sort_by(|&l, &r| entry[r].total_cmp(&entry[l]).then_with(|| r.cmp(&l)));
                order
            }
            None => Vec::new(),
        };
        let mut stop_sums = RiskSetSums::zeros(nvar, self.second_moments);
        let mut unentered = RiskSetSums::zeros(nvar, self.second_moments);
        let mut stop_ptr = n;
        let mut entry_ptr = 0usize;
        let mut deaths = Vec::new();
        let mut end = n;
        while end > 0 {
            let time = self.stop[rows[end - 1]];
            let mut start = end;
            while start > 0 && self.stop[rows[start - 1]] == time {
                start -= 1;
            }
            while stop_ptr > 0 && self.stop[rows[stop_ptr - 1]] >= time {
                let row = rows[stop_ptr - 1];
                stop_sums.add(
                    self.weights[row],
                    self.weights[row] * self.risk[row],
                    self.x.row(row),
                );
                stop_ptr -= 1;
            }
            if let Some(entry) = self.entry {
                while entry_ptr < n && entry[entry_order[entry_ptr]] >= time {
                    let row = entry_order[entry_ptr];
                    unentered.add(
                        self.weights[row],
                        self.weights[row] * self.risk[row],
                        self.x.row(row),
                    );
                    entry_ptr += 1;
                }
            }
            deaths.clear();
            deaths.extend(
                rows[start..end]
                    .iter()
                    .copied()
                    .filter(|&row| self.status[row] == 1),
            );
            if !deaths.is_empty() {
                let mut tied = RiskSetSums::zeros(nvar, self.second_moments);
                for &row in &deaths {
                    tied.add(
                        self.weights[row],
                        self.weights[row] * self.risk[row],
                        self.x.row(row),
                    );
                }
                let risk_set = if self.entry.is_some() {
                    stop_sums.minus(&unentered)
                } else {
                    stop_sums.clone()
                };
                visit(&DeathTime {
                    time,
                    deaths: &deaths,
                    risk_set,
                    tied,
                });
            }
            end = start;
        }
    }
}

/// Value of a right-continuous step function at `t` (0 before the first
/// step): `c(0, values)[findInterval(t, times) + 1]`.
pub(crate) fn step_value(times: &[f64], values: &[f64], t: f64) -> f64 {
    let index = times.partition_point(|&time| time <= t);
    if index == 0 { 0.0 } else { values[index - 1] }
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
                assert_eq!(death.risk_set.cmat.as_ref().unwrap()[(1, 0)], 12.0);
                assert!((death.efron_denom(0) - 5.0).abs() < 1e-12);
            } else {
                assert_eq!(death.risk_set.denom, 6.0);
            }
        });
        assert_eq!(seen, 2);
        assert_eq!(step_value(&[1.0, 2.0], &[0.5, 1.0], 1.5), 0.5);
        assert_eq!(step_value(&[1.0, 2.0], &[0.5, 1.0], 0.5), 0.0);
    }
}
