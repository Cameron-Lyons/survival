//! Efron's correction of the frailty groups that share a tie set's deaths.
//!
//! Death step `k` of a tie set of `d` deaths moves each tied group's share of
//! the risk from `a[g] / d2` to `(a[g] - temp * a2[g]) / d2`, with
//! `temp = k / d` and `d2 = denom - temp * efron_wt`. The group sums `a[g]`,
//! `a2[g]` and `cmat2[.., g]` and the risk-set total `denom` stay fixed across
//! the steps, so the corrections of every step factor into a few sums over the
//! steps and one pass over the groups: O(d + groups) work per tie set rather
//! than O(d * groups). In exact arithmetic this is the per-step correction of
//! R's coxfit5 and agfit5; in floating point it re-associates it.

use super::Kernel;
use ndarray::Array2;

/// Sums over a tie set's death steps, with `r = denom / d2` and
/// `c = meanwt * temp * r`:
///
/// - `s1 = sum(c)`, `s2 = sum(c * r)`, `s3 = sum(c * temp * r)`;
/// - `t[i] = sum(c * tmean[i])` over the step's covariate means.
///
/// `r` lies between 1 and `d`, so no sum depends on the scale of the risks.
#[derive(Debug, Clone)]
pub(super) struct TieSums {
    s1: f64,
    s2: f64,
    s3: f64,
    t: Vec<f64>,
    pending: bool,
}

impl TieSums {
    pub(super) fn new(nvar: usize) -> Self {
        Self {
            s1: 0.0,
            s2: 0.0,
            s3: 0.0,
            t: vec![0.0; nvar],
            pending: false,
        }
    }

    /// Adds one death step with `temp != 0`; `means` are its covariate means.
    pub(super) fn add(&mut self, meanwt: f64, temp: f64, denom: f64, d2: f64, means: &[f64]) {
        let r = denom / d2;
        let c = meanwt * temp * r;
        self.s1 += c;
        self.s2 += c * r;
        self.s3 += c * temp * r;
        for (sum, &mean) in self.t.iter_mut().zip(means) {
            *sum += c * mean;
        }
        self.pending = true;
    }

    fn clear(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
        self.s3 = 0.0;
        self.t.fill(0.0);
        self.pending = false;
    }
}

impl Kernel {
    /// Applies the corrections the tie set just swept made to its frailty
    /// groups. With `a' = a[g] / denom` and `a2' = a2[g] / denom`:
    ///
    /// - `u[g] += a2' * s1`
    /// - `fdiag[g] += -a2' * s1 + 2 * a' * a2' * s2 - a2'^2 * s3`
    /// - `jmat[i, g] += -cmat2[i, g] / denom * s1 + a2' * t[i]`
    ///
    /// It must run before the second moments of the set are cleared. A set
    /// whose steps all had `temp == 0` (a lone death, Breslow ties) has
    /// nothing to apply.
    pub(super) fn finish_tie_set(
        &mut self,
        denom: f64,
        u: &mut [f64],
        jmat: &mut Array2<f64>,
        fdiag: &mut [f64],
    ) {
        if !self.tie_sums.pending {
            return;
        }
        let TieSums { s1, s2, s3, .. } = self.tie_sums;
        for &g in &self.tied {
            let share = self.a[g] / denom;
            let tied_share = self.a2[g] / denom;
            u[g] += tied_share * s1;
            fdiag[g] +=
                -tied_share * s1 + 2.0 * share * tied_share * s2 - tied_share * tied_share * s3;
            for (ii, &t) in self.tie_sums.t.iter().enumerate() {
                jmat[(ii, g)] += -(self.cmat2[(ii, g)] / denom) * s1 + tied_share * t;
            }
        }
        self.tie_sums.clear();
    }
}
