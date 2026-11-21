//! Cox information from two weighted Gram matrices.
//!
//! R's right-censored sweeps (`coxfit6`, `coxfit5`) add each subject's `r x x'`
//! to running risk-set sums and read them at every death time,
//! `O((n + t) p^2)` scalar work for `n` subjects, `t` death times and `p`
//! covariates. The information reads those sums only through
//! `sum_t sum_j wtave (C_t - f_j C2_t) / d_tj`, which is `sum_i omega_i x_i x_i'`:
//! every death time adds `alpha = sum_j wtave / d_j` to the weight of each
//! subject still at risk, and `gamma = sum_j wtave s_j / d_j` (`s_j = 1 - f_j`)
//! instead to each of its deaths. The mean part is `sum_tj wtave m_tj m_tj'` for
//! the step means `m_tj`; a tie set of several Efron steps contributes the two
//! rows of the Cholesky factor of its step sums
//! ([`EfronStepSums::mean_factor`]) instead of one row per step. Both are
//! matrix products, which run far faster than the running sums once there are
//! more than a few covariates, and the weights are sums of positive terms.
//! Rows go through the products in blocks of [`BLOCK`], so the storage does not
//! grow with the data.

use crate::regression::efron_steps::{EfronStepSums, scaled_first};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2, ArrayViewMut1, s};

/// The fewest covariates for which the Gram evaluation replaces R's sweep;
/// below it the scalar sweep is as fast.
pub(crate) const GRAM_MIN_VARIABLES: usize = 8;

/// Rows per matrix product.
const BLOCK: usize = 4096;

/// The Gram matrix `sum_i omega_i x_i x_i' - sum_k w_k m_k m_k'` under
/// construction, with its scratch storage.
#[derive(Debug)]
pub(crate) struct CoxGram {
    /// Up to [`BLOCK`] mean rows and their weights.
    means: Array2<f64>,
    mean_weights: Vec<f64>,
    scratch: Array2<f64>,
    gram: Array2<f64>,
}

impl CoxGram {
    pub(crate) fn new(n: usize, nvar: usize) -> Self {
        let block = BLOCK.min(n.max(1));
        Self {
            means: Array2::zeros((block, nvar)),
            mean_weights: Vec::with_capacity(block),
            scratch: Array2::zeros((block, nvar)),
            gram: Array2::zeros((nvar, nvar)),
        }
    }

    /// Starts a new evaluation.
    pub(crate) fn clear(&mut self) {
        self.mean_weights.clear();
        self.gram.fill(0.0);
    }

    /// The next mean row, to be written and then recorded with
    /// [`Self::push_mean`].
    pub(crate) fn mean_row(&mut self) -> ArrayViewMut1<'_, f64> {
        let slot = self.mean_weights.len();
        self.means.row_mut(slot)
    }

    /// Records the weight of the mean row just written, subtracting the
    /// buffer once it is full.
    pub(crate) fn push_mean(&mut self, weight: f64) {
        self.mean_weights.push(weight);
        if self.mean_weights.len() == self.means.nrows() {
            self.flush_means();
        }
    }

    /// Records a tie set of several Efron steps, of common weight `wtave`,
    /// from its step sums and its risk-set (`a`, `denom`) and deaths' (`a2`)
    /// first moments. `None` when the factor is not finite.
    pub(crate) fn push_tie_set(
        &mut self,
        steps: &EfronStepSums,
        a: &[f64],
        a2: &[f64],
        denom: f64,
        wtave: f64,
    ) -> Option<()> {
        let ((first, cross), (_, second)) = steps.mean_factor()?;
        let scale = 1.0 / denom;
        for factor in [(first, cross), (0.0, second)] {
            for (mean, (&a, &a2)) in self.mean_row().iter_mut().zip(a.iter().zip(a2)) {
                let (survivors, deaths) = scaled_first(a, a2, scale);
                *mean = factor.0 * survivors + factor.1 * deaths;
            }
            self.push_mean(wtave);
        }
        Some(())
    }

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

    /// Adds `sum_i omega_i x_i x_i'` over the rows of `covar` and returns the
    /// finished, symmetric matrix.
    pub(crate) fn finish(&mut self, covar: ArrayView2<'_, f64>, omega: &[f64]) -> &Array2<f64> {
        self.flush_means();
        let n = covar.nrows();
        let mut first = 0;
        while first < n {
            let last = (first + self.scratch.nrows()).min(n);
            let rows = covar.slice(s![first..last, ..]);
            let mut weighted = self.scratch.slice_mut(s![..last - first, ..]);
            for ((mut target, source), &omega) in weighted
                .rows_mut()
                .into_iter()
                .zip(rows.rows())
                .zip(&omega[first..last])
            {
                for (value, &x) in target.iter_mut().zip(source) {
                    *value = omega * x;
                }
            }
            general_mat_mul(1.0, &rows.t(), &weighted, 1.0, &mut self.gram);
            first = last;
        }
        &self.gram
    }
}

/// `high + low + value` as a double-double sum (Knuth's two-sum, then a fast
/// renormalization), so prefix sums keep twice the precision of a double.
pub(crate) fn add_double_double(high: f64, low: f64, value: f64) -> (f64, f64) {
    let sum = high + value;
    let virtual_value = sum - high;
    let error = (high - (sum - virtual_value)) + (value - virtual_value);
    let low = low + error;
    let total = sum + low;
    (total, low - (total - sum))
}

/// The death times of one stratum in increasing order, with the double-double
/// prefix sums of their alphas and their gammas, for weighting counting-process
/// rows: a row at risk over `(entry, stop]` gets the alphas of the death times
/// in that interval, with its own death time's gamma in place of that time's
/// alpha.
#[derive(Debug, Default)]
pub(crate) struct IntervalWeights {
    times: Vec<f64>,
    prefix: Vec<(f64, f64)>,
    gammas: Vec<f64>,
}

impl IntervalWeights {
    /// Starts a stratum from its `(time, alpha, gamma)` in decreasing time.
    pub(crate) fn reset<'a>(
        &mut self,
        decreasing: impl DoubleEndedIterator<Item = &'a (f64, f64, f64)>,
    ) {
        self.times.clear();
        self.gammas.clear();
        self.prefix.clear();
        self.prefix.push((0.0, 0.0));
        for &(time, alpha, gamma) in decreasing.rev() {
            self.times.push(time);
            self.gammas.push(gamma);
            let &(high, low) = self.prefix.last().expect("starts with zero");
            self.prefix.push(add_double_double(high, low, alpha));
        }
    }

    /// The weight of a row at risk over `(entry, stop]`.
    pub(crate) fn share(&self, entry: f64, stop: f64, death: bool) -> f64 {
        let interval = |from: usize, to: usize| {
            let (high_to, low_to) = self.prefix[to];
            let (high_from, low_from) = self.prefix[from];
            (high_to - high_from) + (low_to - low_from)
        };
        let through = self.times.partition_point(|&t| t <= stop);
        let before = self.times.partition_point(|&t| t <= entry);
        if death && through > before && self.times[through - 1] == stop {
            interval(before, through - 1) + self.gammas[through - 1]
        } else {
            interval(before, through)
        }
    }
}
