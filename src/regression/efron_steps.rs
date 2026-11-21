//! Efron's information for a tie set, collapsed over its death steps.
//!
//! R's `coxfit5`, `coxfit6`, `agfit4` and `agfit5` add the information of a
//! tie set of `ndead` deaths one step at a time: step `j` removes the share
//! `1 - s` (`s = 1 - j / ndead` for Efron, `s = 1` for Breslow) of the deaths'
//! risk from the risk set, so its denominator is `d = D - (1 - s) E` and its
//! information entry is
//!
//! ```text
//! (C - (1 - s) C2) / d - (A_i - (1 - s) A2_i) (A_k - (1 - s) A2_k) / d^2
//! ```
//!
//! for the risk-set sums `D`, `A`, `C` and the deaths' sums `E`, `A2`, `C2`.
//! That costs `O(ndead nvar^2)` per tie set. With the survivors' sums
//! `R = A - A2` and `RC = C - C2` the entry is
//! `(RC + s C2) / d - (R_i + s A2_i)(R_k + s A2_k) / d^2`, whose sum over the
//! steps needs only the five step sums of `r`, `s r`, `r^2`, `s r^2` and
//! `s^2 r^2` for the ratio `r = D / d`: `O(ndead + nvar^2)` in all. Every
//! coefficient is a positive sum, so a set in which nearly every subject at
//! risk dies, the worst case, loses only about `log(ndead)` rounding units
//! more than the per-step sums; dividing the moments by `D` keeps the
//! intermediate products as small as the step means.

/// The five sums over a tie set's steps; see the module documentation.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EfronStepSums {
    ratio: f64,
    share: f64,
    ratio_square: f64,
    share_square: f64,
    share_squared_square: f64,
}

impl EfronStepSums {
    /// Adds a step with share `share` of the deaths' risk and denominator
    /// `denom`, out of the full risk set's denominator `total`.
    #[inline]
    pub(crate) fn add(&mut self, share: f64, denom: f64, total: f64) {
        let ratio = total / denom;
        let ratio_square = ratio * ratio;
        self.ratio += ratio;
        self.share += share * ratio;
        self.ratio_square += ratio_square;
        self.share_square += share * ratio_square;
        self.share_squared_square += share * share * ratio_square;
    }

    /// Whether every sum is finite and positive, so that the collapsed
    /// information can replace the per-step sums.
    pub(crate) fn is_usable(&self) -> bool {
        [
            self.ratio,
            self.share,
            self.ratio_square,
            self.share_square,
            self.share_squared_square,
        ]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    }

    /// The two rows whose outer products, weighted by the steps' common
    /// weight, sum to the steps' `sum_j m_j m_j'` for the means
    /// `m_j = (R + s_j A2) r_j / D`: with `R / D` and `A2 / D` the
    /// [`scaled_first`] moments, `sum_j r_j^2 [1, s_j][1, s_j]'` is a positive
    /// semi-definite 2 x 2 matrix, and its Cholesky factor combines them.
    /// `None` when the factor is not finite.
    #[allow(clippy::type_complexity)]
    pub(crate) fn mean_factor(&self) -> Option<((f64, f64), (f64, f64))> {
        let first = self.ratio_square.sqrt();
        let cross = self.share_square / first;
        let second = (self.share_squared_square - cross * cross).max(0.0).sqrt();
        [first, cross, second]
            .iter()
            .all(|value| value.is_finite())
            .then_some(((first, cross), (0.0, second)))
    }

    /// The summed information entry of the steps for variables `i` and `k`,
    /// given each one's [`scaled_first`] moments and the raw second moments
    /// `second` (risk set) and `tied_second` (deaths), with `scale = 1 / D`.
    #[inline]
    pub(crate) fn entry(
        &self,
        i: (f64, f64),
        k: (f64, f64),
        second: f64,
        tied_second: f64,
        scale: f64,
    ) -> f64 {
        let ((survivors_i, deaths_i), (survivors_k, deaths_k)) = (i, k);
        let survivor_second = (second - tied_second) * scale;
        let second = survivor_second * self.ratio + tied_second * scale * self.share;
        let product = survivors_i * survivors_k * self.ratio_square
            + (survivors_i * deaths_k + deaths_i * survivors_k) * self.share_square
            + deaths_i * deaths_k * self.share_squared_square;
        second - product
    }
}

/// A variable's survivors' and deaths' first moments `((A - A2) / D, A2 / D)`.
#[inline]
pub(crate) fn scaled_first(first: f64, tied_first: f64, scale: f64) -> (f64, f64) {
    ((first - tied_first) * scale, tied_first * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One variable, three deaths among five at risk: the collapsed entry
    /// matches the per-step sum of R's terms.
    #[test]
    fn collapsed_entry_matches_the_per_step_sum() {
        let risks = [0.5, 1.5, 2.0, 0.25, 3.0];
        let x = [0.3, -1.2, 0.7, 2.0, -0.4];
        let deaths = [0, 2, 4];
        let sum = |f: &dyn Fn(usize) -> f64, idx: &[usize]| idx.iter().map(|&i| f(i)).sum::<f64>();
        let all: Vec<usize> = (0..5).collect();
        let (d, a, c) = (
            sum(&|i| risks[i], &all),
            sum(&|i| risks[i] * x[i], &all),
            sum(&|i| risks[i] * x[i] * x[i], &all),
        );
        let (e, a2, c2) = (
            sum(&|i| risks[i], &deaths),
            sum(&|i| risks[i] * x[i], &deaths),
            sum(&|i| risks[i] * x[i] * x[i], &deaths),
        );
        let ndead = deaths.len();
        let mut steps = EfronStepSums::default();
        let mut expected = 0.0;
        for j in 0..ndead {
            let fraction = j as f64 / ndead as f64;
            let denom = d - fraction * e;
            let mean = (a - fraction * a2) / denom;
            expected += (c - fraction * c2) / denom - mean * mean;
            steps.add(1.0 - fraction, denom, d);
        }
        assert!(steps.is_usable());
        let moments = scaled_first(a, a2, 1.0 / d);
        let actual = steps.entry(moments, moments, c, c2, 1.0 / d);
        assert!(
            (actual - expected).abs() < 1e-14 * expected.abs(),
            "{actual} vs {expected}"
        );
    }

    /// The factor's two rows reproduce the steps' summed mean outer products.
    #[test]
    fn mean_factor_reproduces_the_step_means() {
        let (d, e) = (10.0, 6.0);
        let (a, a2) = (3.0, 2.5);
        let ndead = 4;
        let mut steps = EfronStepSums::default();
        let mut expected = 0.0;
        for j in 0..ndead {
            let fraction = j as f64 / ndead as f64;
            let denom = d - fraction * e;
            let mean = (a - fraction * a2) / denom;
            expected += mean * mean;
            steps.add(1.0 - fraction, denom, d);
        }
        let ((l11, l21), (_, l22)) = steps.mean_factor().unwrap();
        let (survivors, deaths) = scaled_first(a, a2, 1.0 / d);
        let first = l11 * survivors + l21 * deaths;
        let second = l22 * deaths;
        let actual = first * first + second * second;
        assert!(
            (actual - expected).abs() < 1e-14 * expected,
            "{actual} vs {expected}"
        );
    }

    #[test]
    fn unusable_sums_are_detected() {
        let mut steps = EfronStepSums::default();
        assert!(!steps.is_usable());
        steps.add(1.0, 0.0, 1.0);
        assert!(!steps.is_usable());
        steps = EfronStepSums::default();
        steps.add(1.0, f64::NAN, 1.0);
        assert!(!steps.is_usable());
        steps = EfronStepSums::default();
        steps.add(1.0, 2.0, 1.0);
        assert!(steps.is_usable());
    }
}
