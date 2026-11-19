use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct WeightRange {
    smallest: f64,
    largest: f64,
    rows: usize,
}

impl WeightRange {
    const EMPTY: Self = Self {
        smallest: f64::INFINITY,
        largest: 0.0,
        rows: 0,
    };

    fn include(&mut self, weight: f64) {
        self.rows += 1;
        if weight > 0.0 {
            self.smallest = self.smallest.min(weight);
            self.largest = self.largest.max(weight);
        }
    }

    fn needs_exact_updates(self, counting: bool) -> bool {
        let n = self.rows as f64;
        // Several entries can leave the curve together. Within each right-censored
        // cluster the forward sweep removes one row at a time and resets empty clusters.
        let removed = if counting { n } else { 1.0 };
        // Leave headroom for rounding in both the bound and the running sum.
        // In particular, three copies of the rounded MAX / 3 sum to infinity.
        (self.rows > 1 && self.largest > (f64::MAX / 2.0) / n)
            || (self.smallest < self.largest
                && self.smallest < self.largest * (removed * f64::EPSILON.sqrt()))
    }
}

/// Subtracting a large case weight can erase the remaining risk set.
/// Use exact updates when even a single small weight could lose half its digits
/// against the total, or when summing the finite weights could overflow.
pub(crate) fn needs_exact_updates(rows: &[usize], weights: &[f64], counting: bool) -> bool {
    let mut range = WeightRange::EMPTY;
    for &row in rows {
        range.include(weights[row]);
    }
    range.needs_exact_updates(counting)
}

/// Keep ordinary cluster updates when only the weights of different clusters
/// are far apart. Those weights are never subtracted from each other.
pub(crate) fn cluster_weights(
    rows: &[usize],
    weights: &[f64],
    cluster: &[usize],
    nid: usize,
    counting: bool,
) -> Vec<Option<RiskWeight>> {
    let mut ranges = vec![WeightRange::EMPTY; nid];
    for &row in rows {
        ranges[cluster[row]].include(weights[row]);
    }
    ranges
        .into_iter()
        .map(|range| {
            range
                .needs_exact_updates(counting)
                .then(RiskWeight::default)
        })
        .collect()
}

/// Finite binary64 weights are integers times powers of two. Per-exponent
/// integer totals remain exact across insertions and deletions at any scale:
/// a 53-bit significand times at most usize::MAX rows fits in an i128.
#[derive(Default)]
pub(crate) struct RiskWeight {
    bins: BTreeMap<u16, i128>,
}

impl RiskWeight {
    pub(crate) fn add(&mut self, weight: f64) {
        if weight == 0.0 {
            return;
        }
        let bits = weight.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u16;
        let implicit = u64::from(exponent != 0) << 52;
        let significand = i128::from(implicit | (bits & ((1 << 52) - 1)));
        // Subnormals use the same unit as the smallest normal bin.
        let key = exponent.max(1);
        let total = self.bins.entry(key).or_default();
        *total += if weight < 0.0 {
            -significand
        } else {
            significand
        };
        if *total == 0 {
            self.bins.remove(&key);
        }
    }

    /// The sum of finite weights can exceed f64::MAX. Scaling each exponent
    /// bin by 2^-1023 keeps the normalized sum in range in that case.
    pub(crate) fn log2_total(&self) -> f64 {
        let total = self.total();
        if total.is_finite() {
            return total.log2();
        }
        let scaled = self
            .bins
            .iter()
            .map(|(&exponent, &significand)| {
                (significand as f64 / (1_u64 << 52) as f64)
                    * 2.0_f64.powi(i32::from(exponent) - 2046)
            })
            .sum::<f64>();
        scaled.log2() + 1023.0
    }

    pub(crate) fn total(&self) -> f64 {
        self.bins
            .iter()
            .fold(0.0, |sum, (&exponent, &significand)| {
                let scale = f64::from_bits(u64::from(exponent) << 52);
                sum + (significand as f64 / (1_u64 << 52) as f64) * scale
            })
    }
}

#[cfg(test)]
mod tests {
    use super::RiskWeight;

    #[test]
    fn removing_weights_recovers_every_scale_including_subnormals() {
        let tiny = f64::from_bits(1);
        let mut sum = RiskWeight::default();
        for weight in [tiny, 1e-180, 3.0, 1e120, f64::MAX, f64::MAX] {
            sum.add(weight);
        }
        assert_eq!(sum.total(), f64::INFINITY);
        for (weight, remaining) in [
            (f64::MAX, f64::MAX),
            (f64::MAX, 1e120),
            (1e120, 3.0),
            (3.0, 1e-180),
            (1e-180, tiny),
            (tiny, 0.0),
        ] {
            sum.add(-weight);
            assert_eq!(sum.total(), remaining);
        }
    }
}
