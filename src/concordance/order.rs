//! The orderings the concordance sweep runs over: each predictor's rank among
//! its unique values.
//!
//! They sort contiguous integer keys rather than searching or comparing
//! through the data, whose random loads dominate once the data outgrow the
//! cache.

use crate::concordance::rank_tree::btree;

/// Maps a non-NaN value to an integer of the same order, with `-0.0` and
/// `0.0` equal as they are under `==`.
fn ordered_bits(value: f64) -> u64 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | 1 << 63
    }
}

/// Each row's predictor value as its position among the sorted unique
/// values, which is its level in the sweep's rank tree.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct PredictorRanks {
    pub(crate) rank: Vec<usize>,
    pub(crate) levels: usize,
}

impl PredictorRanks {
    /// One sort of the rows by value, rather than a binary search of the
    /// sorted unique values per row.
    pub(crate) fn new(values: &[f64]) -> Self {
        let mut keyed: Vec<(u64, usize)> =
            values.iter().map(|&v| ordered_bits(v)).zip(0..).collect();
        keyed.sort_unstable_by_key(|&(key, _)| key);
        let mut rank = vec![0; values.len()];
        let mut levels = 0;
        let mut previous = None;
        for &(key, row) in &keyed {
            if previous != Some(key) {
                previous = Some(key);
                levels += 1;
            }
            rank[row] = levels - 1;
        }
        Self { rank, levels }
    }

    /// Each row's node in the rank tree.
    pub(crate) fn nodes(&self) -> Vec<usize> {
        let tree = btree(self.levels);
        self.rank.iter().map(|&rank| tree[rank]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::rng::Rng;

    const SPECIAL: [f64; 14] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
        5e-324,
        -5e-324,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.37,
    ];

    #[test]
    fn ordered_bits_follow_numeric_order() {
        for a in SPECIAL {
            for b in SPECIAL {
                assert_eq!(
                    ordered_bits(a).cmp(&ordered_bits(b)),
                    a.partial_cmp(&b).unwrap(),
                    "{a:e} vs {b:e}"
                );
            }
        }
    }

    /// docount's ranking before the sort of keyed rows: the sorted unique
    /// values, binary-searched per row.
    fn searched_ranks(values: &[f64]) -> PredictorRanks {
        let mut levels = values.to_vec();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        PredictorRanks {
            rank: values
                .iter()
                .map(|value| levels.partition_point(|level| level < value))
                .collect(),
            levels: levels.len(),
        }
    }

    fn draw(rng: &mut Rng, kind: usize) -> f64 {
        match kind {
            0 => [0.0, -0.0, 1.0, -1.0, 2.0, 0.5][rng.usize(0..6)],
            1 => SPECIAL[rng.usize(0..SPECIAL.len())],
            2 => (rng.f64() * 10.0 - 5.0).round() / 2.0,
            _ => rng.f64() * 4.0 - 2.0,
        }
    }

    #[test]
    fn keyed_ranks_match_searched_levels() {
        let mut rng = Rng::with_seed(0x0dd_ba11);
        for trial in 0..4000 {
            let n = rng.usize(0..40) + if trial % 50 == 0 { 2000 } else { 0 };
            let kind = rng.usize(0..4);
            let risk: Vec<f64> = (0..n)
                .map(|_| {
                    let value = draw(&mut rng, kind);
                    if value.is_finite() { value } else { -0.0 }
                })
                .collect();
            assert_eq!(
                PredictorRanks::new(&risk),
                searched_ranks(&risk),
                "trial {trial}"
            );
        }
    }

    #[test]
    fn nodes_place_each_level_in_the_balanced_tree() {
        let ranks = PredictorRanks::new(&[0.5, -0.0, 2.0, 0.0, 0.5]);
        assert_eq!(ranks.rank, [1, 0, 2, 0, 1]);
        assert_eq!(ranks.levels, 3);
        assert_eq!(ranks.nodes(), [0, 1, 2, 1, 0]);
        assert_eq!(PredictorRanks::new(&[]), PredictorRanks::default());
    }
}
