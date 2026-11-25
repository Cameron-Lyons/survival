//! The orderings the concordance sweep runs over: each predictor's rank among
//! its unique values, and the rows in sweep order (R's
//! `order(-y[, 1], y[, 2], risk)`).
//!
//! Both sort contiguous integer keys rather than row indices compared through
//! the data: an indirect comparison loads the stop time, status and predictor
//! of two random rows, which dominates the sort once the data outgrow the
//! cache.

use crate::concordance::rank_tree::btree;
use crate::core::time_order::total_order_key;
use rayon::prelude::*;
use std::ops::Range;

const PARALLEL_ORDER_THRESHOLD: usize = 16_384;

/// Maps a non-NaN value to an integer of the same order, with `-0.0` and
/// `0.0` equal as they are under `==`.
fn ordered_bits(value: f64) -> u64 {
    total_order_key(if value == 0.0 { 0.0 } else { value })
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
        if keyed.len() >= PARALLEL_ORDER_THRESHOLD {
            keyed.par_sort_unstable();
        } else {
            keyed.sort_unstable_by_key(|&(key, _)| key);
        }
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

/// Rows past this count cannot pack their rank and index into the low 64
/// bits of a sort key.
const PACKED_ROWS: usize = 1 << 31;

/// The rows in sweep order (descending stop time, censored rows before
/// deaths at each time, then ascending predictor rank, ties in row order),
/// with the runs of two or more rows that share a stop time and status.
/// Later predictors re-sort only within those runs.
pub(crate) fn sweep_order(
    stop: &[f64],
    status: &[i32],
    rank: &[usize],
) -> (Vec<usize>, Vec<Range<usize>>) {
    if stop.len() < PACKED_ROWS {
        packed_sweep_order(stop, status, rank)
    } else {
        compared_sweep_order(stop, status, rank)
    }
}

/// One `u128` key per row: the reversed stop time, the status, the rank and
/// the row, so that an unstable sort of the distinct keys reproduces the
/// stable sort.
fn packed_sweep_order(
    stop: &[f64],
    status: &[i32],
    rank: &[usize],
) -> (Vec<usize>, Vec<Range<usize>>) {
    let mut keys: Vec<u128> = (0..stop.len())
        .map(|row| {
            u128::from(!ordered_bits(stop[row])) << 64
                | u128::from(status[row] == 1) << 63
                | (rank[row] as u128) << 32
                | row as u128
        })
        .collect();
    if keys.len() >= PARALLEL_ORDER_THRESHOLD {
        keys.par_sort_unstable();
    } else {
        keys.sort_unstable();
    }
    let ties = runs(keys.len(), |a, b| keys[a] >> 63 == keys[b] >> 63);
    (keys.iter().map(|&key| key as u32 as usize).collect(), ties)
}

fn compared_sweep_order(
    stop: &[f64],
    status: &[i32],
    rank: &[usize],
) -> (Vec<usize>, Vec<Range<usize>>) {
    let mut order: Vec<usize> = (0..stop.len()).collect();
    // Stop times are finite or infinite, never NaN.
    order.sort_by(|&a, &b| {
        stop[b]
            .partial_cmp(&stop[a])
            .unwrap()
            .then_with(|| status[a].cmp(&status[b]))
            .then_with(|| rank[a].cmp(&rank[b]))
    });
    let ties = runs(order.len(), |a, b| {
        stop[order[a]] == stop[order[b]] && status[order[a]] == status[order[b]]
    });
    (order, ties)
}

/// The runs of two or more consecutive positions that `same` groups with
/// the run's first position.
fn runs(n: usize, same: impl Fn(usize, usize) -> bool) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut first = 0;
    while first < n {
        let mut end = first + 1;
        while end < n && same(first, end) {
            end += 1;
        }
        if end - first > 1 {
            runs.push(first..end);
        }
        first = end;
    }
    runs
}

/// The sweep order for another predictor: each run of `ties` re-sorted by
/// that predictor's rank, ties in row order.
pub(crate) fn resort_ties(order: &[usize], ties: &[Range<usize>], rank: &[usize]) -> Vec<usize> {
    let mut order = order.to_vec();
    let mut keyed: Vec<(usize, usize)> = Vec::new();
    for run in ties {
        keyed.clear();
        keyed.extend(order[run.clone()].iter().map(|&row| (rank[row], row)));
        keyed.sort_unstable();
        for (slot, &(_, row)) in order[run.clone()].iter_mut().zip(&keyed) {
            *slot = row;
        }
    }
    order
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

    /// prepare_stratum's order before packed keys: a stable sort comparing
    /// the rows' values.
    fn compared_order(stop: &[f64], status: &[i32], risk: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..stop.len()).collect();
        order.sort_by(|&a, &b| {
            stop[b]
                .partial_cmp(&stop[a])
                .unwrap()
                .then_with(|| status[a].cmp(&status[b]))
                .then_with(|| risk[a].partial_cmp(&risk[b]).unwrap())
        });
        order
    }

    fn compared_ties(stop: &[f64], status: &[i32], order: &[usize]) -> Vec<Range<usize>> {
        let n = order.len();
        let mut ties = Vec::new();
        let mut first = 0;
        while first < n {
            let row = order[first];
            let mut end = first + 1;
            while end < n && stop[row] == stop[order[end]] && status[row] == status[order[end]] {
                end += 1;
            }
            if end - first > 1 {
                ties.push(first..end);
            }
            first = end;
        }
        ties
    }

    fn compared_resort(order: &[usize], ties: &[Range<usize>], risk: &[f64]) -> Vec<usize> {
        let mut order = order.to_vec();
        for group in ties {
            order[group.clone()].sort_unstable_by(|&a, &b| {
                risk[a]
                    .partial_cmp(&risk[b])
                    .unwrap()
                    .then_with(|| a.cmp(&b))
            });
        }
        order
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
    fn keyed_orders_match_compared_sorts() {
        let mut rng = Rng::with_seed(0x0dd_ba11);
        for trial in 0..4000 {
            let n = rng.usize(0..40) + if trial % 50 == 0 { 2000 } else { 0 };
            let (stop_kind, risk_kind, other_kind) =
                (rng.usize(0..4), rng.usize(0..4), rng.usize(0..4));
            let stop: Vec<f64> = (0..n).map(|_| draw(&mut rng, stop_kind)).collect();
            let status: Vec<i32> = (0..n).map(|_| i32::from(rng.bool())).collect();
            let finite = |rng: &mut Rng, kind| {
                let value = draw(rng, kind);
                if value.is_finite() { value } else { -0.0 }
            };
            let risk: Vec<f64> = (0..n).map(|_| finite(&mut rng, risk_kind)).collect();
            let other: Vec<f64> = (0..n).map(|_| finite(&mut rng, other_kind)).collect();

            let ranks = PredictorRanks::new(&risk);
            assert_eq!(ranks, searched_ranks(&risk), "trial {trial}");
            let other_ranks = PredictorRanks::new(&other);
            assert_eq!(other_ranks, searched_ranks(&other), "trial {trial}");

            let order = compared_order(&stop, &status, &risk);
            let ties = compared_ties(&stop, &status, &order);
            let expected = (order.clone(), ties.clone());
            assert_eq!(sweep_order(&stop, &status, &ranks.rank), expected);
            assert_eq!(packed_sweep_order(&stop, &status, &ranks.rank), expected);
            assert_eq!(compared_sweep_order(&stop, &status, &ranks.rank), expected);
            assert_eq!(
                resort_ties(&order, &ties, &other_ranks.rank),
                compared_resort(&order, &ties, &other),
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
