//! Canonical index-sort helpers.
//!
//! Each function returns row indices rather than moving the data, so
//! several parallel vectors can be reordered together.  Ordering is
//! `f64::total_cmp`, so `NaN` never panics and sorts after `+inf` (before
//! `-inf` when descending); ties keep their original order (stable), which
//! is what R's `order()` does.
//!
//! The sorts work on `(value, row)` pairs rather than on an index vector:
//! the keys sit next to each other in memory, which is several times faster
//! than an indirect comparison sort at a million rows, and the row breaks
//! ties, so an unstable sort gives the stable order.

use crate::constants::PARALLEL_THRESHOLD_LARGE;
use rayon::prelude::*;
use std::cmp::Ordering;

fn sort_pairs(
    rows: impl Iterator<Item = usize>,
    values: &[f64],
    parallel: bool,
    compare: impl Fn(&(f64, usize), &(f64, usize)) -> Ordering + Sync,
) -> Vec<usize> {
    let mut pairs: Vec<(f64, usize)> = rows.map(|i| (values[i], i)).collect();
    if parallel && pairs.len() > PARALLEL_THRESHOLD_LARGE {
        pairs.par_sort_unstable_by(compare);
    } else {
        pairs.sort_unstable_by(compare);
    }
    pairs.into_iter().map(|(_, i)| i).collect()
}

fn ascending(a: &(f64, usize), b: &(f64, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1))
}

/// `keep[order(values[keep])]` for increasing rows `keep`: the rows of
/// `keep` in ascending order of `values`, ties in row order.  `parallel`
/// splits the sort itself over threads; a caller sorting several subsets
/// at once parallelises over the subsets instead.
pub(crate) fn ordered_subset(keep: &[usize], values: &[f64], parallel: bool) -> Vec<usize> {
    sort_pairs(keep.iter().copied(), values, parallel, ascending)
}

/// Indices that order `values` ascending; ties keep input order. This is
/// R's `order(values)` (zero-based).
pub(crate) fn sorted_indices_by(values: &[f64]) -> Vec<usize> {
    sort_pairs(0..values.len(), values, true, ascending)
}

/// Indices that order `time` descending; ties keep input order. This is the
/// traversal order of the Cox partial likelihood (`coxfit6.c` walks from the
/// largest time down so risk sets accumulate).
pub(crate) fn descending_time_indices(time: &[f64]) -> Vec<usize> {
    sort_pairs(0..time.len(), time, true, |a, b| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descending_time_indices_orders_values_descending() {
        let time = vec![2.0, 5.0, 1.0, 4.0];
        let indices = descending_time_indices(&time);
        let ordered: Vec<f64> = indices.iter().map(|&idx| time[idx]).collect();

        assert_eq!(ordered, vec![5.0, 4.0, 2.0, 1.0]);

        let mut sorted_indices = indices;
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn sorted_indices_by_is_stable_and_ascending() {
        let values = vec![3.0, 1.0, 3.0, -2.0, 1.0];
        assert_eq!(sorted_indices_by(&values), vec![3, 1, 4, 0, 2]);
        assert_eq!(descending_time_indices(&values), vec![0, 2, 1, 4, 3]);
        assert!(sorted_indices_by(&[]).is_empty());
    }

    #[test]
    fn ordered_subset_sorts_only_the_kept_rows() {
        let values = vec![3.0, 1.0, 3.0, -2.0, 1.0];
        assert_eq!(ordered_subset(&[0, 2, 4], &values, false), vec![4, 0, 2]);
        assert!(ordered_subset(&[], &values, true).is_empty());
    }

    #[test]
    fn sorted_indices_by_places_nan_last_without_panicking() {
        let values = vec![f64::NAN, 1.0, f64::INFINITY, -1.0, f64::NEG_INFINITY];
        assert_eq!(sorted_indices_by(&values), vec![4, 3, 1, 2, 0]);
        assert_eq!(descending_time_indices(&values), vec![0, 2, 1, 3, 4]);
    }

    #[test]
    fn parallel_and_serial_paths_agree() {
        let n = PARALLEL_THRESHOLD_LARGE * 2 + 7;
        let values: Vec<f64> = (0..n).map(|i| ((i * 7919) % 97) as f64).collect();
        let parallel = sorted_indices_by(&values);
        let mut serial: Vec<usize> = (0..n).collect();
        serial.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));
        assert_eq!(parallel, serial);
        let all: Vec<usize> = (0..n).collect();
        assert_eq!(ordered_subset(&all, &values, false), serial);

        let parallel = descending_time_indices(&values);
        let mut serial: Vec<usize> = (0..n).collect();
        serial.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
        assert_eq!(parallel, serial);
    }
}
