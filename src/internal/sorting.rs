//! Canonical index-sort helpers.
//!
//! Both functions return a permutation of `0..values.len()` rather than
//! moving the data, so several parallel vectors can be reordered together.
//! Ordering is `f64::total_cmp`, so `NaN` never panics and sorts after
//! `+inf` (before `-inf` when descending); ties keep their original order
//! (stable), which is what R's `order()` does.

use crate::constants::PARALLEL_THRESHOLD_LARGE;
use rayon::prelude::*;
use std::cmp::Ordering;

fn sort_indices_by(n: usize, compare: impl Fn(usize, usize) -> Ordering + Sync) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    if n > PARALLEL_THRESHOLD_LARGE {
        indices.par_sort_by(|&a, &b| compare(a, b));
    } else {
        indices.sort_by(|&a, &b| compare(a, b));
    }
    indices
}

/// Indices that order `values` ascending; ties keep input order. This is
/// R's `order(values)` (zero-based).
// Canonical helper; `surv_analysis/survfitkm.rs` and `surv_analysis/nelson_aalen.rs`
// still carry private copies and are expected to migrate to this one.
#[allow(dead_code)]
pub(crate) fn sorted_indices_by(values: &[f64]) -> Vec<usize> {
    sort_indices_by(values.len(), |a, b| {
        values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b))
    })
}

/// Indices that order `time` descending; ties keep input order. This is the
/// traversal order of the Cox partial likelihood (`coxfit6.c` walks from the
/// largest time down so risk sets accumulate).
pub(crate) fn descending_time_indices(time: &[f64]) -> Vec<usize> {
    sort_indices_by(time.len(), |a, b| {
        time[b].total_cmp(&time[a]).then_with(|| a.cmp(&b))
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

        let parallel = descending_time_indices(&values);
        let mut serial: Vec<usize> = (0..n).collect();
        serial.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then_with(|| a.cmp(&b)));
        assert_eq!(parallel, serial);
    }
}
