use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub enum SurvResponse<'a> {
    Right(&'a SurvivalData),
    Counting(&'a CountingProcessData),
}

impl<'a> SurvResponse<'a> {
    pub(crate) fn start(&self) -> Option<&'a [f64]> {
        match self {
            Self::Right(_) => None,
            Self::Counting(data) => Some(&data.start),
        }
    }

    pub(crate) fn stop(&self) -> &'a [f64] {
        match self {
            Self::Right(data) => &data.time,
            Self::Counting(data) => &data.stop,
        }
    }

    pub(crate) fn status(&self) -> &'a [i32] {
        match self {
            Self::Right(data) => &data.status,
            Self::Counting(data) => &data.event,
        }
    }
}

pub(crate) fn validate_intervals(start: &[f64], stop: &[f64]) -> SurvivalResult<()> {
    if let Some(index) = (0..start.len()).find(|&i| start[i] >= stop[i]) {
        return Err(SurvivalError::invalid_input(format!(
            "Stop time must be > start time (row {index}: {} >= {})",
            start[index], stop[index]
        )));
    }
    Ok(())
}

pub(crate) fn last_of_run(strata: &[i32]) -> Vec<bool> {
    let n = strata.len();
    (0..n)
        .map(|i| i + 1 == n || strata[i + 1] != strata[i])
        .collect()
}

pub(crate) fn first_of_run(strata: &[i32]) -> Vec<bool> {
    (0..strata.len())
        .map(|i| i == 0 || strata[i - 1] != strata[i])
        .collect()
}

pub(crate) fn order_within_strata(
    strata: &[i32],
    within: impl Fn(usize, usize) -> Ordering,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..strata.len()).collect();
    order.sort_by(|&a, &b| strata[a].cmp(&strata[b]).then_with(|| within(a, b)));
    order
}

pub(crate) fn stratum_groups(strata: &[i32]) -> Vec<(i32, Vec<usize>)> {
    let mut order: Vec<usize> = (0..strata.len()).collect();
    order.sort_by_key(|&i| strata[i]);
    let mut groups: Vec<(i32, Vec<usize>)> = Vec::new();
    for i in order {
        match groups.last_mut() {
            Some((label, rows)) if *label == strata[i] => rows.push(i),
            _ => groups.push((strata[i], vec![i])),
        }
    }
    groups
}

/// Reorders `order`, a permutation sorted by (stratum, `total_cmp` time, row),
/// into the (stratum, time, status descending, row) order that the score
/// kernels sort by, in O(n): within each run of the same stratum and time
/// bits the events come first, each part keeping its row order.  Runs split
/// on the bits because the sort keeps -0.0 and +0.0 apart; `status` is 0/1.
pub(crate) fn events_first_within_ties(
    order: &[usize],
    strata: &[i32],
    time: &[f64],
    status: &[i32],
) -> Vec<usize> {
    let key = |row: usize| (strata[row], time[row].to_bits());
    let mut result = Vec::with_capacity(order.len());
    let mut first = 0;
    while first < order.len() {
        let run_key = key(order[first]);
        let end = order[first..]
            .iter()
            .position(|&row| key(row) != run_key)
            .map_or(order.len(), |length| first + length);
        let run = &order[first..end];
        result.extend(run.iter().filter(|&&row| status[row] == 1));
        result.extend(run.iter().filter(|&&row| status[row] != 1));
        first = end;
    }
    result
}

/// The groups of [`stratum_groups`] as one dense id per row: group `g` holds
/// the `g`-th smallest label.  Totals scattered by these ids in row order add
/// each group's rows in the same order as the sorted groups do, without
/// sorting the rows or building a list per group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DenseGroups {
    pub ids: Vec<usize>,
    pub count: usize,
}

impl DenseGroups {
    pub(crate) fn new(labels: &[i32]) -> Self {
        let (Some(&min), Some(&max)) = (labels.iter().min(), labels.iter().max()) else {
            return Self {
                ids: Vec::new(),
                count: 0,
            };
        };
        let slot = |label: i32| (i64::from(label) - i64::from(min)) as usize;
        let span = slot(max) + 1;
        if span <= 4 * labels.len() {
            // Compact labels (cluster codes, ids): rank through a table.
            let mut rank = vec![0usize; span];
            for &label in labels {
                rank[slot(label)] = 1;
            }
            let mut count = 0;
            for entry in &mut rank {
                let present = *entry;
                *entry = count;
                count += present;
            }
            let ids = labels.iter().map(|&label| rank[slot(label)]).collect();
            return Self { ids, count };
        }
        let mut distinct = labels.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let ids = labels
            .iter()
            .map(|label| {
                distinct
                    .binary_search(label)
                    .expect("every label is among the distinct labels")
            })
            .collect();
        Self {
            ids,
            count: distinct.len(),
        }
    }

    /// Every row its own group, as `rowsum(x, seq_len(n))`.
    pub(crate) fn identity(n: usize) -> Self {
        Self {
            ids: (0..n).collect(),
            count: n,
        }
    }
}

/// Adds `term(row, value)` of each entry to its group's total, in row order
/// from zero, as `stratum_groups` followed by a sum per group does.
fn sum_rows_by_group(
    values: ArrayView2<'_, f64>,
    groups: &DenseGroups,
    term: impl Fn(usize, f64) -> f64,
) -> Array2<f64> {
    let ncols = values.ncols();
    let mut sums = Array2::zeros((groups.count, ncols));
    if ncols == 0 {
        // Nothing to add, and rows of width zero cannot be chunked.
        return sums;
    }
    let values = values.as_standard_layout();
    let rows = values.as_slice().expect("standard layout is contiguous");
    let totals = sums.as_slice_mut().expect("a new array is contiguous");
    for (row, (entries, &id)) in rows.chunks_exact(ncols).zip(&groups.ids).enumerate() {
        for (total, &value) in totals[id * ncols..(id + 1) * ncols].iter_mut().zip(entries) {
            *total += term(row, value);
        }
    }
    sums
}

/// R's `rowsum(values * weights, group)` in one pass.  Each row is scaled
/// before it joins its group's total, as scaling a copy first would do.
pub(crate) fn weighted_rowsum(
    values: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    groups: &DenseGroups,
) -> Array2<f64> {
    match weights {
        Some(weights) => sum_rows_by_group(values, groups, |row, value| value * weights[row]),
        None => sum_rows_by_group(values, groups, |_, value| value),
    }
}

pub(crate) fn rowsum(values: ArrayView2<'_, f64>, group: &[i32]) -> Array2<f64> {
    grouped_sum(values, group, false).expect("internal rowsum groups match the matrix rows")
}

pub fn grouped_sum(
    values: ArrayView2<'_, f64>,
    group: &[i32],
    squares: bool,
) -> SurvivalResult<Array2<f64>> {
    if values.nrows() != group.len() {
        return Err(SurvivalError::invalid_input(
            "group must have one value per matrix row",
        ));
    }
    let groups = DenseGroups::new(group);
    if !squares {
        return Ok(weighted_rowsum(values, None, &groups));
    }
    let mut sums = sum_rows_by_group(values, &groups, |_, value| value * value);
    sums.mapv_inplace(f64::sqrt);
    Ok(sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::rng::Rng;

    #[test]
    fn run_markers_follow_label_changes() {
        let strata = [3, 3, 1, 1, 1, 2];
        assert_eq!(
            last_of_run(&strata),
            vec![false, true, false, false, true, true]
        );
        assert_eq!(
            first_of_run(&strata),
            vec![true, false, true, false, false, true]
        );
        assert!(last_of_run(&[]).is_empty());
        assert_eq!(last_of_run(&[7]), vec![true]);
    }

    #[test]
    fn order_within_strata_is_stable_and_stratum_major() {
        let strata = [2, 1, 2, 1, 1];
        let time: [f64; 5] = [5.0, 3.0, 1.0, 3.0, 2.0];
        let order = order_within_strata(&strata, |a, b| time[a].total_cmp(&time[b]));
        assert_eq!(order, vec![4, 1, 3, 2, 0]);
    }

    #[test]
    fn intervals_must_have_positive_length() {
        assert!(validate_intervals(&[0.0, 1.0], &[1.0, 2.0]).is_ok());
        let err = validate_intervals(&[0.0, 2.0], &[1.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("Stop time must be > start time"));
    }

    #[test]
    fn stratum_groups_are_label_sorted() {
        let groups = stratum_groups(&[2, 1, 2, 1]);
        assert_eq!(groups, vec![(1, vec![1, 3]), (2, vec![0, 2])]);
        assert!(stratum_groups(&[]).is_empty());
        let strata = [5, -1, 3, 5, 0, 3];
        assert_eq!(
            stratum_groups(&strata),
            vec![
                (-1, vec![1]),
                (0, vec![4]),
                (3, vec![2, 5]),
                (5, vec![0, 3])
            ]
        );
    }

    #[test]
    fn rowsum_sums_rows_by_ascending_label() {
        let values = ndarray::array![[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]];
        let sums = rowsum(values.view(), &[2, 1, 2, 5]);
        assert_eq!(sums, ndarray::array![[2.0, 6.0], [4.0, 12.0], [4.0, 8.0]]);
        assert_eq!(rowsum(Array2::zeros((0, 2)).view(), &[]).dim(), (0, 2));
    }

    #[test]
    fn grouped_sum_validates_and_preserves_empty_shapes_and_addition_order() {
        let values = ndarray::array![[1e16, 3.0], [-1e16, 4.0], [1.0, 0.0]];
        assert_eq!(
            grouped_sum(values.view(), &[-3, -3, -3], false).unwrap(),
            ndarray::array![[1.0, 7.0]]
        );
        assert_eq!(
            grouped_sum(values.view(), &[1, 2], false)
                .unwrap_err()
                .to_string(),
            "group must have one value per matrix row"
        );
        assert_eq!(
            grouped_sum(Array2::zeros((0, 3)).view(), &[], true)
                .unwrap()
                .dim(),
            (0, 3)
        );
        assert_eq!(
            grouped_sum(Array2::zeros((3, 0)).view(), &[1, 2, 1], false)
                .unwrap()
                .dim(),
            (2, 0)
        );
    }

    /// The sort-based `grouped_sum` that `DenseGroups` replaced.
    fn grouped_sum_by_sorting(
        values: ArrayView2<'_, f64>,
        group: &[i32],
        squares: bool,
    ) -> Array2<f64> {
        let groups = stratum_groups(group);
        let mut sums = Array2::zeros((groups.len(), values.ncols()));
        for (mut sum, (_, rows)) in sums.outer_iter_mut().zip(&groups) {
            for &row in rows {
                if squares {
                    for (total, value) in sum.iter_mut().zip(values.row(row)) {
                        *total += value * value;
                    }
                } else {
                    sum += &values.row(row);
                }
            }
        }
        if squares {
            sums.mapv_inplace(f64::sqrt);
        }
        sums
    }

    fn bits(values: &Array2<f64>) -> Array2<u64> {
        values.mapv(f64::to_bits)
    }

    #[test]
    fn dense_groups_rank_labels_like_the_sorted_groups() {
        let mut draws = Rng::with_seed(7);
        let values = [
            1e16,
            -1e16,
            1.0,
            -0.0,
            0.0,
            3.5,
            -2.25,
            1e-300,
            f64::NAN,
            f64::INFINITY,
        ];
        for case in 0..400 {
            let n = draws.usize(0..40);
            let ncols = draws.usize(0..4);
            let scale = [1, 3, 1000, 1 << 20][case % 4];
            let group: Vec<i32> = (0..n)
                .map(|_| match case % 5 {
                    4 => draws.pick(&[i32::MIN, -1, 0, i32::MAX]),
                    _ => (draws.usize(0..6) as i32 - 2) * scale,
                })
                .collect();
            let matrix = Array2::from_shape_fn((n, ncols), |_| {
                if case.is_multiple_of(3) {
                    draws.pick(&values[..8])
                } else {
                    draws.pick(&values)
                }
            });
            let weights: Vec<f64> = (0..n).map(|_| draws.pick(&[0.0, 0.5, 1.0, 3.0])).collect();
            let groups = DenseGroups::new(&group);
            let sorted = stratum_groups(&group);
            assert_eq!(groups.count, sorted.len());
            for (id, (_, rows)) in sorted.iter().enumerate() {
                assert!(rows.iter().all(|&row| groups.ids[row] == id));
            }
            for squares in [false, true] {
                assert_eq!(
                    bits(&grouped_sum(matrix.view(), &group, squares).unwrap()),
                    bits(&grouped_sum_by_sorting(matrix.view(), &group, squares)),
                );
            }
            let mut scaled = matrix.clone();
            for (mut row, weight) in scaled.outer_iter_mut().zip(&weights) {
                row.mapv_inplace(|value| value * weight);
            }
            assert_eq!(
                bits(&weighted_rowsum(matrix.view(), Some(&weights), &groups)),
                bits(&grouped_sum_by_sorting(scaled.view(), &group, false)),
            );
            let rows: Vec<i32> = (0..n as i32).collect();
            assert_eq!(
                bits(&weighted_rowsum(
                    matrix.view(),
                    Some(&weights),
                    &DenseGroups::identity(n)
                )),
                bits(&grouped_sum_by_sorting(scaled.view(), &rows, false)),
            );
        }
        let transposed = ndarray::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(
            rowsum(transposed.t(), &[1, 0, 1]),
            ndarray::array![[2.0, 5.0], [4.0, 10.0]]
        );
    }

    #[test]
    fn events_first_within_ties_matches_the_status_sort() {
        let mut draws = Rng::with_seed(11);
        for case in 0..2000_usize {
            let n = draws.usize(0..30);
            let strata: Vec<i32> = (0..n)
                .map(|_| {
                    if case.is_multiple_of(2) {
                        0
                    } else {
                        draws.pick(&[-3, 0, 4])
                    }
                })
                .collect();
            let time: Vec<f64> = (0..n)
                .map(|_| draws.pick(&[-0.0, 0.0, 1.0, 2.0, 2.0, -1.5]))
                .collect();
            let status: Vec<i32> = (0..n).map(|_| draws.pick(&[0, 1])).collect();
            let fit_order = order_within_strata(&strata, |a, b| time[a].total_cmp(&time[b]));
            let expected = order_within_strata(&strata, |a, b| {
                time[a]
                    .total_cmp(&time[b])
                    .then_with(|| status[b].cmp(&status[a]))
            });
            assert_eq!(
                events_first_within_ties(&fit_order, &strata, &time, &status),
                expected
            );
        }
    }

    #[test]
    fn grouped_sum_errors_use_quadrature_and_propagate_nonfinite_entries() {
        let values = ndarray::array![[3.0, f64::NAN], [4.0, 2.0], [2.0, f64::INFINITY]];
        let sums = grouped_sum(values.view(), &[2, 2, 1], true).unwrap();
        assert_eq!(sums[(0, 0)], 2.0);
        assert!(sums[(0, 1)].is_infinite());
        assert_eq!(sums[(1, 0)], 5.0);
        assert!(sums[(1, 1)].is_nan());
    }
}
