//! Data conventions shared by the ports of R's Cox C routines: the
//! right-censored / (start, stop] response, strata labels, sort orders and
//! interval checks.
//!
//! Every kernel in `residuals`, `scoring` and `core` takes one integer label
//! per observation and expects the data sorted by stratum first (R sorts
//! with `order(strata, time)`), so equal labels form contiguous runs.  R's C
//! code receives the same information as 0/1 markers, `strata[i] == 1` on
//! the last observation of a run (`coxmart.c`, `coxscho.c`) or on the first
//! (`coxcount1.c`); the helpers below derive those markers from the labels.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
use std::cmp::Ordering;

/// The outcome of a Cox-type routine: a right-censored or a (start, stop]
/// `Surv` object.
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

/// R's `Surv()` rule for counting-process data: every interval must have
/// `start < stop` (the kernels' sweeps assume it).
pub(crate) fn validate_intervals(start: &[f64], stop: &[f64]) -> SurvivalResult<()> {
    if let Some(index) = (0..start.len()).find(|&i| start[i] >= stop[i]) {
        return Err(SurvivalError::invalid_input(format!(
            "Stop time must be > start time (row {index}: {} >= {})",
            start[index], stop[index]
        )));
    }
    Ok(())
}

/// `true` on the last observation of each run of equal labels; the final
/// observation is always a run end (the "failsafe" `strata[n-1] = 1` of
/// `coxmart.c`).
pub(crate) fn last_of_run(strata: &[i32]) -> Vec<bool> {
    let n = strata.len();
    (0..n)
        .map(|i| i + 1 == n || strata[i + 1] != strata[i])
        .collect()
}

/// `true` on the first observation of each run of equal labels.
pub(crate) fn first_of_run(strata: &[i32]) -> Vec<bool> {
    (0..strata.len())
        .map(|i| i == 0 || strata[i - 1] != strata[i])
        .collect()
}

/// Stable permutation that sorts by stratum label and then by `within`, the
/// R idiom `order(strata, ...)` that every kernel here expects.
pub(crate) fn order_within_strata(
    strata: &[i32],
    within: impl Fn(usize, usize) -> Ordering,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..strata.len()).collect();
    order.sort_by(|&a, &b| strata[a].cmp(&strata[b]).then_with(|| within(a, b)));
    order
}

/// Row indices grouped by stratum label, labels ascending; each group keeps
/// the input order.  This is how `concordancefit` and `agreg.fit` iterate
/// over strata, and how `rowsum` groups by cluster.  One stable sort of the
/// row indices followed by a split into runs of equal labels, `O(n log n)`
/// however many labels there are (one cluster per subject is the common
/// case).
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

#[cfg(test)]
mod tests {
    use super::*;

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
        // Every row its own group (one cluster per subject): rows keep
        // their input order within a group and labels come out ascending.
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
}
