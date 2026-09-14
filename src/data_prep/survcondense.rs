//! R's `survcondense` (`R/survcondense.R`): the inverse of `survSplit`.
//! Adjacent rows of a subject whose covariates agree and whose intervals
//! abut, `(0, 10]` followed by `(10, 25]`, are merged into one row
//! `(0, 25]` carrying the values of the last row.
//!
//! As in R, only the covariates (summarised by `row_code`) and the times
//! decide whether rows merge; the status column plays no part, so a
//! caller wanting to keep interior events must fold the status into
//! `row_code`.

use super::id_value::{IdValue, SubjectId};
use crate::error::SurvivalResult;
use crate::internal::validation::{validate_finite, validate_length};
use pyo3::prelude::*;

/// Which rows survive and their (possibly moved back) start times.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvcondenseResult {
    /// Start time of every input row after merging; only the rows in
    /// `keep` are meaningful.
    #[pyo3(get)]
    pub start: Vec<f64>,
    /// Zero-based input rows to keep, in input order.
    #[pyo3(get)]
    pub keep: Vec<usize>,
}

/// Condense `(start, stop]` rows.  `row_code` identifies the covariate
/// combination of each row (equal codes may merge); rows are ordered by
/// stop time within subject, as R does.
pub fn survcondense<I: SubjectId>(
    id: &[I],
    start: &[f64],
    stop: &[f64],
    row_code: &[i64],
) -> SurvivalResult<SurvcondenseResult> {
    let n = id.len();
    validate_length(n, start.len(), "start")?;
    validate_length(n, stop.len(), "stop")?;
    validate_length(n, row_code.len(), "row_code")?;
    validate_finite(start, "start")?;
    validate_finite(stop, "stop")?;

    let keys: Vec<I::Key> = id.iter().map(SubjectId::key).collect();
    let mut index: Vec<usize> = (0..n).collect();
    index.sort_by(|&a, &b| {
        keys[a]
            .cmp(&keys[b])
            .then_with(|| stop[a].total_cmp(&stop[b]))
    });

    // droprow[p]: sorted row p merges into sorted row p + 1.
    let droprow: Vec<bool> = (0..n)
        .map(|p| {
            p + 1 < n && {
                let (cur, next) = (index[p], index[p + 1]);
                keys[cur] == keys[next]
                    && row_code[cur] == row_code[next]
                    && start[next] == stop[cur]
            }
        })
        .collect();

    // The first row after each run of dropped rows inherits the start time
    // of the first row of the run.
    let mut new_start = start.to_vec();
    let mut p = 0;
    while p < n {
        if !droprow[p] {
            p += 1;
            continue;
        }
        let run_start = p;
        while p < n && droprow[p] {
            p += 1;
        }
        // droprow is always false for the last sorted row, so p < n here.
        new_start[index[p]] = start[index[run_start]];
    }
    let mut dropped = vec![false; n];
    for (p, &drop) in droprow.iter().enumerate() {
        if drop {
            dropped[index[p]] = true;
        }
    }
    Ok(SurvcondenseResult {
        start: new_start,
        keep: (0..n).filter(|&i| !dropped[i]).collect(),
    })
}

/// Python entry point of [`survcondense`].
#[pyfunction(name = "survcondense")]
pub fn survcondense_py(
    id: Vec<IdValue>,
    start: Vec<f64>,
    stop: Vec<f64>,
    row_code: Vec<i64>,
) -> PyResult<SurvcondenseResult> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(survcondense(&id, &start, &stop, &row_code)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_abutting_rows_with_equal_covariates() {
        let out = survcondense(
            &[1i64, 1, 1, 2, 2],
            &[0.0, 5.0, 10.0, 0.0, 3.0],
            &[5.0, 10.0, 15.0, 3.0, 8.0],
            &[7, 7, 7, 1, 2],
        )
        .unwrap();
        assert_eq!(out.keep, vec![2, 3, 4]);
        assert_eq!(out.start[2], 0.0);
        assert_eq!(out.start[3], 0.0);
        assert_eq!(out.start[4], 3.0);
    }

    #[test]
    fn gaps_and_covariate_changes_block_merging() {
        let out = survcondense(
            &[1i64, 1, 1],
            &[0.0, 6.0, 10.0],
            &[5.0, 10.0, 15.0],
            &[1, 1, 2],
        )
        .unwrap();
        assert_eq!(out.keep, vec![0, 1, 2]);
        assert_eq!(out.start, vec![0.0, 6.0, 10.0]);
    }

    #[test]
    fn unsorted_input_uses_sorted_positions_not_raw_rows() {
        // Subject 1's rows appear last; a run at the start of the sorted
        // order must not touch raw row 0.
        let out = survcondense(
            &["b", "b", "a", "a", "a"],
            &[0.0, 4.0, 0.0, 2.0, 5.0],
            &[4.0, 9.0, 2.0, 5.0, 9.0],
            &[1, 2, 1, 1, 1],
        )
        .unwrap();
        assert_eq!(out.keep, vec![0, 1, 4]);
        assert_eq!(out.start, vec![0.0, 4.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn every_permutation_of_the_input_gives_the_same_condensed_rows() {
        use crate::tests::common::index_permutations;
        let id = [1i64, 1, 1, 2, 2];
        let start = [0.0, 2.0, 4.0, 0.0, 3.0];
        let stop = [2.0, 4.0, 6.0, 3.0, 5.0];
        for permutation in index_permutations(id.len()) {
            let pid: Vec<i64> = permutation.iter().map(|&i| id[i]).collect();
            let pstart: Vec<f64> = permutation.iter().map(|&i| start[i]).collect();
            let pstop: Vec<f64> = permutation.iter().map(|&i| stop[i]).collect();
            let out = survcondense(&pid, &pstart, &pstop, &[0; 5]).unwrap();
            let mut rows: Vec<(i64, f64, f64)> = out
                .keep
                .iter()
                .map(|&k| (pid[k], out.start[k], pstop[k]))
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(rows, vec![(1, 0.0, 6.0), (2, 0.0, 5.0)]);
        }
    }

    #[test]
    fn rejects_mismatched_or_non_finite_inputs() {
        assert!(survcondense(&[1i64], &[], &[1.0], &[0]).is_err());
        assert!(survcondense(&[1i64], &[f64::NAN], &[1.0], &[0]).is_err());
        assert!(
            survcondense::<i64>(&[], &[], &[], &[])
                .unwrap()
                .keep
                .is_empty()
        );
    }
}
