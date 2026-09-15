//! R's `lvcf` (`R/xtras.R`): last value carried forward within subject.
//!
//! Only the bookkeeping lives here: for every row, the row whose value it
//! should take.  The caller copies values (of any type) and applies R's
//! `first = TRUE` rule, which turns a missing first value of a 0/1 or
//! logical variable into 0.

use super::id_value::{IdValue, SubjectId};
use crate::error::SurvivalResult;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::cmp::Ordering;

/// R's `order(id, time)` treats missing times as larger than any value.
fn compare_time(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (false, false) => left.total_cmp(&right),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

/// For each row, the zero-based row whose value is carried into it: the
/// row itself when its value is present or it is the first row of its
/// subject, otherwise the most recent row of the subject with a value
/// (or that subject's first row when none has one yet).
pub fn lvcf<I: SubjectId>(
    id: &[I],
    missing: &[bool],
    time: Option<&[f64]>,
) -> SurvivalResult<Vec<usize>> {
    let n = id.len();
    validate_length(n, missing.len(), "missing")?;
    if let Some(time) = time {
        validate_length(n, time.len(), "time")?;
    }
    let keys: Vec<I::Key> = id.iter().map(SubjectId::key).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        keys[left]
            .cmp(&keys[right])
            .then_with(|| time.map_or(Ordering::Equal, |t| compare_time(t[left], t[right])))
    });

    let mut source: Vec<usize> = (0..n).collect();
    let mut current = 0;
    let mut previous: Option<usize> = None;
    for &row in &order {
        let new_subject = previous.is_none_or(|p| keys[p] != keys[row]);
        if new_subject || !missing[row] {
            current = row;
        } else {
            source[row] = current;
        }
        previous = Some(row);
    }
    Ok(source)
}

/// Python entry point of [`lvcf`].
#[pyfunction(name = "lvcf")]
#[pyo3(signature = (id, missing, time=None))]
pub fn lvcf_py(
    id: Vec<IdValue>,
    missing: Vec<bool>,
    time: Option<Vec<f64>>,
) -> PyResult<Vec<usize>> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(lvcf(&id, &missing, time.as_deref())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_within_sorted_ids() {
        assert_eq!(
            lvcf(&[2i64, 1, 1], &[false, false, true], None).unwrap(),
            vec![0, 1, 1]
        );
        assert_eq!(
            lvcf(&["a", "a", "a", "b"], &[true, true, false, true], None).unwrap(),
            vec![0, 0, 2, 3]
        );
    }

    #[test]
    fn time_orders_within_subject_with_missing_times_last() {
        let result = lvcf(
            &[0i64, 0, 0],
            &[false, true, false],
            Some(&[1.0, f64::NAN, 2.0]),
        )
        .unwrap();
        assert_eq!(result, vec![0, 2, 2]);
        let result = lvcf(&[0i64, 0], &[false, true], Some(&[2.0, 1.0])).unwrap();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(lvcf(&[1i64], &[], None).is_err());
        assert!(lvcf(&[1i64], &[true], Some(&[])).is_err());
        assert!(lvcf::<i64>(&[], &[], None).unwrap().is_empty());
    }
}
