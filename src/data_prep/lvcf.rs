use super::id_value::{IdValue, SubjectId, first_appearance_codes};
use crate::error::SurvivalResult;
use crate::internal::numpy_utils::FloatVec;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::cmp::Ordering;

fn compare_time(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (false, false) => left.partial_cmp(&right).unwrap(),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

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
    let (subjects, levels) = first_appearance_codes(id);
    let mut current = vec![usize::MAX; levels.len()];
    let mut source_row = |row: usize| {
        let last = &mut current[subjects[row]];
        if !missing[row] || *last == usize::MAX {
            *last = row;
        }
        *last
    };
    let Some(time) = time else {
        // Input order is visit order within each subject, even for interleaved IDs.
        return Ok((0..n).map(source_row).collect());
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        subjects[left]
            .cmp(&subjects[right])
            .then_with(|| compare_time(time[left], time[right]))
    });
    let mut source = vec![0; n];
    for row in order {
        source[row] = source_row(row);
    }
    Ok(source)
}

#[doc = " Python entry point of [`lvcf`]."]
#[pyfunction(name = "lvcf")]
#[pyo3(signature = (id, missing, time=None))]
pub(crate) fn lvcf_py(
    py: Python<'_>,
    id: Vec<IdValue>,
    missing: Vec<bool>,
    time: Option<FloatVec>,
) -> PyResult<Vec<usize>> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(py.detach(|| lvcf(&id, &missing, time.as_deref()))?)
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
    fn ties_preserve_input_order_within_interleaved_subjects() {
        let id = ["a", "b", "b", "a", "a", "b"];
        let missing = [true, false, true, false, true, false];
        let expected = vec![0, 1, 1, 3, 3, 5];
        assert_eq!(lvcf(&id, &missing, None).unwrap(), expected);
        let time = [0.0, 0.0, -0.0, -0.0, 0.0, 0.0];
        assert_eq!(lvcf(&id, &missing, Some(&time)).unwrap(), expected);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(lvcf(&[1i64], &[], None).is_err());
        assert!(lvcf(&[1i64], &[true], Some(&[])).is_err());
        assert!(lvcf::<i64>(&[], &[], None).unwrap().is_empty());
    }
}
