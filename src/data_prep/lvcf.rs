use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;

fn compare_time(left: f64, right: f64) -> Ordering {
    match (left.is_nan(), right.is_nan()) {
        (false, false) => left.total_cmp(&right),
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

#[pyfunction]
#[pyo3(signature = (id, missing, time=None))]
pub fn lvcf_indices(
    py: Python<'_>,
    id: Vec<usize>,
    missing: Vec<bool>,
    time: Option<Vec<f64>>,
) -> PyResult<Vec<usize>> {
    let n = id.len();
    validate_lvcf_lengths(n, missing.len(), time.as_ref().map(Vec::len))?;

    Ok(py.detach(move || lvcf_source_indices_by(id, missing, time, Ord::cmp)))
}

#[pyfunction]
#[pyo3(signature = (id, missing, time=None))]
pub fn lvcf_numeric_indices(
    py: Python<'_>,
    id: &Bound<'_, PyAny>,
    missing: Vec<bool>,
    time: Option<Vec<f64>>,
) -> PyResult<Vec<usize>> {
    if let Ok(integer_id) = id.extract::<Vec<i64>>() {
        let n = integer_id.len();
        validate_lvcf_lengths(n, missing.len(), time.as_ref().map(Vec::len))?;
        return Ok(py.detach(move || lvcf_source_indices_by(integer_id, missing, time, Ord::cmp)));
    }

    let id = id.extract::<Vec<f64>>()?;
    let n = id.len();
    validate_lvcf_lengths(n, missing.len(), time.as_ref().map(Vec::len))?;
    if id.iter().any(|value| value.is_nan()) {
        return Err(PyValueError::new_err("id must not contain missing values"));
    }

    Ok(py.detach(move || lvcf_source_indices_by(id, missing, time, compare_numeric_id)))
}

fn validate_lvcf_lengths(
    id_len: usize,
    missing_len: usize,
    time_len: Option<usize>,
) -> PyResult<()> {
    if missing_len != id_len {
        return Err(PyValueError::new_err(
            "missing must have the same length as id",
        ));
    }
    if time_len.is_some_and(|length| length != id_len) {
        return Err(PyValueError::new_err(
            "time must have the same length as id",
        ));
    }
    Ok(())
}

fn compare_numeric_id(left: &f64, right: &f64) -> Ordering {
    if left == right {
        Ordering::Equal
    } else {
        left.total_cmp(right)
    }
}

fn lvcf_source_indices_by<T, F>(
    id: Vec<T>,
    missing: Vec<bool>,
    time: Option<Vec<f64>>,
    compare_id: F,
) -> Vec<usize>
where
    F: Fn(&T, &T) -> Ordering,
{
    let n = id.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        compare_id(&id[left], &id[right])
            .then_with(|| {
                time.as_ref().map_or(Ordering::Equal, |values| {
                    compare_time(values[left], values[right])
                })
            })
            .then_with(|| left.cmp(&right))
    });

    let mut source: Vec<usize> = (0..n).collect();
    let Some((&first, rest)) = order.split_first() else {
        return source;
    };
    let mut current = first;
    let mut previous = first;
    for &row in rest {
        if !missing[row] || compare_id(&id[row], &id[previous]) != Ordering::Equal {
            current = row;
        } else {
            source[row] = current;
        }
        previous = row;
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_within_sorted_ids() {
        let result =
            lvcf_source_indices_by(vec![2, 1, 1], vec![false, false, true], None, Ord::cmp);
        assert_eq!(result, vec![0, 1, 1]);
    }

    #[test]
    fn sorts_missing_times_last() {
        let result = lvcf_source_indices_by(
            vec![0, 0, 0],
            vec![false, true, false],
            Some(vec![1.0, f64::NAN, 2.0]),
            Ord::cmp,
        );
        assert_eq!(result, vec![0, 2, 2]);
    }

    #[test]
    fn preserves_infinite_time_ordering() {
        let result = lvcf_source_indices_by(
            vec![0, 0, 0],
            vec![false, true, false],
            Some(vec![1.0, f64::NEG_INFINITY, f64::INFINITY]),
            Ord::cmp,
        );
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn empty_inputs_are_supported() {
        assert!(
            lvcf_source_indices_by::<usize, _>(Vec::new(), Vec::new(), None, Ord::cmp).is_empty()
        );
    }

    #[test]
    fn numeric_ids_preserve_python_equality() {
        let result = lvcf_source_indices_by(
            vec![-0.0, 0.0, 2.0],
            vec![false, true, false],
            None,
            compare_numeric_id,
        );
        assert_eq!(result, vec![0, 0, 2]);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(validate_lvcf_lengths(1, 0, None).is_err());
        assert!(validate_lvcf_lengths(1, 1, Some(0)).is_err());
    }
}
