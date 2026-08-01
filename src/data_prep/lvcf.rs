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
    id: Vec<usize>,
    missing: Vec<bool>,
    time: Option<Vec<f64>>,
) -> PyResult<Vec<usize>> {
    let n = id.len();
    if missing.len() != n {
        return Err(PyValueError::new_err(
            "missing must have the same length as id",
        ));
    }
    if time.as_ref().is_some_and(|values| values.len() != n) {
        return Err(PyValueError::new_err(
            "time must have the same length as id",
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| {
        id[left]
            .cmp(&id[right])
            .then_with(|| {
                time.as_ref().map_or(Ordering::Equal, |values| {
                    compare_time(values[left], values[right])
                })
            })
            .then_with(|| left.cmp(&right))
    });

    let mut source: Vec<usize> = (0..n).collect();
    let Some((&first, rest)) = order.split_first() else {
        return Ok(source);
    };
    let mut current = first;
    let mut previous_id = id[first];
    for &row in rest {
        if !missing[row] || id[row] != previous_id {
            current = row;
        } else {
            source[row] = current;
        }
        previous_id = id[row];
    }
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_within_sorted_ids() {
        let result = lvcf_indices(vec![2, 1, 1], vec![false, false, true], None).unwrap();
        assert_eq!(result, vec![0, 1, 1]);
    }

    #[test]
    fn sorts_missing_times_last() {
        let result = lvcf_indices(
            vec![0, 0, 0],
            vec![false, true, false],
            Some(vec![1.0, f64::NAN, 2.0]),
        )
        .unwrap();
        assert_eq!(result, vec![0, 2, 2]);
    }

    #[test]
    fn preserves_infinite_time_ordering() {
        let result = lvcf_indices(
            vec![0, 0, 0],
            vec![false, true, false],
            Some(vec![1.0, f64::NEG_INFINITY, f64::INFINITY]),
        )
        .unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(lvcf_indices(vec![0], vec![], None).is_err());
        assert!(lvcf_indices(vec![0], vec![false], Some(vec![])).is_err());
    }
}
