use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Result of nearest date matching
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NearDateResult {
    /// Index into the reference set (id2/date2) for each query, None if no match
    #[pyo3(get)]
    pub indices: Vec<Option<usize>>,
    /// Distance (in same units as input) to the matched value, None if no match
    #[pyo3(get)]
    pub distances: Vec<Option<f64>>,
    /// Number of successful matches
    #[pyo3(get)]
    pub n_matched: usize,
}

fn validate_dates(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "{} values must be finite, got non-finite value at index {}",
                name, idx
            )));
        }
    }
    Ok(())
}

fn validate_direction(best: Option<&str>) -> PyResult<&'static str> {
    let direction = best.unwrap_or("closest");
    let mut matches = ["prior", "after", "closest"]
        .into_iter()
        .filter(|choice| choice.starts_with(direction));
    let Some(first) = matches.next() else {
        return Err(PyErr::new::<PyValueError, _>(
            "best must be 'prior', 'after', or 'closest'",
        ));
    };
    if direction.is_empty() || matches.next().is_some() {
        return Err(PyErr::new::<PyValueError, _>(
            "best must be 'prior', 'after', or 'closest'",
        ));
    }
    Ok(first)
}

/// Find the closest matching date/value in a reference set.
///
/// For each observation in the query set (id1, date1), finds the closest
/// matching date in the reference set (id2, date2) within the same ID.
///
/// # Arguments
/// * `id1` - IDs for the query observations
/// * `date1` - Dates/values for the query observations
/// * `id2` - IDs for the reference observations
/// * `date2` - Dates/values for the reference observations
/// * `best` - Direction to search: "prior" (<=), "after" (>=), or "closest" (default)
/// * `nomatch` - Value to return for non-matches (index). If None, returns None.
///
/// # Returns
/// * `NearDateResult` with indices into reference set and distances
#[pyfunction]
#[pyo3(signature = (id1, date1, id2, date2, best=None, nomatch=None))]
pub fn neardate(
    id1: Vec<i64>,
    date1: Vec<f64>,
    id2: Vec<i64>,
    date2: Vec<f64>,
    best: Option<&str>,
    nomatch: Option<usize>,
) -> PyResult<NearDateResult> {
    let n1 = id1.len();
    let n2 = id2.len();

    if date1.len() != n1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "id1 and date1 must have same length",
        ));
    }
    if date2.len() != n2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "id2 and date2 must have same length",
        ));
    }
    validate_dates("date1", &date1)?;
    validate_dates("date2", &date2)?;

    let direction = validate_direction(best)?;

    let mut ref_by_id: HashMap<i64, Vec<(usize, f64)>> = HashMap::new();
    for (idx, (&id, &date)) in id2.iter().zip(date2.iter()).enumerate() {
        ref_by_id.entry(id).or_default().push((idx, date));
    }

    for entries in ref_by_id.values_mut() {
        entries.sort_by(|a, b| a.1.total_cmp(&b.1));
    }

    let mut indices = Vec::with_capacity(n1);
    let mut distances = Vec::with_capacity(n1);
    let mut n_matched = 0;

    for (&id, &date) in id1.iter().zip(date1.iter()) {
        let match_result = if let Some(refs) = ref_by_id.get(&id) {
            find_nearest(refs, date, direction)
        } else {
            None
        };

        match match_result {
            Some((idx, dist)) => {
                indices.push(Some(idx));
                distances.push(Some(dist));
                n_matched += 1;
            }
            None => {
                indices.push(nomatch);
                distances.push(None);
            }
        }
    }

    Ok(NearDateResult {
        indices,
        distances,
        n_matched,
    })
}

/// Find nearest value in sorted reference list
fn find_nearest(refs: &[(usize, f64)], target: f64, direction: &str) -> Option<(usize, f64)> {
    if refs.is_empty() {
        return None;
    }

    match direction {
        "prior" => {
            let pos = refs.partition_point(|entry| entry.1 <= target);
            if pos == 0 {
                None
            } else {
                let (idx, val) = refs[pos - 1];
                Some((idx, target - val))
            }
        }
        "after" => {
            let pos = refs.partition_point(|entry| entry.1 < target);
            refs.get(pos).map(|&(idx, val)| (idx, val - target))
        }
        "closest" => {
            let pos = refs.partition_point(|entry| entry.1 < target);
            if pos == 0 {
                let (idx, val) = refs[0];
                return Some((idx, val - target));
            }
            if pos == refs.len() {
                let (idx, val) = refs[refs.len() - 1];
                return Some((idx, target - val));
            }

            let (_, before_val) = refs[pos - 1];
            let (after_idx, after_val) = refs[pos];
            let before_dist = target - before_val;
            let after_dist = after_val - target;
            if before_dist <= after_dist {
                let first_before_pos = refs.partition_point(|entry| entry.1 < before_val);
                Some((refs[first_before_pos].0, before_dist))
            } else {
                Some((after_idx, after_dist))
            }
        }
        _ => None,
    }
}

/// Find nearest date using string IDs
#[pyfunction]
#[pyo3(signature = (id1, date1, id2, date2, best=None, nomatch=None))]
pub fn neardate_str(
    id1: Vec<String>,
    date1: Vec<f64>,
    id2: Vec<String>,
    date2: Vec<f64>,
    best: Option<&str>,
    nomatch: Option<usize>,
) -> PyResult<NearDateResult> {
    let n1 = id1.len();
    let n2 = id2.len();

    if date1.len() != n1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "id1 and date1 must have same length",
        ));
    }
    if date2.len() != n2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "id2 and date2 must have same length",
        ));
    }
    validate_dates("date1", &date1)?;
    validate_dates("date2", &date2)?;

    let direction = validate_direction(best)?;

    let mut ref_by_id: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    for (idx, (id, &date)) in id2.iter().zip(date2.iter()).enumerate() {
        ref_by_id.entry(id.clone()).or_default().push((idx, date));
    }

    for entries in ref_by_id.values_mut() {
        entries.sort_by(|a, b| a.1.total_cmp(&b.1));
    }

    let mut indices = Vec::with_capacity(n1);
    let mut distances = Vec::with_capacity(n1);
    let mut n_matched = 0;

    for (id, &date) in id1.iter().zip(date1.iter()) {
        let match_result = if let Some(refs) = ref_by_id.get(id) {
            find_nearest(refs, date, direction)
        } else {
            None
        };

        match match_result {
            Some((idx, dist)) => {
                indices.push(Some(idx));
                distances.push(Some(dist));
                n_matched += 1;
            }
            None => {
                indices.push(nomatch);
                distances.push(None);
            }
        }
    }

    Ok(NearDateResult {
        indices,
        distances,
        n_matched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neardate_basic() {
        let id1 = vec![1, 1, 2];
        let date1 = vec![5.0, 15.0, 10.0];
        let id2 = vec![1, 1, 1, 2, 2];
        let date2 = vec![1.0, 10.0, 20.0, 5.0, 15.0];

        let result = neardate(id1, date1, id2, date2, Some("closest"), None).unwrap();
        assert_eq!(result.n_matched, 3);
    }

    #[test]
    fn test_neardate_prior() {
        let id1 = vec![1];
        let date1 = vec![15.0];
        let id2 = vec![1, 1, 1];
        let date2 = vec![10.0, 20.0, 5.0];

        let result = neardate(id1, date1, id2, date2, Some("prior"), None).unwrap();
        assert_eq!(result.n_matched, 1);
        assert_eq!(result.indices[0], Some(0));
    }

    #[test]
    fn test_neardate_after() {
        let id1 = vec![1];
        let date1 = vec![15.0];
        let id2 = vec![1, 1, 1];
        let date2 = vec![10.0, 20.0, 25.0];

        let result = neardate(id1, date1, id2, date2, Some("after"), None).unwrap();
        assert_eq!(result.n_matched, 1);
        assert_eq!(result.indices[0], Some(1));
    }

    #[test]
    fn test_neardate_accepts_unique_best_prefixes() {
        let after = neardate(
            vec![1],
            vec![15.0],
            vec![1, 1],
            vec![10.0, 20.0],
            Some("a"),
            None,
        )
        .unwrap();
        assert_eq!(after.indices, vec![Some(1)]);

        let prior = neardate(
            vec![1],
            vec![15.0],
            vec![1, 1],
            vec![10.0, 20.0],
            Some("pr"),
            None,
        )
        .unwrap();
        assert_eq!(prior.indices, vec![Some(0)]);

        let closest = neardate(
            vec![1],
            vec![18.0],
            vec![1, 1],
            vec![10.0, 20.0],
            Some("cl"),
            None,
        )
        .unwrap();
        assert_eq!(closest.indices, vec![Some(1)]);
    }

    #[test]
    fn test_neardate_no_match() {
        let id1 = vec![1];
        let date1 = vec![10.0];
        let id2 = vec![2];
        let date2 = vec![10.0];

        let result = neardate(id1, date1, id2, date2, None, None).unwrap();
        assert_eq!(result.n_matched, 0);
        assert_eq!(result.indices[0], None);
    }

    #[test]
    fn test_neardate_preserves_tie_and_duplicate_behavior() {
        let result = neardate(
            vec![1, 1, 1],
            vec![15.0, 10.0, 11.0],
            vec![1, 1, 1, 1],
            vec![10.0, 20.0, 10.0, 12.0],
            Some("closest"),
            None,
        )
        .unwrap();

        assert_eq!(result.indices, vec![Some(3), Some(0), Some(0)]);
        assert_eq!(result.distances, vec![Some(3.0), Some(0.0), Some(1.0)]);

        let prior = neardate(
            vec![1],
            vec![10.0],
            vec![1, 1],
            vec![10.0, 10.0],
            Some("prior"),
            None,
        )
        .unwrap();
        assert_eq!(prior.indices[0], Some(1));

        let after = neardate(
            vec![1],
            vec![10.0],
            vec![1, 1],
            vec![10.0, 10.0],
            Some("after"),
            None,
        )
        .unwrap();
        assert_eq!(after.indices[0], Some(0));
    }

    #[test]
    fn test_neardate_rejects_nonfinite_dates() {
        assert!(neardate(vec![1], vec![f64::NAN], vec![1], vec![1.0], None, None).is_err());
        assert!(
            neardate_str(
                vec!["a".to_string()],
                vec![1.0],
                vec!["a".to_string()],
                vec![f64::INFINITY],
                None,
                None,
            )
            .is_err()
        );
        assert!(neardate(vec![1], vec![1.0], vec![1], vec![1.0], Some(""), None).is_err());
    }
}
