use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct NearDateResult {
    #[pyo3(get)]
    pub indices: Vec<Option<usize>>,
    #[pyo3(get)]
    pub distances: Vec<Option<f64>>,
    #[pyo3(get)]
    pub n_matched: usize,
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

fn neardate_impl<Id>(
    id1: &[Id],
    date1: &[f64],
    id2: &[Id],
    date2: &[f64],
    direction: &str,
    nomatch: Option<usize>,
) -> PyResult<NearDateResult>
where
    Id: Clone + Eq + Hash,
{
    if date1.len() != id1.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "id1 and date1 must have same length",
        ));
    }
    if date2.len() != id2.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "id2 and date2 must have same length",
        ));
    }

    let query_ids: HashSet<Id> = id1.iter().cloned().collect();
    let mut ref_by_id: HashMap<Id, Vec<(usize, f64)>> = HashMap::new();
    for (idx, (id, &date)) in id2.iter().zip(date2).enumerate() {
        if date.is_nan() || !query_ids.contains(id) {
            continue;
        }
        ref_by_id.entry(id.clone()).or_default().push((idx, date));
    }
    if ref_by_id.is_empty() {
        return Err(PyErr::new::<PyValueError, _>(
            "No valid entries in data set 2",
        ));
    }
    for entries in ref_by_id.values_mut() {
        entries.sort_by(|left, right| left.1.total_cmp(&right.1));
    }

    let mut indices = Vec::with_capacity(id1.len());
    let mut distances = Vec::with_capacity(id1.len());
    let mut n_matched = 0;
    for (id, &date) in id1.iter().zip(date1) {
        let matched = if date.is_nan() {
            None
        } else {
            ref_by_id
                .get(id)
                .and_then(|references| find_nearest(references, date, direction))
        };
        if let Some((idx, distance)) = matched {
            indices.push(Some(idx));
            distances.push(Some(distance));
            n_matched += 1;
        } else {
            indices.push(nomatch);
            distances.push(None);
        }
    }

    Ok(NearDateResult {
        indices,
        distances,
        n_matched,
    })
}

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
    let direction = validate_direction(best)?;
    neardate_impl(&id1, &date1, &id2, &date2, direction, nomatch)
}

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
                Some((idx, ordered_distance(val, target)))
            }
        }
        "after" => {
            let pos = refs.partition_point(|entry| entry.1 < target);
            refs.get(pos)
                .map(|&(idx, val)| (idx, ordered_distance(target, val)))
        }
        "closest" => {
            let pos = refs.partition_point(|entry| entry.1 < target);
            if pos == 0 {
                let (idx, val) = refs[0];
                return Some((idx, ordered_distance(target, val)));
            }
            if pos == refs.len() {
                let (idx, val) = refs[refs.len() - 1];
                return Some((idx, ordered_distance(val, target)));
            }

            let (_, before_val) = refs[pos - 1];
            let (after_idx, after_val) = refs[pos];
            let before_dist = ordered_distance(before_val, target);
            let after_dist = ordered_distance(target, after_val);
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

fn ordered_distance(lower: f64, upper: f64) -> f64 {
    if lower == upper { 0.0 } else { upper - lower }
}

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
    let direction = validate_direction(best)?;
    neardate_impl(&id1, &date1, &id2, &date2, direction, nomatch)
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
        let id1 = vec![1, 2];
        let date1 = vec![10.0, 10.0];
        let id2 = vec![2];
        let date2 = vec![10.0];

        let result = neardate(id1, date1, id2, date2, None, None).unwrap();
        assert_eq!(result.n_matched, 1);
        assert_eq!(result.indices[0], None);
        assert_eq!(result.indices[1], Some(0));
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
    fn test_neardate_skips_missing_dates_and_accepts_infinite_endpoints() {
        let after = neardate(
            vec![1, 1, 1, 1],
            vec![f64::NAN, 2.0, f64::INFINITY, f64::NEG_INFINITY],
            vec![1, 1, 1, 1],
            vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            Some("after"),
            None,
        )
        .unwrap();
        let prior = neardate(
            vec![1, 1, 1, 1],
            vec![f64::NAN, 2.0, f64::INFINITY, f64::NEG_INFINITY],
            vec![1, 1, 1, 1],
            vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            Some("prior"),
            None,
        )
        .unwrap();

        assert_eq!(after.indices, vec![None, Some(2), Some(2), Some(3)]);
        assert_eq!(prior.indices, vec![None, Some(0), Some(2), Some(3)]);
        assert_eq!(
            after.distances,
            vec![None, Some(f64::INFINITY), Some(0.0), Some(0.0)]
        );
        assert_eq!(prior.distances, vec![None, Some(1.0), Some(0.0), Some(0.0)]);
        assert_eq!(after.n_matched, 3);
        assert_eq!(prior.n_matched, 3);
    }

    #[test]
    fn test_neardate_requires_a_valid_reference_for_a_query_id() {
        let missing =
            neardate(vec![1], vec![1.0], vec![1], vec![f64::NAN], None, None).unwrap_err();
        let unmatched = neardate_str(
            vec!["a".to_string()],
            vec![1.0],
            vec!["b".to_string()],
            vec![1.0],
            None,
            None,
        )
        .unwrap_err();

        assert!(
            missing
                .to_string()
                .contains("No valid entries in data set 2")
        );
        assert!(
            unmatched
                .to_string()
                .contains("No valid entries in data set 2")
        );
        assert!(neardate(vec![1], vec![1.0], vec![1], vec![1.0], Some(""), None).is_err());
    }
}
