use pyo3::prelude::*;
use std::collections::HashMap;

/// Result of nearest date matching
#[derive(Debug, Clone)]
#[pyclass]
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

    let direction = best.unwrap_or("closest");
    if !["prior", "after", "closest"].contains(&direction) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "best must be 'prior', 'after', or 'closest'",
        ));
    }

    let mut ref_by_id: HashMap<i64, Vec<(usize, f64)>> = HashMap::new();
    for (idx, (&id, &date)) in id2.iter().zip(date2.iter()).enumerate() {
        ref_by_id.entry(id).or_default().push((idx, date));
    }

    for entries in ref_by_id.values_mut() {
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
            let mut best: Option<(usize, f64)> = None;
            for &(idx, val) in refs {
                if val <= target {
                    best = Some((idx, target - val));
                } else {
                    break;
                }
            }
            best
        }
        "after" => {
            for &(idx, val) in refs {
                if val >= target {
                    return Some((idx, val - target));
                }
            }
            None
        }
        "closest" => {
            let mut best_idx = 0;
            let mut best_dist = (refs[0].1 - target).abs();

            for (i, &(_, val)) in refs.iter().enumerate() {
                let dist = (val - target).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            Some((refs[best_idx].0, best_dist))
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

    let direction = best.unwrap_or("closest");
    if !["prior", "after", "closest"].contains(&direction) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "best must be 'prior', 'after', or 'closest'",
        ));
    }

    // Build index of reference observations by ID
    let mut ref_by_id: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    for (idx, (id, &date)) in id2.iter().zip(date2.iter()).enumerate() {
        ref_by_id.entry(id.clone()).or_default().push((idx, date));
    }

    for entries in ref_by_id.values_mut() {
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
        // For id=1, date=5: closest is date2[1]=10 (idx 1)
        // For id=1, date=15: closest is date2[1]=10 or date2[2]=20
        // For id=2, date=10: closest is date2[3]=5 or date2[4]=15
    }

    #[test]
    fn test_neardate_prior() {
        let id1 = vec![1];
        let date1 = vec![15.0];
        let id2 = vec![1, 1, 1];
        let date2 = vec![10.0, 20.0, 5.0];

        let result = neardate(id1, date1, id2, date2, Some("prior"), None).unwrap();
        assert_eq!(result.n_matched, 1);
        // Should match date2[0]=10 (the largest value <= 15)
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
        // Should match date2[1]=20 (the smallest value >= 15)
        assert_eq!(result.indices[0], Some(1));
    }

    #[test]
    fn test_neardate_no_match() {
        let id1 = vec![1];
        let date1 = vec![10.0];
        let id2 = vec![2]; // Different ID
        let date2 = vec![10.0];

        let result = neardate(id1, date1, id2, date2, None, None).unwrap();
        assert_eq!(result.n_matched, 0);
        assert_eq!(result.indices[0], None);
    }
}
