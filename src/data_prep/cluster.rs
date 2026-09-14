//! R's `cluster(x)` (`R/cluster.R`) is an identity marker in a formula;
//! what the fitting code then needs is `match(x, unique(x))`: the cluster
//! of every observation as a code in order of first appearance.

use super::id_value::{IdValue, SubjectId, first_appearance_codes};
use pyo3::prelude::*;

/// Cluster membership codes.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct ClusterResult {
    /// Zero-based cluster code of each observation, numbered by first
    /// appearance.
    #[pyo3(get)]
    pub codes: Vec<usize>,
    /// Label of each cluster, in code order.
    #[pyo3(get)]
    pub levels: Vec<String>,
    /// Number of observations in each cluster.
    #[pyo3(get)]
    pub sizes: Vec<usize>,
}

/// Code a cluster identifier.
pub fn cluster<I: SubjectId>(id: &[I]) -> ClusterResult {
    let (codes, representatives) = first_appearance_codes(id);
    let mut sizes = vec![0; representatives.len()];
    for &code in &codes {
        sizes[code] += 1;
    }
    ClusterResult {
        codes,
        levels: representatives.iter().map(SubjectId::label).collect(),
        sizes,
    }
}

/// Python entry point of [`cluster`].
#[pyfunction(name = "cluster")]
pub fn cluster_py(id: Vec<IdValue>) -> PyResult<ClusterResult> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    Ok(cluster(&id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_follow_first_appearance() {
        let result = cluster(&[5i64, 5, 2, 2, 2, 3]);
        assert_eq!(result.codes, vec![0, 0, 1, 1, 1, 2]);
        assert_eq!(result.sizes, vec![2, 3, 1]);
        assert_eq!(result.levels, vec!["5", "2", "3"]);
        assert!(cluster::<i64>(&[]).codes.is_empty());
    }

    #[test]
    fn strings_and_mixed_python_values_are_supported() {
        let result = cluster(&["A", "B", "A"]);
        assert_eq!(result.codes, vec![0, 1, 0]);
        let result = cluster(&[
            IdValue::Int(1),
            IdValue::Float(1.0),
            IdValue::Str("1".into()),
        ]);
        assert_eq!(result.codes, vec![0, 0, 1]);
        assert_eq!(result.levels, vec!["1", "1"]);
    }
}
