use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ClusterResult {
    #[pyo3(get)]
    pub cluster_ids: Vec<i32>,
    #[pyo3(get)]
    pub n_clusters: usize,
    #[pyo3(get)]
    pub cluster_sizes: Vec<usize>,
    #[pyo3(get)]
    pub levels: Vec<String>,
}

#[pyfunction]
pub fn cluster(id: Vec<i64>) -> PyResult<ClusterResult> {
    let n = id.len();
    if n == 0 {
        return Ok(ClusterResult {
            cluster_ids: vec![],
            n_clusters: 0,
            cluster_sizes: vec![],
            levels: vec![],
        });
    }

    let mut cluster_map: HashMap<i64, i32> = HashMap::new();
    let mut cluster_ids = Vec::with_capacity(n);
    let mut levels = Vec::new();
    let mut current_cluster_id = 0i32;

    for &val in &id {
        let cluster_id = *cluster_map.entry(val).or_insert_with(|| {
            let id = current_cluster_id;
            levels.push(val.to_string());
            current_cluster_id += 1;
            id
        });
        cluster_ids.push(cluster_id);
    }

    let n_clusters = cluster_map.len();

    let mut cluster_sizes = vec![0usize; n_clusters];
    for &cid in &cluster_ids {
        cluster_sizes[cid as usize] += 1;
    }

    Ok(ClusterResult {
        cluster_ids,
        n_clusters,
        cluster_sizes,
        levels,
    })
}

#[pyfunction]
pub fn cluster_str(id: Vec<String>) -> PyResult<ClusterResult> {
    let n = id.len();
    if n == 0 {
        return Ok(ClusterResult {
            cluster_ids: vec![],
            n_clusters: 0,
            cluster_sizes: vec![],
            levels: vec![],
        });
    }

    let mut cluster_map: HashMap<String, i32> = HashMap::new();
    let mut cluster_ids = Vec::with_capacity(n);
    let mut levels = Vec::new();
    let mut current_cluster_id = 0i32;

    for val in &id {
        let cluster_id = *cluster_map.entry(val.clone()).or_insert_with(|| {
            let id = current_cluster_id;
            levels.push(val.clone());
            current_cluster_id += 1;
            id
        });
        cluster_ids.push(cluster_id);
    }

    let n_clusters = cluster_map.len();

    let mut cluster_sizes = vec![0usize; n_clusters];
    for &cid in &cluster_ids {
        cluster_sizes[cid as usize] += 1;
    }

    Ok(ClusterResult {
        cluster_ids,
        n_clusters,
        cluster_sizes,
        levels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_basic() {
        let id = vec![1, 1, 2, 2, 2, 3];
        let result = cluster(id).unwrap();
        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.cluster_ids, vec![0, 0, 1, 1, 1, 2]);
        assert_eq!(result.cluster_sizes, vec![2, 3, 1]);
    }

    #[test]
    fn test_cluster_empty() {
        let id: Vec<i64> = vec![];
        let result = cluster(id).unwrap();
        assert_eq!(result.n_clusters, 0);
    }

    #[test]
    fn test_cluster_all_same() {
        let id = vec![5, 5, 5, 5];
        let result = cluster(id).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes, vec![4]);
    }

    #[test]
    fn test_cluster_str() {
        let id = vec!["A".to_string(), "B".to_string(), "A".to_string()];
        let result = cluster_str(id).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.cluster_ids, vec![0, 1, 0]);
    }
}
