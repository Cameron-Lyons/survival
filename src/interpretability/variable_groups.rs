#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

type DendrogramHistory = Vec<(usize, usize, f64, usize)>;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum GroupingMethod {
    Hierarchical,
    KMeans,
    Domain,
    Automatic,
}

#[pymethods]
impl GroupingMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "hierarchical" => Ok(GroupingMethod::Hierarchical),
            "kmeans" | "k-means" => Ok(GroupingMethod::KMeans),
            "domain" => Ok(GroupingMethod::Domain),
            "automatic" | "auto" => Ok(GroupingMethod::Automatic),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown grouping method",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum LinkageType {
    Single,
    Complete,
    Average,
    Ward,
}

#[pymethods]
impl LinkageType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "single" => Ok(LinkageType::Single),
            "complete" => Ok(LinkageType::Complete),
            "average" => Ok(LinkageType::Average),
            "ward" => Ok(LinkageType::Ward),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown linkage type",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct VariableGroupingConfig {
    #[pyo3(get, set)]
    pub method: GroupingMethod,
    #[pyo3(get, set)]
    pub n_groups: Option<usize>,
    #[pyo3(get, set)]
    pub correlation_threshold: f64,
    #[pyo3(get, set)]
    pub linkage: LinkageType,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl VariableGroupingConfig {
    #[new]
    #[pyo3(signature = (
        method=GroupingMethod::Automatic,
        n_groups=None,
        correlation_threshold=0.7,
        linkage=LinkageType::Average,
        max_iter=100,
        seed=None
    ))]
    pub fn new(
        method: GroupingMethod,
        n_groups: Option<usize>,
        correlation_threshold: f64,
        linkage: LinkageType,
        max_iter: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&correlation_threshold) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "correlation_threshold must be between 0 and 1",
            ));
        }

        Ok(VariableGroupingConfig {
            method,
            n_groups,
            correlation_threshold,
            linkage,
            max_iter,
            seed,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FeatureGroup {
    #[pyo3(get)]
    pub group_id: usize,
    #[pyo3(get)]
    pub feature_indices: Vec<usize>,
    #[pyo3(get)]
    pub representative_feature: usize,
    #[pyo3(get)]
    pub group_importance: f64,
    #[pyo3(get)]
    pub internal_correlation: f64,
}

#[pymethods]
impl FeatureGroup {
    fn __repr__(&self) -> String {
        format!(
            "FeatureGroup(id={}, n_features={}, importance={:.4})",
            self.group_id,
            self.feature_indices.len(),
            self.group_importance
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct VariableGroupingResult {
    #[pyo3(get)]
    pub groups: Vec<FeatureGroup>,
    #[pyo3(get)]
    pub feature_to_group: Vec<usize>,
    #[pyo3(get)]
    pub correlation_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dendrogram: Option<Vec<(usize, usize, f64, usize)>>,
    #[pyo3(get)]
    pub n_groups: usize,
    #[pyo3(get)]
    pub n_features: usize,
}

#[pymethods]
impl VariableGroupingResult {
    fn __repr__(&self) -> String {
        format!(
            "VariableGroupingResult(n_groups={}, n_features={})",
            self.n_groups, self.n_features
        )
    }

    fn get_group(&self, group_id: usize) -> Option<FeatureGroup> {
        self.groups.iter().find(|g| g.group_id == group_id).cloned()
    }

    fn get_feature_group(&self, feature_idx: usize) -> Option<usize> {
        self.feature_to_group.get(feature_idx).copied()
    }

    fn get_group_by_feature(&self, feature_idx: usize) -> Option<FeatureGroup> {
        let group_id = self.feature_to_group.get(feature_idx)?;
        self.get_group(*group_id)
    }
}

fn compute_shap_correlation_matrix(
    shap_values: &[Vec<Vec<f64>>],
    n_samples: usize,
    n_features: usize,
    n_times: usize,
) -> Vec<Vec<f64>> {
    let aggregated: Vec<Vec<f64>> = (0..n_features)
        .map(|f| {
            (0..n_samples)
                .map(|s| shap_values[s][f].iter().map(|v| v.abs()).sum::<f64>() / n_times as f64)
                .collect()
        })
        .collect();

    let means: Vec<f64> = aggregated
        .iter()
        .map(|f| f.iter().sum::<f64>() / n_samples as f64)
        .collect();

    let stds: Vec<f64> = aggregated
        .iter()
        .zip(means.iter())
        .map(|(f, &m)| {
            let var = f.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n_samples as f64;
            var.sqrt().max(1e-12)
        })
        .collect();

    let mut corr = vec![vec![0.0; n_features]; n_features];

    for i in 0..n_features {
        corr[i][i] = 1.0;

        for j in (i + 1)..n_features {
            let mut cov = 0.0;
            for s in 0..n_samples {
                cov += (aggregated[i][s] - means[i]) * (aggregated[j][s] - means[j]);
            }
            cov /= n_samples as f64;

            let r = cov / (stds[i] * stds[j]);
            let r = r.clamp(-1.0, 1.0);

            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    corr
}

fn hierarchical_clustering(
    dist_matrix: &[Vec<f64>],
    n_features: usize,
    linkage: LinkageType,
    n_groups: Option<usize>,
    threshold: f64,
) -> (Vec<usize>, DendrogramHistory) {
    let mut clusters: Vec<Vec<usize>> = (0..n_features).map(|i| vec![i]).collect();
    let mut dendrogram = Vec::new();

    let dist = dist_matrix.to_vec();

    let target_n = n_groups.unwrap_or(1);

    while clusters.len() > target_n {
        let mut min_dist = f64::INFINITY;
        let mut merge_i = 0;
        let mut merge_j = 0;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d = compute_cluster_distance(&clusters[i], &clusters[j], &dist, linkage);

                if d < min_dist {
                    min_dist = d;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        if n_groups.is_none() && min_dist > threshold {
            break;
        }

        let cluster_j = clusters.remove(merge_j);
        let merged_size = clusters[merge_i].len() + cluster_j.len();

        dendrogram.push((merge_i, merge_j, min_dist, merged_size));

        clusters[merge_i].extend(cluster_j);
    }

    let mut feature_to_group = vec![0; n_features];
    for (group_id, cluster) in clusters.iter().enumerate() {
        for &feature in cluster {
            feature_to_group[feature] = group_id;
        }
    }

    (feature_to_group, dendrogram)
}

fn compute_cluster_distance(
    cluster_a: &[usize],
    cluster_b: &[usize],
    dist_matrix: &[Vec<f64>],
    linkage: LinkageType,
) -> f64 {
    let distances: Vec<f64> = cluster_a
        .iter()
        .flat_map(|&i| cluster_b.iter().map(move |&j| dist_matrix[i][j]))
        .collect();

    match linkage {
        LinkageType::Single => distances.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
        LinkageType::Complete => distances.iter().fold(0.0f64, |a, &b| a.max(b)),
        LinkageType::Average => distances.iter().sum::<f64>() / distances.len() as f64,
        LinkageType::Ward => {
            let n_a = cluster_a.len() as f64;
            let n_b = cluster_b.len() as f64;
            let avg = distances.iter().sum::<f64>() / distances.len() as f64;
            (n_a * n_b / (n_a + n_b)) * avg
        }
    }
}

fn kmeans_clustering(
    features: &[Vec<f64>],
    n_features: usize,
    n_groups: usize,
    max_iter: usize,
    seed: u64,
) -> Vec<usize> {
    if n_features <= n_groups {
        return (0..n_features).collect();
    }

    let dim = features[0].len();

    let mut rng = fastrand::Rng::with_seed(seed);
    let mut centroids: Vec<Vec<f64>> = (0..n_groups)
        .map(|_| {
            let idx = rng.usize(0..n_features);
            features[idx].clone()
        })
        .collect();

    let mut assignments = vec![0; n_features];

    for _ in 0..max_iter {
        let old_assignments = assignments.clone();

        for (i, feature) in features.iter().enumerate() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (k, centroid) in centroids.iter().enumerate() {
                let dist: f64 = feature
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&f, &c)| (f - c).powi(2))
                    .sum();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = k;
                }
            }

            assignments[i] = best_cluster;
        }

        for (k, centroid) in centroids.iter_mut().enumerate() {
            let members: Vec<&Vec<f64>> = features
                .iter()
                .zip(assignments.iter())
                .filter(|(_, a)| **a == k)
                .map(|(f, _)| f)
                .collect();

            if !members.is_empty() {
                for d in 0..dim {
                    centroid[d] = members.iter().map(|m| m[d]).sum::<f64>() / members.len() as f64;
                }
            }
        }

        if assignments == old_assignments {
            break;
        }
    }

    assignments
}

fn compute_group_importance(
    shap_values: &[Vec<Vec<f64>>],
    group_features: &[usize],
    n_samples: usize,
    n_times: usize,
) -> f64 {
    let total: f64 = group_features
        .iter()
        .map(|&f| {
            shap_values
                .iter()
                .flat_map(|s| s[f].iter())
                .map(|v| v.abs())
                .sum::<f64>()
        })
        .sum();

    total / (n_samples * n_times) as f64
}

fn compute_internal_correlation(corr_matrix: &[Vec<f64>], features: &[usize]) -> f64 {
    if features.len() < 2 {
        return 1.0;
    }

    let mut total = 0.0;
    let mut count = 0;

    for i in 0..features.len() {
        for j in (i + 1)..features.len() {
            total += corr_matrix[features[i]][features[j]].abs();
            count += 1;
        }
    }

    if count > 0 { total / count as f64 } else { 1.0 }
}

fn find_representative_feature(
    shap_values: &[Vec<Vec<f64>>],
    features: &[usize],
    n_samples: usize,
    n_times: usize,
) -> usize {
    if features.is_empty() {
        return 0;
    }

    let importances: Vec<(usize, f64)> = features
        .iter()
        .map(|&f| {
            let imp: f64 = shap_values
                .iter()
                .flat_map(|s| s[f].iter())
                .map(|v| v.abs())
                .sum();
            (f, imp)
        })
        .collect();

    importances
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(f, _)| f)
        .unwrap_or(features[0])
}

#[pyfunction]
#[pyo3(signature = (shap_values, n_samples, n_features, n_times, config))]
pub fn group_variables(
    shap_values: Vec<Vec<Vec<f64>>>,
    n_samples: usize,
    n_features: usize,
    n_times: usize,
    config: &VariableGroupingConfig,
) -> PyResult<VariableGroupingResult> {
    if shap_values.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shap_values first dimension must match n_samples",
        ));
    }

    let corr_matrix = compute_shap_correlation_matrix(&shap_values, n_samples, n_features, n_times);

    let dist_matrix: Vec<Vec<f64>> = corr_matrix
        .iter()
        .map(|row| row.iter().map(|&c| 1.0 - c.abs()).collect())
        .collect();

    let (feature_to_group, dendrogram) = match config.method {
        GroupingMethod::Hierarchical | GroupingMethod::Automatic => {
            let (ftg, dend) = hierarchical_clustering(
                &dist_matrix,
                n_features,
                config.linkage,
                config.n_groups,
                1.0 - config.correlation_threshold,
            );
            (ftg, Some(dend))
        }
        GroupingMethod::KMeans => {
            let features: Vec<Vec<f64>> = (0..n_features)
                .map(|f| {
                    (0..n_samples)
                        .map(|s| {
                            shap_values[s][f].iter().map(|v| v.abs()).sum::<f64>() / n_times as f64
                        })
                        .collect()
                })
                .collect();

            let n_groups = config
                .n_groups
                .unwrap_or((n_features as f64).sqrt() as usize)
                .max(1);
            let ftg = kmeans_clustering(
                &features,
                n_features,
                n_groups,
                config.max_iter,
                config.seed.unwrap_or(42),
            );
            (ftg, None)
        }
        GroupingMethod::Domain => {
            let ftg = (0..n_features).collect();
            (ftg, None)
        }
    };

    let n_groups = *feature_to_group.iter().max().unwrap_or(&0) + 1;

    let groups: Vec<FeatureGroup> = (0..n_groups)
        .map(|g| {
            let indices: Vec<usize> = feature_to_group
                .iter()
                .enumerate()
                .filter(|(_, group)| **group == g)
                .map(|(i, _)| i)
                .collect();

            let representative =
                find_representative_feature(&shap_values, &indices, n_samples, n_times);
            let importance = compute_group_importance(&shap_values, &indices, n_samples, n_times);
            let internal_corr = compute_internal_correlation(&corr_matrix, &indices);

            FeatureGroup {
                group_id: g,
                feature_indices: indices,
                representative_feature: representative,
                group_importance: importance,
                internal_correlation: internal_corr,
            }
        })
        .collect();

    Ok(VariableGroupingResult {
        groups,
        feature_to_group,
        correlation_matrix: corr_matrix,
        dendrogram,
        n_groups,
        n_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = VariableGroupingConfig::new(
            GroupingMethod::Hierarchical,
            Some(3),
            0.7,
            LinkageType::Average,
            100,
            None,
        )
        .unwrap();
        assert_eq!(config.n_groups, Some(3));
    }

    #[test]
    fn test_correlation_matrix() {
        let shap = vec![
            vec![vec![1.0, 2.0], vec![2.0, 4.0]],
            vec![vec![1.5, 2.5], vec![3.0, 5.0]],
        ];

        let corr = compute_shap_correlation_matrix(&shap, 2, 2, 2);

        assert_eq!(corr.len(), 2);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let dist = vec![
            vec![0.0, 0.1, 0.5, 0.6],
            vec![0.1, 0.0, 0.4, 0.5],
            vec![0.5, 0.4, 0.0, 0.1],
            vec![0.6, 0.5, 0.1, 0.0],
        ];

        let (assignments, _) =
            hierarchical_clustering(&dist, 4, LinkageType::Average, Some(2), 0.5);

        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_kmeans() {
        let features = vec![
            vec![1.0, 1.0],
            vec![1.1, 0.9],
            vec![5.0, 5.0],
            vec![4.9, 5.1],
        ];

        let assignments = kmeans_clustering(&features, 4, 2, 50, 42);

        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_group_variables() {
        let shap = vec![
            vec![vec![1.0, 2.0, 3.0]; 4],
            vec![vec![1.5, 2.5, 3.5]; 4],
            vec![vec![0.5, 1.5, 2.5]; 4],
        ];

        let config = VariableGroupingConfig::new(
            GroupingMethod::Automatic,
            None,
            0.7,
            LinkageType::Average,
            100,
            Some(42),
        )
        .unwrap();

        let result = group_variables(shap, 3, 4, 3, &config).unwrap();

        assert!(result.n_groups >= 1);
        assert_eq!(result.n_features, 4);
        assert_eq!(result.feature_to_group.len(), 4);
    }
}
