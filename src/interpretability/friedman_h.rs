#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct FriedmanHResult {
    #[pyo3(get)]
    pub feature1_index: usize,
    #[pyo3(get)]
    pub feature2_index: usize,
    #[pyo3(get)]
    pub h_statistic: f64,
    #[pyo3(get)]
    pub interaction_strength: f64,
}

#[pymethods]
impl FriedmanHResult {
    fn __repr__(&self) -> String {
        format!(
            "FriedmanHResult(features=({}, {}), H={:.4})",
            self.feature1_index, self.feature2_index, self.h_statistic
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FeatureImportanceResult {
    #[pyo3(get)]
    pub feature_index: usize,
    #[pyo3(get)]
    pub main_effect: f64,
    #[pyo3(get)]
    pub total_effect: f64,
    #[pyo3(get)]
    pub interaction_effect: f64,
}

#[pymethods]
impl FeatureImportanceResult {
    fn __repr__(&self) -> String {
        format!(
            "FeatureImportanceResult(feature={}, main={:.4}, interaction={:.4})",
            self.feature_index, self.main_effect, self.interaction_effect
        )
    }
}

fn compute_pdp_values(
    covariates: &[Vec<f64>],
    predictions: &[f64],
    feature_index: usize,
    grid_points: &[f64],
) -> Vec<f64> {
    let mut feature_values: Vec<(f64, f64)> = covariates
        .iter()
        .zip(predictions.iter())
        .map(|(c, &p)| (c[feature_index], p))
        .collect();
    feature_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    grid_points
        .iter()
        .map(|&grid_val| {
            let mut total = 0.0;
            let mut count = 0;
            for &(fv, pred) in &feature_values {
                if (fv - grid_val).abs()
                    < (grid_points.last().unwrap_or(&1.0) - grid_points[0])
                        / grid_points.len() as f64
                {
                    total += pred;
                    count += 1;
                }
            }
            if count > 0 {
                total / count as f64
            } else {
                let idx = feature_values
                    .binary_search_by(|(fv, _)| {
                        fv.partial_cmp(&grid_val)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|i| i.min(feature_values.len() - 1));
                feature_values[idx].1
            }
        })
        .collect()
}

fn compute_pdp_2d_values(
    covariates: &[Vec<f64>],
    predictions: &[f64],
    feature1_index: usize,
    feature2_index: usize,
    grid1: &[f64],
    grid2: &[f64],
) -> Vec<Vec<f64>> {
    let n1 = grid1.len();
    let n2 = grid2.len();

    let data: Vec<(f64, f64, f64)> = covariates
        .iter()
        .zip(predictions.iter())
        .map(|(c, &p)| (c[feature1_index], c[feature2_index], p))
        .collect();

    (0..n1)
        .into_par_iter()
        .map(|i| {
            let g1 = grid1[i];
            (0..n2)
                .map(|j| {
                    let g2 = grid2[j];
                    let mut total = 0.0;
                    let mut count = 0;
                    let tol1 =
                        (grid1.last().unwrap_or(&1.0) - grid1[0]) / grid1.len().max(1) as f64;
                    let tol2 =
                        (grid2.last().unwrap_or(&1.0) - grid2[0]) / grid2.len().max(1) as f64;

                    for &(f1, f2, pred) in &data {
                        if (f1 - g1).abs() < tol1 && (f2 - g2).abs() < tol2 {
                            total += pred;
                            count += 1;
                        }
                    }

                    if count > 0 {
                        total / count as f64
                    } else {
                        data.iter()
                            .min_by(|a, b| {
                                let d_a = (a.0 - g1).powi(2) + (a.1 - g2).powi(2);
                                let d_b = (b.0 - g1).powi(2) + (b.1 - g2).powi(2);
                                d_a.partial_cmp(&d_b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|t| t.2)
                            .unwrap_or(0.0)
                    }
                })
                .collect()
        })
        .collect()
}

fn compute_grid_points(values: &[f64], n_points: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    (0..n_points)
        .map(|i| {
            let idx = (i * (n - 1)) / (n_points - 1).max(1);
            sorted[idx]
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature1_index,
    feature2_index,
    n_grid=20
))]
pub fn compute_friedman_h(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature1_index: usize,
    feature2_index: usize,
    n_grid: usize,
) -> PyResult<FriedmanHResult> {
    if covariates.is_empty() || predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inputs must not be empty",
        ));
    }

    let n_features = covariates[0].len();
    if feature1_index >= n_features || feature2_index >= n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature indices out of bounds",
        ));
    }

    if feature1_index == feature2_index {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature indices must be different",
        ));
    }

    let feature1_values: Vec<f64> = covariates.iter().map(|c| c[feature1_index]).collect();
    let feature2_values: Vec<f64> = covariates.iter().map(|c| c[feature2_index]).collect();

    let grid1 = compute_grid_points(&feature1_values, n_grid);
    let grid2 = compute_grid_points(&feature2_values, n_grid);

    let pdp1 = compute_pdp_values(&covariates, &predictions, feature1_index, &grid1);
    let pdp2 = compute_pdp_values(&covariates, &predictions, feature2_index, &grid2);
    let pdp12 = compute_pdp_2d_values(
        &covariates,
        &predictions,
        feature1_index,
        feature2_index,
        &grid1,
        &grid2,
    );

    let mean_pred: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;

    let pdp1_centered: Vec<f64> = pdp1.iter().map(|&v| v - mean_pred).collect();
    let pdp2_centered: Vec<f64> = pdp2.iter().map(|&v| v - mean_pred).collect();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &p1) in pdp1_centered.iter().enumerate() {
        for (j, &p2) in pdp2_centered.iter().enumerate() {
            let additive = p1 + p2 + mean_pred;
            let joint = pdp12[i][j];
            let interaction = joint - additive;

            numerator += interaction.powi(2);
            denominator += joint.powi(2);
        }
    }

    let h_statistic = if denominator > 1e-10 {
        (numerator / denominator).sqrt()
    } else {
        0.0
    };

    let interaction_strength = numerator.sqrt() / (n_grid * n_grid) as f64;

    Ok(FriedmanHResult {
        feature1_index,
        feature2_index,
        h_statistic,
        interaction_strength,
    })
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature_indices=None,
    n_grid=20
))]
pub fn compute_all_pairwise_interactions(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature_indices: Option<Vec<usize>>,
    n_grid: usize,
) -> PyResult<Vec<FriedmanHResult>> {
    if covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must not be empty",
        ));
    }

    let n_features = covariates[0].len();
    let indices: Vec<usize> = feature_indices.unwrap_or_else(|| (0..n_features).collect());

    let pairs: Vec<(usize, usize)> = indices
        .iter()
        .flat_map(|&i| {
            indices
                .iter()
                .filter_map(move |&j| (i < j).then_some((i, j)))
        })
        .collect();

    let results: Vec<FriedmanHResult> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            compute_friedman_h(covariates.clone(), predictions.clone(), i, j, n_grid).ok()
        })
        .collect();

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    n_grid=20
))]
pub fn compute_feature_importance_decomposition(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    n_grid: usize,
) -> PyResult<Vec<FeatureImportanceResult>> {
    if covariates.is_empty() || predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inputs must not be empty",
        ));
    }

    let n_features = covariates[0].len();
    let mean_pred: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;

    let main_effects: Vec<f64> = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let feature_values: Vec<f64> = covariates.iter().map(|c| c[f]).collect();
            let grid = compute_grid_points(&feature_values, n_grid);
            let pdp = compute_pdp_values(&covariates, &predictions, f, &grid);

            let variance: f64 =
                pdp.iter().map(|&v| (v - mean_pred).powi(2)).sum::<f64>() / pdp.len() as f64;
            variance.sqrt()
        })
        .collect();

    let total_variance: f64 = predictions
        .iter()
        .map(|&p| (p - mean_pred).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    let total_effects: Vec<f64> = main_effects
        .iter()
        .map(|&me| (me.powi(2) + total_variance * 0.1).sqrt())
        .collect();

    let results: Vec<FeatureImportanceResult> = (0..n_features)
        .map(|f| FeatureImportanceResult {
            feature_index: f,
            main_effect: main_effects[f],
            total_effect: total_effects[f],
            interaction_effect: total_effects[f] - main_effects[f],
        })
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_points() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let grid = compute_grid_points(&values, 3);
        assert_eq!(grid.len(), 3);
        assert!((grid[0] - 1.0).abs() < 1e-6);
        assert!((grid[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_friedman_h_same_feature_error() {
        let covariates = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let predictions = vec![1.0, 2.0];
        let result = compute_friedman_h(covariates, predictions, 0, 0, 5);
        assert!(result.is_err());
    }
}
