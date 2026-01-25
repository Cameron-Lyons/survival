#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct ICEResult {
    #[pyo3(get)]
    pub grid_values: Vec<f64>,
    #[pyo3(get)]
    pub ice_curves: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub pdp_curve: Vec<f64>,
    #[pyo3(get)]
    pub feature_index: usize,
    #[pyo3(get)]
    pub centered: bool,
}

#[pymethods]
impl ICEResult {
    fn __repr__(&self) -> String {
        format!(
            "ICEResult(feature={}, n_curves={}, centered={})",
            self.feature_index,
            self.ice_curves.len(),
            self.centered
        )
    }

    fn get_curve(&self, index: usize) -> PyResult<Vec<f64>> {
        self.ice_curves.get(index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("curve index out of bounds")
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DICEResult {
    #[pyo3(get)]
    pub grid_values: Vec<f64>,
    #[pyo3(get)]
    pub derivative_curves: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub mean_derivative: Vec<f64>,
    #[pyo3(get)]
    pub feature_index: usize,
}

#[pymethods]
impl DICEResult {
    fn __repr__(&self) -> String {
        format!(
            "DICEResult(feature={}, n_curves={})",
            self.feature_index,
            self.derivative_curves.len()
        )
    }
}

fn compute_grid(values: &[f64], n_points: usize) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_val = sorted[0];
    let max_val = sorted[sorted.len() - 1];

    (0..n_points)
        .map(|i| min_val + (max_val - min_val) * i as f64 / (n_points - 1).max(1) as f64)
        .collect()
}

fn linear_prediction(cov: &[f64], weights: &[f64]) -> f64 {
    cov.iter()
        .zip(weights.iter())
        .map(|(&c, &w)| c * w)
        .sum::<f64>()
        .tanh()
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature_index,
    n_grid=50,
    centered=false,
    sample_size=None
))]
pub fn compute_ice(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature_index: usize,
    n_grid: usize,
    centered: bool,
    sample_size: Option<usize>,
) -> PyResult<ICEResult> {
    if covariates.is_empty() || predictions.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inputs must not be empty",
        ));
    }

    if covariates.len() != predictions.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates and predictions must have same length",
        ));
    }

    let n_features = covariates[0].len();
    if feature_index >= n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_index out of bounds",
        ));
    }

    let sample_indices: Vec<usize> = if let Some(size) = sample_size {
        if size >= covariates.len() {
            (0..covariates.len()).collect()
        } else {
            let step = covariates.len() / size;
            (0..size).map(|i| i * step).collect()
        }
    } else {
        (0..covariates.len()).collect()
    };

    let feature_values: Vec<f64> = covariates.iter().map(|c| c[feature_index]).collect();
    let grid_values = compute_grid(&feature_values, n_grid);

    let weights: Vec<f64> = (0..n_features)
        .map(|f| {
            let sum: f64 = covariates.iter().map(|c| c[f]).sum();
            let mean = sum / covariates.len() as f64;
            let pred_mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
            let cov: f64 = covariates
                .iter()
                .zip(predictions.iter())
                .map(|(c, &p)| (c[f] - mean) * (p - pred_mean))
                .sum();
            let var: f64 = covariates.iter().map(|c| (c[f] - mean).powi(2)).sum();
            if var > 1e-10 { cov / var } else { 0.0 }
        })
        .collect();

    let mut ice_curves: Vec<Vec<f64>> = sample_indices
        .par_iter()
        .map(|&idx| {
            let original_cov = &covariates[idx];
            grid_values
                .iter()
                .map(|&grid_val| {
                    let mut modified = original_cov.clone();
                    modified[feature_index] = grid_val;
                    linear_prediction(&modified, &weights)
                })
                .collect()
        })
        .collect();

    if centered {
        for curve in &mut ice_curves {
            let first = curve[0];
            for v in curve.iter_mut() {
                *v -= first;
            }
        }
    }

    let n_curves = ice_curves.len();
    let pdp_curve: Vec<f64> = (0..n_grid)
        .map(|j| ice_curves.iter().map(|c| c[j]).sum::<f64>() / n_curves as f64)
        .collect();

    Ok(ICEResult {
        grid_values,
        ice_curves,
        pdp_curve,
        feature_index,
        centered,
    })
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    predictions,
    feature_index,
    n_grid=50,
    sample_size=None
))]
pub fn compute_dice(
    covariates: Vec<Vec<f64>>,
    predictions: Vec<f64>,
    feature_index: usize,
    n_grid: usize,
    sample_size: Option<usize>,
) -> PyResult<DICEResult> {
    let ice = compute_ice(
        covariates,
        predictions,
        feature_index,
        n_grid,
        false,
        sample_size,
    )?;

    let derivative_curves: Vec<Vec<f64>> = ice
        .ice_curves
        .par_iter()
        .map(|curve| {
            let mut deriv = vec![0.0; curve.len()];
            for i in 0..curve.len() {
                if i == 0 {
                    deriv[i] = (curve[1] - curve[0])
                        / (ice.grid_values[1] - ice.grid_values[0]).max(1e-10);
                } else if i == curve.len() - 1 {
                    deriv[i] = (curve[i] - curve[i - 1])
                        / (ice.grid_values[i] - ice.grid_values[i - 1]).max(1e-10);
                } else {
                    deriv[i] = (curve[i + 1] - curve[i - 1])
                        / (ice.grid_values[i + 1] - ice.grid_values[i - 1]).max(1e-10);
                }
            }
            deriv
        })
        .collect();

    let n_curves = derivative_curves.len();
    let mean_derivative: Vec<f64> = (0..n_grid)
        .map(|j| derivative_curves.iter().map(|c| c[j]).sum::<f64>() / n_curves as f64)
        .collect();

    Ok(DICEResult {
        grid_values: ice.grid_values,
        derivative_curves,
        mean_derivative,
        feature_index,
    })
}

#[pyfunction]
#[pyo3(signature = (
    covariates,
    survival_predictions,
    time_points,
    feature_index,
    n_grid=50,
    sample_size=None
))]
pub fn compute_survival_ice(
    covariates: Vec<Vec<f64>>,
    survival_predictions: Vec<Vec<f64>>,
    time_points: Vec<f64>,
    feature_index: usize,
    n_grid: usize,
    sample_size: Option<usize>,
) -> PyResult<Vec<ICEResult>> {
    if covariates.is_empty() || survival_predictions.is_empty() || time_points.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "inputs must not be empty",
        ));
    }

    let n_times = time_points.len();
    if survival_predictions[0].len() != n_times {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survival_predictions must have same number of time points",
        ));
    }

    let results: Vec<ICEResult> = (0..n_times)
        .into_par_iter()
        .map(|t| {
            let preds_at_t: Vec<f64> = survival_predictions.iter().map(|p| p[t]).collect();
            compute_ice(
                covariates.clone(),
                preds_at_t,
                feature_index,
                n_grid,
                false,
                sample_size,
            )
            .unwrap()
        })
        .collect();

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (
    ice_result,
    threshold=0.1
))]
pub fn detect_heterogeneity(ice_result: &ICEResult, threshold: f64) -> PyResult<Vec<usize>> {
    let n_curves = ice_result.ice_curves.len();
    let n_grid = ice_result.grid_values.len();

    let deviations: Vec<f64> = ice_result
        .ice_curves
        .par_iter()
        .map(|curve| {
            let mse: f64 = curve
                .iter()
                .zip(ice_result.pdp_curve.iter())
                .map(|(&c, &p)| (c - p).powi(2))
                .sum();
            (mse / n_grid as f64).sqrt()
        })
        .collect();

    let mean_dev: f64 = deviations.iter().sum::<f64>() / n_curves as f64;
    let std_dev: f64 = (deviations
        .iter()
        .map(|&d| (d - mean_dev).powi(2))
        .sum::<f64>()
        / n_curves as f64)
        .sqrt();

    let heterogeneous_indices: Vec<usize> = deviations
        .iter()
        .enumerate()
        .filter(|(_, d)| **d > mean_dev + threshold * std_dev)
        .map(|(i, _)| i)
        .collect();

    Ok(heterogeneous_indices)
}

#[pyfunction]
#[pyo3(signature = (
    ice_result,
    n_clusters=3
))]
pub fn cluster_ice_curves(ice_result: &ICEResult, n_clusters: usize) -> PyResult<Vec<usize>> {
    let n_curves = ice_result.ice_curves.len();

    if n_clusters >= n_curves {
        return Ok((0..n_curves).collect());
    }

    let mut assignments = vec![0usize; n_curves];
    for (i, curve) in ice_result.ice_curves.iter().enumerate() {
        let sum: f64 = curve.iter().sum();
        let bucket = ((sum * n_clusters as f64).abs() as usize) % n_clusters;
        assignments[i] = bucket;
    }

    Ok(assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_computation() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let grid = compute_grid(&values, 5);
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 0.0).abs() < 1e-6);
        assert!((grid[4] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_ice_computation() {
        let covariates = vec![
            vec![0.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
        ];
        let predictions = vec![0.0, 0.5, 0.7, 1.0];
        let result = compute_ice(covariates, predictions, 0, 5, false, None).unwrap();
        assert_eq!(result.ice_curves.len(), 4);
        assert_eq!(result.grid_values.len(), 5);
    }

    #[test]
    fn test_centered_ice() {
        let covariates = vec![vec![0.0], vec![1.0], vec![2.0]];
        let predictions = vec![0.0, 0.5, 1.0];
        let result = compute_ice(covariates, predictions, 0, 3, true, None).unwrap();
        for curve in &result.ice_curves {
            assert!((curve[0]).abs() < 1e-6);
        }
    }
}
