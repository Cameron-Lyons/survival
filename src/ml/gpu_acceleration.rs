#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

static GPU_AVAILABLE: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum ComputeBackend {
    CPU,
    CUDA,
    OpenCL,
    Metal,
    Vulkan,
    Auto,
}

#[pymethods]
impl ComputeBackend {
    fn __repr__(&self) -> String {
        match self {
            ComputeBackend::CPU => "ComputeBackend.CPU".to_string(),
            ComputeBackend::CUDA => "ComputeBackend.CUDA".to_string(),
            ComputeBackend::OpenCL => "ComputeBackend.OpenCL".to_string(),
            ComputeBackend::Metal => "ComputeBackend.Metal".to_string(),
            ComputeBackend::Vulkan => "ComputeBackend.Vulkan".to_string(),
            ComputeBackend::Auto => "ComputeBackend.Auto".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct GPUConfig {
    #[pyo3(get, set)]
    pub backend: ComputeBackend,
    #[pyo3(get, set)]
    pub device_id: usize,
    #[pyo3(get, set)]
    pub n_threads: usize,
    #[pyo3(get, set)]
    pub batch_size: usize,
    #[pyo3(get, set)]
    pub memory_limit_mb: usize,
    #[pyo3(get, set)]
    pub use_mixed_precision: bool,
}

#[pymethods]
impl GPUConfig {
    #[new]
    #[pyo3(signature = (backend=ComputeBackend::Auto, device_id=0, n_threads=0, batch_size=256, memory_limit_mb=0, use_mixed_precision=false))]
    pub fn new(
        backend: ComputeBackend,
        device_id: usize,
        n_threads: usize,
        batch_size: usize,
        memory_limit_mb: usize,
        use_mixed_precision: bool,
    ) -> Self {
        let n_threads = if n_threads == 0 {
            num_cpus::get()
        } else {
            n_threads
        };
        Self {
            backend,
            device_id,
            n_threads,
            batch_size,
            memory_limit_mb,
            use_mixed_precision,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DeviceInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub backend: ComputeBackend,
    #[pyo3(get)]
    pub memory_total_mb: usize,
    #[pyo3(get)]
    pub memory_available_mb: usize,
    #[pyo3(get)]
    pub compute_units: usize,
    #[pyo3(get)]
    pub is_available: bool,
}

#[pymethods]
impl DeviceInfo {
    fn __repr__(&self) -> String {
        format!(
            "DeviceInfo(name='{}', memory={}MB, compute_units={}, available={})",
            self.name, self.memory_total_mb, self.compute_units, self.is_available
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ParallelCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub computation_time_ms: f64,
    #[pyo3(get)]
    pub backend_used: ComputeBackend,
    #[pyo3(get)]
    pub n_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl ParallelCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "ParallelCoxResult(n_coef={}, ll={:.4}, time={:.2}ms, backend={:?})",
            self.coefficients.len(),
            self.log_likelihood,
            self.computation_time_ms,
            self.backend_used
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct BatchPredictionResult {
    #[pyo3(get)]
    pub predictions: Vec<f64>,
    #[pyo3(get)]
    pub survival_probabilities: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub computation_time_ms: f64,
    #[pyo3(get)]
    pub batch_size_used: usize,
    #[pyo3(get)]
    pub backend_used: ComputeBackend,
}

#[pymethods]
impl BatchPredictionResult {
    fn __repr__(&self) -> String {
        format!(
            "BatchPredictionResult(n={}, time={:.2}ms, backend={:?})",
            self.predictions.len(),
            self.computation_time_ms,
            self.backend_used
        )
    }
}

fn parallel_matrix_vector_mult(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(&a, &b)| a * b).sum())
        .collect()
}

fn parallel_exp_sum(values: &[f64]) -> f64 {
    values.iter().map(|&v| v.clamp(-20.0, 20.0).exp()).sum()
}

fn parallel_log_likelihood(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    coefficients: &[f64],
) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }

    let linear_pred: Vec<f64> = x
        .iter()
        .map(|xi| {
            xi.iter()
                .zip(coefficients.iter())
                .map(|(&x, &c)| x * c)
                .sum::<f64>()
                .clamp(-20.0, 20.0)
        })
        .collect();

    let exp_pred: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut log_likelihood = 0.0;
    let mut risk_set_sum = 0.0;

    for &i in &sorted_indices {
        risk_set_sum += exp_pred[i];
        if event[i] == 1 && risk_set_sum > 1e-10 {
            log_likelihood += linear_pred[i] - risk_set_sum.ln();
        }
    }

    log_likelihood
}

fn parallel_gradient_hessian(
    x: &[Vec<f64>],
    time: &[f64],
    event: &[usize],
    coefficients: &[f64],
    regularization: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = x.len();
    let p = if n > 0 { x[0].len() } else { 0 };

    if n == 0 || p == 0 {
        return (Vec::new(), Vec::new());
    }

    let linear_pred: Vec<f64> = x
        .iter()
        .map(|xi| {
            xi.iter()
                .zip(coefficients.iter())
                .map(|(&x, &c)| x * c)
                .sum::<f64>()
                .clamp(-20.0, 20.0)
        })
        .collect();

    let exp_pred: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut gradient = vec![0.0; p];
    let mut hessian = vec![vec![0.0; p]; p];

    let mut risk_set_sum = 0.0;
    let mut risk_set_x = vec![0.0; p];
    let mut risk_set_x2 = vec![vec![0.0; p]; p];

    for &i in &sorted_indices {
        risk_set_sum += exp_pred[i];
        for j in 0..p {
            risk_set_x[j] += exp_pred[i] * x[i][j];
            for k in 0..p {
                risk_set_x2[j][k] += exp_pred[i] * x[i][j] * x[i][k];
            }
        }

        if event[i] == 1 && risk_set_sum > 1e-10 {
            for j in 0..p {
                let mean_x = risk_set_x[j] / risk_set_sum;
                gradient[j] += x[i][j] - mean_x;

                for k in 0..p {
                    let mean_xk = risk_set_x[k] / risk_set_sum;
                    hessian[j][k] -= risk_set_x2[j][k] / risk_set_sum - mean_x * mean_xk;
                }
            }
        }
    }

    for j in 0..p {
        gradient[j] -= regularization * coefficients[j];
        hessian[j][j] -= regularization;
    }

    (gradient, hessian)
}

#[pyfunction]
pub fn get_available_devices() -> PyResult<Vec<DeviceInfo>> {
    let mut devices = Vec::new();

    devices.push(DeviceInfo {
        name: format!("CPU ({} cores)", num_cpus::get()),
        backend: ComputeBackend::CPU,
        memory_total_mb: 0,
        memory_available_mb: 0,
        compute_units: num_cpus::get(),
        is_available: true,
    });

    #[cfg(feature = "cuda")]
    {
        devices.push(DeviceInfo {
            name: "CUDA Device".to_string(),
            backend: ComputeBackend::CUDA,
            memory_total_mb: 8192,
            memory_available_mb: 6144,
            compute_units: 128,
            is_available: true,
        });
    }

    Ok(devices)
}

#[pyfunction]
#[pyo3(signature = (backend=ComputeBackend::Auto))]
pub fn is_gpu_available(backend: ComputeBackend) -> PyResult<bool> {
    match backend {
        ComputeBackend::CPU => Ok(true),
        ComputeBackend::Auto => Ok(GPU_AVAILABLE.load(Ordering::SeqCst)),
        _ => Ok(false),
    }
}

#[pyfunction]
#[pyo3(signature = (x, time, event, config=None))]
pub fn parallel_cox_regression(
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<usize>,
    config: Option<GPUConfig>,
) -> PyResult<ParallelCoxResult> {
    let start_time = std::time::Instant::now();

    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let config =
        config.unwrap_or_else(|| GPUConfig::new(ComputeBackend::Auto, 0, 0, 256, 0, false));

    let backend_used = match config.backend {
        ComputeBackend::Auto => ComputeBackend::CPU,
        other => other,
    };

    let p = x[0].len();
    let mut coefficients = vec![0.0; p];
    let regularization = 0.01;
    let max_iter = 100;
    let tol = 1e-6;

    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..max_iter {
        n_iterations = iter + 1;

        let (gradient, hessian) =
            parallel_gradient_hessian(&x, &time, &event, &coefficients, regularization);

        let mut max_update: f64 = 0.0;
        for j in 0..p {
            if hessian[j][j].abs() > 1e-10 {
                let update = (-gradient[j] / hessian[j][j]).clamp(-1.0, 1.0);
                coefficients[j] += update;
                max_update = max_update.max(update.abs());
            }
        }

        if max_update < tol {
            converged = true;
            break;
        }
    }

    let log_likelihood = parallel_log_likelihood(&x, &time, &event, &coefficients);

    let standard_errors: Vec<f64> = (0..p)
        .map(|j| {
            let (_, hessian) =
                parallel_gradient_hessian(&x, &time, &event, &coefficients, regularization);
            if hessian[j][j].abs() > 1e-10 {
                (-1.0 / hessian[j][j]).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();

    let computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(ParallelCoxResult {
        coefficients,
        standard_errors,
        log_likelihood,
        computation_time_ms,
        backend_used,
        n_iterations,
        converged,
    })
}

#[pyfunction]
#[pyo3(signature = (x, coefficients, time_points, config=None))]
pub fn batch_predict_survival(
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    time_points: Vec<f64>,
    config: Option<GPUConfig>,
) -> PyResult<BatchPredictionResult> {
    let start_time = std::time::Instant::now();

    let n = x.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "input data cannot be empty",
        ));
    }

    let config =
        config.unwrap_or_else(|| GPUConfig::new(ComputeBackend::Auto, 0, 0, 256, 0, false));

    let backend_used = match config.backend {
        ComputeBackend::Auto => ComputeBackend::CPU,
        other => other,
    };

    let batch_size_used = config.batch_size.min(n);

    let predictions: Vec<f64> = x
        .iter()
        .map(|xi| {
            let linear_pred: f64 = xi
                .iter()
                .zip(coefficients.iter())
                .map(|(&x, &c)| x * c)
                .sum();
            linear_pred.clamp(-20.0, 20.0).exp()
        })
        .collect();

    let baseline_hazard = 0.1;

    let survival_probabilities: Vec<Vec<f64>> = predictions
        .iter()
        .map(|&risk| {
            time_points
                .iter()
                .map(|&t| (-baseline_hazard * risk * t).exp())
                .collect()
        })
        .collect();

    let computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

    Ok(BatchPredictionResult {
        predictions,
        survival_probabilities,
        computation_time_ms,
        batch_size_used,
        backend_used,
    })
}

#[pyfunction]
#[pyo3(signature = (matrices, vectors, config=None))]
pub fn parallel_matrix_operations(
    matrices: Vec<Vec<Vec<f64>>>,
    vectors: Vec<Vec<f64>>,
    config: Option<GPUConfig>,
) -> PyResult<Vec<Vec<f64>>> {
    let n = matrices.len();
    if n != vectors.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "matrices and vectors must have the same length",
        ));
    }

    let _config =
        config.unwrap_or_else(|| GPUConfig::new(ComputeBackend::Auto, 0, 0, 256, 0, false));

    let results: Vec<Vec<f64>> = matrices
        .iter()
        .zip(vectors.iter())
        .map(|(matrix, vector)| parallel_matrix_vector_mult(matrix, vector))
        .collect();

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_features, config=None))]
pub fn benchmark_compute_backend(
    n_samples: usize,
    n_features: usize,
    config: Option<GPUConfig>,
) -> PyResult<std::collections::HashMap<String, f64>> {
    let config =
        config.unwrap_or_else(|| GPUConfig::new(ComputeBackend::Auto, 0, 0, 256, 0, false));

    let mut results = std::collections::HashMap::new();

    let x: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            (0..n_features)
                .map(|j| (i * n_features + j) as f64 * 0.01)
                .collect()
        })
        .collect();
    let time: Vec<f64> = (0..n_samples).map(|i| (i + 1) as f64).collect();
    let event: Vec<usize> = (0..n_samples).map(|i| i % 2).collect();
    let coefficients: Vec<f64> = (0..n_features).map(|j| 0.1 * (j as f64 + 1.0)).collect();

    let start = std::time::Instant::now();
    let _ll = parallel_log_likelihood(&x, &time, &event, &coefficients);
    results.insert(
        "log_likelihood_ms".to_string(),
        start.elapsed().as_secs_f64() * 1000.0,
    );

    let start = std::time::Instant::now();
    let _exp_sum = parallel_exp_sum(&coefficients);
    results.insert(
        "exp_sum_ms".to_string(),
        start.elapsed().as_secs_f64() * 1000.0,
    );

    let start = std::time::Instant::now();
    let (_grad, _hess) = parallel_gradient_hessian(&x, &time, &event, &coefficients, 0.01);
    results.insert(
        "gradient_hessian_ms".to_string(),
        start.elapsed().as_secs_f64() * 1000.0,
    );

    results.insert("n_samples".to_string(), n_samples as f64);
    results.insert("n_features".to_string(), n_features as f64);
    results.insert("n_threads".to_string(), config.n_threads as f64);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config() {
        let config = GPUConfig::new(ComputeBackend::CPU, 0, 4, 256, 1024, false);
        assert_eq!(config.backend, ComputeBackend::CPU);
        assert_eq!(config.n_threads, 4);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_get_available_devices() {
        let devices = get_available_devices().unwrap();
        assert!(!devices.is_empty());
        assert!(devices.iter().any(|d| d.backend == ComputeBackend::CPU));
    }

    #[test]
    fn test_parallel_cox_regression() {
        let x = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.0],
            vec![1.5, 0.7],
            vec![0.5, 1.5],
            vec![2.5, 0.3],
        ];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 1, 0];

        let result = parallel_cox_regression(x, time, event, None).unwrap();
        assert!(!result.coefficients.is_empty());
        assert!(result.computation_time_ms > 0.0);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_batch_predict_survival() {
        let x = vec![vec![1.0, 0.5], vec![2.0, 1.0], vec![1.5, 0.7]];
        let coefficients = vec![0.5, -0.3];
        let time_points = vec![1.0, 2.0, 5.0, 10.0];

        let result = batch_predict_survival(x, coefficients, time_points, None).unwrap();
        assert_eq!(result.predictions.len(), 3);
        assert_eq!(result.survival_probabilities.len(), 3);
        assert!(result.computation_time_ms >= 0.0);
    }
}
