//! Generalized estimating equations for pseudo-value regression (Andersen,
//! Klein & Rosthoj 2003); not part of R's `survival`, kept as is.

use crate::constants::{DIVISION_FLOOR, normal_ci_95};
use crate::internal::statistical::normal_cdf;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GEEConfig {
    #[pyo3(get, set)]
    pub correlation_structure: String,
    #[pyo3(get, set)]
    pub link_function: String,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
}

#[pymethods]
impl GEEConfig {
    #[new]
    #[pyo3(signature = (correlation_structure="independence".to_string(), link_function="identity".to_string(), max_iter=100, tol=1e-6))]
    pub fn new(
        correlation_structure: String,
        link_function: String,
        max_iter: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let config = Self {
            correlation_structure,
            link_function,
            max_iter,
            tol,
        };
        validate_gee_config(&config)?;
        Ok(config)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct GEEResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub z_values: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub confidence_intervals: Vec<(f64, f64)>,
    #[pyo3(get)]
    pub qic: f64,
    #[pyo3(get)]
    pub n_iterations: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl GEEResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        z_values: Vec<f64>,
        p_values: Vec<f64>,
        confidence_intervals: Vec<(f64, f64)>,
        qic: f64,
        n_iterations: usize,
        converged: bool,
    ) -> Self {
        Self {
            coefficients,
            std_errors,
            z_values,
            p_values,
            confidence_intervals,
            qic,
            n_iterations,
            converged,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (pseudo_values, covariates, cluster_id=None, config=None))]
pub fn pseudo_gee_regression(
    pseudo_values: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    cluster_id: Option<Vec<usize>>,
    config: Option<GEEConfig>,
) -> PyResult<GEEResult> {
    let config = match config {
        Some(config) => {
            validate_gee_config(&config)?;
            config
        }
        None => GEEConfig::new(
            "independence".to_string(),
            "identity".to_string(),
            100,
            1e-6,
        )?,
    };

    validate_pseudo_gee_inputs(&pseudo_values, &covariates, cluster_id.as_deref())?;

    let n = pseudo_values.len();
    let n_times = pseudo_values[0].len();
    let p = covariates[0].len();
    let cluster_id = cluster_id.unwrap_or_else(|| (0..n).collect());

    let y: Vec<f64> = pseudo_values
        .iter()
        .flat_map(|pv| pv.iter().cloned())
        .collect();
    let n_obs = y.len();

    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for cov in covariates.iter() {
        for _ in 0..n_times {
            x.push(cov.clone());
        }
    }

    let mut beta: Vec<f64> = vec![0.0; p];
    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..config.max_iter {
        n_iterations = iter + 1;

        let eta: Vec<f64> = x
            .iter()
            .map(|xi| xi.iter().zip(beta.iter()).map(|(x, b)| x * b).sum())
            .collect();

        let mu: Vec<f64> = apply_link_inverse(&eta, &config.link_function);

        let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(y, m)| y - m).collect();

        let link_deriv: Vec<f64> = compute_link_derivative(&mu, &config.link_function);

        let mut xtx = vec![vec![0.0; p]; p];
        let mut xty = vec![0.0; p];

        for i in 0..n_obs {
            let w = link_deriv[i].powi(2);
            for j in 0..p {
                xty[j] += x[i][j] * residuals[i] * w;
                for k in 0..p {
                    xtx[j][k] += x[i][j] * x[i][k] * w;
                }
            }
        }

        let xtx_inv = invert_matrix(&xtx);
        let delta: Vec<f64> = (0..p)
            .map(|j| xtx_inv[j].iter().zip(xty.iter()).map(|(a, b)| a * b).sum())
            .collect();

        let delta_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        if delta_norm < config.tol {
            converged = true;
            break;
        }

        for k in 0..p {
            beta[k] += delta[k];
        }
    }

    let eta: Vec<f64> = x
        .iter()
        .map(|xi| xi.iter().zip(beta.iter()).map(|(x, b)| x * b).sum())
        .collect();
    let mu: Vec<f64> = apply_link_inverse(&eta, &config.link_function);
    let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(y, m)| y - m).collect();

    let sandwich_variance =
        compute_sandwich_variance(&x, &residuals, &cluster_id, n_times, p, &config);

    let std_errors: Vec<f64> = (0..p).map(|k| sandwich_variance[k][k].sqrt()).collect();

    let z_values: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(b, se)| if *se > 0.0 { b / se } else { f64::NAN })
        .collect();

    let p_values: Vec<f64> = z_values
        .iter()
        .map(|z| {
            if z.is_finite() {
                2.0 * (1.0 - normal_cdf(z.abs()))
            } else {
                f64::NAN
            }
        })
        .collect();

    let confidence_intervals: Vec<(f64, f64)> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&beta, &std_error)| normal_ci_95(beta, std_error))
        .collect();

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let qic = n_obs as f64 * (rss / n_obs as f64).ln() + 2.0 * p as f64;

    Ok(GEEResult {
        coefficients: beta,
        std_errors,
        z_values,
        p_values,
        confidence_intervals,
        qic,
        n_iterations,
        converged,
    })
}

fn validate_gee_config(config: &GEEConfig) -> PyResult<()> {
    match config.correlation_structure.as_str() {
        "independence" | "exchangeable" | "ar1" => {}
        _ => {
            return Err(PyValueError::new_err(
                "correlation_structure must be 'independence', 'exchangeable', or 'ar1'",
            ));
        }
    }

    match config.link_function.as_str() {
        "identity" | "log" | "logit" | "cloglog" => {}
        _ => {
            return Err(PyValueError::new_err(
                "link_function must be 'identity', 'log', 'logit', or 'cloglog'",
            ));
        }
    }

    if config.max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(PyValueError::new_err(
            "tol must be finite and strictly positive",
        ));
    }

    Ok(())
}

fn validate_matrix_values(
    matrix: &[Vec<f64>],
    name: &'static str,
    require_non_empty_rows: bool,
) -> PyResult<usize> {
    let n_cols = matrix
        .first()
        .ok_or_else(|| PyValueError::new_err("Input data must be non-empty"))?
        .len();
    if require_non_empty_rows && n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} rows must not be empty"
        )));
    }

    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != n_cols {
            return Err(PyValueError::new_err(format!(
                "{name} row {row_idx} has {} columns, expected {n_cols}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if value.is_nan() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains NaN at row {row_idx}, column {col_idx}"
                )));
            }
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} contains non-finite value {value} at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }

    Ok(n_cols)
}

fn validate_pseudo_gee_inputs(
    pseudo_values: &[Vec<f64>],
    covariates: &[Vec<f64>],
    cluster_id: Option<&[usize]>,
) -> PyResult<()> {
    if pseudo_values.is_empty() || covariates.is_empty() {
        return Err(PyValueError::new_err("Input data must be non-empty"));
    }
    if covariates.len() != pseudo_values.len() {
        return Err(PyValueError::new_err(format!(
            "covariates length must equal pseudo_values length; got {} and {}",
            covariates.len(),
            pseudo_values.len()
        )));
    }

    validate_matrix_values(pseudo_values, "pseudo_values", true)?;
    validate_matrix_values(covariates, "covariates", true)?;

    if let Some(cluster_id) = cluster_id
        && cluster_id.len() != pseudo_values.len()
    {
        return Err(PyValueError::new_err(format!(
            "cluster_id length must equal pseudo_values length; got {} and {}",
            cluster_id.len(),
            pseudo_values.len()
        )));
    }

    Ok(())
}

fn apply_link_inverse(eta: &[f64], link: &str) -> Vec<f64> {
    match link {
        "identity" => eta.to_vec(),
        "log" => eta.iter().map(|e| e.exp()).collect(),
        "logit" => eta.iter().map(|e| 1.0 / (1.0 + (-e).exp())).collect(),
        "cloglog" => eta.iter().map(|e| 1.0 - (-e.exp()).exp()).collect(),
        _ => eta.to_vec(),
    }
}

fn compute_link_derivative(mu: &[f64], link: &str) -> Vec<f64> {
    match link {
        "identity" => vec![1.0; mu.len()],
        "log" => mu.iter().map(|m| 1.0 / m.max(DIVISION_FLOOR)).collect(),
        "logit" => mu
            .iter()
            .map(|m| 1.0 / (m.max(DIVISION_FLOOR) * (1.0 - m).max(DIVISION_FLOOR)))
            .collect(),
        "cloglog" => mu
            .iter()
            .map(|m| {
                let m = m.clamp(DIVISION_FLOOR, 1.0 - DIVISION_FLOOR);
                1.0 / ((1.0 - m) * (-(1.0 - m).ln()))
            })
            .collect(),
        _ => vec![1.0; mu.len()],
    }
}

fn compute_sandwich_variance(
    x: &[Vec<f64>],
    residuals: &[f64],
    cluster_id: &[usize],
    n_times: usize,
    p: usize,
    _config: &GEEConfig,
) -> Vec<Vec<f64>> {
    let n_obs = x.len();

    let mut xtx = vec![vec![0.0; p]; p];
    for xi in x.iter() {
        for j in 0..p {
            for k in 0..p {
                xtx[j][k] += xi[j] * xi[k];
            }
        }
    }
    let xtx_inv = invert_matrix(&xtx);

    let mut meat = vec![vec![0.0; p]; p];
    let max_cluster = *cluster_id.iter().max().unwrap_or(&0);

    for c in 0..=max_cluster {
        let mut score = vec![0.0; p];
        for (i, &cluster) in cluster_id.iter().enumerate().take(n_obs / n_times) {
            if cluster == c {
                for t in 0..n_times {
                    let idx = i * n_times + t;
                    for j in 0..p {
                        score[j] += x[idx][j] * residuals[idx];
                    }
                }
            }
        }

        for j in 0..p {
            for k in 0..p {
                meat[j][k] += score[j] * score[k];
            }
        }
    }

    let mut result = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            for k in 0..p {
                for l in 0..p {
                    result[i][j] += xtx_inv[i][k] * meat[k][l] * xtx_inv[l][j];
                }
            }
        }
    }

    result
}

fn invert_matrix(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    if n == 0 {
        return vec![];
    }

    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        let pivot = aug[i][i];
        if pivot.abs() < DIVISION_FLOOR {
            continue;
        }

        for val in aug[i].iter_mut() {
            *val /= pivot;
        }

        let row_i = aug[i].clone();
        for (k, row_k) in aug.iter_mut().enumerate() {
            if k != i {
                let factor = row_k[i];
                for (val, &ri) in row_k.iter_mut().zip(row_i.iter()) {
                    *val -= factor * ri;
                }
            }
        }
    }

    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = aug[i][n + j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_gee_regression() {
        let pseudo_values = vec![vec![0.8], vec![0.7], vec![0.6], vec![0.5], vec![0.4]];
        let covariates = vec![
            vec![1.0, 0.5],
            vec![1.0, 1.0],
            vec![1.0, 1.5],
            vec![1.0, 2.0],
            vec![1.0, 2.5],
        ];

        let config = GEEConfig::new(
            "independence".to_string(),
            "identity".to_string(),
            100,
            1e-6,
        )
        .unwrap();
        let result = pseudo_gee_regression(pseudo_values, covariates, None, Some(config)).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_errors.len(), 2);
    }

    #[test]
    fn test_pseudo_gee_rejects_malformed_inputs() {
        let err = GEEConfig::new("weird".to_string(), "identity".to_string(), 100, 1e-6)
            .expect_err("invalid correlation structure should fail");
        assert!(err.to_string().contains("correlation_structure"));

        let err = GEEConfig::new("independence".to_string(), "identity".to_string(), 0, 1e-6)
            .expect_err("zero max_iter should fail");
        assert!(err.to_string().contains("max_iter"));

        let err = pseudo_gee_regression(
            vec![vec![0.8], vec![0.7, 0.6]],
            vec![vec![1.0], vec![1.0]],
            None,
            None,
        )
        .expect_err("ragged pseudo_values should fail");
        assert!(err.to_string().contains("pseudo_values row 1"));

        let err = pseudo_gee_regression(
            vec![vec![0.8], vec![0.7]],
            vec![vec![1.0], vec![1.0]],
            Some(vec![0]),
            None,
        )
        .expect_err("short cluster_id should fail");
        assert!(err.to_string().contains("cluster_id length"));
    }
}
