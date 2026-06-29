use crate::constants::{COX_MAX_ITER, DEFAULT_BOOTSTRAP_SAMPLES};
use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const SURVREG_DISTRIBUTION_ERROR: &str = "distribution must be one of weibull, exponential, rayleigh, extreme, gaussian, logistic, loggaussian, lognormal, or loglogistic";

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BootstrapResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub bootstrap_samples: Vec<Vec<f64>>,
}
#[pymethods]
impl BootstrapResult {
    #[new]
    fn new(
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        bootstrap_samples: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            coefficients,
            std_errors,
            ci_lower,
            ci_upper,
            bootstrap_samples,
        }
    }
}
pub(crate) struct BootstrapConfig {
    pub n_bootstrap: usize,
    pub confidence_level: f64,
    pub seed: Option<u64>,
}
impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: DEFAULT_BOOTSTRAP_SAMPLES,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

#[inline]
fn bootstrap_sample_indices(n: usize, seed: u64, iteration: usize) -> Vec<usize> {
    let mut rng = fastrand::Rng::with_seed(seed.wrapping_add(iteration as u64));
    (0..n).map(|_| rng.usize(..n)).collect()
}

fn validate_bootstrap_inputs(n_bootstrap: usize, confidence_level: f64) -> PyResult<()> {
    if n_bootstrap < 2 {
        return Err(PyValueError::new_err("n_bootstrap must be at least 2"));
    }

    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err(PyValueError::new_err(
            "confidence_level must be between 0 and 1",
        ));
    }

    Ok(())
}

fn validate_bootstrap_time(time: &[f64], require_positive: bool) -> PyResult<()> {
    if time.is_empty() {
        return Err(PyValueError::new_err("time must not be empty"));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "time contains non-finite value at index {idx}"
            )));
        }
        if require_positive {
            if value <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "time[{idx}] must be positive"
                )));
            }
        } else if value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "time contains negative value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_bootstrap_status_i32(status: &[i32], n: usize) -> PyResult<()> {
    if status.len() != n {
        return Err(PyValueError::new_err(format!(
            "status length mismatch: expected {n}, got {}",
            status.len()
        )));
    }
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(PyValueError::new_err(format!(
                "status must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_bootstrap_status_f64(status: &[f64], n: usize) -> PyResult<()> {
    if status.len() != n {
        return Err(PyValueError::new_err(format!(
            "status length mismatch: expected {n}, got {}",
            status.len()
        )));
    }
    for (idx, &value) in status.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "status contains non-finite value at index {idx}"
            )));
        }
        if value != 0.0 && value != 1.0 && value != 2.0 {
            return Err(PyValueError::new_err(format!(
                "status must contain only 0/1/2 values for bootstrap_survreg_ci; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_bootstrap_covariates(covariates: &[Vec<f64>], n: usize) -> PyResult<usize> {
    if covariates.len() != n {
        return Err(PyValueError::new_err(format!(
            "covariates row count mismatch: expected {n}, got {}",
            covariates.len()
        )));
    }
    let nvar = covariates.first().map_or(0, Vec::len);
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != nvar {
            return Err(PyValueError::new_err(format!(
                "covariates row {row_idx} length mismatch: expected {nvar}, got {}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "covariates contains non-finite value at row {row_idx}, column {col_idx}"
                )));
            }
        }
    }
    Ok(nvar)
}

fn validate_bootstrap_weights(weights: Option<&[f64]>, n: usize) -> PyResult<()> {
    let Some(weights) = weights else {
        return Ok(());
    };
    if weights.len() != n {
        return Err(PyValueError::new_err(format!(
            "weights length mismatch: expected {n}, got {}",
            weights.len()
        )));
    }
    for (idx, &value) in weights.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "weights contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "weights contains negative value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn canonical_survreg_bootstrap_distribution(distribution: &str) -> Option<&'static str> {
    let normalized = distribution.trim().to_lowercase().replace(['-', ' '], "_");
    match normalized.as_str() {
        "weibull" => Some("weibull"),
        "exponential" => Some("exponential"),
        "rayleigh" => Some("rayleigh"),
        "extreme" | "extreme_value" | "extremevalue" => Some("extreme_value"),
        "gaussian" | "normal" => Some("gaussian"),
        "logistic" => Some("logistic"),
        "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => Some("lognormal"),
        "loglogistic" | "log_logistic" => Some("loglogistic"),
        _ => None,
    }
}

fn validate_survreg_bootstrap_distribution(distribution: &str) -> PyResult<&'static str> {
    canonical_survreg_bootstrap_distribution(distribution)
        .ok_or_else(|| PyValueError::new_err(SURVREG_DISTRIBUTION_ERROR))
}

pub(crate) fn bootstrap_cox(
    time: &[f64],
    status: &[i32],
    covariates: &Array2<f64>,
    weights: Option<&[f64]>,
    config: &BootstrapConfig,
) -> Result<BootstrapResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::cox_optimizer::{CoxFitBuilder, Method as CoxMethod};
    use ndarray::Array1;
    let n = time.len();
    let nvar = covariates.nrows();
    let default_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&default_weights);
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].total_cmp(&time[a]));
    let sorted_time: Vec<f64> = sorted_indices.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = sorted_indices.iter().map(|&i| status[i]).collect();
    let sorted_weights: Vec<f64> = sorted_indices.iter().map(|&i| weights[i]).collect();
    let mut sorted_covariates = Array2::zeros((n, nvar));
    for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
        for var in 0..nvar {
            sorted_covariates[[new_idx, var]] = covariates[[var, orig_idx]];
        }
    }
    let time_arr = Array1::from_vec(sorted_time.clone());
    let status_arr = Array1::from_vec(sorted_status.clone());
    let weights_arr = Array1::from_vec(sorted_weights.clone());
    let mut original_fit = CoxFitBuilder::new(time_arr, status_arr, sorted_covariates.clone())
        .weights(weights_arr)
        .method(CoxMethod::Breslow)
        .max_iter(COX_MAX_ITER)
        .eps(1e-9)
        .toler(1e-9)
        .build()?;
    original_fit.fit()?;
    let (original_beta, _, _, _, _, _, _, _) = original_fit.results();
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
    let bootstrap_coefs: Vec<Vec<f64>> = (0..config.n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let indices = bootstrap_sample_indices(n, seed, b);
            let boot_time: Vec<f64> = indices.iter().map(|&i| sorted_time[i]).collect();
            let boot_status: Vec<i32> = indices.iter().map(|&i| sorted_status[i]).collect();
            let boot_weights: Vec<f64> = indices.iter().map(|&i| sorted_weights[i]).collect();
            let mut boot_covariates = Array2::zeros((n, nvar));
            for (new_idx, &orig_idx) in indices.iter().enumerate() {
                for var in 0..nvar {
                    boot_covariates[[new_idx, var]] = sorted_covariates[[orig_idx, var]];
                }
            }
            let mut boot_indices: Vec<usize> = (0..n).collect();
            boot_indices.sort_by(|&a, &b| boot_time[b].total_cmp(&boot_time[a]));
            let resorted_time: Vec<f64> = boot_indices.iter().map(|&i| boot_time[i]).collect();
            let resorted_status: Vec<i32> = boot_indices.iter().map(|&i| boot_status[i]).collect();
            let resorted_weights: Vec<f64> =
                boot_indices.iter().map(|&i| boot_weights[i]).collect();
            let mut resorted_covariates = Array2::zeros((n, nvar));
            for (new_idx, &orig_idx) in boot_indices.iter().enumerate() {
                for var in 0..nvar {
                    resorted_covariates[[new_idx, var]] = boot_covariates[[orig_idx, var]];
                }
            }
            let time_arr = Array1::from_vec(resorted_time);
            let status_arr = Array1::from_vec(resorted_status);
            let weights_arr = Array1::from_vec(resorted_weights);
            match CoxFitBuilder::new(time_arr, status_arr, resorted_covariates)
                .weights(weights_arr)
                .method(CoxMethod::Breslow)
                .max_iter(COX_MAX_ITER)
                .eps(1e-9)
                .toler(1e-9)
                .build()
            {
                Ok(mut fit) => {
                    if fit.fit().is_ok() {
                        let (beta, _, _, _, _, _, _, _) = fit.results();
                        Some(beta)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        })
        .collect();
    let actual_n_bootstrap = bootstrap_coefs.len();
    if actual_n_bootstrap == 0 {
        return Err("All bootstrap iterations failed".into());
    }

    let mut transposed: Vec<Vec<f64>> = vec![Vec::with_capacity(actual_n_bootstrap); nvar];
    for coefs in &bootstrap_coefs {
        for (i, &c) in coefs.iter().enumerate() {
            transposed[i].push(c);
        }
    }

    let means: Vec<f64> = transposed
        .iter()
        .map(|col| col.iter().sum::<f64>() / actual_n_bootstrap as f64)
        .collect();

    let std_errors: Vec<f64> = if actual_n_bootstrap > 1 {
        transposed
            .iter()
            .zip(means.iter())
            .map(|(col, &mean)| {
                let variance = col.iter().map(|&c| (c - mean).powi(2)).sum::<f64>()
                    / (actual_n_bootstrap - 1) as f64;
                variance.sqrt()
            })
            .collect()
    } else {
        vec![0.0; nvar]
    };

    let alpha = 1.0 - config.confidence_level;
    let lower_percentile = (alpha / 2.0 * actual_n_bootstrap as f64) as usize;
    let upper_percentile = ((1.0 - alpha / 2.0) * actual_n_bootstrap as f64) as usize;

    let ci_results: Vec<(f64, f64)> = transposed
        .into_par_iter()
        .map(|mut var_coefs| {
            var_coefs.sort_by(f64::total_cmp);
            (
                var_coefs[lower_percentile.min(actual_n_bootstrap - 1)],
                var_coefs[upper_percentile.min(actual_n_bootstrap - 1)],
            )
        })
        .collect();

    let (ci_lower, ci_upper): (Vec<f64>, Vec<f64>) = ci_results.into_iter().unzip();
    Ok(BootstrapResult {
        coefficients: original_beta,
        std_errors,
        ci_lower,
        ci_upper,
        bootstrap_samples: bootstrap_coefs,
    })
}
/// Compute bootstrap confidence intervals for Cox regression coefficients.
///
/// Parameters
/// ----------
/// time : array-like
///     Survival/censoring times.
/// status : array-like
///     Event indicator (1=event, 0=censored).
/// covariates : list of lists
///     Covariate matrix (n_obs x n_vars).
/// weights : array-like, optional
///     Case weights.
/// n_bootstrap : int, optional
///     Number of bootstrap samples (default 1000).
/// confidence_level : float, optional
///     Confidence level for intervals (default 0.95).
/// seed : int, optional
///     Random seed for reproducibility.
///
/// Returns
/// -------
/// BootstrapResult
///     Object with: coefficients, std_errors, ci_lower, ci_upper, bootstrap_samples.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, weights=None, n_bootstrap=None, confidence_level=None, seed=None))]
pub fn bootstrap_cox_ci(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> PyResult<BootstrapResult> {
    let n_bootstrap = n_bootstrap.unwrap_or(DEFAULT_BOOTSTRAP_SAMPLES);
    let confidence_level = confidence_level.unwrap_or(0.95);
    validate_bootstrap_inputs(n_bootstrap, confidence_level)?;

    let n = time.len();
    validate_bootstrap_time(&time, false)?;
    validate_bootstrap_status_i32(&status, n)?;
    let nvar = validate_bootstrap_covariates(&covariates, n)?;
    validate_bootstrap_weights(weights.as_deref(), n)?;
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = BootstrapConfig {
        n_bootstrap,
        confidence_level,
        seed,
    };
    let weights_ref = weights.as_deref();
    bootstrap_cox(&time, &status, &cov_array, weights_ref, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}
pub(crate) fn bootstrap_survreg(
    time: &[f64],
    status: &[f64],
    covariates: &Array2<f64>,
    distribution: &str,
    config: &BootstrapConfig,
) -> Result<BootstrapResult, Box<dyn std::error::Error + Send + Sync>> {
    use crate::regression::parametric_survival::survreg;
    let n = time.len();
    let nvar = covariates.nrows();
    let cov_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nvar).map(|j| covariates[[j, i]]).collect())
        .collect();
    let dist_str = canonical_survreg_bootstrap_distribution(distribution).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, SURVREG_DISTRIBUTION_ERROR)
    })?;

    let original = survreg(
        time.to_vec(),
        status.to_vec(),
        cov_vecs.clone(),
        None,
        None,
        None,
        None,
        Some(dist_str),
        Some(COX_MAX_ITER),
        Some(1e-5),
        Some(1e-9),
        None,
        None,
        None,
    )?;
    let seed = config.seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);
    let bootstrap_coefs: Vec<Vec<f64>> = (0..config.n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let indices = bootstrap_sample_indices(n, seed, b);
            let boot_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let boot_status: Vec<f64> = indices.iter().map(|&i| status[i]).collect();
            let boot_covariates: Vec<Vec<f64>> =
                indices.iter().map(|&i| cov_vecs[i].clone()).collect();
            match survreg(
                boot_time,
                boot_status,
                boot_covariates,
                None,
                None,
                None,
                None,
                Some(dist_str),
                Some(COX_MAX_ITER),
                Some(1e-5),
                Some(1e-9),
                None,
                None,
                None,
            ) {
                Ok(result) => Some(result.coefficients),
                Err(_) => None,
            }
        })
        .collect();
    let actual_n_bootstrap = bootstrap_coefs.len();
    if actual_n_bootstrap == 0 {
        return Err("All bootstrap iterations failed".into());
    }
    let ncoef = original.coefficients.len();
    let mut means = vec![0.0; ncoef];
    for coefs in &bootstrap_coefs {
        for (i, &c) in coefs.iter().enumerate() {
            if i < ncoef {
                means[i] += c;
            }
        }
    }
    for m in &mut means {
        *m /= actual_n_bootstrap as f64;
    }
    let mut std_errors = vec![0.0; ncoef];
    if actual_n_bootstrap > 1 {
        for coefs in &bootstrap_coefs {
            for (i, &c) in coefs.iter().enumerate() {
                if i < ncoef {
                    std_errors[i] += (c - means[i]).powi(2);
                }
            }
        }
        for se in &mut std_errors {
            *se = (*se / (actual_n_bootstrap - 1) as f64).sqrt();
        }
    }
    let alpha = 1.0 - config.confidence_level;
    let lower_percentile = (alpha / 2.0 * actual_n_bootstrap as f64) as usize;
    let upper_percentile = ((1.0 - alpha / 2.0) * actual_n_bootstrap as f64) as usize;

    let ci_results: Vec<(f64, f64)> = (0..ncoef)
        .into_par_iter()
        .map(|var| {
            let mut var_coefs: Vec<f64> = bootstrap_coefs
                .iter()
                .filter_map(|c| c.get(var).copied())
                .collect();
            if var_coefs.is_empty() {
                return (0.0, 0.0);
            }
            var_coefs.sort_by(f64::total_cmp);
            (
                var_coefs[lower_percentile.min(var_coefs.len() - 1)],
                var_coefs[upper_percentile.min(var_coefs.len() - 1)],
            )
        })
        .collect();

    let (ci_lower, ci_upper): (Vec<f64>, Vec<f64>) = ci_results.into_iter().unzip();
    Ok(BootstrapResult {
        coefficients: original.coefficients,
        std_errors,
        ci_lower,
        ci_upper,
        bootstrap_samples: bootstrap_coefs,
    })
}
#[pyfunction]
#[pyo3(signature = (time, status, covariates, distribution=None, n_bootstrap=None, confidence_level=None, seed=None))]
pub fn bootstrap_survreg_ci(
    time: Vec<f64>,
    status: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    distribution: Option<&str>,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> PyResult<BootstrapResult> {
    let n_bootstrap = n_bootstrap.unwrap_or(DEFAULT_BOOTSTRAP_SAMPLES);
    let confidence_level = confidence_level.unwrap_or(0.95);
    validate_bootstrap_inputs(n_bootstrap, confidence_level)?;

    let n = time.len();
    validate_bootstrap_time(&time, true)?;
    validate_bootstrap_status_f64(&status, n)?;
    let nvar = validate_bootstrap_covariates(&covariates, n)?;
    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };
    let config = BootstrapConfig {
        n_bootstrap,
        confidence_level,
        seed,
    };
    let dist = validate_survreg_bootstrap_distribution(distribution.unwrap_or("weibull"))?;
    bootstrap_survreg(&time, &status, &cov_array, dist, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}

#[cfg(test)]
mod tests {
    use super::{BootstrapConfig, bootstrap_survreg, canonical_survreg_bootstrap_distribution};
    use ndarray::Array2;

    #[test]
    fn test_survreg_bootstrap_distribution_aliases_are_explicit() {
        assert_eq!(
            canonical_survreg_bootstrap_distribution("weibull"),
            Some("weibull")
        );
        assert_eq!(
            canonical_survreg_bootstrap_distribution(" exponential "),
            Some("exponential")
        );
        assert_eq!(
            canonical_survreg_bootstrap_distribution("rayleigh"),
            Some("rayleigh")
        );
        assert_eq!(
            canonical_survreg_bootstrap_distribution("normal"),
            Some("gaussian")
        );
        assert_eq!(
            canonical_survreg_bootstrap_distribution("log-normal"),
            Some("lognormal")
        );
        assert_eq!(
            canonical_survreg_bootstrap_distribution("log logistic"),
            Some("loglogistic")
        );
        assert_eq!(canonical_survreg_bootstrap_distribution("mystery"), None);
    }

    #[test]
    fn test_bootstrap_survreg_transposed_covariates_smoke() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let covariates = Array2::from_shape_vec(
            (2, 6),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        .unwrap();
        let config = BootstrapConfig {
            n_bootstrap: 8,
            confidence_level: 0.95,
            seed: Some(123),
        };

        let result = bootstrap_survreg(&time, &status, &covariates, "weibull", &config).unwrap();

        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.std_errors.len(), 3);
        assert_eq!(result.ci_lower.len(), 3);
        assert_eq!(result.ci_upper.len(), 3);
        assert!(!result.bootstrap_samples.is_empty());
    }
}
