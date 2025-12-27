use ndarray::Array2;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
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

pub struct BootstrapConfig {
    pub n_bootstrap: usize,
    pub confidence_level: f64,
    pub seed: Option<u64>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 1000,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

fn simple_rng(seed: u64, index: usize) -> u64 {
    let mut state = seed.wrapping_add(index as u64);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    state
}

fn bootstrap_sample_indices(n: usize, seed: u64, iteration: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let rng_val = simple_rng(seed.wrapping_add(iteration as u64), i);
        indices.push((rng_val as usize) % n);
    }
    indices
}

pub fn bootstrap_cox(
    time: &[f64],
    status: &[i32],
    covariates: &Array2<f64>,
    weights: Option<&[f64]>,
    config: &BootstrapConfig,
) -> Result<BootstrapResult, Box<dyn std::error::Error>> {
    use crate::regression::coxfit6::{CoxFit, Method as CoxMethod};
    use ndarray::Array1;

    let n = time.len();
    let nvar = covariates.nrows();

    let default_weights: Vec<f64> = vec![1.0; n];
    let weights = weights.unwrap_or(&default_weights);

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| time[b].partial_cmp(&time[a]).unwrap_or(std::cmp::Ordering::Equal));

    let sorted_time: Vec<f64> = sorted_indices.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = sorted_indices.iter().map(|&i| status[i]).collect();
    let sorted_weights: Vec<f64> = sorted_indices.iter().map(|&i| weights[i]).collect();

    let mut sorted_covariates = Array2::zeros((n, nvar));
    for (new_idx, &orig_idx) in sorted_indices.iter().enumerate() {
        for var in 0..nvar {
            sorted_covariates[[new_idx, var]] = covariates[[var, orig_idx]];
        }
    }

    let initial_beta: Vec<f64> = vec![0.0; nvar];

    let time_arr = Array1::from_vec(sorted_time.clone());
    let status_arr = Array1::from_vec(sorted_status.clone());
    let strata_arr = Array1::from_elem(n, 0i32);
    let offset_arr = Array1::from_elem(n, 0.0);
    let weights_arr = Array1::from_vec(sorted_weights.clone());

    let mut original_fit = CoxFit::new(
        time_arr,
        status_arr,
        sorted_covariates.clone(),
        strata_arr,
        offset_arr,
        weights_arr,
        CoxMethod::Breslow,
        25,
        1e-9,
        1e-9,
        vec![true; nvar],
        initial_beta.clone(),
    )?;

    original_fit.fit()?;
    let (original_beta, _, _, _, _, _, _, _) = original_fit.results();

    let seed = config.seed.unwrap_or(42);
    let mut bootstrap_coefs: Vec<Vec<f64>> = Vec::with_capacity(config.n_bootstrap);

    for b in 0..config.n_bootstrap {
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
        boot_indices.sort_by(|&a, &b| boot_time[b].partial_cmp(&boot_time[a]).unwrap_or(std::cmp::Ordering::Equal));

        let resorted_time: Vec<f64> = boot_indices.iter().map(|&i| boot_time[i]).collect();
        let resorted_status: Vec<i32> = boot_indices.iter().map(|&i| boot_status[i]).collect();
        let resorted_weights: Vec<f64> = boot_indices.iter().map(|&i| boot_weights[i]).collect();

        let mut resorted_covariates = Array2::zeros((n, nvar));
        for (new_idx, &orig_idx) in boot_indices.iter().enumerate() {
            for var in 0..nvar {
                resorted_covariates[[new_idx, var]] = boot_covariates[[orig_idx, var]];
            }
        }

        let time_arr = Array1::from_vec(resorted_time);
        let status_arr = Array1::from_vec(resorted_status);
        let strata_arr = Array1::from_elem(n, 0i32);
        let offset_arr = Array1::from_elem(n, 0.0);
        let weights_arr = Array1::from_vec(resorted_weights);

        match CoxFit::new(
            time_arr,
            status_arr,
            resorted_covariates,
            strata_arr,
            offset_arr,
            weights_arr,
            CoxMethod::Breslow,
            25,
            1e-9,
            1e-9,
            vec![true; nvar],
            initial_beta.clone(),
        ) {
            Ok(mut fit) => {
                if fit.fit().is_ok() {
                    let (beta, _, _, _, _, _, _, _) = fit.results();
                    bootstrap_coefs.push(beta);
                }
            }
            Err(_) => continue,
        }
    }

    let actual_n_bootstrap = bootstrap_coefs.len();
    if actual_n_bootstrap == 0 {
        return Err("All bootstrap iterations failed".into());
    }

    let mut means = vec![0.0; nvar];
    for coefs in &bootstrap_coefs {
        for (i, &c) in coefs.iter().enumerate() {
            means[i] += c;
        }
    }
    for m in &mut means {
        *m /= actual_n_bootstrap as f64;
    }

    let mut std_errors = vec![0.0; nvar];
    for coefs in &bootstrap_coefs {
        for (i, &c) in coefs.iter().enumerate() {
            std_errors[i] += (c - means[i]).powi(2);
        }
    }
    for se in &mut std_errors {
        *se = (*se / (actual_n_bootstrap - 1) as f64).sqrt();
    }

    let alpha = 1.0 - config.confidence_level;
    let lower_percentile = (alpha / 2.0 * actual_n_bootstrap as f64) as usize;
    let upper_percentile = ((1.0 - alpha / 2.0) * actual_n_bootstrap as f64) as usize;

    let mut ci_lower = vec![0.0; nvar];
    let mut ci_upper = vec![0.0; nvar];

    for var in 0..nvar {
        let mut var_coefs: Vec<f64> = bootstrap_coefs.iter().map(|c| c[var]).collect();
        var_coefs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ci_lower[var] = var_coefs[lower_percentile.min(actual_n_bootstrap - 1)];
        ci_upper[var] = var_coefs[upper_percentile.min(actual_n_bootstrap - 1)];
    }

    Ok(BootstrapResult {
        coefficients: original_beta,
        std_errors,
        ci_lower,
        ci_upper,
        bootstrap_samples: bootstrap_coefs,
    })
}

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
    let n = time.len();
    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };

    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };

    let config = BootstrapConfig {
        n_bootstrap: n_bootstrap.unwrap_or(1000),
        confidence_level: confidence_level.unwrap_or(0.95),
        seed,
    };

    let weights_ref = weights.as_deref();

    bootstrap_cox(&time, &status, &cov_array, weights_ref, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}

pub fn bootstrap_survreg(
    time: &[f64],
    status: &[f64],
    covariates: &Array2<f64>,
    distribution: &str,
    config: &BootstrapConfig,
) -> Result<BootstrapResult, Box<dyn std::error::Error>> {
    use crate::regression::survreg6::survreg;

    let n = time.len();
    let nvar = covariates.ncols();

    let cov_vecs: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..nvar).map(|j| covariates[[j, i]]).collect())
        .collect();

    let original = survreg(
        time.to_vec(),
        status.to_vec(),
        cov_vecs.clone(),
        None,
        None,
        None,
        None,
        Some(distribution),
        Some(25),
        Some(1e-5),
        Some(1e-9),
    )?;

    let seed = config.seed.unwrap_or(42);
    let mut bootstrap_coefs: Vec<Vec<f64>> = Vec::with_capacity(config.n_bootstrap);

    for b in 0..config.n_bootstrap {
        let indices = bootstrap_sample_indices(n, seed, b);

        let boot_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let boot_status: Vec<f64> = indices.iter().map(|&i| status[i]).collect();
        let boot_covariates: Vec<Vec<f64>> = indices.iter().map(|&i| cov_vecs[i].clone()).collect();

        match survreg(
            boot_time,
            boot_status,
            boot_covariates,
            None,
            None,
            None,
            None,
            Some(distribution),
            Some(25),
            Some(1e-5),
            Some(1e-9),
        ) {
            Ok(result) => {
                bootstrap_coefs.push(result.coefficients);
            }
            Err(_) => continue,
        }
    }

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

    let alpha = 1.0 - config.confidence_level;
    let lower_percentile = (alpha / 2.0 * actual_n_bootstrap as f64) as usize;
    let upper_percentile = ((1.0 - alpha / 2.0) * actual_n_bootstrap as f64) as usize;

    let mut ci_lower = vec![0.0; ncoef];
    let mut ci_upper = vec![0.0; ncoef];

    for var in 0..ncoef {
        let mut var_coefs: Vec<f64> = bootstrap_coefs
            .iter()
            .filter_map(|c| c.get(var).copied())
            .collect();
        if var_coefs.is_empty() {
            continue;
        }
        var_coefs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ci_lower[var] = var_coefs[lower_percentile.min(var_coefs.len() - 1)];
        ci_upper[var] = var_coefs[upper_percentile.min(var_coefs.len() - 1)];
    }

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
    let n = time.len();
    let nvar = if !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };

    let cov_array = if nvar > 0 {
        let flat: Vec<f64> = covariates.into_iter().flatten().collect();
        let temp = Array2::from_shape_vec((n, nvar), flat)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        temp.t().to_owned()
    } else {
        Array2::zeros((0, n))
    };

    let config = BootstrapConfig {
        n_bootstrap: n_bootstrap.unwrap_or(1000),
        confidence_level: confidence_level.unwrap_or(0.95),
        seed,
    };

    let dist = distribution.unwrap_or("weibull");

    bootstrap_survreg(&time, &status, &cov_array, dist, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
}
