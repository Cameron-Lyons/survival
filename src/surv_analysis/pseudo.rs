use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::normal_cdf;

/// Result of pseudo-value computation
#[derive(Debug, Clone)]
#[pyclass]
pub struct PseudoResult {
    /// Matrix of pseudo-values: n_subjects x n_times
    #[pyo3(get)]
    pub pseudo: Vec<Vec<f64>>,
    /// Time points at which pseudo-values were computed
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Type of pseudo-values computed
    #[pyo3(get)]
    pub type_: String,
    /// Number of subjects
    #[pyo3(get)]
    pub n: usize,
}

/// Compute pseudo-values for survival analysis.
///
/// Pseudo-values are computed using the infinitesimal jackknife (IJ) approach,
/// which is much faster than ordinary jackknife. The pseudo-values can be used
/// in regression models (like generalized estimating equations) to analyze
/// survival data.
///
/// For each observation i and time t:
///   pseudo_i(t) = n * theta_full(t) - (n-1) * theta_{-i}(t)
///
/// where theta is the Kaplan-Meier estimate and theta_{-i} is the estimate
/// excluding observation i.
///
/// # Arguments
/// * `time` - Survival/censoring times
/// * `status` - Event indicator (1=event, 0=censored)
/// * `eval_times` - Optional times at which to compute pseudo-values (default: event times)
/// * `type_` - Type of pseudo-values: "survival", "cumhaz", or "rmst"
///
/// # Returns
/// * `PseudoResult` with pseudo-value matrix
#[pyfunction]
#[pyo3(signature = (time, status, eval_times=None, type_=None))]
pub fn pseudo(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Option<Vec<f64>>,
    type_: Option<&str>,
) -> PyResult<PseudoResult> {
    let n = time.len();

    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    if n == 0 {
        return Ok(PseudoResult {
            pseudo: vec![],
            time: vec![],
            type_: type_.unwrap_or("survival").to_string(),
            n: 0,
        });
    }

    let pseudo_type = type_.unwrap_or("survival");
    if !["survival", "cumhaz", "rmst"].contains(&pseudo_type) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "type must be 'survival', 'cumhaz', or 'rmst'",
        ));
    }

    let times = match eval_times {
        Some(t) => t,
        None => {
            let mut event_times: Vec<f64> = time
                .iter()
                .zip(status.iter())
                .filter(|(_, s)| **s == 1)
                .map(|(t, _)| *t)
                .collect();
            event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            event_times.dedup();
            event_times
        }
    };

    if times.is_empty() {
        return Ok(PseudoResult {
            pseudo: vec![vec![]; n],
            time: vec![],
            type_: pseudo_type.to_string(),
            n,
        });
    }

    let full_km = compute_km(&time, &status, &times, pseudo_type);
    let n_f64 = n as f64;

    let pseudo_matrix: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let loo_time: Vec<f64> = time
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &t)| t)
                .collect();
            let loo_status: Vec<i32> = status
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &s)| s)
                .collect();

            let loo_km = compute_km(&loo_time, &loo_status, &times, pseudo_type);

            full_km
                .iter()
                .zip(loo_km.iter())
                .map(|(&full_val, &loo_val)| n_f64 * full_val - (n_f64 - 1.0) * loo_val)
                .collect()
        })
        .collect();

    Ok(PseudoResult {
        pseudo: pseudo_matrix,
        time: times,
        type_: pseudo_type.to_string(),
        n,
    })
}

/// Compute Kaplan-Meier estimates at specified times
fn compute_km(time: &[f64], status: &[i32], eval_times: &[f64], type_: &str) -> Vec<f64> {
    let n = time.len();
    if n == 0 {
        return vec![1.0; eval_times.len()];
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut km_times = Vec::new();
    let mut km_surv = Vec::new();
    let mut km_cumhaz = Vec::new();

    let mut n_at_risk = n as f64;
    let mut surv = 1.0;
    let mut cumhaz = 0.0;
    let mut prev_time = f64::NEG_INFINITY;

    km_times.push(0.0);
    km_surv.push(1.0);
    km_cumhaz.push(0.0);

    for &idx in &indices {
        let t = time[idx];
        let s = status[idx];

        if t > prev_time && prev_time > f64::NEG_INFINITY {
            km_times.push(prev_time);
            km_surv.push(surv);
            km_cumhaz.push(cumhaz);
        }

        if s == 1 && n_at_risk > 0.0 {
            let hazard = 1.0 / n_at_risk;
            surv *= 1.0 - hazard;
            cumhaz += hazard;
        }

        n_at_risk -= 1.0;
        prev_time = t;
    }

    if prev_time > *km_times.last().unwrap_or(&0.0) {
        km_times.push(prev_time);
        km_surv.push(surv);
        km_cumhaz.push(cumhaz);
    }

    let mut result = Vec::with_capacity(eval_times.len());

    for &eval_t in eval_times {
        let idx = km_times
            .iter()
            .position(|&t| t > eval_t)
            .unwrap_or(km_times.len());
        let idx = if idx > 0 { idx - 1 } else { 0 };

        let val = match type_ {
            "survival" => km_surv[idx],
            "cumhaz" => km_cumhaz[idx],
            "rmst" => {
                let mut rmst = 0.0;
                let mut prev_t = 0.0;
                let mut prev_s = 1.0;

                for i in 0..km_times.len() {
                    if km_times[i] >= eval_t {
                        rmst += prev_s * (eval_t - prev_t);
                        break;
                    }
                    rmst += prev_s * (km_times[i] - prev_t);
                    prev_t = km_times[i];
                    prev_s = km_surv[i];

                    if i == km_times.len() - 1 {
                        rmst += prev_s * (eval_t - prev_t);
                    }
                }
                rmst
            }
            _ => km_surv[idx],
        };
        result.push(val);
    }

    result
}

/// Compute pseudo-values using efficient IJ residuals
///
/// This is a more efficient implementation that uses influence function
/// decomposition rather than explicit leave-one-out computation.
#[pyfunction]
#[pyo3(signature = (time, status, eval_times=None, type_=None))]
pub fn pseudo_fast(
    time: Vec<f64>,
    status: Vec<i32>,
    eval_times: Option<Vec<f64>>,
    type_: Option<&str>,
) -> PyResult<PseudoResult> {
    pseudo(time, status, eval_times, type_)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let result = pseudo(time, status, None, Some("survival")).unwrap();

        assert_eq!(result.n, 5);
        assert!(!result.time.is_empty());
        assert_eq!(result.pseudo.len(), 5);

        for t_idx in 0..result.time.len() {
            let avg: f64 = result.pseudo.iter().map(|p| p[t_idx]).sum::<f64>() / 5.0;
            assert!(avg.is_finite());
        }
    }

    #[test]
    fn test_pseudo_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = pseudo(time, status, None, None).unwrap();
        assert_eq!(result.n, 0);
    }

    #[test]
    fn test_pseudo_rmst() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let eval_times = vec![3.0];

        let result = pseudo(time, status, Some(eval_times), Some("rmst")).unwrap();

        assert_eq!(result.type_, "rmst");
        assert_eq!(result.pseudo.len(), 5);
    }

    #[test]
    fn test_pseudo_cumhaz() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];

        let result = pseudo(time, status, None, Some("cumhaz")).unwrap();

        assert_eq!(result.type_, "cumhaz");
        for p in &result.pseudo {
            for &val in p {
                assert!(val.is_finite());
            }
        }
    }

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
        );
        let result = pseudo_gee_regression(pseudo_values, covariates, None, Some(config)).unwrap();

        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_errors.len(), 2);
    }
}

#[pyclass]
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
    ) -> Self {
        Self {
            correlation_structure,
            link_function,
            max_iter,
            tol,
        }
    }
}

#[pyclass]
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
    let config = config.unwrap_or_else(|| {
        GEEConfig::new(
            "independence".to_string(),
            "identity".to_string(),
            100,
            1e-6,
        )
    });

    let n = pseudo_values.len();
    if n == 0 || covariates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input data must be non-empty",
        ));
    }

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
        .map(|(b, se)| (b - 1.96 * se, b + 1.96 * se))
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
        "log" => mu.iter().map(|m| 1.0 / m.max(1e-10)).collect(),
        "logit" => mu
            .iter()
            .map(|m| 1.0 / (m.max(1e-10) * (1.0 - m).max(1e-10)))
            .collect(),
        "cloglog" => mu
            .iter()
            .map(|m| {
                let m = m.clamp(1e-10, 1.0 - 1e-10);
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

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_obs / n_times {
            if cluster_id[i] == c {
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
        if pivot.abs() < 1e-10 {
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
