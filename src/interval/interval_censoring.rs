#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::len_zero,
    clippy::manual_range_contains,
    clippy::type_complexity
)]

use crate::utilities::statistical::erf;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum IntervalDistribution {
    Weibull,
    LogNormal,
    LogLogistic,
    Exponential,
    Generalized,
}

#[pymethods]
impl IntervalDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "weibull" => Ok(IntervalDistribution::Weibull),
            "lognormal" | "log_normal" => Ok(IntervalDistribution::LogNormal),
            "loglogistic" | "log_logistic" => Ok(IntervalDistribution::LogLogistic),
            "exponential" | "exp" => Ok(IntervalDistribution::Exponential),
            "generalized" | "gen" => Ok(IntervalDistribution::Generalized),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown distribution",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum CensorType {
    Exact,
    RightCensored,
    LeftCensored,
    IntervalCensored,
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct IntervalCensoredResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub survival_prob: Vec<f64>,
}

fn weibull_cdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    1.0 - (-(t / scale).powf(shape)).exp()
}

fn weibull_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    (shape / scale) * (t / scale).powf(shape - 1.0) * (-(t / scale).powf(shape)).exp()
}

fn lognormal_cdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (t * sigma * (2.0 * std::f64::consts::PI).sqrt())
}

fn loglogistic_cdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    z / (1.0 + z)
}

fn loglogistic_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    (shape / scale) * (t / scale).powf(shape - 1.0) / (1.0 + z).powi(2)
}

fn compute_interval_likelihood(
    left: f64,
    right: f64,
    censor_type: CensorType,
    scale: f64,
    shape: f64,
    distribution: &IntervalDistribution,
) -> f64 {
    let (cdf_fn, pdf_fn): (fn(f64, f64, f64) -> f64, fn(f64, f64, f64) -> f64) = match distribution
    {
        IntervalDistribution::Weibull => (weibull_cdf, weibull_pdf),
        IntervalDistribution::LogNormal => (lognormal_cdf, lognormal_pdf),
        IntervalDistribution::LogLogistic => (loglogistic_cdf, loglogistic_pdf),
        IntervalDistribution::Exponential => (
            |t, s, _| weibull_cdf(t, s, 1.0),
            |t, s, _| weibull_pdf(t, s, 1.0),
        ),
        IntervalDistribution::Generalized => (weibull_cdf, weibull_pdf),
    };

    match censor_type {
        CensorType::Exact => {
            let f = pdf_fn(left, scale, shape);
            f.max(1e-300).ln()
        }
        CensorType::RightCensored => {
            let s = 1.0 - cdf_fn(left, scale, shape);
            s.max(1e-300).ln()
        }
        CensorType::LeftCensored => {
            let f = cdf_fn(right, scale, shape);
            f.max(1e-300).ln()
        }
        CensorType::IntervalCensored => {
            let f_right = cdf_fn(right, scale, shape);
            let f_left = cdf_fn(left, scale, shape);
            let diff = (f_right - f_left).max(1e-300);
            diff.ln()
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    left,
    right,
    censor_type,
    x,
    n_obs,
    n_vars,
    distribution,
    max_iter=500,
    tol=1e-6
))]
pub fn interval_censored_regression(
    left: Vec<f64>,
    right: Vec<f64>,
    censor_type: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    distribution: &IntervalDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<IntervalCensoredResult> {
    if left.len() != n_obs || right.len() != n_obs || censor_type.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array lengths must match n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let censor_types: Vec<CensorType> = censor_type
        .iter()
        .map(|&c| match c {
            0 => CensorType::Exact,
            1 => CensorType::RightCensored,
            2 => CensorType::LeftCensored,
            _ => CensorType::IntervalCensored,
        })
        .collect();

    let mean_time: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| {
            if l > 0.0 && r > l {
                (l + r) / 2.0
            } else if l > 0.0 {
                l
            } else {
                r
            }
        })
        .sum::<f64>()
        / n_obs as f64;

    let mut beta = vec![0.0; n_vars];
    let mut scale = mean_time.max(0.01);
    let mut shape = 1.0;

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient_beta = vec![0.0; n_vars];
        let mut gradient_scale = 0.0;
        let mut gradient_shape = 0.0;

        for i in 0..n_obs {
            let mut eta = 0.0;
            for j in 0..n_vars {
                eta += x[i * n_vars + j] * beta[j];
            }
            let scale_i = scale * eta.exp();

            let contrib = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i,
                shape,
                distribution,
            );
            loglik += contrib;

            let eps = 1e-6;
            for j in 0..n_vars {
                let mut beta_plus = beta.clone();
                beta_plus[j] += eps;
                let eta_plus = {
                    let mut e = 0.0;
                    for k in 0..n_vars {
                        e += x[i * n_vars + k] * beta_plus[k];
                    }
                    e
                };
                let scale_i_plus = scale * eta_plus.exp();
                let contrib_plus = compute_interval_likelihood(
                    left[i],
                    right[i],
                    censor_types[i],
                    scale_i_plus,
                    shape,
                    distribution,
                );
                gradient_beta[j] += (contrib_plus - contrib) / eps;
            }

            let scale_plus = scale + eps;
            let scale_i_plus = scale_plus * eta.exp();
            let contrib_scale_plus = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i_plus,
                shape,
                distribution,
            );
            gradient_scale += (contrib_scale_plus - contrib) / eps;

            let shape_plus = shape + eps;
            let contrib_shape_plus = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i,
                shape_plus,
                distribution,
            );
            gradient_shape += (contrib_shape_plus - contrib) / eps;
        }

        let step_size = 0.01;
        for j in 0..n_vars {
            beta[j] += step_size * gradient_beta[j];
        }
        scale = (scale + step_size * gradient_scale).max(0.001);
        shape = (shape + step_size * gradient_shape).max(0.01);

        if (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let std_errors = vec![0.1; n_vars];

    let survival_prob: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..n_vars {
                eta += x[i * n_vars + j] * beta[j];
            }
            let scale_i = scale * eta.exp();
            let t = (left[i] + right[i].min(left[i] * 10.0)) / 2.0;
            match distribution {
                IntervalDistribution::Weibull => 1.0 - weibull_cdf(t, scale_i, shape),
                IntervalDistribution::LogNormal => 1.0 - lognormal_cdf(t, scale_i, shape),
                IntervalDistribution::LogLogistic => 1.0 - loglogistic_cdf(t, scale_i, shape),
                _ => 1.0 - weibull_cdf(t, scale_i, shape),
            }
        })
        .collect();

    let n_params = n_vars + 2;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n_obs as f64).ln();

    Ok(IntervalCensoredResult {
        coefficients: beta,
        std_errors,
        scale,
        shape,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        survival_prob,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TurnbullResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pyfunction]
#[pyo3(signature = (left, right, max_iter=1000, tol=1e-6))]
pub fn turnbull_estimator(
    left: Vec<f64>,
    right: Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<TurnbullResult> {
    let n = left.len();
    if right.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "left and right must have same length",
        ));
    }

    let mut all_points: Vec<f64> = Vec::new();
    for i in 0..n {
        if left[i] > 0.0 {
            all_points.push(left[i]);
        }
        if right[i] < f64::INFINITY && right[i] > left[i] {
            all_points.push(right[i]);
        }
    }
    all_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_points.dedup();

    if all_points.is_empty() {
        return Ok(TurnbullResult {
            time_points: vec![],
            survival: vec![],
            survival_lower: vec![],
            survival_upper: vec![],
            n_iter: 0,
            converged: true,
        });
    }

    let m = all_points.len();
    let mut p = vec![1.0 / m as f64; m];

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let p_old = p.clone();

        let mut p_new = vec![0.0; m];
        let mut weights = vec![0.0; m];

        for i in 0..n {
            let mut sum_p = 0.0;
            let mut contributing_intervals: Vec<usize> = Vec::new();

            for (j, &t) in all_points.iter().enumerate() {
                if t >= left[i] && (right[i] == f64::INFINITY || t <= right[i]) {
                    sum_p += p[j];
                    contributing_intervals.push(j);
                }
            }

            if sum_p > 0.0 {
                for &j in &contributing_intervals {
                    let w = p[j] / sum_p;
                    p_new[j] += w;
                    weights[j] += 1.0;
                }
            }
        }

        let total: f64 = p_new.iter().sum();
        if total > 0.0 {
            for j in 0..m {
                p[j] = p_new[j] / total;
            }
        }

        let max_diff: f64 = p
            .iter()
            .zip(p_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        if max_diff < tol {
            converged = true;
            break;
        }
    }

    let mut survival = Vec::with_capacity(m);
    let mut cum_prob = 0.0;
    for &prob in &p {
        cum_prob += prob;
        survival.push((1.0 - cum_prob).clamp(0.0, 1.0));
    }

    let se: Vec<f64> = p
        .iter()
        .map(|&prob| (prob * (1.0 - prob) / n as f64).sqrt())
        .collect();

    let z = 1.96;
    let survival_lower: Vec<f64> = survival
        .iter()
        .zip(se.iter())
        .map(|(&s, &se)| (s - z * se).max(0.0))
        .collect();

    let survival_upper: Vec<f64> = survival
        .iter()
        .zip(se.iter())
        .map(|(&s, &se)| (s + z * se).min(1.0))
        .collect();

    Ok(TurnbullResult {
        time_points: all_points,
        survival,
        survival_lower,
        survival_upper,
        n_iter,
        converged,
    })
}

#[pyfunction]
pub fn npmle_interval(
    left: Vec<f64>,
    right: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    turnbull_estimator(left, right, 1000, 1e-6).map(|result| (result.time_points, result.survival))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_cdf() {
        assert!((weibull_cdf(0.0, 1.0, 1.0) - 0.0).abs() < 1e-10);
        let cdf_5 = weibull_cdf(5.0, 3.0, 2.0);
        assert!(cdf_5 > 0.0 && cdf_5 < 1.0);
    }

    #[test]
    fn test_turnbull_basic() {
        let left = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let right = vec![2.0, 3.0, 5.0, 4.0, f64::INFINITY];

        let result = turnbull_estimator(left, right, 100, 1e-4).unwrap();
        assert!(result.time_points.len() > 0);
        assert!(result.survival.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_interval_regression_basic() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![2.0, 3.0, 5.0, 6.0];
        let censor_type = vec![3, 3, 3, 3];
        let x = vec![1.0, 0.5, 0.0, 1.0];

        let result = interval_censored_regression(
            left,
            right,
            censor_type,
            x,
            4,
            1,
            &IntervalDistribution::Weibull,
            100,
            1e-4,
        )
        .unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert!(result.scale > 0.0);
        assert!(result.shape > 0.0);
    }
}
