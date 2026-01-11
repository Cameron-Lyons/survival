use crate::utilities::statistical::{normal_cdf, normal_inverse_cdf};
use pyo3::prelude::*;

/// Result of Royston's D statistic calculation
#[derive(Debug, Clone)]
#[pyclass]
pub struct RoystonResult {
    /// Royston's D statistic (measure of prognostic separation)
    #[pyo3(get)]
    pub d: f64,
    /// Standard error of D
    #[pyo3(get)]
    pub se: f64,
    /// R-squared based on D (Royston-Sauerbrei)
    #[pyo3(get)]
    pub r_squared_d: f64,
    /// Kent-O'Quigley R-squared
    #[pyo3(get)]
    pub r_squared_ko: f64,
    /// Z-score
    #[pyo3(get)]
    pub z: f64,
    /// P-value (two-sided)
    #[pyo3(get)]
    pub p_value: f64,
    /// Number of events
    #[pyo3(get)]
    pub n_events: usize,
}

/// Compute Royston's D statistic for a Cox model.
///
/// The D statistic measures the prognostic separation achieved by a Cox model.
/// It is based on the linear predictor (prognostic index) and provides a measure
/// of discrimination that is interpretable in terms of hazard ratios.
///
/// D can be interpreted as: subjects with prognostic index one SD above the mean
/// have exp(D) times the hazard of subjects with prognostic index one SD below
/// the mean.
///
/// # Arguments
/// * `linear_predictor` - Linear predictor values (X * beta) from Cox model
/// * `time` - Survival/censoring times
/// * `status` - Event indicator (1=event, 0=censored)
///
/// # Returns
/// * `RoystonResult` with D statistic and related measures
///
/// # References
/// Royston P, Sauerbrei W. (2004) A new measure of prognostic separation in
/// survival data. Statistics in Medicine 23:723-748.
#[pyfunction]
pub fn royston(
    linear_predictor: Vec<f64>,
    time: Vec<f64>,
    status: Vec<i32>,
) -> PyResult<RoystonResult> {
    let n = linear_predictor.len();

    if time.len() != n || status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "linear_predictor, time, and status must have same length",
        ));
    }

    let n_events = status.iter().filter(|&&s| s == 1).count();

    if n_events < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least 2 events required",
        ));
    }

    let event_lp: Vec<f64> = linear_predictor
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 1)
        .map(|(lp, _)| *lp)
        .collect();

    let ranks = compute_ranks(&event_lp);

    let normal_scores: Vec<f64> = ranks
        .iter()
        .map(|&r| {
            let p = (r - 0.375) / (n_events as f64 + 0.25);
            normal_inverse_cdf(p)
        })
        .collect();

    let lp_mean: f64 = event_lp.iter().sum::<f64>() / n_events as f64;
    let ns_mean: f64 = normal_scores.iter().sum::<f64>() / n_events as f64;

    let mut lp_var = 0.0;
    let mut ns_var = 0.0;
    let mut covar = 0.0;

    for ((lp, ns), _) in event_lp.iter().zip(normal_scores.iter()).zip(ranks.iter()) {
        let lp_dev = lp - lp_mean;
        let ns_dev = ns - ns_mean;
        lp_var += lp_dev * lp_dev;
        ns_var += ns_dev * ns_dev;
        covar += lp_dev * ns_dev;
    }

    lp_var /= (n_events - 1) as f64;
    ns_var /= (n_events - 1) as f64;
    covar /= (n_events - 1) as f64;

    let lp_sd = lp_var.sqrt();
    let ns_sd = ns_var.sqrt();

    let correlation = if lp_sd > 0.0 && ns_sd > 0.0 {
        covar / (lp_sd * ns_sd)
    } else {
        0.0
    };

    let kappa = (8.0 / std::f64::consts::PI).sqrt();

    let d = kappa * correlation;

    let se = kappa * ((1.0 - correlation.powi(2)) / (n_events - 2).max(1) as f64).sqrt();

    let z = if se > 0.0 { d / se } else { 0.0 };
    let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

    let d_sq = d.powi(2);
    let r_squared_d = d_sq / (d_sq + std::f64::consts::PI.powi(2) / 6.0);

    let r_squared_ko = 1.0 - (-d_sq * std::f64::consts::PI / 6.0).exp();

    Ok(RoystonResult {
        d,
        se,
        r_squared_d,
        r_squared_ko,
        z,
        p_value,
        n_events,
    })
}

/// Compute Royston's D from model coefficients and data.
///
/// This is a convenience function that computes the linear predictor
/// and then calls royston().
#[pyfunction]
pub fn royston_from_model(
    x: Vec<f64>,
    coef: Vec<f64>,
    n_obs: usize,
    time: Vec<f64>,
    status: Vec<i32>,
) -> PyResult<RoystonResult> {
    let n_vars = coef.len();
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let mut linear_predictor = vec![0.0; n_obs];
    for i in 0..n_obs {
        for j in 0..n_vars {
            linear_predictor[i] += x[i * n_vars + j] * coef[j];
        }
    }

    royston(linear_predictor, time, status)
}

fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];

    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }

        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_royston_basic() {
        let lp = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.5, 0.9, -0.2];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 0, 1, 1, 1, 0];

        let result = royston(lp, time, status).unwrap();

        // D should be positive if higher lp means higher hazard
        assert!(result.d.is_finite());
        assert!(result.se > 0.0);
        assert!(result.r_squared_d >= 0.0 && result.r_squared_d <= 1.0);
        assert!(result.r_squared_ko >= 0.0 && result.r_squared_ko <= 1.0);
    }

    #[test]
    fn test_royston_perfect_separation() {
        // Perfect separation: all high lp subjects have events first
        let lp = vec![1.0, 0.9, 0.8, 0.7, -0.7, -0.8, -0.9, -1.0];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];

        let result = royston(lp, time, status).unwrap();

        // Should have high D value
        assert!(result.d > 1.0);
    }

    #[test]
    fn test_royston_no_separation() {
        // No separation: lp is constant
        let lp = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];

        let result = royston(lp, time, status).unwrap();

        // D should be close to 0
        assert!(result.d.abs() < 0.1);
    }

    #[test]
    fn test_normal_inverse_cdf() {
        // Test known values
        assert!((normal_inverse_cdf(0.5) - 0.0).abs() < 0.001);
        assert!((normal_inverse_cdf(0.975) - 1.96).abs() < 0.01);
        assert!((normal_inverse_cdf(0.025) - (-1.96)).abs() < 0.01);
    }
}
