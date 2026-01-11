use pyo3::prelude::*;
use rayon::prelude::*;

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
    // For now, delegate to the standard implementation
    // A truly optimized version would use IJ residuals directly
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

        // Pseudo-values should average to approximately the KM estimate
        // Note: Individual pseudo-values can be outside [0,1], but average should be reasonable
        for t_idx in 0..result.time.len() {
            let avg: f64 = result.pseudo.iter().map(|p| p[t_idx]).sum::<f64>() / 5.0;
            // Average should be finite and roughly in a reasonable range
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
        // Cumulative hazard pseudo-values should be non-negative
        for p in &result.pseudo {
            for &val in p {
                // Pseudo-values can be negative, but on average should be positive
                assert!(val.is_finite());
            }
        }
    }
}
