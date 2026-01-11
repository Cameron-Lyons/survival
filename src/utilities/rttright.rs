use pyo3::prelude::*;

/// Result of redistribute-to-the-right weight calculation
#[derive(Debug, Clone)]
#[pyclass]
pub struct RttrightResult {
    /// Redistributed weights for each observation
    #[pyo3(get)]
    pub weights: Vec<f64>,
    /// Original time values (sorted)
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Original status values (sorted)
    #[pyo3(get)]
    pub status: Vec<i32>,
    /// Sort order indices
    #[pyo3(get)]
    pub order: Vec<usize>,
}

/// Compute redistribute-to-the-right weights for censored data.
///
/// This implements the IPCW (Inverse Probability of Censoring Weighting)
/// approach where the weight of each censored observation is redistributed
/// to observations with longer survival times.
///
/// The Kaplan-Meier estimator can be derived from this redistribution.
///
/// # Arguments
/// * `time` - Survival/censoring times
/// * `status` - Event indicator (1=event, 0=censored)
/// * `weights` - Optional initial weights (default: 1.0 for all)
///
/// # Returns
/// * `RttrightResult` containing redistributed weights
#[pyfunction]
#[pyo3(signature = (time, status, weights=None))]
pub fn rttright(
    time: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
) -> PyResult<RttrightResult> {
    let n = time.len();

    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let init_weights = weights.unwrap_or_else(|| vec![1.0; n]);
    if init_weights.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must have same length as time",
        ));
    }

    if n == 0 {
        return Ok(RttrightResult {
            weights: vec![],
            time: vec![],
            status: vec![],
            order: vec![],
        });
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
    let sorted_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
    let sorted_weights: Vec<f64> = indices.iter().map(|&i| init_weights[i]).collect();

    let mut rtt_weights = vec![0.0; n];
    let mut cumulative_censored_weight = 0.0;

    for i in 0..n {
        if i > 0 && sorted_status[i - 1] == 0 {
            cumulative_censored_weight += sorted_weights[i - 1];
        }

        if sorted_status[i] == 1 {
            let base_weight = sorted_weights[i];
            let n_at_risk = (n - i) as f64;
            let redistributed = if n_at_risk > 0.0 {
                cumulative_censored_weight / n_at_risk
            } else {
                0.0
            };

            rtt_weights[i] = base_weight + redistributed;
        } else {
            rtt_weights[i] = 0.0;
        }
    }

    let km_weights = compute_km_weights(&sorted_time, &sorted_status, &sorted_weights);

    Ok(RttrightResult {
        weights: km_weights,
        time: sorted_time,
        status: sorted_status,
        order: indices,
    })
}

/// Compute IPCW weights using Kaplan-Meier censoring distribution
fn compute_km_weights(time: &[f64], status: &[i32], init_weights: &[f64]) -> Vec<f64> {
    let n = time.len();
    if n == 0 {
        return vec![];
    }

    let mut g_t = vec![1.0; n];
    let mut n_at_risk = init_weights.iter().sum::<f64>();

    let mut prev_time = f64::NEG_INFINITY;
    let mut current_g = 1.0;

    for i in 0..n {
        if time[i] > prev_time && i > 0 {
            let mut n_censored = 0.0;
            let mut j = i - 1;
            while j < i && time[j] == prev_time {
                if status[j] == 0 {
                    n_censored += init_weights[j];
                }
                if j == 0 {
                    break;
                }
                j -= 1;
            }

            if n_at_risk > 0.0 && n_censored > 0.0 {
                current_g *= 1.0 - n_censored / n_at_risk;
            }
        }

        g_t[i] = current_g;
        n_at_risk -= init_weights[i];
        prev_time = time[i];
    }

    let mut weights = Vec::with_capacity(n);
    for i in 0..n {
        if status[i] == 1 {
            let g = if i > 0 { g_t[i - 1] } else { 1.0 };
            let w = if g > 1e-10 {
                init_weights[i] / g
            } else {
                init_weights[i]
            };
            weights.push(w);
        } else {
            weights.push(0.0);
        }
    }

    weights
}

/// Compute IPCW weights with stratification
#[pyfunction]
#[pyo3(signature = (time, status, strata, weights=None))]
pub fn rttright_stratified(
    time: Vec<f64>,
    status: Vec<i32>,
    strata: Vec<i32>,
    weights: Option<Vec<f64>>,
) -> PyResult<RttrightResult> {
    let n = time.len();

    if status.len() != n || strata.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and strata must have same length",
        ));
    }

    let init_weights = weights.unwrap_or_else(|| vec![1.0; n]);

    let mut strata_indices: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &s) in strata.iter().enumerate() {
        strata_indices.entry(s).or_default().push(i);
    }

    let mut final_weights = vec![0.0; n];
    let mut final_order = vec![0; n];

    let mut offset = 0;
    for indices in strata_indices.values() {
        let strata_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let strata_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();
        let strata_weights: Vec<f64> = indices.iter().map(|&i| init_weights[i]).collect();

        let result = rttright(strata_time, strata_status, Some(strata_weights))?;

        for (j, &orig_idx) in indices.iter().enumerate() {
            final_weights[orig_idx] = result.weights[j];
            final_order[offset + j] = orig_idx;
        }
        offset += indices.len();
    }

    Ok(RttrightResult {
        weights: final_weights,
        time,
        status,
        order: final_order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rttright_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let result = rttright(time, status, None).unwrap();

        // Events should have positive weights
        assert!(result.weights[0] > 0.0);
        assert!(result.weights[2] > 0.0);
        assert!(result.weights[4] > 0.0);

        // Censored should have zero weight
        assert_eq!(result.weights[1], 0.0);
        assert_eq!(result.weights[3], 0.0);
    }

    #[test]
    fn test_rttright_all_events() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 1, 1];

        let result = rttright(time, status, None).unwrap();

        // All should have weight 1.0 (no redistribution needed)
        for w in &result.weights {
            assert!((*w - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rttright_empty() {
        let time: Vec<f64> = vec![];
        let status: Vec<i32> = vec![];

        let result = rttright(time, status, None).unwrap();
        assert!(result.weights.is_empty());
    }
}
