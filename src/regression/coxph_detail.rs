use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoxphDetailRow {
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub n_risk: usize,
    #[pyo3(get)]
    pub n_event: usize,
    #[pyo3(get)]
    pub n_censor: usize,
    #[pyo3(get)]
    pub hazard: f64,
    #[pyo3(get)]
    pub cumhaz: f64,
    #[pyo3(get)]
    pub score: Vec<f64>,
    #[pyo3(get)]
    pub schoenfeld: Option<Vec<f64>>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub imat: Vec<Vec<f64>>,
}

#[pymethods]
impl CoxphDetailRow {
    fn __repr__(&self) -> String {
        format!(
            "CoxphDetailRow(time={:.4}, n_risk={}, n_event={}, hazard={:.6})",
            self.time, self.n_risk, self.n_event, self.hazard
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CoxphDetail {
    #[pyo3(get)]
    pub rows: Vec<CoxphDetailRow>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_observations: usize,
    #[pyo3(get)]
    pub n_covariates: usize,
}

#[pymethods]
impl CoxphDetail {
    fn __repr__(&self) -> String {
        format!(
            "CoxphDetail(n_events={}, n_obs={}, n_times={})",
            self.n_events,
            self.n_observations,
            self.rows.len()
        )
    }

    pub fn times(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.time).collect()
    }

    pub fn hazards(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.hazard).collect()
    }

    pub fn cumulative_hazards(&self) -> Vec<f64> {
        self.rows.iter().map(|r| r.cumhaz).collect()
    }

    pub fn n_risk_at_times(&self) -> Vec<usize> {
        self.rows.iter().map(|r| r.n_risk).collect()
    }

    pub fn schoenfeld_residuals(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .filter_map(|r| r.schoenfeld.clone())
            .collect()
    }
}

pub fn compute_coxph_detail(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    coefficients: &[f64],
    weights: Option<&[f64]>,
) -> CoxphDetail {
    let n = time.len();
    let nvar = coefficients.len();

    if n == 0 || nvar == 0 {
        return CoxphDetail {
            rows: vec![],
            n_events: 0,
            n_observations: 0,
            n_covariates: nvar,
        };
    }

    let risk_scores: Vec<f64> = covariates
        .iter()
        .map(|cov| {
            let mut lp = 0.0;
            for (j, &c) in cov.iter().enumerate() {
                if j < nvar {
                    lp += c * coefficients[j];
                }
            }
            lp.exp()
        })
        .collect();

    let wts: Vec<f64> = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; n]);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times: Vec<f64> = Vec::new();
    let mut event_counts: Vec<usize> = Vec::new();
    let mut censor_counts: Vec<usize> = Vec::new();
    let mut risk_set_sizes: Vec<usize> = Vec::new();
    let mut risk_sums: Vec<f64> = Vec::new();
    let mut weighted_means: Vec<Vec<f64>> = Vec::new();
    let mut event_covariates: Vec<Vec<f64>> = Vec::new();

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut n_event = 0;
        let mut n_censor = 0;
        let mut event_cov_sum = vec![0.0; nvar];
        let start_i = i;

        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 1 {
                n_event += 1;
                for (k, &c) in covariates[indices[i]].iter().enumerate() {
                    if k < nvar {
                        event_cov_sum[k] += c;
                    }
                }
            } else {
                n_censor += 1;
            }
            i += 1;
        }

        if n_event > 0 {
            let risk_set_size = n - start_i;
            let mut risk_sum = 0.0;
            let mut weighted_mean = vec![0.0; nvar];

            for &idx in indices.iter().take(n).skip(start_i) {
                let w = wts[idx] * risk_scores[idx];
                risk_sum += w;
                for (k, &c) in covariates[idx].iter().enumerate() {
                    if k < nvar {
                        weighted_mean[k] += w * c;
                    }
                }
            }

            if risk_sum > 0.0 {
                for wm in weighted_mean.iter_mut().take(nvar) {
                    *wm /= risk_sum;
                }
            }

            unique_times.push(current_time);
            event_counts.push(n_event);
            censor_counts.push(n_censor);
            risk_set_sizes.push(risk_set_size);
            risk_sums.push(risk_sum);
            weighted_means.push(weighted_mean);
            event_covariates.push(event_cov_sum);
        }
    }

    let n_unique = unique_times.len();
    let mut rows = Vec::with_capacity(n_unique);
    let mut cumhaz = 0.0;

    for t in 0..n_unique {
        let hazard = if risk_sums[t] > 0.0 {
            event_counts[t] as f64 / risk_sums[t]
        } else {
            0.0
        };
        cumhaz += hazard;

        let schoenfeld = if event_counts[t] > 0 {
            let mut scho = vec![0.0; nvar];
            for k in 0..nvar {
                scho[k] = event_covariates[t][k] - event_counts[t] as f64 * weighted_means[t][k];
            }
            Some(scho)
        } else {
            None
        };

        let score = vec![0.0; nvar];
        let imat = vec![vec![0.0; nvar]; nvar];

        rows.push(CoxphDetailRow {
            time: unique_times[t],
            n_risk: risk_set_sizes[t],
            n_event: event_counts[t],
            n_censor: censor_counts[t],
            hazard,
            cumhaz,
            score,
            schoenfeld,
            means: weighted_means[t].clone(),
            imat,
        });
    }

    let total_events = event_counts.iter().sum();

    CoxphDetail {
        rows,
        n_events: total_events,
        n_observations: n,
        n_covariates: nvar,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, coefficients, weights=None))]
pub fn coxph_detail(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<CoxphDetail> {
    let n = time.len();
    if status.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and covariates must have the same length",
        ));
    }

    Ok(compute_coxph_detail(
        &time,
        &status,
        &covariates,
        &coefficients,
        weights.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coxph_detail() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let covariates = vec![vec![1.0], vec![2.0], vec![1.5], vec![2.5], vec![3.0]];
        let coefficients = vec![0.5];

        let detail = compute_coxph_detail(&time, &status, &covariates, &coefficients, None);

        assert_eq!(detail.n_events, 3);
        assert_eq!(detail.n_observations, 5);
        assert_eq!(detail.rows.len(), 3);
    }
}
