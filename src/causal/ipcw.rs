use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IPCWInput {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub x_censoring: Vec<f64>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pymethods]
impl IPCWInput {
    #[new]
    #[pyo3(signature = (time, status, x_censoring, n_obs, n_vars))]
    pub fn new(
        time: Vec<f64>,
        status: Vec<i32>,
        x_censoring: Vec<f64>,
        n_obs: usize,
        n_vars: usize,
    ) -> PyResult<Self> {
        let input = Self {
            time,
            status,
            x_censoring,
            n_obs,
            n_vars,
        };
        input.validate()?;
        Ok(input)
    }
}

impl IPCWInput {
    fn validate(&self) -> PyResult<()> {
        if self.time.len() != self.n_obs || self.status.len() != self.n_obs {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "time and status must have length n_obs",
            ));
        }
        if self.x_censoring.len() != self.n_obs * self.n_vars {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "x_censoring dimensions mismatch",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IPCWConfig {
    #[pyo3(get, set)]
    pub stabilized: bool,
    #[pyo3(get, set)]
    pub trim: Option<f64>,
}

#[pymethods]
impl IPCWConfig {
    #[new]
    #[pyo3(signature = (stabilized=true, trim=None))]
    pub fn new(stabilized: bool, trim: Option<f64>) -> Self {
        Self { stabilized, trim }
    }
}

impl Default for IPCWConfig {
    fn default() -> Self {
        Self {
            stabilized: true,
            trim: Some(0.01),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IPCWTreatmentInput {
    pub ipcw: IPCWInput,
    pub treatment: Vec<i32>,
    pub outcome: Vec<f64>,
    pub tau: Option<f64>,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IPCWResult {
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub censoring_probs: Vec<f64>,
    #[pyo3(get)]
    pub treatment_effect: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub n_effective: f64,
}

fn fit_logistic_model(
    x: &[f64],
    y: &[i32],
    at_risk_indices: &[usize],
    p: usize,
    max_iter: usize,
) -> Vec<f64> {
    let mut beta = vec![0.0; p];

    for _ in 0..max_iter {
        let mut gradient = vec![0.0; p];
        let mut hessian_diag = vec![0.0; p];

        for (idx, &obs) in at_risk_indices.iter().enumerate() {
            let base = obs * p;
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[base + j] * beta[j];
            }
            let prob = 1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp());
            let residual = y[idx] as f64 - prob;

            for j in 0..p {
                let xij = x[base + j];
                gradient[j] += xij * residual;
                hessian_diag[j] += xij * xij * prob * (1.0 - prob);
            }
        }

        let mut max_change: f64 = 0.0;
        for j in 0..p {
            if hessian_diag[j].abs() > crate::constants::DIVISION_FLOOR {
                let update = gradient[j] / hessian_diag[j];
                beta[j] += update;
                max_change = max_change.max(update.abs());
            }
        }

        if max_change < 1e-6 {
            break;
        }
    }

    beta
}

fn predict_probs(x: &[f64], beta: &[f64], at_risk_indices: &[usize], p: usize) -> Vec<f64> {
    at_risk_indices
        .iter()
        .map(|&obs| {
            let base = obs * p;
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[base + j] * beta[j];
            }
            1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp())
        })
        .collect()
}

fn compute_ipcw_weights_typed(input: &IPCWInput, config: &IPCWConfig) -> PyResult<IPCWResult> {
    input.validate()?;

    let time = &input.time;
    let status = &input.status;
    let x_censoring = &input.x_censoring;
    let n_obs = input.n_obs;
    let n_vars = input.n_vars;

    let censored: Vec<i32> = status.iter().map(|&s| if s == 0 { 1 } else { 0 }).collect();

    let mut unique_times: Vec<f64> = time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();

    let mut sorted_indices: Vec<usize> = (0..n_obs).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut censoring_probs = vec![1.0; n_obs];
    let mut risk_start = 0usize;

    for &t in &unique_times {
        while risk_start < n_obs && time[sorted_indices[risk_start]] < t {
            risk_start += 1;
        }
        let at_risk = &sorted_indices[risk_start..];

        if at_risk.is_empty() {
            continue;
        }

        let mut y_risk = Vec::with_capacity(at_risk.len());
        let mut has_events = false;
        for &i in at_risk {
            let yi = if (time[i] - t).abs() < crate::constants::DIVISION_FLOOR && censored[i] == 1 {
                1
            } else {
                0
            };
            has_events |= yi == 1;
            y_risk.push(yi);
        }

        if !has_events {
            continue;
        }

        let beta = fit_logistic_model(x_censoring, &y_risk, at_risk, n_vars, 50);
        let censor_probs_t = predict_probs(x_censoring, &beta, at_risk, n_vars);

        for (idx, &i) in at_risk.iter().enumerate() {
            censoring_probs[i] *= 1.0 - censor_probs_t[idx];
        }
    }

    let trim_threshold = config.trim.unwrap_or(0.01);
    for prob in &mut censoring_probs {
        *prob = prob.max(trim_threshold);
    }

    let mut weights: Vec<f64> = censoring_probs.iter().map(|&p| 1.0 / p).collect();

    if config.stabilized {
        let km_survival = compute_km_censoring(time, status, n_obs);
        for i in 0..n_obs {
            weights[i] *= km_survival[i];
        }
    }

    let n_effective = weights.iter().map(|&w| w.powi(2)).sum::<f64>().recip()
        * weights.iter().sum::<f64>().powi(2);

    Ok(IPCWResult {
        weights,
        censoring_probs,
        treatment_effect: 0.0,
        std_error: 0.0,
        ci_lower: 0.0,
        ci_upper: 0.0,
        n_effective,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, x_censoring, n_obs, n_vars, stabilized=true, trim=None))]
pub fn compute_ipcw_weights(
    time: Vec<f64>,
    status: Vec<i32>,
    x_censoring: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    stabilized: bool,
    trim: Option<f64>,
) -> PyResult<IPCWResult> {
    let input = IPCWInput::new(time, status, x_censoring, n_obs, n_vars)?;
    let config = IPCWConfig::new(stabilized, trim);
    compute_ipcw_weights_typed(&input, &config)
}

fn compute_km_censoring(time: &[f64], status: &[i32], n: usize) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut km_surv = vec![1.0; n];
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;

        let start_i = i;
        while i < n && (time[indices[i]] - current_time).abs() < crate::constants::TIME_EPSILON {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        for j in start_i..i {
            km_surv[indices[j]] = cum_surv;
        }

        at_risk -= i - start_i;
    }

    km_surv
}

fn ipcw_treatment_effect_typed(input: IPCWTreatmentInput) -> PyResult<IPCWResult> {
    input.ipcw.validate()?;
    if input.treatment.len() != input.ipcw.n_obs || input.outcome.len() != input.ipcw.n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "treatment and outcome must have length n_obs",
        ));
    }

    let ipcw = compute_ipcw_weights_typed(&input.ipcw, &IPCWConfig::default())?;

    let time = &input.ipcw.time;
    let status = &input.ipcw.status;
    let n_obs = input.ipcw.n_obs;
    let tau_val = input
        .tau
        .unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;
    let mut n_treated = 0.0;
    let mut n_control = 0.0;

    for i in 0..n_obs {
        let contrib = if (time[i] <= tau_val && status[i] == 1) || time[i] > tau_val {
            input.outcome[i] * ipcw.weights[i]
        } else {
            continue;
        };

        if input.treatment[i] == 1 {
            sum_treated += contrib;
            n_treated += ipcw.weights[i];
        } else {
            sum_control += contrib;
            n_control += ipcw.weights[i];
        }
    }

    let mean_treated = if n_treated > 0.0 {
        sum_treated / n_treated
    } else {
        0.0
    };
    let mean_control = if n_control > 0.0 {
        sum_control / n_control
    } else {
        0.0
    };
    let treatment_effect = mean_treated - mean_control;

    let mut var_sum = 0.0;
    for i in 0..n_obs {
        if time[i] <= tau_val || status[i] == 1 {
            let resid = if input.treatment[i] == 1 {
                input.outcome[i] - mean_treated
            } else {
                input.outcome[i] - mean_control
            };
            var_sum += ipcw.weights[i] * ipcw.weights[i] * resid * resid;
        }
    }

    let std_error = (var_sum / (n_treated + n_control).powi(2)).sqrt();
    let z = 1.96;
    let ci_lower = treatment_effect - z * std_error;
    let ci_upper = treatment_effect + z * std_error;

    Ok(IPCWResult {
        weights: ipcw.weights,
        censoring_probs: ipcw.censoring_probs,
        treatment_effect,
        std_error,
        ci_lower,
        ci_upper,
        n_effective: ipcw.n_effective,
    })
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (time, status, treatment, outcome, x_confounders, n_obs, n_vars, tau=None))]
pub fn ipcw_treatment_effect(
    time: Vec<f64>,
    status: Vec<i32>,
    treatment: Vec<i32>,
    outcome: Vec<f64>,
    x_confounders: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    tau: Option<f64>,
) -> PyResult<IPCWResult> {
    let input = IPCWTreatmentInput {
        ipcw: IPCWInput::new(time, status, x_confounders, n_obs, n_vars)?,
        treatment,
        outcome,
        tau,
    };
    ipcw_treatment_effect_typed(input)
}

#[pyfunction]
#[pyo3(signature = (time, status, x_censoring, n_obs, n_vars, time_points))]
pub fn ipcw_kaplan_meier(
    time: Vec<f64>,
    status: Vec<i32>,
    x_censoring: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time_points: Vec<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let ipcw_input = IPCWInput::new(time.clone(), status.clone(), x_censoring, n_obs, n_vars)?;
    let ipcw = compute_ipcw_weights_typed(&ipcw_input, &IPCWConfig::default())?;

    let mut survival = Vec::with_capacity(time_points.len());
    let mut variance = Vec::with_capacity(time_points.len());

    for &t in &time_points {
        let mut numer = 0.0;
        let mut denom = 0.0;
        let mut var_sum = 0.0;

        for i in 0..n_obs {
            let w = ipcw.weights[i];

            denom += w;

            if time[i] <= t && status[i] == 1 {
                numer += w;
                var_sum += w * w;
            }
        }

        let surv = if denom > 0.0 {
            1.0 - numer / denom
        } else {
            1.0
        };
        let var = if denom > 0.0 {
            var_sum / (denom * denom)
        } else {
            0.0
        };

        survival.push(surv);
        variance.push(var);
    }

    let ci_width: Vec<f64> = variance.iter().map(|&v| 1.96 * v.sqrt()).collect();

    Ok((time_points, survival, ci_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipcw_weights() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5];

        let result = compute_ipcw_weights(time, status, x, 5, 1, true, Some(0.01)).unwrap();
        assert_eq!(result.weights.len(), 5);
        assert!(result.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_km_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![1, 0, 1, 0];
        let km = compute_km_censoring(&time, &status, 4);
        assert_eq!(km.len(), 4);
        assert!(km.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_ipcw_weights_dimension_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0];
        let x = vec![1.0, 0.5, 0.0];
        let result = compute_ipcw_weights(time, status, x, 3, 1, true, Some(0.01));
        assert!(result.is_err());
    }

    #[test]
    fn test_ipcw_weights_x_dimension_mismatch() {
        let time = vec![1.0, 2.0, 3.0];
        let status = vec![1, 0, 1];
        let x = vec![1.0, 0.5];
        let result = compute_ipcw_weights(time, status, x, 3, 1, true, Some(0.01));
        assert!(result.is_err());
    }

    #[test]
    fn test_ipcw_weights_all_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5];

        let result = compute_ipcw_weights(time, status, x, 5, 1, true, Some(0.01)).unwrap();
        assert_eq!(result.weights.len(), 5);
        assert!(result.weights.iter().all(|&w| w > 0.0));
        assert!(result.n_effective > 0.0);
    }

    #[test]
    fn test_ipcw_weights_unstabilized() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5];

        let stabilized = compute_ipcw_weights(
            time.clone(),
            status.clone(),
            x.clone(),
            5,
            1,
            true,
            Some(0.01),
        )
        .unwrap();
        let unstabilized = compute_ipcw_weights(time, status, x, 5, 1, false, Some(0.01)).unwrap();

        assert_eq!(stabilized.weights.len(), unstabilized.weights.len());
    }

    #[test]
    fn test_ipcw_treatment_effect_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let treatment = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let outcome = vec![2.0, 1.0, 2.5, 1.5, 3.0, 0.5, 2.2, 1.3, 2.8, 0.8];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4];

        let result = ipcw_treatment_effect_typed(IPCWTreatmentInput {
            ipcw: IPCWInput::new(time, status, x, 10, 1).unwrap(),
            treatment,
            outcome,
            tau: None,
        })
        .unwrap();

        assert_eq!(result.weights.len(), 10);
        assert_eq!(result.censoring_probs.len(), 10);
        assert!(result.treatment_effect.is_finite());
        assert!(result.std_error >= 0.0);
        assert!(result.ci_lower <= result.ci_upper);
    }

    #[test]
    fn test_ipcw_kaplan_meier_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let x = vec![1.0, 0.5, 0.0, 1.0, 0.5, 0.3, 0.7, 0.2];
        let time_points = vec![2.0, 4.0, 6.0, 8.0];

        let (tp, survival, ci_width) =
            ipcw_kaplan_meier(time, status, x, 8, 1, time_points).unwrap();

        assert_eq!(tp.len(), 4);
        assert_eq!(survival.len(), 4);
        assert_eq!(ci_width.len(), 4);
        assert!(survival.iter().all(|&s| (0.0..=1.0).contains(&s)));
        assert!(ci_width.iter().all(|&c| c >= 0.0));
    }

    #[test]
    fn test_km_censoring_all_censored() {
        let time = vec![1.0, 2.0, 3.0, 4.0];
        let status = vec![0, 0, 0, 0];
        let km = compute_km_censoring(&time, &status, 4);
        assert_eq!(km.len(), 4);
        assert!(km.iter().all(|&s| s <= 1.0));
    }
}
