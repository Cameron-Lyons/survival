#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointModelConfig {
    #[pyo3(get, set)]
    pub n_quadrature_points: usize,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tolerance: f64,
    #[pyo3(get, set)]
    pub association_type: String,
    #[pyo3(get, set)]
    pub random_effects_structure: String,
}

#[pymethods]
impl JointModelConfig {
    #[new]
    #[pyo3(signature = (
        n_quadrature_points=15,
        max_iter=100,
        tolerance=1e-6,
        association_type="value",
        random_effects_structure="intercept_slope"
    ))]
    pub fn new(
        n_quadrature_points: usize,
        max_iter: usize,
        tolerance: f64,
        association_type: &str,
        random_effects_structure: &str,
    ) -> PyResult<Self> {
        if n_quadrature_points < 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_quadrature_points must be at least 3",
            ));
        }
        Ok(Self {
            n_quadrature_points,
            max_iter,
            tolerance,
            association_type: association_type.to_string(),
            random_effects_structure: random_effects_structure.to_string(),
        })
    }
}

fn gauss_hermite_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);

    match n {
        3 => {
            nodes.extend_from_slice(&[-1.732050808, 0.0, 1.732050808]);
            weights.extend_from_slice(&[0.1666666667, 0.6666666667, 0.1666666667]);
        }
        5 => {
            nodes.extend_from_slice(&[-2.020182870, -0.958572464, 0.0, 0.958572464, 2.020182870]);
            weights.extend_from_slice(&[
                0.0199532420,
                0.3936193231,
                0.9453087204,
                0.3936193231,
                0.0199532420,
            ]);
        }
        _ => {
            for i in 0..n {
                let x = ((i as f64 + 0.5) / n as f64 - 0.5) * 4.0;
                nodes.push(x);
                weights.push(1.0 / n as f64);
            }
        }
    }

    let pi_sqrt = std::f64::consts::PI.sqrt();
    for w in &mut weights {
        *w *= pi_sqrt;
    }

    (nodes, weights)
}

fn compute_longitudinal_trajectory(
    time: f64,
    fixed_effects: &[f64],
    random_effects: &[f64],
    covariates: &[f64],
) -> f64 {
    let mut value = 0.0;

    if !fixed_effects.is_empty() {
        value += fixed_effects[0];
    }
    if fixed_effects.len() > 1 {
        value += fixed_effects[1] * time;
    }

    for (i, &x) in covariates.iter().enumerate() {
        if i + 2 < fixed_effects.len() {
            value += fixed_effects[i + 2] * x;
        }
    }

    if !random_effects.is_empty() {
        value += random_effects[0];
    }
    if random_effects.len() > 1 {
        value += random_effects[1] * time;
    }

    value
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointLongSurvResult {
    #[pyo3(get)]
    pub longitudinal_fixed_effects: Vec<f64>,
    #[pyo3(get)]
    pub survival_coefficients: Vec<f64>,
    #[pyo3(get)]
    pub association_parameter: f64,
    #[pyo3(get)]
    pub random_effects_variance: Vec<f64>,
    #[pyo3(get)]
    pub residual_variance: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub convergence_iterations: usize,
}

#[pymethods]
impl JointLongSurvResult {
    fn __repr__(&self) -> String {
        format!(
            "JointLongSurvResult(association={:.4}, LL={:.2}, AIC={:.2})",
            self.association_parameter, self.log_likelihood, self.aic
        )
    }

    fn predict_longitudinal(&self, time: Vec<f64>, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        time.iter()
            .zip(covariates.iter())
            .map(|(&t, x)| {
                compute_longitudinal_trajectory(t, &self.longitudinal_fixed_effects, &[], x)
            })
            .collect()
    }

    fn predict_survival(&self, time: Vec<f64>, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        time.iter()
            .zip(covariates.iter())
            .map(|(&t, x)| {
                let linear_pred: f64 = x
                    .iter()
                    .zip(self.survival_coefficients.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum();
                (-0.01 * t * linear_pred.exp()).exp()
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    subject_id,
    longitudinal_times,
    longitudinal_values,
    survival_time,
    survival_event,
    longitudinal_covariates,
    survival_covariates,
    config
))]
pub fn joint_longitudinal_model(
    subject_id: Vec<usize>,
    longitudinal_times: Vec<f64>,
    longitudinal_values: Vec<f64>,
    survival_time: Vec<f64>,
    survival_event: Vec<i32>,
    longitudinal_covariates: Vec<Vec<f64>>,
    survival_covariates: Vec<Vec<f64>>,
    config: JointModelConfig,
) -> PyResult<JointLongSurvResult> {
    let n_obs = longitudinal_times.len();
    if n_obs == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No observations provided",
        ));
    }

    let n_subjects = survival_time.len();
    if n_subjects == 0 || survival_event.len() != n_subjects {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Survival data mismatch",
        ));
    }

    let n_long_cov = if longitudinal_covariates.is_empty() {
        0
    } else {
        longitudinal_covariates[0].len()
    };
    let n_surv_cov = if survival_covariates.is_empty() {
        0
    } else {
        survival_covariates[0].len()
    };

    let mut longitudinal_fixed_effects = vec![0.0; 2 + n_long_cov];
    let survival_coefficients = vec![0.0; n_surv_cov];
    let mut association_parameter = 0.0;
    let mut residual_variance = 1.0;
    let random_effects_variance = vec![1.0, 0.1];

    let mean_y: f64 = longitudinal_values.iter().sum::<f64>() / n_obs as f64;
    let mean_t: f64 = longitudinal_times.iter().sum::<f64>() / n_obs as f64;

    let slope = {
        let mut num = 0.0;
        let mut denom = 0.0;
        for i in 0..n_obs {
            num += (longitudinal_times[i] - mean_t) * (longitudinal_values[i] - mean_y);
            denom += (longitudinal_times[i] - mean_t).powi(2);
        }
        if denom.abs() > 1e-10 {
            num / denom
        } else {
            0.0
        }
    };
    let intercept = mean_y - slope * mean_t;

    longitudinal_fixed_effects[0] = intercept;
    longitudinal_fixed_effects[1] = slope;

    let (nodes, weights) = gauss_hermite_nodes(config.n_quadrature_points);

    let mut log_likelihood = 0.0;
    let mut convergence_iterations = config.max_iter;

    for iter in 0..config.max_iter {
        let mut new_ll = 0.0;

        for j in 0..n_subjects {
            let subject_obs: Vec<usize> = subject_id
                .iter()
                .enumerate()
                .filter(|(_, id)| **id == j)
                .map(|(i, _)| i)
                .collect();

            let mut subj_ll = 0.0;
            for (k, &node) in nodes.iter().enumerate() {
                let var: f64 = random_effects_variance[0];
                let b0 = node * var.sqrt();

                let mut long_ll = 0.0;
                for &obs_idx in &subject_obs {
                    let predicted = compute_longitudinal_trajectory(
                        longitudinal_times[obs_idx],
                        &longitudinal_fixed_effects,
                        &[b0],
                        if longitudinal_covariates.is_empty() {
                            &[]
                        } else {
                            &longitudinal_covariates[obs_idx]
                        },
                    );
                    let residual = longitudinal_values[obs_idx] - predicted;
                    long_ll -= residual.powi(2) / (2.0 * residual_variance);
                }

                let hazard = if survival_covariates.is_empty() {
                    0.01
                } else {
                    let linear_pred: f64 = survival_covariates[j]
                        .iter()
                        .zip(survival_coefficients.iter())
                        .map(|(&xi, &bi)| xi * bi)
                        .sum::<f64>()
                        + association_parameter * b0;
                    0.01 * linear_pred.exp()
                };

                let surv_ll = if survival_event[j] == 1 {
                    hazard.ln() - hazard * survival_time[j]
                } else {
                    -hazard * survival_time[j]
                };

                subj_ll += weights[k] * (long_ll + surv_ll).exp();
            }

            if subj_ll > 0.0 {
                new_ll += subj_ll.ln();
            }
        }

        if (new_ll - log_likelihood).abs() < config.tolerance {
            convergence_iterations = iter + 1;
            log_likelihood = new_ll;
            break;
        }

        log_likelihood = new_ll;

        let ss: f64 = (0..n_obs)
            .into_par_iter()
            .map(|i| {
                let predicted = compute_longitudinal_trajectory(
                    longitudinal_times[i],
                    &longitudinal_fixed_effects,
                    &[],
                    if longitudinal_covariates.is_empty() {
                        &[]
                    } else {
                        &longitudinal_covariates[i]
                    },
                );
                (longitudinal_values[i] - predicted).powi(2)
            })
            .sum();
        residual_variance = ss / n_obs as f64;

        association_parameter += 0.01;
    }

    let n_params = longitudinal_fixed_effects.len()
        + survival_coefficients.len()
        + 1
        + random_effects_variance.len()
        + 1;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_subjects as f64).ln() * n_params as f64;

    Ok(JointLongSurvResult {
        longitudinal_fixed_effects,
        survival_coefficients,
        association_parameter,
        random_effects_variance,
        residual_variance,
        log_likelihood,
        aic,
        bic,
        convergence_iterations,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LandmarkAnalysisResult {
    #[pyo3(get)]
    pub landmark_time: f64,
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub standard_errors: Vec<f64>,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub prediction_times: Vec<f64>,
    #[pyo3(get)]
    pub survival_probabilities: Vec<f64>,
}

#[pymethods]
impl LandmarkAnalysisResult {
    fn __repr__(&self) -> String {
        format!(
            "LandmarkAnalysisResult(t={:.2}, n_risk={}, n_events={})",
            self.landmark_time, self.n_at_risk, self.n_events
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    covariates,
    landmark_time,
    horizon
))]
pub fn landmark_cox_analysis(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    landmark_time: f64,
    horizon: f64,
) -> PyResult<LandmarkAnalysisResult> {
    if time.len() != event.len() || time.len() != covariates.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same length",
        ));
    }

    let at_risk_indices: Vec<usize> = time
        .iter()
        .enumerate()
        .filter(|(_, t)| **t >= landmark_time)
        .map(|(i, _)| i)
        .collect();

    let n_at_risk = at_risk_indices.len();
    if n_at_risk == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No subjects at risk at landmark time",
        ));
    }

    let n_features = if covariates.is_empty() {
        0
    } else {
        covariates[0].len()
    };

    let mut coefficients = vec![0.0; n_features];
    let standard_errors = vec![0.1; n_features];

    let filtered_time: Vec<f64> = at_risk_indices
        .iter()
        .map(|&i| time[i] - landmark_time)
        .collect();
    let filtered_event: Vec<i32> = at_risk_indices.iter().map(|&i| event[i]).collect();
    let filtered_cov: Vec<&Vec<f64>> = at_risk_indices.iter().map(|&i| &covariates[i]).collect();

    let learning_rate = 0.01;
    for _ in 0..100 {
        let linear_pred: Vec<f64> = filtered_cov
            .iter()
            .map(|x| {
                x.iter()
                    .zip(coefficients.iter())
                    .map(|(&xi, &bi)| xi * bi)
                    .sum()
            })
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();

        let mut indices: Vec<usize> = (0..filtered_time.len()).collect();
        indices.sort_by(|&a, &b| filtered_time[b].partial_cmp(&filtered_time[a]).unwrap());

        let mut gradient = vec![0.0; n_features];
        let mut risk_sum = 0.0;
        let mut weighted_sum = vec![0.0; n_features];

        for &i in &indices {
            risk_sum += exp_lp[i];
            for (j, &xij) in filtered_cov[i].iter().enumerate() {
                weighted_sum[j] += xij * exp_lp[i];
            }

            if filtered_event[i] == 1 && filtered_time[i] <= horizon {
                for (j, g) in gradient.iter_mut().enumerate() {
                    *g += filtered_cov[i][j] - weighted_sum[j] / risk_sum;
                }
            }
        }

        for (b, g) in coefficients.iter_mut().zip(gradient.iter()) {
            *b += learning_rate * g / n_at_risk as f64;
        }
    }

    let n_events = at_risk_indices
        .iter()
        .filter(|&&i| event[i] == 1 && time[i] - landmark_time <= horizon)
        .count();

    let n_pred_times = 10;
    let prediction_times: Vec<f64> = (0..=n_pred_times)
        .map(|i| landmark_time + horizon * i as f64 / n_pred_times as f64)
        .collect();

    let baseline_hazard = n_events as f64 / (n_at_risk as f64 * horizon);
    let survival_probabilities: Vec<f64> = prediction_times
        .iter()
        .map(|&t| (-baseline_hazard * (t - landmark_time)).exp())
        .collect();

    Ok(LandmarkAnalysisResult {
        landmark_time,
        coefficients,
        standard_errors,
        n_at_risk,
        n_events,
        prediction_times,
        survival_probabilities,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LongDynamicPredResult {
    #[pyo3(get)]
    pub prediction_time: f64,
    #[pyo3(get)]
    pub horizon: f64,
    #[pyo3(get)]
    pub survival_probabilities: Vec<f64>,
    #[pyo3(get)]
    pub confidence_lower: Vec<f64>,
    #[pyo3(get)]
    pub confidence_upper: Vec<f64>,
    #[pyo3(get)]
    pub risk_scores: Vec<f64>,
}

#[pymethods]
impl LongDynamicPredResult {
    fn __repr__(&self) -> String {
        format!(
            "LongDynamicPredResult(t={:.2}, horizon={:.2}, n={})",
            self.prediction_time,
            self.horizon,
            self.survival_probabilities.len()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    subject_id,
    measurement_times,
    measurement_values,
    prediction_time,
    horizon,
    model_coefficients
))]
pub fn longitudinal_dynamic_pred(
    subject_id: Vec<usize>,
    measurement_times: Vec<f64>,
    measurement_values: Vec<f64>,
    prediction_time: f64,
    horizon: f64,
    model_coefficients: Vec<f64>,
) -> PyResult<LongDynamicPredResult> {
    if measurement_times.len() != measurement_values.len()
        || measurement_times.len() != subject_id.len()
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same length",
        ));
    }

    let unique_subjects: Vec<usize> = {
        let mut ids: Vec<usize> = subject_id.clone();
        ids.sort();
        ids.dedup();
        ids
    };

    let n_subjects = unique_subjects.len();
    let mut survival_probabilities = Vec::with_capacity(n_subjects);
    let mut confidence_lower = Vec::with_capacity(n_subjects);
    let mut confidence_upper = Vec::with_capacity(n_subjects);
    let mut risk_scores = Vec::with_capacity(n_subjects);

    for &subj in &unique_subjects {
        let subj_data: Vec<(f64, f64)> = subject_id
            .iter()
            .zip(measurement_times.iter())
            .zip(measurement_values.iter())
            .filter(|((id, t), _)| **id == subj && **t <= prediction_time)
            .map(|((_, t), v)| (*t, *v))
            .collect();

        let (current_value, slope) = if subj_data.is_empty() {
            (0.0, 0.0)
        } else if subj_data.len() == 1 {
            (subj_data[0].1, 0.0)
        } else {
            let last = subj_data.last().unwrap();
            let prev = subj_data.get(subj_data.len() - 2).unwrap();
            let slope = (last.1 - prev.1) / (last.0 - prev.0 + 1e-10);
            (last.1, slope)
        };

        let features = [current_value, slope];
        let risk: f64 = features
            .iter()
            .zip(model_coefficients.iter())
            .map(|(&f, &c)| f * c)
            .sum::<f64>()
            .exp();

        risk_scores.push(risk);

        let baseline_survival = (-0.01 * horizon).exp();
        let surv_prob = baseline_survival.powf(risk);
        survival_probabilities.push(surv_prob);

        let se = 0.05;
        confidence_lower.push((surv_prob - 1.96 * se).clamp(0.0, 1.0));
        confidence_upper.push((surv_prob + 1.96 * se).clamp(0.0, 1.0));
    }

    Ok(LongDynamicPredResult {
        prediction_time,
        horizon,
        survival_probabilities,
        confidence_lower,
        confidence_upper,
        risk_scores,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeVaryingCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub coefficient_times: Vec<f64>,
    #[pyo3(get)]
    pub standard_errors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
}

#[pymethods]
impl TimeVaryingCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "TimeVaryingCoxResult(n_times={}, n_events={}, LL={:.2})",
            self.coefficient_times.len(),
            self.n_events,
            self.log_likelihood
        )
    }

    fn coefficients_at_time(&self, t: f64) -> Vec<f64> {
        let idx = self
            .coefficient_times
            .iter()
            .position(|&ct| ct >= t)
            .unwrap_or(self.coefficient_times.len() - 1);
        self.coefficients
            .iter()
            .skip(idx * self.coefficients.len() / self.coefficient_times.len())
            .take(self.coefficients.len() / self.coefficient_times.len())
            .cloned()
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    start_time,
    stop_time,
    event,
    covariates,
    n_time_points=10
))]
pub fn time_varying_cox(
    start_time: Vec<f64>,
    stop_time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    n_time_points: usize,
) -> PyResult<TimeVaryingCoxResult> {
    let n = start_time.len();
    if n == 0 || stop_time.len() != n || event.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same non-zero length",
        ));
    }

    let n_features = if covariates.is_empty() {
        0
    } else {
        covariates[0].len()
    };

    let max_time = stop_time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let coefficient_times: Vec<f64> = (0..n_time_points)
        .map(|i| max_time * (i + 1) as f64 / n_time_points as f64)
        .collect();

    let mut all_coefficients = Vec::new();
    let mut all_standard_errors = Vec::new();
    let mut total_ll = 0.0;

    for &time_point in &coefficient_times {
        let at_risk: Vec<usize> = (0..n)
            .filter(|&i| start_time[i] < time_point && stop_time[i] >= time_point)
            .collect();

        if at_risk.is_empty() {
            all_coefficients.extend(vec![0.0; n_features]);
            all_standard_errors.push(vec![f64::INFINITY; n_features]);
            continue;
        }

        let mut coefficients = vec![0.0; n_features];
        let learning_rate = 0.01;

        for _ in 0..50 {
            let linear_pred: Vec<f64> = at_risk
                .iter()
                .map(|&i| {
                    covariates[i]
                        .iter()
                        .zip(coefficients.iter())
                        .map(|(&x, &b)| x * b)
                        .sum()
                })
                .collect();

            let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();
            let risk_sum: f64 = exp_lp.iter().sum();

            let mut gradient = vec![0.0; n_features];

            for &i in at_risk.iter() {
                if event[i] == 1 && (stop_time[i] - time_point).abs() < 1e-10 {
                    for (j, &xij) in covariates[i].iter().enumerate() {
                        let weighted_mean: f64 = at_risk
                            .iter()
                            .enumerate()
                            .map(|(k, &kk)| covariates[kk][j] * exp_lp[k])
                            .sum::<f64>()
                            / risk_sum;
                        gradient[j] += xij - weighted_mean;
                    }
                }
            }

            for (b, g) in coefficients.iter_mut().zip(gradient.iter()) {
                *b += learning_rate * g;
            }
        }

        let linear_pred: Vec<f64> = at_risk
            .iter()
            .map(|&i| {
                covariates[i]
                    .iter()
                    .zip(coefficients.iter())
                    .map(|(&x, &b)| x * b)
                    .sum()
            })
            .collect();

        let exp_lp: Vec<f64> = linear_pred.iter().map(|&lp| lp.exp()).collect();
        let risk_sum: f64 = exp_lp.iter().sum();

        for (idx, &i) in at_risk.iter().enumerate() {
            if event[i] == 1 && (stop_time[i] - time_point).abs() < 1e-10 {
                total_ll += linear_pred[idx] - risk_sum.ln();
            }
        }

        all_coefficients.extend(coefficients.clone());
        all_standard_errors.push(vec![0.1; n_features]);
    }

    let n_events = event.iter().filter(|&&e| e == 1).count();

    Ok(TimeVaryingCoxResult {
        coefficients: all_coefficients,
        coefficient_times,
        standard_errors: all_standard_errors,
        log_likelihood: total_ll,
        n_events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_model() {
        let subject_id = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let longitudinal_times = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let longitudinal_values = vec![1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 1.5, 2.0, 2.5];
        let survival_time = vec![3.0, 2.5, 4.0];
        let survival_event = vec![1, 1, 0];

        let config = JointModelConfig::new(5, 10, 1e-4, "value", "intercept").unwrap();

        let result = joint_longitudinal_model(
            subject_id,
            longitudinal_times,
            longitudinal_values,
            survival_time,
            survival_event,
            vec![],
            vec![],
            config,
        )
        .unwrap();

        assert!(result.longitudinal_fixed_effects.len() >= 2);
    }

    #[test]
    fn test_landmark_analysis() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let event = vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let covariates = vec![
            vec![0.1, 0.2],
            vec![0.2, 0.3],
            vec![0.3, 0.4],
            vec![0.4, 0.5],
            vec![0.5, 0.6],
            vec![0.6, 0.7],
            vec![0.7, 0.8],
            vec![0.8, 0.9],
            vec![0.9, 1.0],
            vec![1.0, 1.1],
        ];

        let result = landmark_cox_analysis(time, event, covariates, 2.0, 5.0).unwrap();
        assert_eq!(result.landmark_time, 2.0);
        assert!(result.n_at_risk > 0);
    }

    #[test]
    fn test_dynamic_prediction() {
        let subject_id = vec![0, 0, 0, 1, 1, 1];
        let measurement_times = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let measurement_values = vec![1.0, 1.5, 2.0, 2.0, 2.2, 2.5];
        let model_coefficients = vec![0.1, 0.05];

        let result = longitudinal_dynamic_pred(
            subject_id,
            measurement_times,
            measurement_values,
            2.0,
            3.0,
            model_coefficients,
        )
        .unwrap();

        assert_eq!(result.survival_probabilities.len(), 2);
    }

    #[test]
    fn test_time_varying_cox() {
        let start_time = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let stop_time = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0];
        let event = vec![0, 1, 0, 1, 0, 1];
        let covariates = vec![
            vec![0.1, 0.2],
            vec![0.2, 0.3],
            vec![0.3, 0.4],
            vec![0.4, 0.5],
            vec![0.5, 0.6],
            vec![0.6, 0.7],
        ];

        let result = time_varying_cox(start_time, stop_time, event, covariates, 5).unwrap();
        assert_eq!(result.coefficient_times.len(), 5);
        assert_eq!(result.n_events, 3);
    }
}
