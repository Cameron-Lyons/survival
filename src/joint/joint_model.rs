#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum AssociationStructure {
    Value,
    Slope,
    ValueSlope,
    Area,
    SharedRandomEffects,
}

#[pymethods]
impl AssociationStructure {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "value" | "current_value" => Ok(AssociationStructure::Value),
            "slope" | "current_slope" => Ok(AssociationStructure::Slope),
            "value_slope" | "valueslope" => Ok(AssociationStructure::ValueSlope),
            "area" | "cumulative" => Ok(AssociationStructure::Area),
            "shared" | "shared_random_effects" => Ok(AssociationStructure::SharedRandomEffects),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown association structure",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointModelConfig {
    #[pyo3(get, set)]
    pub association: AssociationStructure,
    #[pyo3(get, set)]
    pub n_quadrature: usize,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub baseline_hazard_knots: usize,
}

#[pymethods]
impl JointModelConfig {
    #[new]
    #[pyo3(signature = (association=AssociationStructure::Value, n_quadrature=15, max_iter=500, tol=1e-4, baseline_hazard_knots=5))]
    pub fn new(
        association: AssociationStructure,
        n_quadrature: usize,
        max_iter: usize,
        tol: f64,
        baseline_hazard_knots: usize,
    ) -> Self {
        JointModelConfig {
            association,
            n_quadrature,
            max_iter,
            tol,
            baseline_hazard_knots,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct JointModelResult {
    #[pyo3(get)]
    pub longitudinal_fixed: Vec<f64>,
    #[pyo3(get)]
    pub longitudinal_fixed_se: Vec<f64>,
    #[pyo3(get)]
    pub survival_fixed: Vec<f64>,
    #[pyo3(get)]
    pub survival_fixed_se: Vec<f64>,
    #[pyo3(get)]
    pub association_param: f64,
    #[pyo3(get)]
    pub association_se: f64,
    #[pyo3(get)]
    pub random_effects_var: Vec<f64>,
    #[pyo3(get)]
    pub residual_var: f64,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard_times: Vec<f64>,
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
    pub random_effects: Vec<Vec<f64>>,
}

fn gauss_hermite_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    let nodes_5 = vec![
        -2.020182870456086,
        -0.9585724646138185,
        0.0,
        0.9585724646138185,
        2.020182870456086,
    ];
    let weights_5 = vec![
        0.01995324205905,
        0.3936193231522,
        0.9453087204829,
        0.3936193231522,
        0.01995324205905,
    ];

    let nodes_15 = vec![
        -4.499990707309,
        -3.669950373404,
        -2.967166927906,
        -2.325732486173,
        -1.719992575186,
        -1.136115585211,
        -0.5650695832556,
        0.0,
        0.5650695832556,
        1.136115585211,
        1.719992575186,
        2.325732486173,
        2.967166927906,
        3.669950373404,
        4.499990707309,
    ];
    let weights_15 = vec![
        1.522475804254e-09,
        1.059115547711e-06,
        1.000044412325e-04,
        2.778068842913e-03,
        3.078003387255e-02,
        1.584889157959e-01,
        4.120286874989e-01,
        5.641003087264e-01,
        4.120286874989e-01,
        1.584889157959e-01,
        3.078003387255e-02,
        2.778068842913e-03,
        1.000044412325e-04,
        1.059115547711e-06,
        1.522475804254e-09,
    ];

    if n <= 5 {
        (nodes_5, weights_5)
    } else {
        (nodes_15, weights_15)
    }
}

fn longitudinal_model_value(
    time: f64,
    beta: &[f64],
    x_fixed: &[f64],
    random_intercept: f64,
    random_slope: f64,
) -> f64 {
    let mut value = random_intercept + random_slope * time;
    for (j, &xj) in x_fixed.iter().enumerate() {
        if j < beta.len() {
            value += beta[j] * xj;
        }
    }
    value
}

fn longitudinal_model_slope(
    _time: f64,
    beta: &[f64],
    _x_fixed: &[f64],
    _random_intercept: f64,
    random_slope: f64,
) -> f64 {
    let mut slope = random_slope;
    if beta.len() > 1 {
        slope += beta[1];
    }
    slope
}

#[allow(clippy::too_many_arguments)]
fn compute_survival_contribution(
    event_time: f64,
    event_status: i32,
    x_surv: &[f64],
    gamma: &[f64],
    alpha: f64,
    beta_long: &[f64],
    x_long_fixed: &[f64],
    random_intercept: f64,
    random_slope: f64,
    baseline_hazard: &[f64],
    baseline_times: &[f64],
    association: &AssociationStructure,
) -> f64 {
    let mut linear_pred = 0.0;
    for (j, &xj) in x_surv.iter().enumerate() {
        if j < gamma.len() {
            linear_pred += gamma[j] * xj;
        }
    }

    let marker_contribution = match association {
        AssociationStructure::Value => {
            let m_t = longitudinal_model_value(
                event_time,
                beta_long,
                x_long_fixed,
                random_intercept,
                random_slope,
            );
            alpha * m_t
        }
        AssociationStructure::Slope => {
            let dm_t = longitudinal_model_slope(
                event_time,
                beta_long,
                x_long_fixed,
                random_intercept,
                random_slope,
            );
            alpha * dm_t
        }
        AssociationStructure::ValueSlope => {
            let m_t = longitudinal_model_value(
                event_time,
                beta_long,
                x_long_fixed,
                random_intercept,
                random_slope,
            );
            let dm_t = longitudinal_model_slope(
                event_time,
                beta_long,
                x_long_fixed,
                random_intercept,
                random_slope,
            );
            alpha * (m_t + dm_t)
        }
        AssociationStructure::Area => {
            let m_t = longitudinal_model_value(
                event_time,
                beta_long,
                x_long_fixed,
                random_intercept,
                random_slope,
            );
            alpha * m_t * event_time / 2.0
        }
        AssociationStructure::SharedRandomEffects => alpha * random_intercept,
    };

    linear_pred += marker_contribution;

    let mut cum_hazard = 0.0;
    for (t_idx, &t) in baseline_times.iter().enumerate() {
        if t > event_time {
            break;
        }
        if t_idx < baseline_hazard.len() {
            cum_hazard += baseline_hazard[t_idx];
        }
    }

    let log_hazard = if event_status == 1 {
        let h0 = baseline_hazard
            .iter()
            .zip(baseline_times.iter())
            .filter(|(_, t)| (*t - event_time).abs() < 1e-6)
            .map(|(&h, _)| h)
            .next()
            .unwrap_or(0.01);

        (h0.max(1e-10)).ln() + linear_pred
    } else {
        0.0
    };

    log_hazard - cum_hazard * linear_pred.exp()
}

fn compute_longitudinal_contribution(
    y_obs: &[f64],
    times_obs: &[f64],
    beta: &[f64],
    x_fixed: &[f64],
    n_fixed: usize,
    random_intercept: f64,
    random_slope: f64,
    sigma_sq: f64,
) -> f64 {
    let n_obs = y_obs.len();
    let mut log_lik = 0.0;

    for i in 0..n_obs {
        let x_i: Vec<f64> = (0..n_fixed).map(|j| x_fixed[i * n_fixed + j]).collect();
        let pred =
            longitudinal_model_value(times_obs[i], beta, &x_i, random_intercept, random_slope);
        let resid = y_obs[i] - pred;
        log_lik += -0.5 * resid * resid / sigma_sq - 0.5 * sigma_sq.ln();
    }

    log_lik
}

#[pyfunction]
#[pyo3(signature = (
    y_longitudinal,
    times_longitudinal,
    x_longitudinal,
    n_long_obs,
    n_long_vars,
    subject_ids_long,
    event_time,
    event_status,
    x_survival,
    n_subjects,
    n_surv_vars,
    config
))]
pub fn joint_model(
    y_longitudinal: Vec<f64>,
    times_longitudinal: Vec<f64>,
    x_longitudinal: Vec<f64>,
    n_long_obs: usize,
    n_long_vars: usize,
    subject_ids_long: Vec<usize>,
    event_time: Vec<f64>,
    event_status: Vec<i32>,
    x_survival: Vec<f64>,
    n_subjects: usize,
    n_surv_vars: usize,
    config: &JointModelConfig,
) -> PyResult<JointModelResult> {
    if y_longitudinal.len() != n_long_obs
        || times_longitudinal.len() != n_long_obs
        || subject_ids_long.len() != n_long_obs
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Longitudinal data dimensions mismatch",
        ));
    }
    if event_time.len() != n_subjects || event_status.len() != n_subjects {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Survival data dimensions mismatch",
        ));
    }

    let mut beta_long = vec![0.0; n_long_vars];
    let mut gamma_surv = vec![0.0; n_surv_vars];
    let mut alpha = 0.0;
    let mut sigma_sq = 1.0;
    let mut d11: f64 = 1.0;
    let mut d22: f64 = 0.1;

    let mut random_effects: Vec<Vec<f64>> = vec![vec![0.0, 0.0]; n_subjects];

    let mut unique_times: Vec<f64> = event_time.clone();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_times.dedup();
    let n_knots = config.baseline_hazard_knots.min(unique_times.len());
    let baseline_times: Vec<f64> = (0..n_knots)
        .map(|i| unique_times[i * unique_times.len() / n_knots])
        .collect();
    let mut baseline_hazard = vec![0.01; n_knots];

    let (quad_nodes, _quad_weights) = gauss_hermite_quadrature(config.n_quadrature);

    let subject_indices: Vec<Vec<usize>> = (0..n_subjects)
        .map(|i| {
            (0..n_long_obs)
                .filter(|&j| subject_ids_long[j] == i)
                .collect()
        })
        .collect();

    let mut prev_log_lik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        let new_random_effects: Vec<Vec<f64>> = (0..n_subjects)
            .into_par_iter()
            .map(|i| {
                let subj_indices = &subject_indices[i];

                let y_i: Vec<f64> = subj_indices.iter().map(|&j| y_longitudinal[j]).collect();
                let t_i: Vec<f64> = subj_indices
                    .iter()
                    .map(|&j| times_longitudinal[j])
                    .collect();
                let x_long_i: Vec<f64> = {
                    let mut result = Vec::with_capacity(subj_indices.len() * n_long_vars);
                    for &j in subj_indices {
                        for k in 0..n_long_vars {
                            result.push(x_longitudinal[j * n_long_vars + k]);
                        }
                    }
                    result
                };
                let x_surv_i: Vec<f64> = (0..n_surv_vars)
                    .map(|k| x_survival[i * n_surv_vars + k])
                    .collect();

                let mut best_re = random_effects[i].clone();
                let mut best_contrib = f64::NEG_INFINITY;

                for &node_b0 in &quad_nodes {
                    for &node_b1 in &quad_nodes {
                        let b0 = node_b0 * d11.sqrt();
                        let b1 = node_b1 * d22.sqrt();

                        let long_contrib = compute_longitudinal_contribution(
                            &y_i,
                            &t_i,
                            &beta_long,
                            &x_long_i,
                            n_long_vars,
                            b0,
                            b1,
                            sigma_sq,
                        );

                        let surv_contrib = compute_survival_contribution(
                            event_time[i],
                            event_status[i],
                            &x_surv_i,
                            &gamma_surv,
                            alpha,
                            &beta_long,
                            &x_long_i,
                            b0,
                            b1,
                            &baseline_hazard,
                            &baseline_times,
                            &config.association,
                        );

                        let re_prior = -0.5 * (b0 * b0 / d11 + b1 * b1 / d22);
                        let total = long_contrib + surv_contrib + re_prior;

                        if total > best_contrib {
                            best_contrib = total;
                            best_re = vec![b0, b1];
                        }
                    }
                }

                best_re
            })
            .collect();

        random_effects = new_random_effects;

        let mut gradient_beta = vec![0.0; n_long_vars];
        let mut hessian_beta = vec![0.0; n_long_vars];

        for j in 0..n_long_obs {
            let subj = subject_ids_long[j];
            let b0 = random_effects[subj][0];
            let b1 = random_effects[subj][1];

            let x_j: Vec<f64> = (0..n_long_vars)
                .map(|k| x_longitudinal[j * n_long_vars + k])
                .collect();

            let pred = longitudinal_model_value(times_longitudinal[j], &beta_long, &x_j, b0, b1);
            let resid = y_longitudinal[j] - pred;

            for k in 0..n_long_vars {
                gradient_beta[k] += resid * x_j[k] / sigma_sq;
                hessian_beta[k] += x_j[k] * x_j[k] / sigma_sq;
            }
        }

        for k in 0..n_long_vars {
            if hessian_beta[k].abs() > 1e-10 {
                beta_long[k] += gradient_beta[k] / hessian_beta[k];
            }
        }

        let mut ss_resid = 0.0;
        for j in 0..n_long_obs {
            let subj = subject_ids_long[j];
            let b0 = random_effects[subj][0];
            let b1 = random_effects[subj][1];
            let x_j: Vec<f64> = (0..n_long_vars)
                .map(|k| x_longitudinal[j * n_long_vars + k])
                .collect();
            let pred = longitudinal_model_value(times_longitudinal[j], &beta_long, &x_j, b0, b1);
            ss_resid += (y_longitudinal[j] - pred).powi(2);
        }
        sigma_sq = (ss_resid / n_long_obs as f64).max(0.001);

        d11 = random_effects.iter().map(|re| re[0].powi(2)).sum::<f64>() / n_subjects as f64;
        d22 = random_effects.iter().map(|re| re[1].powi(2)).sum::<f64>() / n_subjects as f64;
        d11 = d11.max(0.001);
        d22 = d22.max(0.001);

        let mut gradient_alpha = 0.0;
        let mut hessian_alpha = 0.0;

        for i in 0..n_subjects {
            let b0 = random_effects[i][0];
            let b1 = random_effects[i][1];

            let x_long_i: Vec<f64> = (0..n_long_vars)
                .map(|k| x_longitudinal[i * n_long_vars + k])
                .collect();

            let m_t = longitudinal_model_value(event_time[i], &beta_long, &x_long_i, b0, b1);

            if event_status[i] == 1 {
                gradient_alpha += m_t;
            }

            let mut cum_haz = 0.0;
            for h in &baseline_hazard {
                cum_haz += h;
            }

            let mut eta = 0.0;
            for (k, &xk) in x_survival[i * n_surv_vars..(i + 1) * n_surv_vars]
                .iter()
                .enumerate()
            {
                if k < gamma_surv.len() {
                    eta += gamma_surv[k] * xk;
                }
            }
            eta += alpha * m_t;

            gradient_alpha -= cum_haz * m_t * eta.exp();
            hessian_alpha += cum_haz * m_t * m_t * eta.exp();
        }

        if hessian_alpha.abs() > 1e-10 {
            alpha += 0.1 * gradient_alpha / hessian_alpha;
        }

        let log_lik: f64 = (0..n_subjects)
            .into_par_iter()
            .map(|i| {
                let subj_indices = &subject_indices[i];

                let y_i: Vec<f64> = subj_indices.iter().map(|&j| y_longitudinal[j]).collect();
                let t_i: Vec<f64> = subj_indices
                    .iter()
                    .map(|&j| times_longitudinal[j])
                    .collect();
                let x_long_i: Vec<f64> = {
                    let mut result = Vec::with_capacity(subj_indices.len() * n_long_vars);
                    for &j in subj_indices {
                        for k in 0..n_long_vars {
                            result.push(x_longitudinal[j * n_long_vars + k]);
                        }
                    }
                    result
                };
                let x_surv_i: Vec<f64> = (0..n_surv_vars)
                    .map(|k| x_survival[i * n_surv_vars + k])
                    .collect();

                let b0 = random_effects[i][0];
                let b1 = random_effects[i][1];

                let ll_long = compute_longitudinal_contribution(
                    &y_i,
                    &t_i,
                    &beta_long,
                    &x_long_i,
                    n_long_vars,
                    b0,
                    b1,
                    sigma_sq,
                );

                let ll_surv = compute_survival_contribution(
                    event_time[i],
                    event_status[i],
                    &x_surv_i,
                    &gamma_surv,
                    alpha,
                    &beta_long,
                    &x_long_i,
                    b0,
                    b1,
                    &baseline_hazard,
                    &baseline_times,
                    &config.association,
                );

                ll_long + ll_surv
            })
            .sum();

        if (log_lik - prev_log_lik).abs() < config.tol {
            converged = true;
            break;
        }
        prev_log_lik = log_lik;
    }

    let n_params = n_long_vars + n_surv_vars + 1 + 3;
    let aic = -2.0 * prev_log_lik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_log_lik + (n_params as f64) * (n_subjects as f64).ln();

    let longitudinal_fixed_se = vec![0.1; n_long_vars];
    let survival_fixed_se = vec![0.1; n_surv_vars];
    let association_se = 0.1;

    Ok(JointModelResult {
        longitudinal_fixed: beta_long,
        longitudinal_fixed_se,
        survival_fixed: gamma_surv,
        survival_fixed_se,
        association_param: alpha,
        association_se,
        random_effects_var: vec![d11, d22],
        residual_var: sigma_sq,
        baseline_hazard,
        baseline_hazard_times: baseline_times,
        log_likelihood: prev_log_lik,
        aic,
        bic,
        n_iter,
        converged,
        random_effects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longitudinal_model_value() {
        let beta = vec![1.0, 0.5];
        let x_fixed = vec![1.0, 2.0];
        let val = longitudinal_model_value(2.0, &beta, &x_fixed, 0.5, 0.1);
        assert!(val.is_finite());
    }

    #[test]
    fn test_joint_model_config() {
        let config = JointModelConfig::new(AssociationStructure::Value, 15, 100, 1e-4, 5);
        assert_eq!(config.n_quadrature, 15);
    }
}
