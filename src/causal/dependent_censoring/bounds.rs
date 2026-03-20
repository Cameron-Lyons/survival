
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SensitivityBoundsConfig {
    #[pyo3(get, set)]
    pub gamma_range: Vec<f64>,
    #[pyo3(get, set)]
    pub n_grid: usize,
    #[pyo3(get, set)]
    pub method: String,
}

#[pymethods]
impl SensitivityBoundsConfig {
    #[new]
    #[pyo3(signature = (gamma_range=None, n_grid=100, method="rosenbaum"))]
    pub fn new(gamma_range: Option<Vec<f64>>, n_grid: usize, method: &str) -> Self {
        SensitivityBoundsConfig {
            gamma_range: gamma_range.unwrap_or_else(|| vec![1.0, 1.5, 2.0, 2.5, 3.0]),
            n_grid,
            method: method.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SensitivityBoundsResult {
    #[pyo3(get)]
    pub gamma_values: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub rmst_lower: Vec<f64>,
    #[pyo3(get)]
    pub rmst_upper: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_lower: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratio_upper: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub point_estimate: f64,
}

#[pyfunction]
#[pyo3(signature = (time, event, treatment, covariates, tau, config))]
pub fn sensitivity_bounds_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    treatment: Vec<i32>,
    covariates: Vec<f64>,
    tau: f64,
    config: &SensitivityBoundsConfig,
) -> PyResult<SensitivityBoundsResult> {
    let n = time.len();
    if event.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }
    if !covariates.is_empty() && (n == 0 || !covariates.len().is_multiple_of(n)) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must be empty or have length n_obs * n_covariates",
        ));
    }

    let max_time = tau.min(time.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let eval_times: Vec<f64> = (0..config.n_grid)
        .map(|i| (i as f64 + 1.0) / config.n_grid as f64 * max_time)
        .collect();

    let treated_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] == 1).collect();
    let control_idx: Vec<usize> = (0..n).filter(|&i| treatment[i] == 0).collect();

    let treated_time: Vec<f64> = treated_idx.iter().map(|&i| time[i]).collect();
    let treated_event: Vec<i32> = treated_idx.iter().map(|&i| event[i]).collect();
    let control_time: Vec<f64> = control_idx.iter().map(|&i| time[i]).collect();
    let control_event: Vec<i32> = control_idx.iter().map(|&i| event[i]).collect();

    let km_treated = estimate_km(&treated_time, &treated_event, &eval_times);
    let km_control = estimate_km(&control_time, &control_event, &eval_times);

    let point_estimate =
        compute_rmst(&km_treated, &eval_times, tau) - compute_rmst(&km_control, &eval_times, tau);

    let mut survival_lower = Vec::new();
    let mut survival_upper = Vec::new();
    let mut rmst_lower = Vec::new();
    let mut rmst_upper = Vec::new();
    let mut hazard_ratio_lower = Vec::new();
    let mut hazard_ratio_upper = Vec::new();

    for &gamma in &config.gamma_range {
        let adjustment = (gamma - 1.0) / (gamma + 1.0);

        let surv_lower: Vec<f64> = km_treated
            .iter()
            .zip(km_control.iter())
            .map(|(&st, &sc)| {
                let diff = st - sc;
                (diff - adjustment * (1.0 - st.min(sc))).clamp(-1.0, 1.0)
            })
            .collect();

        let surv_upper: Vec<f64> = km_treated
            .iter()
            .zip(km_control.iter())
            .map(|(&st, &sc)| {
                let diff = st - sc;
                (diff + adjustment * st.min(sc)).clamp(-1.0, 1.0)
            })
            .collect();

        survival_lower.push(surv_lower.clone());
        survival_upper.push(surv_upper.clone());

        let rmst_l = point_estimate - adjustment * tau * 0.5;
        let rmst_u = point_estimate + adjustment * tau * 0.5;
        rmst_lower.push(rmst_l);
        rmst_upper.push(rmst_u);

        let hr_point = estimate_hazard_ratio(&time, &event, &treatment);
        let hr_l = hr_point * (1.0 / gamma);
        let hr_u = hr_point * gamma;
        hazard_ratio_lower.push(hr_l);
        hazard_ratio_upper.push(hr_u);
    }

    Ok(SensitivityBoundsResult {
        gamma_values: config.gamma_range.clone(),
        survival_lower,
        survival_upper,
        rmst_lower,
        rmst_upper,
        hazard_ratio_lower,
        hazard_ratio_upper,
        eval_times,
        point_estimate,
    })
}

fn compute_rmst(survival: &[f64], times: &[f64], tau: f64) -> f64 {
    let mut rmst = 0.0;
    let mut prev_time = 0.0;

    for (i, &t) in times.iter().enumerate() {
        if t > tau {
            break;
        }
        let dt = t - prev_time;
        let s = if i > 0 { survival[i - 1] } else { 1.0 };
        rmst += s * dt;
        prev_time = t;
    }

    rmst
}

fn estimate_hazard_ratio(time: &[f64], event: &[i32], treatment: &[i32]) -> f64 {
    let n = time.len();

    let mut beta = 0.0;

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[b]
            .partial_cmp(&time[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for _ in 0..50 {
        let mut gradient = 0.0;
        let mut hessian = 0.0;
        let mut risk_sum = 0.0;
        let mut weighted_t = 0.0;
        let mut weighted_tt = 0.0;

        for &i in &sorted_indices {
            let t_i = treatment[i] as f64;
            let exp_bt = (beta * t_i).clamp(-700.0, 700.0).exp();

            risk_sum += exp_bt;
            weighted_t += exp_bt * t_i;
            weighted_tt += exp_bt * t_i * t_i;

            if event[i] == 1 && risk_sum > 0.0 {
                let t_bar = weighted_t / risk_sum;
                let tt_bar = weighted_tt / risk_sum;
                gradient += t_i - t_bar;
                hessian += tt_bar - t_bar * t_bar;
            }
        }

        if hessian.abs() > crate::constants::DIVISION_FLOOR {
            beta += gradient / hessian;
            beta = beta.clamp(-10.0, 10.0);
        }
    }

    beta.exp()
}
