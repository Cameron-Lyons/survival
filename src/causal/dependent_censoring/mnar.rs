
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MNARSurvivalConfig {
    #[pyo3(get, set)]
    pub delta_range: Vec<f64>,
    #[pyo3(get, set)]
    pub pattern: String,
}

#[pymethods]
impl MNARSurvivalConfig {
    #[new]
    #[pyo3(signature = (delta_range=None, pattern="tilt"))]
    pub fn new(delta_range: Option<Vec<f64>>, pattern: &str) -> Self {
        MNARSurvivalConfig {
            delta_range: delta_range.unwrap_or_else(|| vec![-1.0, -0.5, 0.0, 0.5, 1.0]),
            pattern: pattern.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MNARSurvivalResult {
    #[pyo3(get)]
    pub delta_values: Vec<f64>,
    #[pyo3(get)]
    pub adjusted_survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub adjusted_rmst: Vec<f64>,
    #[pyo3(get)]
    pub adjusted_median: Vec<f64>,
    #[pyo3(get)]
    pub eval_times: Vec<f64>,
    #[pyo3(get)]
    pub reference_survival: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (time, event, covariates, config))]
pub fn mnar_sensitivity_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    covariates: Vec<f64>,
    config: &MNARSurvivalConfig,
) -> PyResult<MNARSurvivalResult> {
    let n = time.len();
    if event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have same length",
        ));
    }
    if !covariates.is_empty() && (n == 0 || !covariates.len().is_multiple_of(n)) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must be empty or have length n_obs * n_covariates",
        ));
    }

    let max_time = time.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_grid = 100;
    let eval_times: Vec<f64> = (0..n_grid)
        .map(|i| (i as f64 + 1.0) / n_grid as f64 * max_time)
        .collect();

    let reference_survival = estimate_km(&time, &event, &eval_times);

    let mut adjusted_survival = Vec::new();
    let mut adjusted_rmst = Vec::new();
    let mut adjusted_median = Vec::new();

    for &delta in &config.delta_range {
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                if event[i] == 0 {
                    (delta * time[i] / max_time).exp()
                } else {
                    1.0
                }
            })
            .collect();

        let adj_surv = estimate_weighted_km(&time, &event, &weights, &eval_times);
        adjusted_survival.push(adj_surv.clone());

        let rmst = compute_rmst(&adj_surv, &eval_times, max_time);
        adjusted_rmst.push(rmst);

        let median = adj_surv
            .iter()
            .zip(eval_times.iter())
            .find(|(s, _)| **s <= 0.5)
            .map(|(_, t)| *t)
            .unwrap_or(max_time);
        adjusted_median.push(median);
    }

    Ok(MNARSurvivalResult {
        delta_values: config.delta_range.clone(),
        adjusted_survival,
        adjusted_rmst,
        adjusted_median,
        eval_times,
        reference_survival,
    })
}

fn estimate_weighted_km(
    time: &[f64],
    event: &[i32],
    weights: &[f64],
    eval_times: &[f64],
) -> Vec<f64> {
    let n = time.len();
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut survival = 1.0;
    let total_weight: f64 = weights.iter().sum();
    let mut at_risk_weight = total_weight;
    let mut km_times = Vec::new();
    let mut km_values = Vec::new();

    km_times.push(0.0);
    km_values.push(1.0);

    for &i in &sorted_indices {
        if event[i] == 1 {
            survival *= 1.0 - weights[i] / at_risk_weight;
            km_times.push(time[i]);
            km_values.push(survival);
        }
        at_risk_weight -= weights[i];
    }

    eval_times
        .iter()
        .map(|&t| {
            let idx = km_times.iter().rposition(|&kt| kt <= t).unwrap_or(0);
            km_values[idx]
        })
        .collect()
}

