
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RDSurvivalConfig {
    #[pyo3(get, set)]
    pub bandwidth: f64,
    #[pyo3(get, set)]
    pub kernel: String,
    #[pyo3(get, set)]
    pub polynomial_order: usize,
    #[pyo3(get, set)]
    pub fuzzy: bool,
}

#[pymethods]
impl RDSurvivalConfig {
    #[new]
    #[pyo3(signature = (bandwidth=None, kernel="triangular", polynomial_order=1, fuzzy=false))]
    pub fn new(
        bandwidth: Option<f64>,
        kernel: &str,
        polynomial_order: usize,
        fuzzy: bool,
    ) -> PyResult<Self> {
        Ok(RDSurvivalConfig {
            bandwidth: bandwidth.unwrap_or(0.0),
            kernel: kernel.to_string(),
            polynomial_order,
            fuzzy,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct RDSurvivalResult {
    #[pyo3(get)]
    pub treatment_effect: f64,
    #[pyo3(get)]
    pub se: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub z_score: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub bandwidth_used: f64,
    #[pyo3(get)]
    pub n_left: usize,
    #[pyo3(get)]
    pub n_right: usize,
    #[pyo3(get)]
    pub survival_left: f64,
    #[pyo3(get)]
    pub survival_right: f64,
}

fn kernel_weight(x: f64, bandwidth: f64, kernel: &str) -> f64 {
    let u = x / bandwidth;
    if u.abs() > 1.0 {
        return 0.0;
    }
    match kernel {
        "triangular" => 1.0 - u.abs(),
        "uniform" | "rectangular" => 1.0,
        "epanechnikov" => 0.75 * (1.0 - u * u),
        _ => 1.0 - u.abs(),
    }
}

#[pyfunction]
#[pyo3(signature = (time, event, running_var, cutoff, treatment, covariates, config))]
pub fn rd_survival(
    time: Vec<f64>,
    event: Vec<i32>,
    running_var: Vec<f64>,
    cutoff: f64,
    treatment: Vec<f64>,
    covariates: Vec<f64>,
    config: &RDSurvivalConfig,
) -> PyResult<RDSurvivalResult> {
    let n = time.len();
    if event.len() != n || running_var.len() != n || treatment.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }
    if !covariates.is_empty() && (n == 0 || !covariates.len().is_multiple_of(n)) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must be empty or have length n_obs * n_covariates",
        ));
    }

    let bandwidth = if config.bandwidth > 0.0 {
        config.bandwidth
    } else {
        let centered: Vec<f64> = running_var.iter().map(|&r| r - cutoff).collect();
        let iqr = {
            let mut sorted = centered.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = sorted[n / 4];
            let q3 = sorted[3 * n / 4];
            q3 - q1
        };
        1.06 * iqr * (n as f64).powf(-0.2)
    };

    let left_indices: Vec<usize> = (0..n)
        .filter(|&i| running_var[i] < cutoff && running_var[i] >= cutoff - bandwidth)
        .collect();

    let right_indices: Vec<usize> = (0..n)
        .filter(|&i| running_var[i] >= cutoff && running_var[i] <= cutoff + bandwidth)
        .collect();

    let n_left = left_indices.len();
    let n_right = right_indices.len();

    if n_left < 10 || n_right < 10 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Insufficient observations near cutoff",
        ));
    }

    let survival_left = estimate_km_survival(
        &left_indices,
        &time,
        &event,
        &running_var,
        cutoff,
        bandwidth,
        &config.kernel,
    );

    let survival_right = estimate_km_survival(
        &right_indices,
        &time,
        &event,
        &running_var,
        cutoff,
        bandwidth,
        &config.kernel,
    );

    let treatment_effect = survival_right - survival_left;

    let se_left = (survival_left * (1.0 - survival_left) / n_left as f64).sqrt();
    let se_right = (survival_right * (1.0 - survival_right) / n_right as f64).sqrt();
    let se = (se_left.powi(2) + se_right.powi(2)).sqrt();

    let z_score = if se > crate::constants::DIVISION_FLOOR {
        treatment_effect / se
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    let ci_lower = treatment_effect - 1.96 * se;
    let ci_upper = treatment_effect + 1.96 * se;

    Ok(RDSurvivalResult {
        treatment_effect,
        se,
        ci_lower,
        ci_upper,
        z_score,
        p_value,
        bandwidth_used: bandwidth,
        n_left,
        n_right,
        survival_left,
        survival_right,
    })
}

fn estimate_km_survival(
    indices: &[usize],
    time: &[f64],
    event: &[i32],
    running_var: &[f64],
    cutoff: f64,
    bandwidth: f64,
    kernel: &str,
) -> f64 {
    if indices.is_empty() {
        return 1.0;
    }

    let mut sorted_indices: Vec<usize> = indices.to_vec();
    sorted_indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let max_time = time[sorted_indices[sorted_indices.len() / 2]];

    let mut survival = 1.0;
    let mut at_risk = 0.0;

    for &i in indices {
        at_risk += kernel_weight(running_var[i] - cutoff, bandwidth, kernel);
    }

    for &i in &sorted_indices {
        if time[i] > max_time {
            break;
        }

        let weight = kernel_weight(running_var[i] - cutoff, bandwidth, kernel);

        if event[i] == 1 && at_risk > crate::constants::DIVISION_FLOOR {
            survival *= 1.0 - weight / at_risk;
        }

        at_risk -= weight;
    }

    survival.clamp(0.0, 1.0)
}
