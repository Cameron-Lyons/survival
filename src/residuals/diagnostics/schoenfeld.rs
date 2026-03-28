
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SchoenfeldSmoothResult {
    #[pyo3(get)]
    pub times: Vec<f64>,
    #[pyo3(get)]
    pub smoothed_residuals: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub coefficient_path: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub slope_test_stats: Vec<f64>,
    #[pyo3(get)]
    pub slope_p_values: Vec<f64>,
    #[pyo3(get)]
    pub non_proportional_vars: Vec<usize>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pyfunction]
#[pyo3(signature = (event_times, schoenfeld_residuals, n_covariates, coefficients, bandwidth=None, transform="identity"))]
pub fn smooth_schoenfeld(
    event_times: Vec<f64>,
    schoenfeld_residuals: Vec<f64>,
    n_covariates: usize,
    coefficients: Vec<f64>,
    bandwidth: Option<f64>,
    transform: &str,
) -> PyResult<SchoenfeldSmoothResult> {
    let n_events = event_times.len();
    if schoenfeld_residuals.len() != n_events * n_covariates {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "schoenfeld_residuals must have length n_events * n_covariates",
        ));
    }

    let transformed_times: Vec<f64> = match transform.to_lowercase().as_str() {
        "identity" => event_times.clone(),
        "log" => event_times.iter().map(|&t| (t.max(1e-10)).ln()).collect(),
        "km" => {
            let mut km = vec![0.0; n_events];
            let mut n_risk = n_events as f64;
            for km_i in km.iter_mut().take(n_events) {
                *km_i = 1.0 - 1.0 / n_risk;
                n_risk -= 1.0;
            }
            km
        }
        "rank" => {
            let ranks: Vec<f64> = (1..=n_events)
                .map(|i| i as f64 / (n_events as f64 + 1.0))
                .collect();
            ranks
        }
        _ => event_times.clone(),
    };

    let h = bandwidth.unwrap_or_else(|| {
        let time_range = transformed_times
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - transformed_times
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        0.2 * time_range
    });

    let mut smoothed: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n_events];
    let mut coefficient_path: Vec<Vec<f64>> = vec![vec![0.0; n_covariates]; n_events];

    for i in 0..n_events {
        let t_i = transformed_times[i];

        for j in 0..n_covariates {
            let mut sum_w = 0.0;
            let mut sum_wy = 0.0;
            let mut _sum_wty = 0.0;
            let mut _sum_wt = 0.0;
            let mut _sum_wtt = 0.0;

            for k in 0..n_events {
                let t_k = transformed_times[k];
                let diff = (t_i - t_k) / h;
                let w = (-0.5 * diff * diff).exp();

                let y = schoenfeld_residuals[k * n_covariates + j];

                sum_w += w;
                sum_wy += w * y;
                _sum_wty += w * t_k * y;
                _sum_wt += w * t_k;
                _sum_wtt += w * t_k * t_k;
            }

            if sum_w > 1e-10 {
                let y_mean = sum_wy / sum_w;
                smoothed[i][j] = y_mean;

                coefficient_path[i][j] = coefficients[j] + y_mean;
            }
        }
    }

    let mut slope_test_stats = vec![0.0; n_covariates];
    let mut slope_p_values = vec![0.0; n_covariates];

    for j in 0..n_covariates {
        let (slope, se_slope) =
            compute_slope_test(&transformed_times, &schoenfeld_residuals, j, n_covariates);

        if se_slope > 1e-10 {
            let z = slope / se_slope;
            slope_test_stats[j] = z;
            slope_p_values[j] = 2.0 * (1.0 - normal_cdf(z.abs()));
        }
    }

    let non_proportional_vars: Vec<usize> = (0..n_covariates)
        .filter(|&j| slope_p_values[j] < 0.05)
        .collect();

    Ok(SchoenfeldSmoothResult {
        times: transformed_times,
        smoothed_residuals: smoothed,
        coefficient_path,
        slope_test_stats,
        slope_p_values,
        non_proportional_vars,
        n_events,
        n_vars: n_covariates,
    })
}

fn compute_slope_test(
    times: &[f64],
    residuals: &[f64],
    var_idx: usize,
    n_covariates: usize,
) -> (f64, f64) {
    let n = times.len();

    let mean_t: f64 = times.iter().sum::<f64>() / n as f64;
    let mean_r: f64 = (0..n)
        .map(|i| residuals[i * n_covariates + var_idx])
        .sum::<f64>()
        / n as f64;

    let mut sum_tt = 0.0;
    let mut sum_tr = 0.0;
    let mut sum_rr = 0.0;

    for i in 0..n {
        let t_diff = times[i] - mean_t;
        let r_diff = residuals[i * n_covariates + var_idx] - mean_r;
        sum_tt += t_diff * t_diff;
        sum_tr += t_diff * r_diff;
        sum_rr += r_diff * r_diff;
    }

    let slope = if sum_tt > 1e-10 { sum_tr / sum_tt } else { 0.0 };

    let residual_var = if n > 2 && sum_tt > 1e-10 {
        (sum_rr - slope * slope * sum_tt) / (n - 2) as f64
    } else {
        0.0
    };

    let se_slope = if sum_tt > 1e-10 {
        (residual_var / sum_tt).sqrt()
    } else {
        0.0
    };

    (slope, se_slope)
}
