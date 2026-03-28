
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ConcordanceComparisonResult {
    #[pyo3(get)]
    pub c_index_1: f64,
    #[pyo3(get)]
    pub c_index_2: f64,
    #[pyo3(get)]
    pub difference: f64,
    #[pyo3(get)]
    pub variance_diff: f64,
    #[pyo3(get)]
    pub std_error_diff: f64,
    #[pyo3(get)]
    pub z_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
}

#[pymethods]
impl ConcordanceComparisonResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        c_index_1: f64,
        c_index_2: f64,
        difference: f64,
        variance_diff: f64,
        std_error_diff: f64,
        z_statistic: f64,
        p_value: f64,
        ci_lower: f64,
        ci_upper: f64,
    ) -> Self {
        Self {
            c_index_1,
            c_index_2,
            difference,
            variance_diff,
            std_error_diff,
            z_statistic,
            p_value,
            ci_lower,
            ci_upper,
        }
    }
}

pub(crate) fn compare_uno_c_indices_core(
    time: &[f64],
    status: &[i32],
    risk_score_1: &[f64],
    risk_score_2: &[f64],
    tau: Option<f64>,
) -> ConcordanceComparisonResult {
    let n = time.len();

    if n == 0 {
        return ConcordanceComparisonResult {
            c_index_1: 0.5,
            c_index_2: 0.5,
            difference: 0.0,
            variance_diff: 0.0,
            std_error_diff: 0.0,
            z_statistic: 0.0,
            p_value: 1.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
        };
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let (km_times, km_values) = compute_censoring_km(time, status);

    let min_g = IPCW_SURVIVAL_FLOOR;

    let mut concordant_1 = 0.0;
    let mut concordant_2 = 0.0;
    let mut total_pairs = 0.0;

    let mut influence_1 = vec![0.0; n];
    let mut influence_2 = vec![0.0; n];

    for i in 0..n {
        if status[i] != 1 || time[i] > tau_val {
            continue;
        }

        let g_ti = km_step_prob_at(time[i], &km_times, &km_values).max(min_g);
        let weight = 1.0 / (g_ti * g_ti);

        for j in 0..n {
            if i == j || time[j] <= time[i] {
                continue;
            }

            total_pairs += weight;

            let contrib_1 = if risk_score_1[i] > risk_score_1[j] {
                weight
            } else if risk_score_1[i] < risk_score_1[j] {
                0.0
            } else {
                0.5 * weight
            };

            let contrib_2 = if risk_score_2[i] > risk_score_2[j] {
                weight
            } else if risk_score_2[i] < risk_score_2[j] {
                0.0
            } else {
                0.5 * weight
            };

            concordant_1 += contrib_1;
            concordant_2 += contrib_2;

            influence_1[i] += contrib_1;
            influence_1[j] -= contrib_1;
            influence_2[i] += contrib_2;
            influence_2[j] -= contrib_2;
        }
    }

    let c_index_1 = if total_pairs > 0.0 {
        concordant_1 / total_pairs
    } else {
        0.5
    };

    let c_index_2 = if total_pairs > 0.0 {
        concordant_2 / total_pairs
    } else {
        0.5
    };

    let difference = c_index_1 - c_index_2;

    let variance_diff = if total_pairs > 0.0 {
        let n_f = n as f64;
        let mut var_sum = 0.0;

        for k in 0..n {
            let diff_inf = (influence_1[k] - influence_2[k]) / total_pairs;
            var_sum += diff_inf * diff_inf;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error_diff = variance_diff.sqrt();

    let z_statistic = if std_error_diff > 1e-10 {
        difference / std_error_diff
    } else {
        0.0
    };

    let p_value = 2.0 * (1.0 - normal_cdf(z_statistic.abs()));

    let z = 1.96;
    let ci_lower = difference - z * std_error_diff;
    let ci_upper = difference + z * std_error_diff;

    ConcordanceComparisonResult {
        c_index_1,
        c_index_2,
        difference,
        variance_diff,
        std_error_diff,
        z_statistic,
        p_value,
        ci_lower,
        ci_upper,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_score_1, risk_score_2, tau=None))]
pub fn compare_uno_c_indices(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_score_1: Vec<f64>,
    risk_score_2: Vec<f64>,
    tau: Option<f64>,
) -> PyResult<ConcordanceComparisonResult> {
    let n = time.len();
    if n != status.len() || n != risk_score_1.len() || n != risk_score_2.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    Ok(compare_uno_c_indices_core(
        &time,
        &status,
        &risk_score_1,
        &risk_score_2,
        tau,
    ))
}
