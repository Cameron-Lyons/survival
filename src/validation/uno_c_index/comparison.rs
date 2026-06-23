
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

struct UnoComparisonAccumulator {
    concordant_1: f64,
    concordant_2: f64,
    total_pairs: f64,
    influence_1: Vec<f64>,
    influence_2: Vec<f64>,
}

impl UnoComparisonAccumulator {
    fn new(n: usize) -> Self {
        Self {
            concordant_1: 0.0,
            concordant_2: 0.0,
            total_pairs: 0.0,
            influence_1: vec![0.0; n],
            influence_2: vec![0.0; n],
        }
    }

    fn merge(&mut self, other: Self) {
        self.concordant_1 += other.concordant_1;
        self.concordant_2 += other.concordant_2;
        self.total_pairs += other.total_pairs;
        for (left, right) in self.influence_1.iter_mut().zip(other.influence_1.iter()) {
            *left += right;
        }
        for (left, right) in self.influence_2.iter_mut().zip(other.influence_2.iter()) {
            *left += right;
        }
    }
}

struct UnoComparisonContext<'a> {
    time: &'a [f64],
    status: &'a [i32],
    risk_score_1: &'a [f64],
    risk_score_2: &'a [f64],
    km_times: &'a [f64],
    km_values: &'a [f64],
    tau: f64,
    min_g: f64,
}

fn accumulate_uno_comparison_row(
    accumulator: &mut UnoComparisonAccumulator,
    context: &UnoComparisonContext<'_>,
    i: usize,
) {
    if context.status[i] != 1 || !at_or_before_tau(context.time[i], context.tau) {
        return;
    }

    let g_ti = km_step_prob_at(context.time[i], context.km_times, context.km_values)
        .max(context.min_g);
    let weight = 1.0 / (g_ti * g_ti);

    for j in 0..context.time.len() {
        if i == j || !after_event_time(context.time[j], context.time[i]) {
            continue;
        }

        accumulator.total_pairs += weight;

        let contrib_1 = if context.risk_score_1[i] > context.risk_score_1[j] {
            weight
        } else if context.risk_score_1[i] < context.risk_score_1[j] {
            0.0
        } else {
            0.5 * weight
        };

        let contrib_2 = if context.risk_score_2[i] > context.risk_score_2[j] {
            weight
        } else if context.risk_score_2[i] < context.risk_score_2[j] {
            0.0
        } else {
            0.5 * weight
        };

        accumulator.concordant_1 += contrib_1;
        accumulator.concordant_2 += contrib_2;

        accumulator.influence_1[i] += contrib_1;
        accumulator.influence_1[j] -= contrib_1;
        accumulator.influence_2[i] += contrib_2;
        accumulator.influence_2[j] -= contrib_2;
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

    let context = UnoComparisonContext {
        time,
        status,
        risk_score_1,
        risk_score_2,
        km_times: &km_times,
        km_values: &km_values,
        tau: tau_val,
        min_g,
    };

    let accumulator = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n)
            .into_par_iter()
            .fold(
                || UnoComparisonAccumulator::new(n),
                |mut accumulator, i| {
                    accumulate_uno_comparison_row(&mut accumulator, &context, i);
                    accumulator
                },
            )
            .reduce(
                || UnoComparisonAccumulator::new(n),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            )
    } else {
        let mut accumulator = UnoComparisonAccumulator::new(n);
        for i in 0..n {
            accumulate_uno_comparison_row(&mut accumulator, &context, i);
        }
        accumulator
    };

    let c_index_1 = if accumulator.total_pairs > 0.0 {
        accumulator.concordant_1 / accumulator.total_pairs
    } else {
        0.5
    };

    let c_index_2 = if accumulator.total_pairs > 0.0 {
        accumulator.concordant_2 / accumulator.total_pairs
    } else {
        0.5
    };

    let difference = c_index_1 - c_index_2;

    let variance_diff = if accumulator.total_pairs > 0.0 {
        let n_f = n as f64;
        let mut var_sum = 0.0;

        for k in 0..n {
            let diff_inf =
                (accumulator.influence_1[k] - accumulator.influence_2[k]) / accumulator.total_pairs;
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

    let (ci_lower, ci_upper) = normal_ci_95(difference, std_error_diff);

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
    validate_uno_time_status(&time, &status)?;
    validate_uno_risk_score(&risk_score_1, "risk_score_1")?;
    validate_uno_risk_score(&risk_score_2, "risk_score_2")?;
    validate_uno_tau(tau)?;

    Ok(compare_uno_c_indices_core(
        &time,
        &status,
        &risk_score_1,
        &risk_score_2,
        tau,
    ))
}
