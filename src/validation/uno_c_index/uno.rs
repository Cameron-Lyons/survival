
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]

pub struct UnoCIndexResult {
    pub c_index: f64,
    pub concordant: f64,
    pub discordant: f64,
    pub tied_risk: f64,
    pub comparable_pairs: f64,
    pub variance: f64,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub tau: f64,
}

impl fmt::Display for UnoCIndexResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnoCIndexResult(c_index={:.4}, se={:.4}, ci=[{:.4}, {:.4}])",
            self.c_index, self.std_error, self.ci_lower, self.ci_upper
        )
    }
}

#[pymethods]
impl UnoCIndexResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        c_index: f64,
        concordant: f64,
        discordant: f64,
        tied_risk: f64,
        comparable_pairs: f64,
        variance: f64,
        std_error: f64,
        ci_lower: f64,
        ci_upper: f64,
        tau: f64,
    ) -> Self {
        Self {
            c_index,
            concordant,
            discordant,
            tied_risk,
            comparable_pairs,
            variance,
            std_error,
            ci_lower,
            ci_upper,
            tau,
        }
    }
}

struct UnoCIndexAccumulator {
    concordant: f64,
    discordant: f64,
    tied: f64,
    total_weight: f64,
    influence_sums: Vec<f64>,
}

impl UnoCIndexAccumulator {
    fn new(n: usize) -> Self {
        Self {
            concordant: 0.0,
            discordant: 0.0,
            tied: 0.0,
            total_weight: 0.0,
            influence_sums: vec![0.0; n],
        }
    }

    fn merge(&mut self, other: Self) {
        self.concordant += other.concordant;
        self.discordant += other.discordant;
        self.tied += other.tied;
        self.total_weight += other.total_weight;
        for (left, right) in self
            .influence_sums
            .iter_mut()
            .zip(other.influence_sums.iter())
        {
            *left += right;
        }
    }
}

struct UnoCIndexContext<'a> {
    time: &'a [f64],
    status: &'a [i32],
    risk_score: &'a [f64],
    km_times: &'a [f64],
    km_values: &'a [f64],
    tau: f64,
    min_g: f64,
}

fn accumulate_uno_c_index_row(
    accumulator: &mut UnoCIndexAccumulator,
    context: &UnoCIndexContext<'_>,
    i: usize,
) {
    if context.status[i] != 1 || !at_or_before_tau(context.time[i], context.tau) {
        return;
    }

    let g_ti = km_step_prob_at(context.time[i], context.km_times, context.km_values).max(context.min_g);
    let weight = 1.0 / (g_ti * g_ti);

    for j in 0..context.time.len() {
        if i == j {
            continue;
        }

        if !after_event_time(context.time[j], context.time[i]) {
            continue;
        }

        accumulator.total_weight += weight;

        if context.risk_score[i] > context.risk_score[j] {
            accumulator.concordant += weight;
            accumulator.influence_sums[i] += weight;
            accumulator.influence_sums[j] -= weight;
        } else if context.risk_score[i] < context.risk_score[j] {
            accumulator.discordant += weight;
            accumulator.influence_sums[i] -= weight;
            accumulator.influence_sums[j] += weight;
        } else {
            accumulator.tied += weight;
        }
    }
}

pub(crate) fn uno_c_index_core(
    time: &[f64],
    status: &[i32],
    risk_score: &[f64],
    tau: Option<f64>,
) -> UnoCIndexResult {
    let n = time.len();

    if n == 0 {
        return UnoCIndexResult {
            c_index: 0.5,
            concordant: 0.0,
            discordant: 0.0,
            tied_risk: 0.0,
            comparable_pairs: 0.0,
            variance: 0.0,
            std_error: 0.0,
            ci_lower: 0.5,
            ci_upper: 0.5,
            tau: 0.0,
        };
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let (km_times, km_values) = compute_censoring_km(time, status);

    let min_g = IPCW_SURVIVAL_FLOOR;

    let context = UnoCIndexContext {
        time,
        status,
        risk_score,
        km_times: &km_times,
        km_values: &km_values,
        tau: tau_val,
        min_g,
    };

    let accumulator = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n)
            .into_par_iter()
            .fold(
                || UnoCIndexAccumulator::new(n),
                |mut accumulator, i| {
                    accumulate_uno_c_index_row(&mut accumulator, &context, i);
                    accumulator
                },
            )
            .reduce(
                || UnoCIndexAccumulator::new(n),
                |mut left, right| {
                    left.merge(right);
                    left
                },
            )
    } else {
        let mut accumulator = UnoCIndexAccumulator::new(n);
        for i in 0..n {
            accumulate_uno_c_index_row(&mut accumulator, &context, i);
        }
        accumulator
    };

    let c_index = if accumulator.total_weight > 0.0 {
        (accumulator.concordant + 0.5 * accumulator.tied) / accumulator.total_weight
    } else {
        0.5
    };

    let variance = if accumulator.total_weight > 0.0 {
        let n_f = n as f64;
        let mut var_sum = 0.0;

        for &inf in &accumulator.influence_sums {
            let normalized_inf = inf / accumulator.total_weight;
            var_sum += normalized_inf * normalized_inf;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error = variance.sqrt();
    let (ci_lower, ci_upper) = clamped_normal_ci_95(c_index, std_error, 0.0, 1.0);

    UnoCIndexResult {
        c_index,
        concordant: accumulator.concordant,
        discordant: accumulator.discordant,
        tied_risk: accumulator.tied,
        comparable_pairs: accumulator.total_weight,
        variance,
        std_error,
        ci_lower,
        ci_upper,
        tau: tau_val,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_score, tau=None))]
pub fn uno_c_index(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_score: Vec<f64>,
    tau: Option<f64>,
) -> PyResult<UnoCIndexResult> {
    if time.len() != status.len() || time.len() != risk_score.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and risk_score must have the same length",
        ));
    }
    validate_uno_time_status(&time, &status)?;
    validate_uno_risk_score(&risk_score, "risk_score")?;
    validate_uno_tau(tau)?;

    Ok(uno_c_index_core(&time, &status, &risk_score, tau))
}
