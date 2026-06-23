
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct CIndexDecompositionResult {
    pub c_index: f64,
    pub c_index_ee: f64,
    pub c_index_ec: f64,
    pub alpha: f64,
    pub n_event_event_pairs: usize,
    pub n_event_censored_pairs: usize,
    pub concordant_ee: f64,
    pub concordant_ec: f64,
    pub discordant_ee: f64,
    pub discordant_ec: f64,
    pub tied_ee: f64,
    pub tied_ec: f64,
}

impl fmt::Display for CIndexDecompositionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CIndexDecomposition(C={:.4}, C_ee={:.4}, C_ec={:.4}, alpha={:.4})",
            self.c_index, self.c_index_ee, self.c_index_ec, self.alpha
        )
    }
}

#[pymethods]
impl CIndexDecompositionResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        c_index: f64,
        c_index_ee: f64,
        c_index_ec: f64,
        alpha: f64,
        n_event_event_pairs: usize,
        n_event_censored_pairs: usize,
        concordant_ee: f64,
        concordant_ec: f64,
        discordant_ee: f64,
        discordant_ec: f64,
        tied_ee: f64,
        tied_ec: f64,
    ) -> Self {
        Self {
            c_index,
            c_index_ee,
            c_index_ec,
            alpha,
            n_event_event_pairs,
            n_event_censored_pairs,
            concordant_ee,
            concordant_ec,
            discordant_ee,
            discordant_ec,
            tied_ee,
            tied_ec,
        }
    }
}

struct CIndexDecompositionAccumulator {
    concordant_ee: f64,
    concordant_ec: f64,
    discordant_ee: f64,
    discordant_ec: f64,
    tied_ee: f64,
    tied_ec: f64,
    n_ee_pairs: usize,
    n_ec_pairs: usize,
}

impl CIndexDecompositionAccumulator {
    fn new() -> Self {
        Self {
            concordant_ee: 0.0,
            concordant_ec: 0.0,
            discordant_ee: 0.0,
            discordant_ec: 0.0,
            tied_ee: 0.0,
            tied_ec: 0.0,
            n_ee_pairs: 0,
            n_ec_pairs: 0,
        }
    }

    fn merge(&mut self, other: Self) {
        self.concordant_ee += other.concordant_ee;
        self.concordant_ec += other.concordant_ec;
        self.discordant_ee += other.discordant_ee;
        self.discordant_ec += other.discordant_ec;
        self.tied_ee += other.tied_ee;
        self.tied_ec += other.tied_ec;
        self.n_ee_pairs += other.n_ee_pairs;
        self.n_ec_pairs += other.n_ec_pairs;
    }
}

struct CIndexDecompositionContext<'a> {
    time: &'a [f64],
    status: &'a [i32],
    risk_score: &'a [f64],
    km_times: &'a [f64],
    km_values: &'a [f64],
    tau: f64,
    min_g: f64,
}

fn accumulate_c_index_decomposition_row(
    accumulator: &mut CIndexDecompositionAccumulator,
    context: &CIndexDecompositionContext<'_>,
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

        let is_event_event = context.status[j] == 1 && at_or_before_tau(context.time[j], context.tau);

        if is_event_event {
            accumulator.n_ee_pairs += 1;
            if context.risk_score[i] > context.risk_score[j] {
                accumulator.concordant_ee += weight;
            } else if context.risk_score[i] < context.risk_score[j] {
                accumulator.discordant_ee += weight;
            } else {
                accumulator.tied_ee += weight;
            }
        } else {
            accumulator.n_ec_pairs += 1;
            if context.risk_score[i] > context.risk_score[j] {
                accumulator.concordant_ec += weight;
            } else if context.risk_score[i] < context.risk_score[j] {
                accumulator.discordant_ec += weight;
            } else {
                accumulator.tied_ec += weight;
            }
        }
    }
}

pub(crate) fn c_index_decomposition_core(
    time: &[f64],
    status: &[i32],
    risk_score: &[f64],
    tau: Option<f64>,
) -> CIndexDecompositionResult {
    let n = time.len();

    if n == 0 {
        return CIndexDecompositionResult {
            c_index: 0.5,
            c_index_ee: 0.5,
            c_index_ec: 0.5,
            alpha: 0.5,
            n_event_event_pairs: 0,
            n_event_censored_pairs: 0,
            concordant_ee: 0.0,
            concordant_ec: 0.0,
            discordant_ee: 0.0,
            discordant_ec: 0.0,
            tied_ee: 0.0,
            tied_ec: 0.0,
        };
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));

    let (km_times, km_values) = compute_censoring_km(time, status);
    let min_g = IPCW_SURVIVAL_FLOOR;

    let context = CIndexDecompositionContext {
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
            .fold(CIndexDecompositionAccumulator::new, |mut accumulator, i| {
                accumulate_c_index_decomposition_row(&mut accumulator, &context, i);
                accumulator
            })
            .reduce(CIndexDecompositionAccumulator::new, |mut left, right| {
                left.merge(right);
                left
            })
    } else {
        let mut accumulator = CIndexDecompositionAccumulator::new();
        for i in 0..n {
            accumulate_c_index_decomposition_row(&mut accumulator, &context, i);
        }
        accumulator
    };

    let total_ee = accumulator.concordant_ee + accumulator.discordant_ee + accumulator.tied_ee;
    let total_ec = accumulator.concordant_ec + accumulator.discordant_ec + accumulator.tied_ec;
    let total_pairs = total_ee + total_ec;

    let c_index_ee = if total_ee > 0.0 {
        (accumulator.concordant_ee + 0.5 * accumulator.tied_ee) / total_ee
    } else {
        0.5
    };

    let c_index_ec = if total_ec > 0.0 {
        (accumulator.concordant_ec + 0.5 * accumulator.tied_ec) / total_ec
    } else {
        0.5
    };

    let c_index = if total_pairs > 0.0 {
        (accumulator.concordant_ee
            + accumulator.concordant_ec
            + 0.5 * (accumulator.tied_ee + accumulator.tied_ec))
            / total_pairs
    } else {
        0.5
    };

    let correctly_ordered_ee = accumulator.concordant_ee + 0.5 * accumulator.tied_ee;
    let correctly_ordered_ec = accumulator.concordant_ec + 0.5 * accumulator.tied_ec;
    let total_correctly_ordered = correctly_ordered_ee + correctly_ordered_ec;

    let alpha = if total_correctly_ordered > 0.0 {
        correctly_ordered_ee / total_correctly_ordered
    } else {
        0.5
    };

    CIndexDecompositionResult {
        c_index,
        c_index_ee,
        c_index_ec,
        alpha,
        n_event_event_pairs: accumulator.n_ee_pairs,
        n_event_censored_pairs: accumulator.n_ec_pairs,
        concordant_ee: accumulator.concordant_ee,
        concordant_ec: accumulator.concordant_ec,
        discordant_ee: accumulator.discordant_ee,
        discordant_ec: accumulator.discordant_ec,
        tied_ee: accumulator.tied_ee,
        tied_ec: accumulator.tied_ec,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, risk_score, tau=None))]
pub fn c_index_decomposition(
    time: Vec<f64>,
    status: Vec<i32>,
    risk_score: Vec<f64>,
    tau: Option<f64>,
) -> PyResult<CIndexDecompositionResult> {
    if time.len() != status.len() || time.len() != risk_score.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and risk_score must have the same length",
        ));
    }
    validate_uno_time_status(&time, &status)?;
    validate_uno_risk_score(&risk_score, "risk_score")?;
    validate_uno_tau(tau)?;

    Ok(c_index_decomposition_core(&time, &status, &risk_score, tau))
}
