
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

    let mut concordant_ee = 0.0;
    let mut concordant_ec = 0.0;
    let mut discordant_ee = 0.0;
    let mut discordant_ec = 0.0;
    let mut tied_ee = 0.0;
    let mut tied_ec = 0.0;
    let mut n_ee_pairs = 0usize;
    let mut n_ec_pairs = 0usize;

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

            let is_event_event = status[j] == 1 && time[j] <= tau_val;

            if is_event_event {
                n_ee_pairs += 1;
                if risk_score[i] > risk_score[j] {
                    concordant_ee += weight;
                } else if risk_score[i] < risk_score[j] {
                    discordant_ee += weight;
                } else {
                    tied_ee += weight;
                }
            } else {
                n_ec_pairs += 1;
                if risk_score[i] > risk_score[j] {
                    concordant_ec += weight;
                } else if risk_score[i] < risk_score[j] {
                    discordant_ec += weight;
                } else {
                    tied_ec += weight;
                }
            }
        }
    }

    let total_ee = concordant_ee + discordant_ee + tied_ee;
    let total_ec = concordant_ec + discordant_ec + tied_ec;
    let total_pairs = total_ee + total_ec;

    let c_index_ee = if total_ee > 0.0 {
        (concordant_ee + 0.5 * tied_ee) / total_ee
    } else {
        0.5
    };

    let c_index_ec = if total_ec > 0.0 {
        (concordant_ec + 0.5 * tied_ec) / total_ec
    } else {
        0.5
    };

    let c_index = if total_pairs > 0.0 {
        (concordant_ee + concordant_ec + 0.5 * (tied_ee + tied_ec)) / total_pairs
    } else {
        0.5
    };

    let correctly_ordered_ee = concordant_ee + 0.5 * tied_ee;
    let correctly_ordered_ec = concordant_ec + 0.5 * tied_ec;
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
        n_event_event_pairs: n_ee_pairs,
        n_event_censored_pairs: n_ec_pairs,
        concordant_ee,
        concordant_ec,
        discordant_ee,
        discordant_ec,
        tied_ee,
        tied_ec,
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

    Ok(c_index_decomposition_core(&time, &status, &risk_score, tau))
}
