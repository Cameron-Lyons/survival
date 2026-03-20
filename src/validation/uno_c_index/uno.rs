
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

    let compute_pair_contributions = |i: usize| -> (f64, f64, f64, f64, Vec<f64>) {
        let mut concordant = 0.0;
        let mut discordant = 0.0;
        let mut tied = 0.0;
        let mut total_weight = 0.0;
        let mut influence = vec![0.0; n];

        if status[i] != 1 || time[i] > tau_val {
            return (concordant, discordant, tied, total_weight, influence);
        }

        let g_ti = km_step_prob_at(time[i], &km_times, &km_values).max(min_g);
        let weight = 1.0 / (g_ti * g_ti);

        for j in 0..n {
            if i == j {
                continue;
            }

            if time[j] <= time[i] {
                continue;
            }

            total_weight += weight;

            if risk_score[i] > risk_score[j] {
                concordant += weight;
                influence[i] += weight;
                influence[j] -= weight;
            } else if risk_score[i] < risk_score[j] {
                discordant += weight;
                influence[i] -= weight;
                influence[j] += weight;
            } else {
                tied += weight;
            }
        }

        (concordant, discordant, tied, total_weight, influence)
    };

    let results: Vec<(f64, f64, f64, f64, Vec<f64>)> = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n)
            .into_par_iter()
            .map(compute_pair_contributions)
            .collect()
    } else {
        (0..n).map(compute_pair_contributions).collect()
    };

    let mut total_concordant = 0.0;
    let mut total_discordant = 0.0;
    let mut total_tied = 0.0;
    let mut total_pairs = 0.0;
    let mut influence_sums = vec![0.0; n];

    for (concordant, discordant, tied, pairs, influence) in results {
        total_concordant += concordant;
        total_discordant += discordant;
        total_tied += tied;
        total_pairs += pairs;
        for (k, &inf) in influence.iter().enumerate() {
            influence_sums[k] += inf;
        }
    }

    let c_index = if total_pairs > 0.0 {
        (total_concordant + 0.5 * total_tied) / total_pairs
    } else {
        0.5
    };

    let variance = if total_pairs > 0.0 {
        let n_f = n as f64;
        let mut var_sum = 0.0;

        for &inf in &influence_sums {
            let normalized_inf = inf / total_pairs;
            var_sum += normalized_inf * normalized_inf;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error = variance.sqrt();
    let z = 1.96;
    let ci_lower = (c_index - z * std_error).clamp(0.0, 1.0);
    let ci_upper = (c_index + z * std_error).clamp(0.0, 1.0);

    UnoCIndexResult {
        c_index,
        concordant: total_concordant,
        discordant: total_discordant,
        tied_risk: total_tied,
        comparable_pairs: total_pairs,
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

    Ok(uno_c_index_core(&time, &status, &risk_score, tau))
}
