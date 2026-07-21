
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
}

#[inline]
fn uno_rank_counts(tree: &FenwickTree, rank: usize) -> (f64, f64, f64) {
    let below = if rank == 0 {
        0.0
    } else {
        tree.prefix_sum(rank - 1)
    };
    let inclusive = tree.prefix_sum(rank);
    (below, inclusive - below, tree.total() - inclusive)
}

fn uno_risk_ranks(risk_score: &[f64]) -> (usize, Vec<usize>) {
    let mut levels = risk_score.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|left, right| *left == *right);
    let ranks = risk_score
        .iter()
        .map(|&score| levels.partition_point(|&level| level < score))
        .collect();
    (levels.len(), ranks)
}

fn uno_event_weights(
    time: &[f64],
    status: &[i32],
    km_times: &[f64],
    km_values: &[f64],
    tau: f64,
) -> Vec<f64> {
    time.iter()
        .enumerate()
        .map(|(idx, &event_time)| {
            if status[idx] != 1 || !at_or_before_tau(event_time, tau) {
                return 0.0;
            }
            let censoring_survival =
                km_step_prob_at(event_time, km_times, km_values).max(IPCW_SURVIVAL_FLOOR);
            1.0 / (censoring_survival * censoring_survival)
        })
        .collect()
}

fn uno_c_index_ranked_accumulator(
    time: &[f64],
    risk_score: &[f64],
    event_weights: &[f64],
) -> UnoCIndexAccumulator {
    let n = time.len();
    let (n_ranks, risk_ranks) = uno_risk_ranks(risk_score);
    let mut subjects_desc: Vec<usize> = (0..n).collect();
    subjects_desc.sort_by(|&left, &right| {
        time[right]
            .total_cmp(&time[left])
            .then_with(|| left.cmp(&right))
    });
    let mut events_desc: Vec<usize> = (0..n)
        .filter(|&idx| event_weights[idx] > 0.0)
        .collect();
    events_desc.sort_by(|&left, &right| {
        time[right]
            .total_cmp(&time[left])
            .then_with(|| left.cmp(&right))
    });

    // Sweep backward through event times, adding only subjects far enough in
    // the future to satisfy the package's epsilon-aware comparability rule.
    let mut accumulator = {
        let mut later_risk_counts = FenwickTree::new(n_ranks);
        let mut subject_cursor = 0;
        let mut pair_counts = vec![(0.0, 0.0, 0.0); n];
        for &event_idx in &events_desc {
            while subject_cursor < n
                && after_event_time(
                    time[subjects_desc[subject_cursor]],
                    time[event_idx],
                )
            {
                let subject_idx = subjects_desc[subject_cursor];
                later_risk_counts.update(risk_ranks[subject_idx], 1.0);
                subject_cursor += 1;
            }
            pair_counts[event_idx] = uno_rank_counts(&later_risk_counts, risk_ranks[event_idx]);
        }

        // Preserve the old row-order accumulation for stable floating-point
        // behavior while replacing each row's pair scan with its rank counts.
        let mut accumulator = UnoCIndexAccumulator::new(n);
        for idx in 0..n {
            let weight = event_weights[idx];
            if weight == 0.0 {
                continue;
            }
            let (lower, tied, higher) = pair_counts[idx];
            accumulator.concordant += weight * lower;
            accumulator.discordant += weight * higher;
            accumulator.tied += weight * tied;
            accumulator.total_weight += weight * (lower + tied + higher);
            accumulator.influence_sums[idx] += weight * (lower - higher);
        }
        accumulator
    };

    // Sweep forward to recover every later subject's signed influence from
    // the IPCW weights of earlier comparable events.
    subjects_desc.reverse();
    events_desc.reverse();
    let mut earlier_event_weights = FenwickTree::new(n_ranks);
    let mut event_cursor = 0;
    for &subject_idx in &subjects_desc {
        while event_cursor < events_desc.len()
            && after_event_time(time[subject_idx], time[events_desc[event_cursor]])
        {
            let event_idx = events_desc[event_cursor];
            earlier_event_weights.update(risk_ranks[event_idx], event_weights[event_idx]);
            event_cursor += 1;
        }
        let (lower, _tied, higher) =
            uno_rank_counts(&earlier_event_weights, risk_ranks[subject_idx]);
        accumulator.influence_sums[subject_idx] += lower - higher;
    }

    accumulator
}

#[cfg(test)]
fn uno_c_index_quadratic_accumulator(
    time: &[f64],
    risk_score: &[f64],
    event_weights: &[f64],
) -> UnoCIndexAccumulator {
    let mut accumulator = UnoCIndexAccumulator::new(time.len());
    for i in 0..time.len() {
        let weight = event_weights[i];
        if weight == 0.0 {
            continue;
        }
        for j in 0..time.len() {
            if i == j || !after_event_time(time[j], time[i]) {
                continue;
            }
            accumulator.total_weight += weight;
            if risk_score[i] > risk_score[j] {
                accumulator.concordant += weight;
                accumulator.influence_sums[i] += weight;
                accumulator.influence_sums[j] -= weight;
            } else if risk_score[i] < risk_score[j] {
                accumulator.discordant += weight;
                accumulator.influence_sums[i] -= weight;
                accumulator.influence_sums[j] += weight;
            } else {
                accumulator.tied += weight;
            }
        }
    }
    accumulator
}

fn uno_c_index_result(
    accumulator: UnoCIndexAccumulator,
    n: usize,
    tau: f64,
) -> UnoCIndexResult {
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
        tau,
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
        return uno_c_index_result(UnoCIndexAccumulator::new(0), 0, 0.0);
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let (km_times, km_values) = compute_censoring_km(time, status);
    let event_weights = uno_event_weights(time, status, &km_times, &km_values, tau_val);
    uno_c_index_result(
        uno_c_index_ranked_accumulator(time, risk_score, &event_weights),
        n,
        tau_val,
    )
}

#[cfg(test)]
fn uno_c_index_core_quadratic(
    time: &[f64],
    status: &[i32],
    risk_score: &[f64],
    tau: Option<f64>,
) -> UnoCIndexResult {
    let n = time.len();
    if n == 0 {
        return uno_c_index_result(UnoCIndexAccumulator::new(0), 0, 0.0);
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let (km_times, km_values) = compute_censoring_km(time, status);
    let event_weights = uno_event_weights(time, status, &km_times, &km_values, tau_val);
    uno_c_index_result(
        uno_c_index_quadratic_accumulator(time, risk_score, &event_weights),
        n,
        tau_val,
    )
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
