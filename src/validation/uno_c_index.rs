use crate::constants::PARALLEL_THRESHOLD_LARGE;
use crate::utilities::statistical::normal_cdf;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
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

fn compute_censoring_km(time: &[f64], status: &[i32]) -> (Vec<f64>, Vec<f64>) {
    let n = time.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut unique_times = Vec::new();
    let mut km_values = Vec::new();
    let mut cum_surv = 1.0;
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut censored_count = 0;
        let mut total_at_time = 0;

        let _start_i = i;
        while i < n && (time[indices[i]] - current_time).abs() < 1e-10 {
            if status[indices[i]] == 0 {
                censored_count += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        if censored_count > 0 && at_risk > 0 {
            cum_surv *= 1.0 - censored_count as f64 / at_risk as f64;
        }

        unique_times.push(current_time);
        km_values.push(cum_surv);

        at_risk -= total_at_time;
    }

    (unique_times, km_values)
}

fn get_censoring_prob(t: f64, unique_times: &[f64], km_values: &[f64]) -> f64 {
    if unique_times.is_empty() {
        return 1.0;
    }

    if t < unique_times[0] {
        return 1.0;
    }

    let mut left = 0;
    let mut right = unique_times.len();

    while left < right {
        let mid = (left + right) / 2;
        if unique_times[mid] <= t {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if left == 0 { 1.0 } else { km_values[left - 1] }
}

pub fn uno_c_index_core(
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

    let min_g = 0.01;

    let compute_pair_contributions = |i: usize| -> (f64, f64, f64, f64, Vec<f64>) {
        let mut concordant = 0.0;
        let mut discordant = 0.0;
        let mut tied = 0.0;
        let mut total_weight = 0.0;
        let mut influence = vec![0.0; n];

        if status[i] != 1 || time[i] > tau_val {
            return (concordant, discordant, tied, total_weight, influence);
        }

        let g_ti = get_censoring_prob(time[i], &km_times, &km_values).max(min_g);
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

#[derive(Debug, Clone)]
#[pyclass]
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

pub fn compare_uno_c_indices_core(
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

    let min_g = 0.01;

    let mut concordant_1 = 0.0;
    let mut concordant_2 = 0.0;
    let mut total_pairs = 0.0;

    let mut influence_1 = vec![0.0; n];
    let mut influence_2 = vec![0.0; n];

    for i in 0..n {
        if status[i] != 1 || time[i] > tau_val {
            continue;
        }

        let g_ti = get_censoring_prob(time[i], &km_times, &km_values).max(min_g);
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

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
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

pub fn c_index_decomposition_core(
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
    let min_g = 0.01;

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

        let g_ti = get_censoring_prob(time[i], &km_times, &km_values).max(min_g);
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

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct GonenHellerResult {
    pub cpe: f64,
    pub n_pairs: usize,
    pub n_ties: usize,
    pub variance: f64,
    pub std_error: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

impl fmt::Display for GonenHellerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GonenHellerResult(cpe={:.4}, se={:.4}, ci=[{:.4}, {:.4}])",
            self.cpe, self.std_error, self.ci_lower, self.ci_upper
        )
    }
}

#[pymethods]
impl GonenHellerResult {
    #[new]
    fn new(
        cpe: f64,
        n_pairs: usize,
        n_ties: usize,
        variance: f64,
        std_error: f64,
        ci_lower: f64,
        ci_upper: f64,
    ) -> Self {
        Self {
            cpe,
            n_pairs,
            n_ties,
            variance,
            std_error,
            ci_lower,
            ci_upper,
        }
    }
}

pub fn gonen_heller_core(linear_predictor: &[f64]) -> GonenHellerResult {
    let n = linear_predictor.len();

    if n < 2 {
        return GonenHellerResult {
            cpe: 0.5,
            n_pairs: 0,
            n_ties: 0,
            variance: 0.0,
            std_error: 0.0,
            ci_lower: 0.5,
            ci_upper: 0.5,
        };
    }

    let compute_contributions = |i: usize| -> (f64, usize, usize, Vec<f64>) {
        let mut sum = 0.0;
        let mut pairs = 0usize;
        let mut ties = 0usize;
        let mut influence = vec![0.0; n];

        for j in (i + 1)..n {
            let diff = linear_predictor[i] - linear_predictor[j];

            if diff.abs() < 1e-10 {
                ties += 1;
                continue;
            }

            pairs += 1;
            let contribution = 1.0 / (1.0 + (-diff.abs()).exp());
            sum += contribution;

            let deriv = contribution * (1.0 - contribution);
            if diff > 0.0 {
                influence[i] += deriv;
                influence[j] -= deriv;
            } else {
                influence[i] -= deriv;
                influence[j] += deriv;
            }
        }

        (sum, pairs, ties, influence)
    };

    let results: Vec<(f64, usize, usize, Vec<f64>)> = if n > PARALLEL_THRESHOLD_LARGE {
        (0..n).into_par_iter().map(compute_contributions).collect()
    } else {
        (0..n).map(compute_contributions).collect()
    };

    let mut total_sum = 0.0;
    let mut total_pairs = 0usize;
    let mut total_ties = 0usize;
    let mut influence_sums = vec![0.0; n];

    for (sum, pairs, ties, influence) in results {
        total_sum += sum;
        total_pairs += pairs;
        total_ties += ties;
        for (k, &inf) in influence.iter().enumerate() {
            influence_sums[k] += inf;
        }
    }

    let cpe = if total_pairs > 0 {
        total_sum / total_pairs as f64
    } else {
        0.5
    };

    let variance = if total_pairs > 0 {
        let n_f = n as f64;
        let pairs_f = total_pairs as f64;
        let mut var_sum = 0.0;

        for &inf in &influence_sums {
            let normalized = inf / pairs_f;
            var_sum += normalized * normalized;
        }

        var_sum / n_f
    } else {
        0.0
    };

    let std_error = variance.sqrt();
    let z = 1.96;
    let ci_lower = (cpe - z * std_error).clamp(0.0, 1.0);
    let ci_upper = (cpe + z * std_error).clamp(0.0, 1.0);

    GonenHellerResult {
        cpe,
        n_pairs: total_pairs,
        n_ties: total_ties,
        variance,
        std_error,
        ci_lower,
        ci_upper,
    }
}

#[pyfunction]
pub fn gonen_heller_concordance(linear_predictor: Vec<f64>) -> PyResult<GonenHellerResult> {
    if linear_predictor.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "linear_predictor must not be empty",
        ));
    }

    Ok(gonen_heller_core(&linear_predictor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uno_c_index_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!(result.c_index > 0.9);
        assert!(result.std_error >= 0.0);
        assert!(result.ci_lower <= result.c_index);
        assert!(result.ci_upper >= result.c_index);
    }

    #[test]
    fn test_uno_c_index_random_prediction() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score = vec![0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4, 0.7, 0.35];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
    }

    #[test]
    fn test_uno_c_index_with_tau() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result_full = uno_c_index_core(&time, &status, &risk_score, None);
        let result_tau = uno_c_index_core(&time, &status, &risk_score, Some(5.0));

        assert!(result_tau.tau <= 5.0);
        assert!(result_tau.comparable_pairs <= result_full.comparable_pairs);
    }

    #[test]
    fn test_uno_c_index_heavy_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 0, 1, 0, 0, 1, 0];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
    }

    #[test]
    fn test_uno_c_index_empty() {
        let result = uno_c_index_core(&[], &[], &[], None);
        assert!((result.c_index - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compare_uno_c_indices() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let risk_score_1 = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let risk_score_2 = vec![0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1];

        let result = compare_uno_c_indices_core(&time, &status, &risk_score_1, &risk_score_2, None);

        assert!((0.0..=1.0).contains(&result.c_index_1));
        assert!((0.0..=1.0).contains(&result.c_index_2));
        assert!((result.difference - (result.c_index_1 - result.c_index_2)).abs() < 1e-10);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_censoring_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let (times, values) = compute_censoring_km(&time, &status);

        assert!(!times.is_empty());
        assert_eq!(times.len(), values.len());
        for &v in &values {
            assert!((0.0..=1.0).contains(&v));
        }
        for i in 1..values.len() {
            assert!(values[i] <= values[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_c_index_decomposition_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!((0.0..=1.0).contains(&result.c_index_ee));
        assert!((0.0..=1.0).contains(&result.c_index_ec));
        assert!((0.0..=1.0).contains(&result.alpha));
        assert!(result.n_event_event_pairs > 0);
        assert!(result.n_event_censored_pairs > 0);
    }

    #[test]
    fn test_c_index_decomposition_all_events() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 1, 1, 1];
        let risk_score = vec![0.9, 0.7, 0.5, 0.3, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!(result.c_index > 0.9);
        assert!(result.c_index_ee > 0.9);
        assert_eq!(result.n_event_censored_pairs, 0);
        assert!((result.alpha - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_c_index_decomposition_heavy_censoring() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 0, 0, 0, 1, 0, 0, 0];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = c_index_decomposition_core(&time, &status, &risk_score, None);

        assert!((0.0..=1.0).contains(&result.c_index));
        assert!(result.n_event_censored_pairs > result.n_event_event_pairs);
        assert!(result.alpha < 0.5);
    }

    #[test]
    fn test_c_index_decomposition_empty() {
        let result = c_index_decomposition_core(&[], &[], &[], None);

        assert!((result.c_index - 0.5).abs() < 1e-10);
        assert!((result.c_index_ee - 0.5).abs() < 1e-10);
        assert!((result.c_index_ec - 0.5).abs() < 1e-10);
        assert_eq!(result.n_event_event_pairs, 0);
        assert_eq!(result.n_event_censored_pairs, 0);
    }

    #[test]
    fn test_c_index_decomposition_consistency() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 0, 1, 0, 1];
        let risk_score = vec![0.9, 0.7, 0.8, 0.5, 0.4, 0.2];

        let decomp = c_index_decomposition_core(&time, &status, &risk_score, None);
        let uno = uno_c_index_core(&time, &status, &risk_score, None);

        assert!((decomp.c_index - uno.c_index).abs() < 0.01);
    }

    #[test]
    fn test_gonen_heller_basic() {
        let lp = vec![2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5];

        let result = gonen_heller_core(&lp);

        assert!((0.0..=1.0).contains(&result.cpe));
        assert!(result.cpe > 0.5);
        assert!(result.n_pairs > 0);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_gonen_heller_good_discrimination() {
        let lp = vec![10.0, 8.0, 6.0, 4.0, 2.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!(result.cpe > 0.9);
        assert_eq!(result.n_ties, 0);
    }

    #[test]
    fn test_gonen_heller_no_discrimination() {
        let lp = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!((result.cpe - 0.5).abs() < 1e-10);
        assert_eq!(result.n_pairs, 0);
        assert!(result.n_ties > 0);
    }

    #[test]
    fn test_gonen_heller_symmetric() {
        let lp1 = vec![1.0, 0.5, 0.0, -0.5, -1.0];
        let lp2: Vec<f64> = lp1.iter().map(|x| -x).collect();

        let result1 = gonen_heller_core(&lp1);
        let result2 = gonen_heller_core(&lp2);

        assert!((result1.cpe - result2.cpe).abs() < 1e-10);
    }

    #[test]
    fn test_gonen_heller_small_sample() {
        let lp = vec![1.0, 0.0];

        let result = gonen_heller_core(&lp);

        assert!((0.0..=1.0).contains(&result.cpe));
        assert_eq!(result.n_pairs, 1);
    }

    #[test]
    fn test_gonen_heller_single_element() {
        let lp = vec![1.0];

        let result = gonen_heller_core(&lp);

        assert!((result.cpe - 0.5).abs() < 1e-10);
        assert_eq!(result.n_pairs, 0);
    }
}
