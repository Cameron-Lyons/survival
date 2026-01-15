use crate::constants::PARALLEL_THRESHOLD_LARGE;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct UnoCIndexResult {
    #[pyo3(get)]
    pub c_index: f64,
    #[pyo3(get)]
    pub concordant: f64,
    #[pyo3(get)]
    pub discordant: f64,
    #[pyo3(get)]
    pub tied_risk: f64,
    #[pyo3(get)]
    pub comparable_pairs: f64,
    #[pyo3(get)]
    pub variance: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub ci_lower: f64,
    #[pyo3(get)]
    pub ci_upper: f64,
    #[pyo3(get)]
    pub tau: f64,
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

fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989423 * (-x * x / 2.0).exp();
    let p =
        d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if x > 0.0 { 1.0 - p } else { p }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uno_c_index_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let risk_score = vec![0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        let result = uno_c_index_core(&time, &status, &risk_score, None);

        assert!(result.c_index >= 0.0 && result.c_index <= 1.0);
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

        assert!(result.c_index >= 0.0 && result.c_index <= 1.0);
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

        assert!(result.c_index >= 0.0 && result.c_index <= 1.0);
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

        assert!(result.c_index_1 >= 0.0 && result.c_index_1 <= 1.0);
        assert!(result.c_index_2 >= 0.0 && result.c_index_2 <= 1.0);
        assert!((result.difference - (result.c_index_1 - result.c_index_2)).abs() < 1e-10);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_censoring_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let (times, values) = compute_censoring_km(&time, &status);

        assert!(!times.is_empty());
        assert_eq!(times.len(), values.len());
        for &v in &values {
            assert!(v >= 0.0 && v <= 1.0);
        }
        for i in 1..values.len() {
            assert!(values[i] <= values[i - 1] + 1e-10);
        }
    }
}
