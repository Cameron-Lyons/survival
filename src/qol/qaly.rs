#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct QALYResult {
    #[pyo3(get)]
    pub qaly: f64,
    #[pyo3(get)]
    pub life_years: f64,
    #[pyo3(get)]
    pub mean_utility: f64,
    #[pyo3(get)]
    pub qaly_by_period: Vec<f64>,
    #[pyo3(get)]
    pub discounted_qaly: f64,
    #[pyo3(get)]
    pub qaly_se: f64,
    #[pyo3(get)]
    pub qaly_ci_lower: f64,
    #[pyo3(get)]
    pub qaly_ci_upper: f64,
}

fn qaly_calculation_internal(
    time: &[f64],
    status: &[i32],
    utility_values: &[f64],
    utility_times: &[f64],
    discount_rate: f64,
    tau: f64,
) -> Option<(f64, f64, f64, Vec<f64>)> {
    let unique_times: Vec<f64> = {
        let mut times = utility_times.to_vec();
        times.push(0.0);
        times.push(tau);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        times.dedup();
        times.into_iter().filter(|&t| t <= tau).collect()
    };

    let get_utility = |t: f64| -> f64 {
        for (i, &ut) in utility_times.iter().enumerate().rev() {
            if t >= ut && i < utility_values.len() {
                return utility_values[i];
            }
        }
        utility_values.first().copied().unwrap_or(1.0)
    };

    let survival_km = kaplan_meier_estimate(time, status, &unique_times);

    let mut total_qaly = 0.0;
    let mut total_ly = 0.0;
    let mut total_discounted_qaly = 0.0;
    let mut qaly_by_period = Vec::new();

    for i in 1..unique_times.len() {
        let t0 = unique_times[i - 1];
        let t1 = unique_times[i];
        let dt = t1 - t0;

        let s0 = survival_km[i - 1];
        let s1 = survival_km[i];
        let avg_surv = (s0 + s1) / 2.0;

        let utility = get_utility(t0);
        let period_qaly = avg_surv * utility * dt;
        let period_ly = avg_surv * dt;

        let mid_time = (t0 + t1) / 2.0;
        let discount_factor = (1.0 + discount_rate).powf(-mid_time);
        let discounted_period_qaly = period_qaly * discount_factor;

        total_qaly += period_qaly;
        total_ly += period_ly;
        total_discounted_qaly += discounted_period_qaly;
        qaly_by_period.push(period_qaly);
    }

    Some((total_qaly, total_ly, total_discounted_qaly, qaly_by_period))
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    utility_values,
    utility_times,
    discount_rate=0.03,
    horizon=None
))]
pub fn qaly_calculation(
    time: Vec<f64>,
    status: Vec<i32>,
    utility_values: Vec<f64>,
    utility_times: Vec<f64>,
    discount_rate: f64,
    horizon: Option<f64>,
) -> PyResult<QALYResult> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have same length",
        ));
    }

    let tau = horizon.unwrap_or_else(|| time.iter().copied().fold(0.0, f64::max));

    let (total_qaly, total_ly, total_discounted_qaly, qaly_by_period) = qaly_calculation_internal(
        &time,
        &status,
        &utility_values,
        &utility_times,
        discount_rate,
        tau,
    )
    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to compute QALY"))?;

    let mean_utility = if total_ly > 0.0 {
        total_qaly / total_ly
    } else {
        0.0
    };

    let (qaly_se, qaly_ci_lower, qaly_ci_upper) = bootstrap_qaly_ci(
        &time,
        &status,
        &utility_values,
        &utility_times,
        tau,
        discount_rate,
        500,
    );

    Ok(QALYResult {
        qaly: total_qaly,
        life_years: total_ly,
        mean_utility,
        qaly_by_period,
        discounted_qaly: total_discounted_qaly,
        qaly_se,
        qaly_ci_lower,
        qaly_ci_upper,
    })
}

fn kaplan_meier_estimate(time: &[f64], status: &[i32], eval_times: &[f64]) -> Vec<f64> {
    let n = time.len();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        time[a]
            .partial_cmp(&time[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut survival = Vec::with_capacity(eval_times.len());
    let mut cum_surv = 1.0;
    let mut at_risk = n;
    let mut time_idx = 0;

    for &t in eval_times {
        while time_idx < n && time[indices[time_idx]] <= t {
            let idx = indices[time_idx];
            if status[idx] == 1 && at_risk > 0 {
                cum_surv *= 1.0 - 1.0 / at_risk as f64;
            }
            at_risk = at_risk.saturating_sub(1);
            time_idx += 1;
        }
        survival.push(cum_surv);
    }

    survival
}

fn bootstrap_qaly_ci(
    time: &[f64],
    status: &[i32],
    utility_values: &[f64],
    utility_times: &[f64],
    tau: f64,
    discount_rate: f64,
    n_bootstrap: usize,
) -> (f64, f64, f64) {
    let n = time.len();

    let qalys: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = fastrand::Rng::with_seed(b as u64 + 98765);
            let indices: Vec<usize> = (0..n).map(|_| rng.usize(0..n)).collect();

            let boot_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
            let boot_status: Vec<i32> = indices.iter().map(|&i| status[i]).collect();

            qaly_calculation_internal(
                &boot_time,
                &boot_status,
                utility_values,
                utility_times,
                discount_rate,
                tau,
            )
            .map(|(qaly, _, _, _)| qaly)
        })
        .collect();

    if qalys.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let mean = qalys.iter().sum::<f64>() / qalys.len() as f64;
    let var = qalys.iter().map(|&q| (q - mean).powi(2)).sum::<f64>() / (qalys.len() - 1) as f64;
    let se = var.sqrt();

    let mut sorted = qalys.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ci_lower = sorted[(qalys.len() as f64 * 0.025) as usize];
    let ci_upper = sorted[(qalys.len() as f64 * 0.975) as usize];

    (se, ci_lower, ci_upper)
}

#[pyfunction]
#[pyo3(signature = (
    time_treated,
    status_treated,
    utility_treated,
    time_control,
    status_control,
    utility_control,
    utility_times,
    discount_rate=0.03,
    horizon=None,
    n_bootstrap=1000
))]
pub fn qaly_comparison(
    time_treated: Vec<f64>,
    status_treated: Vec<i32>,
    utility_treated: Vec<f64>,
    time_control: Vec<f64>,
    status_control: Vec<i32>,
    utility_control: Vec<f64>,
    utility_times: Vec<f64>,
    discount_rate: f64,
    horizon: Option<f64>,
    n_bootstrap: usize,
) -> PyResult<(QALYResult, QALYResult, f64, f64, f64)> {
    let result_treated = qaly_calculation(
        time_treated.clone(),
        status_treated.clone(),
        utility_treated.clone(),
        utility_times.clone(),
        discount_rate,
        horizon,
    )?;

    let result_control = qaly_calculation(
        time_control.clone(),
        status_control.clone(),
        utility_control.clone(),
        utility_times.clone(),
        discount_rate,
        horizon,
    )?;

    let qaly_diff = result_treated.qaly - result_control.qaly;

    let n_treated = time_treated.len();
    let n_control = time_control.len();
    let tau = horizon.unwrap_or_else(|| {
        time_treated
            .iter()
            .chain(time_control.iter())
            .copied()
            .fold(0.0, f64::max)
    });

    let boot_diffs: Vec<f64> = (0..n_bootstrap)
        .into_par_iter()
        .filter_map(|b| {
            let mut rng = fastrand::Rng::with_seed(b as u64 + 11111);

            let idx_t: Vec<usize> = (0..n_treated).map(|_| rng.usize(0..n_treated)).collect();
            let idx_c: Vec<usize> = (0..n_control).map(|_| rng.usize(0..n_control)).collect();

            let boot_time_t: Vec<f64> = idx_t.iter().map(|&i| time_treated[i]).collect();
            let boot_status_t: Vec<i32> = idx_t.iter().map(|&i| status_treated[i]).collect();
            let boot_time_c: Vec<f64> = idx_c.iter().map(|&i| time_control[i]).collect();
            let boot_status_c: Vec<i32> = idx_c.iter().map(|&i| status_control[i]).collect();

            match (
                qaly_calculation_internal(
                    &boot_time_t,
                    &boot_status_t,
                    &utility_treated,
                    &utility_times,
                    discount_rate,
                    tau,
                ),
                qaly_calculation_internal(
                    &boot_time_c,
                    &boot_status_c,
                    &utility_control,
                    &utility_times,
                    discount_rate,
                    tau,
                ),
            ) {
                (Some((qaly_t, _, _, _)), Some((qaly_c, _, _, _))) => Some(qaly_t - qaly_c),
                _ => None,
            }
        })
        .collect();

    let mut sorted_diffs = boot_diffs.clone();
    sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ci_lower = sorted_diffs[(boot_diffs.len() as f64 * 0.025) as usize];
    let ci_upper = sorted_diffs[(boot_diffs.len() as f64 * 0.975) as usize];

    Ok((
        result_treated,
        result_control,
        qaly_diff,
        ci_lower,
        ci_upper,
    ))
}

#[pyfunction]
pub fn incremental_cost_effectiveness(
    qaly_treated: f64,
    qaly_control: f64,
    cost_treated: f64,
    cost_control: f64,
    wtp_threshold: Option<f64>,
) -> PyResult<(f64, f64, bool)> {
    let delta_qaly = qaly_treated - qaly_control;
    let delta_cost = cost_treated - cost_control;

    let icer = if delta_qaly.abs() > 1e-10 {
        delta_cost / delta_qaly
    } else {
        f64::INFINITY
    };

    let threshold = wtp_threshold.unwrap_or(50000.0);
    let cost_effective = icer <= threshold;

    let nmb = delta_qaly * threshold - delta_cost;

    Ok((icer, nmb, cost_effective))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qaly_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let utility_values = vec![0.8, 0.6];
        let utility_times = vec![0.0, 2.0];

        let result =
            qaly_calculation(time, status, utility_values, utility_times, 0.03, None).unwrap();

        assert!(result.qaly >= 0.0);
        assert!(result.life_years >= 0.0);
        assert!(result.mean_utility >= 0.0 && result.mean_utility <= 1.0);
    }

    #[test]
    fn test_icer() {
        let (icer, nmb, cost_effective) =
            incremental_cost_effectiveness(10.0, 8.0, 50000.0, 30000.0, Some(50000.0)).unwrap();

        assert_eq!(icer, 10000.0);
        assert!(cost_effective);
    }
}
