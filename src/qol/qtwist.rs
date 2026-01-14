#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    unused_assignments,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct QTWISTResult {
    #[pyo3(get)]
    pub qtwist: f64,
    #[pyo3(get)]
    pub tox: f64,
    #[pyo3(get)]
    pub twistt: f64,
    #[pyo3(get)]
    pub rel: f64,
    #[pyo3(get)]
    pub total_time: f64,
    #[pyo3(get)]
    pub utility_tox: f64,
    #[pyo3(get)]
    pub utility_rel: f64,
    #[pyo3(get)]
    pub qtwist_difference: Option<f64>,
    #[pyo3(get)]
    pub ci_lower: Option<f64>,
    #[pyo3(get)]
    pub ci_upper: Option<f64>,
}

#[pyfunction]
#[pyo3(signature = (
    time,
    status,
    toxicity_start,
    toxicity_end,
    relapse_time,
    utility_tox=0.5,
    utility_rel=0.5,
    tau=None
))]
pub fn qtwist_analysis(
    time: Vec<f64>,
    status: Vec<i32>,
    toxicity_start: Vec<Option<f64>>,
    toxicity_end: Vec<Option<f64>>,
    relapse_time: Vec<Option<f64>>,
    utility_tox: f64,
    utility_rel: f64,
    tau: Option<f64>,
) -> PyResult<QTWISTResult> {
    let n = time.len();
    if status.len() != n
        || toxicity_start.len() != n
        || toxicity_end.len() != n
        || relapse_time.len() != n
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input arrays must have same length",
        ));
    }

    let tau_val = tau.unwrap_or_else(|| time.iter().copied().fold(0.0, f64::max));

    let mut total_tox = 0.0;
    let mut total_twistt = 0.0;
    let mut total_rel = 0.0;
    let mut total_time_sum = 0.0;

    for i in 0..n {
        let obs_time = time[i].min(tau_val);

        let tox_duration = match (toxicity_start[i], toxicity_end[i]) {
            (Some(start), Some(end)) => (end.min(obs_time) - start.max(0.0)).max(0.0),
            (Some(start), None) => (obs_time - start.max(0.0)).max(0.0),
            _ => 0.0,
        };

        let relapse_duration = match relapse_time[i] {
            Some(rel) if rel < obs_time => obs_time - rel,
            _ => 0.0,
        };

        let twistt_duration = obs_time - tox_duration - relapse_duration;

        total_tox += tox_duration;
        total_twistt += twistt_duration.max(0.0);
        total_rel += relapse_duration;
        total_time_sum += obs_time;
    }

    let mean_tox = total_tox / n as f64;
    let mean_twistt = total_twistt / n as f64;
    let mean_rel = total_rel / n as f64;
    let mean_total = total_time_sum / n as f64;

    let qtwist = utility_tox * mean_tox + mean_twistt + utility_rel * mean_rel;

    Ok(QTWISTResult {
        qtwist,
        tox: mean_tox,
        twistt: mean_twistt,
        rel: mean_rel,
        total_time: mean_total,
        utility_tox,
        utility_rel,
        qtwist_difference: None,
        ci_lower: None,
        ci_upper: None,
    })
}

#[pyfunction]
#[pyo3(signature = (
    time_treated,
    status_treated,
    tox_start_treated,
    tox_end_treated,
    relapse_treated,
    time_control,
    status_control,
    tox_start_control,
    tox_end_control,
    relapse_control,
    utility_tox=0.5,
    utility_rel=0.5,
    tau=None,
    n_bootstrap=1000
))]
pub fn qtwist_comparison(
    time_treated: Vec<f64>,
    status_treated: Vec<i32>,
    tox_start_treated: Vec<Option<f64>>,
    tox_end_treated: Vec<Option<f64>>,
    relapse_treated: Vec<Option<f64>>,
    time_control: Vec<f64>,
    status_control: Vec<i32>,
    tox_start_control: Vec<Option<f64>>,
    tox_end_control: Vec<Option<f64>>,
    relapse_control: Vec<Option<f64>>,
    utility_tox: f64,
    utility_rel: f64,
    tau: Option<f64>,
    n_bootstrap: usize,
) -> PyResult<(QTWISTResult, QTWISTResult, f64, f64, f64)> {
    let result_treated = qtwist_analysis(
        time_treated.clone(),
        status_treated.clone(),
        tox_start_treated.clone(),
        tox_end_treated.clone(),
        relapse_treated.clone(),
        utility_tox,
        utility_rel,
        tau,
    )?;

    let result_control = qtwist_analysis(
        time_control.clone(),
        status_control.clone(),
        tox_start_control.clone(),
        tox_end_control.clone(),
        relapse_control.clone(),
        utility_tox,
        utility_rel,
        tau,
    )?;

    let qtwist_diff = result_treated.qtwist - result_control.qtwist;

    let n_treated = time_treated.len();
    let n_control = time_control.len();

    let mut boot_diffs: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for b in 0..n_bootstrap {
        let mut rng = fastrand::Rng::with_seed(b as u64);

        let boot_treated_idx: Vec<usize> =
            (0..n_treated).map(|_| rng.usize(0..n_treated)).collect();
        let boot_control_idx: Vec<usize> =
            (0..n_control).map(|_| rng.usize(0..n_control)).collect();

        let boot_time_t: Vec<f64> = boot_treated_idx.iter().map(|&i| time_treated[i]).collect();
        let boot_status_t: Vec<i32> = boot_treated_idx
            .iter()
            .map(|&i| status_treated[i])
            .collect();
        let boot_tox_start_t: Vec<Option<f64>> = boot_treated_idx
            .iter()
            .map(|&i| tox_start_treated[i])
            .collect();
        let boot_tox_end_t: Vec<Option<f64>> = boot_treated_idx
            .iter()
            .map(|&i| tox_end_treated[i])
            .collect();
        let boot_relapse_t: Vec<Option<f64>> = boot_treated_idx
            .iter()
            .map(|&i| relapse_treated[i])
            .collect();

        let boot_time_c: Vec<f64> = boot_control_idx.iter().map(|&i| time_control[i]).collect();
        let boot_status_c: Vec<i32> = boot_control_idx
            .iter()
            .map(|&i| status_control[i])
            .collect();
        let boot_tox_start_c: Vec<Option<f64>> = boot_control_idx
            .iter()
            .map(|&i| tox_start_control[i])
            .collect();
        let boot_tox_end_c: Vec<Option<f64>> = boot_control_idx
            .iter()
            .map(|&i| tox_end_control[i])
            .collect();
        let boot_relapse_c: Vec<Option<f64>> = boot_control_idx
            .iter()
            .map(|&i| relapse_control[i])
            .collect();

        if let (Ok(res_t), Ok(res_c)) = (
            qtwist_analysis(
                boot_time_t,
                boot_status_t,
                boot_tox_start_t,
                boot_tox_end_t,
                boot_relapse_t,
                utility_tox,
                utility_rel,
                tau,
            ),
            qtwist_analysis(
                boot_time_c,
                boot_status_c,
                boot_tox_start_c,
                boot_tox_end_c,
                boot_relapse_c,
                utility_tox,
                utility_rel,
                tau,
            ),
        ) {
            boot_diffs.push(res_t.qtwist - res_c.qtwist);
        }
    }

    boot_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ci_lower = boot_diffs[(n_bootstrap as f64 * 0.025) as usize];
    let ci_upper = boot_diffs[(n_bootstrap as f64 * 0.975) as usize];

    Ok((
        result_treated,
        result_control,
        qtwist_diff,
        ci_lower,
        ci_upper,
    ))
}

#[pyfunction]
pub fn qtwist_sensitivity(
    time: Vec<f64>,
    status: Vec<i32>,
    toxicity_start: Vec<Option<f64>>,
    toxicity_end: Vec<Option<f64>>,
    relapse_time: Vec<Option<f64>>,
    utility_tox_range: Vec<f64>,
    utility_rel_range: Vec<f64>,
    tau: Option<f64>,
) -> PyResult<Vec<(f64, f64, f64)>> {
    let mut results = Vec::new();

    for &u_tox in &utility_tox_range {
        for &u_rel in &utility_rel_range {
            if let Ok(result) = qtwist_analysis(
                time.clone(),
                status.clone(),
                toxicity_start.clone(),
                toxicity_end.clone(),
                relapse_time.clone(),
                u_tox,
                u_rel,
                tau,
            ) {
                results.push((u_tox, u_rel, result.qtwist));
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qtwist_basic() {
        let time = vec![10.0, 15.0, 20.0, 25.0];
        let status = vec![1, 0, 1, 0];
        let tox_start = vec![Some(1.0), Some(2.0), None, Some(1.0)];
        let tox_end = vec![Some(5.0), Some(7.0), None, Some(3.0)];
        let relapse = vec![None, None, Some(15.0), None];

        let result =
            qtwist_analysis(time, status, tox_start, tox_end, relapse, 0.5, 0.5, None).unwrap();

        assert!(result.qtwist >= 0.0);
        assert!(result.tox >= 0.0);
        assert!(result.twistt >= 0.0);
    }
}
