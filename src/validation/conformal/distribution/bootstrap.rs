use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct BootstrapConformalResult {
    #[pyo3(get)]
    pub lower_bound: Vec<f64>,
    #[pyo3(get)]
    pub upper_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub n_bootstrap: usize,
    #[pyo3(get)]
    pub bootstrap_quantile_lower: f64,
    #[pyo3(get)]
    pub bootstrap_quantile_upper: f64,
}

pub(crate) fn bootstrap_sample_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n);
    let mut rng_state = seed;

    for _ in 0..n {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let idx = (rng_state as usize) % n;
        indices.push(idx);
    }

    indices
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, predicted_new, coverage_level=None, n_bootstrap=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn bootstrap_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    predicted_new: Vec<f64>,
    coverage_level: Option<f64>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
) -> PyResult<BootstrapConformalResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    let n_boot = n_bootstrap.unwrap_or(200);
    let base_seed = seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let alpha = 1.0 - coverage;

    let bootstrap_thresholds: Vec<(f64, f64)> = (0..n_boot)
        .into_par_iter()
        .map(|b| {
            let boot_seed = base_seed.wrapping_add(b as u64);
            let indices = bootstrap_sample_indices(n, boot_seed);

            let mut lower_scores = Vec::new();
            let mut upper_scores = Vec::new();

            for &i in &indices {
                if status[i] == 1 {
                    lower_scores.push(predicted[i] - time[i]);
                    upper_scores.push(time[i] - predicted[i]);
                }
            }

            if lower_scores.is_empty() {
                return (0.0, 0.0);
            }

            let weights: Vec<f64> = vec![1.0; lower_scores.len()];
            let q_level =
                (1.0 - alpha / 2.0) * (lower_scores.len() as f64 + 1.0) / lower_scores.len() as f64;
            let q_level = q_level.min(1.0);

            let lower_q = weighted_quantile(&lower_scores, &weights, q_level);
            let upper_q = weighted_quantile(&upper_scores, &weights, q_level);

            (lower_q, upper_q)
        })
        .collect();

    let mut all_lower: Vec<f64> = bootstrap_thresholds.iter().map(|(l, _)| *l).collect();
    let mut all_upper: Vec<f64> = bootstrap_thresholds.iter().map(|(_, u)| *u).collect();

    all_lower.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_upper.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let upper_idx = ((1.0 - alpha / 2.0) * n_boot as f64) as usize;
    let upper_idx = upper_idx.min(n_boot - 1);

    let final_lower_q = all_lower[upper_idx];
    let final_upper_q = all_upper[upper_idx];

    let lower_bound: Vec<f64> = predicted_new
        .iter()
        .map(|&p| (p - final_lower_q).max(0.0))
        .collect();

    let upper_bound: Vec<f64> = predicted_new.iter().map(|&p| p + final_upper_q).collect();

    Ok(BootstrapConformalResult {
        lower_bound,
        upper_bound,
        predicted_time: predicted_new,
        coverage_level: coverage,
        n_bootstrap: n_boot,
        bootstrap_quantile_lower: final_lower_q,
        bootstrap_quantile_upper: final_upper_q,
    })
}
