use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CoverageSelectionResult {
    #[pyo3(get)]
    pub optimal_coverage: f64,
    #[pyo3(get)]
    pub coverage_candidates: Vec<f64>,
    #[pyo3(get)]
    pub mean_widths: Vec<f64>,
    #[pyo3(get)]
    pub empirical_coverages: Vec<f64>,
    #[pyo3(get)]
    pub efficiency_scores: Vec<f64>,
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, n_folds=None, coverage_candidates=None, seed=None))]
#[allow(clippy::too_many_arguments)]
pub fn conformal_coverage_cv(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    n_folds: Option<usize>,
    coverage_candidates: Option<Vec<f64>>,
    seed: Option<u64>,
) -> PyResult<CoverageSelectionResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let k = n_folds.unwrap_or(5);
    let candidates = coverage_candidates.unwrap_or_else(|| vec![0.80, 0.85, 0.90, 0.95, 0.99]);
    let base_seed = seed.unwrap_or(crate::constants::DEFAULT_RANDOM_SEED);

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = base_seed;
    for i in (1..n).rev() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let j = (rng_state as usize) % (i + 1);
        indices.swap(i, j);
    }

    let fold_size = n / k;
    let folds: Vec<Vec<usize>> = (0..k)
        .map(|i| {
            let start = i * fold_size;
            let end = if i == k - 1 { n } else { (i + 1) * fold_size };
            indices[start..end].to_vec()
        })
        .collect();

    let results: Vec<(f64, f64, f64)> = candidates
        .par_iter()
        .map(|&coverage| {
            let mut total_width = 0.0;
            let mut total_covered = 0usize;
            let mut total_events = 0usize;

            for fold_idx in 0..k {
                let test_indices = &folds[fold_idx];
                let train_indices: Vec<usize> = (0..k)
                    .filter(|&i| i != fold_idx)
                    .flat_map(|i| folds[i].iter().copied())
                    .collect();

                let train_time: Vec<f64> = train_indices.iter().map(|&i| time[i]).collect();
                let train_status: Vec<i32> = train_indices.iter().map(|&i| status[i]).collect();
                let train_pred: Vec<f64> = train_indices.iter().map(|&i| predicted[i]).collect();

                let mut scores = Vec::new();
                for i in 0..train_time.len() {
                    if train_status[i] == 1 {
                        scores.push(train_time[i] - train_pred[i]);
                    }
                }

                if scores.is_empty() {
                    continue;
                }

                let n_scores = scores.len();
                let q_level = (1.0 - coverage) * (n_scores as f64 + 1.0) / n_scores as f64;
                let q_level = q_level.min(1.0);
                let weights: Vec<f64> = vec![1.0; n_scores];
                let threshold = weighted_quantile(&scores, &weights, q_level);

                for &i in test_indices {
                    if status[i] == 1 {
                        total_events += 1;
                        let lb = (predicted[i] - threshold).max(0.0);
                        total_width += predicted[i] - lb;
                        if time[i] >= lb {
                            total_covered += 1;
                        }
                    }
                }
            }

            let emp_coverage = if total_events > 0 {
                total_covered as f64 / total_events as f64
            } else {
                0.0
            };
            let mean_width = if total_events > 0 {
                total_width / total_events as f64
            } else {
                f64::INFINITY
            };

            (coverage, emp_coverage, mean_width)
        })
        .collect();

    let coverage_candidates: Vec<f64> = results.iter().map(|(c, _, _)| *c).collect();
    let empirical_coverages: Vec<f64> = results.iter().map(|(_, e, _)| *e).collect();
    let mean_widths: Vec<f64> = results.iter().map(|(_, _, w)| *w).collect();

    let efficiency_scores: Vec<f64> = results
        .iter()
        .map(|(target, emp, width)| {
            let coverage_gap = (emp - target).abs();
            if *width > 0.0 && width.is_finite() {
                (1.0 - coverage_gap) / width
            } else {
                0.0
            }
        })
        .collect();

    let optimal_idx = efficiency_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let optimal_coverage = coverage_candidates[optimal_idx];

    Ok(CoverageSelectionResult {
        optimal_coverage,
        coverage_candidates,
        mean_widths,
        empirical_coverages,
        efficiency_scores,
    })
}
