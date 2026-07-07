use super::super::*;

const DEFAULT_COVERAGE_CANDIDATES: &[f64] = &[0.80, 0.85, 0.90, 0.95, 0.99];

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
    if n < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least two observations are required for cross-validation",
        ));
    }
    if status.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and predicted must have the same length",
        ));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    for (idx, &value) in status.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "status must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }
    for (idx, &value) in predicted.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "predicted contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "predicted must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    if !status.contains(&1) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one uncensored observation is required",
        ));
    }

    let k = n_folds.unwrap_or_else(|| 5.min(n));
    if k < 2 || k > n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "n_folds must be between 2 and the number of observations ({n})"
        )));
    }
    let candidates = coverage_candidates
        .as_deref()
        .unwrap_or(DEFAULT_COVERAGE_CANDIDATES);
    if candidates.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_candidates cannot be empty",
        ));
    }
    for (idx, &coverage) in candidates.iter().enumerate() {
        if !coverage.is_finite() || !(0.0..1.0).contains(&coverage) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "coverage_candidates must contain finite values between 0 and 1; got {coverage} at index {idx}"
            )));
        }
    }
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
    let remainder = n % k;
    let mut start = 0;
    let folds: Vec<Vec<usize>> = (0..k)
        .map(|i| {
            let size = fold_size + usize::from(i < remainder);
            let end = start + size;
            let fold = indices[start..end].to_vec();
            start = end;
            fold
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
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_coverage_candidates_match_explicit_grid() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1];
        let predicted = vec![1.1, 1.9, 3.2, 3.8, 5.2, 5.8, 7.1, 7.9];

        let default_result = conformal_coverage_cv(
            time.clone(),
            status.clone(),
            predicted.clone(),
            Some(4),
            None,
            Some(7),
        )
        .unwrap();
        let explicit_result = conformal_coverage_cv(
            time,
            status,
            predicted,
            Some(4),
            Some(DEFAULT_COVERAGE_CANDIDATES.to_vec()),
            Some(7),
        )
        .unwrap();

        assert_eq!(
            default_result.coverage_candidates,
            explicit_result.coverage_candidates
        );
        assert_eq!(default_result.mean_widths, explicit_result.mean_widths);
        assert_eq!(
            default_result.empirical_coverages,
            explicit_result.empirical_coverages
        );
        assert_eq!(
            default_result.efficiency_scores,
            explicit_result.efficiency_scores
        );
        assert_eq!(
            default_result.optimal_coverage,
            explicit_result.optimal_coverage
        );
    }
}
