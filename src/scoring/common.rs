use crate::constants::PARALLEL_THRESHOLD_MEDIUM;
use crate::internal::validation::{
    ValidationError, validate_binary_f64, validate_finite, validate_length, validate_non_negative,
};
use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

#[inline]
pub(crate) fn apply_deltas_add<F>(
    indices: &[usize],
    nvar: usize,
    matrix: &mut Array2<f64>,
    compute_deltas: F,
) where
    F: Fn(usize) -> Vec<f64> + Sync + Send,
{
    if indices.len() > PARALLEL_THRESHOLD_MEDIUM {
        let updates: Vec<(usize, Vec<f64>)> = indices
            .par_iter()
            .map(|&idx| (idx, compute_deltas(idx)))
            .collect();
        for (idx, deltas) in updates {
            for j in 0..nvar {
                matrix[[j, idx]] += deltas[j];
            }
        }
    } else {
        for &idx in indices {
            let deltas = compute_deltas(idx);
            for j in 0..nvar {
                matrix[[j, idx]] += deltas[j];
            }
        }
    }
}

#[inline]
pub(crate) fn apply_deltas_set<F>(
    indices: &[usize],
    nvar: usize,
    matrix: &mut Array2<f64>,
    compute_deltas: F,
) where
    F: Fn(usize) -> Vec<f64> + Sync + Send,
{
    if indices.len() > PARALLEL_THRESHOLD_MEDIUM {
        let updates: Vec<(usize, Vec<f64>)> = indices
            .par_iter()
            .map(|&idx| (idx, compute_deltas(idx)))
            .collect();
        for (idx, deltas) in updates {
            for j in 0..nvar {
                matrix[[j, idx]] = deltas[j];
            }
        }
    } else {
        for &idx in indices {
            let deltas = compute_deltas(idx);
            for j in 0..nvar {
                matrix[[j, idx]] = deltas[j];
            }
        }
    }
}

fn validation_err_to_pyresult<T>(result: Result<T, ValidationError>) -> PyResult<T> {
    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn validate_scoring_inputs(
    time_data: &[f64],
    covariates: &[f64],
    strata: &[i32],
    score: &[f64],
    weights: &[f64],
    method: i32,
) -> PyResult<(usize, usize)> {
    let n = weights.len();
    if n == 0 {
        return Err(PyValueError::new_err("No observations provided"));
    }
    let expected_time_len = 3usize
        .checked_mul(n)
        .ok_or_else(|| PyValueError::new_err("3 * n exceeds supported array size"))?;
    validation_err_to_pyresult(validate_length(
        expected_time_len,
        time_data.len(),
        "time_data",
    ))?;
    if !covariates.len().is_multiple_of(n) {
        return Err(PyValueError::new_err(
            "Covariates length should be divisible by number of observations",
        ));
    }
    let nvar = covariates.len() / n;
    validation_err_to_pyresult(validate_length(n, strata.len(), "strata"))?;
    validation_err_to_pyresult(validate_length(n, score.len(), "score"))?;
    if method != 0 && method != 1 {
        return Err(PyValueError::new_err(
            "method must be 0 (Breslow) or 1 (Efron)",
        ));
    }

    validation_err_to_pyresult(validate_finite(time_data, "time_data"))?;
    validation_err_to_pyresult(validate_binary_f64(
        &time_data[2 * n..expected_time_len],
        "event",
    ))?;
    validation_err_to_pyresult(validate_finite(covariates, "covariates"))?;
    validation_err_to_pyresult(validate_finite(score, "score"))?;
    validation_err_to_pyresult(validate_non_negative(score, "score"))?;
    validation_err_to_pyresult(validate_finite(weights, "weights"))?;
    validation_err_to_pyresult(validate_non_negative(weights, "weights"))?;

    Ok((n, nvar))
}
pub(crate) fn compute_summary_stats(residuals: &[f64], n: usize, nvar: usize) -> Vec<f64> {
    if n > PARALLEL_THRESHOLD_MEDIUM && nvar > 1 {
        (0..nvar)
            .into_par_iter()
            .flat_map(|i| {
                let start_idx = i * n;
                let end_idx = (i + 1) * n;
                let var_residuals = &residuals[start_idx..end_idx];
                let mean = var_residuals.iter().sum::<f64>() / n as f64;
                let variance = if n > 1 {
                    var_residuals
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (n - 1) as f64
                } else {
                    0.0
                };
                vec![mean, variance]
            })
            .collect()
    } else {
        let mut summary_stats = Vec::with_capacity(nvar * 2);
        for i in 0..nvar {
            let start_idx = i * n;
            let end_idx = (i + 1) * n;
            let var_residuals = &residuals[start_idx..end_idx];
            let mean = var_residuals.iter().sum::<f64>() / n as f64;
            let variance = if n > 1 {
                var_residuals
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / (n - 1) as f64
            } else {
                0.0
            };
            summary_stats.push(mean);
            summary_stats.push(variance);
        }
        summary_stats
    }
}
pub(crate) fn build_score_result(
    py: Python<'_>,
    residuals: Vec<f64>,
    n: usize,
    nvar: usize,
    method: i32,
) -> PyResult<Py<PyDict>> {
    let summary_stats = compute_summary_stats(&residuals, n, nvar);
    let dict = PyDict::new(py);
    dict.set_item("residuals", residuals)?;
    dict.set_item("n_observations", n)?;
    dict.set_item("n_variables", nvar)?;
    dict.set_item("method", if method == 0 { "breslow" } else { "efron" })?;
    dict.set_item("summary_stats", summary_stats)?;
    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn apply_deltas_add_accumulates() {
        let mut matrix = Array2::from_elem((1, 3), 1.0);
        apply_deltas_add(&[0, 1, 2], 1, &mut matrix, |idx| vec![idx as f64]);
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]], 2.0);
        assert_eq!(matrix[[0, 2]], 3.0);
    }

    #[test]
    fn apply_deltas_set_overwrites() {
        let mut matrix = Array2::from_elem((1, 3), 10.0);
        apply_deltas_set(&[0, 1, 2], 1, &mut matrix, |idx| vec![idx as f64]);
        assert_eq!(matrix[[0, 0]], 0.0);
        assert_eq!(matrix[[0, 1]], 1.0);
        assert_eq!(matrix[[0, 2]], 2.0);
    }

    #[test]
    fn compute_summary_stats_known_values() {
        let residuals = vec![1.0, 2.0, 3.0];
        let stats = compute_summary_stats(&residuals, 3, 1);
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 2.0).abs() < 1e-12);
        assert!((stats[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn compute_summary_stats_single_observation() {
        let residuals = vec![5.0];
        let stats = compute_summary_stats(&residuals, 1, 1);
        assert_eq!(stats.len(), 2);
        assert!((stats[0] - 5.0).abs() < 1e-12);
        assert_eq!(stats[1], 0.0);
    }

    #[test]
    fn compute_summary_stats_two_variables() {
        let residuals = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let stats = compute_summary_stats(&residuals, 3, 2);
        assert_eq!(stats.len(), 4);
        assert!((stats[0] - 2.0).abs() < 1e-12);
        assert!((stats[1] - 1.0).abs() < 1e-12);
        assert!((stats[2] - 20.0).abs() < 1e-12);
        assert!((stats[3] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn validate_scoring_inputs_accepts_strata_labels() {
        let result = validate_scoring_inputs(
            &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0],
            &[0.5, 1.0, 1.5],
            &[2, 2, 4],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
            0,
        )
        .expect("strata are labels, not binary flags");

        assert_eq!(result, (3, 1));
    }

    #[test]
    fn validate_scoring_inputs_rejects_bad_public_values() {
        let time_data = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0];
        let covariates = vec![0.5, 1.0, 1.5];
        let strata = vec![0, 0, 0];
        let score = vec![1.0, 1.0, 1.0];
        let weights = vec![1.0, 1.0, 1.0];

        let method_err =
            validate_scoring_inputs(&time_data, &covariates, &strata, &score, &weights, 2)
                .expect_err("unsupported method should fail");
        assert!(
            method_err
                .to_string()
                .contains("method must be 0 (Breslow) or 1 (Efron)")
        );

        let mut bad_time = time_data.clone();
        bad_time[0] = f64::NAN;
        let time_err =
            validate_scoring_inputs(&bad_time, &covariates, &strata, &score, &weights, 0)
                .expect_err("NaN time_data should fail");
        assert!(
            time_err
                .to_string()
                .contains("time_data contains non-finite")
        );

        let mut bad_event = time_data.clone();
        bad_event[7] = 0.5;
        let event_err =
            validate_scoring_inputs(&bad_event, &covariates, &strata, &score, &weights, 0)
                .expect_err("non-binary event should fail");
        assert!(
            event_err
                .to_string()
                .contains("event values must be 0 or 1")
        );

        let mut bad_score = score.clone();
        bad_score[1] = -1.0;
        let score_err =
            validate_scoring_inputs(&time_data, &covariates, &strata, &bad_score, &weights, 0)
                .expect_err("negative score should fail");
        assert!(
            score_err
                .to_string()
                .contains("score contains negative value")
        );

        let mut bad_weights = weights;
        bad_weights[2] = -1.0;
        let weight_err =
            validate_scoring_inputs(&time_data, &covariates, &strata, &score, &bad_weights, 0)
                .expect_err("negative weight should fail");
        assert!(
            weight_err
                .to_string()
                .contains("weights contains negative value")
        );
    }
}
