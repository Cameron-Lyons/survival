use crate::constants::same_time;
use crate::internal::numpy_utils::{extract_vec_f64, extract_vec_i32};
use crate::internal::statistical::chi2_sf;
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_no_nan, validate_non_negative,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;

const RANK_TIE_TOLERANCE: f64 = 1e-10;

fn validate_survobrien_inputs(
    time: &[f64],
    status: &[i32],
    covariate: &[f64],
    strata: Option<&[i32]>,
) -> PyResult<()> {
    validate_length(time.len(), status.len(), "status")?;
    validate_length(time.len(), covariate.len(), "covariate")?;
    if let Some(strata) = strata {
        validate_length(time.len(), strata.len(), "strata")?;
    }
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    validate_binary_i32(status, "status")?;
    validate_no_nan(covariate, "covariate")?;
    validate_finite(covariate, "covariate")?;
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvObrienResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub scores: Vec<f64>,
    #[pyo3(get)]
    pub score_sum: f64,
    #[pyo3(get)]
    pub expected: f64,
    #[pyo3(get)]
    pub variance: f64,
}

#[pymethods]
impl SurvObrienResult {
    #[new]
    fn new(
        statistic: f64,
        p_value: f64,
        df: usize,
        scores: Vec<f64>,
        score_sum: f64,
        expected: f64,
        variance: f64,
    ) -> Self {
        Self {
            statistic,
            p_value,
            df,
            scores,
            score_sum,
            expected,
            variance,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, covariate, strata=None))]
pub fn survobrien(
    time: &Bound<'_, PyAny>,
    status: &Bound<'_, PyAny>,
    covariate: &Bound<'_, PyAny>,
    strata: Option<&Bound<'_, PyAny>>,
) -> PyResult<SurvObrienResult> {
    let time_vec = extract_vec_f64(time)?;
    let status_vec = extract_vec_i32(status)?;
    let covariate_vec = extract_vec_f64(covariate)?;
    let strata_vec = strata.map(extract_vec_i32).transpose()?;
    let strata = strata_vec.as_deref();

    validate_survobrien_inputs(&time_vec, &status_vec, &covariate_vec, strata)?;
    let result = compute_survobrien(&time_vec, &status_vec, &covariate_vec, strata);
    Ok(result)
}

fn validate_survobrien_transform_groups(
    columns: &[Vec<f64>],
    row_indices: &[usize],
    group_sizes: &[usize],
) -> PyResult<()> {
    let n_rows = columns.first().map_or(0, Vec::len);
    for column in columns {
        if column.len() != n_rows {
            return Err(PyValueError::new_err(
                "survobrien transform columns must have equal lengths",
            ));
        }
        validate_finite(column, "columns")?;
    }

    if columns.is_empty() && !row_indices.is_empty() {
        return Err(PyValueError::new_err(
            "survobrien transform requires columns when row indices are present",
        ));
    }
    if row_indices.iter().any(|&index| index >= n_rows) {
        return Err(PyValueError::new_err(
            "survobrien transform row index is out of bounds",
        ));
    }

    let grouped_rows = group_sizes.iter().try_fold(0usize, |total, &size| {
        total
            .checked_add(size)
            .ok_or_else(|| PyValueError::new_err("survobrien transform group sizes overflowed"))
    })?;
    if grouped_rows != row_indices.len() {
        return Err(PyValueError::new_err(
            "survobrien transform group sizes must sum to the row index count",
        ));
    }
    Ok(())
}

fn transform_survobrien_groups(
    columns: &[Vec<f64>],
    row_indices: &[usize],
    group_sizes: &[usize],
) -> Vec<Vec<f64>> {
    let mut transformed = columns
        .iter()
        .map(|_| vec![0.0; row_indices.len()])
        .collect::<Vec<_>>();

    for (column, output) in columns.iter().zip(transformed.iter_mut()) {
        let mut offset = 0;
        let mut order = Vec::new();
        for &group_size in group_sizes {
            let group_end = offset + group_size;
            order.clear();
            order.extend(offset..group_end);
            order.sort_unstable_by(|&left, &right| {
                let left_value = column[row_indices[left]];
                let right_value = column[row_indices[right]];
                if left_value == right_value {
                    Ordering::Equal
                } else {
                    left_value.total_cmp(&right_value)
                }
            });

            let mut start = 0;
            while start < group_size {
                let mut end = start + 1;
                let tie_value = column[row_indices[order[start]]];
                while end < group_size && column[row_indices[order[end]]] == tie_value {
                    end += 1;
                }
                let rank = ((start + 1) as f64 + end as f64) / 2.0;
                let probability = (rank - 0.5) / group_size as f64;
                let value = (probability / (1.0 - probability)).ln();
                for position in start..end {
                    output[order[position]] = value;
                }
                start = end;
            }
            offset = group_end;
        }
    }
    transformed
}

#[pyfunction]
pub fn survobrien_transform_groups(
    py: Python<'_>,
    columns: Vec<Vec<f64>>,
    row_indices: Vec<usize>,
    group_sizes: Vec<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_survobrien_transform_groups(&columns, &row_indices, &group_sizes)?;
    Ok(py.detach(move || transform_survobrien_groups(&columns, &row_indices, &group_sizes)))
}

fn compute_survobrien(
    time: &[f64],
    status: &[i32],
    covariate: &[f64],
    strata: Option<&[i32]>,
) -> SurvObrienResult {
    let n = time.len();
    if n == 0 {
        return SurvObrienResult {
            statistic: 0.0,
            p_value: 1.0,
            df: 1,
            scores: Vec::new(),
            score_sum: 0.0,
            expected: 0.0,
            variance: 0.0,
        };
    }

    let mut scores = vec![0.0; n];

    let mut total_score_sum = 0.0;
    let mut total_variance = 0.0;

    if let Some(strata) = strata {
        let mut unique_strata: Vec<i32> = strata.to_vec();
        unique_strata.sort();
        unique_strata.dedup();

        for &stratum in &unique_strata {
            let mut sorted_indices: Vec<usize> = (0..n).filter(|&i| strata[i] == stratum).collect();
            if sorted_indices.is_empty() {
                continue;
            }
            sorted_indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
            let (score_sum, variance) =
                compute_survobrien_stratum(time, status, covariate, &sorted_indices, &mut scores);
            total_score_sum += score_sum;
            total_variance += variance;
        }
    } else {
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
        let (score_sum, variance) =
            compute_survobrien_stratum(time, status, covariate, &sorted_indices, &mut scores);
        total_score_sum += score_sum;
        total_variance += variance;
    }

    let statistic = if total_variance > 0.0 {
        total_score_sum * total_score_sum / total_variance
    } else {
        0.0
    };

    let p_value = chi2_sf(statistic, 1);

    SurvObrienResult {
        statistic,
        p_value,
        df: 1,
        scores,
        score_sum: total_score_sum,
        expected: 0.0,
        variance: total_variance,
    }
}

fn compute_survobrien_stratum(
    time: &[f64],
    status: &[i32],
    covariate: &[f64],
    sorted_indices: &[usize],
    scores: &mut [f64],
) -> (f64, f64) {
    let n_stratum = sorted_indices.len();
    let mut score_sum = 0.0;
    let mut variance = 0.0;

    if n_stratum == 0 {
        return (score_sum, variance);
    }

    let mut at_risk: Vec<bool> = vec![true; n_stratum];
    let mut at_risk_values: Vec<(usize, f64)> = Vec::with_capacity(n_stratum);
    let mut ranks = vec![0.0; n_stratum];

    let mut i = 0;
    while i < n_stratum {
        let current_time = time[sorted_indices[i]];

        let mut event_indices: Vec<usize> = Vec::new();
        let mut j = i;
        while j < n_stratum && same_time(time[sorted_indices[j]], current_time) {
            if status[sorted_indices[j]] == 1 {
                event_indices.push(j);
            }
            j += 1;
        }

        if !event_indices.is_empty() {
            at_risk_values.clear();
            for (k, &idx) in sorted_indices.iter().enumerate() {
                if at_risk[k] {
                    at_risk_values.push((k, covariate[idx]));
                }
            }

            let n_at_risk = at_risk_values.len();
            if n_at_risk > 0 {
                at_risk_values.sort_by(|a, b| a.1.total_cmp(&b.1));

                let mut k = 0;
                while k < n_at_risk {
                    let current_value = at_risk_values[k].1;
                    let mut tie_count = 1;
                    let mut rank_sum = (k + 1) as f64;

                    while k + tie_count < n_at_risk
                        && (at_risk_values[k + tie_count].1 - current_value).abs()
                            < RANK_TIE_TOLERANCE
                    {
                        rank_sum += (k + tie_count + 1) as f64;
                        tie_count += 1;
                    }

                    let avg_rank = rank_sum / tie_count as f64;
                    for t in 0..tie_count {
                        ranks[at_risk_values[k + t].0] = avg_rank;
                    }
                    k += tie_count;
                }

                let mean_rank = (n_at_risk as f64 + 1.0) / 2.0;
                let var_rank = (n_at_risk as f64 * n_at_risk as f64 - 1.0) / 12.0;

                for &event_local_idx in &event_indices {
                    let rank = ranks[event_local_idx];
                    let orig_idx = sorted_indices[event_local_idx];
                    if var_rank > 0.0 {
                        scores[orig_idx] = (rank - mean_rank) / var_rank.sqrt();
                    } else {
                        scores[orig_idx] = 0.0;
                    }
                    score_sum += scores[orig_idx];
                }

                let n_events = event_indices.len() as f64;
                if var_rank > 0.0 {
                    variance += n_events / var_rank * var_rank;
                }
            }
        }

        for item in at_risk.iter_mut().take(j).skip(i) {
            *item = false;
        }

        i = j;
    }

    (score_sum, variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survobrien_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let covariate = vec![10.0, 20.0, 15.0, 30.0, 25.0];
        let strata = vec![1, 1, 1, 1, 1];

        let result = compute_survobrien(&time, &status, &covariate, Some(&strata));

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_survobrien_empty() {
        let result = compute_survobrien(&[], &[], &[], None);
        assert_eq!(result.statistic, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_survobrien_stratified() {
        let time = vec![1.0, 2.0, 1.0, 2.0];
        let status = vec![1, 0, 1, 0];
        let covariate = vec![10.0, 20.0, 30.0, 40.0];
        let strata = vec![1, 1, 2, 2];

        let result = compute_survobrien(&time, &status, &covariate, Some(&strata));

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_survobrien_default_strata_matches_explicit_single_stratum() {
        let time = vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 1, 0, 1];
        let covariate = vec![10.0, 20.0, 15.0, 30.0, 25.0, 35.0];
        let strata = vec![1; time.len()];

        let default = compute_survobrien(&time, &status, &covariate, None);
        let explicit = compute_survobrien(&time, &status, &covariate, Some(&strata));

        assert!((default.statistic - explicit.statistic).abs() < 1e-12);
        assert!((default.p_value - explicit.p_value).abs() < 1e-12);
        assert!((default.score_sum - explicit.score_sum).abs() < 1e-12);
        assert!((default.expected - explicit.expected).abs() < 1e-12);
        assert!((default.variance - explicit.variance).abs() < 1e-12);
        for (default, explicit) in default.scores.iter().zip(explicit.scores.iter()) {
            assert!((*default - *explicit).abs() < 1e-12);
        }
    }

    #[test]
    fn test_survobrien_groups_near_tied_event_times() {
        let exact_time = vec![1.0, 1.0, 2.0, 3.0];
        let near_time = vec![1.0, 1.0 + crate::constants::TIME_EPSILON / 2.0, 2.0, 3.0];
        let status = vec![1, 1, 0, 0];
        let covariate = vec![10.0, 30.0, 20.0, 40.0];
        let strata = vec![1, 1, 1, 1];

        let expected = compute_survobrien(&exact_time, &status, &covariate, Some(&strata));
        let actual = compute_survobrien(&near_time, &status, &covariate, Some(&strata));

        assert!((actual.statistic - expected.statistic).abs() < 1e-12);
        assert!((actual.p_value - expected.p_value).abs() < 1e-12);
        assert!((actual.score_sum - expected.score_sum).abs() < 1e-12);
        assert!((actual.variance - expected.variance).abs() < 1e-12);
        for (actual, expected) in actual.scores.iter().zip(expected.scores.iter()) {
            assert!((*actual - *expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_survobrien_validates_public_inputs() {
        let err = validate_survobrien_inputs(&[1.0, 2.0], &[1], &[0.1, 0.2], Some(&[1, 1]))
            .expect_err("status length mismatch should fail");
        assert!(err.to_string().contains("status length mismatch"));

        let err = validate_survobrien_inputs(&[1.0], &[2], &[0.1], Some(&[1]))
            .expect_err("non-binary status should fail");
        assert!(err.to_string().contains("status must contain only 0/1"));

        let err = validate_survobrien_inputs(&[1.0], &[1], &[f64::INFINITY], Some(&[1]))
            .expect_err("non-finite covariate should fail");
        assert!(err.to_string().contains("covariate contains non-finite"));

        let err = validate_survobrien_inputs(&[1.0, 2.0], &[1, 0], &[0.1, 0.2], Some(&[1]))
            .expect_err("strata length mismatch should fail");
        assert!(err.to_string().contains("strata length mismatch"));
    }

    #[test]
    fn test_survobrien_transform_groups_handles_ties_and_empty_groups() {
        let columns = vec![vec![4.0, 1.0, 1.0, 3.0], vec![0.0, 2.0, 1.0, 2.0]];
        let rows = vec![0, 1, 2, 3, 2];
        let actual = transform_survobrien_groups(&columns, &rows, &[4, 0, 1]);

        let low_tie = (0.25_f64 / 0.75).ln();
        let high = (0.875_f64 / 0.125).ln();
        assert_eq!(actual.len(), 2);
        assert!((actual[0][0] - high).abs() < 1e-12);
        assert!((actual[0][1] - low_tie).abs() < 1e-12);
        assert!((actual[0][2] - low_tie).abs() < 1e-12);
        assert_eq!(actual[0][4], 0.0);
        assert_eq!(actual[1][4], 0.0);
    }

    #[test]
    fn test_survobrien_transform_groups_ties_signed_zero() {
        let columns = vec![vec![-0.0, 0.0, 1.0]];
        let actual = transform_survobrien_groups(&columns, &[0, 1, 2], &[3]);

        assert_eq!(actual[0][0], actual[0][1]);
        assert!(actual[0][0] < 0.0);
        assert!(actual[0][2] > 0.0);
    }

    #[test]
    fn test_survobrien_transform_groups_validates_shape_and_indices() {
        let err = validate_survobrien_transform_groups(&[vec![1.0, 2.0], vec![1.0]], &[0], &[1])
            .expect_err("unequal column lengths should fail");
        assert!(err.to_string().contains("equal lengths"));

        let err = validate_survobrien_transform_groups(&[vec![1.0]], &[1], &[1])
            .expect_err("out-of-bounds index should fail");
        assert!(err.to_string().contains("out of bounds"));

        let err = validate_survobrien_transform_groups(&[vec![1.0]], &[0], &[0])
            .expect_err("incorrect group size total should fail");
        assert!(err.to_string().contains("must sum"));

        let err = validate_survobrien_transform_groups(&[vec![f64::INFINITY]], &[0], &[1])
            .expect_err("non-finite values should fail");
        assert!(err.to_string().contains("non-finite"));
    }
}
