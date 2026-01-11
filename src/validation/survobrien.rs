//! O'Brien's Test for Association of a Single Variable with Survival
//!
//! O'Brien's test is a generalization of the Wilcoxon rank-sum test to censored data.
//! At each failure time, it ranks the covariate values among subjects still at risk
//! and computes a test statistic based on these ranks.

use crate::utilities::numpy_utils::{extract_vec_f64, extract_vec_i32};
use crate::utilities::statistical::chi2_sf;
use pyo3::prelude::*;

/// Result of O'Brien's test for survival association
#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvObrienResult {
    /// Chi-squared test statistic
    #[pyo3(get)]
    pub statistic: f64,
    /// P-value from chi-squared distribution
    #[pyo3(get)]
    pub p_value: f64,
    /// Degrees of freedom
    #[pyo3(get)]
    pub df: usize,
    /// O'Brien scores for each observation
    #[pyo3(get)]
    pub scores: Vec<f64>,
    /// Sum of scores at event times
    #[pyo3(get)]
    pub score_sum: f64,
    /// Expected sum under null hypothesis
    #[pyo3(get)]
    pub expected: f64,
    /// Variance of the test statistic
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

/// O'Brien's Test for Association with Survival
///
/// Tests whether a continuous covariate is associated with survival time.
/// The test ranks covariate values among subjects at risk at each failure time
/// and computes a test statistic analogous to the Wilcoxon rank-sum test.
///
/// # Arguments
/// * `time` - Survival/censoring times
/// * `status` - Event indicator (1=event, 0=censored)
/// * `covariate` - Continuous covariate to test
/// * `strata` - Optional stratification variable
///
/// # Returns
/// SurvObrienResult with test statistic, p-value, and scores
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
    let strata_vec = match strata {
        Some(s) => extract_vec_i32(s)?,
        None => vec![1; time_vec.len()],
    };

    let result = compute_survobrien(&time_vec, &status_vec, &covariate_vec, &strata_vec);
    Ok(result)
}

/// Compute O'Brien's test statistic
fn compute_survobrien(
    time: &[f64],
    status: &[i32],
    covariate: &[f64],
    strata: &[i32],
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

    let mut unique_strata: Vec<i32> = strata.to_vec();
    unique_strata.sort();
    unique_strata.dedup();

    let mut scores = vec![0.0; n];

    let mut total_score_sum = 0.0;
    let total_expected = 0.0;
    let mut total_variance = 0.0;

    for &stratum in &unique_strata {
        let stratum_indices: Vec<usize> = (0..n).filter(|&i| strata[i] == stratum).collect();

        if stratum_indices.is_empty() {
            continue;
        }

        let mut sorted_indices = stratum_indices.clone();
        sorted_indices.sort_by(|&a, &b| {
            time[a]
                .partial_cmp(&time[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n_stratum = sorted_indices.len();

        let mut at_risk: Vec<bool> = vec![true; n_stratum];

        let mut i = 0;
        while i < n_stratum {
            let current_time = time[sorted_indices[i]];

            let mut event_indices: Vec<usize> = Vec::new();
            let mut j = i;
            while j < n_stratum && time[sorted_indices[j]] == current_time {
                if status[sorted_indices[j]] == 1 {
                    event_indices.push(j);
                }
                j += 1;
            }

            if !event_indices.is_empty() {
                let mut at_risk_values: Vec<(usize, f64)> = Vec::new();
                for (k, &idx) in sorted_indices.iter().enumerate() {
                    if at_risk[k] {
                        at_risk_values.push((k, covariate[idx]));
                    }
                }

                let n_at_risk = at_risk_values.len();
                if n_at_risk > 0 {
                    at_risk_values
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    let mut ranks: std::collections::HashMap<usize, f64> =
                        std::collections::HashMap::new();
                    let mut k = 0;
                    while k < n_at_risk {
                        let current_value = at_risk_values[k].1;
                        let mut tie_count = 1;
                        let mut rank_sum = (k + 1) as f64;

                        while k + tie_count < n_at_risk
                            && (at_risk_values[k + tie_count].1 - current_value).abs() < 1e-10
                        {
                            rank_sum += (k + tie_count + 1) as f64;
                            tie_count += 1;
                        }

                        let avg_rank = rank_sum / tie_count as f64;
                        for t in 0..tie_count {
                            ranks.insert(at_risk_values[k + t].0, avg_rank);
                        }
                        k += tie_count;
                    }

                    let mean_rank = (n_at_risk as f64 + 1.0) / 2.0;
                    let var_rank = (n_at_risk as f64 * n_at_risk as f64 - 1.0) / 12.0;

                    for &event_local_idx in &event_indices {
                        if let Some(&rank) = ranks.get(&event_local_idx) {
                            let orig_idx = sorted_indices[event_local_idx];
                            if var_rank > 0.0 {
                                scores[orig_idx] = (rank - mean_rank) / var_rank.sqrt();
                            } else {
                                scores[orig_idx] = 0.0;
                            }
                            total_score_sum += scores[orig_idx];
                        }
                    }

                    let n_events = event_indices.len() as f64;
                    if var_rank > 0.0 {
                        total_variance += n_events / var_rank * var_rank;
                    }
                }
            }

            for item in at_risk.iter_mut().take(j).skip(i) {
                *item = false;
            }

            i = j;
        }
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
        expected: total_expected,
        variance: total_variance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survobrien_basic() {
        // Simple test case
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let covariate = vec![10.0, 20.0, 15.0, 30.0, 25.0];
        let strata = vec![1, 1, 1, 1, 1];

        let result = compute_survobrien(&time, &status, &covariate, &strata);

        // Should have a valid test statistic
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df, 1);
    }

    #[test]
    fn test_survobrien_empty() {
        let result = compute_survobrien(&[], &[], &[], &[]);
        assert_eq!(result.statistic, 0.0);
        assert_eq!(result.p_value, 1.0);
    }

    #[test]
    fn test_survobrien_stratified() {
        let time = vec![1.0, 2.0, 1.0, 2.0];
        let status = vec![1, 0, 1, 0];
        let covariate = vec![10.0, 20.0, 30.0, 40.0];
        let strata = vec![1, 1, 2, 2];

        let result = compute_survobrien(&time, &status, &covariate, &strata);

        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
}
