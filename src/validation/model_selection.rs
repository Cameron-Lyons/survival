#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use pyo3::prelude::*;

use crate::utilities::statistical::lower_incomplete_gamma;

#[derive(Debug, Clone)]
#[pyclass]
pub struct ModelSelectionCriteria {
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_params: usize,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub aicc: f64,
    #[pyo3(get)]
    pub hqc: f64,
    #[pyo3(get)]
    pub deviance: f64,
    #[pyo3(get)]
    pub r_squared_mcfadden: f64,
    #[pyo3(get)]
    pub r_squared_cox_snell: f64,
    #[pyo3(get)]
    pub r_squared_nagelkerke: f64,
}

#[pymethods]
impl ModelSelectionCriteria {
    fn __repr__(&self) -> String {
        format!(
            "ModelSelectionCriteria(AIC={:.2}, BIC={:.2}, AICc={:.2})",
            self.aic, self.bic, self.aicc
        )
    }

    fn summary(&self) -> String {
        format!(
            "Model Selection Criteria:\n\
             Log-Likelihood: {:.4}\n\
             Parameters: {}\n\
             Observations: {} (Events: {})\n\
             AIC: {:.4}\n\
             BIC: {:.4}\n\
             AICc: {:.4}\n\
             HQC: {:.4}\n\
             Deviance: {:.4}\n\
             McFadden R²: {:.4}\n\
             Cox-Snell R²: {:.4}\n\
             Nagelkerke R²: {:.4}",
            self.log_likelihood,
            self.n_params,
            self.n_obs,
            self.n_events,
            self.aic,
            self.bic,
            self.aicc,
            self.hqc,
            self.deviance,
            self.r_squared_mcfadden,
            self.r_squared_cox_snell,
            self.r_squared_nagelkerke
        )
    }
}

#[pyfunction]
#[pyo3(signature = (log_likelihood, n_params, n_obs, n_events, null_log_likelihood=None))]
pub fn compute_model_selection_criteria(
    log_likelihood: f64,
    n_params: usize,
    n_obs: usize,
    n_events: usize,
    null_log_likelihood: Option<f64>,
) -> PyResult<ModelSelectionCriteria> {
    if n_obs == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_obs must be greater than 0",
        ));
    }

    let k = n_params as f64;
    let n = n_obs as f64;

    let aic = -2.0 * log_likelihood + 2.0 * k;

    let bic = -2.0 * log_likelihood + k * n.ln();

    let aicc = if n > k + 1.0 {
        aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
    } else {
        f64::INFINITY
    };

    let hqc = -2.0 * log_likelihood + 2.0 * k * n.ln().ln();

    let deviance = -2.0 * log_likelihood;

    let null_ll =
        null_log_likelihood.unwrap_or_else(|| -(n_events as f64) * (n_events as f64 / n).ln());

    let r_squared_mcfadden = if null_ll.abs() > 1e-10 {
        1.0 - log_likelihood / null_ll
    } else {
        0.0
    };

    let r_squared_cox_snell = 1.0 - (2.0 * (null_ll - log_likelihood) / n).exp().recip();

    let max_r2 = 1.0 - (2.0 * null_ll / n).exp();
    let r_squared_nagelkerke = if max_r2.abs() > 1e-10 {
        r_squared_cox_snell / max_r2
    } else {
        0.0
    };

    Ok(ModelSelectionCriteria {
        log_likelihood,
        n_params,
        n_obs,
        n_events,
        aic,
        bic,
        aicc,
        hqc,
        deviance,
        r_squared_mcfadden: r_squared_mcfadden.clamp(0.0, 1.0),
        r_squared_cox_snell: r_squared_cox_snell.clamp(0.0, 1.0),
        r_squared_nagelkerke: r_squared_nagelkerke.clamp(0.0, 1.0),
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvivalModelComparison {
    #[pyo3(get)]
    pub model_names: Vec<String>,
    #[pyo3(get)]
    pub log_likelihoods: Vec<f64>,
    #[pyo3(get)]
    pub n_params: Vec<usize>,
    #[pyo3(get)]
    pub aic_values: Vec<f64>,
    #[pyo3(get)]
    pub bic_values: Vec<f64>,
    #[pyo3(get)]
    pub aic_weights: Vec<f64>,
    #[pyo3(get)]
    pub bic_weights: Vec<f64>,
    #[pyo3(get)]
    pub delta_aic: Vec<f64>,
    #[pyo3(get)]
    pub delta_bic: Vec<f64>,
    #[pyo3(get)]
    pub best_model_aic: String,
    #[pyo3(get)]
    pub best_model_bic: String,
    #[pyo3(get)]
    pub likelihood_ratio_tests: Vec<(String, String, f64, f64, f64)>,
}

#[pymethods]
impl SurvivalModelComparison {
    fn __repr__(&self) -> String {
        format!(
            "SurvivalModelComparison(n_models={}, best_aic={}, best_bic={})",
            self.model_names.len(),
            self.best_model_aic,
            self.best_model_bic
        )
    }

    fn get_ranking_by_aic(&self) -> Vec<(String, f64)> {
        let mut ranking: Vec<(String, f64)> = self
            .model_names
            .iter()
            .zip(self.aic_values.iter())
            .map(|(n, &a)| (n.clone(), a))
            .collect();
        ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ranking
    }

    fn get_ranking_by_bic(&self) -> Vec<(String, f64)> {
        let mut ranking: Vec<(String, f64)> = self
            .model_names
            .iter()
            .zip(self.bic_values.iter())
            .map(|(n, &b)| (n.clone(), b))
            .collect();
        ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        ranking
    }
}

fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0)
}

#[pyfunction]
#[pyo3(signature = (model_names, log_likelihoods, n_params, n_obs))]
pub fn compare_models(
    model_names: Vec<String>,
    log_likelihoods: Vec<f64>,
    n_params: Vec<usize>,
    n_obs: usize,
) -> PyResult<SurvivalModelComparison> {
    let n_models = model_names.len();
    if log_likelihoods.len() != n_models || n_params.len() != n_models {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    if n_models == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one model is required",
        ));
    }

    let n = n_obs as f64;

    let aic_values: Vec<f64> = log_likelihoods
        .iter()
        .zip(n_params.iter())
        .map(|(&ll, &k)| -2.0 * ll + 2.0 * k as f64)
        .collect();

    let bic_values: Vec<f64> = log_likelihoods
        .iter()
        .zip(n_params.iter())
        .map(|(&ll, &k)| -2.0 * ll + k as f64 * n.ln())
        .collect();

    let min_aic = aic_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_aic: Vec<f64> = aic_values.iter().map(|&a| a - min_aic).collect();

    let min_bic = bic_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let delta_bic: Vec<f64> = bic_values.iter().map(|&b| b - min_bic).collect();

    let exp_delta_aic: Vec<f64> = delta_aic.iter().map(|&d| (-0.5 * d).exp()).collect();
    let sum_exp_aic: f64 = exp_delta_aic.iter().sum();
    let aic_weights: Vec<f64> = exp_delta_aic.iter().map(|&e| e / sum_exp_aic).collect();

    let exp_delta_bic: Vec<f64> = delta_bic.iter().map(|&d| (-0.5 * d).exp()).collect();
    let sum_exp_bic: f64 = exp_delta_bic.iter().sum();
    let bic_weights: Vec<f64> = exp_delta_bic.iter().map(|&e| e / sum_exp_bic).collect();

    let best_aic_idx = aic_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_bic_idx = bic_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut likelihood_ratio_tests: Vec<(String, String, f64, f64, f64)> = Vec::new();

    for i in 0..n_models {
        for j in (i + 1)..n_models {
            if n_params[i] != n_params[j] {
                let (nested_idx, full_idx) = if n_params[i] < n_params[j] {
                    (i, j)
                } else {
                    (j, i)
                };

                let lr_stat = 2.0 * (log_likelihoods[full_idx] - log_likelihoods[nested_idx]);
                let df = (n_params[full_idx] - n_params[nested_idx]) as f64;
                let p_value = 1.0 - chi_squared_cdf(lr_stat.max(0.0), df);

                likelihood_ratio_tests.push((
                    model_names[nested_idx].clone(),
                    model_names[full_idx].clone(),
                    lr_stat,
                    df,
                    p_value,
                ));
            }
        }
    }

    Ok(SurvivalModelComparison {
        model_names: model_names.clone(),
        log_likelihoods,
        n_params,
        aic_values,
        bic_values,
        aic_weights,
        bic_weights,
        delta_aic,
        delta_bic,
        best_model_aic: model_names[best_aic_idx].clone(),
        best_model_bic: model_names[best_bic_idx].clone(),
        likelihood_ratio_tests,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CrossValidatedScore {
    #[pyo3(get)]
    pub mean_score: f64,
    #[pyo3(get)]
    pub std_score: f64,
    #[pyo3(get)]
    pub fold_scores: Vec<f64>,
    #[pyo3(get)]
    pub metric: String,
    #[pyo3(get)]
    pub n_folds: usize,
}

#[pymethods]
impl CrossValidatedScore {
    fn __repr__(&self) -> String {
        format!(
            "CrossValidatedScore({}={:.4} +/- {:.4}, folds={})",
            self.metric, self.mean_score, self.std_score, self.n_folds
        )
    }

    fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        let z = 1.96;
        let margin = z * self.std_score / (self.n_folds as f64).sqrt();
        (self.mean_score - margin, self.mean_score + margin)
    }
}

#[pyfunction]
pub fn compute_cv_score(fold_scores: Vec<f64>, metric: String) -> PyResult<CrossValidatedScore> {
    let n_folds = fold_scores.len();
    if n_folds == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "At least one fold score is required",
        ));
    }

    let mean_score = fold_scores.iter().sum::<f64>() / n_folds as f64;
    let variance = fold_scores
        .iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>()
        / n_folds as f64;
    let std_score = variance.sqrt();

    Ok(CrossValidatedScore {
        mean_score,
        std_score,
        fold_scores,
        metric,
        n_folds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selection_criteria() {
        let criteria = compute_model_selection_criteria(-100.0, 5, 200, 50, None).unwrap();

        assert!(criteria.aic > 0.0);
        assert!(criteria.bic > 0.0);
        assert!(criteria.aicc > 0.0);
        assert!(criteria.deviance > 0.0);
    }

    #[test]
    fn test_compare_models() {
        let model_names = vec![
            "Model1".to_string(),
            "Model2".to_string(),
            "Model3".to_string(),
        ];
        let log_likelihoods = vec![-100.0, -95.0, -90.0];
        let n_params = vec![3, 5, 7];

        let result = compare_models(model_names, log_likelihoods, n_params, 200).unwrap();

        assert_eq!(result.model_names.len(), 3);
        assert!(!result.aic_weights.is_empty());
        let weight_sum: f64 = result.aic_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cv_score() {
        let fold_scores = vec![0.75, 0.78, 0.72, 0.76, 0.74];
        let result = compute_cv_score(fold_scores, "c_index".to_string()).unwrap();

        assert_eq!(result.n_folds, 5);
        assert!(result.mean_score > 0.7 && result.mean_score < 0.8);
        assert!(result.std_score >= 0.0);
    }

    #[test]
    fn test_likelihood_ratio_test() {
        let model_names = vec!["Nested".to_string(), "Full".to_string()];
        let log_likelihoods = vec![-105.0, -100.0];
        let n_params = vec![3, 5];

        let result = compare_models(model_names, log_likelihoods, n_params, 200).unwrap();

        assert!(!result.likelihood_ratio_tests.is_empty());
        let (_, _, lr_stat, df, p_value) = &result.likelihood_ratio_tests[0];
        assert!(*lr_stat > 0.0);
        assert!(*df == 2.0);
        assert!(*p_value >= 0.0 && *p_value <= 1.0);
    }
}
