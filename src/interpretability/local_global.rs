#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum ViewRecommendation {
    UseGlobal,
    UseLocal,
    UseBoth,
}

#[pymethods]
impl ViewRecommendation {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "global" | "useglobal" => Ok(ViewRecommendation::UseGlobal),
            "local" | "uselocal" => Ok(ViewRecommendation::UseLocal),
            "both" | "useboth" => Ok(ViewRecommendation::UseBoth),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown recommendation. Use 'global', 'local', or 'both'",
            )),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            ViewRecommendation::UseGlobal => "UseGlobal".to_string(),
            ViewRecommendation::UseLocal => "UseLocal".to_string(),
            ViewRecommendation::UseBoth => "UseBoth".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LocalGlobalConfig {
    #[pyo3(get, set)]
    pub heterogeneity_threshold: f64,
    #[pyo3(get, set)]
    pub monotonicity_threshold: f64,
    #[pyo3(get, set)]
    pub interaction_threshold: f64,
    #[pyo3(get, set)]
    pub n_bootstrap: usize,
    #[pyo3(get, set)]
    pub confidence_level: f64,
}

#[pymethods]
impl LocalGlobalConfig {
    #[new]
    #[pyo3(signature = (
        heterogeneity_threshold=0.3,
        monotonicity_threshold=0.8,
        interaction_threshold=0.2,
        n_bootstrap=100,
        confidence_level=0.95
    ))]
    pub fn new(
        heterogeneity_threshold: f64,
        monotonicity_threshold: f64,
        interaction_threshold: f64,
        n_bootstrap: usize,
        confidence_level: f64,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&heterogeneity_threshold) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "heterogeneity_threshold must be between 0 and 1",
            ));
        }
        if !(0.0..=1.0).contains(&monotonicity_threshold) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "monotonicity_threshold must be between 0 and 1",
            ));
        }

        Ok(LocalGlobalConfig {
            heterogeneity_threshold,
            monotonicity_threshold,
            interaction_threshold,
            n_bootstrap,
            confidence_level,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FeatureViewAnalysis {
    #[pyo3(get)]
    pub feature_idx: usize,
    #[pyo3(get)]
    pub recommendation: ViewRecommendation,
    #[pyo3(get)]
    pub heterogeneity_score: f64,
    #[pyo3(get)]
    pub monotonicity_score: f64,
    #[pyo3(get)]
    pub interaction_score: f64,
    #[pyo3(get)]
    pub global_mean_importance: f64,
    #[pyo3(get)]
    pub local_importance_std: f64,
    #[pyo3(get)]
    pub confidence_interval_width: f64,
    #[pyo3(get)]
    pub explanation: String,
}

#[pymethods]
impl FeatureViewAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "FeatureViewAnalysis(feature={}, rec={:?}, het={:.3})",
            self.feature_idx, self.recommendation, self.heterogeneity_score
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LocalGlobalResult {
    #[pyo3(get)]
    pub analyses: Vec<FeatureViewAnalysis>,
    #[pyo3(get)]
    pub global_features: Vec<usize>,
    #[pyo3(get)]
    pub local_features: Vec<usize>,
    #[pyo3(get)]
    pub both_features: Vec<usize>,
    #[pyo3(get)]
    pub summary_statistics: LocalGlobalSummary,
}

#[pymethods]
impl LocalGlobalResult {
    fn __repr__(&self) -> String {
        format!(
            "LocalGlobalResult(global={}, local={}, both={})",
            self.global_features.len(),
            self.local_features.len(),
            self.both_features.len()
        )
    }

    fn get_feature_analysis(&self, feature_idx: usize) -> Option<FeatureViewAnalysis> {
        self.analyses
            .iter()
            .find(|a| a.feature_idx == feature_idx)
            .cloned()
    }

    fn features_by_recommendation(&self, rec: ViewRecommendation) -> Vec<usize> {
        self.analyses
            .iter()
            .filter(|a| a.recommendation == rec)
            .map(|a| a.feature_idx)
            .collect()
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct LocalGlobalSummary {
    #[pyo3(get)]
    pub mean_heterogeneity: f64,
    #[pyo3(get)]
    pub mean_monotonicity: f64,
    #[pyo3(get)]
    pub mean_interaction: f64,
    #[pyo3(get)]
    pub n_features: usize,
    #[pyo3(get)]
    pub proportion_global: f64,
    #[pyo3(get)]
    pub proportion_local: f64,
    #[pyo3(get)]
    pub proportion_both: f64,
}

fn compute_heterogeneity(shap_values: &[f64]) -> f64 {
    if shap_values.is_empty() {
        return 0.0;
    }

    let n = shap_values.len() as f64;
    let mean = shap_values.iter().sum::<f64>() / n;

    if mean.abs() < 1e-12 {
        return 0.0;
    }

    let variance = shap_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let cv = variance.sqrt() / mean.abs();

    (cv / (1.0 + cv)).clamp(0.0, 1.0)
}

fn compute_monotonicity(feature_values: &[f64], shap_values: &[f64]) -> f64 {
    if feature_values.len() < 2 || feature_values.len() != shap_values.len() {
        return 1.0;
    }

    let mut indices: Vec<usize> = (0..feature_values.len()).collect();
    indices.sort_by(|&a, &b| {
        feature_values[a]
            .partial_cmp(&feature_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_shap: Vec<f64> = indices.iter().map(|&i| shap_values[i]).collect();

    let mut increasing = 0;
    let mut decreasing = 0;
    let mut total = 0;

    for i in 1..sorted_shap.len() {
        let diff = sorted_shap[i] - sorted_shap[i - 1];
        if diff > 1e-10 {
            increasing += 1;
        } else if diff < -1e-10 {
            decreasing += 1;
        }
        total += 1;
    }

    if total == 0 {
        return 1.0;
    }

    let max_direction = increasing.max(decreasing) as f64;
    max_direction / total as f64
}

fn compute_interaction_score(
    shap_values: &[Vec<Vec<f64>>],
    feature_idx: usize,
    n_samples: usize,
    n_features: usize,
    n_times: usize,
) -> f64 {
    let target_shap: Vec<f64> = (0..n_samples)
        .flat_map(|s| shap_values[s][feature_idx].iter().copied())
        .collect();

    let target_mean = target_shap.iter().sum::<f64>() / target_shap.len() as f64;
    let target_var: f64 = target_shap.iter().map(|&x| (x - target_mean).powi(2)).sum();

    if target_var < 1e-12 {
        return 0.0;
    }

    let mut total_corr = 0.0;
    let mut count = 0;

    for other_f in 0..n_features {
        if other_f == feature_idx {
            continue;
        }

        let other_shap: Vec<f64> = (0..n_samples)
            .flat_map(|s| shap_values[s][other_f].iter().copied())
            .collect();

        let other_mean = other_shap.iter().sum::<f64>() / other_shap.len() as f64;
        let other_var: f64 = other_shap.iter().map(|&x| (x - other_mean).powi(2)).sum();

        if other_var < 1e-12 {
            continue;
        }

        let covar: f64 = target_shap
            .iter()
            .zip(other_shap.iter())
            .map(|(&a, &b)| (a - target_mean) * (b - other_mean))
            .sum();

        let corr = covar / (target_var.sqrt() * other_var.sqrt());
        total_corr += corr.abs();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    total_corr / count as f64
}

fn compute_confidence_interval_width(
    shap_values: &[f64],
    n_bootstrap: usize,
    confidence_level: f64,
    seed: u64,
) -> f64 {
    if shap_values.is_empty() || n_bootstrap == 0 {
        return 0.0;
    }

    let n = shap_values.len();
    let mut rng = fastrand::Rng::with_seed(seed);

    let bootstrap_means: Vec<f64> = (0..n_bootstrap)
        .map(|_| {
            let sum: f64 = (0..n).map(|_| shap_values[rng.usize(0..n)]).sum();
            sum / n as f64
        })
        .collect();

    let mut sorted = bootstrap_means;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    let lower = sorted.get(lower_idx).copied().unwrap_or(0.0);
    let upper = sorted
        .get(upper_idx.min(n_bootstrap - 1))
        .copied()
        .unwrap_or(0.0);

    upper - lower
}

fn generate_explanation(
    heterogeneity: f64,
    monotonicity: f64,
    interaction: f64,
    recommendation: ViewRecommendation,
    config: &LocalGlobalConfig,
) -> String {
    let mut reasons = Vec::new();

    match recommendation {
        ViewRecommendation::UseGlobal => {
            reasons.push("Feature has consistent effects across samples".to_string());
            if monotonicity >= config.monotonicity_threshold {
                reasons.push(format!(
                    "monotonic relationship (score: {:.2})",
                    monotonicity
                ));
            }
            if heterogeneity < config.heterogeneity_threshold {
                reasons.push(format!("low heterogeneity (score: {:.2})", heterogeneity));
            }
        }
        ViewRecommendation::UseLocal => {
            reasons.push("Feature effects vary significantly across samples".to_string());
            if heterogeneity >= config.heterogeneity_threshold {
                reasons.push(format!("high heterogeneity (score: {:.2})", heterogeneity));
            }
            if interaction >= config.interaction_threshold {
                reasons.push(format!(
                    "significant interactions (score: {:.2})",
                    interaction
                ));
            }
        }
        ViewRecommendation::UseBoth => {
            reasons.push("Feature shows mixed behavior".to_string());
            reasons
                .push("global view for overall trend, local for individual variation".to_string());
        }
    }

    reasons.join("; ")
}

fn determine_recommendation(
    heterogeneity: f64,
    monotonicity: f64,
    interaction: f64,
    config: &LocalGlobalConfig,
) -> ViewRecommendation {
    let is_heterogeneous = heterogeneity >= config.heterogeneity_threshold;
    let is_monotonic = monotonicity >= config.monotonicity_threshold;
    let has_interactions = interaction >= config.interaction_threshold;

    if !is_heterogeneous && is_monotonic && !has_interactions {
        ViewRecommendation::UseGlobal
    } else if is_heterogeneous || has_interactions {
        if is_monotonic {
            ViewRecommendation::UseBoth
        } else {
            ViewRecommendation::UseLocal
        }
    } else {
        ViewRecommendation::UseBoth
    }
}

#[pyfunction]
#[pyo3(signature = (
    shap_values,
    feature_values,
    n_samples,
    n_features,
    n_times,
    config,
    seed=None
))]
pub fn analyze_local_global(
    shap_values: Vec<Vec<Vec<f64>>>,
    feature_values: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    n_times: usize,
    config: &LocalGlobalConfig,
    seed: Option<u64>,
) -> PyResult<LocalGlobalResult> {
    if shap_values.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shap_values first dimension must match n_samples",
        ));
    }
    if feature_values.len() != n_samples * n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_values length must match n_samples * n_features",
        ));
    }

    let base_seed = seed.unwrap_or(42);

    let analyses: Vec<FeatureViewAnalysis> = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let aggregated_shap: Vec<f64> = (0..n_samples)
                .map(|s| shap_values[s][f].iter().map(|v| v.abs()).sum::<f64>() / n_times as f64)
                .collect();

            let feature_vals: Vec<f64> = (0..n_samples)
                .map(|s| feature_values[s * n_features + f])
                .collect();

            let heterogeneity = compute_heterogeneity(&aggregated_shap);
            let monotonicity = compute_monotonicity(&feature_vals, &aggregated_shap);
            let interaction =
                compute_interaction_score(&shap_values, f, n_samples, n_features, n_times);

            let recommendation =
                determine_recommendation(heterogeneity, monotonicity, interaction, config);

            let global_mean = aggregated_shap.iter().sum::<f64>() / n_samples as f64;
            let local_std = {
                let var = aggregated_shap
                    .iter()
                    .map(|&x| (x - global_mean).powi(2))
                    .sum::<f64>()
                    / n_samples as f64;
                var.sqrt()
            };

            let ci_width = compute_confidence_interval_width(
                &aggregated_shap,
                config.n_bootstrap,
                config.confidence_level,
                base_seed + f as u64,
            );

            let explanation = generate_explanation(
                heterogeneity,
                monotonicity,
                interaction,
                recommendation,
                config,
            );

            FeatureViewAnalysis {
                feature_idx: f,
                recommendation,
                heterogeneity_score: heterogeneity,
                monotonicity_score: monotonicity,
                interaction_score: interaction,
                global_mean_importance: global_mean,
                local_importance_std: local_std,
                confidence_interval_width: ci_width,
                explanation,
            }
        })
        .collect();

    let global_features: Vec<usize> = analyses
        .iter()
        .filter(|a| a.recommendation == ViewRecommendation::UseGlobal)
        .map(|a| a.feature_idx)
        .collect();

    let local_features: Vec<usize> = analyses
        .iter()
        .filter(|a| a.recommendation == ViewRecommendation::UseLocal)
        .map(|a| a.feature_idx)
        .collect();

    let both_features: Vec<usize> = analyses
        .iter()
        .filter(|a| a.recommendation == ViewRecommendation::UseBoth)
        .map(|a| a.feature_idx)
        .collect();

    let mean_heterogeneity =
        analyses.iter().map(|a| a.heterogeneity_score).sum::<f64>() / n_features as f64;
    let mean_monotonicity =
        analyses.iter().map(|a| a.monotonicity_score).sum::<f64>() / n_features as f64;
    let mean_interaction =
        analyses.iter().map(|a| a.interaction_score).sum::<f64>() / n_features as f64;

    let summary = LocalGlobalSummary {
        mean_heterogeneity,
        mean_monotonicity,
        mean_interaction,
        n_features,
        proportion_global: global_features.len() as f64 / n_features as f64,
        proportion_local: local_features.len() as f64 / n_features as f64,
        proportion_both: both_features.len() as f64 / n_features as f64,
    };

    Ok(LocalGlobalResult {
        analyses,
        global_features,
        local_features,
        both_features,
        summary_statistics: summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config = LocalGlobalConfig::new(0.3, 0.8, 0.2, 100, 0.95).unwrap();
        assert!((config.heterogeneity_threshold - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_heterogeneity_uniform() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let het = compute_heterogeneity(&values);
        assert!(het < 0.1);
    }

    #[test]
    fn test_heterogeneity_varied() {
        let values = vec![0.1, 5.0, 0.2, 4.5, 0.3];
        let het = compute_heterogeneity(&values);
        assert!(het > 0.3);
    }

    #[test]
    fn test_monotonicity_perfect() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shap = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mono = compute_monotonicity(&features, &shap);
        assert!(mono > 0.9);
    }

    #[test]
    fn test_monotonicity_random() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let shap = vec![0.5, 0.1, 0.4, 0.2, 0.3];
        let mono = compute_monotonicity(&features, &shap);
        assert!(mono < 0.8);
    }

    #[test]
    fn test_recommendation_global() {
        let rec = determine_recommendation(
            0.1,
            0.9,
            0.1,
            &LocalGlobalConfig::new(0.3, 0.8, 0.2, 100, 0.95).unwrap(),
        );
        assert_eq!(rec, ViewRecommendation::UseGlobal);
    }

    #[test]
    fn test_recommendation_local() {
        let rec = determine_recommendation(
            0.5,
            0.5,
            0.3,
            &LocalGlobalConfig::new(0.3, 0.8, 0.2, 100, 0.95).unwrap(),
        );
        assert_eq!(rec, ViewRecommendation::UseLocal);
    }

    #[test]
    fn test_analyze_local_global() {
        let n_samples = 10;
        let n_features = 3;
        let n_times = 5;

        let shap_values: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|s| {
                (0..n_features)
                    .map(|f| (0..n_times).map(|t| (s * f * t) as f64 * 0.01).collect())
                    .collect()
            })
            .collect();

        let feature_values: Vec<f64> = (0..n_samples * n_features)
            .map(|i| i as f64 * 0.1)
            .collect();

        let config = LocalGlobalConfig::new(0.3, 0.8, 0.2, 50, 0.95).unwrap();

        let result = analyze_local_global(
            shap_values,
            feature_values,
            n_samples,
            n_features,
            n_times,
            &config,
            Some(42),
        )
        .unwrap();

        assert_eq!(result.analyses.len(), n_features);
        assert_eq!(
            result.global_features.len() + result.local_features.len() + result.both_features.len(),
            n_features
        );
    }
}
