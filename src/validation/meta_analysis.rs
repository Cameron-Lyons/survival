use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::constants::{
    DEFAULT_CONFIDENCE_LEVEL, Z_SCORE_95, normal_ci, normal_ci_bounds, z_score_for_confidence,
};
use crate::internal::statistical::{normal_cdf, student_t_cdf};
use crate::internal::validation::{validate_confidence_level, validate_positive_finite_slice};

const REML_MAX_ITERATIONS: usize = 100;
const REML_TOLERANCE: f64 = 1e-8;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn validate_finite_slice(values: &[f64], field: &str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{field} contains non-finite value {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_method(method: &str) -> PyResult<()> {
    match method {
        "fixed" | "random" => Ok(()),
        _ => Err(value_error("method must be 'fixed' or 'random'")),
    }
}

fn validate_tau_method(tau_method: &str) -> PyResult<()> {
    match tau_method {
        "dl" | "reml" | "pm" => Ok(()),
        _ => Err(value_error("tau_method must be 'dl', 'reml', or 'pm'")),
    }
}

fn validate_config(config: &MetaAnalysisConfig) -> PyResult<()> {
    validate_method(&config.method)?;
    validate_confidence_level(config.confidence_level)?;
    validate_tau_method(&config.tau_method)
}

fn validate_meta_inputs(effects: &[f64], std_errors: &[f64], min_studies: usize) -> PyResult<()> {
    if effects.len() < min_studies || std_errors.len() != effects.len() {
        return Err(value_error(format!(
            "Need at least {min_studies} studies with matching effect sizes and standard errors",
        )));
    }
    validate_finite_slice(effects, "effects")?;
    validate_positive_finite_slice(std_errors, "std_errors")
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MetaAnalysisConfig {
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub confidence_level: f64,
    #[pyo3(get, set)]
    pub tau_method: String,
}

#[pymethods]
impl MetaAnalysisConfig {
    #[new]
    #[pyo3(signature = (method="random".to_string(), confidence_level=0.95, tau_method="dl".to_string()))]
    pub fn new(method: String, confidence_level: f64, tau_method: String) -> PyResult<Self> {
        let config = Self {
            method,
            confidence_level,
            tau_method,
        };
        validate_config(&config)?;
        Ok(config)
    }
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MetaAnalysisResult {
    #[pyo3(get)]
    pub pooled_effect: f64,
    #[pyo3(get)]
    pub pooled_se: f64,
    #[pyo3(get)]
    pub lower_ci: f64,
    #[pyo3(get)]
    pub upper_ci: f64,
    #[pyo3(get)]
    pub z_value: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub tau_squared: f64,
    #[pyo3(get)]
    pub i_squared: f64,
    #[pyo3(get)]
    pub q_statistic: f64,
    #[pyo3(get)]
    pub q_df: usize,
    #[pyo3(get)]
    pub q_pvalue: f64,
    #[pyo3(get)]
    pub h_squared: f64,
    #[pyo3(get)]
    pub study_weights: Vec<f64>,
    #[pyo3(get)]
    pub prediction_interval: (f64, f64),
}

#[pymethods]
impl MetaAnalysisResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pooled_effect: f64,
        pooled_se: f64,
        lower_ci: f64,
        upper_ci: f64,
        z_value: f64,
        p_value: f64,
        tau_squared: f64,
        i_squared: f64,
        q_statistic: f64,
        q_df: usize,
        q_pvalue: f64,
        h_squared: f64,
        study_weights: Vec<f64>,
        prediction_interval: (f64, f64),
    ) -> Self {
        Self {
            pooled_effect,
            pooled_se,
            lower_ci,
            upper_ci,
            z_value,
            p_value,
            tau_squared,
            i_squared,
            q_statistic,
            q_df,
            q_pvalue,
            h_squared,
            study_weights,
            prediction_interval,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (effects, std_errors, config=None))]
pub fn survival_meta_analysis(
    effects: Vec<f64>,
    std_errors: Vec<f64>,
    config: Option<MetaAnalysisConfig>,
) -> PyResult<MetaAnalysisResult> {
    let config = config.unwrap_or(MetaAnalysisConfig {
        method: "random".to_string(),
        confidence_level: DEFAULT_CONFIDENCE_LEVEL,
        tau_method: "dl".to_string(),
    });
    validate_config(&config)?;

    let k = effects.len();
    validate_meta_inputs(&effects, &std_errors, 2)?;

    let variances: Vec<f64> = std_errors.iter().map(|se| se * se).collect();

    let (q_stat, q_df) = compute_q_statistic(&effects, &variances);
    let q_pvalue = 1.0 - chi_square_cdf(q_stat, q_df);

    let tau_squared = match config.tau_method.as_str() {
        "dl" => compute_tau_squared_dl(&effects, &variances, q_stat, k),
        "reml" => compute_tau_squared_reml(&effects, &variances),
        "pm" => compute_tau_squared_pm(&effects, &variances),
        _ => return Err(value_error("tau_method must be 'dl', 'reml', or 'pm'")),
    };

    let (pooled_effect, pooled_se, study_weights) = match config.method.as_str() {
        "fixed" => compute_fixed_effects(&effects, &variances),
        "random" => compute_random_effects(&effects, &variances, tau_squared),
        _ => return Err(value_error("method must be 'fixed' or 'random'")),
    };

    let z = z_score_for_confidence(config.confidence_level);
    let (lower_ci, upper_ci) = normal_ci(pooled_effect, pooled_se, z);

    let z_value = if pooled_se > 0.0 {
        pooled_effect / pooled_se
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal_cdf(z_value.abs()));

    let i_squared = if q_stat > k as f64 - 1.0 {
        100.0 * (q_stat - (k as f64 - 1.0)) / q_stat
    } else {
        0.0
    };

    let h_squared = if k > 1 {
        q_stat / (k as f64 - 1.0)
    } else {
        1.0
    };

    let pred_se = (pooled_se.powi(2) + tau_squared).sqrt();
    let t_crit = t_distribution_quantile(0.975, k - 1);
    let prediction_interval = (
        pooled_effect - t_crit * pred_se,
        pooled_effect + t_crit * pred_se,
    );

    Ok(MetaAnalysisResult {
        pooled_effect,
        pooled_se,
        lower_ci,
        upper_ci,
        z_value,
        p_value,
        tau_squared,
        i_squared,
        q_statistic: q_stat,
        q_df,
        q_pvalue,
        h_squared,
        study_weights,
        prediction_interval,
    })
}

fn compute_q_statistic(effects: &[f64], variances: &[f64]) -> (f64, usize) {
    let k = effects.len();
    let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let sum_weights: f64 = weights.iter().sum();
    let weighted_mean: f64 = effects
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * w)
        .sum::<f64>()
        / sum_weights;

    let q: f64 = effects
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| w * (e - weighted_mean).powi(2))
        .sum();

    (q, k - 1)
}

fn compute_tau_squared_dl(_effects: &[f64], variances: &[f64], q: f64, k: usize) -> f64 {
    let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let sum_w: f64 = weights.iter().sum();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();

    let c = sum_w - sum_w2 / sum_w;

    let tau2 = (q - (k as f64 - 1.0)) / c;
    tau2.max(0.0)
}

fn compute_tau_squared_reml(effects: &[f64], variances: &[f64]) -> f64 {
    let k = effects.len();
    let mut tau2: f64 = 0.0;

    for _ in 0..REML_MAX_ITERATIONS {
        let weights: Vec<f64> = variances.iter().map(|v| 1.0 / (v + tau2)).collect();
        let sum_w: f64 = weights.iter().sum();
        let weighted_mean: f64 = effects
            .iter()
            .zip(weights.iter())
            .map(|(e, w)| e * w)
            .sum::<f64>()
            / sum_w;

        let q: f64 = effects
            .iter()
            .zip(weights.iter())
            .map(|(e, w)| w * (e - weighted_mean).powi(2))
            .sum();

        let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
        let c = sum_w - sum_w2 / sum_w;

        let tau2_new = (q - (k as f64 - 1.0)) / c;
        let tau2_new = tau2_new.max(0.0);

        if (tau2_new - tau2).abs() < REML_TOLERANCE {
            break;
        }
        tau2 = tau2_new;
    }

    tau2
}

fn compute_tau_squared_pm(effects: &[f64], variances: &[f64]) -> f64 {
    let k = effects.len();

    let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let sum_w: f64 = weights.iter().sum();
    let weighted_mean: f64 = effects
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * w)
        .sum::<f64>()
        / sum_w;

    let ss: f64 = effects.iter().map(|e| (e - weighted_mean).powi(2)).sum();

    let mean_var: f64 = variances.iter().sum::<f64>() / k as f64;

    let tau2 = ss / (k as f64 - 1.0) - mean_var;
    tau2.max(0.0)
}

fn compute_fixed_effects(effects: &[f64], variances: &[f64]) -> (f64, f64, Vec<f64>) {
    let weights: Vec<f64> = variances.iter().map(|v| 1.0 / v).collect();
    let sum_weights: f64 = weights.iter().sum();

    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum_weights).collect();

    let pooled_effect: f64 = effects
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * w)
        .sum::<f64>()
        / sum_weights;

    let pooled_variance = 1.0 / sum_weights;
    let pooled_se = pooled_variance.sqrt();

    (pooled_effect, pooled_se, normalized_weights)
}

fn compute_random_effects(
    effects: &[f64],
    variances: &[f64],
    tau_squared: f64,
) -> (f64, f64, Vec<f64>) {
    let weights: Vec<f64> = variances.iter().map(|v| 1.0 / (v + tau_squared)).collect();
    let sum_weights: f64 = weights.iter().sum();

    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum_weights).collect();

    let pooled_effect: f64 = effects
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * w)
        .sum::<f64>()
        / sum_weights;

    let pooled_variance = 1.0 / sum_weights;
    let pooled_se = pooled_variance.sqrt();

    (pooled_effect, pooled_se, normalized_weights)
}

fn chi_square_cdf(x: f64, df: usize) -> f64 {
    if df == 0 || x <= 0.0 {
        return 0.0;
    }
    1.0 - crate::internal::statistical::chi2_sf(x, df)
}

fn t_distribution_quantile(_p: f64, df: usize) -> f64 {
    if df <= 1 {
        return 12.71;
    }
    if df <= 2 {
        return 4.30;
    }
    if df <= 5 {
        return 2.57;
    }
    if df <= 10 {
        return 2.23;
    }
    if df <= 30 {
        return 2.04;
    }
    Z_SCORE_95
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct MetaForestPlotData {
    #[pyo3(get)]
    pub study_names: Vec<String>,
    #[pyo3(get)]
    pub effects: Vec<f64>,
    #[pyo3(get)]
    pub lower_ci: Vec<f64>,
    #[pyo3(get)]
    pub upper_ci: Vec<f64>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub pooled_effect: f64,
    #[pyo3(get)]
    pub pooled_lower: f64,
    #[pyo3(get)]
    pub pooled_upper: f64,
    #[pyo3(get)]
    pub i_squared: f64,
}

#[pymethods]
impl MetaForestPlotData {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        study_names: Vec<String>,
        effects: Vec<f64>,
        lower_ci: Vec<f64>,
        upper_ci: Vec<f64>,
        weights: Vec<f64>,
        pooled_effect: f64,
        pooled_lower: f64,
        pooled_upper: f64,
        i_squared: f64,
    ) -> Self {
        Self {
            study_names,
            effects,
            lower_ci,
            upper_ci,
            weights,
            pooled_effect,
            pooled_lower,
            pooled_upper,
            i_squared,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (study_names, effects, std_errors, config=None))]
pub fn generate_forest_plot_data(
    study_names: Vec<String>,
    effects: Vec<f64>,
    std_errors: Vec<f64>,
    config: Option<MetaAnalysisConfig>,
) -> PyResult<MetaForestPlotData> {
    let k = effects.len();
    if k != study_names.len() || k != std_errors.len() {
        return Err(value_error("All input vectors must have the same length"));
    }
    validate_meta_inputs(&effects, &std_errors, 2)?;

    let config = config.unwrap_or(MetaAnalysisConfig {
        method: "random".to_string(),
        confidence_level: DEFAULT_CONFIDENCE_LEVEL,
        tau_method: "dl".to_string(),
    });
    validate_config(&config)?;
    let z = z_score_for_confidence(config.confidence_level);
    let (lower_ci, upper_ci) = normal_ci_bounds(&effects, &std_errors, z);

    let meta_result = survival_meta_analysis(effects.clone(), std_errors.clone(), Some(config))?;

    Ok(MetaForestPlotData {
        study_names,
        effects,
        lower_ci,
        upper_ci,
        weights: meta_result.study_weights,
        pooled_effect: meta_result.pooled_effect,
        pooled_lower: meta_result.lower_ci,
        pooled_upper: meta_result.upper_ci,
        i_squared: meta_result.i_squared,
    })
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PublicationBiasResult {
    #[pyo3(get)]
    pub egger_intercept: f64,
    #[pyo3(get)]
    pub egger_se: f64,
    #[pyo3(get)]
    pub egger_t: f64,
    #[pyo3(get)]
    pub egger_p: f64,
    #[pyo3(get)]
    pub begg_z: f64,
    #[pyo3(get)]
    pub begg_p: f64,
    #[pyo3(get)]
    pub trim_fill_n: usize,
    #[pyo3(get)]
    pub trim_fill_effect: f64,
}

#[pymethods]
impl PublicationBiasResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        egger_intercept: f64,
        egger_se: f64,
        egger_t: f64,
        egger_p: f64,
        begg_z: f64,
        begg_p: f64,
        trim_fill_n: usize,
        trim_fill_effect: f64,
    ) -> Self {
        Self {
            egger_intercept,
            egger_se,
            egger_t,
            egger_p,
            begg_z,
            begg_p,
            trim_fill_n,
            trim_fill_effect,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (effects, std_errors))]
/// Test publication bias using Egger regression and Begg rank correlation.
///
/// Egger regresses standardized effects on precision with an intercept and uses
/// a two-sided Student-t test with `n - 2` degrees of freedom. Its fields are NaN
/// when the precision slope is numerically unidentifiable or standardized
/// effects are not representable. With zero residual variance, a nonzero
/// intercept has an infinite statistic and zero probability; a zero intercept
/// has an undefined statistic and probability.
pub fn publication_bias_tests(
    effects: Vec<f64>,
    std_errors: Vec<f64>,
) -> PyResult<PublicationBiasResult> {
    let k = effects.len();
    validate_meta_inputs(&effects, &std_errors, 3)?;

    // Scaling the precision column leaves the intercept and its test unchanged.
    // This avoids overflowing reciprocals and makes rank independent of units.
    let min_se = std_errors.iter().copied().fold(f64::INFINITY, f64::min);
    let precisions: Vec<f64> = std_errors.iter().map(|se| min_se / se).collect();
    let standardized: Vec<f64> = effects
        .iter()
        .zip(std_errors.iter())
        .map(|(e, se)| e / se)
        .collect();

    let (intercept, se_intercept, egger_t) =
        egger_regression(&precisions, &standardized).unwrap_or((f64::NAN, f64::NAN, f64::NAN));
    let egger_p = if egger_t.is_nan() {
        f64::NAN
    } else {
        2.0 * student_t_cdf(-egger_t.abs(), (k - 2) as f64)
    };

    let (begg_z, begg_p) = kendall_tau_test(&effects, &std_errors);

    let (trim_fill_n, trim_fill_effect) = trim_and_fill(&effects, &std_errors);

    Ok(PublicationBiasResult {
        egger_intercept: intercept,
        egger_se: se_intercept,
        egger_t,
        egger_p,
        begg_z,
        begg_p,
        trim_fill_n,
        trim_fill_effect,
    })
}

fn egger_regression(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    let n = x.len() as f64;
    if y.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let y_scale = y.iter().map(|value| value.abs()).fold(0.0, f64::max);
    let y_scale = if y_scale == 0.0 { 1.0 } else { y_scale };
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().map(|value| value / y_scale).sum::<f64>() / n;
    let sxx = x.iter().map(|value| (value - mean_x).powi(2)).sum::<f64>();
    let sum_xx = x.iter().map(|value| value * value).sum::<f64>();
    // R's lm.fit uses a default relative column-norm rank tolerance of 1e-7.
    // A dropped precision slope does not identify the two-parameter Egger test.
    if sxx <= 1e-14 * sum_xx {
        return None;
    }

    let sxy = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| (xi - mean_x) * (yi / y_scale - mean_y))
        .sum::<f64>();
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let sse = x
        .iter()
        .zip(y)
        .map(|(xi, yi)| ((yi / y_scale - mean_y) - slope * (xi - mean_x)).powi(2))
        .sum::<f64>();
    // Var(intercept) = MSE * (1/n + mean(x)^2 / Sxx).
    // Scale the response to avoid overflow when squaring residuals.
    let se_intercept = (sse / (n - 2.0)).sqrt() * (1.0 / n + mean_x * mean_x / sxx).sqrt();

    Some((
        intercept * y_scale,
        se_intercept * y_scale,
        intercept / se_intercept,
    ))
}

fn kendall_tau_test(effects: &[f64], std_errors: &[f64]) -> (f64, f64) {
    let k = effects.len();
    if k < 3 {
        return (0.0, 1.0);
    }

    let mut concordant = 0;
    let mut discordant = 0;

    for i in 0..k {
        for j in (i + 1)..k {
            let effect_diff = effects[i] - effects[j];
            let se_diff = std_errors[i] - std_errors[j];

            if effect_diff * se_diff > 0.0 {
                concordant += 1;
            } else if effect_diff * se_diff < 0.0 {
                discordant += 1;
            }
        }
    }

    let n_pairs = k * (k - 1) / 2;
    let tau = (concordant as f64 - discordant as f64) / n_pairs as f64;

    let var_tau = (2.0 * (2.0 * k as f64 + 5.0)) / (9.0 * k as f64 * (k as f64 - 1.0));
    let z = tau / var_tau.sqrt();
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));

    (z, p)
}

fn trim_and_fill(effects: &[f64], std_errors: &[f64]) -> (usize, f64) {
    let variances: Vec<f64> = std_errors.iter().map(|se| se * se).collect();
    let meta_result = survival_meta_analysis(effects.to_vec(), std_errors.to_vec(), None);

    let pooled = match &meta_result {
        Ok(r) => r.pooled_effect,
        Err(_) => effects.iter().sum::<f64>() / effects.len() as f64,
    };

    let deviations: Vec<f64> = effects.iter().map(|e| e - pooled).collect();

    let n_positive = deviations.iter().filter(|&&d| d > 0.0).count();
    let n_negative = deviations.iter().filter(|&&d| d < 0.0).count();

    let n_missing = n_positive.saturating_sub(n_negative);

    let mut augmented_effects = effects.to_vec();
    let mut augmented_variances = variances.clone();

    for i in 0..n_missing.min(effects.len()) {
        let idx = effects.len() - 1 - i;
        let mirrored = 2.0 * pooled - effects[idx];
        augmented_effects.push(mirrored);
        augmented_variances.push(variances[idx]);
    }

    let augmented_se: Vec<f64> = augmented_variances.iter().map(|v| v.sqrt()).collect();
    let adjusted_result = survival_meta_analysis(augmented_effects, augmented_se, None);

    let adjusted_effect = match adjusted_result {
        Ok(r) => r.pooled_effect,
        Err(_) => pooled,
    };

    (n_missing, adjusted_effect)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survival_meta_analysis_fixed() {
        let effects = vec![0.5, 0.7, 0.4, 0.6, 0.55];
        let std_errors = vec![0.1, 0.15, 0.12, 0.11, 0.13];

        let config = MetaAnalysisConfig::new("fixed".to_string(), 0.95, "dl".to_string()).unwrap();
        let result = survival_meta_analysis(effects, std_errors, Some(config)).unwrap();

        assert!(result.pooled_effect > 0.0);
        assert!(result.pooled_se > 0.0);
        assert!(result.i_squared >= 0.0 && result.i_squared <= 100.0);
    }

    #[test]
    fn test_survival_meta_analysis_random() {
        let effects = vec![0.5, 0.7, 0.4, 0.6, 0.55];
        let std_errors = vec![0.1, 0.15, 0.12, 0.11, 0.13];

        let config = MetaAnalysisConfig::new("random".to_string(), 0.95, "dl".to_string()).unwrap();
        let result = survival_meta_analysis(effects, std_errors, Some(config)).unwrap();

        assert!(result.pooled_effect > 0.0);
        assert!(result.tau_squared >= 0.0);
    }

    #[test]
    fn test_forest_plot_data() {
        let study_names = vec![
            "Study A".to_string(),
            "Study B".to_string(),
            "Study C".to_string(),
        ];
        let effects = vec![0.5, 0.7, 0.4];
        let std_errors = vec![0.1, 0.15, 0.12];

        let result = generate_forest_plot_data(study_names, effects, std_errors, None).unwrap();

        assert_eq!(result.study_names.len(), 3);
        assert_eq!(result.effects.len(), 3);
        assert_eq!(result.weights.len(), 3);
    }

    #[test]
    fn test_publication_bias() {
        let effects = vec![0.5, 0.7, 0.4, 0.6, 0.55];
        let std_errors = vec![0.1, 0.15, 0.12, 0.11, 0.13];

        let result = publication_bias_tests(effects, std_errors).unwrap();

        assert!(result.egger_p >= 0.0 && result.egger_p <= 1.0);
        assert!(result.begg_p >= 0.0 && result.begg_p <= 1.0);
    }

    #[test]
    fn egger_matches_r_inference_across_units_and_degrees_of_freedom() {
        #[derive(serde::Deserialize)]
        struct Reference {
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            effects: Vec<f64>,
            std_errors: Vec<f64>,
            egger_intercept: f64,
            egger_se: f64,
            egger_t: f64,
            egger_p: f64,
        }
        let reference: Reference = serde_json::from_str(include_str!(
            "../../python/tests/fixtures/egger_r_reference.json"
        ))
        .unwrap();
        for case in reference.cases {
            for units in [1e-8, 1.0, 1e8] {
                for effect_scale in [-1.0, 1.0, 1e160] {
                    let result = publication_bias_tests(
                        case.effects
                            .iter()
                            .map(|e| e * units * effect_scale)
                            .collect(),
                        case.std_errors.iter().map(|se| se * units).collect(),
                    )
                    .unwrap();
                    let actual = [
                        result.egger_intercept / effect_scale,
                        result.egger_se / effect_scale.abs(),
                        result.egger_t * effect_scale.signum(),
                        result.egger_p,
                    ];
                    let expected = [
                        case.egger_intercept,
                        case.egger_se,
                        case.egger_t,
                        case.egger_p,
                    ];
                    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate()
                    {
                        // Do not hide a lost small probability with absolute tolerance.
                        let absolute = if index == 3 { 0.0 } else { 2e-12 };
                        assert!(
                            (actual - expected).abs() <= absolute + 2e-10 * expected.abs(),
                            "{}: units={units}, effect_scale={effect_scale}, field={index}, {actual} != {expected}",
                            case.id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn egger_marks_unidentifiable_regressions_unavailable() {
        for std_errors in [vec![1.0; 3], vec![1.0, 1.0 + 1e-10, 1.0 + 2e-10]] {
            for units in [1e-8, 1.0, 1e8] {
                let result = publication_bias_tests(
                    vec![units, 2.0 * units, 4.0 * units],
                    std_errors.iter().map(|se| se * units).collect(),
                )
                .unwrap();
                assert!(result.egger_intercept.is_nan());
                assert!(result.egger_se.is_nan());
                assert!(result.egger_t.is_nan());
                assert!(result.egger_p.is_nan());
                assert!(result.begg_p.is_finite());
                assert!(result.trim_fill_effect.is_finite());
            }
        }
        let unrepresentable =
            publication_bias_tests(vec![f64::MAX; 3], vec![0.1, 0.2, 0.3]).unwrap();
        assert!(unrepresentable.egger_intercept.is_nan());
        assert!(unrepresentable.egger_se.is_nan());
        assert!(unrepresentable.egger_t.is_nan());
        assert!(unrepresentable.egger_p.is_nan());
    }

    #[test]
    fn egger_rank_cutoff_retains_estimable_precision_columns() {
        // R lm(y ~ x), x = 1 + (0:2) * delta, y = c(1, 2, 4):
        // rank 1 at delta=1e-7, rank 2 at delta=2e-7.
        for (delta, estimable) in [(1e-7, false), (2e-7, true)] {
            for units in [1e-8, 1.0, 1e8] {
                let std_errors: Vec<f64> = (0..3)
                    .map(|i| units / (1.0 + f64::from(i) * delta))
                    .collect();
                let effects = std_errors
                    .iter()
                    .zip([1.0, 2.0, 4.0])
                    .map(|(se, y)| se * y)
                    .collect();
                let result = publication_bias_tests(effects, std_errors).unwrap();
                for value in [
                    result.egger_intercept,
                    result.egger_se,
                    result.egger_t,
                    result.egger_p,
                ] {
                    assert_eq!(value.is_finite(), estimable);
                }
                if estimable {
                    // This nearly aliased fit loses a few digits in both QR and
                    // centered arithmetic; the inferential result still agrees.
                    assert!((result.egger_p - 0.12103775436571447).abs() < 1e-8);
                }
            }
        }
    }

    #[test]
    fn egger_zero_residual_variance_preserves_undefined_and_infinite_statistics() {
        let std_errors = vec![1.0, 0.5, 0.25];
        let zero = publication_bias_tests(vec![0.0; 3], std_errors.clone()).unwrap();
        assert_eq!(zero.egger_intercept, 0.0);
        assert_eq!(zero.egger_se, 0.0);
        assert!(zero.egger_t.is_nan());
        assert!(zero.egger_p.is_nan());
        for intercept in [-2.0, 2.0] {
            let result = publication_bias_tests(
                std_errors.iter().map(|se| intercept * se).collect(),
                std_errors.clone(),
            )
            .unwrap();
            assert_eq!(result.egger_intercept, intercept);
            assert_eq!(result.egger_se, 0.0);
            assert_eq!(result.egger_t, f64::INFINITY.copysign(intercept));
            assert_eq!(result.egger_p, 0.0);
        }
    }

    #[test]
    fn meta_analysis_rejects_malformed_public_inputs() {
        assert!(MetaAnalysisConfig::new("weird".to_string(), 0.95, "dl".to_string()).is_err());
        assert!(MetaAnalysisConfig::new("random".to_string(), f64::NAN, "dl".to_string()).is_err());
        assert!(MetaAnalysisConfig::new("random".to_string(), 0.95, "bad".to_string()).is_err());

        assert!(survival_meta_analysis(vec![0.5], vec![0.1], None).is_err());
        assert!(survival_meta_analysis(vec![0.5, f64::NAN], vec![0.1, 0.2], None).is_err());
        assert!(survival_meta_analysis(vec![0.5, 0.7], vec![0.1, 0.0], None).is_err());
        assert!(survival_meta_analysis(vec![0.5, 0.7], vec![0.1], None).is_err());
        assert!(
            generate_forest_plot_data(
                vec!["A".to_string(), "B".to_string()],
                vec![0.5, 0.7],
                vec![0.1, f64::INFINITY],
                None,
            )
            .is_err()
        );
        assert!(publication_bias_tests(vec![0.5, 0.7, 0.4], vec![0.1, -0.2, 0.3]).is_err());
    }
}
