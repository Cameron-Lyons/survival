use pyo3::prelude::*;

use crate::utilities::statistical::normal_cdf;

#[pyclass]
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
    pub fn new(method: String, confidence_level: f64, tau_method: String) -> Self {
        Self {
            method,
            confidence_level,
            tau_method,
        }
    }
}

#[pyclass]
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
    let config = config
        .unwrap_or_else(|| MetaAnalysisConfig::new("random".to_string(), 0.95, "dl".to_string()));

    let k = effects.len();
    if k < 2 || std_errors.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 studies with matching effect sizes and standard errors",
        ));
    }

    let variances: Vec<f64> = std_errors.iter().map(|se| se * se).collect();

    let (q_stat, q_df) = compute_q_statistic(&effects, &variances);
    let q_pvalue = 1.0 - chi_square_cdf(q_stat, q_df);

    let tau_squared = match config.tau_method.as_str() {
        "dl" => compute_tau_squared_dl(&effects, &variances, q_stat, k),
        "reml" => compute_tau_squared_reml(&effects, &variances),
        "pm" => compute_tau_squared_pm(&effects, &variances),
        _ => compute_tau_squared_dl(&effects, &variances, q_stat, k),
    };

    let (pooled_effect, pooled_se, study_weights) = match config.method.as_str() {
        "fixed" => compute_fixed_effects(&effects, &variances),
        "random" => compute_random_effects(&effects, &variances, tau_squared),
        _ => compute_random_effects(&effects, &variances, tau_squared),
    };

    let z = 1.96;
    let lower_ci = pooled_effect - z * pooled_se;
    let upper_ci = pooled_effect + z * pooled_se;

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

    for _ in 0..100 {
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

        if (tau2_new - tau2).abs() < 1e-8 {
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
    1.0 - crate::utilities::statistical::chi2_sf(x, df)
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
    1.96
}

#[pyclass]
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
        return Err(pyo3::exceptions::PyValueError::new_err(
            "All input vectors must have the same length",
        ));
    }

    let z = 1.96;
    let lower_ci: Vec<f64> = effects
        .iter()
        .zip(std_errors.iter())
        .map(|(e, se)| e - z * se)
        .collect();
    let upper_ci: Vec<f64> = effects
        .iter()
        .zip(std_errors.iter())
        .map(|(e, se)| e + z * se)
        .collect();

    let meta_result = survival_meta_analysis(effects.clone(), std_errors.clone(), config)?;

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

#[pyclass]
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
pub fn publication_bias_tests(
    effects: Vec<f64>,
    std_errors: Vec<f64>,
) -> PyResult<PublicationBiasResult> {
    let k = effects.len();
    if k < 3 || std_errors.len() != k {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 3 studies",
        ));
    }

    let precisions: Vec<f64> = std_errors.iter().map(|se| 1.0 / se).collect();
    let standardized: Vec<f64> = effects
        .iter()
        .zip(std_errors.iter())
        .map(|(e, se)| e / se)
        .collect();

    let (slope, intercept, se_intercept) = weighted_regression(&precisions, &standardized);
    let _ = slope;
    let egger_t = if se_intercept > 0.0 {
        intercept / se_intercept
    } else {
        0.0
    };
    let egger_p = 2.0 * (1.0 - t_distribution_cdf(egger_t.abs(), k - 2));

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

fn weighted_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return (0.0, 0.0, f64::INFINITY);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let residuals: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| yi - (slope * xi + intercept))
        .collect();
    let sse: f64 = residuals.iter().map(|r| r * r).sum();
    let mse = sse / (n - 2.0);

    let se_intercept = (mse * sum_xx / (n * denom)).sqrt();

    (slope, intercept, se_intercept)
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

fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    let t_crits = [
        (1, 6.314),
        (2, 2.920),
        (3, 2.353),
        (5, 2.015),
        (10, 1.812),
        (20, 1.725),
        (30, 1.697),
    ];

    for &(d, crit) in &t_crits {
        if df <= d {
            if t.abs() >= crit {
                return 0.95 + 0.05 * (t.abs() - crit) / crit;
            } else {
                return 0.5 + 0.45 * t.abs() / crit;
            }
        }
    }

    normal_cdf(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_survival_meta_analysis_fixed() {
        let effects = vec![0.5, 0.7, 0.4, 0.6, 0.55];
        let std_errors = vec![0.1, 0.15, 0.12, 0.11, 0.13];

        let config = MetaAnalysisConfig::new("fixed".to_string(), 0.95, "dl".to_string());
        let result = survival_meta_analysis(effects, std_errors, Some(config)).unwrap();

        assert!(result.pooled_effect > 0.0);
        assert!(result.pooled_se > 0.0);
        assert!(result.i_squared >= 0.0 && result.i_squared <= 100.0);
    }

    #[test]
    fn test_survival_meta_analysis_random() {
        let effects = vec![0.5, 0.7, 0.4, 0.6, 0.55];
        let std_errors = vec![0.1, 0.15, 0.12, 0.11, 0.13];

        let config = MetaAnalysisConfig::new("random".to_string(), 0.95, "dl".to_string());
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
}
