#![allow(
    unused_variables,
    unused_imports,
    clippy::too_many_arguments,
    clippy::needless_range_loop
)]

use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utilities::statistical::{ln_gamma, lower_incomplete_gamma};

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum TimeVaryingTestType {
    SlopeTest,
    VarianceTest,
    BreakpointTest,
}

#[pymethods]
impl TimeVaryingTestType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "slope" | "slopetest" => Ok(TimeVaryingTestType::SlopeTest),
            "variance" | "variancetest" => Ok(TimeVaryingTestType::VarianceTest),
            "breakpoint" | "breakpointtest" => Ok(TimeVaryingTestType::BreakpointTest),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown test type. Use 'slope', 'variance', or 'breakpoint'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeVaryingTestConfig {
    #[pyo3(get, set)]
    pub test_type: TimeVaryingTestType,
    #[pyo3(get, set)]
    pub n_windows: usize,
    #[pyo3(get, set)]
    pub min_window_size: usize,
    #[pyo3(get, set)]
    pub significance_level: f64,
    #[pyo3(get, set)]
    pub n_permutations: usize,
}

#[pymethods]
impl TimeVaryingTestConfig {
    #[new]
    #[pyo3(signature = (
        test_type=TimeVaryingTestType::SlopeTest,
        n_windows=5,
        min_window_size=10,
        significance_level=0.05,
        n_permutations=1000
    ))]
    pub fn new(
        test_type: TimeVaryingTestType,
        n_windows: usize,
        min_window_size: usize,
        significance_level: f64,
        n_permutations: usize,
    ) -> PyResult<Self> {
        if n_windows == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "n_windows must be positive",
            ));
        }
        if !(0.0..1.0).contains(&significance_level) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "significance_level must be between 0 and 1",
            ));
        }

        Ok(TimeVaryingTestConfig {
            test_type,
            n_windows,
            min_window_size,
            significance_level,
            n_permutations,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeVaryingTestResult {
    #[pyo3(get)]
    pub feature_idx: usize,
    #[pyo3(get)]
    pub is_time_varying: bool,
    #[pyo3(get)]
    pub test_statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub slope: Option<f64>,
    #[pyo3(get)]
    pub slope_se: Option<f64>,
    #[pyo3(get)]
    pub window_means: Option<Vec<f64>>,
    #[pyo3(get)]
    pub window_variances: Option<Vec<f64>>,
    #[pyo3(get)]
    pub breakpoint_time: Option<f64>,
    #[pyo3(get)]
    pub effect_size: f64,
}

#[pymethods]
impl TimeVaryingTestResult {
    fn __repr__(&self) -> String {
        format!(
            "TimeVaryingTestResult(feature={}, time_varying={}, p={:.4})",
            self.feature_idx, self.is_time_varying, self.p_value
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct TimeVaryingAnalysis {
    #[pyo3(get)]
    pub results: Vec<TimeVaryingTestResult>,
    #[pyo3(get)]
    pub time_varying_features: Vec<usize>,
    #[pyo3(get)]
    pub stable_features: Vec<usize>,
    #[pyo3(get)]
    pub feature_rankings: Vec<(usize, f64)>,
}

#[pymethods]
impl TimeVaryingAnalysis {
    fn __repr__(&self) -> String {
        format!(
            "TimeVaryingAnalysis(n_time_varying={}, n_stable={})",
            self.time_varying_features.len(),
            self.stable_features.len()
        )
    }

    fn get_feature_result(&self, feature_idx: usize) -> Option<TimeVaryingTestResult> {
        self.results
            .iter()
            .find(|r| r.feature_idx == feature_idx)
            .cloned()
    }
}

fn compute_slope_test(
    shap_values: &[f64],
    time_points: &[f64],
    n_times: usize,
) -> (f64, f64, f64, f64) {
    if n_times < 2 {
        return (0.0, 0.0, f64::NAN, 1.0);
    }

    let mean_t: f64 = time_points.iter().sum::<f64>() / n_times as f64;
    let mean_y: f64 = shap_values.iter().sum::<f64>() / n_times as f64;

    let mut ss_tt = 0.0;
    let mut ss_ty = 0.0;

    for i in 0..n_times {
        let t_diff = time_points[i] - mean_t;
        let y_diff = shap_values[i] - mean_y;
        ss_tt += t_diff * t_diff;
        ss_ty += t_diff * y_diff;
    }

    if ss_tt.abs() < 1e-12 {
        return (0.0, 0.0, f64::NAN, 1.0);
    }

    let slope = ss_ty / ss_tt;

    let mut ss_res = 0.0;
    for i in 0..n_times {
        let predicted = mean_y + slope * (time_points[i] - mean_t);
        let residual = shap_values[i] - predicted;
        ss_res += residual * residual;
    }

    let mse = ss_res / (n_times - 2).max(1) as f64;
    let slope_se = (mse / ss_tt).sqrt();

    let t_stat = if slope_se > 1e-12 {
        slope / slope_se
    } else {
        0.0
    };

    let df = (n_times - 2) as f64;
    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));

    (slope, slope_se, t_stat, p_value)
}

fn t_distribution_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }

    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5;

    let beta_cdf = incomplete_beta(a, b, x);

    if t >= 0.0 {
        1.0 - 0.5 * beta_cdf
    } else {
        0.5 * beta_cdf
    }
}

fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    let symmetry_transform = x < (a + 1.0) / (a + b + 2.0);

    if symmetry_transform {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=100 {
        let m = m as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < 1e-10 {
            break;
        }
    }

    h
}

fn compute_variance_test(
    shap_values: &[f64],
    time_points: &[f64],
    n_times: usize,
    n_windows: usize,
) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let window_size = (n_times / n_windows).max(1);
    let mut window_means = Vec::with_capacity(n_windows);
    let mut window_variances = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let start = w * window_size;
        let end = if w == n_windows - 1 {
            n_times
        } else {
            (start + window_size).min(n_times)
        };

        if start >= n_times {
            break;
        }

        let window_vals: Vec<f64> = shap_values[start..end].to_vec();
        let n = window_vals.len();

        if n == 0 {
            continue;
        }

        let mean = window_vals.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            window_vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        window_means.push(mean);
        window_variances.push(var);
    }

    let k = window_variances.len();
    if k < 2 {
        return (window_means, window_variances, 0.0, 1.0);
    }

    let n_total = n_times as f64;
    let pooled_var = window_variances.iter().sum::<f64>() / k as f64;

    let mut bartlett_num = 0.0;
    let mut bartlett_denom = 0.0;

    for (w, &var) in window_variances.iter().enumerate() {
        let n_w = window_size as f64;
        if var > 1e-12 && pooled_var > 1e-12 {
            bartlett_num += (n_w - 1.0) * (var / pooled_var).ln();
        }
        bartlett_denom += 1.0 / (n_w - 1.0);
    }

    let c = 1.0 + (1.0 / (3.0 * (k as f64 - 1.0))) * (bartlett_denom - 1.0 / (n_total - k as f64));

    let chi2_stat = if c > 1e-12 { -bartlett_num / c } else { 0.0 };

    let df = (k - 1) as f64;
    let p_value = 1.0 - chi_squared_cdf(chi2_stat.abs(), df);

    (window_means, window_variances, chi2_stat, p_value)
}

fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(df / 2.0, x / 2.0)
}

fn compute_breakpoint_test(
    shap_values: &[f64],
    time_points: &[f64],
    n_times: usize,
    min_segment: usize,
) -> (Option<f64>, f64, f64) {
    if n_times < 2 * min_segment {
        return (None, 0.0, 1.0);
    }

    let total_mean: f64 = shap_values.iter().sum::<f64>() / n_times as f64;
    let total_ss: f64 = shap_values.iter().map(|&v| (v - total_mean).powi(2)).sum();

    let mut min_ss = total_ss;
    let mut best_breakpoint = None;

    for k in min_segment..(n_times - min_segment) {
        let left = &shap_values[..k];
        let right = &shap_values[k..];

        let left_mean = left.iter().sum::<f64>() / k as f64;
        let right_mean = right.iter().sum::<f64>() / (n_times - k) as f64;

        let left_ss: f64 = left.iter().map(|&v| (v - left_mean).powi(2)).sum();
        let right_ss: f64 = right.iter().map(|&v| (v - right_mean).powi(2)).sum();

        let combined_ss = left_ss + right_ss;

        if combined_ss < min_ss {
            min_ss = combined_ss;
            best_breakpoint = Some(time_points[k]);
        }
    }

    let f_stat = if min_ss > 1e-12 && n_times > 3 {
        ((total_ss - min_ss) / 1.0) / (min_ss / (n_times - 3) as f64)
    } else {
        0.0
    };

    let p_value = 1.0 - f_distribution_cdf(f_stat, 1.0, (n_times - 3) as f64);

    (best_breakpoint, f_stat, p_value)
}

fn f_distribution_cdf(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let x = df2 / (df2 + df1 * f);
    incomplete_beta(df2 / 2.0, df1 / 2.0, x)
}

#[pyfunction]
#[pyo3(signature = (
    shap_values,
    time_points,
    n_samples,
    n_features,
    config
))]
pub fn detect_time_varying_features(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    config: &TimeVaryingTestConfig,
) -> PyResult<TimeVaryingAnalysis> {
    let n_times = time_points.len();

    if shap_values.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shap_values first dimension must match n_samples",
        ));
    }

    let results: Vec<TimeVaryingTestResult> = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let aggregated_shap: Vec<f64> = (0..n_times)
                .map(|t| {
                    shap_values
                        .iter()
                        .map(|sample| sample[f][t].abs())
                        .sum::<f64>()
                        / n_samples as f64
                })
                .collect();

            match config.test_type {
                TimeVaryingTestType::SlopeTest => {
                    let (slope, slope_se, t_stat, p_value) =
                        compute_slope_test(&aggregated_shap, &time_points, n_times);

                    let effect_size = slope.abs()
                        * (time_points.last().unwrap_or(&1.0)
                            - time_points.first().unwrap_or(&0.0));

                    TimeVaryingTestResult {
                        feature_idx: f,
                        is_time_varying: p_value < config.significance_level,
                        test_statistic: t_stat,
                        p_value,
                        slope: Some(slope),
                        slope_se: Some(slope_se),
                        window_means: None,
                        window_variances: None,
                        breakpoint_time: None,
                        effect_size,
                    }
                }
                TimeVaryingTestType::VarianceTest => {
                    let (window_means, window_variances, chi2_stat, p_value) =
                        compute_variance_test(
                            &aggregated_shap,
                            &time_points,
                            n_times,
                            config.n_windows,
                        );

                    let max_var = window_variances.iter().fold(0.0f64, |a, &b| a.max(b));
                    let min_var = window_variances
                        .iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                    let effect_size = if min_var > 1e-12 {
                        (max_var / min_var).ln()
                    } else {
                        0.0
                    };

                    TimeVaryingTestResult {
                        feature_idx: f,
                        is_time_varying: p_value < config.significance_level,
                        test_statistic: chi2_stat,
                        p_value,
                        slope: None,
                        slope_se: None,
                        window_means: Some(window_means),
                        window_variances: Some(window_variances),
                        breakpoint_time: None,
                        effect_size,
                    }
                }
                TimeVaryingTestType::BreakpointTest => {
                    let (breakpoint, f_stat, p_value) = compute_breakpoint_test(
                        &aggregated_shap,
                        &time_points,
                        n_times,
                        config.min_window_size,
                    );

                    let effect_size = f_stat.sqrt();

                    TimeVaryingTestResult {
                        feature_idx: f,
                        is_time_varying: p_value < config.significance_level,
                        test_statistic: f_stat,
                        p_value,
                        slope: None,
                        slope_se: None,
                        window_means: None,
                        window_variances: None,
                        breakpoint_time: breakpoint,
                        effect_size,
                    }
                }
            }
        })
        .collect();

    let time_varying_features: Vec<usize> = results
        .iter()
        .filter(|r| r.is_time_varying)
        .map(|r| r.feature_idx)
        .collect();

    let stable_features: Vec<usize> = results
        .iter()
        .filter(|r| !r.is_time_varying)
        .map(|r| r.feature_idx)
        .collect();

    let mut feature_rankings: Vec<(usize, f64)> = results
        .iter()
        .map(|r| (r.feature_idx, r.effect_size))
        .collect();
    feature_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(TimeVaryingAnalysis {
        results,
        time_varying_features,
        stable_features,
        feature_rankings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config =
            TimeVaryingTestConfig::new(TimeVaryingTestType::SlopeTest, 5, 10, 0.05, 1000).unwrap();
        assert_eq!(config.n_windows, 5);
    }

    #[test]
    fn test_slope_test() {
        let shap = vec![0.12, 0.18, 0.32, 0.38, 0.52];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (slope, se, t_stat, _p_value) = compute_slope_test(&shap, &time, 5);

        assert!((slope - 0.1).abs() < 0.05);
        assert!(se > 0.0);
        assert!(t_stat.abs() > 0.0);
    }

    #[test]
    fn test_variance_test() {
        let shap = vec![0.1, 0.15, 0.12, 0.5, 0.6, 0.55, 0.1, 0.12, 0.11, 0.58];
        let time: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (means, vars, stat, p) = compute_variance_test(&shap, &time, 10, 2);

        assert_eq!(means.len(), 2);
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_breakpoint_test() {
        let shap: Vec<f64> = (0..20).map(|i| if i < 10 { 0.1 } else { 0.5 }).collect();
        let time: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (bp, stat, p) = compute_breakpoint_test(&shap, &time, 20, 3);

        assert!(bp.is_some());
    }

    #[test]
    fn test_detect_time_varying() {
        let n_samples = 5;
        let n_features = 3;
        let n_times = 10;

        let shap_values: Vec<Vec<Vec<f64>>> = (0..n_samples)
            .map(|_| {
                (0..n_features)
                    .map(|f| {
                        (0..n_times)
                            .map(|t| if f == 0 { t as f64 * 0.1 } else { 0.5 })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let time_points: Vec<f64> = (0..n_times).map(|t| t as f64).collect();

        let config =
            TimeVaryingTestConfig::new(TimeVaryingTestType::SlopeTest, 5, 2, 0.05, 100).unwrap();

        let result =
            detect_time_varying_features(shap_values, time_points, n_samples, n_features, &config)
                .unwrap();

        assert_eq!(result.results.len(), n_features);
    }
}
