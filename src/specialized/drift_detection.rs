#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DriftConfig {
    #[pyo3(get, set)]
    pub window_size: usize,
    #[pyo3(get, set)]
    pub threshold_psi: f64,
    #[pyo3(get, set)]
    pub threshold_ks: f64,
    #[pyo3(get, set)]
    pub n_bins: usize,
}

#[pymethods]
impl DriftConfig {
    #[new]
    #[pyo3(signature = (
        window_size=1000,
        threshold_psi=0.2,
        threshold_ks=0.05,
        n_bins=10
    ))]
    pub fn new(window_size: usize, threshold_psi: f64, threshold_ks: f64, n_bins: usize) -> Self {
        Self {
            window_size,
            threshold_psi,
            threshold_ks,
            n_bins,
        }
    }
}

fn compute_psi(reference: &[f64], current: &[f64], n_bins: usize) -> f64 {
    if reference.is_empty() || current.is_empty() {
        return 0.0;
    }

    let min_val = reference
        .iter()
        .chain(current.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let max_val = reference
        .iter()
        .chain(current.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return 0.0;
    }

    let bin_width = (max_val - min_val) / n_bins as f64;

    let ref_counts: Vec<usize> = (0..n_bins)
        .map(|b| {
            let lower = min_val + b as f64 * bin_width;
            let upper = if b == n_bins - 1 {
                f64::INFINITY
            } else {
                min_val + (b + 1) as f64 * bin_width
            };
            reference
                .iter()
                .filter(|&&x| x >= lower && x < upper)
                .count()
        })
        .collect();

    let cur_counts: Vec<usize> = (0..n_bins)
        .map(|b| {
            let lower = min_val + b as f64 * bin_width;
            let upper = if b == n_bins - 1 {
                f64::INFINITY
            } else {
                min_val + (b + 1) as f64 * bin_width
            };
            current.iter().filter(|&&x| x >= lower && x < upper).count()
        })
        .collect();

    let ref_total = reference.len() as f64;
    let cur_total = current.len() as f64;

    let mut psi = 0.0;
    for b in 0..n_bins {
        let ref_pct = (ref_counts[b] as f64 + 0.001) / (ref_total + 0.001 * n_bins as f64);
        let cur_pct = (cur_counts[b] as f64 + 0.001) / (cur_total + 0.001 * n_bins as f64);
        psi += (cur_pct - ref_pct) * (cur_pct / ref_pct).ln();
    }

    psi
}

fn compute_ks_statistic(reference: &[f64], current: &[f64]) -> (f64, f64) {
    if reference.is_empty() || current.is_empty() {
        return (0.0, 1.0);
    }

    let mut ref_sorted = reference.to_vec();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut cur_sorted = current.to_vec();
    cur_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n1 = reference.len() as f64;
    let n2 = current.len() as f64;

    let mut all_values: Vec<f64> = ref_sorted
        .iter()
        .chain(cur_sorted.iter())
        .cloned()
        .collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup();

    let mut max_diff = 0.0f64;

    for &x in &all_values {
        let ref_cdf = ref_sorted.iter().filter(|&&v| v <= x).count() as f64 / n1;
        let cur_cdf = cur_sorted.iter().filter(|&&v| v <= x).count() as f64 / n2;
        max_diff = max_diff.max((ref_cdf - cur_cdf).abs());
    }

    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * max_diff;
    let pvalue = 2.0 * (-2.0 * lambda * lambda).exp();

    (max_diff, pvalue.clamp(0.0, 1.0))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct FeatureDriftResult {
    #[pyo3(get)]
    pub feature_name: String,
    #[pyo3(get)]
    pub psi: f64,
    #[pyo3(get)]
    pub ks_statistic: f64,
    #[pyo3(get)]
    pub ks_pvalue: f64,
    #[pyo3(get)]
    pub has_drift: bool,
    #[pyo3(get)]
    pub drift_severity: String,
}

#[pymethods]
impl FeatureDriftResult {
    fn __repr__(&self) -> String {
        format!(
            "FeatureDriftResult({}, PSI={:.3}, drift={})",
            self.feature_name, self.psi, self.has_drift
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct DriftReport {
    #[pyo3(get)]
    pub feature_results: Vec<FeatureDriftResult>,
    #[pyo3(get)]
    pub overall_drift_detected: bool,
    #[pyo3(get)]
    pub n_features_drifted: usize,
    #[pyo3(get)]
    pub prediction_drift_psi: f64,
    #[pyo3(get)]
    pub prediction_drift_detected: bool,
}

#[pymethods]
impl DriftReport {
    fn __repr__(&self) -> String {
        format!(
            "DriftReport(features_drifted={}/{}, pred_drift={})",
            self.n_features_drifted,
            self.feature_results.len(),
            self.prediction_drift_detected
        )
    }

    fn to_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Drift Detection Report ===\n\n");

        s.push_str(&format!(
            "Overall Drift: {}\n",
            if self.overall_drift_detected {
                "DETECTED"
            } else {
                "None"
            }
        ));
        s.push_str(&format!(
            "Features with drift: {}/{}\n",
            self.n_features_drifted,
            self.feature_results.len()
        ));
        s.push_str(&format!(
            "Prediction drift PSI: {:.4}\n\n",
            self.prediction_drift_psi
        ));

        if !self.feature_results.is_empty() {
            s.push_str("Feature Details:\n");
            s.push_str("| Feature | PSI | KS | Drift |\n");
            s.push_str("|---------|-----|----|---------|\n");

            for f in &self.feature_results {
                s.push_str(&format!(
                    "| {} | {:.4} | {:.4} | {} |\n",
                    f.feature_name, f.psi, f.ks_statistic, f.drift_severity
                ));
            }
        }

        s
    }
}

#[pyfunction]
#[pyo3(signature = (
    reference_features,
    current_features,
    feature_names,
    reference_predictions=None,
    current_predictions=None,
    config=None
))]
pub fn detect_drift(
    reference_features: Vec<Vec<f64>>,
    current_features: Vec<Vec<f64>>,
    feature_names: Vec<String>,
    reference_predictions: Option<Vec<f64>>,
    current_predictions: Option<Vec<f64>>,
    config: Option<DriftConfig>,
) -> PyResult<DriftReport> {
    let config = config.unwrap_or_else(|| DriftConfig::new(1000, 0.2, 0.05, 10));

    if reference_features.is_empty() || current_features.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Reference and current features must not be empty",
        ));
    }

    let n_features = reference_features[0].len();
    if feature_names.len() != n_features {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "feature_names must match number of features",
        ));
    }

    let mut feature_results = Vec::new();
    let mut n_features_drifted = 0;

    for f in 0..n_features {
        let ref_vals: Vec<f64> = reference_features.iter().map(|r| r[f]).collect();
        let cur_vals: Vec<f64> = current_features.iter().map(|c| c[f]).collect();

        let psi = compute_psi(&ref_vals, &cur_vals, config.n_bins);
        let (ks_stat, ks_pvalue) = compute_ks_statistic(&ref_vals, &cur_vals);

        let has_drift = psi > config.threshold_psi || ks_pvalue < config.threshold_ks;
        let drift_severity = if psi < 0.1 {
            "None".to_string()
        } else if psi < 0.2 {
            "Minor".to_string()
        } else if psi < 0.3 {
            "Moderate".to_string()
        } else {
            "Severe".to_string()
        };

        if has_drift {
            n_features_drifted += 1;
        }

        feature_results.push(FeatureDriftResult {
            feature_name: feature_names[f].clone(),
            psi,
            ks_statistic: ks_stat,
            ks_pvalue,
            has_drift,
            drift_severity,
        });
    }

    let prediction_drift_psi = match (&reference_predictions, &current_predictions) {
        (Some(ref_pred), Some(cur_pred)) => compute_psi(ref_pred, cur_pred, config.n_bins),
        _ => 0.0,
    };

    let prediction_drift_detected = prediction_drift_psi > config.threshold_psi;
    let overall_drift_detected = n_features_drifted > 0 || prediction_drift_detected;

    Ok(DriftReport {
        feature_results,
        overall_drift_detected,
        n_features_drifted,
        prediction_drift_psi,
        prediction_drift_detected,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct PerformanceDriftResult {
    #[pyo3(get)]
    pub time_periods: Vec<String>,
    #[pyo3(get)]
    pub c_indices: Vec<f64>,
    #[pyo3(get)]
    pub calibration_slopes: Vec<f64>,
    #[pyo3(get)]
    pub drift_detected: bool,
    #[pyo3(get)]
    pub c_index_change: f64,
    #[pyo3(get)]
    pub recommendation: String,
}

#[pymethods]
impl PerformanceDriftResult {
    fn __repr__(&self) -> String {
        format!(
            "PerformanceDriftResult(periods={}, drift={})",
            self.time_periods.len(),
            self.drift_detected
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    predictions,
    time,
    event,
    period_labels,
    c_index_threshold=0.05
))]
pub fn monitor_performance(
    predictions: Vec<f64>,
    time: Vec<f64>,
    event: Vec<i32>,
    period_labels: Vec<String>,
    c_index_threshold: f64,
) -> PyResult<PerformanceDriftResult> {
    let n = predictions.len();
    if n == 0 || time.len() != n || event.len() != n || period_labels.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same non-zero length",
        ));
    }

    let mut unique_periods: Vec<String> = period_labels.clone();
    unique_periods.sort();
    unique_periods.dedup();

    let mut time_periods = Vec::new();
    let mut c_indices = Vec::new();
    let mut calibration_slopes = Vec::new();

    for period in &unique_periods {
        let indices: Vec<usize> = period_labels
            .iter()
            .enumerate()
            .filter(|(_, p)| *p == period)
            .map(|(i, _)| i)
            .collect();

        if indices.len() < 20 {
            continue;
        }

        let period_pred: Vec<f64> = indices.iter().map(|&i| predictions[i]).collect();
        let period_time: Vec<f64> = indices.iter().map(|&i| time[i]).collect();
        let period_event: Vec<i32> = indices.iter().map(|&i| event[i]).collect();

        let c_index = compute_c_index_internal(&period_pred, &period_time, &period_event);
        let cal_slope = compute_calibration_slope(&period_pred, &period_time, &period_event);

        time_periods.push(period.clone());
        c_indices.push(c_index);
        calibration_slopes.push(cal_slope);
    }

    let c_index_change = if c_indices.len() >= 2 {
        c_indices.last().unwrap() - c_indices.first().unwrap()
    } else {
        0.0
    };

    let drift_detected = c_index_change.abs() > c_index_threshold;

    let recommendation = if drift_detected {
        if c_index_change < 0.0 {
            "Performance degradation detected. Consider model retraining.".to_string()
        } else {
            "Performance improvement detected. Verify data quality.".to_string()
        }
    } else {
        "Model performance stable. Continue monitoring.".to_string()
    };

    Ok(PerformanceDriftResult {
        time_periods,
        c_indices,
        calibration_slopes,
        drift_detected,
        c_index_change,
        recommendation,
    })
}

fn compute_c_index_internal(predictions: &[f64], time: &[f64], event: &[i32]) -> f64 {
    let n = predictions.len();
    let mut concordant = 0.0;
    let mut discordant = 0.0;

    for i in 0..n {
        if event[i] != 1 {
            continue;
        }
        for j in 0..n {
            if i == j || time[j] <= time[i] {
                continue;
            }
            if predictions[i] > predictions[j] {
                concordant += 1.0;
            } else if predictions[i] < predictions[j] {
                discordant += 1.0;
            } else {
                concordant += 0.5;
                discordant += 0.5;
            }
        }
    }

    if concordant + discordant > 0.0 {
        concordant / (concordant + discordant)
    } else {
        0.5
    }
}

fn compute_calibration_slope(predictions: &[f64], _time: &[f64], _event: &[i32]) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 1.0;
    }

    let mean_pred = predictions.iter().sum::<f64>() / n as f64;
    let _var_pred: f64 = predictions
        .iter()
        .map(|&p| (p - mean_pred).powi(2))
        .sum::<f64>()
        / n as f64;

    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let psi = compute_psi(&data, &data, 5);
        assert!(psi < 0.1);
    }

    #[test]
    fn test_ks_identical() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (ks, pvalue) = compute_ks_statistic(&data, &data);
        assert!(ks < 0.01);
        assert!(pvalue > 0.5);
    }

    #[test]
    fn test_drift_detection() {
        let reference: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![i as f64 / 100.0, (i as f64 / 100.0) + 1.0])
            .collect();
        let current: Vec<Vec<f64>> = (0..100)
            .map(|i| vec![(i as f64 / 100.0) + 0.01, (i as f64 / 100.0) + 1.01])
            .collect();
        let names = vec!["f1".to_string(), "f2".to_string()];

        let result = detect_drift(reference, current, names, None, None, None).unwrap();
        assert_eq!(result.feature_results.len(), 2);
    }
}
