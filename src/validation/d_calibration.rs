use crate::utilities::statistical::chi2_sf;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct DCalibrationResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: usize,
    #[pyo3(get)]
    pub n_bins: usize,
    #[pyo3(get)]
    pub observed_counts: Vec<usize>,
    #[pyo3(get)]
    pub expected_counts: Vec<f64>,
    #[pyo3(get)]
    pub bin_edges: Vec<f64>,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub is_calibrated: bool,
}

#[pymethods]
impl DCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        statistic: f64,
        p_value: f64,
        degrees_of_freedom: usize,
        n_bins: usize,
        observed_counts: Vec<usize>,
        expected_counts: Vec<f64>,
        bin_edges: Vec<f64>,
        n_events: usize,
        is_calibrated: bool,
    ) -> Self {
        Self {
            statistic,
            p_value,
            degrees_of_freedom,
            n_bins,
            observed_counts,
            expected_counts,
            bin_edges,
            n_events,
            is_calibrated,
        }
    }
}

pub fn d_calibration_core(
    survival_probs: &[f64],
    status: &[i32],
    n_bins: usize,
) -> DCalibrationResult {
    let events: Vec<f64> = survival_probs
        .iter()
        .zip(status.iter())
        .filter(|(_, s)| **s == 1)
        .map(|(p, _)| *p)
        .collect();

    let n_events = events.len();

    if n_events < n_bins * 2 {
        return DCalibrationResult {
            statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
            n_bins,
            observed_counts: vec![],
            expected_counts: vec![],
            bin_edges: vec![],
            n_events,
            is_calibrated: true,
        };
    }

    let mut bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
    bin_edges[0] = 0.0;
    bin_edges[n_bins] = 1.0 + 1e-10;

    let mut observed_counts = vec![0usize; n_bins];
    for &p in &events {
        for bin_idx in 0..n_bins {
            if p >= bin_edges[bin_idx] && p < bin_edges[bin_idx + 1] {
                observed_counts[bin_idx] += 1;
                break;
            }
        }
    }

    let expected_per_bin = n_events as f64 / n_bins as f64;
    let expected_counts: Vec<f64> = vec![expected_per_bin; n_bins];

    let mut chi2_stat = 0.0;
    for bin_idx in 0..n_bins {
        let observed = observed_counts[bin_idx] as f64;
        let expected = expected_counts[bin_idx];
        if expected > 0.0 {
            chi2_stat += (observed - expected).powi(2) / expected;
        }
    }

    let df = n_bins - 1;
    let p_value = chi2_sf(chi2_stat, df);

    let is_calibrated = p_value >= 0.05;

    bin_edges.pop();

    DCalibrationResult {
        statistic: chi2_stat,
        p_value,
        degrees_of_freedom: df,
        n_bins,
        observed_counts,
        expected_counts,
        bin_edges,
        n_events,
        is_calibrated,
    }
}

#[pyfunction]
#[pyo3(signature = (survival_probs, status, n_bins=None))]
pub fn d_calibration(
    survival_probs: Vec<f64>,
    status: Vec<i32>,
    n_bins: Option<usize>,
) -> PyResult<DCalibrationResult> {
    if survival_probs.len() != status.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survival_probs and status must have the same length",
        ));
    }

    let n_bins = n_bins.unwrap_or(10);
    if n_bins < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_bins must be at least 2",
        ));
    }

    Ok(d_calibration_core(&survival_probs, &status, n_bins))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct OneCalibrationResult {
    #[pyo3(get)]
    pub time_point: f64,
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: usize,
    #[pyo3(get)]
    pub n_groups: usize,
    #[pyo3(get)]
    pub predicted_survival: Vec<f64>,
    #[pyo3(get)]
    pub observed_survival: Vec<f64>,
    #[pyo3(get)]
    pub n_per_group: Vec<usize>,
    #[pyo3(get)]
    pub n_events_per_group: Vec<usize>,
    #[pyo3(get)]
    pub is_calibrated: bool,
}

#[pymethods]
impl OneCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        time_point: f64,
        statistic: f64,
        p_value: f64,
        degrees_of_freedom: usize,
        n_groups: usize,
        predicted_survival: Vec<f64>,
        observed_survival: Vec<f64>,
        n_per_group: Vec<usize>,
        n_events_per_group: Vec<usize>,
        is_calibrated: bool,
    ) -> Self {
        Self {
            time_point,
            statistic,
            p_value,
            degrees_of_freedom,
            n_groups,
            predicted_survival,
            observed_survival,
            n_per_group,
            n_events_per_group,
            is_calibrated,
        }
    }
}

pub fn one_calibration_core(
    time: &[f64],
    status: &[i32],
    predicted_survival_at_t: &[f64],
    time_point: f64,
    n_groups: usize,
) -> OneCalibrationResult {
    let n = time.len();

    if n < n_groups * 5 {
        return OneCalibrationResult {
            time_point,
            statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
            n_groups,
            predicted_survival: vec![],
            observed_survival: vec![],
            n_per_group: vec![],
            n_events_per_group: vec![],
            is_calibrated: true,
        };
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predicted_survival_at_t[a]
            .partial_cmp(&predicted_survival_at_t[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let group_size = n / n_groups;
    let remainder = n % n_groups;

    let mut predicted_survival = Vec::with_capacity(n_groups);
    let mut observed_survival = Vec::with_capacity(n_groups);
    let mut n_per_group = Vec::with_capacity(n_groups);
    let mut n_events_per_group = Vec::with_capacity(n_groups);

    let mut start = 0;
    for g in 0..n_groups {
        let extra = if g < remainder { 1 } else { 0 };
        let end = start + group_size + extra;

        if end <= start {
            continue;
        }

        let group_indices: Vec<usize> = indices[start..end].to_vec();
        let n_in_group = group_indices.len();

        let sum_pred: f64 = group_indices
            .iter()
            .map(|&i| predicted_survival_at_t[i])
            .sum();
        let mean_pred = sum_pred / n_in_group as f64;

        let events_before_t: usize = group_indices
            .iter()
            .filter(|&&i| time[i] <= time_point && status[i] == 1)
            .count();

        let obs_surv = if n_in_group > 0 {
            1.0 - (events_before_t as f64 / n_in_group as f64)
        } else {
            1.0
        };

        predicted_survival.push(mean_pred);
        observed_survival.push(obs_surv);
        n_per_group.push(n_in_group);
        n_events_per_group.push(events_before_t);

        start = end;
    }

    let actual_groups = predicted_survival.len();
    if actual_groups < 2 {
        return OneCalibrationResult {
            time_point,
            statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
            n_groups: actual_groups,
            predicted_survival,
            observed_survival,
            n_per_group,
            n_events_per_group,
            is_calibrated: true,
        };
    }

    let mut chi2_stat = 0.0;
    for g in 0..actual_groups {
        let n_g = n_per_group[g] as f64;
        let pred = predicted_survival[g];

        let expected_events = n_g * (1.0 - pred);
        let observed_events = n_events_per_group[g] as f64;

        if expected_events > 0.0 && expected_events < n_g {
            let variance = n_g * pred * (1.0 - pred);
            if variance > 1e-10 {
                chi2_stat += (observed_events - expected_events).powi(2) / variance;
            }
        }
    }

    let df = actual_groups.saturating_sub(1);
    let p_value = if df > 0 { chi2_sf(chi2_stat, df) } else { 1.0 };

    let is_calibrated = p_value >= 0.05;

    OneCalibrationResult {
        time_point,
        statistic: chi2_stat,
        p_value,
        degrees_of_freedom: df,
        n_groups: actual_groups,
        predicted_survival,
        observed_survival,
        n_per_group,
        n_events_per_group,
        is_calibrated,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_survival_at_t, time_point, n_groups=None))]
pub fn one_calibration(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_survival_at_t: Vec<f64>,
    time_point: f64,
    n_groups: Option<usize>,
) -> PyResult<OneCalibrationResult> {
    let n = time.len();
    if n != status.len() || n != predicted_survival_at_t.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_groups = n_groups.unwrap_or(10);
    if n_groups < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_groups must be at least 2",
        ));
    }

    Ok(one_calibration_core(
        &time,
        &status,
        &predicted_survival_at_t,
        time_point,
        n_groups,
    ))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct CalibrationPlotData {
    #[pyo3(get)]
    pub predicted: Vec<f64>,
    #[pyo3(get)]
    pub observed: Vec<f64>,
    #[pyo3(get)]
    pub n_per_group: Vec<usize>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub ici: f64,
    #[pyo3(get)]
    pub e50: f64,
    #[pyo3(get)]
    pub e90: f64,
    #[pyo3(get)]
    pub emax: f64,
}

#[pymethods]
impl CalibrationPlotData {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        predicted: Vec<f64>,
        observed: Vec<f64>,
        n_per_group: Vec<usize>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        ici: f64,
        e50: f64,
        e90: f64,
        emax: f64,
    ) -> Self {
        Self {
            predicted,
            observed,
            n_per_group,
            ci_lower,
            ci_upper,
            ici,
            e50,
            e90,
            emax,
        }
    }
}

pub fn calibration_plot_data_core(
    time: &[f64],
    status: &[i32],
    predicted_survival_at_t: &[f64],
    time_point: f64,
    n_groups: usize,
) -> CalibrationPlotData {
    let n = time.len();

    if n < n_groups * 2 {
        return CalibrationPlotData {
            predicted: vec![],
            observed: vec![],
            n_per_group: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            ici: 0.0,
            e50: 0.0,
            e90: 0.0,
            emax: 0.0,
        };
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        predicted_survival_at_t[a]
            .partial_cmp(&predicted_survival_at_t[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let group_size = n / n_groups;
    let remainder = n % n_groups;

    let mut predicted = Vec::with_capacity(n_groups);
    let mut observed = Vec::with_capacity(n_groups);
    let mut n_per_group_vec = Vec::with_capacity(n_groups);
    let mut ci_lower = Vec::with_capacity(n_groups);
    let mut ci_upper = Vec::with_capacity(n_groups);
    let mut absolute_errors = Vec::new();

    let mut start = 0;
    for g in 0..n_groups {
        let extra = if g < remainder { 1 } else { 0 };
        let end = start + group_size + extra;

        if end <= start {
            continue;
        }

        let group_indices: Vec<usize> = indices[start..end].to_vec();
        let n_in_group = group_indices.len();

        let sum_pred: f64 = group_indices
            .iter()
            .map(|&i| predicted_survival_at_t[i])
            .sum();
        let mean_pred = sum_pred / n_in_group as f64;

        let events_before_t: usize = group_indices
            .iter()
            .filter(|&&i| time[i] <= time_point && status[i] == 1)
            .count();

        let obs_surv = 1.0 - (events_before_t as f64 / n_in_group as f64);

        let se = if n_in_group > 1 && obs_surv > 0.0 && obs_surv < 1.0 {
            (obs_surv * (1.0 - obs_surv) / n_in_group as f64).sqrt()
        } else {
            0.0
        };

        let z = 1.96;
        let lower = (obs_surv - z * se).max(0.0);
        let upper = (obs_surv + z * se).min(1.0);

        predicted.push(mean_pred);
        observed.push(obs_surv);
        n_per_group_vec.push(n_in_group);
        ci_lower.push(lower);
        ci_upper.push(upper);
        absolute_errors.push((mean_pred - obs_surv).abs());

        start = end;
    }

    let ici = if !absolute_errors.is_empty() {
        absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64
    } else {
        0.0
    };

    absolute_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let e50 = if !absolute_errors.is_empty() {
        let idx = absolute_errors.len() / 2;
        absolute_errors[idx]
    } else {
        0.0
    };

    let e90 = if !absolute_errors.is_empty() {
        let idx = (absolute_errors.len() as f64 * 0.9).floor() as usize;
        absolute_errors[idx.min(absolute_errors.len() - 1)]
    } else {
        0.0
    };

    let emax = absolute_errors.last().copied().unwrap_or(0.0);

    CalibrationPlotData {
        predicted,
        observed,
        n_per_group: n_per_group_vec,
        ci_lower,
        ci_upper,
        ici,
        e50,
        e90,
        emax,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_survival_at_t, time_point, n_groups=None))]
pub fn calibration_plot(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_survival_at_t: Vec<f64>,
    time_point: f64,
    n_groups: Option<usize>,
) -> PyResult<CalibrationPlotData> {
    let n = time.len();
    if n != status.len() || n != predicted_survival_at_t.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_groups = n_groups.unwrap_or(10);
    if n_groups < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_groups must be at least 2",
        ));
    }

    Ok(calibration_plot_data_core(
        &time,
        &status,
        &predicted_survival_at_t,
        time_point,
        n_groups,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_calibration_uniform() {
        let survival_probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let status = vec![1; 100];

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert!(result.p_value > 0.05);
        assert!(result.is_calibrated);
        assert_eq!(result.n_events, 100);
        assert_eq!(result.n_bins, 10);
    }

    #[test]
    fn test_d_calibration_non_uniform() {
        let mut survival_probs = vec![0.1; 50];
        survival_probs.extend(vec![0.9; 50]);
        let status = vec![1; 100];

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert!(result.p_value < 0.05);
        assert!(!result.is_calibrated);
    }

    #[test]
    fn test_d_calibration_with_censoring() {
        let survival_probs: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let mut status = vec![1; 100];
        for i in (0..100).step_by(2) {
            status[i] = 0;
        }

        let result = d_calibration_core(&survival_probs, &status, 10);

        assert_eq!(result.n_events, 50);
    }

    #[test]
    fn test_d_calibration_empty() {
        let result = d_calibration_core(&[], &[], 10);
        assert_eq!(result.n_events, 0);
        assert!(result.is_calibrated);
    }

    #[test]
    fn test_one_calibration_basic() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = time.iter().map(|&t| (-0.01 * t).exp()).collect();

        let result = one_calibration_core(&time, &status, &predicted, 50.0, 5);

        assert_eq!(result.n_groups, 5);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_calibration_plot_basic() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = time.iter().map(|&t| (-0.01 * t).exp()).collect();

        let result = calibration_plot_data_core(&time, &status, &predicted, 50.0, 5);

        assert_eq!(result.predicted.len(), 5);
        assert_eq!(result.observed.len(), 5);
        assert!(result.ici >= 0.0);
        assert!(result.emax >= result.e90);
        assert!(result.e90 >= result.e50);
    }

    #[test]
    fn test_calibration_metrics() {
        let time: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let status = vec![1; 100];
        let predicted: Vec<f64> = (0..100).map(|i| 1.0 - i as f64 / 100.0).collect();

        let result = calibration_plot_data_core(&time, &status, &predicted, 50.0, 10);

        assert!(result.ici >= 0.0 && result.ici <= 1.0);
    }
}
