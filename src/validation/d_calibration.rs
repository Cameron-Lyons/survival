use crate::simd_ops::{dot_product_simd, mean_simd, subtract_scalar_simd, sum_of_squares_simd};
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

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct BrierCalibrationResult {
    pub time_point: f64,
    pub brier_score: f64,
    pub calibration_slope: f64,
    pub calibration_intercept: f64,
    pub ici: f64,
    pub e50: f64,
    pub e90: f64,
    pub emax: f64,
    pub predicted: Vec<f64>,
    pub observed: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub n_per_group: Vec<usize>,
}

impl std::fmt::Display for BrierCalibrationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BrierCalibrationResult(t={:.2}, brier={:.4}, slope={:.3}, ici={:.4})",
            self.time_point, self.brier_score, self.calibration_slope, self.ici
        )
    }
}

#[pymethods]
impl BrierCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        time_point: f64,
        brier_score: f64,
        calibration_slope: f64,
        calibration_intercept: f64,
        ici: f64,
        e50: f64,
        e90: f64,
        emax: f64,
        predicted: Vec<f64>,
        observed: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        n_per_group: Vec<usize>,
    ) -> Self {
        Self {
            time_point,
            brier_score,
            calibration_slope,
            calibration_intercept,
            ici,
            e50,
            e90,
            emax,
            predicted,
            observed,
            ci_lower,
            ci_upper,
            n_per_group,
        }
    }
}

fn compute_calibration_slope_intercept(predicted: &[f64], observed: &[f64]) -> (f64, f64) {
    let n = predicted.len();
    if n < 2 {
        return (1.0, 0.0);
    }

    let mean_pred = mean_simd(predicted);
    let mean_obs = mean_simd(observed);

    let centered_pred = subtract_scalar_simd(predicted, mean_pred);
    let centered_obs = subtract_scalar_simd(observed, mean_obs);

    let numerator = dot_product_simd(&centered_pred, &centered_obs);
    let denominator = sum_of_squares_simd(&centered_pred);

    let slope = if denominator > 1e-10 {
        numerator / denominator
    } else {
        1.0
    };

    let intercept = mean_obs - slope * mean_pred;

    (slope, intercept)
}

pub fn brier_calibration_core(
    time: &[f64],
    status: &[i32],
    predicted_survival_at_t: &[f64],
    time_point: f64,
    n_groups: usize,
) -> BrierCalibrationResult {
    let n = time.len();

    if n == 0 {
        return BrierCalibrationResult {
            time_point,
            brier_score: 0.0,
            calibration_slope: 1.0,
            calibration_intercept: 0.0,
            ici: 0.0,
            e50: 0.0,
            e90: 0.0,
            emax: 0.0,
            predicted: vec![],
            observed: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_per_group: vec![],
        };
    }

    let mut brier_sum = 0.0;
    let mut brier_count = 0;

    for i in 0..n {
        let outcome = if time[i] <= time_point && status[i] == 1 {
            0.0
        } else if time[i] > time_point {
            1.0
        } else {
            continue;
        };

        let pred = predicted_survival_at_t[i];
        brier_sum += (pred - outcome).powi(2);
        brier_count += 1;
    }

    let brier_score = if brier_count > 0 {
        brier_sum / brier_count as f64
    } else {
        0.0
    };

    let plot_data =
        calibration_plot_data_core(time, status, predicted_survival_at_t, time_point, n_groups);

    let (slope, intercept) =
        compute_calibration_slope_intercept(&plot_data.predicted, &plot_data.observed);

    BrierCalibrationResult {
        time_point,
        brier_score,
        calibration_slope: slope,
        calibration_intercept: intercept,
        ici: plot_data.ici,
        e50: plot_data.e50,
        e90: plot_data.e90,
        emax: plot_data.emax,
        predicted: plot_data.predicted,
        observed: plot_data.observed,
        ci_lower: plot_data.ci_lower,
        ci_upper: plot_data.ci_upper,
        n_per_group: plot_data.n_per_group,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_survival_at_t, time_point, n_groups=None))]
pub fn brier_calibration(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_survival_at_t: Vec<f64>,
    time_point: f64,
    n_groups: Option<usize>,
) -> PyResult<BrierCalibrationResult> {
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

    Ok(brier_calibration_core(
        &time,
        &status,
        &predicted_survival_at_t,
        time_point,
        n_groups,
    ))
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct MultiTimeCalibrationResult {
    pub time_points: Vec<f64>,
    pub brier_scores: Vec<f64>,
    pub integrated_brier: f64,
    pub calibration_slopes: Vec<f64>,
    pub calibration_intercepts: Vec<f64>,
    pub ici_values: Vec<f64>,
    pub mean_ici: f64,
    pub mean_slope: f64,
}

impl std::fmt::Display for MultiTimeCalibrationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MultiTimeCalibrationResult(n_times={}, ibs={:.4}, mean_slope={:.3}, mean_ici={:.4})",
            self.time_points.len(),
            self.integrated_brier,
            self.mean_slope,
            self.mean_ici
        )
    }
}

#[pymethods]
impl MultiTimeCalibrationResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        time_points: Vec<f64>,
        brier_scores: Vec<f64>,
        integrated_brier: f64,
        calibration_slopes: Vec<f64>,
        calibration_intercepts: Vec<f64>,
        ici_values: Vec<f64>,
        mean_ici: f64,
        mean_slope: f64,
    ) -> Self {
        Self {
            time_points,
            brier_scores,
            integrated_brier,
            calibration_slopes,
            calibration_intercepts,
            ici_values,
            mean_ici,
            mean_slope,
        }
    }
}

pub fn multi_time_calibration_core(
    time: &[f64],
    status: &[i32],
    survival_predictions: &[Vec<f64>],
    prediction_times: &[f64],
    n_groups: usize,
) -> MultiTimeCalibrationResult {
    let n_times = prediction_times.len();

    if n_times == 0 || survival_predictions.is_empty() {
        return MultiTimeCalibrationResult {
            time_points: vec![],
            brier_scores: vec![],
            integrated_brier: 0.0,
            calibration_slopes: vec![],
            calibration_intercepts: vec![],
            ici_values: vec![],
            mean_ici: 0.0,
            mean_slope: 1.0,
        };
    }

    let mut brier_scores = Vec::with_capacity(n_times);
    let mut calibration_slopes = Vec::with_capacity(n_times);
    let mut calibration_intercepts = Vec::with_capacity(n_times);
    let mut ici_values = Vec::with_capacity(n_times);

    for (t_idx, &t) in prediction_times.iter().enumerate() {
        let preds_at_t: Vec<f64> = survival_predictions.iter().map(|row| row[t_idx]).collect();

        let result = brier_calibration_core(time, status, &preds_at_t, t, n_groups);

        brier_scores.push(result.brier_score);
        calibration_slopes.push(result.calibration_slope);
        calibration_intercepts.push(result.calibration_intercept);
        ici_values.push(result.ici);
    }

    let integrated_brier = if n_times >= 2 {
        let mut integrated = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n_times - 1 {
            let dt = prediction_times[i + 1] - prediction_times[i];
            let avg_brier = (brier_scores[i] + brier_scores[i + 1]) / 2.0;
            integrated += avg_brier * dt;
            total_weight += dt;
        }

        if total_weight > 0.0 {
            integrated / total_weight
        } else {
            brier_scores.iter().sum::<f64>() / n_times as f64
        }
    } else {
        brier_scores.first().copied().unwrap_or(0.0)
    };

    let mean_ici = if !ici_values.is_empty() {
        ici_values.iter().sum::<f64>() / ici_values.len() as f64
    } else {
        0.0
    };

    let mean_slope = if !calibration_slopes.is_empty() {
        calibration_slopes.iter().sum::<f64>() / calibration_slopes.len() as f64
    } else {
        1.0
    };

    MultiTimeCalibrationResult {
        time_points: prediction_times.to_vec(),
        brier_scores,
        integrated_brier,
        calibration_slopes,
        calibration_intercepts,
        ici_values,
        mean_ici,
        mean_slope,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, survival_predictions, prediction_times, n_groups=None))]
pub fn multi_time_calibration(
    time: Vec<f64>,
    status: Vec<i32>,
    survival_predictions: Vec<Vec<f64>>,
    prediction_times: Vec<f64>,
    n_groups: Option<usize>,
) -> PyResult<MultiTimeCalibrationResult> {
    let n = time.len();
    if n != status.len() || n != survival_predictions.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and survival_predictions must have the same length",
        ));
    }

    for (i, row) in survival_predictions.iter().enumerate() {
        if row.len() != prediction_times.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "survival_predictions row {} has {} elements, expected {}",
                i,
                row.len(),
                prediction_times.len()
            )));
        }
    }

    let n_groups = n_groups.unwrap_or(10);
    if n_groups < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_groups must be at least 2",
        ));
    }

    Ok(multi_time_calibration_core(
        &time,
        &status,
        &survival_predictions,
        &prediction_times,
        n_groups,
    ))
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all)]
pub struct SmoothedCalibrationCurve {
    pub predicted_grid: Vec<f64>,
    pub smoothed_observed: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub bandwidth: f64,
}

impl std::fmt::Display for SmoothedCalibrationCurve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SmoothedCalibrationCurve(n_points={}, bandwidth={:.3})",
            self.predicted_grid.len(),
            self.bandwidth
        )
    }
}

#[pymethods]
impl SmoothedCalibrationCurve {
    #[new]
    fn new(
        predicted_grid: Vec<f64>,
        smoothed_observed: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        bandwidth: f64,
    ) -> Self {
        Self {
            predicted_grid,
            smoothed_observed,
            ci_lower,
            ci_upper,
            bandwidth,
        }
    }
}

fn gaussian_kernel(x: f64, bandwidth: f64) -> f64 {
    let z = x / bandwidth;
    (-0.5 * z * z).exp() / (bandwidth * (2.0 * std::f64::consts::PI).sqrt())
}

pub fn smoothed_calibration_core(
    time: &[f64],
    status: &[i32],
    predicted_survival_at_t: &[f64],
    time_point: f64,
    n_grid_points: usize,
    bandwidth: Option<f64>,
) -> SmoothedCalibrationCurve {
    let n = time.len();

    if n == 0 {
        return SmoothedCalibrationCurve {
            predicted_grid: vec![],
            smoothed_observed: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            bandwidth: 0.0,
        };
    }

    let outcomes: Vec<f64> = (0..n)
        .filter_map(|i| {
            if time[i] <= time_point && status[i] == 1 {
                Some(0.0)
            } else if time[i] > time_point {
                Some(1.0)
            } else {
                None
            }
        })
        .collect();

    let preds: Vec<f64> = (0..n)
        .filter_map(|i| {
            if time[i] > time_point || (time[i] <= time_point && status[i] == 1) {
                Some(predicted_survival_at_t[i])
            } else {
                None
            }
        })
        .collect();

    if preds.is_empty() {
        return SmoothedCalibrationCurve {
            predicted_grid: vec![],
            smoothed_observed: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            bandwidth: 0.0,
        };
    }

    let h = bandwidth
        .unwrap_or_else(|| {
            let mut sorted_preds = preds.clone();
            sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1_idx = sorted_preds.len() / 4;
            let q3_idx = 3 * sorted_preds.len() / 4;
            let iqr = sorted_preds[q3_idx] - sorted_preds[q1_idx];
            0.9 * iqr.min(sorted_preds.iter().copied().fold(0.0_f64, f64::max) / 4.0)
                * (preds.len() as f64).powf(-0.2)
        })
        .max(0.05);

    let min_pred = preds.iter().copied().fold(f64::INFINITY, f64::min);
    let max_pred = preds.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let grid_step = (max_pred - min_pred) / (n_grid_points - 1) as f64;
    let predicted_grid: Vec<f64> = (0..n_grid_points)
        .map(|i| min_pred + i as f64 * grid_step)
        .collect();

    let mut smoothed_observed = Vec::with_capacity(n_grid_points);
    let mut ci_lower = Vec::with_capacity(n_grid_points);
    let mut ci_upper = Vec::with_capacity(n_grid_points);

    for &x in &predicted_grid {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weighted_sq_sum = 0.0;

        for (i, &pred) in preds.iter().enumerate() {
            let w = gaussian_kernel(x - pred, h);
            weighted_sum += w * outcomes[i];
            weight_sum += w;
            weighted_sq_sum += w * outcomes[i] * outcomes[i];
        }

        let smoothed = if weight_sum > 1e-10 {
            weighted_sum / weight_sum
        } else {
            0.5
        };

        let variance = if weight_sum > 1e-10 {
            let mean_sq = weighted_sq_sum / weight_sum;
            (mean_sq - smoothed * smoothed).max(0.0)
        } else {
            0.0
        };

        let se = (variance / weight_sum.max(1.0)).sqrt();
        let z = 1.96;

        smoothed_observed.push(smoothed);
        ci_lower.push((smoothed - z * se).clamp(0.0, 1.0));
        ci_upper.push((smoothed + z * se).clamp(0.0, 1.0));
    }

    SmoothedCalibrationCurve {
        predicted_grid,
        smoothed_observed,
        ci_lower,
        ci_upper,
        bandwidth: h,
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted_survival_at_t, time_point, n_grid_points=None, bandwidth=None))]
pub fn smoothed_calibration(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted_survival_at_t: Vec<f64>,
    time_point: f64,
    n_grid_points: Option<usize>,
    bandwidth: Option<f64>,
) -> PyResult<SmoothedCalibrationCurve> {
    let n = time.len();
    if n != status.len() || n != predicted_survival_at_t.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All input vectors must have the same length",
        ));
    }

    let n_grid_points = n_grid_points.unwrap_or(100);
    if n_grid_points < 10 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_grid_points must be at least 10",
        ));
    }

    Ok(smoothed_calibration_core(
        &time,
        &status,
        &predicted_survival_at_t,
        time_point,
        n_grid_points,
        bandwidth,
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
