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
pub enum ChangepointMethod {
    PELT,
    BinarySegment,
    BottomUp,
}

#[pymethods]
impl ChangepointMethod {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "pelt" => Ok(ChangepointMethod::PELT),
            "binary" | "binarysegment" | "binary_segment" => Ok(ChangepointMethod::BinarySegment),
            "bottomup" | "bottom_up" => Ok(ChangepointMethod::BottomUp),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown method. Use 'pelt', 'binary_segment', or 'bottom_up'",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass]
pub enum CostFunction {
    L2,
    L1,
    Normal,
    Poisson,
}

#[pymethods]
impl CostFunction {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "l2" | "quadratic" | "normal_mean" => Ok(CostFunction::L2),
            "l1" | "absolute" => Ok(CostFunction::L1),
            "normal" | "normal_meanvar" => Ok(CostFunction::Normal),
            "poisson" => Ok(CostFunction::Poisson),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown cost function",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ChangepointConfig {
    #[pyo3(get, set)]
    pub method: ChangepointMethod,
    #[pyo3(get, set)]
    pub cost: CostFunction,
    #[pyo3(get, set)]
    pub penalty: f64,
    #[pyo3(get, set)]
    pub min_size: usize,
    #[pyo3(get, set)]
    pub max_changepoints: Option<usize>,
}

#[pymethods]
impl ChangepointConfig {
    #[new]
    #[pyo3(signature = (
        method=ChangepointMethod::PELT,
        cost=CostFunction::L2,
        penalty=1.0,
        min_size=2,
        max_changepoints=None
    ))]
    pub fn new(
        method: ChangepointMethod,
        cost: CostFunction,
        penalty: f64,
        min_size: usize,
        max_changepoints: Option<usize>,
    ) -> PyResult<Self> {
        if penalty < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "penalty must be non-negative",
            ));
        }
        if min_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "min_size must be positive",
            ));
        }

        Ok(ChangepointConfig {
            method,
            cost,
            penalty,
            min_size,
            max_changepoints,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct Changepoint {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub time: f64,
    #[pyo3(get)]
    pub cost_improvement: f64,
    #[pyo3(get)]
    pub mean_before: f64,
    #[pyo3(get)]
    pub mean_after: f64,
}

#[pymethods]
impl Changepoint {
    fn __repr__(&self) -> String {
        format!(
            "Changepoint(idx={}, time={:.2}, delta={:.4})",
            self.index,
            self.time,
            self.mean_after - self.mean_before
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ChangepointResult {
    #[pyo3(get)]
    pub feature_idx: usize,
    #[pyo3(get)]
    pub changepoints: Vec<Changepoint>,
    #[pyo3(get)]
    pub segments: Vec<(usize, usize)>,
    #[pyo3(get)]
    pub segment_means: Vec<f64>,
    #[pyo3(get)]
    pub total_cost: f64,
    #[pyo3(get)]
    pub n_changepoints: usize,
}

#[pymethods]
impl ChangepointResult {
    fn __repr__(&self) -> String {
        format!(
            "ChangepointResult(feature={}, n_changepoints={})",
            self.feature_idx, self.n_changepoints
        )
    }

    fn get_segment_at(&self, time_idx: usize) -> usize {
        for (seg_idx, &(start, end)) in self.segments.iter().enumerate() {
            if time_idx >= start && time_idx < end {
                return seg_idx;
            }
        }
        self.segments.len().saturating_sub(1)
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AllChangepointsResult {
    #[pyo3(get)]
    pub results: Vec<ChangepointResult>,
    #[pyo3(get)]
    pub features_with_changes: Vec<usize>,
    #[pyo3(get)]
    pub most_unstable_features: Vec<(usize, usize)>,
}

#[pymethods]
impl AllChangepointsResult {
    fn __repr__(&self) -> String {
        format!(
            "AllChangepointsResult(n_features={}, with_changes={})",
            self.results.len(),
            self.features_with_changes.len()
        )
    }
}

fn compute_segment_cost(data: &[f64], start: usize, end: usize, cost: CostFunction) -> f64 {
    if end <= start {
        return 0.0;
    }

    let segment = &data[start..end];
    let n = segment.len() as f64;

    match cost {
        CostFunction::L2 => {
            let mean = segment.iter().sum::<f64>() / n;
            segment.iter().map(|&x| (x - mean).powi(2)).sum()
        }
        CostFunction::L1 => {
            let mut sorted: Vec<f64> = segment.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if sorted.len().is_multiple_of(2) {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            };
            segment.iter().map(|&x| (x - median).abs()).sum()
        }
        CostFunction::Normal => {
            let mean = segment.iter().sum::<f64>() / n;
            let var = segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            if var > 1e-12 { n * (1.0 + var.ln()) } else { n }
        }
        CostFunction::Poisson => {
            let mean = segment.iter().sum::<f64>() / n;
            if mean > 1e-12 {
                2.0 * segment
                    .iter()
                    .map(|&x| {
                        let x = x.max(1e-12);
                        x * (x / mean).ln() - x + mean
                    })
                    .sum::<f64>()
            } else {
                0.0
            }
        }
    }
}

fn pelt(data: &[f64], penalty: f64, min_size: usize, cost: CostFunction) -> Vec<usize> {
    let n = data.len();
    if n < 2 * min_size {
        return vec![];
    }

    let mut f = vec![f64::INFINITY; n + 1];
    let mut cp = vec![0usize; n + 1];
    let mut r: Vec<usize> = vec![0];

    f[0] = -penalty;

    for t in min_size..=n {
        let mut new_r = Vec::new();

        for &s in &r {
            if t - s >= min_size {
                let cost_val = compute_segment_cost(data, s, t, cost);
                let candidate = f[s] + cost_val + penalty;

                if candidate < f[t] {
                    f[t] = candidate;
                    cp[t] = s;
                }

                if f[s] + cost_val + penalty <= f[t] + penalty {
                    new_r.push(s);
                }
            }
        }

        new_r.push(t);
        r = new_r;
    }

    let mut changepoints = Vec::new();
    let mut idx = n;
    while cp[idx] > 0 {
        changepoints.push(cp[idx]);
        idx = cp[idx];
    }

    changepoints.reverse();
    changepoints
}

fn binary_segmentation(
    data: &[f64],
    penalty: f64,
    min_size: usize,
    cost: CostFunction,
    max_cp: Option<usize>,
) -> Vec<usize> {
    let n = data.len();
    let max_changepoints = max_cp.unwrap_or(n / (2 * min_size));

    let mut changepoints = Vec::new();
    let mut segments: Vec<(usize, usize)> = vec![(0, n)];

    while changepoints.len() < max_changepoints && !segments.is_empty() {
        let mut best_gain = 0.0;
        let mut best_cp = None;
        let mut best_seg_idx = 0;

        for (seg_idx, &(start, end)) in segments.iter().enumerate() {
            if end - start < 2 * min_size {
                continue;
            }

            let full_cost = compute_segment_cost(data, start, end, cost);

            for cp in (start + min_size)..(end - min_size + 1) {
                let left_cost = compute_segment_cost(data, start, cp, cost);
                let right_cost = compute_segment_cost(data, cp, end, cost);
                let gain = full_cost - left_cost - right_cost - penalty;

                if gain > best_gain {
                    best_gain = gain;
                    best_cp = Some(cp);
                    best_seg_idx = seg_idx;
                }
            }
        }

        if let Some(cp) = best_cp {
            let (start, end) = segments.remove(best_seg_idx);
            segments.push((start, cp));
            segments.push((cp, end));
            changepoints.push(cp);
        } else {
            break;
        }
    }

    changepoints.sort();
    changepoints
}

fn bottom_up(
    data: &[f64],
    penalty: f64,
    min_size: usize,
    cost: CostFunction,
    max_cp: Option<usize>,
) -> Vec<usize> {
    let n = data.len();
    let max_changepoints = max_cp.unwrap_or(n / min_size);

    let mut changepoints: Vec<usize> = (min_size..n).step_by(min_size).collect();

    if changepoints.is_empty() {
        return vec![];
    }

    while changepoints.len() > max_changepoints {
        let mut min_cost_increase = f64::INFINITY;
        let mut merge_idx = 0;

        for i in 0..changepoints.len() {
            let start = if i == 0 { 0 } else { changepoints[i - 1] };
            let mid = changepoints[i];
            let end = if i + 1 < changepoints.len() {
                changepoints[i + 1]
            } else {
                n
            };

            let left_cost = compute_segment_cost(data, start, mid, cost);
            let right_cost = compute_segment_cost(data, mid, end, cost);
            let merged_cost = compute_segment_cost(data, start, end, cost);

            let cost_increase = merged_cost - left_cost - right_cost + penalty;

            if cost_increase < min_cost_increase {
                min_cost_increase = cost_increase;
                merge_idx = i;
            }
        }

        if min_cost_increase > penalty {
            break;
        }

        changepoints.remove(merge_idx);
    }

    changepoints
}

fn detect_changepoints_single(
    shap_values: &[f64],
    time_points: &[f64],
    feature_idx: usize,
    config: &ChangepointConfig,
) -> ChangepointResult {
    let n = shap_values.len();

    let cp_indices = match config.method {
        ChangepointMethod::PELT => pelt(shap_values, config.penalty, config.min_size, config.cost),
        ChangepointMethod::BinarySegment => binary_segmentation(
            shap_values,
            config.penalty,
            config.min_size,
            config.cost,
            config.max_changepoints,
        ),
        ChangepointMethod::BottomUp => bottom_up(
            shap_values,
            config.penalty,
            config.min_size,
            config.cost,
            config.max_changepoints,
        ),
    };

    let mut segments: Vec<(usize, usize)> = Vec::new();
    let mut prev = 0;
    for &cp in &cp_indices {
        segments.push((prev, cp));
        prev = cp;
    }
    segments.push((prev, n));

    let segment_means: Vec<f64> = segments
        .iter()
        .map(|&(start, end)| {
            if end > start {
                shap_values[start..end].iter().sum::<f64>() / (end - start) as f64
            } else {
                0.0
            }
        })
        .collect();

    let total_cost: f64 = segments
        .iter()
        .map(|&(start, end)| compute_segment_cost(shap_values, start, end, config.cost))
        .sum();

    let changepoints: Vec<Changepoint> = cp_indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let mean_before = segment_means[i];
            let mean_after = segment_means[i + 1];

            let start = if i == 0 { 0 } else { cp_indices[i - 1] };
            let end = if i + 1 < cp_indices.len() {
                cp_indices[i + 1]
            } else {
                n
            };

            let cost_without =
                compute_segment_cost(shap_values, start, end, config.cost) + config.penalty;
            let cost_with = compute_segment_cost(shap_values, start, idx, config.cost)
                + compute_segment_cost(shap_values, idx, end, config.cost);

            Changepoint {
                index: idx,
                time: time_points.get(idx).copied().unwrap_or(idx as f64),
                cost_improvement: cost_without - cost_with,
                mean_before,
                mean_after,
            }
        })
        .collect();

    ChangepointResult {
        feature_idx,
        changepoints,
        segments,
        segment_means,
        total_cost,
        n_changepoints: cp_indices.len(),
    }
}

#[pyfunction]
#[pyo3(signature = (shap_values, time_points, n_samples, n_features, config))]
pub fn detect_changepoints(
    shap_values: Vec<Vec<Vec<f64>>>,
    time_points: Vec<f64>,
    n_samples: usize,
    n_features: usize,
    config: &ChangepointConfig,
) -> PyResult<AllChangepointsResult> {
    let n_times = time_points.len();

    if shap_values.len() != n_samples {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "shap_values first dimension must match n_samples",
        ));
    }

    let results: Vec<ChangepointResult> = (0..n_features)
        .into_par_iter()
        .map(|f| {
            let aggregated: Vec<f64> = (0..n_times)
                .map(|t| {
                    shap_values
                        .iter()
                        .map(|sample| sample[f][t].abs())
                        .sum::<f64>()
                        / n_samples as f64
                })
                .collect();

            detect_changepoints_single(&aggregated, &time_points, f, config)
        })
        .collect();

    let features_with_changes: Vec<usize> = results
        .iter()
        .filter(|r| r.n_changepoints > 0)
        .map(|r| r.feature_idx)
        .collect();

    let mut most_unstable_features: Vec<(usize, usize)> = results
        .iter()
        .map(|r| (r.feature_idx, r.n_changepoints))
        .collect();
    most_unstable_features.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(AllChangepointsResult {
        results,
        features_with_changes,
        most_unstable_features,
    })
}

#[pyfunction]
#[pyo3(signature = (data, time_points, config))]
pub fn detect_changepoints_single_series(
    data: Vec<f64>,
    time_points: Vec<f64>,
    config: &ChangepointConfig,
) -> PyResult<ChangepointResult> {
    if data.len() != time_points.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "data and time_points must have equal length",
        ));
    }

    Ok(detect_changepoints_single(&data, &time_points, 0, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config() {
        let config =
            ChangepointConfig::new(ChangepointMethod::PELT, CostFunction::L2, 1.0, 2, None)
                .unwrap();
        assert_eq!(config.min_size, 2);
    }

    #[test]
    fn test_segment_cost_l2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cost = compute_segment_cost(&data, 0, 5, CostFunction::L2);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_pelt_clear_changepoint() {
        let mut data: Vec<f64> = vec![1.0; 20];
        data.extend(vec![5.0; 20]);

        let cp = binary_segmentation(&data, 5.0, 5, CostFunction::L2, Some(3));
        assert!(!cp.is_empty());
        assert!((cp[0] as i32 - 20).abs() <= 3);
    }

    #[test]
    fn test_binary_segmentation() {
        let mut data: Vec<f64> = vec![1.0; 15];
        data.extend(vec![5.0; 15]);

        let cp = binary_segmentation(&data, 5.0, 5, CostFunction::L2, Some(3));
        assert!(!cp.is_empty());
    }

    #[test]
    fn test_bottom_up() {
        let mut data: Vec<f64> = vec![1.0; 20];
        data.extend(vec![5.0; 20]);

        let cp = bottom_up(&data, 10.0, 5, CostFunction::L2, Some(5));
        assert!(!cp.is_empty());
    }

    #[test]
    fn test_detect_single_series() {
        let data: Vec<f64> = (0..30).map(|i| if i < 15 { 1.0 } else { 5.0 }).collect();
        let time: Vec<f64> = (0..30).map(|i| i as f64).collect();

        let config = ChangepointConfig::new(
            ChangepointMethod::BinarySegment,
            CostFunction::L2,
            5.0,
            5,
            None,
        )
        .unwrap();

        let result = detect_changepoints_single_series(data, time, &config).unwrap();
        assert!(result.n_changepoints >= 1);
    }
}
