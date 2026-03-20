use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MondrianDiagnostics {
    #[pyo3(get)]
    pub group_labels: Vec<i32>,
    #[pyo3(get)]
    pub group_sizes: Vec<usize>,
    #[pyo3(get)]
    pub group_thresholds: Vec<f64>,
    #[pyo3(get)]
    pub n_small_groups: usize,
    #[pyo3(get)]
    pub global_threshold: f64,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MondrianCalibrationResult {
    #[pyo3(get)]
    pub group_thresholds: std::collections::HashMap<i32, f64>,
    #[pyo3(get)]
    pub global_threshold: f64,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub group_sizes: std::collections::HashMap<i32, usize>,
    #[pyo3(get)]
    pub min_group_size: usize,
    #[pyo3(get)]
    pub diagnostics: MondrianDiagnostics,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct MondrianConformalResult {
    #[pyo3(get)]
    pub lower_predictive_bound: Vec<f64>,
    #[pyo3(get)]
    pub predicted_time: Vec<f64>,
    #[pyo3(get)]
    pub coverage_level: f64,
    #[pyo3(get)]
    pub applied_thresholds: Vec<f64>,
    #[pyo3(get)]
    pub group_labels_used: Vec<i32>,
    #[pyo3(get)]
    pub used_global_fallback: Vec<bool>,
    #[pyo3(get)]
    pub diagnostics: MondrianDiagnostics,
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, group_labels, coverage_level=None, min_group_size=None))]
#[allow(clippy::too_many_arguments)]
pub fn mondrian_conformal_calibrate(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    group_labels: Vec<i32>,
    coverage_level: Option<f64>,
    min_group_size: Option<usize>,
) -> PyResult<MondrianCalibrationResult> {
    let n = time.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if status.len() != n || predicted.len() != n || group_labels.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, predicted, and group_labels must have the same length",
        ));
    }

    let coverage = coverage_level.unwrap_or(DEFAULT_CONFORMAL_COVERAGE);
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coverage_level must be between 0 and 1",
        ));
    }

    let min_size = min_group_size.unwrap_or(DEFAULT_MIN_GROUP_SIZE);

    let mut group_scores: std::collections::HashMap<i32, Vec<f64>> =
        std::collections::HashMap::new();
    let mut all_scores = Vec::new();

    for i in 0..n {
        if status[i] == 1 {
            let score = time[i] - predicted[i];
            all_scores.push(score);
            group_scores.entry(group_labels[i]).or_default().push(score);
        }
    }

    if all_scores.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No uncensored observations in calibration set",
        ));
    }

    let n_all = all_scores.len();
    let global_q_level = (1.0 - coverage) * (n_all as f64 + 1.0) / n_all as f64;
    let global_q_level = global_q_level.min(1.0);
    let global_weights: Vec<f64> = vec![1.0; n_all];
    let global_threshold = weighted_quantile(&all_scores, &global_weights, global_q_level);

    let mut group_thresholds: std::collections::HashMap<i32, f64> =
        std::collections::HashMap::new();
    let mut group_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut n_small_groups = 0usize;

    let mut diag_group_labels = Vec::new();
    let mut diag_group_sizes = Vec::new();
    let mut diag_group_thresholds = Vec::new();

    for (&group, scores) in &group_scores {
        let size = scores.len();
        group_sizes.insert(group, size);
        diag_group_labels.push(group);
        diag_group_sizes.push(size);

        if size >= min_size {
            let q_level = (1.0 - coverage) * (size as f64 + 1.0) / size as f64;
            let q_level = q_level.min(1.0);
            let weights: Vec<f64> = vec![1.0; size];
            let threshold = weighted_quantile(scores, &weights, q_level);
            group_thresholds.insert(group, threshold);
            diag_group_thresholds.push(threshold);
        } else {
            group_thresholds.insert(group, global_threshold);
            diag_group_thresholds.push(global_threshold);
            n_small_groups += 1;
        }
    }

    let diagnostics = MondrianDiagnostics {
        group_labels: diag_group_labels,
        group_sizes: diag_group_sizes,
        group_thresholds: diag_group_thresholds,
        n_small_groups,
        global_threshold,
    };

    Ok(MondrianCalibrationResult {
        group_thresholds,
        global_threshold,
        coverage_level: coverage,
        group_sizes,
        min_group_size: min_size,
        diagnostics,
    })
}

#[pyfunction]
#[pyo3(signature = (calibration, predicted_new, group_labels_new))]
pub fn mondrian_conformal_predict(
    calibration: &MondrianCalibrationResult,
    predicted_new: Vec<f64>,
    group_labels_new: Vec<i32>,
) -> PyResult<MondrianConformalResult> {
    let n_new = predicted_new.len();
    if n_new == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new cannot be empty",
        ));
    }
    if group_labels_new.len() != n_new {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "predicted_new and group_labels_new must have the same length",
        ));
    }

    let mut lower_predictive_bound = Vec::with_capacity(n_new);
    let mut applied_thresholds = Vec::with_capacity(n_new);
    let mut used_global_fallback = Vec::with_capacity(n_new);

    for i in 0..n_new {
        let group = group_labels_new[i];
        let (threshold, used_fallback) = if let Some(&t) = calibration.group_thresholds.get(&group)
        {
            let size = calibration.group_sizes.get(&group).copied().unwrap_or(0);
            if size >= calibration.min_group_size {
                (t, false)
            } else {
                (calibration.global_threshold, true)
            }
        } else {
            (calibration.global_threshold, true)
        };

        lower_predictive_bound.push((predicted_new[i] - threshold).max(0.0));
        applied_thresholds.push(threshold);
        used_global_fallback.push(used_fallback);
    }

    Ok(MondrianConformalResult {
        lower_predictive_bound,
        predicted_time: predicted_new,
        coverage_level: calibration.coverage_level,
        applied_thresholds,
        group_labels_used: group_labels_new,
        used_global_fallback,
        diagnostics: calibration.diagnostics.clone(),
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, predicted, group_labels, predicted_new, group_labels_new, coverage_level=None, min_group_size=None))]
#[allow(clippy::too_many_arguments)]
pub fn mondrian_conformal_survival(
    time: Vec<f64>,
    status: Vec<i32>,
    predicted: Vec<f64>,
    group_labels: Vec<i32>,
    predicted_new: Vec<f64>,
    group_labels_new: Vec<i32>,
    coverage_level: Option<f64>,
    min_group_size: Option<usize>,
) -> PyResult<MondrianConformalResult> {
    let calibration = mondrian_conformal_calibrate(
        time,
        status,
        predicted,
        group_labels,
        coverage_level,
        min_group_size,
    )?;

    mondrian_conformal_predict(&calibration, predicted_new, group_labels_new)
}
