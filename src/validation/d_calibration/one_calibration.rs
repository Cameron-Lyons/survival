
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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

pub(crate) fn one_calibration_core(
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
