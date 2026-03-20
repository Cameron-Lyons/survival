
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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

pub(crate) fn calibration_plot_data_core(
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
