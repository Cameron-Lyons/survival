
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
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

pub(crate) fn smoothed_calibration_core(
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

