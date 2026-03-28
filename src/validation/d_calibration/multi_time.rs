
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
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

pub(crate) fn multi_time_calibration_core(
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
