
#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
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

pub(crate) fn brier_calibration_core(
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
