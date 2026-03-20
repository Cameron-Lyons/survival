use super::super::*;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ConformalCalibrationPlot {
    #[pyo3(get)]
    pub coverage_levels: Vec<f64>,
    #[pyo3(get)]
    pub empirical_coverages: Vec<f64>,
    #[pyo3(get)]
    pub ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_test: usize,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ConformalWidthAnalysis {
    #[pyo3(get)]
    pub mean_width: f64,
    #[pyo3(get)]
    pub median_width: f64,
    #[pyo3(get)]
    pub std_width: f64,
    #[pyo3(get)]
    pub min_width: f64,
    #[pyo3(get)]
    pub max_width: f64,
    #[pyo3(get)]
    pub quantile_25: f64,
    #[pyo3(get)]
    pub quantile_75: f64,
    #[pyo3(get)]
    pub width_by_predicted: Vec<(f64, f64)>,
}

#[pyfunction]
#[pyo3(signature = (time_test, status_test, lower_bounds, upper_bounds=None, n_levels=None))]
pub fn conformal_calibration_plot(
    time_test: Vec<f64>,
    status_test: Vec<i32>,
    lower_bounds: Vec<Vec<f64>>,
    upper_bounds: Option<Vec<Vec<f64>>>,
    n_levels: Option<usize>,
) -> PyResult<ConformalCalibrationPlot> {
    let n_test = time_test.len();
    if n_test == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Test data cannot be empty",
        ));
    }

    let n_levels_val = n_levels.unwrap_or(10);
    let coverage_levels: Vec<f64> = (1..=n_levels_val)
        .map(|i| i as f64 / n_levels_val as f64)
        .collect();

    let n_bounds = lower_bounds.len();
    if n_bounds != n_levels_val {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "lower_bounds length must match n_levels",
        ));
    }

    let has_upper = upper_bounds.is_some();
    let upper = upper_bounds.unwrap_or_else(|| vec![vec![f64::INFINITY; n_test]; n_levels_val]);

    let mut empirical_coverages = Vec::with_capacity(n_levels_val);
    let mut ci_lower = Vec::with_capacity(n_levels_val);
    let mut ci_upper = Vec::with_capacity(n_levels_val);

    for level_idx in 0..n_levels_val {
        let lb = &lower_bounds[level_idx];
        let ub = &upper[level_idx];

        let mut covered = 0usize;
        let mut total = 0usize;

        for i in 0..n_test {
            if status_test[i] == 1 {
                total += 1;
                let above_lower = time_test[i] >= lb[i];
                let below_upper = !has_upper || time_test[i] <= ub[i];
                if above_lower && below_upper {
                    covered += 1;
                }
            }
        }

        let emp_cov = if total > 0 {
            covered as f64 / total as f64
        } else {
            0.0
        };
        let se = (emp_cov * (1.0 - emp_cov) / total.max(1) as f64).sqrt();
        let z = 1.96;

        empirical_coverages.push(emp_cov);
        ci_lower.push((emp_cov - z * se).max(0.0));
        ci_upper.push((emp_cov + z * se).min(1.0));
    }

    Ok(ConformalCalibrationPlot {
        coverage_levels,
        empirical_coverages,
        ci_lower,
        ci_upper,
        n_test,
    })
}

#[pyfunction]
#[pyo3(signature = (lower_bounds, upper_bounds, predicted))]
pub fn conformal_width_analysis(
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    predicted: Vec<f64>,
) -> PyResult<ConformalWidthAnalysis> {
    let n = lower_bounds.len();
    if n == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }
    if upper_bounds.len() != n || predicted.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All arrays must have the same length",
        ));
    }

    let widths: Vec<f64> = lower_bounds
        .iter()
        .zip(upper_bounds.iter())
        .map(|(&l, &u)| if u.is_finite() { u - l } else { f64::INFINITY })
        .filter(|&w| w.is_finite())
        .collect();

    if widths.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No finite interval widths",
        ));
    }

    let n_finite = widths.len();
    let mean_width = widths.iter().sum::<f64>() / n_finite as f64;

    let mut sorted_widths = widths.clone();
    sorted_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_width = if n_finite.is_multiple_of(2) {
        (sorted_widths[n_finite / 2 - 1] + sorted_widths[n_finite / 2]) / 2.0
    } else {
        sorted_widths[n_finite / 2]
    };

    let variance = widths
        .iter()
        .map(|&w| (w - mean_width).powi(2))
        .sum::<f64>()
        / n_finite as f64;
    let std_width = variance.sqrt();

    let min_width = sorted_widths[0];
    let max_width = sorted_widths[n_finite - 1];
    let quantile_25 = sorted_widths[(0.25 * n_finite as f64) as usize];
    let quantile_75 = sorted_widths[((0.75 * n_finite as f64) as usize).min(n_finite - 1)];

    let mut width_by_predicted: Vec<(f64, f64)> = predicted
        .iter()
        .zip(lower_bounds.iter().zip(upper_bounds.iter()))
        .filter(|(_, (_, u))| u.is_finite())
        .map(|(p, (l, u))| (*p, *u - *l))
        .collect();
    width_by_predicted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(ConformalWidthAnalysis {
        mean_width,
        median_width,
        std_width,
        min_width,
        max_width,
        quantile_25,
        quantile_75,
        width_by_predicted,
    })
}
