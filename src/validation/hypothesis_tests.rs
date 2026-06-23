use crate::internal::matrix::matrix_inverse;
use crate::internal::statistical::chi2_sf;
use ndarray::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn invalid_test_result(test_name: &'static str, df: usize) -> TestResult {
    TestResult {
        statistic: f64::NAN,
        df,
        p_value: f64::NAN,
        test_name: test_name.to_string(),
    }
}

fn validate_finite_slice(values: &[f64], field: &'static str) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{field} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_positive_finite_slice(values: &[f64], field: &'static str) -> PyResult<()> {
    validate_finite_slice(values, field)?;
    for (idx, &value) in values.iter().enumerate() {
        if value <= 0.0 {
            return Err(value_error(format!(
                "{field} must contain positive values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_square_finite_matrix(
    matrix: &[Vec<f64>],
    expected_width: usize,
    field: &'static str,
) -> PyResult<()> {
    if matrix.len() != expected_width {
        return Err(value_error(format!(
            "{field} must have {expected_width} rows"
        )));
    }
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != expected_width {
            return Err(value_error(format!(
                "{field} must be a square matrix; row {row_idx} has length {}, expected {expected_width}",
                row.len()
            )));
        }
        validate_finite_slice(row, field)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TestResult {
    #[pyo3(get)]
    pub statistic: f64,
    #[pyo3(get)]
    pub df: usize,
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub test_name: String,
}
#[pymethods]
impl TestResult {
    #[new]
    fn new(statistic: f64, df: usize, p_value: f64, test_name: String) -> Self {
        Self {
            statistic,
            df,
            p_value,
            test_name,
        }
    }
    fn __repr__(&self) -> String {
        format!(
            "{}(statistic={:.4}, df={}, p_value={:.4})",
            self.test_name, self.statistic, self.df, self.p_value
        )
    }
}
pub(crate) fn likelihood_ratio_test(
    loglik_full: f64,
    loglik_reduced: f64,
    df: usize,
) -> TestResult {
    let statistic = 2.0 * (loglik_full - loglik_reduced);
    let p_value = chi2_sf(statistic, df);
    TestResult {
        statistic,
        df,
        p_value,
        test_name: "LikelihoodRatioTest".to_string(),
    }
}
pub fn wald_test(coefficients: &[f64], std_errors: &[f64]) -> TestResult {
    let n = coefficients.len();
    let mut statistic = 0.0;
    for i in 0..n {
        if std_errors[i] > 0.0 {
            let z = coefficients[i] / std_errors[i];
            statistic += z * z;
        }
    }
    let p_value = chi2_sf(statistic, n);
    TestResult {
        statistic,
        df: n,
        p_value,
        test_name: "WaldTest".to_string(),
    }
}
pub fn score_test(score_vector: &[f64], information_matrix: &[Vec<f64>]) -> TestResult {
    let n = score_vector.len();
    if n == 0 {
        return TestResult {
            statistic: 0.0,
            df: 0,
            p_value: 1.0,
            test_name: "ScoreTest".to_string(),
        };
    }

    let mat = match vec_to_square_array2(information_matrix, n) {
        Some(mat) if score_vector.iter().all(|value| value.is_finite()) => mat,
        _ => return invalid_test_result("ScoreTest", n),
    };
    let inv_info = match matrix_inverse(&mat) {
        Some(inv) => inv,
        None => return invalid_test_result("ScoreTest", n),
    };

    let mut statistic = 0.0;
    for i in 0..n {
        for j in 0..n {
            statistic += score_vector[i] * inv_info[[i, j]] * score_vector[j];
        }
    }
    if !statistic.is_finite() {
        return invalid_test_result("ScoreTest", n);
    }
    let p_value = chi2_sf(statistic, n);
    if !p_value.is_finite() {
        return invalid_test_result("ScoreTest", n);
    }
    TestResult {
        statistic,
        df: n,
        p_value,
        test_name: "ScoreTest".to_string(),
    }
}

fn vec_to_square_array2(matrix: &[Vec<f64>], n: usize) -> Option<Array2<f64>> {
    if n == 0 {
        return Some(Array2::zeros((0, 0)));
    }
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut arr = Array2::zeros((n, n));
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if !val.is_finite() {
                return None;
            }
            arr[[i, j]] = val;
        }
    }
    Some(arr)
}
#[pyfunction]
pub fn lrt_test(loglik_full: f64, loglik_reduced: f64, df: usize) -> PyResult<TestResult> {
    if !loglik_full.is_finite() {
        return Err(value_error("loglik_full must be finite"));
    }
    if !loglik_reduced.is_finite() {
        return Err(value_error("loglik_reduced must be finite"));
    }
    if df == 0 {
        return Err(value_error("df must be positive"));
    }
    Ok(likelihood_ratio_test(loglik_full, loglik_reduced, df))
}
#[pyfunction]
pub(crate) fn wald_test_py(coefficients: Vec<f64>, std_errors: Vec<f64>) -> PyResult<TestResult> {
    if coefficients.len() != std_errors.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coefficients and std_errors must have the same length",
        ));
    }
    if coefficients.is_empty() {
        return Err(value_error("coefficients cannot be empty"));
    }
    validate_finite_slice(&coefficients, "coefficients")?;
    validate_positive_finite_slice(&std_errors, "std_errors")?;
    Ok(wald_test(&coefficients, &std_errors))
}
#[pyfunction]
pub(crate) fn score_test_py(
    score_vector: Vec<f64>,
    information_matrix: Vec<Vec<f64>>,
) -> PyResult<TestResult> {
    if score_vector.len() != information_matrix.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "score_vector length must match information_matrix dimensions",
        ));
    }
    if score_vector.is_empty() {
        return Err(value_error("score_vector cannot be empty"));
    }
    validate_finite_slice(&score_vector, "score_vector")?;
    validate_square_finite_matrix(
        &information_matrix,
        score_vector.len(),
        "information_matrix",
    )?;
    let result = score_test(&score_vector, &information_matrix);
    if !result.statistic.is_finite() || !result.p_value.is_finite() {
        return Err(value_error(
            "information_matrix is singular or invalid for score test",
        ));
    }
    Ok(result)
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ProportionalityTest {
    #[pyo3(get)]
    pub variable_names: Vec<String>,
    #[pyo3(get)]
    pub chi2_values: Vec<f64>,
    #[pyo3(get)]
    pub p_values: Vec<f64>,
    #[pyo3(get)]
    pub global_chi2: f64,
    #[pyo3(get)]
    pub global_df: usize,
    #[pyo3(get)]
    pub global_p_value: f64,
}
#[pymethods]
impl ProportionalityTest {
    #[new]
    fn new(
        variable_names: Vec<String>,
        chi2_values: Vec<f64>,
        p_values: Vec<f64>,
        global_chi2: f64,
        global_df: usize,
        global_p_value: f64,
    ) -> Self {
        Self {
            variable_names,
            chi2_values,
            p_values,
            global_chi2,
            global_df,
            global_p_value,
        }
    }
}
pub(crate) fn proportional_hazards_test(
    schoenfeld_residuals: &[Vec<f64>],
    event_times: &[f64],
    _weights: Option<&[f64]>,
) -> ProportionalityTest {
    let n_events = schoenfeld_residuals.len();
    let n_vars = if n_events > 0 {
        schoenfeld_residuals[0].len()
    } else {
        0
    };
    if n_events < 2 || n_vars == 0 {
        return ProportionalityTest {
            variable_names: vec![],
            chi2_values: vec![],
            p_values: vec![],
            global_chi2: 0.0,
            global_df: 0,
            global_p_value: 1.0,
        };
    }
    let mut sorted_indices: Vec<usize> = (0..n_events).collect();
    sorted_indices.sort_by(|&a, &b| {
        event_times[a]
            .partial_cmp(&event_times[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let transformed_time: Vec<f64> = sorted_indices.iter().map(|&idx| event_times[idx]).collect();
    let mut chi2_values = Vec::with_capacity(n_vars);
    let mut p_values = Vec::with_capacity(n_vars);
    let mut global_chi2 = 0.0;
    for var in 0..n_vars {
        let residuals: Vec<f64> = sorted_indices
            .iter()
            .filter_map(|&i| {
                schoenfeld_residuals
                    .get(i)
                    .and_then(|row| row.get(var).copied())
            })
            .collect();
        let mean_time: f64 = transformed_time.iter().sum::<f64>() / n_events as f64;
        let mean_resid: f64 = residuals.iter().sum::<f64>() / n_events as f64;
        let mut cov = 0.0;
        let mut var_time = 0.0;
        let mut var_resid = 0.0;
        for i in 0..n_events {
            let r_diff = transformed_time[i] - mean_time;
            let resid_diff = residuals[i] - mean_resid;
            cov += r_diff * resid_diff;
            var_time += r_diff * r_diff;
            var_resid += resid_diff * resid_diff;
        }
        let correlation = if var_time > 0.0 && var_resid > 0.0 {
            cov / (var_time.sqrt() * var_resid.sqrt())
        } else {
            0.0
        };
        let chi2 = correlation * correlation * (n_events - 2) as f64;
        let p_value = chi2_sf(chi2, 1);
        chi2_values.push(chi2);
        p_values.push(p_value);
        global_chi2 += chi2;
    }
    let global_p_value = chi2_sf(global_chi2, n_vars);
    ProportionalityTest {
        variable_names: (0..n_vars).map(|i| format!("var{}", i)).collect(),
        chi2_values,
        p_values,
        global_chi2,
        global_df: n_vars,
        global_p_value,
    }
}
#[pyfunction]
pub fn ph_test(
    schoenfeld_residuals: Vec<Vec<f64>>,
    event_times: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<ProportionalityTest> {
    if event_times.len() != schoenfeld_residuals.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "event_times must have the same length as schoenfeld_residuals",
        ));
    }
    if event_times.len() < 2 {
        return Err(value_error("at least two event_times are required"));
    }
    validate_finite_slice(&event_times, "event_times")?;
    if let Some(first_row) = schoenfeld_residuals.first() {
        let width = first_row.len();
        if width == 0 {
            return Err(value_error("schoenfeld_residuals rows cannot be empty"));
        }
        if schoenfeld_residuals.iter().any(|row| row.len() != width) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "schoenfeld_residuals must be rectangular",
            ));
        }
        for row in &schoenfeld_residuals {
            validate_finite_slice(row, "schoenfeld_residuals")?;
        }
    } else {
        return Err(value_error("schoenfeld_residuals cannot be empty"));
    }
    if let Some(weights) = weights.as_ref() {
        if weights.len() != event_times.len() {
            return Err(value_error(
                "weights must have the same length as event_times",
            ));
        }
        validate_finite_slice(weights, "weights")?;
    }
    let weights_ref = weights.as_deref();
    Ok(proportional_hazards_test(
        &schoenfeld_residuals,
        &event_times,
        weights_ref,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_hazards_test_uses_supplied_transformed_time_axis() {
        let residuals = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
        let rank_time = vec![1.0, 2.0, 3.0, 4.0];
        let uneven_time = vec![1.0, 2.0, 10.0, 11.0];

        let rank_result = proportional_hazards_test(&residuals, &rank_time, None);
        let uneven_result = proportional_hazards_test(&residuals, &uneven_time, None);

        assert_ne!(rank_result.chi2_values[0], uneven_result.chi2_values[0]);
    }

    #[test]
    fn ph_test_validates_time_length_and_rectangular_residuals() {
        assert!(ph_test(vec![vec![1.0], vec![2.0]], vec![1.0], None).is_err());
        assert!(ph_test(vec![vec![1.0], vec![2.0, 3.0]], vec![1.0, 2.0], None).is_err());
    }

    #[test]
    fn public_hypothesis_tests_validate_numeric_inputs() {
        assert!(lrt_test(f64::NAN, -12.0, 1).is_err());
        assert!(lrt_test(-10.0, -12.0, 0).is_err());
        assert!(wald_test_py(vec![], vec![]).is_err());
        assert!(wald_test_py(vec![f64::INFINITY], vec![1.0]).is_err());
        assert!(wald_test_py(vec![1.0], vec![0.0]).is_err());
        assert!(score_test_py(vec![], vec![]).is_err());
        assert!(score_test_py(vec![1.0], vec![vec![f64::NAN]]).is_err());
        assert!(score_test_py(vec![1.0, 2.0], vec![vec![1.0], vec![0.0, 1.0]]).is_err());
        assert!(ph_test(vec![vec![1.0]], vec![1.0], None).is_err());
        assert!(ph_test(vec![vec![f64::NAN], vec![2.0]], vec![1.0, 2.0], None).is_err());
        assert!(ph_test(vec![vec![1.0], vec![2.0]], vec![1.0, 2.0], Some(vec![1.0])).is_err());
        assert!(
            ph_test(
                vec![vec![1.0], vec![2.0]],
                vec![1.0, 2.0],
                Some(vec![1.0, f64::INFINITY]),
            )
            .is_err()
        );
    }

    #[test]
    fn direct_score_test_handles_bad_matrix_shape_without_panicking() {
        let result = score_test(&[1.0, 2.0], &[vec![1.0], vec![0.0, 1.0]]);

        assert!(result.statistic.is_nan());
        assert!(result.p_value.is_nan());
    }
}
