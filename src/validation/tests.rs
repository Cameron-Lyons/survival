use crate::utilities::matrix::matrix_inverse;
use crate::utilities::statistical::chi2_sf;
use ndarray::Array2;
use pyo3::prelude::*;
#[derive(Debug, Clone)]
#[pyclass]
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
pub fn likelihood_ratio_test(loglik_full: f64, loglik_reduced: f64, df: usize) -> TestResult {
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

    let mat = vec_to_array2(information_matrix);
    let inv_info = match matrix_inverse(&mat) {
        Some(inv) => inv,
        None => {
            return TestResult {
                statistic: f64::NAN,
                df: n,
                p_value: f64::NAN,
                test_name: "ScoreTest".to_string(),
            };
        }
    };

    let mut statistic = 0.0;
    for i in 0..n {
        for j in 0..n {
            statistic += score_vector[i] * inv_info[[i, j]] * score_vector[j];
        }
    }
    let p_value = chi2_sf(statistic, n);
    TestResult {
        statistic,
        df: n,
        p_value,
        test_name: "ScoreTest".to_string(),
    }
}

fn vec_to_array2(matrix: &[Vec<f64>]) -> Array2<f64> {
    let n = matrix.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }
    let m = matrix[0].len();
    let mut arr = Array2::zeros((n, m));
    for (i, row) in matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }
    arr
}
#[pyfunction]
pub fn lrt_test(loglik_full: f64, loglik_reduced: f64, df: usize) -> PyResult<TestResult> {
    Ok(likelihood_ratio_test(loglik_full, loglik_reduced, df))
}
#[pyfunction]
pub fn wald_test_py(coefficients: Vec<f64>, std_errors: Vec<f64>) -> PyResult<TestResult> {
    if coefficients.len() != std_errors.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "coefficients and std_errors must have the same length",
        ));
    }
    Ok(wald_test(&coefficients, &std_errors))
}
#[pyfunction]
pub fn score_test_py(
    score_vector: Vec<f64>,
    information_matrix: Vec<Vec<f64>>,
) -> PyResult<TestResult> {
    if score_vector.len() != information_matrix.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "score_vector length must match information_matrix dimensions",
        ));
    }
    Ok(score_test(&score_vector, &information_matrix))
}
#[derive(Debug, Clone)]
#[pyclass]
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
pub fn proportional_hazards_test(
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
    if n_events == 0 || n_vars == 0 {
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
    let ranks: Vec<f64> = (1..=n_events).map(|r| r as f64).collect();
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
        let mean_rank = (n_events as f64 + 1.0) / 2.0;
        let mean_resid: f64 = residuals.iter().sum::<f64>() / n_events as f64;
        let mut cov = 0.0;
        let mut var_rank = 0.0;
        let mut var_resid = 0.0;
        for i in 0..n_events {
            let r_diff = ranks[i] - mean_rank;
            let resid_diff = residuals[i] - mean_resid;
            cov += r_diff * resid_diff;
            var_rank += r_diff * r_diff;
            var_resid += resid_diff * resid_diff;
        }
        let correlation = if var_rank > 0.0 && var_resid > 0.0 {
            cov / (var_rank.sqrt() * var_resid.sqrt())
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
    let weights_ref = weights.as_deref();
    Ok(proportional_hazards_test(
        &schoenfeld_residuals,
        &event_times,
        weights_ref,
    ))
}
