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

fn chi2_sf(x: f64, df: usize) -> f64 {
    if x <= 0.0 || df == 0 {
        return 1.0;
    }

    let k = df as f64 / 2.0;
    let x_half = x / 2.0;

    let ln_gamma_k = ln_gamma(k);
    let regularized_gamma = lower_incomplete_gamma(k, x_half) / ln_gamma_k.exp();

    1.0 - regularized_gamma
}

fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];

    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();

    let mut ser = 1.000000000190015;
    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }

    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn lower_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        ln_gamma(a).exp() - gamma_continued_fraction(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = 100;

    let mut sum = 1.0 / a;
    let mut term = sum;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let eps = 1e-10;
    let max_iter = 100;

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / 1e-30;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = b + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }

    (-x + a * x.ln() - ln_gamma(a)).exp() * h
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

    let inv_info = invert_matrix(information_matrix);

    let mut statistic = 0.0;
    for i in 0..n {
        for j in 0..n {
            statistic += score_vector[i] * inv_info[i][j] * score_vector[j];
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

fn invert_matrix(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }

    let mut aug: Vec<Vec<f64>> = matrix.to_vec();
    for i in 0..n {
        aug[i].extend(vec![0.0; n]);
        aug[i][n + i] = 1.0;
    }

    for i in 0..n {
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-10 {
            continue;
        }

        let pivot = aug[i][i];
        for val in aug[i].iter_mut().take(2 * n) {
            *val /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                let aug_i_clone: Vec<f64> = aug[i].clone();
                for (j, aug_kj) in aug[k].iter_mut().enumerate().take(2 * n) {
                    *aug_kj -= factor * aug_i_clone[j];
                }
            }
        }
    }

    aug.iter().map(|row| row[n..].to_vec()).collect()
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
    sorted_indices.sort_by(|&a, &b| event_times[a].partial_cmp(&event_times[b]).unwrap_or(std::cmp::Ordering::Equal));

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
