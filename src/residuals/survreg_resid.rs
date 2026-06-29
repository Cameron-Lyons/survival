use crate::internal::statistical::{ln_gamma, normal_cdf};
use pyo3::prelude::*;

const LOG_PROBABILITY_FLOOR: f64 = -690.0;
const PROBABILITY_FLOOR: f64 = 1e-300;
const SURVREG_MATRIX_STEP: f64 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SurvregResidType {
    Response,
    Deviance,
    Dfbeta,
    Dfbetas,
    Working,
    Ldcase,
    Ldresp,
    Ldshape,
    Matrix,
}

impl SurvregResidType {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "response" => Some(SurvregResidType::Response),
            "deviance" => Some(SurvregResidType::Deviance),
            "dfbeta" => Some(SurvregResidType::Dfbeta),
            "dfbetas" => Some(SurvregResidType::Dfbetas),
            "working" => Some(SurvregResidType::Working),
            "ldcase" => Some(SurvregResidType::Ldcase),
            "ldresp" => Some(SurvregResidType::Ldresp),
            "ldshape" => Some(SurvregResidType::Ldshape),
            "matrix" => Some(SurvregResidType::Matrix),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvregResiduals {
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub residual_type: String,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl SurvregResiduals {
    fn __repr__(&self) -> String {
        format!(
            "SurvregResiduals(type='{}', n={})",
            self.residual_type, self.n
        )
    }
}

fn extreme_value_cdf(z: f64) -> f64 {
    1.0 - (-z.exp()).exp()
}

fn extreme_value_pdf(z: f64) -> f64 {
    let ez = z.exp();
    ez * (-ez).exp()
}

fn logistic_cdf(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn logistic_pdf(z: f64) -> f64 {
    let ez = (-z).exp();
    ez / ((1.0 + ez) * (1.0 + ez))
}

fn gaussian_cdf(z: f64) -> f64 {
    normal_cdf(z)
}

fn gaussian_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h
}

fn student_t_pdf(z: f64, df: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_nan() { f64::NAN } else { 0.0 };
    }
    let log_coef = ln_gamma((df + 1.0) / 2.0)
        - ln_gamma(df / 2.0)
        - 0.5 * (df.ln() + std::f64::consts::PI.ln());
    (log_coef - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln()).exp()
}

fn student_t_cdf(z: f64, df: f64) -> f64 {
    if z == f64::INFINITY {
        return 1.0;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if z == 0.0 {
        return 0.5;
    }
    let x = df / (df + z * z);
    let ibeta = regularized_beta(df / 2.0, 0.5, x);
    if z > 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

fn student_t_pdf_derivative(z: f64, df: f64) -> f64 {
    student_t_pdf(z, df) * (-(df + 1.0) * z / (df + z * z))
}

fn distribution_key(distribution: &str) -> String {
    distribution.to_lowercase().replace('-', "_")
}

fn is_valid_distribution_key(key: &str) -> bool {
    matches!(
        key,
        "weibull"
            | "exponential"
            | "rayleigh"
            | "extreme"
            | "extreme_value"
            | "extremevalue"
            | "logistic"
            | "gaussian"
            | "normal"
            | "lognormal"
            | "log_normal"
            | "loggaussian"
            | "log_gaussian"
            | "loglogistic"
            | "log_logistic"
            | "t"
            | "student"
            | "student_t"
            | "studentt"
    )
}

fn is_student_t_distribution_key(key: &str) -> bool {
    matches!(key, "t" | "student" | "student_t" | "studentt")
}

fn invalid_distribution_error() -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "distribution must be one of weibull, exponential, rayleigh, extreme, gaussian, logistic, loggaussian, lognormal, loglogistic, or t",
    )
}

fn validate_distribution(distribution: &str) -> PyResult<()> {
    let key = distribution_key(distribution);
    if is_valid_distribution_key(&key) {
        return Ok(());
    }
    Err(invalid_distribution_error())
}

fn validated_distribution_key(distribution: &str) -> String {
    let key = distribution_key(distribution);
    debug_assert!(
        is_valid_distribution_key(&key),
        "distribution was validated"
    );
    key
}

fn response_uses_log_transform_key(key: &str) -> bool {
    if matches!(
        key,
        "weibull"
            | "exponential"
            | "rayleigh"
            | "lognormal"
            | "log_normal"
            | "loggaussian"
            | "log_gaussian"
            | "loglogistic"
            | "log_logistic"
    ) {
        return true;
    }
    match key {
        "extreme" | "extreme_value" | "extremevalue" | "gaussian" | "normal" | "logistic" | "t"
        | "student" | "student_t" | "studentt" => false,
        _ => unreachable!("distribution was validated"),
    }
}

fn response_uses_log_transform(distribution: &str) -> bool {
    response_uses_log_transform_key(&validated_distribution_key(distribution))
}

fn response_time_value(time: f64, distribution: &str) -> f64 {
    if response_uses_log_transform(distribution) {
        time.ln()
    } else {
        time
    }
}

fn inverse_response_time_value(value: f64, distribution: &str) -> f64 {
    if response_uses_log_transform(distribution) {
        value.exp()
    } else {
        value
    }
}

fn response_time_value_for_key(time: f64, key: &str) -> f64 {
    if response_uses_log_transform_key(key) {
        time.ln()
    } else {
        time
    }
}

fn inverse_response_time_value_for_key(value: f64, key: &str) -> f64 {
    if response_uses_log_transform_key(key) {
        value.exp()
    } else {
        value
    }
}

fn validated_distribution_parameter_for_key(
    key: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Option<f64>> {
    match key {
        key if is_student_t_distribution_key(key) => {
            let df = distribution_parameter.unwrap_or(4.0);
            if !df.is_finite() || df <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "distribution_parameter for Student-t residuals must be a positive finite degrees-of-freedom value",
                ));
            }
            Ok(Some(df))
        }
        _ => {
            if distribution_parameter.is_some() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "distribution_parameter is only supported for distribution='t'",
                ));
            }
            Ok(None)
        }
    }
}

fn cdf_for_key(key: &str, distribution_parameter: Option<f64>, z: f64) -> f64 {
    match key {
        "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value" | "extremevalue" => {
            extreme_value_cdf(z)
        }
        "logistic" | "loglogistic" | "log_logistic" => logistic_cdf(z),
        "gaussian" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" | "normal" => {
            gaussian_cdf(z)
        }
        "t" | "student" | "student_t" | "studentt" => student_t_cdf(
            z,
            distribution_parameter.expect("Student-t df was validated"),
        ),
        _ => unreachable!("distribution was validated"),
    }
}

fn pdf_for_key(key: &str, distribution_parameter: Option<f64>, z: f64) -> f64 {
    match key {
        "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value" | "extremevalue" => {
            extreme_value_pdf(z)
        }
        "logistic" | "loglogistic" | "log_logistic" => logistic_pdf(z),
        "gaussian" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" | "normal" => {
            gaussian_pdf(z)
        }
        "t" | "student" | "student_t" | "studentt" => student_t_pdf(
            z,
            distribution_parameter.expect("Student-t df was validated"),
        ),
        _ => unreachable!("distribution was validated"),
    }
}

fn pdf_derivative_for_key(key: &str, distribution_parameter: Option<f64>, z: f64) -> f64 {
    match key {
        "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value" | "extremevalue" => {
            let ez = z.exp();
            ez * (-ez).exp() * (1.0 - ez)
        }
        "logistic" | "loglogistic" | "log_logistic" => {
            let ez = (-z).exp();
            let denom = (1.0 + ez).powi(3);
            ez * (ez - 1.0) / denom
        }
        "gaussian" | "normal" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
            -z * gaussian_pdf(z)
        }
        "t" | "student" | "student_t" | "studentt" => student_t_pdf_derivative(
            z,
            distribution_parameter.expect("Student-t df was validated"),
        ),
        _ => unreachable!("distribution was validated"),
    }
}

fn log_expm1_positive(value: f64) -> f64 {
    if value < 40.0 {
        value.exp_m1().ln()
    } else {
        value
    }
}

fn log_one_minus_exp_neg(value: f64) -> f64 {
    if value <= 0.0 {
        return LOG_PROBABILITY_FLOOR;
    }
    let probability = -(-value).exp_m1();
    log_positive(probability)
}

fn transformed_interval_width(
    time: &[f64],
    time2: Option<&[f64]>,
    idx: usize,
    scale: f64,
    distribution: &str,
) -> f64 {
    let upper = time2.expect("validated time2 length")[idx];
    (response_time_value(upper, distribution) - response_time_value(time[idx], distribution))
        / scale
}

fn survreg_saturated_center_loglik(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    idx: usize,
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<(f64, f64)> {
    let y = response_time_value(time[idx], distribution);
    let event = status[idx];
    let key = validated_distribution_key(distribution);
    let distribution_parameter =
        validated_distribution_parameter_for_key(&key, distribution_parameter)?;

    if event != 3 {
        let loglik = if event == 1 {
            match key.as_str() {
                "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value"
                | "extremevalue" => -(1.0 + scale.ln()),
                "logistic" | "loglogistic" | "log_logistic" => -(4.0 * scale).ln(),
                "gaussian" | "normal" | "lognormal" | "log_normal" | "loggaussian"
                | "log_gaussian" => -(std::f64::consts::TAU.sqrt() * scale).ln(),
                "t" | "student" | "student_t" | "studentt" => -(student_t_pdf(
                    0.0,
                    distribution_parameter.expect("Student-t df was validated"),
                ) * scale)
                    .ln(),
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "distribution must be one of weibull, exponential, rayleigh, extreme, gaussian, logistic, loggaussian, lognormal, loglogistic, or t",
                    ));
                }
            }
        } else {
            0.0
        };
        return Ok((y, loglik));
    }

    let width = transformed_interval_width(time, time2, idx, scale, distribution);
    let upper = response_time_value(time2.expect("validated time2 length")[idx], distribution);
    match key.as_str() {
        "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value" | "extremevalue" => {
            let log_temp = width.ln() - log_expm1_positive(width);
            let center = y - log_temp;
            let temp = log_temp.exp();
            let loglik = -temp + log_one_minus_exp_neg(width.exp());
            Ok((center, loglik))
        }
        "logistic" | "loglogistic" | "log_logistic" => {
            let center = (y + upper) / 2.0;
            let loglik = log_positive((width / 4.0).tanh());
            Ok((center, loglik))
        }
        "gaussian" | "normal" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
            let center = (y + upper) / 2.0;
            let probability = 2.0 * normal_cdf(width / 2.0) - 1.0;
            Ok((center, log_positive(probability)))
        }
        "t" | "student" | "student_t" | "studentt" => {
            let center = (y + upper) / 2.0;
            let df = distribution_parameter.expect("Student-t df was validated");
            Ok((center, (1.0 - 2.0 * student_t_cdf(width / 2.0, df)).ln()))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "distribution must be one of weibull, exponential, rayleigh, extreme, gaussian, logistic, loggaussian, lognormal, loglogistic, or t",
        )),
    }
}

fn log_positive(value: f64) -> f64 {
    if value > PROBABILITY_FLOOR {
        value.ln()
    } else {
        LOG_PROBABILITY_FLOOR
    }
}

fn interval_probability_for_key(
    key: &str,
    distribution_parameter: Option<f64>,
    lower_z: f64,
    upper_z: f64,
) -> f64 {
    if lower_z > 0.0 {
        (1.0 - cdf_for_key(key, distribution_parameter, lower_z))
            - (1.0 - cdf_for_key(key, distribution_parameter, upper_z))
    } else {
        cdf_for_key(key, distribution_parameter, upper_z)
            - cdf_for_key(key, distribution_parameter, lower_z)
    }
}

fn has_interval_censoring(status: &[i32]) -> bool {
    status.iter().any(|&value| value == 2 || value == 3)
}

fn validate_positive_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must not be empty"
        )));
    }
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if value <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{name}[{idx}] must be positive"
            )));
        }
    }
    Ok(())
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_status_values(status: &[i32]) -> PyResult<()> {
    for &value in status {
        if !matches!(value, 0..=3) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "status must contain only 0/1/2/3 values",
            ));
        }
    }
    Ok(())
}

fn validate_scale(scale: f64) -> PyResult<()> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "scale must be a finite positive value",
        ));
    }
    Ok(())
}

fn validate_time2_for_interval_residuals(
    time: &[f64],
    status: &[i32],
    time2: Option<&[f64]>,
) -> PyResult<()> {
    let has_interval_rows = status.contains(&3);
    if !has_interval_rows && time2.is_none() {
        return Ok(());
    }
    let Some(values) = time2 else {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time2 is required for interval-censored rows",
        ));
    };
    if values.len() != time.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time2 must have the same length as time",
        ));
    }
    for (idx, ((&start, &end), &event)) in time
        .iter()
        .zip(values.iter())
        .zip(status.iter())
        .enumerate()
    {
        if event != 3 {
            continue;
        }
        if !end.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time2 contains non-finite interval endpoint at index {idx}"
            )));
        }
        if end <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time2[{idx}] must be positive"
            )));
        }
        if end <= start {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time2[{idx}] must be greater than time[{idx}] for interval-censored rows"
            )));
        }
    }
    Ok(())
}

fn validate_survreg_residual_inputs(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
) -> PyResult<()> {
    validate_positive_finite("time", time)?;
    validate_status_values(status)?;
    validate_finite_values("linear_pred", linear_pred)?;
    validate_scale(scale)
}

fn validate_covariates(covariates: &[Vec<f64>]) -> PyResult<usize> {
    let width = covariates.first().map_or(0, Vec::len);
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != width {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "covariates row {row_idx} has {} columns but expected {width}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariates[{row_idx}][{col_idx}] contains non-finite value"
                )));
            }
        }
    }
    Ok(width)
}

fn validate_variance_matrix(var_matrix: &[Vec<f64>], width: usize) -> PyResult<()> {
    if var_matrix.len() < width {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "var_matrix must have at least {width} rows"
        )));
    }
    for (row_idx, row) in var_matrix.iter().take(width).enumerate() {
        if row.len() < width {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "var_matrix row {row_idx} has {} columns but expected at least {width}",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().take(width).enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "var_matrix[{row_idx}][{col_idx}] contains non-finite value"
                )));
            }
        }
    }
    Ok(())
}

fn validate_derivative_matrix(matrix: &[Vec<f64>]) -> PyResult<()> {
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != 6 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "derivative_matrix row {row_idx} has {} columns but expected 6",
                row.len()
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "derivative_matrix[{row_idx}][{col_idx}] contains non-finite value"
                )));
            }
        }
    }
    Ok(())
}

fn validate_scales_and_strata(scales: &[f64], strata: &[usize], n: usize) -> PyResult<()> {
    if strata.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "strata must have the same length as derivative_matrix",
        ));
    }
    validate_positive_finite("scales", scales)?;
    for (idx, &stratum) in strata.iter().enumerate() {
        if stratum >= scales.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "strata[{idx}] is out of bounds for {} scale value(s)",
                scales.len()
            )));
        }
    }
    Ok(())
}

pub(crate) fn compute_response_residuals(
    time: &[f64],
    linear_pred: &[f64],
    distribution: &str,
) -> Vec<f64> {
    let key = validated_distribution_key(distribution);
    time.iter()
        .zip(linear_pred.iter())
        .map(|(&t, &lp)| {
            inverse_response_time_value_for_key(response_time_value_for_key(t, &key), &key)
                - inverse_response_time_value_for_key(lp, &key)
        })
        .collect()
}

#[cfg(test)]
fn compute_response_residuals_censored(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> PyResult<Vec<f64>> {
    compute_response_residuals_censored_with_parameter(
        time,
        time2,
        status,
        linear_pred,
        scale,
        distribution,
        None,
    )
}

pub(crate) fn compute_response_residuals_censored_with_parameter(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<f64>> {
    validate_time2_for_interval_residuals(time, status, time2)?;
    let mut residuals = Vec::with_capacity(time.len());
    for (idx, &linear_predictor) in linear_pred.iter().enumerate().take(time.len()) {
        let (center, _) = survreg_saturated_center_loglik(
            time,
            time2,
            status,
            idx,
            scale,
            distribution,
            distribution_parameter,
        )?;
        residuals.push(
            inverse_response_time_value(center, distribution)
                - inverse_response_time_value(linear_predictor, distribution),
        );
    }
    Ok(residuals)
}

#[cfg(test)]
fn compute_deviance_residuals_survreg(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> PyResult<Vec<f64>> {
    compute_deviance_residuals_survreg_with_parameter(
        time,
        time2,
        status,
        linear_pred,
        scale,
        distribution,
        None,
    )
}

pub(crate) fn compute_deviance_residuals_survreg_with_parameter(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<f64>> {
    let derivative_matrix = compute_survreg_residual_matrix_with_parameter(
        time,
        time2,
        status,
        linear_pred,
        scale,
        distribution,
        distribution_parameter,
    )?;
    compute_deviance_residuals_from_derivative_matrix_with_parameter(
        &derivative_matrix,
        time,
        time2,
        status,
        scale,
        distribution,
        distribution_parameter,
    )
}

#[cfg(test)]
fn compute_deviance_residuals_from_derivative_matrix(
    derivative_matrix: &[Vec<f64>],
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    scale: f64,
    distribution: &str,
) -> PyResult<Vec<f64>> {
    compute_deviance_residuals_from_derivative_matrix_with_parameter(
        derivative_matrix,
        time,
        time2,
        status,
        scale,
        distribution,
        None,
    )
}

pub(crate) fn compute_deviance_residuals_from_derivative_matrix_with_parameter(
    derivative_matrix: &[Vec<f64>],
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<f64>> {
    validate_derivative_matrix(derivative_matrix)?;
    validate_time2_for_interval_residuals(time, status, time2)?;
    let working = compute_working_residuals_from_derivative_matrix(derivative_matrix)?;
    let mut residuals = Vec::with_capacity(time.len());

    for idx in 0..time.len() {
        let (_, saturated_loglik) = survreg_saturated_center_loglik(
            time,
            time2,
            status,
            idx,
            scale,
            distribution,
            distribution_parameter,
        )?;
        let magnitude = (2.0 * (saturated_loglik - derivative_matrix[idx][0]))
            .max(0.0)
            .sqrt();
        residuals.push(if working[idx] > 0.0 {
            magnitude
        } else if working[idx] < 0.0 {
            -magnitude
        } else {
            0.0
        });
    }

    Ok(residuals)
}

#[cfg(test)]
fn compute_working_residuals(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> Vec<f64> {
    compute_working_residuals_with_parameter(time, status, linear_pred, scale, distribution, None)
}

pub(crate) fn compute_working_residuals_with_parameter(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    let key = validated_distribution_key(distribution);
    let distribution_parameter =
        validated_distribution_parameter_for_key(&key, distribution_parameter)
            .expect("distribution parameter was validated");

    for i in 0..n {
        let y = response_time_value_for_key(time[i], &key);
        let z = (y - linear_pred[i]) / scale;

        let resid = if status[i] == 1 {
            let f = pdf_for_key(&key, distribution_parameter, z);
            let f_prime = pdf_derivative_for_key(&key, distribution_parameter, z);
            if f.abs() > 1e-300 { -f_prime / f } else { 0.0 }
        } else {
            let surv = 1.0 - cdf_for_key(&key, distribution_parameter, z);
            let f = pdf_for_key(&key, distribution_parameter, z);
            if surv.abs() > 1e-300 { f / surv } else { 0.0 }
        };

        residuals.push(resid);
    }

    residuals
}

pub(crate) fn compute_dfbeta_survreg_with_parameter(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    linear_pred: &[f64],
    scale: f64,
    var_matrix: &[Vec<f64>],
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> Vec<Vec<f64>> {
    let n = time.len();
    let nvar = if n > 0 && !covariates.is_empty() {
        covariates[0].len()
    } else {
        return vec![];
    };

    let working = compute_working_residuals_with_parameter(
        time,
        status,
        linear_pred,
        scale,
        distribution,
        distribution_parameter,
    );

    let mut dfbeta = Vec::with_capacity(n);

    for i in 0..n {
        let mut row = Vec::with_capacity(nvar);
        for j in 0..nvar {
            let mut val = 0.0;
            for k in 0..nvar {
                if k < var_matrix.len() && j < var_matrix[k].len() {
                    val += var_matrix[k][j] * covariates[i][k] * working[i];
                }
            }
            row.push(val);
        }
        dfbeta.push(row);
    }

    dfbeta
}

pub(crate) fn compute_ldcase_with_parameter(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<f64>> {
    validate_time2_for_interval_residuals(time, status, time2)?;
    let n = time.len();

    let mut ld = Vec::with_capacity(n);

    let key = validated_distribution_key(distribution);
    let distribution_parameter =
        validated_distribution_parameter_for_key(&key, distribution_parameter)?;

    for i in 0..n {
        let y = response_time_value_for_key(time[i], &key);
        let z = (y - linear_pred[i]) / scale;

        let contrib = match status[i] {
            1 => {
                let f = pdf_for_key(&key, distribution_parameter, z) / scale;
                if f > 1e-300 { f.ln() } else { -690.0 }
            }
            0 => {
                let surv = 1.0 - cdf_for_key(&key, distribution_parameter, z);
                if surv > 1e-300 { surv.ln() } else { -690.0 }
            }
            2 => {
                let cdf = cdf_for_key(&key, distribution_parameter, z);
                if cdf > 1e-300 { cdf.ln() } else { -690.0 }
            }
            3 => {
                let end =
                    response_time_value_for_key(time2.expect("validated time2 length")[i], &key);
                let z2 = (end - linear_pred[i]) / scale;
                let prob = cdf_for_key(&key, distribution_parameter, z2)
                    - cdf_for_key(&key, distribution_parameter, z);
                if prob > 1e-300 { prob.ln() } else { -690.0 }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "status must contain only 0/1/2/3 values",
                ));
            }
        };

        ld.push(contrib);
    }

    Ok(ld)
}

fn survreg_loglik_contribution(
    time: f64,
    time2: Option<f64>,
    status: i32,
    linear_pred: f64,
    log_scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<f64> {
    let scale = log_scale.exp();
    let key = validated_distribution_key(distribution);
    let distribution_parameter =
        validated_distribution_parameter_for_key(&key, distribution_parameter)?;
    let y = response_time_value_for_key(time, &key);
    let z = (y - linear_pred) / scale;

    match status {
        1 => Ok(log_positive(
            pdf_for_key(&key, distribution_parameter, z) / scale,
        )),
        0 => Ok(log_positive(
            1.0 - cdf_for_key(&key, distribution_parameter, z),
        )),
        2 => Ok(log_positive(cdf_for_key(&key, distribution_parameter, z))),
        3 => {
            let Some(end) = time2 else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "time2 is required for interval-censored rows",
                ));
            };
            let z2 = (response_time_value_for_key(end, &key) - linear_pred) / scale;
            Ok(log_positive(interval_probability_for_key(
                &key,
                distribution_parameter,
                z,
                z2,
            )))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "status must contain only 0/1/2/3 values",
        )),
    }
}

fn eta_derivative_step(scale: f64) -> f64 {
    (SURVREG_MATRIX_STEP * scale).clamp(1e-6, 1e-3)
}

#[cfg(test)]
fn compute_survreg_residual_matrix(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> PyResult<Vec<Vec<f64>>> {
    compute_survreg_residual_matrix_with_parameter(
        time,
        time2,
        status,
        linear_pred,
        scale,
        distribution,
        None,
    )
}

pub(crate) fn compute_survreg_residual_matrix_with_parameter(
    time: &[f64],
    time2: Option<&[f64]>,
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    validate_time2_for_interval_residuals(time, status, time2)?;
    let key = validated_distribution_key(distribution);
    let distribution_parameter =
        validated_distribution_parameter_for_key(&key, distribution_parameter)?;
    let log_scale = scale.ln();
    let h_eta = eta_derivative_step(scale);
    let h_scale = SURVREG_MATRIX_STEP;
    let mut matrix = Vec::with_capacity(time.len());

    for i in 0..time.len() {
        let time2_i = time2.map(|values| values[i]);
        let eta = linear_pred[i];
        let eval = |eta_shift: f64, scale_shift: f64| {
            survreg_loglik_contribution(
                time[i],
                time2_i,
                status[i],
                eta + eta_shift,
                log_scale + scale_shift,
                distribution,
                distribution_parameter,
            )
        };

        let g = eval(0.0, 0.0)?;
        let eta_plus = eval(h_eta, 0.0)?;
        let eta_minus = eval(-h_eta, 0.0)?;
        let scale_plus = eval(0.0, h_scale)?;
        let scale_minus = eval(0.0, -h_scale)?;

        let dg = (eta_plus - eta_minus) / (2.0 * h_eta);
        let ddg = (eta_plus - 2.0 * g + eta_minus) / (h_eta * h_eta);
        let ds = (scale_plus - scale_minus) / (2.0 * h_scale);
        let dds = (scale_plus - 2.0 * g + scale_minus) / (h_scale * h_scale);
        let dsg = (eval(h_eta, h_scale)? - eval(h_eta, -h_scale)? - eval(-h_eta, h_scale)?
            + eval(-h_eta, -h_scale)?)
            / (4.0 * h_eta * h_scale);

        matrix.push(vec![g, dg, ddg, ds, dds, dsg]);
    }

    Ok(matrix)
}

pub(crate) fn compute_working_residuals_from_derivative_matrix(
    derivative_matrix: &[Vec<f64>],
) -> PyResult<Vec<f64>> {
    validate_derivative_matrix(derivative_matrix)?;
    Ok(derivative_matrix
        .iter()
        .map(|row| {
            let curvature = row[2];
            if curvature.abs() > 1e-12 {
                -row[1] / curvature
            } else {
                0.0
            }
        })
        .collect())
}

fn survreg_influence_score_row(
    deriv: &[f64],
    covariates: &[f64],
    scale: f64,
    stratum: usize,
    nstrat: usize,
    residual_type: SurvregResidType,
    rsigma: bool,
) -> Vec<f64> {
    let nvar = covariates.len();
    let mut score = vec![0.0; nvar + usize::from(rsigma) * nstrat];

    match residual_type {
        SurvregResidType::Ldcase => {
            for (col_idx, &value) in covariates.iter().enumerate() {
                score[col_idx] = deriv[1] * value;
            }
            if rsigma {
                score[nvar + stratum] = deriv[3];
            }
        }
        SurvregResidType::Ldresp => {
            for (col_idx, &value) in covariates.iter().enumerate() {
                score[col_idx] = deriv[2] * value * scale;
            }
            if rsigma {
                score[nvar + stratum] = deriv[5] * scale;
            }
        }
        SurvregResidType::Ldshape => {
            for (col_idx, &value) in covariates.iter().enumerate() {
                score[col_idx] = deriv[5] * value;
            }
            if rsigma {
                score[nvar + stratum] = deriv[4];
            }
        }
        _ => unreachable!(),
    }

    score
}

fn quadratic_row(score: &[f64], var_matrix: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for col_idx in 0..score.len() {
        let mut temp = 0.0;
        for row_idx in 0..score.len() {
            temp += score[row_idx] * var_matrix[row_idx][col_idx];
        }
        total += score[col_idx] * temp;
    }
    total
}

fn multiply_row_by_variance(score: &[f64], var_matrix: &[Vec<f64>]) -> Vec<f64> {
    let mut result = vec![0.0; score.len()];
    for col_idx in 0..score.len() {
        for row_idx in 0..score.len() {
            result[col_idx] += score[row_idx] * var_matrix[row_idx][col_idx];
        }
    }
    result
}

pub(crate) fn compute_survreg_influence_residuals(
    derivative_matrix: &[Vec<f64>],
    covariates: &[Vec<f64>],
    scales: &[f64],
    strata: &[usize],
    var_matrix: &[Vec<f64>],
    residual_type: SurvregResidType,
    rsigma: bool,
) -> PyResult<Vec<f64>> {
    let n = derivative_matrix.len();
    if covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must have the same number of rows as derivative_matrix",
        ));
    }
    validate_derivative_matrix(derivative_matrix)?;
    let nvar = validate_covariates(covariates)?;
    validate_scales_and_strata(scales, strata, n)?;
    let width = nvar + if rsigma { scales.len() } else { 0 };
    validate_variance_matrix(var_matrix, width)?;

    let mut residuals = Vec::with_capacity(n);
    for row_idx in 0..n {
        let stratum = strata[row_idx];
        let score = survreg_influence_score_row(
            &derivative_matrix[row_idx],
            &covariates[row_idx],
            scales[stratum],
            stratum,
            scales.len(),
            residual_type,
            rsigma,
        );
        residuals.push(quadratic_row(&score, var_matrix));
    }

    Ok(residuals)
}

pub(crate) fn compute_survreg_dfbeta_residuals(
    derivative_matrix: &[Vec<f64>],
    covariates: &[Vec<f64>],
    scales: &[f64],
    strata: &[usize],
    var_matrix: &[Vec<f64>],
    rsigma: bool,
    standardized: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let n = derivative_matrix.len();
    if covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "covariates must have the same number of rows as derivative_matrix",
        ));
    }
    validate_derivative_matrix(derivative_matrix)?;
    let nvar = validate_covariates(covariates)?;
    validate_scales_and_strata(scales, strata, n)?;
    let width = nvar + if rsigma { scales.len() } else { 0 };
    validate_variance_matrix(var_matrix, width)?;
    let scales_by_column: Vec<f64> = if standardized {
        (0..width)
            .map(|idx| var_matrix[idx][idx].abs().sqrt().max(1e-12))
            .collect()
    } else {
        vec![1.0; width]
    };

    let mut rows = Vec::with_capacity(n);
    for row_idx in 0..n {
        let stratum = strata[row_idx];
        let score = survreg_influence_score_row(
            &derivative_matrix[row_idx],
            &covariates[row_idx],
            scales[stratum],
            stratum,
            scales.len(),
            SurvregResidType::Ldcase,
            rsigma,
        );
        let mut row = multiply_row_by_variance(&score, var_matrix);
        for (col_idx, value) in row.iter_mut().enumerate() {
            *value /= scales_by_column[col_idx];
        }
        rows.push(row);
    }

    Ok(rows)
}

#[pyfunction]
#[pyo3(signature = (time, status, linear_pred, scale, distribution, time2=None, distribution_parameter=None))]
pub fn survreg_residual_matrix(
    time: Vec<f64>,
    status: Vec<i32>,
    linear_pred: Vec<f64>,
    scale: f64,
    distribution: String,
    time2: Option<Vec<f64>>,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let n = time.len();
    if status.len() != n || linear_pred.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and linear_pred must have the same length",
        ));
    }
    validate_survreg_residual_inputs(&time, &status, &linear_pred, scale)?;
    validate_distribution(&distribution)?;
    let key = validated_distribution_key(&distribution);
    validated_distribution_parameter_for_key(&key, distribution_parameter)?;

    compute_survreg_residual_matrix_with_parameter(
        &time,
        time2.as_deref(),
        &status,
        &linear_pred,
        scale,
        &distribution,
        distribution_parameter,
    )
}

#[pyfunction]
#[pyo3(signature = (derivative_matrix, covariates, scales, strata, var_matrix, rsigma=true, standardized=false))]
pub fn survreg_dfbeta_residuals(
    derivative_matrix: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    scales: Vec<f64>,
    strata: Vec<usize>,
    var_matrix: Vec<Vec<f64>>,
    rsigma: bool,
    standardized: bool,
) -> PyResult<Vec<Vec<f64>>> {
    compute_survreg_dfbeta_residuals(
        &derivative_matrix,
        &covariates,
        &scales,
        &strata,
        &var_matrix,
        rsigma,
        standardized,
    )
}

#[pyfunction]
#[pyo3(signature = (derivative_matrix, covariates, scales, strata, var_matrix, residual_type, rsigma=true))]
pub fn survreg_influence_residuals(
    derivative_matrix: Vec<Vec<f64>>,
    covariates: Vec<Vec<f64>>,
    scales: Vec<f64>,
    strata: Vec<usize>,
    var_matrix: Vec<Vec<f64>>,
    residual_type: String,
    rsigma: bool,
) -> PyResult<Vec<f64>> {
    let resid_type = SurvregResidType::from_str(&residual_type).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown survreg influence residual type: {residual_type}. Valid types: ldcase, ldresp, ldshape",
        ))
    })?;
    if !matches!(
        resid_type,
        SurvregResidType::Ldcase | SurvregResidType::Ldresp | SurvregResidType::Ldshape
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survreg influence residual type must be ldcase, ldresp, or ldshape",
        ));
    }

    compute_survreg_influence_residuals(
        &derivative_matrix,
        &covariates,
        &scales,
        &strata,
        &var_matrix,
        resid_type,
        rsigma,
    )
}

#[pyfunction]
#[pyo3(signature = (time, status, linear_pred, scale, distribution, residual_type="deviance".to_string(), time2=None, distribution_parameter=None))]
pub fn residuals_survreg(
    time: Vec<f64>,
    status: Vec<i32>,
    linear_pred: Vec<f64>,
    scale: f64,
    distribution: String,
    residual_type: String,
    time2: Option<Vec<f64>>,
    distribution_parameter: Option<f64>,
) -> PyResult<SurvregResiduals> {
    let n = time.len();
    if status.len() != n || linear_pred.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and linear_pred must have the same length",
        ));
    }

    let resid_type = SurvregResidType::from_str(&residual_type).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown residual type: {}. Valid types: response, deviance, working, ldcase, ldresp, ldshape, dfbeta, dfbetas, matrix",
            residual_type
        ))
    })?;
    if matches!(
        resid_type,
        SurvregResidType::Dfbeta | SurvregResidType::Dfbetas
    ) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survreg dfbeta residuals require covariates and a variance matrix; use dfbeta_survreg",
        ));
    }
    if matches!(resid_type, SurvregResidType::Matrix) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "survreg matrix residuals are matrix-valued; use survreg_residual_matrix",
        ));
    }
    validate_survreg_residual_inputs(&time, &status, &linear_pred, scale)?;
    validate_distribution(&distribution)?;
    let key = validated_distribution_key(&distribution);
    validated_distribution_parameter_for_key(&key, distribution_parameter)?;

    let residuals = match resid_type {
        SurvregResidType::Response => {
            if has_interval_censoring(&status) {
                compute_response_residuals_censored_with_parameter(
                    &time,
                    time2.as_deref(),
                    &status,
                    &linear_pred,
                    scale,
                    &distribution,
                    distribution_parameter,
                )?
            } else {
                compute_response_residuals(&time, &linear_pred, &distribution)
            }
        }
        SurvregResidType::Deviance => compute_deviance_residuals_survreg_with_parameter(
            &time,
            time2.as_deref(),
            &status,
            &linear_pred,
            scale,
            &distribution,
            distribution_parameter,
        )?,
        SurvregResidType::Working => {
            if has_interval_censoring(&status) || is_student_t_distribution_key(&key) {
                let derivative_matrix = compute_survreg_residual_matrix_with_parameter(
                    &time,
                    time2.as_deref(),
                    &status,
                    &linear_pred,
                    scale,
                    &distribution,
                    distribution_parameter,
                )?;
                compute_working_residuals_from_derivative_matrix(&derivative_matrix)?
            } else {
                compute_working_residuals_with_parameter(
                    &time,
                    &status,
                    &linear_pred,
                    scale,
                    &distribution,
                    distribution_parameter,
                )
            }
        }
        SurvregResidType::Ldcase | SurvregResidType::Ldresp | SurvregResidType::Ldshape => {
            compute_ldcase_with_parameter(
                &time,
                time2.as_deref(),
                &status,
                &linear_pred,
                scale,
                &distribution,
                distribution_parameter,
            )?
        }
        SurvregResidType::Dfbeta | SurvregResidType::Dfbetas => unreachable!(),
        SurvregResidType::Matrix => unreachable!(),
    };

    Ok(SurvregResiduals {
        residuals,
        residual_type,
        n,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, linear_pred, scale, var_matrix, distribution, time2=None, distribution_parameter=None))]
#[allow(clippy::too_many_arguments)]
pub fn dfbeta_survreg(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    linear_pred: Vec<f64>,
    scale: f64,
    var_matrix: Vec<Vec<f64>>,
    distribution: String,
    time2: Option<Vec<f64>>,
    distribution_parameter: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let n = time.len();
    if status.len() != n || linear_pred.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same length",
        ));
    }
    validate_survreg_residual_inputs(&time, &status, &linear_pred, scale)?;
    validate_distribution(&distribution)?;
    let key = validated_distribution_key(&distribution);
    validated_distribution_parameter_for_key(&key, distribution_parameter)?;
    let width = validate_covariates(&covariates)?;
    validate_variance_matrix(&var_matrix, width)?;

    if has_interval_censoring(&status) {
        let derivative_matrix = compute_survreg_residual_matrix_with_parameter(
            &time,
            time2.as_deref(),
            &status,
            &linear_pred,
            scale,
            &distribution,
            distribution_parameter,
        )?;
        let scales = vec![scale];
        let strata = vec![0; n];
        return compute_survreg_dfbeta_residuals(
            &derivative_matrix,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            false,
            false,
        );
    }

    Ok(compute_dfbeta_survreg_with_parameter(
        &time,
        &status,
        &covariates,
        &linear_pred,
        scale,
        &var_matrix,
        &distribution,
        distribution_parameter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_residuals() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let linear_pred = vec![0.0, 0.5, 1.0, 1.2, 1.5];
        let weibull = compute_response_residuals(&time, &linear_pred, "weibull");
        let rayleigh = compute_response_residuals(&time, &linear_pred, "rayleigh");
        let gaussian = compute_response_residuals(&time, &linear_pred, "gaussian");

        assert_eq!(weibull.len(), 5);
        assert!((weibull[0] - 0.0).abs() < 1e-10);
        assert!((weibull[1] - (2.0 - 0.5_f64.exp())).abs() < 1e-10);
        assert_eq!(rayleigh, weibull);
        assert_eq!(gaussian.len(), 5);
        assert!((gaussian[0] - 1.0).abs() < 1e-10);
        assert!((gaussian[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_residual_distribution_aliases_are_canonicalized() {
        let response = compute_response_residuals(&[2.0], &[0.0], "log-normal");
        assert!((response[0] - 1.0).abs() < 1e-12);

        let loggaussian = compute_response_residuals(&[2.0], &[0.0], "loggaussian");
        assert!((loggaussian[0] - 1.0).abs() < 1e-12);

        let matrix =
            compute_survreg_residual_matrix(&[1.0], None, &[1], &[0.0], 1.0, "extreme-value")
                .unwrap();
        assert_eq!(matrix.len(), 1);
        assert!(matrix[0].iter().all(|value| value.is_finite()));

        let rayleigh =
            compute_survreg_residual_matrix(&[1.0], None, &[1], &[0.0], 0.5, "rayleigh").unwrap();
        assert_eq!(rayleigh.len(), 1);
        assert!(rayleigh[0].iter().all(|value| value.is_finite()));
    }

    #[test]
    #[should_panic(expected = "distribution was validated")]
    fn test_residual_helpers_do_not_default_unknown_distribution() {
        let _ = compute_working_residuals(&[1.0], &[1], &[0.0], 1.0, "mystery");
    }

    #[test]
    fn test_deviance_residuals() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let linear_pred = vec![0.0, 0.5, 1.0, 1.2, 1.5];
        let scale = 1.0;

        let resid = compute_deviance_residuals_survreg(
            &time,
            None,
            &status,
            &linear_pred,
            scale,
            "weibull",
        )
        .unwrap();

        assert_eq!(resid.len(), 5);
        assert!(resid.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_survreg_residual_matrix_gaussian_exact_derivatives() {
        let matrix =
            compute_survreg_residual_matrix(&[1.5], None, &[1], &[1.0], 1.0, "gaussian").unwrap();
        let z = 0.5;
        let expected_loglik = gaussian_pdf(z).ln();

        assert_eq!(matrix.len(), 1);
        assert_eq!(matrix[0].len(), 6);
        assert!((matrix[0][0] - expected_loglik).abs() < 1e-10);
        assert!((matrix[0][1] - z).abs() < 1e-7);
        assert!((matrix[0][2] + 1.0).abs() < 1e-6);
        assert!((matrix[0][3] - (z * z - 1.0)).abs() < 1e-7);
        assert!((matrix[0][4] + 2.0 * z * z).abs() < 1e-6);
        assert!((matrix[0][5] + 2.0 * z).abs() < 1e-6);
    }

    #[test]
    fn test_working_residuals_from_derivative_matrix() {
        let matrix = vec![
            vec![0.0, 2.0, -4.0, 0.0, 0.0, 0.0],
            vec![0.0, -3.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let working = compute_working_residuals_from_derivative_matrix(&matrix).unwrap();

        assert!((working[0] - 0.5).abs() < 1e-12);
        assert!((working[1] + 0.5).abs() < 1e-12);
        assert_eq!(working[2], 0.0);
    }

    #[test]
    fn test_censored_response_and_deviance_residuals_use_saturated_model() {
        let time = vec![1.0, 1.0, 1.0, 1.0];
        let time2 = vec![1.0, 1.0, 2.0, 1.0];
        let status = vec![1, 2, 3, 0];
        let linear_pred = vec![0.0, 0.0, 0.0, 0.0];
        let matrix = compute_survreg_residual_matrix(
            &time,
            Some(&time2),
            &status,
            &linear_pred,
            1.0,
            "weibull",
        )
        .unwrap();

        let response = compute_response_residuals_censored(
            &time,
            Some(&time2),
            &status,
            &linear_pred,
            1.0,
            "weibull",
        )
        .unwrap();
        let deviance = compute_deviance_residuals_from_derivative_matrix(
            &matrix,
            &time,
            Some(&time2),
            &status,
            1.0,
            "weibull",
        )
        .unwrap();

        let interval_response = 1.0 / 2.0_f64.ln() - 1.0;
        assert!(response[0].abs() < 1e-12);
        assert!(response[1].abs() < 1e-12);
        assert!((response[2] - interval_response).abs() < 1e-12);
        assert!(response[3].abs() < 1e-12);
        assert!(deviance.iter().all(|value| value.is_finite()));
        assert!(deviance[0].abs() < 1e-8);
    }

    #[test]
    fn test_survreg_influence_residuals_use_scale_columns() {
        let deriv = vec![vec![0.0, 2.0, 3.0, 5.0, 7.0, 11.0]];
        let covariates = vec![vec![1.0, 4.0]];
        let scales = vec![1.5];
        let strata = vec![0];
        let var_matrix = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.3, 3.0],
        ];

        let ldcase = compute_survreg_influence_residuals(
            &deriv,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            SurvregResidType::Ldcase,
            true,
        )
        .unwrap();
        let ldresp = compute_survreg_influence_residuals(
            &deriv,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            SurvregResidType::Ldresp,
            true,
        )
        .unwrap();
        let ldshape = compute_survreg_influence_residuals(
            &deriv,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            SurvregResidType::Ldshape,
            true,
        )
        .unwrap();

        assert!((ldcase[0] - 238.2).abs() < 1e-10);
        assert!((ldresp[0] - 1709.1).abs() < 1e-10);
        assert!((ldshape[0] - 4452.4).abs() < 1e-10);
    }

    #[test]
    fn test_survreg_dfbeta_residuals_match_score_times_variance() {
        let deriv = vec![vec![0.0, 2.0, 3.0, 5.0, 7.0, 11.0]];
        let covariates = vec![vec![1.0, 4.0]];
        let scales = vec![1.5];
        let strata = vec![0];
        let var_matrix = vec![
            vec![1.0, 0.1, 0.2],
            vec![0.1, 2.0, 0.3],
            vec![0.2, 0.3, 3.0],
        ];

        let dfbeta = compute_survreg_dfbeta_residuals(
            &deriv,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            true,
            false,
        )
        .unwrap();
        let dfbetas = compute_survreg_dfbeta_residuals(
            &deriv,
            &covariates,
            &scales,
            &strata,
            &var_matrix,
            true,
            true,
        )
        .unwrap();

        assert_eq!(dfbeta.len(), 1);
        assert!((dfbeta[0][0] - 3.8).abs() < 1e-10);
        assert!((dfbeta[0][1] - 17.7).abs() < 1e-10);
        assert!((dfbeta[0][2] - 17.8).abs() < 1e-10);
        assert!((dfbetas[0][0] - 3.8).abs() < 1e-10);
        assert!((dfbetas[0][1] - (17.7 / 2.0_f64.sqrt())).abs() < 1e-10);
        assert!((dfbetas[0][2] - (17.8 / 3.0_f64.sqrt())).abs() < 1e-10);
    }
}
