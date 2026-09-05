use crate::internal::statistical::{ln_gamma, student_t_cdf, student_t_pdf};
use pyo3::prelude::*;

const LOG_PROBABILITY_FLOOR: f64 = -690.0;
const PROBABILITY_FLOOR: f64 = 1e-300;

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

#[cfg(test)]
fn gaussian_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / std::f64::consts::TAU.sqrt()
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

fn standardized_interval_width(lower: f64, upper: f64, scale: f64, log_transform: bool) -> f64 {
    let width = if log_transform {
        let relative_width = (upper - lower) / lower;
        if relative_width.is_finite() {
            relative_width.ln_1p()
        } else {
            upper.ln() - lower.ln()
        }
    } else {
        upper - lower
    };
    width / scale
}

fn transformed_interval_width(
    time: &[f64],
    time2: Option<&[f64]>,
    idx: usize,
    scale: f64,
    distribution: &str,
) -> f64 {
    let upper = time2.expect("validated time2 length")[idx];
    standardized_interval_width(
        time[idx],
        upper,
        scale,
        response_uses_log_transform(distribution),
    )
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
            let probability = libm::erf(width / (2.0 * std::f64::consts::SQRT_2));
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
    let key = validated_distribution_key(distribution);
    let parameter = validated_distribution_parameter_for_key(&key, distribution_parameter)
        .expect("distribution parameter was validated");
    let family = ResidualDistribution::from_key(&key, parameter);
    time.iter()
        .zip(status)
        .zip(linear_pred)
        .map(|((&time, &event), &eta)| {
            let z = (response_time_value_for_key(time, &key) - eta) / scale;
            let row = family.single(z, scale, event);
            -row[1] / row[2]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
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

    let key = validated_distribution_key(distribution);
    let parameter = validated_distribution_parameter_for_key(&key, distribution_parameter)
        .expect("distribution parameter was validated");
    let family = ResidualDistribution::from_key(&key, parameter);

    let mut dfbeta = Vec::with_capacity(n);

    for i in 0..n {
        let z = (response_time_value_for_key(time[i], &key) - linear_pred[i]) / scale;
        let score = family.single(z, scale, status[i])[1];
        let mut row = Vec::with_capacity(nvar);
        for j in 0..nvar {
            let mut val = 0.0;
            for k in 0..nvar {
                if k < var_matrix.len() && j < var_matrix[k].len() {
                    val += var_matrix[k][j] * covariates[i][k] * score;
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
    let key = validated_distribution_key(distribution);
    let parameter = validated_distribution_parameter_for_key(&key, distribution_parameter)?;
    let family = ResidualDistribution::from_key(&key, parameter);
    let log_transform = response_uses_log_transform_key(&key);
    let transform = |value: f64| if log_transform { value.ln() } else { value };
    Ok(time
        .iter()
        .zip(status)
        .zip(linear_pred)
        .enumerate()
        .map(|(idx, ((&time, &event), &eta))| {
            let z = (transform(time) - eta) / scale;
            if event == 3 {
                let upper = time2.expect("validated time2 length")[idx];
                let width = standardized_interval_width(time, upper, scale, log_transform);
                family.interval(z, width, scale)[0]
            } else {
                family.single(z, scale, event)[0]
            }
        })
        .collect())
}

/// Distribution calculations shared by all residual types. The two tails are
/// evaluated directly, so a small survival probability is never `1 - cdf`.
#[derive(Clone, Copy)]
enum ResidualDistribution {
    Extreme,
    Logistic,
    Gaussian,
    StudentT { df: f64, log_normalizer: f64 },
}

#[derive(Clone, Copy)]
struct ResidualDensity {
    log_density: f64,
    score: f64,
    curvature: f64,
}

fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// The normal hazard and its small difference from z. Retaining the continued
/// fraction's correction avoids both subtracting large log probabilities and
/// subtracting almost equal hazard and z values in the tail curvature.
fn normal_tail_hazard(z: f64) -> (f64, f64) {
    let mut denominator = z;
    for numerator in (2..=32).rev() {
        denominator = z + f64::from(numerator) / denominator;
    }
    let correction = denominator.recip();
    (z + correction, correction)
}

/// Log of the upper normal tail for a nonnegative argument. The continued
/// fraction also covers tails smaller than f64 probabilities.
fn normal_log_upper_tail(z: f64) -> f64 {
    if z < 20.0 {
        return (0.5 * libm::erfc(z / std::f64::consts::SQRT_2)).ln();
    }
    let (hazard, _) = normal_tail_hazard(z);
    -0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - hazard.ln()
}

/// R's interval diagnostic columns use a different scale convention from a
/// likelihood Hessian. Keep this conversion shared by integration and tail limits.
fn interval_scale_convention(mut row: [f64; 6], scale: f64) -> [f64; 6] {
    let ds = row[3];
    row[3] = -ds;
    row[4] += 2.0 * ds;
    row[5] = scale * row[5] + row[1] * (scale * (1.0 + ds) - 1.0 + ds);
    row
}

impl ResidualDistribution {
    fn from_key(key: &str, parameter: Option<f64>) -> Self {
        match key {
            "weibull" | "exponential" | "rayleigh" | "extreme" | "extreme_value"
            | "extremevalue" => Self::Extreme,
            "logistic" | "loglogistic" | "log_logistic" => Self::Logistic,
            "gaussian" | "normal" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" => {
                Self::Gaussian
            }
            "t" | "student" | "student_t" | "studentt" => {
                let df = parameter.expect("Student-t df was validated");
                Self::StudentT {
                    df,
                    log_normalizer: ln_gamma((df + 1.0) / 2.0)
                        - ln_gamma(df / 2.0)
                        - 0.5 * (df * std::f64::consts::PI).ln(),
                }
            }
            _ => unreachable!("distribution was validated"),
        }
    }

    fn density(self, z: f64) -> ResidualDensity {
        let (log_density, score, curvature) = match self {
            Self::Extreme => {
                let exponential = z.exp();
                (z - exponential, 1.0 - exponential, -exponential)
            }
            Self::Logistic => {
                let log_density = -z.abs() - 2.0 * (-z.abs()).exp().ln_1p();
                (log_density, -(z / 2.0).tanh(), -2.0 * log_density.exp())
            }
            Self::Gaussian => (-0.5 * z * z - 0.5 * std::f64::consts::TAU.ln(), -z, -1.0),
            Self::StudentT { df, log_normalizer } => {
                let denominator = df + z * z;
                (
                    log_normalizer - 0.5 * (df + 1.0) * (z * z / df).ln_1p(),
                    -(df + 1.0) * z / denominator,
                    (df + 1.0) * (z * z - df) / denominator / denominator,
                )
            }
        };
        ResidualDensity {
            log_density,
            score,
            curvature,
        }
    }

    fn log_tails(self, z: f64) -> (f64, f64) {
        match self {
            Self::Extreme => {
                let exponential = z.exp();
                // exp(z) may underflow even while log(F(z)) is representable.
                let log_cdf = if exponential == 0.0 {
                    z
                } else {
                    (-(-exponential).exp_m1()).ln()
                };
                (log_cdf, -exponential)
            }
            Self::Logistic => (-softplus(-z), -softplus(z)),
            Self::Gaussian => {
                let log_small = normal_log_upper_tail(z.abs());
                let log_large = (-log_small.exp()).ln_1p();
                if z < 0.0 {
                    (log_small, log_large)
                } else {
                    (log_large, log_small)
                }
            }
            Self::StudentT { df, .. } => {
                let log_small = student_t_cdf(-z.abs(), df).ln();
                let log_large = (-log_small.exp()).ln_1p();
                if z < 0.0 {
                    (log_small, log_large)
                } else {
                    (log_large, log_small)
                }
            }
        }
    }

    fn single(self, z: f64, scale: f64, status: i32) -> [f64; 6] {
        let density = self.density(z);
        let (g, score, curvature) = if status == 1 {
            (
                density.log_density - scale.ln(),
                density.score,
                density.curvature,
            )
        } else {
            let (log_cdf, log_survival) = self.log_tails(z);
            match self {
                Self::Extreme if status == 0 => (-z.exp(), -z.exp(), -z.exp()),
                Self::Extreme => {
                    let exponential = z.exp();
                    let ratio = if exponential < 1e-4 {
                        1.0 - exponential / 2.0 + exponential * exponential / 12.0
                    } else {
                        (density.log_density - log_cdf).exp()
                    };
                    let curvature = if exponential < 1e-4 {
                        -exponential / 2.0 + exponential * exponential / 6.0
                    } else {
                        ratio * (1.0 - exponential - ratio)
                    };
                    (log_cdf, ratio, curvature)
                }
                Self::Logistic => {
                    let score = if status == 0 {
                        -log_cdf.exp()
                    } else {
                        log_survival.exp()
                    };
                    (
                        if status == 0 { log_survival } else { log_cdf },
                        score,
                        -density.log_density.exp(),
                    )
                }
                Self::Gaussian if (status == 0 && z >= 20.0) || (status == 2 && z <= -20.0) => {
                    let (hazard, correction) = normal_tail_hazard(z.abs());
                    let g = if status == 0 { log_survival } else { log_cdf };
                    let score = if status == 0 { -hazard } else { hazard };
                    (g, score, -hazard * correction)
                }
                _ => {
                    let g = if status == 0 { log_survival } else { log_cdf };
                    let ratio = (density.log_density - g).exp();
                    let score = if status == 0 { -ratio } else { ratio };
                    (g, score, score * (density.score - score))
                }
            }
        };
        [
            g,
            -score / scale,
            curvature / scale / scale,
            -z * score - f64::from(status == 1),
            z * (score + z * curvature),
            (score + z * curvature) / scale,
        ]
    }

    fn interval(self, lower: f64, width: f64, scale: f64) -> [f64; 6] {
        let upper = lower + width;
        let lower_density = self.density(lower);
        let upper_density = self.density(upper);
        // Integrating the conditional scores avoids subtracting almost equal
        // endpoint densities. Check log-density variation as well as width:
        // the extreme-value density changes at exp(z), much faster than z.
        let density_variation = width
            * (lower_density.score.abs().max(upper_density.score.abs())
                + lower_density
                    .curvature
                    .abs()
                    .max(upper_density.curvature.abs())
                    .sqrt());
        if density_variation < 1e-3 {
            return self.narrow_interval(lower, width, scale);
        }
        // P = A - B, using upper tails on the right and lower tails on
        // the left. Combine their log-likelihood derivatives directly: forming
        // density/probability ratios loses curvature in distant Gaussian tails.
        let (larger, smaller) = if lower > 0.0 {
            (self.single(lower, scale, 0), self.single(upper, scale, 0))
        } else {
            (self.single(upper, scale, 2), self.single(lower, scale, 2))
        };
        let log_ratio = if matches!(self, Self::Gaussian) && (lower >= 20.0 || upper <= -20.0) {
            let (near, far) = if lower > 0.0 {
                (lower, upper)
            } else {
                (-upper, -lower)
            };
            let (near_hazard, near_correction) = normal_tail_hazard(near);
            let (_, far_correction) = normal_tail_hazard(far);
            let hazard_difference = width + far_correction - near_correction;
            -0.5 * width * (near + far) - (hazard_difference / near_hazard).ln_1p()
        } else {
            smaller[0] - larger[0]
        };
        let ratio = (-log_ratio).exp_m1().recip();
        if ratio == 0.0 {
            return interval_scale_convention(larger, scale);
        }
        let location_difference = larger[1] - smaller[1];
        let scale_difference = larger[3] - smaller[3];
        let covariance_weight = ratio * (1.0 + ratio);
        let g = larger[0] + (-log_ratio.exp_m1()).ln();
        let dg = larger[1] + ratio * location_difference;
        let ddg = larger[2] + ratio * (larger[2] - smaller[2])
            - covariance_weight * location_difference * location_difference;
        let ds = larger[3] + ratio * scale_difference;
        let dds = larger[4] + ratio * (larger[4] - smaller[4])
            - covariance_weight * scale_difference * scale_difference;
        let dsg = larger[5] + ratio * (larger[5] - smaller[5])
            - covariance_weight * location_difference * scale_difference;
        interval_scale_convention([g, dg, ddg, ds, dds, dsg], scale)
    }

    fn narrow_interval(self, lower: f64, width: f64, scale: f64) -> [f64; 6] {
        const NODES: [f64; 4] = [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ];
        const WEIGHTS: [f64; 4] = [
            0.34785484513745385,
            0.6521451548625461,
            0.6521451548625461,
            0.34785484513745385,
        ];
        let half_width = width / 2.0;
        let center = lower + half_width;
        let rows = NODES.map(|node| self.single(center + node * half_width, scale, 1));
        let maximum = rows
            .iter()
            .map(|row| row[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: [f64; 4] = std::array::from_fn(|i| WEIGHTS[i] * (rows[i][0] - maximum).exp());
        let total = weights.iter().sum::<f64>();
        let dg = (0..4).map(|i| weights[i] * rows[i][1]).sum::<f64>() / total;
        let ds_true = (0..4).map(|i| weights[i] * rows[i][3]).sum::<f64>() / total;
        let mut ddg = 0.0;
        let mut dds_true = 0.0;
        let mut dsg_true = 0.0;
        for i in 0..4 {
            let location_delta = rows[i][1] - dg;
            let scale_delta = rows[i][3] - ds_true;
            ddg += weights[i] * (rows[i][2] + location_delta * location_delta);
            dds_true += weights[i] * (rows[i][4] + scale_delta * scale_delta);
            dsg_true += weights[i] * (rows[i][5] + location_delta * scale_delta);
        }
        let g = half_width.ln() + scale.ln() + maximum + total.ln();
        interval_scale_convention(
            [
                g,
                dg,
                ddg / total,
                ds_true,
                dds_true / total,
                dsg_true / total,
            ],
            scale,
        )
    }
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
    let family = ResidualDistribution::from_key(&key, distribution_parameter);
    let log_transform = response_uses_log_transform_key(&key);
    let transform = |value: f64| if log_transform { value.ln() } else { value };
    let mut matrix = Vec::with_capacity(time.len());

    for i in 0..time.len() {
        let z = (transform(time[i]) - linear_pred[i]) / scale;
        let row = if status[i] == 3 {
            let upper = time2.expect("validated time2 length")[i];
            let width = standardized_interval_width(time[i], upper, scale, log_transform);
            family.interval(z, width, scale)
        } else {
            family.single(z, scale, status[i])
        };
        matrix.push(row.to_vec());
    }

    Ok(matrix)
}

pub(crate) fn compute_working_residuals_from_derivative_matrix(
    derivative_matrix: &[Vec<f64>],
) -> PyResult<Vec<f64>> {
    validate_derivative_matrix(derivative_matrix)?;
    Ok(derivative_matrix
        .iter()
        .map(|row| -row[1] / row[2])
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
#[allow(clippy::too_many_arguments)]
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

    fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gaussian_exact_derivatives_preserve_extreme_scales() {
        for scale in [1e-12, 1.0, 1e12] {
            let matrix = compute_survreg_residual_matrix(
                &[2.5 * scale],
                None,
                &[1],
                &[0.5 * scale],
                scale,
                "gaussian",
            )
            .unwrap();
            let expected = [
                -2.0 - 0.5 * std::f64::consts::TAU.ln() - scale.ln(),
                2.0 / scale,
                -1.0 / scale / scale,
                3.0,
                -8.0,
                -4.0 / scale,
            ];
            for (actual, expected) in matrix[0].iter().zip(expected) {
                assert_relative(*actual, expected, 2e-14);
            }
            let working = compute_working_residuals_from_derivative_matrix(&matrix).unwrap();
            assert_relative(working[0], 2.0 * scale, 2e-14);
        }
    }

    #[test]
    fn censored_gaussian_tail_derivatives_match_r_reference() {
        let matrix = compute_survreg_residual_matrix(
            &[9.0, 1.0],
            None,
            &[0, 2],
            &[0.0, 10.0],
            1.0,
            "gaussian",
        )
        .unwrap();
        let expected = [
            -43.62814911333211,
            9.108523105002869,
            -0.9884852093452707,
            81.97670794502582,
            -162.0440099019934,
            -18.00488998911032,
        ];
        for (i, &value) in expected.iter().enumerate() {
            assert_relative(matrix[0][i], value, 2e-12);
            let reflected = if i == 1 || i == 5 { -value } else { value };
            assert_relative(matrix[1][i], reflected, 2e-12);
        }
    }

    #[test]
    fn censored_logistic_tail_retains_small_curvature() {
        let matrix =
            compute_survreg_residual_matrix(&[40.0], None, &[0], &[0.0], 1.0, "logistic").unwrap();
        assert_eq!(matrix[0][0], -40.0);
        assert_eq!(matrix[0][1], 1.0);
        assert_relative(matrix[0][2] / (-40.0_f64).exp(), -1.0, 1e-14);
        assert_eq!(matrix[0][3], 40.0);
        let working = compute_working_residuals_from_derivative_matrix(&matrix).unwrap();
        assert_relative(working[0], 40.0_f64.exp(), 1e-14);
    }

    #[test]
    fn narrow_gaussian_interval_has_density_limit_and_r_scale_columns() {
        let time = 1.5;
        let upper = time + 1e-9;
        let eta = 0.2;
        let scale = 1.3;
        let z = ((time + upper) / 2.0 - eta) / scale;
        let matrix = compute_survreg_residual_matrix(
            &[time],
            Some(&[upper]),
            &[3],
            &[eta],
            scale,
            "gaussian",
        )
        .unwrap();
        let expected = [
            gaussian_pdf(z).ln() + ((upper - time) / scale).ln(),
            z / scale,
            -1.0 / scale / scale,
            1.0 - z * z,
            -2.0,
            z * z * z - 2.0 * z + (z * z * z - 2.0 * z) / scale,
        ];
        for (actual, expected) in matrix[0].iter().zip(expected) {
            assert_relative(*actual, expected, 2e-14);
        }
    }

    #[test]
    fn large_gaussian_tail_keeps_negative_unit_curvature() {
        for z in [1e4, 1e5, 1e8] {
            let matrix =
                compute_survreg_residual_matrix(&[z], None, &[0], &[0.0], 1.0, "gaussian").unwrap();
            // The Mills expansion gives h(z)=z+1/z+O(z^-3) and
            // d² log(S(z))/dz²=-1+1/z²+O(z^-4).
            assert_relative(matrix[0][1], z + 1.0 / z, 1e-14);
            assert_relative(matrix[0][2], -1.0 + 1.0 / (z * z), 1e-14);
            let working = compute_working_residuals_from_derivative_matrix(&matrix).unwrap();
            assert!(working[0] > 0.0);
        }
    }

    #[test]
    fn narrow_extreme_interval_checks_density_variation() {
        for lower in [15.0_f64, 20.0] {
            let upper = lower + 1e-5;
            let matrix = compute_survreg_residual_matrix(
                &[lower],
                Some(&[upper]),
                &[3],
                &[0.0],
                1.0,
                "extreme",
            )
            .unwrap();
            let mass_delta = lower.exp() * (upper - lower).exp_m1();
            let expected = -lower.exp() + (-(-mass_delta).exp_m1()).ln();
            assert_relative(matrix[0][0], expected, 2e-15);
        }
    }

    #[test]
    fn narrow_log_interval_preserves_adjacent_response_bounds() {
        let lower = 1e6_f64;
        let upper = f64::from_bits(lower.to_bits() + 1);
        let matrix = compute_survreg_residual_matrix(
            &[lower],
            Some(&[upper]),
            &[3],
            &[0.0],
            1.0,
            "lognormal",
        )
        .unwrap();
        let z = lower.ln();
        let width = ((upper - lower) / lower).ln_1p();
        let expected = gaussian_pdf(z).ln() + width.ln();
        assert_relative(matrix[0][0], expected, 2e-15);
    }

    #[test]
    fn distant_gaussian_interval_matches_dominant_censored_bound() {
        for (lower, eta, status) in [(1e5, 0.0, 0), (1.0, 100002.0, 2)] {
            let upper = lower + 1.0;
            let interval = compute_survreg_residual_matrix(
                &[lower],
                Some(&[upper]),
                &[3],
                &[eta],
                1.0,
                "gaussian",
            )
            .unwrap();
            let bound = if status == 0 { lower } else { upper };
            let censored =
                compute_survreg_residual_matrix(&[bound], None, &[status], &[eta], 1.0, "gaussian")
                    .unwrap();
            for i in 0..3 {
                assert_eq!(interval[0][i], censored[0][i]);
            }
        }
    }

    #[test]
    fn distant_gaussian_interval_retains_curvature_with_two_relevant_bounds() {
        let lower = 1e5_f64;
        let upper = lower + 1e-5;
        let width = upper - lower;
        let matrix = compute_survreg_residual_matrix(
            &[lower],
            Some(&[upper]),
            &[3],
            &[0.0],
            1.0,
            "gaussian",
        )
        .unwrap();
        // For a normal interval, the location score is its conditional mean
        // and curvature is Var(Z | interval)-1. Both have independent bounds.
        assert!(matrix[0][1] >= lower && matrix[0][1] <= upper);
        assert!(matrix[0][2] >= -1.0 - 1e-14);
        assert!(matrix[0][2] <= -1.0 + width * width / 4.0 + 1e-14);
    }

    #[test]
    fn flat_student_interval_preserves_density_limit() {
        let lower = 1e10;
        let upper = lower + 1.0;
        let matrix = compute_survreg_residual_matrix_with_parameter(
            &[lower],
            Some(&[upper]),
            &[3],
            &[0.0],
            1.0,
            "t",
            Some(4.0),
        )
        .unwrap();
        // For df=4 the normalizer is 3/8; the relative density change is
        // 5e-10 and the midpoint integration error is smaller than 1e-19.
        let midpoint = (lower + upper) / 2.0;
        let expected = (3.0_f64 / 8.0).ln() - 2.5 * (midpoint * midpoint / 4.0).ln_1p();
        assert_relative(matrix[0][0], expected, 2e-15);
    }

    #[test]
    fn narrow_log_interval_deviance_uses_the_same_preserved_width() {
        let lower = 1e6_f64;
        let upper = f64::from_bits(lower.to_bits() + 1);
        let residuals = compute_deviance_residuals_survreg(
            &[lower],
            Some(&[upper]),
            &[3],
            &[0.0],
            1.0,
            "lognormal",
        )
        .unwrap();
        assert_relative(residuals[0], lower.ln(), 2e-14);
    }

    #[test]
    fn location_dfbeta_uses_score_instead_of_working_residual() {
        let dfbeta = compute_dfbeta_survreg_with_parameter(
            &[5.0],
            &[1],
            &[vec![1.0]],
            &[1.0],
            2.0,
            &[vec![3.0]],
            "gaussian",
            None,
        );
        assert_eq!(dfbeta, vec![vec![3.0]]);
        assert_eq!(
            compute_working_residuals(&[5.0], &[1], &[1.0], 2.0, "gaussian"),
            vec![4.0]
        );
    }

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
        assert_eq!(working[2], f64::NEG_INFINITY);
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
