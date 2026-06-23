use crate::internal::statistical::normal_inverse_cdf;
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SurvregPredictType {
    Response,
    Lp,
    Terms,
}

impl SurvregPredictType {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "response" => Some(SurvregPredictType::Response),
            "link" | "lp" | "linear" => Some(SurvregPredictType::Lp),
            "terms" => Some(SurvregPredictType::Terms),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvregPrediction {
    #[pyo3(get)]
    pub predictions: Vec<f64>,
    #[pyo3(get)]
    pub se: Option<Vec<f64>>,
    #[pyo3(get)]
    pub prediction_type: String,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl SurvregPrediction {
    fn __repr__(&self) -> String {
        format!(
            "SurvregPrediction(type='{}', n={}, has_se={})",
            self.prediction_type,
            self.n,
            self.se.is_some()
        )
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct SurvregQuantilePrediction {
    #[pyo3(get)]
    pub quantiles: Vec<f64>,
    #[pyo3(get)]
    pub predictions: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl SurvregQuantilePrediction {
    fn __repr__(&self) -> String {
        format!(
            "SurvregQuantilePrediction(n={}, n_quantiles={})",
            self.n,
            self.quantiles.len()
        )
    }
}

fn extreme_value_quantile(p: f64) -> f64 {
    (-(1.0 - p).ln()).ln()
}

fn logistic_quantile(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn distribution_key(distribution: &str) -> String {
    distribution.to_lowercase().replace('-', "_")
}

fn validate_distribution(distribution: &str) -> PyResult<()> {
    let key = distribution_key(distribution);
    if matches!(
        key.as_str(),
        "weibull"
            | "exponential"
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
    ) {
        return Ok(());
    }
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "distribution must be one of weibull, exponential, gaussian, logistic, lognormal, or loglogistic",
    ))
}

pub(crate) fn compute_linear_predictor(
    covariates: &[Vec<f64>],
    coefficients: &[f64],
    offset: Option<&[f64]>,
) -> Vec<f64> {
    let n = covariates.len();
    let nvar = coefficients.len();

    let mut lp = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0.0;
        for j in 0..nvar.min(covariates[i].len()) {
            val += covariates[i][j] * coefficients[j];
        }
        if let Some(off) = offset
            && i < off.len()
        {
            val += off[i];
        }
        lp.push(val);
    }
    lp
}

pub(crate) fn compute_response_prediction(linear_pred: &[f64], distribution: &str) -> Vec<f64> {
    match distribution_key(distribution).as_str() {
        "weibull" | "exponential" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian"
        | "loglogistic" | "log_logistic" => linear_pred.iter().map(|&lp| lp.exp()).collect(),
        "extreme" | "extreme_value" | "extremevalue" | "gaussian" | "normal" | "logistic" => {
            linear_pred.to_vec()
        }
        _ => linear_pred.to_vec(),
    }
}

pub(crate) fn compute_quantile_prediction(
    linear_pred: &[f64],
    scale: f64,
    quantiles: &[f64],
    distribution: &str,
) -> Vec<Vec<f64>> {
    let n = linear_pred.len();
    let nq = quantiles.len();

    let key = distribution_key(distribution);
    let quantile_fn: fn(f64) -> f64 = match key.as_str() {
        "weibull" | "exponential" | "extreme" | "extreme_value" | "extremevalue" => {
            extreme_value_quantile
        }
        "logistic" | "loglogistic" | "log_logistic" => logistic_quantile,
        "gaussian" | "lognormal" | "log_normal" | "loggaussian" | "log_gaussian" | "normal" => {
            normal_inverse_cdf
        }
        _ => extreme_value_quantile,
    };
    let uses_log_transform = matches!(
        key.as_str(),
        "weibull"
            | "exponential"
            | "lognormal"
            | "log_normal"
            | "loggaussian"
            | "log_gaussian"
            | "loglogistic"
            | "log_logistic"
    );

    let mut predictions = Vec::with_capacity(n);
    for lp in linear_pred.iter().take(n) {
        let mut row = Vec::with_capacity(nq);
        for &q in quantiles {
            let z = quantile_fn(q);
            let linear_quantile = lp + scale * z;
            row.push(if uses_log_transform {
                linear_quantile.exp()
            } else {
                linear_quantile
            });
        }
        predictions.push(row);
    }
    predictions
}

pub(crate) fn compute_se_linear_predictor(
    covariates: &[Vec<f64>],
    var_matrix: &[Vec<f64>],
) -> Vec<f64> {
    let nvar = var_matrix.len();

    let mut se = Vec::with_capacity(covariates.len());
    for cov in covariates {
        let mut var = 0.0;
        for j in 0..nvar.min(cov.len()) {
            for k in 0..nvar.min(cov.len()) {
                if j < var_matrix.len() && k < var_matrix[j].len() {
                    var += cov[j] * var_matrix[j][k] * cov[k];
                }
            }
        }
        se.push(var.sqrt());
    }
    se
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{} contains non-finite value at index {}",
                name, idx
            )));
        }
    }
    Ok(())
}

fn validate_rectangular_matrix(name: &str, matrix: &[Vec<f64>]) -> PyResult<usize> {
    let width = matrix.first().map_or(0, Vec::len);
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != width {
            return Err(value_error(format!(
                "{name} row {row_idx} has length {}, expected {width}",
                row.len()
            )));
        }
        validate_finite_values(name, row)?;
    }
    Ok(width)
}

fn validate_square_matrix(name: &str, matrix: &[Vec<f64>]) -> PyResult<usize> {
    let width = matrix.len();
    for (row_idx, row) in matrix.iter().enumerate() {
        if row.len() != width {
            return Err(value_error(format!(
                "{name} row {row_idx} has length {}, expected {width}",
                row.len()
            )));
        }
        validate_finite_values(name, row)?;
    }
    Ok(width)
}

fn validate_scale(scale: f64) -> PyResult<()> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "scale must be a finite positive value",
        ));
    }
    Ok(())
}

#[pyfunction]
pub fn survreg_quantile_prediction_se_matrix(
    rows: Vec<Vec<f64>>,
    scales: Vec<f64>,
    strata: Vec<usize>,
    variance: Vec<Vec<f64>>,
    quantile_scores: Vec<f64>,
    predictions: Vec<Vec<f64>>,
    transform_se: bool,
) -> PyResult<Vec<Vec<f64>>> {
    let nvar = validate_rectangular_matrix("rows", &rows)?;
    let n = rows.len();
    if strata.len() != n {
        return Err(value_error("strata length must match rows length"));
    }
    if predictions.len() != n {
        return Err(value_error("predictions length must match rows length"));
    }
    if scales.is_empty() {
        return Err(value_error("scales must not be empty"));
    }
    validate_finite_values("scales", &scales)?;
    validate_finite_values("quantile_scores", &quantile_scores)?;
    for (row_idx, row) in predictions.iter().enumerate() {
        if row.len() != quantile_scores.len() {
            return Err(value_error(format!(
                "predictions row {row_idx} has length {}, expected {}",
                row.len(),
                quantile_scores.len()
            )));
        }
        validate_finite_values("predictions", row)?;
    }

    let width = validate_square_matrix("variance", &variance)?;
    let full_width = nvar + scales.len();
    if width != nvar && width != full_width {
        return Err(value_error(
            "variance width must match row width or row width plus scale count",
        ));
    }
    for (row_idx, &stratum) in strata.iter().enumerate() {
        if stratum >= scales.len() {
            return Err(value_error(format!(
                "strata contains value {stratum} at row {row_idx}, expected < {}",
                scales.len()
            )));
        }
    }

    let has_scale_variance = width == full_width;
    let mut result = Vec::with_capacity(n);
    for (row_idx, row) in rows.iter().enumerate() {
        let mut result_row = Vec::with_capacity(quantile_scores.len());
        for (score_idx, &score) in quantile_scores.iter().enumerate() {
            let mut variance_value = 0.0;
            for left in 0..nvar {
                for right in 0..nvar {
                    variance_value += row[left] * variance[left][right] * row[right];
                }
            }
            if has_scale_variance {
                let stratum = strata[row_idx];
                let scale_col = nvar + stratum;
                let scale_value = score * scales[stratum];
                for col in 0..nvar {
                    variance_value += row[col] * variance[col][scale_col] * scale_value;
                    variance_value += scale_value * variance[scale_col][col] * row[col];
                }
                variance_value += scale_value * variance[scale_col][scale_col] * scale_value;
            }
            let mut se = variance_value.max(0.0).sqrt();
            if transform_se {
                se *= predictions[row_idx][score_idx].abs();
            }
            result_row.push(se);
        }
        result.push(result_row);
    }
    Ok(result)
}

fn validate_covariates(covariates: &[Vec<f64>], nvar: usize) -> PyResult<()> {
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != nvar {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "covariates row {} has {} columns but coefficients has {}",
                row_idx,
                row.len(),
                nvar
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "covariates[{}][{}] contains non-finite value",
                    row_idx, col_idx
                )));
            }
        }
    }
    Ok(())
}

fn validate_offset(offset: Option<&[f64]>, n: usize) -> PyResult<()> {
    if let Some(values) = offset {
        if values.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "offset has {} values but covariates has {} rows",
                values.len(),
                n
            )));
        }
        validate_finite_values("offset", values)?;
    }
    Ok(())
}

fn validate_var_matrix(var_matrix: &[Vec<f64>], nvar: usize) -> PyResult<()> {
    if var_matrix.len() < nvar {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "var_matrix must have at least {} rows",
            nvar
        )));
    }
    for (row_idx, row) in var_matrix.iter().enumerate() {
        if row_idx < nvar && row.len() < nvar {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "var_matrix row {} has {} columns but expected at least {}",
                row_idx,
                row.len(),
                nvar
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "var_matrix[{}][{}] contains non-finite value",
                    row_idx, col_idx
                )));
            }
        }
    }
    Ok(())
}

fn validate_quantiles(quantiles: &[f64]) -> PyResult<()> {
    for &q in quantiles {
        if !q.is_finite() || q <= 0.0 || q >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Quantiles must be between 0 and 1 (exclusive)",
            ));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (covariates, coefficients, scale, distribution, predict_type="response".to_string(), offset=None, var_matrix=None, se_fit=false))]
#[allow(clippy::too_many_arguments)]
pub fn predict_survreg(
    covariates: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    scale: f64,
    distribution: String,
    predict_type: String,
    offset: Option<Vec<f64>>,
    var_matrix: Option<Vec<Vec<f64>>>,
    se_fit: bool,
) -> PyResult<SurvregPrediction> {
    let n = covariates.len();
    let nvar = coefficients.len();
    validate_scale(scale)?;
    validate_distribution(&distribution)?;
    validate_finite_values("coefficients", &coefficients)?;
    validate_covariates(&covariates, nvar)?;
    validate_offset(offset.as_deref(), n)?;
    if let Some(values) = var_matrix.as_ref() {
        validate_var_matrix(values, nvar)?;
    }

    let pred_type = SurvregPredictType::from_str(&predict_type).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown prediction type: {}. Valid types: response, lp/linear, quantile",
            predict_type
        ))
    })?;

    let linear_pred = compute_linear_predictor(&covariates, &coefficients, offset.as_deref());

    let predictions = match pred_type {
        SurvregPredictType::Lp | SurvregPredictType::Terms => linear_pred.clone(),
        SurvregPredictType::Response => compute_response_prediction(&linear_pred, &distribution),
    };

    let se = if se_fit {
        var_matrix
            .as_ref()
            .map(|vm| compute_se_linear_predictor(&covariates, vm))
    } else {
        None
    };

    Ok(SurvregPrediction {
        predictions,
        se,
        prediction_type: predict_type,
        n,
    })
}

#[pyfunction]
#[pyo3(signature = (covariates, coefficients, scale, distribution, quantiles, offset=None))]
pub fn predict_survreg_quantile(
    covariates: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    scale: f64,
    distribution: String,
    quantiles: Vec<f64>,
    offset: Option<Vec<f64>>,
) -> PyResult<SurvregQuantilePrediction> {
    let n = covariates.len();
    let nvar = coefficients.len();
    validate_scale(scale)?;
    validate_distribution(&distribution)?;
    validate_finite_values("coefficients", &coefficients)?;
    validate_covariates(&covariates, nvar)?;
    validate_offset(offset.as_deref(), n)?;
    validate_quantiles(&quantiles)?;

    let linear_pred = compute_linear_predictor(&covariates, &coefficients, offset.as_deref());

    let predictions = compute_quantile_prediction(&linear_pred, scale, &quantiles, &distribution);

    Ok(SurvregQuantilePrediction {
        quantiles,
        predictions,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_predictor() {
        let covariates = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0]];
        let coefficients = vec![0.5, 0.3];

        let lp = compute_linear_predictor(&covariates, &coefficients, None);

        assert_eq!(lp.len(), 3);
        assert!((lp[0] - 1.1).abs() < 1e-10);
        assert!((lp[1] - 1.9).abs() < 1e-10);
        assert!((lp[2] - 2.7).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_prediction() {
        let linear_pred = vec![1.0, 2.0, 3.0];
        let scale = 1.0;
        let quantiles = vec![0.5];

        let pred = compute_quantile_prediction(&linear_pred, scale, &quantiles, "weibull");

        assert_eq!(pred.len(), 3);
        assert_eq!(pred[0].len(), 1);
    }

    #[test]
    fn test_quantile_prediction_respects_distribution_transform() {
        let linear_pred = vec![1.0];
        let scale = 0.5;
        let quantiles = vec![0.5];

        let gaussian = compute_quantile_prediction(&linear_pred, scale, &quantiles, "gaussian");
        let lognormal = compute_quantile_prediction(&linear_pred, scale, &quantiles, "lognormal");

        assert!((gaussian[0][0] - (1.0 + scale * normal_inverse_cdf(0.5))).abs() < 1e-10);
        assert!((lognormal[0][0] - gaussian[0][0].exp()).abs() < 1e-10);
    }

    #[test]
    fn test_survreg_quantile_prediction_se_matrix_uses_scale_columns() {
        let se = survreg_quantile_prediction_se_matrix(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![2.0, 3.0],
            vec![0, 1],
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
            vec![0.5, 1.0],
            vec![vec![10.0, 20.0], vec![30.0, 40.0]],
            false,
        )
        .expect("quantile SE matrix should compute");

        assert!((se[0][0] - 6.0_f64.sqrt()).abs() < 1e-12);
        assert!((se[0][1] - 3.0).abs() < 1e-12);
        assert!((se[1][0] - 27.25_f64.sqrt()).abs() < 1e-12);
        assert!((se[1][1] - 34.0_f64.sqrt()).abs() < 1e-12);

        let transformed = survreg_quantile_prediction_se_matrix(
            vec![vec![1.0, 2.0]],
            vec![2.0],
            vec![0],
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            vec![0.5],
            vec![vec![10.0]],
            true,
        )
        .expect("transformed quantile SE matrix should compute");
        assert!((transformed[0][0] - 10.0 * 6.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_response_prediction_respects_distribution_transform() {
        let linear_pred = vec![0.0, 1.0];

        let weibull = compute_response_prediction(&linear_pred, "weibull");
        assert!((weibull[0] - 1.0).abs() < 1e-10);
        assert!((weibull[1] - std::f64::consts::E).abs() < 1e-10);

        let exponential = compute_response_prediction(&linear_pred, "exponential");
        assert!((exponential[1] - std::f64::consts::E).abs() < 1e-10);

        assert_eq!(
            compute_response_prediction(&linear_pred, "gaussian"),
            linear_pred
        );
        assert_eq!(
            compute_response_prediction(&linear_pred, "logistic"),
            linear_pred
        );
        assert_eq!(
            compute_response_prediction(&linear_pred, "extreme_value"),
            linear_pred
        );
    }

    #[test]
    fn test_extreme_value_quantile() {
        let q = extreme_value_quantile(0.5);
        assert!((q - (-0.3665129)).abs() < 1e-5);
    }
}
