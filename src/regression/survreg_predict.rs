use crate::internal::statistical::normal_inverse_cdf;
use pyo3::prelude::*;

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
    match distribution.to_lowercase().as_str() {
        "weibull" | "extreme" | "extreme_value" | "extremevalue" | "lognormal" | "gaussian"
        | "loglogistic" | "logistic" => linear_pred.iter().map(|&lp| lp.exp()).collect(),
        _ => linear_pred.iter().map(|&lp| lp.exp()).collect(),
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

    let quantile_fn: fn(f64) -> f64 = match distribution.to_lowercase().as_str() {
        "weibull" | "extreme" | "extreme_value" | "extremevalue" => extreme_value_quantile,
        "logistic" | "loglogistic" => logistic_quantile,
        "gaussian" | "lognormal" | "normal" => normal_inverse_cdf,
        _ => extreme_value_quantile,
    };

    let mut predictions = Vec::with_capacity(n);
    for lp in linear_pred.iter().take(n) {
        let mut row = Vec::with_capacity(nq);
        for &q in quantiles {
            let z = quantile_fn(q);
            let log_time = lp + scale * z;
            row.push(log_time.exp());
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
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "scale must be a finite positive value",
        ));
    }
    let n = covariates.len();

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

    for &q in &quantiles {
        if q <= 0.0 || q >= 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Quantiles must be between 0 and 1 (exclusive)",
            ));
        }
    }

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
    fn test_extreme_value_quantile() {
        let q = extreme_value_quantile(0.5);
        assert!((q - (-0.3665129)).abs() < 1e-5);
    }
}
