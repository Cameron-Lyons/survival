use pyo3::prelude::*;

#[allow(dead_code)]
type DistFn = fn(f64) -> f64;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SurvregResidType {
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
    pub fn from_str(s: &str) -> Option<Self> {
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
#[pyclass]
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
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn gaussian_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[allow(dead_code)]
pub fn compute_standardized_residuals(
    time: &[f64],
    _status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    _distribution: &str,
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let y = time[i].ln();
        let z = (y - linear_pred[i]) / scale;
        residuals.push(z);
    }

    residuals
}

pub fn compute_response_residuals(time: &[f64], linear_pred: &[f64]) -> Vec<f64> {
    time.iter()
        .zip(linear_pred.iter())
        .map(|(&t, &lp)| t.ln() - lp)
        .collect()
}

pub fn compute_deviance_residuals_survreg(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    let (cdf_fn, pdf_fn): (DistFn, DistFn) = match distribution.to_lowercase().as_str() {
        "weibull" | "extreme" | "extreme_value" | "extremevalue" => {
            (extreme_value_cdf, extreme_value_pdf)
        }
        "logistic" | "loglogistic" => (logistic_cdf, logistic_pdf),
        "gaussian" | "lognormal" | "normal" => (gaussian_cdf, gaussian_pdf),
        _ => (extreme_value_cdf, extreme_value_pdf),
    };

    for i in 0..n {
        let y = time[i].ln();
        let z = (y - linear_pred[i]) / scale;

        let surv = 1.0 - cdf_fn(z);
        let dens = pdf_fn(z) / scale;

        let dev = if status[i] == 1 {
            let log_dens = if dens > 1e-300 { dens.ln() } else { -690.0 };
            -2.0 * log_dens
        } else {
            let log_surv = if surv > 1e-300 { surv.ln() } else { -690.0 };
            -2.0 * log_surv
        };

        let sign = if z >= 0.0 { 1.0 } else { -1.0 };
        residuals.push(sign * dev.abs().sqrt());
    }

    residuals
}

pub fn compute_working_residuals(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    let (cdf_fn, pdf_fn): (DistFn, DistFn) = match distribution.to_lowercase().as_str() {
        "weibull" | "extreme" | "extreme_value" | "extremevalue" => {
            (extreme_value_cdf, extreme_value_pdf)
        }
        "logistic" | "loglogistic" => (logistic_cdf, logistic_pdf),
        "gaussian" | "lognormal" | "normal" => (gaussian_cdf, gaussian_pdf),
        _ => (extreme_value_cdf, extreme_value_pdf),
    };

    for i in 0..n {
        let y = time[i].ln();
        let z = (y - linear_pred[i]) / scale;

        let resid = if status[i] == 1 {
            let f = pdf_fn(z);
            let f_prime = match distribution.to_lowercase().as_str() {
                "weibull" | "extreme" | "extreme_value" | "extremevalue" => {
                    let ez = z.exp();
                    ez * (-ez).exp() * (1.0 - ez)
                }
                "logistic" | "loglogistic" => {
                    let ez = (-z).exp();
                    let denom = (1.0 + ez).powi(3);
                    ez * (ez - 1.0) / denom
                }
                _ => -z * f,
            };
            if f.abs() > 1e-300 { -f_prime / f } else { 0.0 }
        } else {
            let surv = 1.0 - cdf_fn(z);
            let f = pdf_fn(z);
            if surv.abs() > 1e-300 { f / surv } else { 0.0 }
        };

        residuals.push(resid);
    }

    residuals
}

pub fn compute_dfbeta_survreg(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    linear_pred: &[f64],
    scale: f64,
    var_matrix: &[Vec<f64>],
    distribution: &str,
) -> Vec<Vec<f64>> {
    let n = time.len();
    let nvar = if n > 0 && !covariates.is_empty() {
        covariates[0].len()
    } else {
        return vec![];
    };

    let working = compute_working_residuals(time, status, linear_pred, scale, distribution);

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

pub fn compute_ldcase(
    time: &[f64],
    status: &[i32],
    linear_pred: &[f64],
    scale: f64,
    distribution: &str,
) -> Vec<f64> {
    let n = time.len();
    let mut ld = Vec::with_capacity(n);

    let (cdf_fn, pdf_fn): (DistFn, DistFn) = match distribution.to_lowercase().as_str() {
        "weibull" | "extreme" | "extreme_value" | "extremevalue" => {
            (extreme_value_cdf, extreme_value_pdf)
        }
        "logistic" | "loglogistic" => (logistic_cdf, logistic_pdf),
        "gaussian" | "lognormal" | "normal" => (gaussian_cdf, gaussian_pdf),
        _ => (extreme_value_cdf, extreme_value_pdf),
    };

    for i in 0..n {
        let y = time[i].ln();
        let z = (y - linear_pred[i]) / scale;

        let contrib = if status[i] == 1 {
            let f = pdf_fn(z);
            if f > 1e-300 {
                f.ln() - scale.ln()
            } else {
                -690.0
            }
        } else {
            let surv = 1.0 - cdf_fn(z);
            if surv > 1e-300 { surv.ln() } else { -690.0 }
        };

        ld.push(contrib);
    }

    ld
}

#[pyfunction]
#[pyo3(signature = (time, status, linear_pred, scale, distribution, residual_type="deviance".to_string()))]
pub fn residuals_survreg(
    time: Vec<f64>,
    status: Vec<i32>,
    linear_pred: Vec<f64>,
    scale: f64,
    distribution: String,
    residual_type: String,
) -> PyResult<SurvregResiduals> {
    let n = time.len();
    if status.len() != n || linear_pred.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time, status, and linear_pred must have the same length",
        ));
    }

    let resid_type = SurvregResidType::from_str(&residual_type).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown residual type: {}. Valid types: response, deviance, working, ldcase",
            residual_type
        ))
    })?;

    let residuals = match resid_type {
        SurvregResidType::Response => compute_response_residuals(&time, &linear_pred),
        SurvregResidType::Deviance => {
            compute_deviance_residuals_survreg(&time, &status, &linear_pred, scale, &distribution)
        }
        SurvregResidType::Working => {
            compute_working_residuals(&time, &status, &linear_pred, scale, &distribution)
        }
        SurvregResidType::Ldcase | SurvregResidType::Ldresp | SurvregResidType::Ldshape => {
            compute_ldcase(&time, &status, &linear_pred, scale, &distribution)
        }
        _ => compute_deviance_residuals_survreg(&time, &status, &linear_pred, scale, &distribution),
    };

    Ok(SurvregResiduals {
        residuals,
        residual_type,
        n,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, linear_pred, scale, var_matrix, distribution))]
pub fn dfbeta_survreg(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    linear_pred: Vec<f64>,
    scale: f64,
    var_matrix: Vec<Vec<f64>>,
    distribution: String,
) -> PyResult<Vec<Vec<f64>>> {
    let n = time.len();
    if status.len() != n || linear_pred.len() != n || covariates.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "All inputs must have the same length",
        ));
    }

    Ok(compute_dfbeta_survreg(
        &time,
        &status,
        &covariates,
        &linear_pred,
        scale,
        &var_matrix,
        &distribution,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_residuals() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let linear_pred = vec![0.0, 0.5, 1.0, 1.2, 1.5];
        let resid = compute_response_residuals(&time, &linear_pred);

        assert_eq!(resid.len(), 5);
        assert!((resid[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_deviance_residuals() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let linear_pred = vec![0.0, 0.5, 1.0, 1.2, 1.5];
        let scale = 1.0;

        let resid =
            compute_deviance_residuals_survreg(&time, &status, &linear_pred, scale, "weibull");

        assert_eq!(resid.len(), 5);
    }
}
