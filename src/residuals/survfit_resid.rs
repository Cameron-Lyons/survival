use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SurvfitResiduals {
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub residual_type: String,
}

#[pymethods]
impl SurvfitResiduals {
    fn __repr__(&self) -> String {
        format!(
            "SurvfitResiduals(type='{}', n={})",
            self.residual_type,
            self.residuals.len()
        )
    }
}

pub fn compute_martingale_survfit(
    time: &[f64],
    status: &[i32],
    surv_time: &[f64],
    surv: &[f64],
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let cumhaz = get_cumhaz_at_time(time[i], surv_time, surv);
        let resid = status[i] as f64 - cumhaz;
        residuals.push(resid);
    }

    residuals
}

pub fn compute_deviance_survfit(
    time: &[f64],
    status: &[i32],
    surv_time: &[f64],
    surv: &[f64],
) -> Vec<f64> {
    let martingale = compute_martingale_survfit(time, status, surv_time, surv);

    martingale
        .iter()
        .zip(status.iter())
        .map(|(&m, &d)| {
            let sign = if m >= 0.0 { 1.0 } else { -1.0 };
            let abs_term = if d == 1 {
                -2.0 * (m - 1.0 + (1.0 - m).max(1e-10).ln())
            } else {
                -2.0 * m
            };
            sign * abs_term.abs().sqrt()
        })
        .collect()
}

fn get_cumhaz_at_time(t: f64, surv_time: &[f64], surv: &[f64]) -> f64 {
    for i in (0..surv_time.len()).rev() {
        if surv_time[i] <= t {
            let s = surv[i];
            if s > 0.0 && s <= 1.0 {
                return -s.ln();
            }
            return 0.0;
        }
    }
    0.0
}

pub fn compute_quantile_residuals(
    time: &[f64],
    status: &[i32],
    surv_time: &[f64],
    surv: &[f64],
) -> Vec<f64> {
    let n = time.len();
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let s_at_t = get_surv_at_time(time[i], surv_time, surv);

        let resid = if status[i] == 1 {
            if s_at_t > 0.0 && s_at_t < 1.0 {
                normal_quantile(1.0 - s_at_t)
            } else if s_at_t <= 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else if s_at_t > 0.0 && s_at_t < 1.0 {
            normal_quantile(1.0 - s_at_t / 2.0)
        } else {
            0.0
        };

        residuals.push(resid);
    }

    residuals
}

fn get_surv_at_time(t: f64, surv_time: &[f64], surv: &[f64]) -> f64 {
    for i in (0..surv_time.len()).rev() {
        if surv_time[i] <= t {
            return surv[i];
        }
    }
    1.0
}

fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    #[allow(clippy::excessive_precision)]
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    #[allow(clippy::excessive_precision)]
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    #[allow(clippy::excessive_precision)]
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    #[allow(clippy::excessive_precision)]
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[pyfunction]
#[pyo3(signature = (time, status, surv_time, surv, residual_type="martingale".to_string()))]
pub fn residuals_survfit(
    time: Vec<f64>,
    status: Vec<i32>,
    surv_time: Vec<f64>,
    surv: Vec<f64>,
    residual_type: String,
) -> PyResult<SurvfitResiduals> {
    let n = time.len();
    if status.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and status must have the same length",
        ));
    }

    if surv_time.len() != surv.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "surv_time and surv must have the same length",
        ));
    }

    let residuals = match residual_type.to_lowercase().as_str() {
        "martingale" => compute_martingale_survfit(&time, &status, &surv_time, &surv),
        "deviance" => compute_deviance_survfit(&time, &status, &surv_time, &surv),
        "quantile" => compute_quantile_residuals(&time, &status, &surv_time, &surv),
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown residual type: {}. Valid types: martingale, deviance, quantile",
                residual_type
            )));
        }
    };

    Ok(SurvfitResiduals {
        residuals,
        time,
        residual_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martingale_survfit() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let surv_time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let surv = vec![0.9, 0.85, 0.75, 0.7, 0.6];

        let resid = compute_martingale_survfit(&time, &status, &surv_time, &surv);

        assert_eq!(resid.len(), 5);
    }

    #[test]
    fn test_deviance_survfit() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];
        let surv_time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let surv = vec![0.9, 0.85, 0.75, 0.7, 0.6];

        let resid = compute_deviance_survfit(&time, &status, &surv_time, &surv);

        assert_eq!(resid.len(), 5);
    }
}
