use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass]
pub enum ReliabilityScale {
    Surv,
    Cumhaz,
    ClogLog,
    LogLogistic,
    Probit,
}

#[pymethods]
impl ReliabilityScale {
    #[staticmethod]
    #[pyo3(name = "from_str")]
    pub fn from_string(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "surv" | "survival" => Ok(ReliabilityScale::Surv),
            "cumhaz" | "cumulative_hazard" => Ok(ReliabilityScale::Cumhaz),
            "cloglog" | "complementary_log_log" => Ok(ReliabilityScale::ClogLog),
            "loglogistic" | "log_logistic" => Ok(ReliabilityScale::LogLogistic),
            "probit" => Ok(ReliabilityScale::Probit),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown scale: {}. Valid scales: surv, cumhaz, cloglog, loglogistic, probit",
                s
            ))),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ReliabilityResult {
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub estimate: Vec<f64>,
    #[pyo3(get)]
    pub std_err: Option<Vec<f64>>,
    #[pyo3(get)]
    pub lower: Option<Vec<f64>>,
    #[pyo3(get)]
    pub upper: Option<Vec<f64>>,
    #[pyo3(get)]
    pub scale: String,
}

#[pymethods]
impl ReliabilityResult {
    #[new]
    #[pyo3(signature = (time, estimate, std_err=None, lower=None, upper=None, scale="surv".to_string()))]
    pub fn new(
        time: Vec<f64>,
        estimate: Vec<f64>,
        std_err: Option<Vec<f64>>,
        lower: Option<Vec<f64>>,
        upper: Option<Vec<f64>>,
        scale: String,
    ) -> Self {
        Self {
            time,
            estimate,
            std_err,
            lower,
            upper,
            scale,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ReliabilityResult(n_times={}, scale='{}')",
            self.time.len(),
            self.scale
        )
    }
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

fn probit(p: f64) -> f64 {
    normal_quantile(p)
}

fn probit_inverse(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
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

pub fn surv_to_cumhaz(surv: f64) -> f64 {
    if surv <= 0.0 {
        f64::INFINITY
    } else {
        -surv.ln()
    }
}

pub fn cumhaz_to_surv(cumhaz: f64) -> f64 {
    (-cumhaz).exp()
}

pub fn surv_to_cloglog(surv: f64) -> f64 {
    if surv <= 0.0 {
        f64::INFINITY
    } else if surv >= 1.0 {
        f64::NEG_INFINITY
    } else {
        (-surv.ln()).ln()
    }
}

pub fn cloglog_to_surv(cloglog: f64) -> f64 {
    (-cloglog.exp()).exp()
}

pub fn surv_to_loglogistic(surv: f64) -> f64 {
    if surv <= 0.0 {
        f64::INFINITY
    } else if surv >= 1.0 {
        f64::NEG_INFINITY
    } else {
        ((1.0 - surv) / surv).ln()
    }
}

pub fn loglogistic_to_surv(logit: f64) -> f64 {
    1.0 / (1.0 + logit.exp())
}

pub fn surv_to_probit(surv: f64) -> f64 {
    if surv <= 0.0 {
        f64::NEG_INFINITY
    } else if surv >= 1.0 {
        f64::INFINITY
    } else {
        probit(1.0 - surv)
    }
}

pub fn probit_to_surv(z: f64) -> f64 {
    1.0 - probit_inverse(z)
}

#[pyfunction]
#[pyo3(signature = (time, surv, std_err=None, conf_level=0.95, scale="cumhaz".to_string()))]
pub fn reliability(
    time: Vec<f64>,
    surv: Vec<f64>,
    std_err: Option<Vec<f64>>,
    conf_level: f64,
    scale: String,
) -> PyResult<ReliabilityResult> {
    if time.len() != surv.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and surv must have the same length",
        ));
    }

    let scale_enum = ReliabilityScale::from_string(&scale)?;

    let transform: fn(f64) -> f64 = match scale_enum {
        ReliabilityScale::Surv => |s| s,
        ReliabilityScale::Cumhaz => surv_to_cumhaz,
        ReliabilityScale::ClogLog => surv_to_cloglog,
        ReliabilityScale::LogLogistic => surv_to_loglogistic,
        ReliabilityScale::Probit => surv_to_probit,
    };

    let estimate: Vec<f64> = surv.iter().map(|&s| transform(s)).collect();

    let (transformed_se, lower, upper) = if let Some(se) = std_err {
        if se.len() != surv.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "std_err must have the same length as surv",
            ));
        }

        let z = normal_quantile(1.0 - (1.0 - conf_level) / 2.0);

        let mut trans_se = Vec::with_capacity(surv.len());
        let mut lo = Vec::with_capacity(surv.len());
        let mut hi = Vec::with_capacity(surv.len());

        for i in 0..surv.len() {
            let s = surv[i];
            let se_val = se[i];

            let (new_se, l, u) = match scale_enum {
                ReliabilityScale::Surv => {
                    let l = (s - z * se_val).max(0.0);
                    let u = (s + z * se_val).min(1.0);
                    (se_val, l, u)
                }
                ReliabilityScale::Cumhaz => {
                    if s > 0.0 && se_val > 0.0 {
                        let h = -s.ln();
                        let se_h = se_val / s;
                        let l = (h - z * se_h).max(0.0);
                        let u = h + z * se_h;
                        (se_h, l, u)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    }
                }
                ReliabilityScale::ClogLog => {
                    if s > 0.0 && s < 1.0 && se_val > 0.0 {
                        let h = -s.ln();
                        let se_cll = se_val / (s * h);
                        let cll = h.ln();
                        let l = cll - z * se_cll;
                        let u = cll + z * se_cll;
                        (se_cll, l, u)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    }
                }
                ReliabilityScale::LogLogistic => {
                    if s > 0.0 && s < 1.0 && se_val > 0.0 {
                        let se_logit = se_val / (s * (1.0 - s));
                        let logit = ((1.0 - s) / s).ln();
                        let l = logit - z * se_logit;
                        let u = logit + z * se_logit;
                        (se_logit, l, u)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    }
                }
                ReliabilityScale::Probit => {
                    if s > 0.0 && s < 1.0 && se_val > 0.0 {
                        let p = 1.0 - s;
                        let phi =
                            (-0.5 * probit(p).powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();
                        let se_probit = se_val / phi;
                        let z_val = probit(p);
                        let l = z_val - z * se_probit;
                        let u = z_val + z * se_probit;
                        (se_probit, l, u)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    }
                }
            };

            trans_se.push(new_se);
            lo.push(l);
            hi.push(u);
        }

        (Some(trans_se), Some(lo), Some(hi))
    } else {
        (None, None, None)
    };

    Ok(ReliabilityResult {
        time,
        estimate,
        std_err: transformed_se,
        lower,
        upper,
        scale,
    })
}

#[pyfunction]
pub fn reliability_inverse(estimate: Vec<f64>, scale: String) -> PyResult<Vec<f64>> {
    let scale_enum = ReliabilityScale::from_string(&scale)?;

    let inverse: fn(f64) -> f64 = match scale_enum {
        ReliabilityScale::Surv => |s| s,
        ReliabilityScale::Cumhaz => cumhaz_to_surv,
        ReliabilityScale::ClogLog => cloglog_to_surv,
        ReliabilityScale::LogLogistic => loglogistic_to_surv,
        ReliabilityScale::Probit => probit_to_surv,
    };

    Ok(estimate.iter().map(|&e| inverse(e)).collect())
}

#[pyfunction]
pub fn hazard_to_reliability(time: Vec<f64>, hazard: Vec<f64>) -> PyResult<ReliabilityResult> {
    if time.len() != hazard.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and hazard must have the same length",
        ));
    }

    let mut cumhaz = Vec::with_capacity(hazard.len());
    let mut cum = 0.0;
    for &h in &hazard {
        cum += h;
        cumhaz.push(cum);
    }

    let surv: Vec<f64> = cumhaz.iter().map(|&h| (-h).exp()).collect();

    Ok(ReliabilityResult {
        time,
        estimate: surv,
        std_err: None,
        lower: None,
        upper: None,
        scale: "surv".to_string(),
    })
}

#[pyfunction]
pub fn failure_probability(surv: Vec<f64>) -> Vec<f64> {
    surv.iter().map(|&s| 1.0 - s).collect()
}

#[pyfunction]
pub fn conditional_reliability(
    time: Vec<f64>,
    surv: Vec<f64>,
    t0: f64,
) -> PyResult<ReliabilityResult> {
    if time.len() != surv.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and surv must have the same length",
        ));
    }

    let s0 = time
        .iter()
        .zip(surv.iter())
        .rev()
        .find(|(t, _)| **t <= t0)
        .map(|(_, s)| *s)
        .unwrap_or(1.0);

    if s0 <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Survival probability at t0 is zero; cannot compute conditional reliability",
        ));
    }

    let conditional: Vec<f64> = surv.iter().map(|&s| s / s0).collect();
    let cond_time: Vec<f64> = time.iter().map(|&t| t - t0).collect();

    Ok(ReliabilityResult {
        time: cond_time,
        estimate: conditional,
        std_err: None,
        lower: None,
        upper: None,
        scale: "conditional_surv".to_string(),
    })
}

#[pyfunction]
pub fn mean_residual_life(time: Vec<f64>, surv: Vec<f64>, at_time: f64) -> PyResult<f64> {
    if time.len() != surv.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and surv must have the same length",
        ));
    }

    if time.len() < 2 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Need at least 2 time points",
        ));
    }

    let s_at = time
        .iter()
        .zip(surv.iter())
        .rev()
        .find(|(t, _)| **t <= at_time)
        .map(|(_, s)| *s)
        .unwrap_or(1.0);

    if s_at <= 0.0 {
        return Ok(0.0);
    }

    let mut integral = 0.0;
    for i in 0..time.len() - 1 {
        if time[i] >= at_time {
            let dt = time[i + 1] - time[i];
            let s_i = surv[i];
            let s_ip1 = surv[i + 1];
            let avg_surv = (s_i + s_ip1) / 2.0;
            integral += dt * avg_surv;
        }
    }

    Ok(integral / s_at)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surv_transformations() {
        let surv = 0.5;

        let cumhaz = surv_to_cumhaz(surv);
        let back = cumhaz_to_surv(cumhaz);
        assert!((back - surv).abs() < 1e-10);

        let cloglog = surv_to_cloglog(surv);
        let back = cloglog_to_surv(cloglog);
        assert!((back - surv).abs() < 1e-10);

        let logit = surv_to_loglogistic(surv);
        let back = loglogistic_to_surv(logit);
        assert!((back - surv).abs() < 1e-10);

        let z = surv_to_probit(surv);
        let back = probit_to_surv(z);
        assert!((back - surv).abs() < 1e-6);
    }

    #[test]
    fn test_reliability_function() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let surv = vec![0.9, 0.8, 0.7, 0.6, 0.5];
        let result = reliability(time, surv, None, 0.95, "cumhaz".to_string()).unwrap();

        assert_eq!(result.estimate.len(), 5);
        assert!((result.estimate[0] - (-0.9_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_failure_probability() {
        let surv = vec![1.0, 0.9, 0.8, 0.5];
        let fp = failure_probability(surv);
        assert!((fp[0] - 0.0).abs() < 1e-10);
        assert!((fp[3] - 0.5).abs() < 1e-10);
    }
}
