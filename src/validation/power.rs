use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct SampleSizeResult {
    #[pyo3(get)]
    pub n_total: usize,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_per_group: Vec<usize>,
    #[pyo3(get)]
    pub power: f64,
    #[pyo3(get)]
    pub alpha: f64,
    #[pyo3(get)]
    pub hazard_ratio: f64,
    #[pyo3(get)]
    pub method: String,
}

#[pymethods]
impl SampleSizeResult {
    #[new]
    fn new(
        n_total: usize,
        n_events: usize,
        n_per_group: Vec<usize>,
        power: f64,
        alpha: f64,
        hazard_ratio: f64,
        method: String,
    ) -> Self {
        Self {
            n_total,
            n_events,
            n_per_group,
            power,
            alpha,
            hazard_ratio,
            method,
        }
    }
}

#[allow(clippy::excessive_precision)]
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
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

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
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
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub fn sample_size_logrank(
    hazard_ratio: f64,
    power: f64,
    alpha: f64,
    allocation_ratio: f64,
    sided: usize,
) -> SampleSizeResult {
    let alpha_adj = if sided == 1 { alpha } else { alpha / 2.0 };

    let z_alpha = norm_ppf(1.0 - alpha_adj);
    let z_beta = norm_ppf(power);

    let theta = hazard_ratio.ln();
    let r = allocation_ratio;

    let n_events = ((z_alpha + z_beta).powi(2) * (1.0 + r).powi(2)) / (r * theta.powi(2));
    let n_events = n_events.ceil() as usize;

    let n1 = (n_events as f64 / (1.0 + r)).ceil() as usize;
    let n2 = (n_events as f64 * r / (1.0 + r)).ceil() as usize;
    let n_total = n1 + n2;

    SampleSizeResult {
        n_total,
        n_events,
        n_per_group: vec![n1, n2],
        power,
        alpha,
        hazard_ratio,
        method: "Schoenfeld".to_string(),
    }
}

pub fn sample_size_freedman(
    hazard_ratio: f64,
    power: f64,
    alpha: f64,
    prob_event_control: f64,
    allocation_ratio: f64,
    sided: usize,
) -> SampleSizeResult {
    let alpha_adj = if sided == 1 { alpha } else { alpha / 2.0 };

    let z_alpha = norm_ppf(1.0 - alpha_adj);
    let z_beta = norm_ppf(power);

    let hr = hazard_ratio;
    let r = allocation_ratio;
    let p1 = prob_event_control;
    let p2 = 1.0 - (1.0 - p1).powf(hr);

    let p_avg = (p1 + r * p2) / (1.0 + r);

    let n_events = ((z_alpha + z_beta).powi(2) * (hr + 1.0).powi(2)) / (p_avg * (hr - 1.0).powi(2));
    let n_events = n_events.ceil() as usize;

    let n_total = (n_events as f64 / p_avg).ceil() as usize;
    let n1 = (n_total as f64 / (1.0 + r)).ceil() as usize;
    let n2 = n_total - n1;

    SampleSizeResult {
        n_total,
        n_events,
        n_per_group: vec![n1, n2],
        power,
        alpha,
        hazard_ratio,
        method: "Freedman".to_string(),
    }
}

pub fn power_logrank(
    n_events: usize,
    hazard_ratio: f64,
    alpha: f64,
    allocation_ratio: f64,
    sided: usize,
) -> f64 {
    let alpha_adj = if sided == 1 { alpha } else { alpha / 2.0 };
    let z_alpha = norm_ppf(1.0 - alpha_adj);

    let theta = hazard_ratio.ln();
    let r = allocation_ratio;

    let se = ((1.0 + r).powi(2) / (r * n_events as f64)).sqrt();
    let z = theta.abs() / se;

    norm_cdf(z - z_alpha)
}

#[pyfunction]
#[pyo3(signature = (hazard_ratio, power=None, alpha=None, allocation_ratio=None, sided=None))]
pub fn sample_size_survival(
    hazard_ratio: f64,
    power: Option<f64>,
    alpha: Option<f64>,
    allocation_ratio: Option<f64>,
    sided: Option<usize>,
) -> PyResult<SampleSizeResult> {
    let power = power.unwrap_or(0.8);
    let alpha = alpha.unwrap_or(0.05);
    let allocation_ratio = allocation_ratio.unwrap_or(1.0);
    let sided = sided.unwrap_or(2);

    if hazard_ratio <= 0.0 || hazard_ratio == 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }

    Ok(sample_size_logrank(
        hazard_ratio,
        power,
        alpha,
        allocation_ratio,
        sided,
    ))
}

#[pyfunction]
#[pyo3(signature = (hazard_ratio, prob_event, power=None, alpha=None, allocation_ratio=None, sided=None))]
pub fn sample_size_survival_freedman(
    hazard_ratio: f64,
    prob_event: f64,
    power: Option<f64>,
    alpha: Option<f64>,
    allocation_ratio: Option<f64>,
    sided: Option<usize>,
) -> PyResult<SampleSizeResult> {
    let power = power.unwrap_or(0.8);
    let alpha = alpha.unwrap_or(0.05);
    let allocation_ratio = allocation_ratio.unwrap_or(1.0);
    let sided = sided.unwrap_or(2);

    if hazard_ratio <= 0.0 || hazard_ratio == 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }

    if prob_event <= 0.0 || prob_event >= 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "prob_event must be between 0 and 1",
        ));
    }

    Ok(sample_size_freedman(
        hazard_ratio,
        power,
        alpha,
        prob_event,
        allocation_ratio,
        sided,
    ))
}

#[pyfunction]
#[pyo3(signature = (n_events, hazard_ratio, alpha=None, allocation_ratio=None, sided=None))]
pub fn power_survival(
    n_events: usize,
    hazard_ratio: f64,
    alpha: Option<f64>,
    allocation_ratio: Option<f64>,
    sided: Option<usize>,
) -> PyResult<f64> {
    let alpha = alpha.unwrap_or(0.05);
    let allocation_ratio = allocation_ratio.unwrap_or(1.0);
    let sided = sided.unwrap_or(2);

    if hazard_ratio <= 0.0 || hazard_ratio == 1.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }

    Ok(power_logrank(
        n_events,
        hazard_ratio,
        alpha,
        allocation_ratio,
        sided,
    ))
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct AccrualResult {
    #[pyo3(get)]
    pub n_total: usize,
    #[pyo3(get)]
    pub accrual_time: f64,
    #[pyo3(get)]
    pub followup_time: f64,
    #[pyo3(get)]
    pub study_duration: f64,
    #[pyo3(get)]
    pub expected_events: f64,
}

#[pymethods]
impl AccrualResult {
    #[new]
    fn new(
        n_total: usize,
        accrual_time: f64,
        followup_time: f64,
        study_duration: f64,
        expected_events: f64,
    ) -> Self {
        Self {
            n_total,
            accrual_time,
            followup_time,
            study_duration,
            expected_events,
        }
    }
}

pub fn expected_events_exponential(
    n_total: usize,
    hazard_control: f64,
    hazard_ratio: f64,
    accrual_time: f64,
    followup_time: f64,
    allocation_ratio: f64,
    dropout_rate: f64,
) -> f64 {
    let r = allocation_ratio;
    let n1 = n_total as f64 / (1.0 + r);
    let n2 = n_total as f64 * r / (1.0 + r);

    let lambda1 = hazard_control;
    let lambda2 = hazard_control * hazard_ratio;

    let study_duration = accrual_time + followup_time;

    let prob_event = |lambda: f64| -> f64 {
        let effective_lambda = lambda + dropout_rate;
        if accrual_time <= 0.0 {
            1.0 - (-lambda * followup_time).exp()
        } else {
            let term1 = 1.0 - (-effective_lambda * followup_time).exp();
            let term2 = (1.0 - (-effective_lambda * study_duration).exp())
                / (effective_lambda * accrual_time);
            term1.min(term2) * (lambda / effective_lambda)
        }
    };

    n1 * prob_event(lambda1) + n2 * prob_event(lambda2)
}

#[pyfunction]
#[pyo3(signature = (n_total, hazard_control, hazard_ratio, accrual_time, followup_time, allocation_ratio=None, dropout_rate=None))]
pub fn expected_events(
    n_total: usize,
    hazard_control: f64,
    hazard_ratio: f64,
    accrual_time: f64,
    followup_time: f64,
    allocation_ratio: Option<f64>,
    dropout_rate: Option<f64>,
) -> PyResult<AccrualResult> {
    let allocation_ratio = allocation_ratio.unwrap_or(1.0);
    let dropout_rate = dropout_rate.unwrap_or(0.0);

    let events = expected_events_exponential(
        n_total,
        hazard_control,
        hazard_ratio,
        accrual_time,
        followup_time,
        allocation_ratio,
        dropout_rate,
    );

    Ok(AccrualResult {
        n_total,
        accrual_time,
        followup_time,
        study_duration: accrual_time + followup_time,
        expected_events: events,
    })
}
