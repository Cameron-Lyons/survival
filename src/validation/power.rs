use crate::constants::{DEFAULT_ALLOCATION_RATIO, DEFAULT_ALPHA, DEFAULT_POWER, DEFAULT_SIDED};
use crate::internal::statistical::{normal_cdf as norm_cdf, normal_inverse_cdf as norm_ppf};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_probability_open(value: f64, field: &'static str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{field} must be finite")));
    }
    if !(value > 0.0 && value < 1.0) {
        return Err(PyValueError::new_err(format!(
            "{field} must be greater than 0 and less than 1"
        )));
    }
    Ok(())
}

fn validate_positive_finite(value: f64, field: &'static str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{field} must be finite")));
    }
    if value <= 0.0 {
        return Err(PyValueError::new_err(format!("{field} must be positive")));
    }
    Ok(())
}

fn validate_nonnegative_finite(value: f64, field: &'static str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!("{field} must be finite")));
    }
    if value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{field} must be non-negative"
        )));
    }
    Ok(())
}

fn validate_hazard_ratio(value: f64) -> PyResult<()> {
    validate_positive_finite(value, "hazard_ratio")?;
    if value == 1.0 {
        return Err(PyValueError::new_err(
            "hazard_ratio must be positive and not equal to 1",
        ));
    }
    Ok(())
}

fn validate_sided(value: usize) -> PyResult<()> {
    if value != 1 && value != 2 {
        return Err(PyValueError::new_err("sided must be 1 or 2"));
    }
    Ok(())
}

fn validate_sample_size_options(
    power: f64,
    alpha: f64,
    allocation_ratio: f64,
    sided: usize,
) -> PyResult<()> {
    validate_probability_open(power, "power")?;
    validate_probability_open(alpha, "alpha")?;
    validate_positive_finite(allocation_ratio, "allocation_ratio")?;
    validate_sided(sided)
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
pub(crate) fn sample_size_logrank(
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
pub(crate) fn sample_size_freedman(
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
pub(crate) fn power_logrank(
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
    let power = power.unwrap_or(DEFAULT_POWER);
    let alpha = alpha.unwrap_or(DEFAULT_ALPHA);
    let allocation_ratio = allocation_ratio.unwrap_or(DEFAULT_ALLOCATION_RATIO);
    let sided = sided.unwrap_or(DEFAULT_SIDED);
    validate_hazard_ratio(hazard_ratio)?;
    validate_sample_size_options(power, alpha, allocation_ratio, sided)?;
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
    let power = power.unwrap_or(DEFAULT_POWER);
    let alpha = alpha.unwrap_or(DEFAULT_ALPHA);
    let allocation_ratio = allocation_ratio.unwrap_or(DEFAULT_ALLOCATION_RATIO);
    let sided = sided.unwrap_or(DEFAULT_SIDED);
    validate_hazard_ratio(hazard_ratio)?;
    validate_probability_open(prob_event, "prob_event")?;
    validate_sample_size_options(power, alpha, allocation_ratio, sided)?;
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
    let alpha = alpha.unwrap_or(DEFAULT_ALPHA);
    let allocation_ratio = allocation_ratio.unwrap_or(DEFAULT_ALLOCATION_RATIO);
    let sided = sided.unwrap_or(DEFAULT_SIDED);
    if n_events == 0 {
        return Err(PyValueError::new_err("n_events must be positive"));
    }
    validate_hazard_ratio(hazard_ratio)?;
    validate_probability_open(alpha, "alpha")?;
    validate_positive_finite(allocation_ratio, "allocation_ratio")?;
    validate_sided(sided)?;
    Ok(power_logrank(
        n_events,
        hazard_ratio,
        alpha,
        allocation_ratio,
        sided,
    ))
}
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
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
pub(crate) fn expected_events_exponential(
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
    if n_total == 0 {
        return Err(PyValueError::new_err("n_total must be positive"));
    }
    validate_positive_finite(hazard_control, "hazard_control")?;
    validate_positive_finite(hazard_ratio, "hazard_ratio")?;
    validate_nonnegative_finite(accrual_time, "accrual_time")?;
    validate_nonnegative_finite(followup_time, "followup_time")?;
    if accrual_time == 0.0 && followup_time == 0.0 {
        return Err(PyValueError::new_err(
            "accrual_time and followup_time cannot both be zero",
        ));
    }
    validate_positive_finite(allocation_ratio, "allocation_ratio")?;
    validate_nonnegative_finite(dropout_rate, "dropout_rate")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_sample_size_survival_validates_parameters() {
        assert!(sample_size_survival(0.7, None, None, None, None).is_ok());
        assert!(sample_size_survival(f64::NAN, None, None, None, None).is_err());
        assert!(sample_size_survival(1.0, None, None, None, None).is_err());
        assert!(sample_size_survival(0.7, Some(1.0), None, None, None).is_err());
        assert!(sample_size_survival(0.7, None, Some(0.0), None, None).is_err());
        assert!(sample_size_survival(0.7, None, None, Some(0.0), None).is_err());
        assert!(sample_size_survival(0.7, None, None, None, Some(3)).is_err());
    }

    #[test]
    fn public_freedman_and_power_validate_parameters() {
        assert!(sample_size_survival_freedman(0.7, 0.6, None, None, None, None).is_ok());
        assert!(sample_size_survival_freedman(0.7, f64::INFINITY, None, None, None, None).is_err());
        assert!(sample_size_survival_freedman(0.7, 0.0, None, None, None, None).is_err());
        assert!(power_survival(64, 0.7, None, None, None).is_ok());
        assert!(power_survival(0, 0.7, None, None, None).is_err());
        assert!(power_survival(64, 0.7, Some(f64::NAN), None, None).is_err());
        assert!(power_survival(64, 0.7, None, Some(-1.0), None).is_err());
    }

    #[test]
    fn public_expected_events_validates_parameters() {
        assert!(expected_events(100, 0.1, 0.7, 12.0, 6.0, None, None).is_ok());
        assert!(expected_events(0, 0.1, 0.7, 12.0, 6.0, None, None).is_err());
        assert!(expected_events(100, f64::NAN, 0.7, 12.0, 6.0, None, None).is_err());
        assert!(expected_events(100, 0.1, 0.0, 12.0, 6.0, None, None).is_err());
        assert!(expected_events(100, 0.1, 0.7, -1.0, 6.0, None, None).is_err());
        assert!(expected_events(100, 0.1, 0.7, 0.0, 0.0, None, None).is_err());
        assert!(expected_events(100, 0.1, 0.7, 12.0, 6.0, Some(0.0), None).is_err());
        assert!(expected_events(100, 0.1, 0.7, 12.0, 6.0, None, Some(-0.1)).is_err());
    }
}
