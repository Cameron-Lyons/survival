#![allow(clippy::too_many_arguments)]

use pyo3::prelude::*;

#[derive(Debug, Clone)]
#[pyclass]
pub struct WarrantyConfig {
    #[pyo3(get, set)]
    pub warranty_period: f64,
    #[pyo3(get, set)]
    pub cost_per_failure: f64,
    #[pyo3(get, set)]
    pub cost_per_repair: f64,
    #[pyo3(get, set)]
    pub discount_rate: f64,
}

#[pymethods]
impl WarrantyConfig {
    #[new]
    #[pyo3(signature = (
        warranty_period,
        cost_per_failure,
        cost_per_repair=None,
        discount_rate=0.0
    ))]
    pub fn new(
        warranty_period: f64,
        cost_per_failure: f64,
        cost_per_repair: Option<f64>,
        discount_rate: f64,
    ) -> Self {
        Self {
            warranty_period,
            cost_per_failure,
            cost_per_repair: cost_per_repair.unwrap_or(cost_per_failure * 0.5),
            discount_rate,
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct WarrantyResult {
    #[pyo3(get)]
    pub expected_failures: f64,
    #[pyo3(get)]
    pub expected_cost: f64,
    #[pyo3(get)]
    pub cost_per_unit: f64,
    #[pyo3(get)]
    pub failure_probability: f64,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_failures: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_cost: Vec<f64>,
}

#[pymethods]
impl WarrantyResult {
    fn __repr__(&self) -> String {
        format!(
            "WarrantyResult(E[failures]={:.2}, E[cost]={:.2})",
            self.expected_failures, self.expected_cost
        )
    }
}

fn estimate_survival(time: &[f64], event: &[i32], eval_times: &[f64]) -> Vec<f64> {
    let _n = time.len();
    let mut unique_times: Vec<f64> = time.to_vec();
    unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_times.dedup();

    let mut survival = vec![1.0; eval_times.len()];

    for (i, &t) in eval_times.iter().enumerate() {
        let mut surv = 1.0;

        for &ut in &unique_times {
            if ut > t {
                break;
            }

            let at_risk = time.iter().filter(|&&ti| ti >= ut).count() as f64;
            let events = time
                .iter()
                .zip(event.iter())
                .filter(|&(&ti, &ei)| (ti - ut).abs() < 1e-10 && ei == 1)
                .count() as f64;

            if at_risk > 0.0 {
                surv *= 1.0 - events / at_risk;
            }
        }

        survival[i] = surv;
    }

    survival
}

#[pyfunction]
#[pyo3(signature = (
    time,
    event,
    n_units,
    config
))]
pub fn warranty_analysis(
    time: Vec<f64>,
    event: Vec<i32>,
    n_units: usize,
    config: WarrantyConfig,
) -> PyResult<WarrantyResult> {
    let n = time.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "time and event must have the same non-zero length",
        ));
    }

    let n_points = 100;
    let time_points: Vec<f64> = (0..=n_points)
        .map(|i| config.warranty_period * i as f64 / n_points as f64)
        .collect();

    let survival = estimate_survival(&time, &event, &time_points);

    let failure_probs: Vec<f64> = survival.iter().map(|&s| 1.0 - s).collect();

    let failure_probability = *failure_probs.last().unwrap_or(&0.0);

    let expected_failures = n_units as f64 * failure_probability;

    let cumulative_failures: Vec<f64> = failure_probs.iter().map(|&p| n_units as f64 * p).collect();

    let mut cumulative_cost = Vec::new();
    for (i, &t) in time_points.iter().enumerate() {
        let discount = if config.discount_rate > 0.0 {
            (-config.discount_rate * t).exp()
        } else {
            1.0
        };
        cumulative_cost.push(cumulative_failures[i] * config.cost_per_failure * discount);
    }

    let expected_cost = *cumulative_cost.last().unwrap_or(&0.0);
    let cost_per_unit = expected_cost / n_units as f64;

    Ok(WarrantyResult {
        expected_failures,
        expected_cost,
        cost_per_unit,
        failure_probability,
        time_points,
        cumulative_failures,
        cumulative_cost,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct RenewalResult {
    #[pyo3(get)]
    pub expected_renewals: f64,
    #[pyo3(get)]
    pub renewal_variance: f64,
    #[pyo3(get)]
    pub mtbf: f64,
    #[pyo3(get)]
    pub availability: f64,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub renewal_function: Vec<f64>,
}

#[pymethods]
impl RenewalResult {
    fn __repr__(&self) -> String {
        format!(
            "RenewalResult(E[renewals]={:.2}, MTBF={:.2})",
            self.expected_renewals, self.mtbf
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    failure_times,
    event,
    time_horizon,
    repair_time=None
))]
pub fn renewal_analysis(
    failure_times: Vec<f64>,
    event: Vec<i32>,
    time_horizon: f64,
    repair_time: Option<f64>,
) -> PyResult<RenewalResult> {
    let n = failure_times.len();
    if n == 0 || event.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "failure_times and event must have the same non-zero length",
        ));
    }

    let repair_time = repair_time.unwrap_or(0.0);

    let event_times: Vec<f64> = failure_times
        .iter()
        .zip(event.iter())
        .filter(|&(_, &e)| e == 1)
        .map(|(&t, _)| t)
        .collect();

    let mtbf = if !event_times.is_empty() {
        event_times.iter().sum::<f64>() / event_times.len() as f64
    } else {
        time_horizon
    };

    let expected_renewals = if mtbf > 0.0 { time_horizon / mtbf } else { 0.0 };

    let variance: f64 = if event_times.len() > 1 {
        let mean = event_times.iter().sum::<f64>() / event_times.len() as f64;
        event_times.iter().map(|&t| (t - mean).powi(2)).sum::<f64>()
            / (event_times.len() - 1) as f64
    } else {
        mtbf * mtbf
    };

    let renewal_variance = if mtbf > 0.0 {
        (time_horizon / mtbf.powi(3)) * variance
    } else {
        0.0
    };

    let availability = if repair_time > 0.0 {
        mtbf / (mtbf + repair_time)
    } else {
        1.0
    };

    let n_points = 100;
    let time_points: Vec<f64> = (0..=n_points)
        .map(|i| time_horizon * i as f64 / n_points as f64)
        .collect();

    let renewal_function: Vec<f64> = time_points
        .iter()
        .map(|&t| if mtbf > 0.0 { t / mtbf } else { 0.0 })
        .collect();

    Ok(RenewalResult {
        expected_renewals,
        renewal_variance,
        mtbf,
        availability,
        time_points,
        renewal_function,
    })
}

#[derive(Debug, Clone)]
#[pyclass]
pub struct ReliabilityGrowthResult {
    #[pyo3(get)]
    pub initial_mtbf: f64,
    #[pyo3(get)]
    pub final_mtbf: f64,
    #[pyo3(get)]
    pub growth_rate: f64,
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub mtbf_trajectory: Vec<f64>,
}

#[pymethods]
impl ReliabilityGrowthResult {
    fn __repr__(&self) -> String {
        format!(
            "ReliabilityGrowthResult(MTBF: {:.2} -> {:.2}, rate={:.3})",
            self.initial_mtbf, self.final_mtbf, self.growth_rate
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    failure_times,
    cumulative_time
))]
pub fn reliability_growth(
    failure_times: Vec<f64>,
    cumulative_time: Vec<f64>,
) -> PyResult<ReliabilityGrowthResult> {
    let n = failure_times.len();
    if n < 2 || cumulative_time.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Need at least 2 failure times and matching cumulative times",
        ));
    }

    let ln_times: Vec<f64> = cumulative_time.iter().map(|&t| t.max(1e-10).ln()).collect();
    let ln_failures: Vec<f64> = (1..=n).map(|i| (i as f64).ln()).collect();

    let mean_x = ln_times.iter().sum::<f64>() / n as f64;
    let mean_y = ln_failures.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut denom = 0.0;
    for i in 0..n {
        num += (ln_times[i] - mean_x) * (ln_failures[i] - mean_y);
        denom += (ln_times[i] - mean_x).powi(2);
    }

    let beta = if denom.abs() > 1e-10 {
        num / denom
    } else {
        1.0
    };

    let growth_rate = 1.0 - beta;

    let max_time = cumulative_time
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let initial_mtbf = cumulative_time.first().unwrap_or(&1.0) / 1.0;
    let final_mtbf = max_time / n as f64;

    let n_points = 100;
    let time_points: Vec<f64> = (1..=n_points)
        .map(|i| max_time * i as f64 / n_points as f64)
        .collect();

    let mtbf_trajectory: Vec<f64> = time_points
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let expected_failures = ((i + 1) as f64).max(1.0);
            t / expected_failures
        })
        .collect();

    Ok(ReliabilityGrowthResult {
        initial_mtbf,
        final_mtbf,
        growth_rate,
        time_points,
        mtbf_trajectory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warranty_analysis() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 0, 1, 0, 1];
        let config = WarrantyConfig::new(5.0, 100.0, None, 0.0);

        let result = warranty_analysis(time, event, 1000, config).unwrap();
        assert!(result.expected_failures >= 0.0);
        assert!(result.failure_probability >= 0.0);
        assert!(result.failure_probability <= 1.0);
    }

    #[test]
    fn test_renewal_analysis() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = vec![1, 1, 1, 1, 1];

        let result = renewal_analysis(times, event, 100.0, Some(0.5)).unwrap();
        assert!(result.mtbf > 0.0);
        assert!(result.expected_renewals > 0.0);
        assert!(result.availability > 0.0);
        assert!(result.availability <= 1.0);
    }

    #[test]
    fn test_reliability_growth() {
        let failure_times = vec![10.0, 25.0, 45.0, 70.0, 100.0];
        let cumulative_time = vec![10.0, 25.0, 45.0, 70.0, 100.0];

        let result = reliability_growth(failure_times, cumulative_time).unwrap();
        assert!(result.final_mtbf > 0.0);
    }
}
