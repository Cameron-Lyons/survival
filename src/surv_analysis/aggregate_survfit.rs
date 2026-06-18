use crate::constants::TIME_EPSILON;
use crate::internal::statistical::normal_inverse_cdf;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

fn value_error(message: impl Into<String>) -> PyErr {
    PyErr::new::<PyValueError, _>(message.into())
}

fn validate_probability_curve(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(value_error(format!(
                "{name} values must be between 0 and 1; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_nonnegative_finite_curve(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "{name} values must be non-negative; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_time_curve(name: &str, values: &[f64]) -> PyResult<()> {
    let mut previous = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "{name} values must be non-negative; got {value} at index {idx}"
            )));
        }
        if value + TIME_EPSILON < previous {
            return Err(value_error(format!(
                "{name} values must be sorted in non-decreasing order"
            )));
        }
        previous = value;
    }
    Ok(())
}

fn validate_conf_level(conf_level: f64) -> PyResult<()> {
    if !conf_level.is_finite() || !(0.0..1.0).contains(&conf_level) {
        return Err(value_error("conf_level must be finite and between 0 and 1"));
    }
    Ok(())
}

fn normalized_weights(weights: Option<Vec<f64>>, n_curves: usize) -> PyResult<Vec<f64>> {
    match weights {
        Some(wts) => {
            if wts.len() != n_curves {
                return Err(value_error(
                    "weights must have same length as number of curves",
                ));
            }
            validate_nonnegative_finite_curve("weights", &wts)?;
            let sum: f64 = wts.iter().sum();
            if sum <= 0.0 {
                return Err(value_error(
                    "weights must include at least one positive value",
                ));
            }
            Ok(wts.iter().map(|&x| x / sum).collect())
        }
        None => Ok(vec![1.0 / n_curves as f64; n_curves]),
    }
}

fn validate_curve_inputs(
    times: &[Vec<f64>],
    survs: &[Vec<f64>],
    std_errs: Option<&[Vec<f64>]>,
) -> PyResult<()> {
    for (idx, (time, surv)) in times.iter().zip(survs.iter()).enumerate() {
        if time.len() != surv.len() {
            return Err(value_error(format!(
                "times[{idx}] and survs[{idx}] must have the same length"
            )));
        }
        validate_time_curve(&format!("times[{idx}]"), time)?;
        validate_probability_curve(&format!("survs[{idx}]"), surv)?;
    }

    if let Some(ses) = std_errs {
        if ses.len() != times.len() {
            return Err(value_error(
                "std_errs must have same length as number of curves",
            ));
        }
        for (idx, se) in ses.iter().enumerate() {
            if se.len() != times[idx].len() {
                return Err(value_error(format!(
                    "std_errs[{idx}] must have the same length as times[{idx}]"
                )));
            }
            validate_nonnegative_finite_curve(&format!("std_errs[{idx}]"), se)?;
        }
    }
    Ok(())
}

/// Result of aggregating survival curves
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct AggregateSurvfitResult {
    /// Aggregated time points
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Aggregated survival estimates
    #[pyo3(get)]
    pub surv: Vec<f64>,
    /// Aggregated standard errors
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    /// Lower confidence bounds
    #[pyo3(get)]
    pub lower: Vec<f64>,
    /// Upper confidence bounds
    #[pyo3(get)]
    pub upper: Vec<f64>,
    /// Number of curves aggregated
    #[pyo3(get)]
    pub n_curves: usize,
    /// Weights used for aggregation
    #[pyo3(get)]
    pub weights: Vec<f64>,
}

/// Aggregate (average) multiple survival curves.
///
/// This function computes the weighted average of multiple survival curves,
/// typically used for computing marginal survival estimates from Cox models
/// or for meta-analysis of survival curves.
///
/// # Arguments
/// * `times` - Vector of time vectors (one per curve)
/// * `survs` - Vector of survival estimate vectors (one per curve)
/// * `std_errs` - Optional vector of standard error vectors
/// * `weights` - Optional weights for each curve (default: equal weights)
/// * `conf_level` - Confidence level for intervals (default: 0.95)
///
/// # Returns
/// * `AggregateSurvfitResult` with aggregated estimates
#[pyfunction]
#[pyo3(signature = (times, survs, std_errs=None, weights=None, conf_level=None))]
pub fn aggregate_survfit(
    times: Vec<Vec<f64>>,
    survs: Vec<Vec<f64>>,
    std_errs: Option<Vec<Vec<f64>>>,
    weights: Option<Vec<f64>>,
    conf_level: Option<f64>,
) -> PyResult<AggregateSurvfitResult> {
    let n_curves = times.len();

    if survs.len() != n_curves {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "times and survs must have same length",
        ));
    }

    if n_curves == 0 {
        if let Some(wts) = weights
            && !wts.is_empty()
        {
            return Err(value_error(
                "weights must be empty when no curves are supplied",
            ));
        }
        if let Some(ses) = std_errs
            && !ses.is_empty()
        {
            return Err(value_error(
                "std_errs must be empty when no curves are supplied",
            ));
        }
        return Ok(AggregateSurvfitResult {
            time: vec![],
            surv: vec![],
            std_err: vec![],
            lower: vec![],
            upper: vec![],
            n_curves: 0,
            weights: vec![],
        });
    }

    let conf = conf_level.unwrap_or(0.95);
    validate_conf_level(conf)?;
    let z = z_score(conf);

    validate_curve_inputs(&times, &survs, std_errs.as_deref())?;
    let w = normalized_weights(weights, n_curves)?;

    let mut all_times: Vec<f64> = times.iter().flatten().cloned().collect();
    all_times.sort_by(|a, b| a.total_cmp(b));
    all_times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);

    let mut interpolated_survs = vec![vec![0.0; all_times.len()]; n_curves];
    let mut interpolated_ses = vec![vec![0.0; all_times.len()]; n_curves];

    for (i, (t, s)) in times.iter().zip(survs.iter()).enumerate() {
        for (j, &eval_t) in all_times.iter().enumerate() {
            interpolated_survs[i][j] = interpolate_step(t, s, eval_t, 1.0);

            if let Some(ref ses) = std_errs
                && i < ses.len()
            {
                interpolated_ses[i][j] = interpolate_step(t, &ses[i], eval_t, 0.0);
            }
        }
    }

    let mut agg_surv = vec![0.0; all_times.len()];
    let mut agg_se = vec![0.0; all_times.len()];

    for j in 0..all_times.len() {
        for i in 0..n_curves {
            agg_surv[j] += w[i] * interpolated_survs[i][j];
            agg_se[j] += w[i] * w[i] * interpolated_ses[i][j].powi(2);
        }
        agg_se[j] = agg_se[j].sqrt();
    }

    let lower: Vec<f64> = agg_surv
        .iter()
        .zip(agg_se.iter())
        .map(|(&s, &se)| (s - z * se).max(0.0))
        .collect();

    let upper: Vec<f64> = agg_surv
        .iter()
        .zip(agg_se.iter())
        .map(|(&s, &se)| (s + z * se).min(1.0))
        .collect();

    Ok(AggregateSurvfitResult {
        time: all_times,
        surv: agg_surv,
        std_err: agg_se,
        lower,
        upper,
        n_curves,
        weights: w,
    })
}

/// Interpolate step function at a given point
fn interpolate_step(times: &[f64], values: &[f64], at: f64, default_value: f64) -> f64 {
    if times.is_empty() || values.is_empty() {
        return default_value;
    }

    if at + TIME_EPSILON < times[0] {
        return default_value;
    }

    let idx = times
        .iter()
        .position(|&t| t > at + TIME_EPSILON)
        .unwrap_or(times.len());

    if idx == 0 { 1.0 } else { values[idx - 1] }
}

fn z_score(conf_level: f64) -> f64 {
    let p = (1.0 + conf_level) / 2.0;
    normal_inverse_cdf(p)
}

/// Average survival curves by group
#[pyfunction]
#[pyo3(signature = (times, survs, groups, weights=None))]
pub fn aggregate_survfit_by_group(
    times: Vec<Vec<f64>>,
    survs: Vec<Vec<f64>>,
    groups: Vec<i32>,
    weights: Option<Vec<f64>>,
) -> PyResult<Vec<AggregateSurvfitResult>> {
    let n = times.len();
    if survs.len() != n || groups.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "times, survs, and groups must have same length",
        ));
    }
    if let Some(values) = weights.as_ref()
        && values.len() != n
    {
        return Err(value_error(
            "weights must have same length as number of curves",
        ));
    }

    let mut grouped: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, &g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i);
    }

    let mut results = Vec::new();

    for (_group, indices) in grouped {
        let group_times: Vec<Vec<f64>> = indices.iter().map(|&i| times[i].clone()).collect();
        let group_survs: Vec<Vec<f64>> = indices.iter().map(|&i| survs[i].clone()).collect();
        let group_weights: Option<Vec<f64>> = weights
            .as_ref()
            .map(|w| indices.iter().map(|&i| w[i]).collect());

        let result = aggregate_survfit(group_times, group_survs, None, group_weights, None)?;
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_survfit_basic() {
        let times = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let survs = vec![vec![0.9, 0.8, 0.7], vec![0.95, 0.85, 0.75]];

        let result = aggregate_survfit(times, survs, None, None, None).unwrap();

        assert_eq!(result.n_curves, 2);
        assert!(!result.time.is_empty());

        for s in &result.surv {
            assert!(*s >= 0.7 && *s <= 0.95);
        }
    }

    #[test]
    fn test_aggregate_survfit_weighted() {
        let times = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let survs = vec![vec![0.9, 0.8], vec![0.8, 0.6]];
        let weights = vec![0.75, 0.25];

        let result = aggregate_survfit(times, survs, None, Some(weights), None).unwrap();

        assert!(result.surv[0] > 0.85);
    }

    #[test]
    fn test_aggregate_survfit_validates_inputs() {
        assert!(
            aggregate_survfit(vec![vec![1.0, 2.0]], vec![vec![0.9]], None, None, None).is_err()
        );
        assert!(aggregate_survfit(vec![vec![1.0]], vec![vec![1.1]], None, None, None).is_err());
        assert!(
            aggregate_survfit(
                vec![vec![1.0]],
                vec![vec![0.9]],
                Some(vec![vec![-0.1]]),
                None,
                None
            )
            .is_err()
        );
        assert!(
            aggregate_survfit(
                vec![vec![1.0]],
                vec![vec![0.9]],
                None,
                Some(vec![0.0]),
                None
            )
            .is_err()
        );
        assert!(
            aggregate_survfit(vec![vec![1.0]], vec![vec![0.9]], None, None, Some(1.0)).is_err()
        );
    }

    #[test]
    fn test_aggregate_survfit_deduplicates_near_times() {
        let result = aggregate_survfit(
            vec![vec![1.0, 2.0], vec![1.0 + TIME_EPSILON / 2.0, 2.0]],
            vec![vec![0.9, 0.8], vec![0.95, 0.85]],
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(result.time, vec![1.0, 2.0]);
        assert_eq!(result.surv, vec![0.925, 0.825]);
    }

    #[test]
    fn test_aggregate_survfit_empty() {
        let times: Vec<Vec<f64>> = vec![];
        let survs: Vec<Vec<f64>> = vec![];

        let result = aggregate_survfit(times, survs, None, None, None).unwrap();
        assert_eq!(result.n_curves, 0);
    }
}
