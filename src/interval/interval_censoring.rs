use crate::constants::{PARALLEL_THRESHOLD_XLARGE, clamped_normal_ci_bounds_95};
use crate::internal::statistical::erf;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

type DistributionFn = fn(f64, f64, f64) -> f64;
type DistributionFns = (DistributionFn, DistributionFn);
type TimeSurvivalCurve = (Vec<f64>, Vec<f64>);

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum IntervalDistribution {
    Weibull,
    LogNormal,
    LogLogistic,
    Exponential,
    Generalized,
}

#[pymethods]
impl IntervalDistribution {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "weibull" => Ok(IntervalDistribution::Weibull),
            "lognormal" | "log_normal" => Ok(IntervalDistribution::LogNormal),
            "loglogistic" | "log_logistic" => Ok(IntervalDistribution::LogLogistic),
            "exponential" | "exp" => Ok(IntervalDistribution::Exponential),
            "generalized" | "gen" => Ok(IntervalDistribution::Generalized),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown distribution",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub(crate) enum CensorType {
    Exact,
    RightCensored,
    LeftCensored,
    IntervalCensored,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct IntervalCensoredResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub scale: f64,
    #[pyo3(get)]
    pub shape: f64,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub survival_prob: Vec<f64>,
}

fn weibull_cdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    1.0 - (-(t / scale).powf(shape)).exp()
}

fn weibull_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    (shape / scale) * (t / scale).powf(shape - 1.0) * (-(t / scale).powf(shape)).exp()
}

fn lognormal_cdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn lognormal_pdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let z = (t.ln() - mu) / sigma;
    (-0.5 * z * z).exp() / (t * sigma * (2.0 * std::f64::consts::PI).sqrt())
}

fn loglogistic_cdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    z / (1.0 + z)
}

fn loglogistic_pdf(t: f64, scale: f64, shape: f64) -> f64 {
    if t <= 0.0 || scale <= 0.0 || shape <= 0.0 {
        return 0.0;
    }
    let z = (t / scale).powf(shape);
    (shape / scale) * (t / scale).powf(shape - 1.0) / (1.0 + z).powi(2)
}

fn compute_interval_likelihood(
    left: f64,
    right: f64,
    censor_type: CensorType,
    scale: f64,
    shape: f64,
    distribution: &IntervalDistribution,
) -> f64 {
    let (cdf_fn, pdf_fn): DistributionFns = match distribution {
        IntervalDistribution::Weibull => (weibull_cdf, weibull_pdf),
        IntervalDistribution::LogNormal => (lognormal_cdf, lognormal_pdf),
        IntervalDistribution::LogLogistic => (loglogistic_cdf, loglogistic_pdf),
        IntervalDistribution::Exponential => (
            |t, s, _| weibull_cdf(t, s, 1.0),
            |t, s, _| weibull_pdf(t, s, 1.0),
        ),
        IntervalDistribution::Generalized => (weibull_cdf, weibull_pdf),
    };

    match censor_type {
        CensorType::Exact => {
            let f = pdf_fn(left, scale, shape);
            f.max(1e-300).ln()
        }
        CensorType::RightCensored => {
            let s = 1.0 - cdf_fn(left, scale, shape);
            s.max(1e-300).ln()
        }
        CensorType::LeftCensored => {
            let f = cdf_fn(right, scale, shape);
            f.max(1e-300).ln()
        }
        CensorType::IntervalCensored => {
            let f_right = cdf_fn(right, scale, shape);
            let f_left = cdf_fn(left, scale, shape);
            let diff = (f_right - f_left).max(1e-300);
            diff.ln()
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    left,
    right,
    censor_type,
    x,
    n_obs,
    n_vars,
    distribution,
    max_iter=500,
    tol=1e-6
))]
#[allow(clippy::too_many_arguments)]
pub fn interval_censored_regression(
    left: Vec<f64>,
    right: Vec<f64>,
    censor_type: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    distribution: &IntervalDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<IntervalCensoredResult> {
    if left.len() != n_obs || right.len() != n_obs || censor_type.len() != n_obs {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array lengths must match n_obs",
        ));
    }
    if x.len() != n_obs * n_vars {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "x length must equal n_obs * n_vars",
        ));
    }

    let censor_types: Vec<CensorType> = censor_type
        .iter()
        .map(|&c| match c {
            0 => CensorType::Exact,
            1 => CensorType::RightCensored,
            2 => CensorType::LeftCensored,
            _ => CensorType::IntervalCensored,
        })
        .collect();

    let mean_time: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| {
            if l > 0.0 && r > l {
                (l + r) / 2.0
            } else if l > 0.0 {
                l
            } else {
                r
            }
        })
        .sum::<f64>()
        / n_obs as f64;

    let mut beta = vec![0.0; n_vars];
    let mut scale = mean_time.max(0.01);
    let mut shape = 1.0;

    let mut prev_loglik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut loglik = 0.0;
        let mut gradient_beta = vec![0.0; n_vars];
        let mut gradient_scale = 0.0;
        let mut gradient_shape = 0.0;

        for i in 0..n_obs {
            let mut eta = 0.0;
            for j in 0..n_vars {
                eta += x[i * n_vars + j] * beta[j];
            }
            let scale_i = scale * eta.exp();

            let contrib = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i,
                shape,
                distribution,
            );
            loglik += contrib;

            let eps = 1e-6;
            for j in 0..n_vars {
                let mut beta_plus = beta.clone();
                beta_plus[j] += eps;
                let eta_plus = {
                    let mut e = 0.0;
                    for k in 0..n_vars {
                        e += x[i * n_vars + k] * beta_plus[k];
                    }
                    e
                };
                let scale_i_plus = scale * eta_plus.exp();
                let contrib_plus = compute_interval_likelihood(
                    left[i],
                    right[i],
                    censor_types[i],
                    scale_i_plus,
                    shape,
                    distribution,
                );
                gradient_beta[j] += (contrib_plus - contrib) / eps;
            }

            let scale_plus = scale + eps;
            let scale_i_plus = scale_plus * eta.exp();
            let contrib_scale_plus = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i_plus,
                shape,
                distribution,
            );
            gradient_scale += (contrib_scale_plus - contrib) / eps;

            let shape_plus = shape + eps;
            let contrib_shape_plus = compute_interval_likelihood(
                left[i],
                right[i],
                censor_types[i],
                scale_i,
                shape_plus,
                distribution,
            );
            gradient_shape += (contrib_shape_plus - contrib) / eps;
        }

        let step_size = 0.01;
        for j in 0..n_vars {
            beta[j] += step_size * gradient_beta[j];
        }
        scale = (scale + step_size * gradient_scale).max(0.001);
        shape = (shape + step_size * gradient_shape).max(0.01);

        if (loglik - prev_loglik).abs() < tol {
            converged = true;
            break;
        }
        prev_loglik = loglik;
    }

    let std_errors = vec![0.1; n_vars];

    let survival_prob: Vec<f64> = (0..n_obs)
        .map(|i| {
            let mut eta = 0.0;
            for j in 0..n_vars {
                eta += x[i * n_vars + j] * beta[j];
            }
            let scale_i = scale * eta.exp();
            let t = (left[i] + right[i].min(left[i] * 10.0)) / 2.0;
            match distribution {
                IntervalDistribution::Weibull => 1.0 - weibull_cdf(t, scale_i, shape),
                IntervalDistribution::LogNormal => 1.0 - lognormal_cdf(t, scale_i, shape),
                IntervalDistribution::LogLogistic => 1.0 - loglogistic_cdf(t, scale_i, shape),
                _ => 1.0 - weibull_cdf(t, scale_i, shape),
            }
        })
        .collect();

    let n_params = n_vars + 2;
    let aic = -2.0 * prev_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * prev_loglik + (n_params as f64) * (n_obs as f64).ln();

    Ok(IntervalCensoredResult {
        coefficients: beta,
        std_errors,
        scale,
        shape,
        log_likelihood: prev_loglik,
        aic,
        bic,
        n_iter,
        converged,
        survival_prob,
    })
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct TurnbullResult {
    #[pyo3(get)]
    pub time_points: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GroupedTurnbullResult {
    #[pyo3(get)]
    pub groups: Vec<i32>,
    #[pyo3(get)]
    pub time_points: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_iter: Vec<usize>,
    #[pyo3(get)]
    pub converged: Vec<bool>,
}

impl GroupedTurnbullResult {
    fn from_curves(curves: Vec<(i32, TurnbullResult)>) -> Self {
        let curve_count = curves.len();
        let mut output = Self {
            groups: Vec::with_capacity(curve_count),
            time_points: Vec::with_capacity(curve_count),
            survival: Vec::with_capacity(curve_count),
            survival_lower: Vec::with_capacity(curve_count),
            survival_upper: Vec::with_capacity(curve_count),
            n_iter: Vec::with_capacity(curve_count),
            converged: Vec::with_capacity(curve_count),
        };
        for (group, curve) in curves {
            output.groups.push(group);
            output.time_points.push(curve.time_points);
            output.survival.push(curve.survival);
            output.survival_lower.push(curve.survival_lower);
            output.survival_upper.push(curve.survival_upper);
            output.n_iter.push(curve.n_iter);
            output.converged.push(curve.converged);
        }
        output
    }
}

#[inline]
fn turnbull_case_weight(weights: Option<&[f64]>, index: usize) -> f64 {
    weights.map_or(1.0, |values| values[index])
}

#[inline]
fn turnbull_support_range(all_points: &[f64], left: f64, right: f64) -> (usize, usize) {
    if left.is_nan() || right.is_nan() {
        return (0, 0);
    }
    let start = all_points.partition_point(|&time| time < left);
    let end = if right == f64::INFINITY {
        all_points.len()
    } else {
        all_points.partition_point(|&time| time <= right)
    };
    (start, end.max(start))
}

fn validate_turnbull_inputs(left: &[f64], right: &[f64], weights: Option<&[f64]>) -> PyResult<()> {
    let n = left.len();
    if right.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "left and right must have same length",
        ));
    }
    let weights_ref = weights;
    if let Some(values) = weights_ref {
        if values.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must have same length as left and right",
            ));
        }
        let mut has_positive = false;
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights contains non-finite value at index {}",
                    idx
                )));
            }
            if value < 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "weights must be non-negative; got {} at index {}",
                    value, idx
                )));
            }
            has_positive |= value > 0.0;
        }
        if !has_positive {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weights must include at least one positive value",
            ));
        }
    }
    Ok(())
}

fn compute_turnbull_estimator(
    left: &[f64],
    right: &[f64],
    max_iter: usize,
    tol: f64,
    weights: Option<&[f64]>,
) -> TurnbullResult {
    let n = left.len();
    let total_weight = weights.map_or(n as f64, |values| values.iter().sum());

    let mut all_points: Vec<f64> = Vec::new();
    for i in 0..n {
        if left[i] > 0.0 {
            all_points.push(left[i]);
        }
        if right[i] < f64::INFINITY && right[i] > left[i] {
            all_points.push(right[i]);
        }
    }
    all_points.sort_by(f64::total_cmp);
    all_points.dedup();

    if all_points.is_empty() {
        return TurnbullResult {
            time_points: vec![],
            survival: vec![],
            survival_lower: vec![],
            survival_upper: vec![],
            n_iter: 0,
            converged: true,
        };
    }

    let m = all_points.len();
    let support_ranges: Vec<(usize, usize)> = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| turnbull_support_range(&all_points, left, right))
        .collect();
    let mut p = vec![1.0 / m as f64; m];

    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let p_old = p.clone();

        let mut p_new = vec![0.0; m];

        for (i, &(start, end)) in support_ranges.iter().enumerate() {
            let case_weight = turnbull_case_weight(weights, i);
            if case_weight == 0.0 {
                continue;
            }
            let mut sum_p = 0.0;
            for &probability in &p[start..end] {
                sum_p += probability;
            }

            if sum_p > 0.0 {
                for j in start..end {
                    let w = p[j] / sum_p;
                    p_new[j] += case_weight * w;
                }
            }
        }

        let total: f64 = p_new.iter().sum();
        if total > 0.0 {
            for j in 0..m {
                p[j] = p_new[j] / total;
            }
        }

        let max_diff: f64 = p
            .iter()
            .zip(p_old.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0, f64::max);

        if max_diff < tol {
            converged = true;
            break;
        }
    }

    let mut survival = Vec::with_capacity(m);
    let mut cum_prob = 0.0;
    for &prob in &p {
        cum_prob += prob;
        survival.push((1.0 - cum_prob).clamp(0.0, 1.0));
    }

    let se: Vec<f64> = p
        .iter()
        .map(|&prob| (prob * (1.0 - prob) / total_weight).sqrt())
        .collect();

    let (survival_lower, survival_upper) = clamped_normal_ci_bounds_95(&survival, &se, 0.0, 1.0);

    TurnbullResult {
        time_points: all_points,
        survival,
        survival_lower,
        survival_upper,
        n_iter,
        converged,
    }
}

#[pyfunction]
#[pyo3(signature = (left, right, max_iter=1000, tol=1e-6, weights=None))]
pub fn turnbull_estimator(
    left: Vec<f64>,
    right: Vec<f64>,
    max_iter: usize,
    tol: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<TurnbullResult> {
    validate_turnbull_inputs(&left, &right, weights.as_deref())?;
    Ok(compute_turnbull_estimator(
        &left,
        &right,
        max_iter,
        tol,
        weights.as_deref(),
    ))
}

fn compute_grouped_turnbull(
    left: &[f64],
    right: &[f64],
    groups: &[i32],
    weights: &[f64],
    max_iter: usize,
    tol: f64,
) -> GroupedTurnbullResult {
    let mut indices_by_group: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, &group) in groups.iter().enumerate() {
        indices_by_group.entry(group).or_default().push(idx);
    }
    let grouped_indices: Vec<(i32, Vec<usize>)> = indices_by_group.into_iter().collect();
    let compute = |(group, indices): &(i32, Vec<usize>)| {
        let group_left: Vec<f64> = indices.iter().map(|&idx| left[idx]).collect();
        let group_right: Vec<f64> = indices.iter().map(|&idx| right[idx]).collect();
        let group_weights: Vec<f64> = indices.iter().map(|&idx| weights[idx]).collect();
        (
            *group,
            compute_turnbull_estimator(
                &group_left,
                &group_right,
                max_iter,
                tol,
                Some(&group_weights),
            ),
        )
    };
    let curves = if left.len() >= PARALLEL_THRESHOLD_XLARGE && grouped_indices.len() > 1 {
        grouped_indices.par_iter().map(compute).collect()
    } else {
        grouped_indices.iter().map(compute).collect()
    };
    GroupedTurnbullResult::from_curves(curves)
}

#[pyfunction]
#[pyo3(signature = (left, right, groups, max_iter=1000, tol=1e-6, weights=None))]
pub fn turnbull_estimator_grouped(
    py: Python<'_>,
    left: Vec<f64>,
    right: Vec<f64>,
    groups: Vec<i32>,
    max_iter: usize,
    tol: f64,
    weights: Option<Vec<f64>>,
) -> PyResult<GroupedTurnbullResult> {
    validate_turnbull_inputs(&left, &right, weights.as_deref())?;
    if groups.len() != left.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "groups must have same length as left and right",
        ));
    }
    let weights = weights.unwrap_or_else(|| vec![1.0; left.len()]);
    let mut group_has_positive_weight = BTreeMap::new();
    for (&group, &weight) in groups.iter().zip(&weights) {
        group_has_positive_weight
            .entry(group)
            .and_modify(|has_positive| *has_positive |= weight > 0.0)
            .or_insert(weight > 0.0);
    }
    if group_has_positive_weight.values().any(|value| !value) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "weights must include at least one positive value per group",
        ));
    }

    Ok(
        py.detach(move || {
            compute_grouped_turnbull(&left, &right, &groups, &weights, max_iter, tol)
        }),
    )
}

#[pyfunction]
pub fn npmle_interval(
    left: Vec<f64>,
    right: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<TimeSurvivalCurve> {
    turnbull_estimator(left, right, 1000, 1e-6, weights)
        .map(|result| (result.time_points, result.survival))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_cdf() {
        assert!((weibull_cdf(0.0, 1.0, 1.0) - 0.0).abs() < 1e-10);
        let cdf_5 = weibull_cdf(5.0, 3.0, 2.0);
        assert!(cdf_5 > 0.0 && cdf_5 < 1.0);
    }

    #[test]
    fn test_turnbull_basic() {
        let left = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let right = vec![2.0, 3.0, 5.0, 4.0, f64::INFINITY];

        let result = turnbull_estimator(left, right, 100, 1e-4, None).unwrap();
        assert!(!result.time_points.is_empty());
        assert!(result.survival.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_turnbull_support_range_matches_interval_membership() {
        let points = vec![-1.0, 1.0, 2.0, 3.0, 5.0, f64::INFINITY];
        let cases = [
            (0.0, 1.0),
            (2.0, 3.0),
            (3.0, f64::INFINITY),
            (4.0, 2.0),
            (f64::NEG_INFINITY, f64::INFINITY),
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            (f64::NAN, f64::INFINITY),
            (0.0, f64::NAN),
        ];

        for (left, right) in cases {
            let (start, end) = turnbull_support_range(&points, left, right);
            let actual: Vec<usize> = (start..end).collect();
            let expected: Vec<usize> = points
                .iter()
                .enumerate()
                .filter_map(|(idx, &time)| {
                    (time >= left && (right == f64::INFINITY || time <= right)).then_some(idx)
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_turnbull_unweighted_matches_unit_weights() {
        let left = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let right = vec![2.0, 3.0, 5.0, 4.0, f64::INFINITY];

        let unweighted = turnbull_estimator(left.clone(), right.clone(), 100, 1e-8, None).unwrap();
        let unit_weighted = turnbull_estimator(left, right, 100, 1e-8, Some(vec![1.0; 5])).unwrap();

        assert_eq!(unweighted.time_points, unit_weighted.time_points);
        assert_eq!(unweighted.survival, unit_weighted.survival);
        assert_eq!(unweighted.survival_lower, unit_weighted.survival_lower);
        assert_eq!(unweighted.survival_upper, unit_weighted.survival_upper);
        assert_eq!(unweighted.n_iter, unit_weighted.n_iter);
        assert_eq!(unweighted.converged, unit_weighted.converged);
    }

    #[test]
    fn test_turnbull_weights_match_replicated_rows() {
        let left = vec![0.0, 1.0, 2.0];
        let right = vec![1.0, 3.0, f64::INFINITY];
        let weights = vec![2.0, 1.0, 3.0];

        let weighted =
            turnbull_estimator(left.clone(), right.clone(), 100, 1e-8, Some(weights)).unwrap();

        let replicated_left = vec![0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        let replicated_right = vec![1.0, 1.0, 3.0, f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let replicated =
            turnbull_estimator(replicated_left, replicated_right, 100, 1e-8, None).unwrap();

        assert_eq!(weighted.time_points, replicated.time_points);
        for (actual, expected) in weighted.survival.iter().zip(replicated.survival.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_grouped_turnbull_matches_individual_weighted_curves() {
        let left = vec![0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 4.0, 3.0];
        let right = vec![1.0, 3.0, f64::INFINITY, 2.0, 2.0, 5.0, 4.0, f64::INFINITY];
        let groups = vec![7, 3, 7, 3, 7, 3, 7, 3];
        let weights = vec![1.0, 0.5, 1.5, 2.0, 0.75, 1.25, 2.5, 1.0];

        let grouped = compute_grouped_turnbull(&left, &right, &groups, &weights, 1000, 1e-6);
        assert_eq!(grouped.groups, vec![3, 7]);

        for (curve_idx, &group) in grouped.groups.iter().enumerate() {
            let indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter_map(|(idx, &value)| (value == group).then_some(idx))
                .collect();
            let group_left: Vec<f64> = indices.iter().map(|&idx| left[idx]).collect();
            let group_right: Vec<f64> = indices.iter().map(|&idx| right[idx]).collect();
            let group_weights: Vec<f64> = indices.iter().map(|&idx| weights[idx]).collect();
            let expected = compute_turnbull_estimator(
                &group_left,
                &group_right,
                1000,
                1e-6,
                Some(&group_weights),
            );

            assert_eq!(grouped.time_points[curve_idx], expected.time_points);
            assert_eq!(grouped.survival[curve_idx], expected.survival);
            assert_eq!(grouped.survival_lower[curve_idx], expected.survival_lower);
            assert_eq!(grouped.survival_upper[curve_idx], expected.survival_upper);
            assert_eq!(grouped.n_iter[curve_idx], expected.n_iter);
            assert_eq!(grouped.converged[curve_idx], expected.converged);
        }
    }

    #[test]
    fn test_interval_regression_basic() {
        let left = vec![1.0, 2.0, 3.0, 4.0];
        let right = vec![2.0, 3.0, 5.0, 6.0];
        let censor_type = vec![3, 3, 3, 3];
        let x = vec![1.0, 0.5, 0.0, 1.0];

        let result = interval_censored_regression(
            left,
            right,
            censor_type,
            x,
            4,
            1,
            &IntervalDistribution::Weibull,
            100,
            1e-4,
        )
        .unwrap();

        assert_eq!(result.coefficients.len(), 1);
        assert!(result.scale > 0.0);
        assert!(result.shape > 0.0);
    }
}
