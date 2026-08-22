use crate::constants::PARALLEL_THRESHOLD_XLARGE;
use crate::internal::statistical::erf;
use crate::surv_analysis::{
    KaplanMeierConfig, SurvFitKMOutput, compute_robust_survfitkm_with_timefix, compute_survfitkm,
};
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
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub survival: Vec<f64>,
    #[pyo3(get)]
    pub std_err: Vec<f64>,
    #[pyo3(get)]
    pub survival_lower: Vec<f64>,
    #[pyo3(get)]
    pub survival_upper: Vec<f64>,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub logse: bool,
    #[pyo3(get)]
    pub conf_level: f64,
    #[pyo3(get)]
    pub conf_type: String,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct GroupedTurnbullResult {
    #[pyo3(get)]
    pub groups: Vec<i32>,
    #[pyo3(get)]
    pub time_points: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_risk: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_event: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_censor: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub std_err: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_lower: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub survival_upper: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub n_iter: Vec<usize>,
    #[pyo3(get)]
    pub converged: Vec<bool>,
    #[pyo3(get)]
    pub logse: Vec<bool>,
    #[pyo3(get)]
    pub conf_level: f64,
    #[pyo3(get)]
    pub conf_type: String,
}

impl GroupedTurnbullResult {
    fn from_curves(curves: Vec<(i32, TurnbullResult)>) -> Self {
        let curve_count = curves.len();
        let mut output = Self {
            groups: Vec::with_capacity(curve_count),
            time_points: Vec::with_capacity(curve_count),
            n_risk: Vec::with_capacity(curve_count),
            n_event: Vec::with_capacity(curve_count),
            n_censor: Vec::with_capacity(curve_count),
            survival: Vec::with_capacity(curve_count),
            std_err: Vec::with_capacity(curve_count),
            survival_lower: Vec::with_capacity(curve_count),
            survival_upper: Vec::with_capacity(curve_count),
            n_iter: Vec::with_capacity(curve_count),
            converged: Vec::with_capacity(curve_count),
            logse: Vec::with_capacity(curve_count),
            conf_level: curves.first().map_or(0.95, |(_, curve)| curve.conf_level),
            conf_type: curves
                .first()
                .map_or_else(|| "log".to_string(), |(_, curve)| curve.conf_type.clone()),
        };
        for (group, curve) in curves {
            output.groups.push(group);
            output.time_points.push(curve.time_points);
            output.n_risk.push(curve.n_risk);
            output.n_event.push(curve.n_event);
            output.n_censor.push(curve.n_censor);
            output.survival.push(curve.survival);
            output.std_err.push(curve.std_err);
            output.survival_lower.push(curve.survival_lower);
            output.survival_upper.push(curve.survival_upper);
            output.n_iter.push(curve.n_iter);
            output.converged.push(curve.converged);
            output.logse.push(curve.logse);
        }
        output
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnbullCensoring {
    Right,
    Exact,
    Left,
    Interval,
}

#[derive(Clone, Copy, Debug)]
struct TurnbullObservation {
    left: f64,
    right: f64,
    censoring: TurnbullCensoring,
    weight: f64,
}

fn turnbull_censoring(left: f64, right: f64) -> TurnbullCensoring {
    if left == right {
        TurnbullCensoring::Exact
    } else if right == f64::INFINITY {
        TurnbullCensoring::Right
    } else if left == f64::NEG_INFINITY {
        TurnbullCensoring::Left
    } else {
        TurnbullCensoring::Interval
    }
}

fn validate_turnbull_inputs(left: &[f64], right: &[f64], weights: Option<&[f64]>) -> PyResult<()> {
    let n = left.len();
    if right.len() != n {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "left and right must have same length",
        ));
    }
    for (idx, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
        if left_value.is_nan() || right_value.is_nan() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "left and right must not contain NaN; invalid interval at index {}",
                idx
            )));
        }
        if left_value == f64::INFINITY
            || right_value == f64::NEG_INFINITY
            || left_value > right_value
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "left must be less than or equal to right at index {}",
                idx
            )));
        }
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

fn turnbull_support_points(observations: &[TurnbullObservation]) -> Vec<f64> {
    let mut endpoints = Vec::with_capacity(observations.len() * 2);
    let mut exact = Vec::new();
    for observation in observations {
        match observation.censoring {
            TurnbullCensoring::Exact => {
                endpoints.push((observation.left, 0_u8));
                exact.push(observation.left);
            }
            TurnbullCensoring::Left => endpoints.push((observation.right, 1_u8)),
            TurnbullCensoring::Right => endpoints.push((observation.left, 2_u8)),
            TurnbullCensoring::Interval => {
                endpoints.push((observation.left, 2_u8));
                endpoints.push((observation.right, 1_u8));
            }
        }
    }
    endpoints.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    for pair in endpoints.windows(2) {
        if pair[0].1 != 1 && pair[1].1 == 1 {
            exact.push(pair[0].0 + (pair[1].0 - pair[0].0) / 2.0);
        }
    }
    exact.sort_by(f64::total_cmp);
    exact.dedup();
    exact
}

fn turnbull_support_ranges(
    observations: &[TurnbullObservation],
    support: &[f64],
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let mut ranges = Vec::new();
    let mut weights = Vec::new();
    for observation in observations {
        let range = match observation.censoring {
            TurnbullCensoring::Left => Some((
                0,
                support.partition_point(|&time| time <= observation.right),
            )),
            TurnbullCensoring::Interval => Some((
                support.partition_point(|&time| time <= observation.left),
                support.partition_point(|&time| time <= observation.right),
            )),
            TurnbullCensoring::Right | TurnbullCensoring::Exact => None,
        };
        if let Some((start, end)) = range {
            ranges.push((start, end));
            weights.push(observation.weight);
        }
    }
    if ranges.is_empty() && !support.is_empty() {
        ranges.push((0, support.len()));
        weights.push(1.0);
    }
    (ranges, weights)
}

fn turnbull_range_totals(values: &[f64], ranges: &[(usize, usize)]) -> Vec<f64> {
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    for value in values {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value);
    }
    ranges
        .iter()
        .map(|&(start, end)| prefix[end] - prefix[start])
        .collect()
}

fn turnbull_initial_mass(support_len: usize, ranges: &[(usize, usize)]) -> Vec<f64> {
    let mut difference = vec![0.0; support_len + 1];
    for &(start, end) in ranges {
        difference[start] += 1.0;
        difference[end] -= 1.0;
    }
    let mut active = 0.0;
    let mut mass = Vec::with_capacity(support_len);
    for delta in difference.into_iter().take(support_len) {
        active += delta;
        mass.push(active);
    }
    let total: f64 = mass.iter().sum();
    if total > 0.0 {
        for value in &mut mass {
            *value /= total;
        }
    }
    mass
}

fn turnbull_redistributed_weights(
    jumps: &[f64],
    ranges: &[(usize, usize)],
    case_weights: &[f64],
) -> Vec<f64> {
    let totals = turnbull_range_totals(jumps, ranges);
    let mut difference = vec![0.0; jumps.len() + 1];
    for ((&(start, end), &case_weight), &total) in ranges.iter().zip(case_weights).zip(&totals) {
        if total > 0.0 {
            let contribution = case_weight / total;
            difference[start] += contribution;
            difference[end] -= contribution;
        }
    }
    let mut active = 0.0;
    jumps
        .iter()
        .zip(difference)
        .map(|(&jump, delta)| {
            active += delta;
            jump * active
        })
        .collect()
}

fn turnbull_loglik(jumps: &[f64], ranges: &[(usize, usize)]) -> f64 {
    turnbull_range_totals(jumps, ranges)
        .into_iter()
        .try_fold(0.0, |total, value| {
            (value > 0.0 && value.is_finite()).then(|| total + value.ln())
        })
        .unwrap_or(f64::NEG_INFINITY)
}

fn turnbull_artificial_data(
    observations: &[TurnbullObservation],
    support: &[f64],
    redistributed: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let real_count = observations
        .iter()
        .filter(|observation| {
            matches!(
                observation.censoring,
                TurnbullCensoring::Right | TurnbullCensoring::Exact
            )
        })
        .count();
    let mut time = Vec::with_capacity(real_count + support.len());
    let mut status = Vec::with_capacity(real_count + support.len());
    let mut weights = Vec::with_capacity(real_count + support.len());
    for observation in observations {
        match observation.censoring {
            TurnbullCensoring::Right => {
                time.push(observation.left);
                status.push(0.0);
                weights.push(observation.weight);
            }
            TurnbullCensoring::Exact => {
                time.push(observation.left);
                status.push(1.0);
                weights.push(observation.weight);
            }
            TurnbullCensoring::Left | TurnbullCensoring::Interval => {}
        }
    }
    time.extend_from_slice(support);
    status.extend(std::iter::repeat_n(1.0, support.len()));
    weights.extend_from_slice(redistributed);
    (time, status, weights)
}

fn turnbull_survival_at_support(curve: &SurvFitKMOutput, support: &[f64]) -> Vec<f64> {
    let mut output = Vec::with_capacity(support.len());
    let mut curve_idx = 0;
    let mut survival = 1.0;
    for &time in support {
        while curve_idx < curve.time.len() && curve.time[curve_idx] <= time {
            survival = curve.estimate[curve_idx];
            curve_idx += 1;
        }
        output.push(survival);
    }
    output
}

fn turnbull_final_curve(
    observations: &[TurnbullObservation],
    support: &[f64],
    redistributed: &[f64],
    robust: bool,
    config: &KaplanMeierConfig,
) -> SurvFitKMOutput {
    let (time, status, weights) = turnbull_artificial_data(observations, support, redistributed);
    if robust {
        let clusters: Vec<i32> = (0..time.len()).map(|index| index as i32).collect();
        compute_robust_survfitkm_with_timefix(&time, &status, &weights, &clusters, config, true)
    } else {
        compute_survfitkm(&time, &status, &weights, None, &vec![0; time.len()], config)
    }
}

fn compute_turnbull_estimator(
    left: &[f64],
    right: &[f64],
    max_iter: usize,
    tol: f64,
    weights: Option<&[f64]>,
    robust: bool,
    config: &KaplanMeierConfig,
) -> TurnbullResult {
    let mut observations: Vec<TurnbullObservation> = left
        .iter()
        .zip(right)
        .enumerate()
        .map(|(index, (&left, &right))| TurnbullObservation {
            left,
            right,
            censoring: turnbull_censoring(left, right),
            weight: weights.map_or(1.0, |values| values[index]),
        })
        .collect();
    let mut support = turnbull_support_points(&observations);
    let original_minimum = support.first().copied().unwrap_or(f64::INFINITY);
    for observation in &mut observations {
        if observation.censoring == TurnbullCensoring::Left && observation.right < original_minimum
        {
            observation.left = observation.right;
            observation.censoring = TurnbullCensoring::Exact;
            support.push(observation.right);
        }
    }
    support.sort_by(f64::total_cmp);
    support.dedup();

    if support.is_empty() {
        return TurnbullResult {
            time_points: vec![],
            n_risk: vec![],
            n_event: vec![],
            n_censor: vec![],
            survival: vec![],
            std_err: vec![],
            survival_lower: vec![],
            survival_upper: vec![],
            n_iter: 0,
            converged: true,
            logse: !robust,
            conf_level: config.conf_level,
            conf_type: config.conf_type.clone(),
        };
    }
    let (ranges, interval_weights) = turnbull_support_ranges(&observations, &support);
    let initial_mass = turnbull_initial_mass(support.len(), &ranges);
    let mut old_survival = Vec::with_capacity(support.len());
    let mut cumulative = 0.0;
    for &mass in &initial_mass {
        cumulative += mass;
        old_survival.push((1.0 - cumulative).max(0.0));
    }
    let mut current_survival = old_survival.clone();
    let mut redistributed = initial_mass.clone();
    let mut converged = false;
    let mut n_iter = 0;
    let mut jump1 = vec![0.0; support.len()];
    let mut jump2 = vec![0.0; support.len()];
    let mut aitken1 = vec![0.0; support.len()];
    let iteration_config = KaplanMeierConfig {
        conf_type: "none".to_string(),
        ..config.clone()
    };

    for iteration in 1..=max_iter {
        n_iter = iteration;
        let mut jumps = Vec::with_capacity(support.len());
        let mut previous = 1.0;
        for &survival in &current_survival {
            jumps.push((previous - survival).max(0.0));
            previous = survival;
        }

        let aitken2 = aitken1.clone();
        for index in 0..jumps.len() {
            aitken1[index] = jumps[index] - jump1[index];
        }
        let saved_jumps = jumps.clone();
        if iteration % 5 == 0 {
            let old_loglik = turnbull_loglik(&jumps, &ranges);
            for index in 0..jumps.len() {
                let denominator = aitken1[index] - aitken2[index];
                let candidate = jump2[index] - aitken2[index] * aitken2[index] / denominator;
                jumps[index] =
                    if candidate > 8.0 * f64::EPSILON && candidate < 1.0 && candidate.is_finite() {
                        candidate
                    } else {
                        saved_jumps[index]
                    };
            }
            if turnbull_loglik(&jumps, &ranges) < old_loglik {
                jumps.clone_from(&saved_jumps);
            }
        }
        jump2 = jump1;
        jump1 = saved_jumps;

        redistributed = turnbull_redistributed_weights(&jumps, &ranges, &interval_weights);
        let (time, status, artificial_weights) =
            turnbull_artificial_data(&observations, &support, &redistributed);
        let curve = compute_survfitkm(
            &time,
            &status,
            &artificial_weights,
            None,
            &vec![0; time.len()],
            &iteration_config,
        );
        current_survival = turnbull_survival_at_support(&curve, &support);
        if iteration % 5 >= 2 {
            let difference = old_survival
                .iter()
                .zip(&current_survival)
                .map(|(&old, &new)| (old - new).abs())
                .fold(0.0, f64::max);
            if difference <= tol {
                converged = true;
                break;
            }
        }
        old_survival.clone_from(&current_survival);
    }
    let mut curve = turnbull_final_curve(&observations, &support, &redistributed, robust, config);
    for index in 0..curve.time.len() {
        if curve.time[index] < original_minimum && curve.n_event[index] > 0.0 {
            curve.n_event[index] = 0.0;
        }
        if curve.estimate[index] <= 0.0 {
            curve.estimate[index] = 0.0;
            curve.std_err[index] = 0.0;
            if !curve.conf_lower.is_empty() {
                curve.conf_lower[index] = 0.0;
                curve.conf_upper[index] = 0.0;
            }
        }
    }
    let std_err = if robust {
        curve.std_err
    } else {
        curve
            .std_err
            .iter()
            .zip(&curve.estimate)
            .map(|(&standard_error, &survival)| {
                if survival > 0.0 {
                    standard_error / survival
                } else {
                    standard_error
                }
            })
            .collect()
    };

    TurnbullResult {
        time_points: curve.time,
        n_risk: curve.n_risk,
        n_event: curve.n_event,
        n_censor: curve.n_censor,
        survival: curve.estimate,
        std_err,
        survival_lower: curve.conf_lower,
        survival_upper: curve.conf_upper,
        n_iter,
        converged,
        logse: !robust,
        conf_level: config.conf_level,
        conf_type: config.conf_type.clone(),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (left, right, max_iter=1000, tol=5e-5, weights=None, robust=true, conf_level=0.95, conf_type="log".to_string()))]
pub fn turnbull_estimator(
    left: Vec<f64>,
    right: Vec<f64>,
    max_iter: usize,
    tol: f64,
    weights: Option<Vec<f64>>,
    robust: bool,
    conf_level: f64,
    conf_type: String,
) -> PyResult<TurnbullResult> {
    validate_turnbull_inputs(&left, &right, weights.as_deref())?;
    if max_iter == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_iter must be positive",
        ));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tol must be finite and positive",
        ));
    }
    let config =
        KaplanMeierConfig::create(Some(false), Some(0), Some(conf_level), Some(conf_type))?;
    Ok(compute_turnbull_estimator(
        &left,
        &right,
        max_iter,
        tol,
        weights.as_deref(),
        robust,
        &config,
    ))
}

#[allow(clippy::too_many_arguments)]
fn compute_grouped_turnbull(
    left: &[f64],
    right: &[f64],
    groups: &[i32],
    weights: &[f64],
    max_iter: usize,
    tol: f64,
    robust: bool,
    config: &KaplanMeierConfig,
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
                robust,
                config,
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
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (left, right, groups, max_iter=1000, tol=5e-5, weights=None, robust=true, conf_level=0.95, conf_type="log".to_string()))]
pub fn turnbull_estimator_grouped(
    py: Python<'_>,
    left: Vec<f64>,
    right: Vec<f64>,
    groups: Vec<i32>,
    max_iter: usize,
    tol: f64,
    weights: Option<Vec<f64>>,
    robust: bool,
    conf_level: f64,
    conf_type: String,
) -> PyResult<GroupedTurnbullResult> {
    validate_turnbull_inputs(&left, &right, weights.as_deref())?;
    if groups.len() != left.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "groups must have same length as left and right",
        ));
    }
    if max_iter == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "max_iter must be positive",
        ));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tol must be finite and positive",
        ));
    }
    let config =
        KaplanMeierConfig::create(Some(false), Some(0), Some(conf_level), Some(conf_type))?;
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

    Ok(py.detach(move || {
        compute_grouped_turnbull(
            &left, &right, &groups, &weights, max_iter, tol, robust, &config,
        )
    }))
}

#[pyfunction]
pub fn npmle_interval(
    left: Vec<f64>,
    right: Vec<f64>,
    weights: Option<Vec<f64>>,
) -> PyResult<TimeSurvivalCurve> {
    turnbull_estimator(
        left,
        right,
        1000,
        5e-5,
        weights,
        true,
        0.95,
        "log".to_string(),
    )
    .map(|result| (result.time_points, result.survival))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turnbull_config() -> KaplanMeierConfig {
        KaplanMeierConfig::default()
    }

    fn fit_turnbull(
        left: Vec<f64>,
        right: Vec<f64>,
        max_iter: usize,
        tol: f64,
        weights: Option<Vec<f64>>,
    ) -> TurnbullResult {
        turnbull_estimator(
            left,
            right,
            max_iter,
            tol,
            weights,
            true,
            0.95,
            "log".to_string(),
        )
        .unwrap()
    }

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

        let result = fit_turnbull(left, right, 100, 1e-4, None);
        assert!(!result.time_points.is_empty());
        assert!(result.survival.iter().all(|&s| (0.0..=1.0).contains(&s)));
    }

    #[test]
    fn test_turnbull_matches_reference_weighted_interval_fit() {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for time in 1..=4 {
            let value = time as f64;
            left.extend([value, value, f64::NEG_INFINITY]);
            right.extend([value, f64::INFINITY, value]);
        }
        let weights = vec![12.0, 3.0, 2.0, 6.0, 2.0, 4.0, 2.0, 0.0, 2.0, 3.0, 3.0, 5.0];
        let result = fit_turnbull(left, right, 1000, 5e-5, Some(weights));

        assert_eq!(result.time_points, vec![1.0, 2.0, 3.0, 4.0]);
        for (actual, expected) in result.survival.iter().zip([
            0.537567714669201,
            0.294594032546376,
            0.209760214997243,
            0.0948457531854178,
        ]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in result.std_err.iter().zip([
            0.202635155653773,
            0.146753572309286,
            0.127982471861457,
            0.0915918887134962,
        ]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(result.n_censor, vec![3.0, 2.0, 0.0, 3.0]);
        assert!(result.converged);
    }

    #[test]
    fn test_turnbull_support_uses_exact_times_and_open_closed_midpoints() {
        let observations = vec![
            TurnbullObservation {
                left: 1.0,
                right: 1.0,
                censoring: TurnbullCensoring::Exact,
                weight: 1.0,
            },
            TurnbullObservation {
                left: 2.0,
                right: 4.0,
                censoring: TurnbullCensoring::Interval,
                weight: 1.0,
            },
            TurnbullObservation {
                left: 5.0,
                right: f64::INFINITY,
                censoring: TurnbullCensoring::Right,
                weight: 1.0,
            },
            TurnbullObservation {
                left: f64::NEG_INFINITY,
                right: 6.0,
                censoring: TurnbullCensoring::Left,
                weight: 1.0,
            },
        ];
        assert_eq!(turnbull_support_points(&observations), vec![1.0, 3.0, 5.5]);
    }

    #[test]
    fn test_turnbull_unweighted_matches_unit_weights() {
        let left = vec![1.0, 2.0, 3.0, 1.0, 2.0];
        let right = vec![2.0, 3.0, 5.0, 4.0, f64::INFINITY];

        let unweighted = fit_turnbull(left.clone(), right.clone(), 100, 1e-8, None);
        let unit_weighted = fit_turnbull(left, right, 100, 1e-8, Some(vec![1.0; 5]));

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

        let weighted = fit_turnbull(left.clone(), right.clone(), 100, 1e-8, Some(weights));

        let replicated_left = vec![0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        let replicated_right = vec![1.0, 1.0, 3.0, f64::INFINITY, f64::INFINITY, f64::INFINITY];
        let replicated = fit_turnbull(replicated_left, replicated_right, 100, 1e-8, None);

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

        let config = turnbull_config();
        let grouped =
            compute_grouped_turnbull(&left, &right, &groups, &weights, 1000, 1e-6, true, &config);
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
                true,
                &config,
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
