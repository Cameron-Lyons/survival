use super::turnbull_em::TurnbullWeights;
use crate::data_prep::aeq_surv;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::sorting::sorted_indices_by;
use crate::internal::validation::{validate_finite, validate_length, validate_non_negative};
use crate::regression::survreg_kernels::{BuiltinKernel, with_builtin_kernel};
use crate::regression::{SurvregControl, SurvregData, SurvregDistribution, survreg_fit};
use crate::surv_analysis::survfitkm::{rows_by_curve, select_items, strata_index};
use crate::surv_analysis::{
    ConfType, StackedCurves, SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm,
};
use ndarray::Array2;
use pyo3::prelude::*;
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object)]
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
#[pyclass(module = "survival._survival", from_py_object)]
pub(crate) enum CensorType {
    Exact,
    RightCensored,
    LeftCensored,
    IntervalCensored,
}

#[derive(Debug, Clone)]
#[pyclass(module = "survival._survival", from_py_object)]
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

/// Fit an accelerated failure time model with an implicit location intercept
/// and a flat row-major covariate matrix. Censor codes are 0 for exact, 1 for
/// right, 2 for left and 3 for interval. The baseline time scale is the
/// exponentiated intercept. Coefficients and standard errors contain only the
/// supplied covariates. Aliased slopes have a NaN coefficient and zero variance.
#[allow(clippy::too_many_arguments)]
pub fn interval_censored_regression(
    mut left: Vec<f64>,
    right: Vec<f64>,
    mut censor_type: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    distribution: &IntervalDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<IntervalCensoredResult> {
    if left.len() != n_obs || right.len() != n_obs || censor_type.len() != n_obs {
        return Err(SurvivalError::invalid_input("Input array lengths must match n_obs").into());
    }
    if n_obs == 0 {
        return Err(SurvivalError::invalid_input("n_obs must be positive").into());
    }
    if n_obs.checked_mul(n_vars) != Some(x.len()) {
        return Err(SurvivalError::invalid_input("x length must equal n_obs * n_vars").into());
    }

    // Translate to Surv's status codes and record the time for each fitted
    // survival probability before survreg applies its response transform.
    let mut prediction_times = Vec::with_capacity(n_obs);
    for (i, code) in censor_type.iter_mut().enumerate() {
        let time = match *code {
            0 => {
                *code = 1;
                left[i]
            }
            1 => {
                *code = 0;
                left[i]
            }
            2 => {
                left[i] = right[i];
                right[i]
            }
            3 => {
                if left[i] == right[i] {
                    return Err(SurvivalError::invalid_input(format!(
                        "interval endpoints must differ at index {i}; use code 0 for an exact observation"
                    ))
                    .into());
                }
                left[i].midpoint(right[i])
            }
            _ => {
                return Err(SurvivalError::invalid_input(format!(
                    "censor_type must contain codes 0, 1, 2 or 3; got {code} at index {i}"
                ))
                .into());
            }
        };
        prediction_times.push(time);
    }
    let design = Array2::from_shape_fn((n_obs, n_vars + 1), |(i, j)| {
        if j == 0 { 1.0 } else { x[i * n_vars + (j - 1)] }
    });
    drop(x);
    let name = match distribution {
        IntervalDistribution::Weibull | IntervalDistribution::Generalized => "weibull",
        IntervalDistribution::LogNormal => "lognormal",
        IntervalDistribution::LogLogistic => "loglogistic",
        IntervalDistribution::Exponential => "exponential",
    };
    let family = SurvregDistribution::from_name(name, None)?;
    let control = SurvregControl {
        iter_max: max_iter,
        rel_tolerance: tol,
        ..SurvregControl::default()
    };
    let data = SurvregData {
        time: left,
        time2: Some(right),
        status: censor_type,
        covariates: design,
        weights: None,
        offset: None,
        strata: None,
        cluster: None,
    };
    let fit = survreg_fit(&data, &family, None, 0.0, &control, false)?;
    let sigma = fit.scale[0];
    let coefficients = fit.coefficients[1..=n_vars].to_vec();
    let std_errors = (1..=n_vars)
        .map(|j| fit.variance_matrix[j][j].sqrt())
        .collect();
    let survival_prob = with_builtin_kernel!(&family, kernel => {
        prediction_times.iter().zip(&fit.linear_predictors).map(|(time, eta)| {
            let z = (time.ln() - eta) / sigma;
            kernel.censored(z)[1]
        }).collect()
    });

    Ok(IntervalCensoredResult {
        coefficients,
        std_errors,
        scale: fit.coefficients[0].exp(),
        shape: if *distribution == IntervalDistribution::LogNormal {
            sigma
        } else {
            sigma.recip()
        },
        log_likelihood: fit.log_likelihood,
        aic: -2.0 * fit.log_likelihood + 2.0 * fit.df,
        bic: -2.0 * fit.log_likelihood + (n_obs as f64).ln() * fit.df,
        n_iter: fit.iterations,
        converged: fit.converged,
        survival_prob,
    })
}

/// Fit a parametric accelerated failure time model with an implicit intercept.
/// Supply covariates as a flat row-major array. Censor codes 0 and 1 mean
/// exact and right-censored at left, 2 means left-censored at right, and 3
/// means interval-censored between left and right. survival_prob is evaluated
/// at that endpoint for codes 0–2 and at the interval midpoint for code 3.
///
/// scale is exp(intercept). shape is the standard deviation of log time for
/// lognormal, the Weibull or log-logistic shape for those families, and one
/// for exponential. Generalized uses the Weibull family.
#[cfg(feature = "python")]
#[pyfunction(name = "interval_censored_regression")]
#[pyo3(signature = (
    left, right, censor_type, x, n_obs, n_vars, distribution, max_iter=500, tol=1e-6
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn interval_censored_regression_py(
    py: Python<'_>,
    left: Vec<f64>,
    right: Vec<f64>,
    censor_type: Vec<i32>,
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    distribution: IntervalDistribution,
    max_iter: usize,
    tol: f64,
) -> PyResult<IntervalCensoredResult> {
    py.detach(|| {
        interval_censored_regression(
            left,
            right,
            censor_type,
            x,
            n_obs,
            n_vars,
            &distribution,
            max_iter,
            tol,
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalStatus {
    Right,
    Exact,
    Left,
    Interval,
}

impl IntervalStatus {
    fn from_code(code: i32) -> SurvivalResult<Self> {
        match code {
            0 => Ok(Self::Right),
            1 => Ok(Self::Exact),
            2 => Ok(Self::Left),
            3 => Ok(Self::Interval),
            other => Err(SurvivalError::invalid_input(format!(
                "interval status must be 0, 1, 2 or 3; got {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TurnbullInput<'a> {
    pub time1: &'a [f64],
    pub time2: &'a [f64],
    pub status: &'a [i32],
    pub weights: Option<&'a [f64]>,
    pub group: Option<&'a [i32]>,
    pub conf_level: f64,
    pub conf_type: &'a str,
    pub timefix: bool,
    pub se_fit: bool,
    pub robust: Option<bool>,
}

#[doc = " One curve of [`TurnbullResult`]: the fields of the `survfit` object."]
#[doc = " `std_err` is the one `survfitKM` reports for the final EM step, which"]
#[doc = " `survfitTurnbull` keeps as it is: the standard error of `surv` under the"]
#[doc = " robust (infinitesimal jackknife) variance and of `log(surv)` otherwise."]
#[doc = " `std_err` is empty without `se_fit`, and so are the limits without"]
#[doc = " `se_fit` or with `conf_type = \"none\"`."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object, get_all)]
pub struct TurnbullCurve {
    pub group: i32,
    pub n: usize,
    pub time: Vec<f64>,
    pub n_risk: Vec<f64>,
    pub n_event: Vec<f64>,
    pub n_censor: Vec<f64>,
    pub surv: Vec<f64>,
    pub std_err: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub iterations: usize,
}

#[doc = " R's `survfit` object for interval-censored data (`type = \"interval\"`)."]
#[doc = " `fit` holds the curves stacked as `survfit0`, `summary.survfit` and"]
#[doc = " `quantile.survfit` read them."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object, get_all)]
pub struct TurnbullResult {
    pub curves: Vec<TurnbullCurve>,
    pub conf_type: String,
    pub conf_int: f64,
    pub fit: SurvfitKMResult,
}

const TURNBULL_EPS: f64 = 0.00005;
const TURNBULL_MAX_ITER: usize = 100_000;

fn turnbull_jump_points(
    time1: &[f64],
    time2: &[f64],
    status: &mut [IntervalStatus],
) -> (Vec<f64>, f64) {
    let mut brackets: Vec<(f64, u8)> = Vec::with_capacity(time1.len() * 2);
    for (i, &code) in status.iter().enumerate() {
        match code {
            IntervalStatus::Right => brackets.push((time1[i], 2)),
            IntervalStatus::Exact => brackets.push((time1[i], 0)),
            IntervalStatus::Left => brackets.push((time1[i], 1)),
            IntervalStatus::Interval => {
                brackets.push((time1[i], 2));
                brackets.push((time2[i], 1));
            }
        }
    }
    // R orders tied finite endpoints by bracket type, including -0 and +0.
    let by_time = |a: &f64, b: &f64| a.partial_cmp(b).expect("validated finite endpoints");
    brackets.sort_by(|a, b| by_time(&a.0, &b.0).then_with(|| a.1.cmp(&b.1)));
    let mut jtimes: Vec<f64> = brackets
        .iter()
        .filter(|(_, kind)| *kind == 0)
        .map(|(t, _)| *t)
        .collect();
    jtimes.extend(
        brackets
            .windows(2)
            .filter(|pair| pair[0].1 != 1 && pair[1].1 == 1)
            .map(|pair| pair[0].0.midpoint(pair[1].0)),
    );
    let mintime = jtimes.iter().copied().fold(f64::INFINITY, f64::min);
    for i in 0..status.len() {
        if status[i] == IntervalStatus::Left && time1[i] < mintime {
            status[i] = IntervalStatus::Exact;
            jtimes.push(time1[i]);
        }
    }
    jtimes.sort_by(by_time);
    jtimes.dedup();
    (jtimes, mintime)
}

/// Weighted Kaplan–Meier step on the fixed Turnbull jump grid. Every event
/// (observed or imputed) lies on a jump. Rows between jumps are right censors.
fn product_limit_at_jumps(
    order: &[usize],
    jump_starts: &[usize],
    status: &[i32],
    weights: &[f64],
    survival: &mut [f64],
) {
    let mut risk = 0.0;
    let mut end = order.len();
    for (&start, factor) in jump_starts.iter().zip(survival.iter_mut()).rev() {
        let mut deaths = 0.0;
        // Match the KM kernel's accumulation order, including rows tied at a jump.
        for &row in order[start..end].iter().rev() {
            risk += weights[row];
            if status[row] == 1 {
                deaths += weights[row];
            }
        }
        *factor = if deaths > 0.0 {
            (risk - deaths) / risk
        } else {
            1.0
        };
        end = start;
    }
    let mut estimate = 1.0;
    for value in survival {
        estimate *= *value;
        *value = estimate;
    }
}

fn turnbull_curve(
    time1: &[f64],
    time2: &[f64],
    status: &mut [IntervalStatus],
    weights: &[f64],
    group: i32,
    final_fit: &SurvfitKMOptions,
) -> SurvivalResult<TurnbullCurve> {
    let n = time1.len();
    let (jtimes, mintime) = turnbull_jump_points(time1, time2, status);
    let njump = jtimes.len();
    if njump == 0 {
        return Err(SurvivalError::invalid_input(
            "no exact or interval-censored observations: the curve has no jump points",
        ));
    }
    let real: Vec<usize> = (0..n)
        .filter(|&i| matches!(status[i], IntervalStatus::Right | IntervalStatus::Exact))
        .collect();
    let censored: Vec<usize> = (0..n)
        .filter(|&i| matches!(status[i], IntervalStatus::Left | IntervalStatus::Interval))
        .collect();
    let mut fit_time: Vec<f64> = real.iter().map(|&i| time1[i]).collect();
    fit_time.extend_from_slice(&jtimes);
    let mut fit_status: Vec<i32> = real
        .iter()
        .map(|&i| i32::from(status[i] == IntervalStatus::Exact))
        .collect();
    fit_status.extend(std::iter::repeat_n(1, njump));
    let order = sorted_indices_by(&fit_time);
    let mut pos = 0;
    let jump_starts: Vec<usize> = jtimes
        .iter()
        .map(|&jump| {
            while pos < order.len() && fit_time[order[pos]] < jump {
                pos += 1;
            }
            pos
        })
        .collect();
    let mut fit_weights: Vec<f64> = real.iter().map(|&i| weights[i]).collect();
    fit_weights.resize(fit_time.len(), 0.0);

    // Each row of R's binary membership matrix is a contiguous range of jumps.
    // Keep its bounds rather than allocating censored.len() * njump values.
    let (supports, lwt): (Vec<std::ops::Range<usize>>, Vec<f64>) = if censored.is_empty() {
        (std::iter::once(0..njump).collect(), vec![1.0])
    } else {
        (
            censored
                .iter()
                .map(|&i| {
                    if status[i] == IntervalStatus::Left {
                        0..jtimes.partition_point(|&t| t <= time1[i])
                    } else {
                        jtimes.partition_point(|&t| t <= time1[i])
                            ..jtimes.partition_point(|&t| t <= time2[i])
                    }
                })
                .collect(),
            censored.iter().map(|&i| weights[i]).collect(),
        )
    };

    // The starting distribution counts overlapping intervals without case weights.
    // Endpoint updates make this linear in the number of intervals and jumps.
    let mut changes = vec![0isize; njump + 1];
    for support in &supports {
        changes[support.start] += 1;
        changes[support.end] -= 1;
    }
    let mut count = 0isize;
    let column_sums: Vec<f64> = changes[..njump]
        .iter()
        .map(|change| {
            count += change;
            count as f64
        })
        .collect();
    let total: f64 = column_sums.iter().sum();
    let mut cumulative = 0.0;
    let mut current_surv: Vec<f64> = column_sums
        .iter()
        .map(|c| {
            cumulative += c;
            1.0 - cumulative / total
        })
        .collect();
    let mut next_surv = vec![0.0; njump];
    let mut iter = 0usize;
    let mut eps = 1.0;
    let mut jump_history: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; njump]);
    let mut accelerated = Vec::new();
    let mut imputation = TurnbullWeights::new(&supports, &lwt, njump);
    while eps > TURNBULL_EPS {
        iter += 1;
        if iter > TURNBULL_MAX_ITER {
            return Err(SurvivalError::not_converged(TURNBULL_MAX_ITER));
        }
        jump_history.rotate_left(1);
        let [older, previous, current] = &mut jump_history;
        let mut preceding_surv = 1.0;
        for (jump, &s) in current.iter_mut().zip(&current_surv) {
            *jump = preceding_surv - s;
            preceding_surv = s;
        }
        let jumps: &[f64] = if iter.is_multiple_of(5) {
            accelerated.clear();
            for j in 0..njump {
                let d1 = previous[j] - older[j];
                let d2 = current[j] - previous[j];
                let value = older[j] - d1.powi(2) / (d2 - d1);
                accelerated.push(if value > 0.0 && value < 1.0 {
                    value
                } else {
                    current[j]
                });
            }
            if imputation.loglik(&accelerated) < imputation.loglik(current) {
                current
            } else {
                &accelerated
            }
        } else {
            current
        };

        imputation.redistribute(jumps, &mut fit_weights[real.len()..])?;
        validate_finite(&fit_weights, "weights")?;
        validate_non_negative(&fit_weights, "weights")?;
        product_limit_at_jumps(
            &order,
            &jump_starts,
            &fit_status,
            &fit_weights,
            &mut next_surv,
        );
        eps = if iter % 5 < 2 {
            1.0
        } else {
            current_surv
                .iter()
                .zip(&next_surv)
                .map(|(o, s)| (o - s).abs())
                .fold(0.0, f64::max)
        };
        std::mem::swap(&mut current_surv, &mut next_surv);
    }
    let data = SurvfitKMData::try_new(
        None,
        fit_time,
        fit_status,
        Some(fit_weights),
        None,
        None,
        None,
    )?;
    let mut km = survfitkm(&data, final_fit)?;
    for (&time, events) in km.time.iter().zip(&mut km.n_event) {
        if time < mintime && *events > 0.0 {
            *events = 0.0;
        }
    }
    Ok(TurnbullCurve {
        group,
        n,
        time: km.time,
        n_risk: km.n_risk,
        n_event: km.n_event,
        n_censor: km.n_censor,
        surv: km.surv,
        std_err: km.std_err.unwrap_or_default(),
        lower: km.lower.unwrap_or_default(),
        upper: km.upper.unwrap_or_default(),
        iterations: iter,
    })
}

fn validate_turnbull(input: &TurnbullInput<'_>) -> SurvivalResult<Vec<IntervalStatus>> {
    let n = input.time1.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input(
            "No (non-missing) observations",
        ));
    }
    validate_length(n, input.time2.len(), "time2")?;
    validate_length(n, input.status.len(), "status")?;
    let status: Vec<IntervalStatus> = input
        .status
        .iter()
        .map(|&code| IntervalStatus::from_code(code))
        .collect::<SurvivalResult<_>>()?;
    for (i, code) in status.iter().enumerate() {
        if !input.time1[i].is_finite() {
            return Err(SurvivalError::invalid_input(format!(
                "time1[{i}] must be finite"
            )));
        }
        if *code == IntervalStatus::Interval {
            if !input.time2[i].is_finite() {
                return Err(SurvivalError::invalid_input(format!(
                    "time2[{i}] must be finite for an interval-censored observation"
                )));
            }
            if input.time2[i] < input.time1[i] {
                return Err(SurvivalError::invalid_input(format!(
                    "time2[{i}] must not be less than time1[{i}]"
                )));
            }
        }
    }
    if let Some(weights) = input.weights {
        validate_length(n, weights.len(), "weights")?;
        validate_finite(weights, "weights")?;
        if weights.iter().any(|&w| w < 0.0) {
            return Err(SurvivalError::invalid_input("weights must be non-negative"));
        }
    }
    if let Some(group) = input.group {
        validate_length(n, group.len(), "group")?;
    }
    if !(0.0..1.0).contains(&input.conf_level) {
        return Err(SurvivalError::invalid_input(
            "conf_level must be between 0 and 1",
        ));
    }
    Ok(status)
}

pub fn turnbull(input: &TurnbullInput<'_>) -> SurvivalResult<TurnbullResult> {
    let mut status = validate_turnbull(input)?;
    let n = input.time1.len();
    let conf_type = ConfType::parse(input.conf_type)?;
    let (time1, time2) = if input.timefix {
        // Keep the interval pairs aligned so aeqSurv detects newly collapsed
        // intervals. Other observations have no upper endpoint.
        let mut upper = input.time2.to_vec();
        for (t, code) in upper.iter_mut().zip(&status) {
            if *code != IntervalStatus::Interval {
                *t = f64::NAN;
            }
        }
        let fixed = aeq_surv(input.time1, Some(&upper), None)?;
        (
            Cow::Owned(fixed.time),
            Cow::Owned(fixed.time2.expect("second time column supplied")),
        )
    } else {
        (Cow::Borrowed(input.time1), Cow::Borrowed(input.time2))
    };
    for (i, code) in status.iter_mut().enumerate() {
        if *code == IntervalStatus::Interval && time1[i] == time2[i] {
            *code = IntervalStatus::Exact;
        }
    }
    let weights = input
        .weights
        .map_or_else(|| Cow::Owned(vec![1.0; n]), Cow::Borrowed);
    let final_fit = SurvfitKMOptions {
        se_fit: input.se_fit,
        conf_int: input.conf_level,
        conf_type,
        robust: input.robust,
        timefix: false,
        ..SurvfitKMOptions::default()
    };
    let curves = if let Some(groups) = input.group {
        let (labels, codes) = strata_index(Some(groups), n);
        let rows = rows_by_curve(&codes, labels.len());
        labels
            .into_iter()
            .zip(rows)
            .map(|(label, rows)| {
                turnbull_curve(
                    &select_items(&time1, &rows),
                    &select_items(&time2, &rows),
                    &mut select_items(&status, &rows),
                    &select_items(&weights, &rows),
                    label,
                    &final_fit,
                )
            })
            .collect::<SurvivalResult<_>>()?
    } else {
        vec![turnbull_curve(
            &time1,
            &time2,
            &mut status,
            &weights,
            1,
            &final_fit,
        )?]
    };
    let fit = stacked_fit(&curves, input.conf_level, conf_type, input.se_fit)?;
    Ok(TurnbullResult {
        curves,
        conf_type: conf_type.as_str().to_string(),
        conf_int: input.conf_level,
        fit,
    })
}

fn stacked_fit(
    curves: &[TurnbullCurve],
    conf_level: f64,
    conf_type: ConfType,
    se_fit: bool,
) -> SurvivalResult<SurvfitKMResult> {
    let stack = |field: fn(&TurnbullCurve) -> &Vec<f64>| -> Vec<f64> {
        curves
            .iter()
            .flat_map(|curve| field(curve).iter().copied())
            .collect()
    };
    let time = stack(|curve| &curve.time);
    let std_err = se_fit.then(|| stack(|curve| &curve.std_err));
    let with_limits = se_fit && conf_type != ConfType::None;
    let t0 = time.iter().copied().fold(0.0, f64::min);
    SurvfitKMResult::from_stacked(StackedCurves {
        n_censor: Some(stack(|curve| &curve.n_censor)),
        std_chaz: std_err.clone(),
        std_err,
        lower: with_limits.then(|| stack(|curve| &curve.lower)),
        upper: with_limits.then(|| stack(|curve| &curve.upper)),
        conf_int: conf_level,
        conf_type: conf_type.as_str().to_string(),
        type_: "interval".to_string(),
        t0,
        ..StackedCurves::new(
            time,
            stack(|curve| &curve.n_risk),
            stack(|curve| &curve.n_event),
            stack(|curve| &curve.surv),
            (curves.len() > 1).then(|| curves.iter().map(|curve| curve.time.len()).collect()),
            curves.iter().map(|curve| curve.n).collect(),
        )
    })
}

#[doc = " Python entry point: `turnbull(time1, time2, status, weights=None,"]
#[doc = " group=None, conf_level=0.95, conf_type=\"log\", timefix=True,"]
#[doc = " se_fit=True, robust=None)`; `status` uses R's `interval` coding (0"]
#[doc = " right, 1 exact, 2 left, 3 interval)."]
#[pyfunction(name = "turnbull")]
#[pyo3(signature = (time1, time2, status, weights=None, group=None, conf_level=0.95, conf_type="log", timefix=true, se_fit=true, robust=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn turnbull_py(
    py: Python<'_>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
    group: Option<Vec<i32>>,
    conf_level: f64,
    conf_type: &str,
    timefix: bool,
    se_fit: bool,
    robust: Option<bool>,
) -> PyResult<TurnbullResult> {
    Ok(py.detach(|| {
        turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: weights.as_deref(),
            group: group.as_deref(),
            conf_level,
            conf_type,
            timefix,
            se_fit,
            robust,
        })
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_intercept_only_uses_total_time_at_risk() {
        let result = interval_censored_regression(
            vec![1.0, 2.0, 4.0],
            vec![f64::NAN; 3],
            vec![0, 1, 0],
            Vec::new(),
            3,
            0,
            &IntervalDistribution::Exponential,
            30,
            1e-10,
        )
        .unwrap();
        let scale: f64 = 7.0 / 2.0;
        let loglik = -2.0 * (scale.ln() + 1.0);
        assert!(result.converged);
        assert!(result.coefficients.is_empty() && result.std_errors.is_empty());
        assert_eq!(result.shape, 1.0);
        assert!((result.scale - scale).abs() < 1e-8);
        assert!((result.log_likelihood - loglik).abs() < 1e-12);
        assert!((result.aic - (2.0 - 2.0 * loglik)).abs() < 1e-12);
        assert!((result.survival_prob[1] - (-2.0 / scale).exp()).abs() < 1e-8);
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

    fn synthetic_interval() -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        (
            vec![1.0, 2.0, 2.0, 4.0, 5.0, 3.0, 6.0, 5.0, 2.0, 7.0],
            vec![
                3.0,
                4.0,
                f64::NAN,
                6.0,
                f64::NAN,
                f64::NAN,
                8.0,
                f64::NAN,
                3.0,
                f64::NAN,
            ],
            vec![3, 3, 2, 3, 1, 0, 3, 2, 3, 0],
        )
    }

    #[test]
    fn jump_points_follow_the_bracket_rule() {
        let (time1, time2, codes) = synthetic_interval();
        let mut status: Vec<IntervalStatus> = codes
            .iter()
            .map(|&c| IntervalStatus::from_code(c).unwrap())
            .collect();
        let (jtimes, mintime) = turnbull_jump_points(&time1, &time2, &mut status);
        assert_eq!(jtimes, vec![1.5, 2.5, 3.5, 5.0, 7.5]);
        assert_eq!(mintime, 1.5);
    }

    #[test]
    fn signed_zero_endpoints_follow_the_same_bracket_order_as_r() {
        for lower_zero in [-0.0, 0.0] {
            for upper_zero in [-0.0, 0.0] {
                // (-2, 0] and (0, 2] have disjoint support, regardless of sign.
                let (jumps, first) = turnbull_jump_points(
                    &[-2.0, lower_zero],
                    &[upper_zero, 2.0],
                    &mut [IntervalStatus::Interval; 2],
                );
                assert_eq!(jumps, [-1.0, 1.0]);
                assert_eq!(first, -1.0);
            }
        }
        // Like sort(unique(jtimes)) in R, keep the first exact zero.
        for first in [-0.0_f64, 0.0] {
            let (jumps, _) = turnbull_jump_points(
                &[first, -first],
                &[f64::NAN; 2],
                &mut [IntervalStatus::Exact; 2],
            );
            assert_eq!(jumps.len(), 1);
            assert_eq!(jumps[0].to_bits(), first.to_bits());
        }
    }

    #[test]
    fn finite_extreme_intervals_have_finite_midpoints() {
        for (lower, upper) in [(1.0e308_f64, 1.5e308_f64), (-1.5e308, -1.0e308)] {
            let fit = turnbull(&TurnbullInput {
                time1: &[lower],
                time2: &[upper],
                status: &[3],
                weights: None,
                group: None,
                conf_level: 0.95,
                conf_type: "log",
                timefix: true,
                se_fit: false,
                robust: None,
            })
            .unwrap();
            assert_eq!(fit.fit.time, [lower / 2.0 + upper / 2.0]);
            assert_eq!(fit.fit.surv, [0.0]);
        }
    }

    #[test]
    fn jump_sweep_matches_km_with_tied_events_censors_and_zero_mass() {
        let time = vec![2., 3., 2., 1., 0.5, 1.5, 2., 4., 1., 2., 3.];
        let status = vec![1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1];
        let weights = vec![1e16, 0.75, 1., 1., 0.5, 1e16, 1.25, 2., 0.5, 0.5, 0.];
        let mut estimates = [0.; 3];
        let order = sorted_indices_by(&time);
        product_limit_at_jumps(&order, &[1, 4, 8], &status, &weights, &mut estimates);
        let data =
            SurvfitKMData::try_new(None, time, status, Some(weights), None, None, None).unwrap();
        let km = survfitkm(
            &data,
            &SurvfitKMOptions {
                se_fit: false,
                timefix: false,
                ..Default::default()
            },
        )
        .unwrap();
        for (&jump, &estimate) in [1., 2., 3.].iter().zip(&estimates) {
            let row = km.time.iter().position(|time| *time == jump).unwrap();
            assert_eq!(estimate.to_bits(), km.surv[row].to_bits());
        }
    }

    #[test]
    fn turnbull_matches_r_survfit_on_the_fixture_data() {
        let (time1, time2, status) = synthetic_interval();
        let result = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        let curve = &result.curves[0];
        assert_eq!(curve.n, 10);
        assert_eq!(curve.time, vec![1.5, 2.5, 3.0, 3.5, 5.0, 7.0, 7.5]);
        let expected_surv = [
            0.846299203204601,
            0.538905789955341,
            0.538905789955341,
            0.538869070577718,
            0.245567865491214,
            0.245567865491214,
            0.0,
        ];
        for (actual, expected) in curve.surv.iter().zip(expected_surv) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(curve.n_censor, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        assert!((curve.n_risk[0] - 10.0).abs() < 1e-12);
        assert!((curve.n_event[0] - 1.53700796795399).abs() < 1e-6);
    }

    #[test]
    fn disjoint_intervals_match_exact_events_at_their_midpoints() {
        let n = 257;
        let time1: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let time2: Vec<f64> = time1.iter().map(|&t| t + 1.0).collect();
        let midpoints: Vec<f64> = time1.iter().map(|&t| t + 0.5).collect();
        let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i % 7) as f64 * 0.25).collect();
        let result = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &vec![3; n],
            weights: Some(&weights),
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: false,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        let data = SurvfitKMData::try_new(
            None,
            midpoints.clone(),
            vec![1; n],
            Some(weights),
            None,
            None,
            None,
        )
        .unwrap();
        let km = survfitkm(&data, &SurvfitKMOptions::default()).unwrap();
        let curve = &result.curves[0];
        assert_eq!(curve.time, midpoints);
        assert_close(&curve.n_risk, &km.n_risk, 1e-11, "n_risk");
        assert_close(&curve.n_event, &km.n_event, 1e-12, "n_event");
        assert_close(&curve.surv, &km.surv, 1e-12, "surv");
        assert_close(
            &curve.std_err,
            km.std_err.as_ref().unwrap(),
            1e-12,
            "std_err",
        );
    }

    #[test]
    fn weighted_interval_boundaries_match_stock_r() {
        // survival::survfit(Surv(time1, time2, status, type="interval") ~ 1,
        //                   weights=weights, timefix=FALSE), survival 3.8-11.
        let nan = f64::NAN;
        let result = turnbull(&TurnbullInput {
            time1: &[1.0, 2.0, 3.0, 1.0, 2.0, 4.0],
            time2: &[nan, nan, nan, 2.0, nan, nan],
            status: &[1, 1, 1, 3, 2, 0],
            weights: Some(&[1.0, 2.0, 0.5, 3.0, 0.75, 1.25]),
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: false,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        let curve = &result.curves[0];
        assert_eq!(curve.time, vec![1.0, 2.0, 3.0, 4.0]);
        assert_close(&curve.n_risk, &[8.5, 7.375, 1.75, 1.25], 1e-12, "n_risk");
        assert_close(&curve.n_event, &[1.125, 5.625, 0.5, 0.0], 1e-12, "n_event");
        assert_close(
            &curve.surv,
            &[
                0.8676470588235294,
                0.2058823529411765,
                0.14705882352941177,
                0.14705882352941177,
            ],
            1e-12,
            "surv",
        );
    }

    #[test]
    fn an_interval_without_representable_support_is_rejected() {
        // The bracket midpoint rounds down to the excluded lower endpoint.
        // A sparse update must not silently discard even a zero-weight row.
        for weight in [0.0, 1.0] {
            let err = turnbull(&TurnbullInput {
                time1: &[1.0, 2.0],
                time2: &[1.0_f64.next_up(), f64::NAN],
                status: &[3, 1],
                weights: Some(&[weight, 1.0]),
                group: None,
                conf_level: 0.95,
                conf_type: "log",
                timefix: false,
                se_fit: false,
                robust: None,
            })
            .unwrap_err();
            assert!(err.to_string().contains("no finite probability mass"));
        }
        // A nonempty support can also lose all its mass on a later iteration.
        let err = turnbull(&TurnbullInput {
            time1: &[1.0, 3.0],
            time2: &[2.0, 4.0],
            status: &[3, 3],
            weights: Some(&[0.0, 1.0]),
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: false,
            se_fit: false,
            robust: None,
        })
        .unwrap_err();
        assert!(err.to_string().contains("no finite probability mass"));
    }

    #[test]
    fn timefix_rejects_collapsed_intervals_and_preserves_exact_observations() {
        let time = [1.0, 2.0];
        // The second endpoint is unused on exact rows.
        let adjacent = [1.0_f64.next_up(), f64::NAN];
        for weight in [0.0, 1.0] {
            let weights = [weight, 1.0];
            let input = TurnbullInput {
                time1: &time,
                time2: &adjacent,
                status: &[3, 1],
                weights: Some(&weights),
                group: None,
                conf_level: 0.95,
                conf_type: "none",
                timefix: true,
                se_fit: false,
                robust: None,
            };
            let error = turnbull(&input).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("an interval has effective length 0")
            );
            let equal = turnbull(&TurnbullInput {
                time2: &[1.0, f64::NAN],
                ..input
            })
            .unwrap();
            let exact = turnbull(&TurnbullInput {
                status: &[1, 1],
                ..input
            })
            .unwrap();
            assert_eq!(equal.curves, exact.curves);
        }
    }

    #[test]
    fn grouped_curves_are_fitted_separately() {
        let (time1, time2, status) = synthetic_interval();
        let group = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
        let result = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: None,
            group: Some(&group),
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        assert_eq!(result.curves.len(), 2);
        assert_eq!(result.curves[0].time, vec![1.5, 2.5, 5.0, 7.0]);
        assert_eq!(result.curves[0].n, 5);
        let expected = [0.7, 0.4, 0.2, 0.0];
        for (actual, expected) in result.curves[0].surv.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    fn eight_rows(robust: Option<bool>) -> TurnbullResult {
        let nan = f64::NAN;
        turnbull(&TurnbullInput {
            time1: &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 7.0],
            time2: &[nan, nan, nan, 6.0, nan, 8.0, 4.0, nan],
            status: &[1, 0, 1, 3, 1, 3, 3, 0],
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust,
        })
        .unwrap()
    }

    #[test]
    fn robust_follows_survfit_km() {
        let robust = [
            0.116925805875275,
            0.116925805875275,
            0.185546918046783,
            0.185546918544061,
            0.172840380546378,
            0.172840379860608,
            0.172840379860608,
            0.0,
        ];
        for fit in [eight_rows(None), eight_rows(Some(true))] {
            assert_close(&fit.curves[0].std_err, &robust, 1e-6, "robust std.err");
        }
        let plain = eight_rows(Some(false));
        let std_err = &plain.curves[0].std_err;
        assert_close(
            &std_err[..7],
            &[
                0.133630620956212,
                0.133630620956212,
                0.318081270529207,
                0.318104505140176,
                0.592563375150846,
                0.592613260221602,
                0.592613260221602,
            ],
            1e-6,
            "std.err of log(surv)",
        );
        assert_eq!(std_err[7], f64::INFINITY);
        assert!(plain.curves[0].lower[7].is_nan());
        assert_close(
            &plain.curves[0].lower[..2],
            &[0.673381936505955, 0.673381936505955],
            1e-6,
            "lower",
        );

        let integer = |robust| {
            turnbull(&TurnbullInput {
                time1: &[1.0, 2.0, 3.0, 5.0],
                time2: &[f64::NAN; 4],
                status: &[0, 0, 1, 0],
                weights: None,
                group: None,
                conf_level: 0.95,
                conf_type: "log",
                timefix: true,
                se_fit: true,
                robust,
            })
            .unwrap()
            .curves[0]
                .std_err
                .clone()
        };
        let greenwood = 0.816496580927726;
        assert_close(
            &integer(None),
            &[0.0, 0.0, greenwood, greenwood],
            1e-12,
            "default",
        );
        let jackknife = 0.272165526975909;
        assert_close(
            &integer(Some(true)),
            &[0.0, 0.0, jackknife, jackknife],
            1e-12,
            "robust",
        );
    }

    #[test]
    fn the_stacked_fit_is_what_survfit0_and_summary_read() {
        let result = eight_rows(None);
        let fit = &result.fit;
        let curve = &result.curves[0];
        assert_eq!((fit.strata.as_ref(), fit.n.as_slice()), (None, &[8][..]));
        assert_eq!(
            (fit.type_.as_str(), fit.t0, fit.logse),
            ("interval", 0.0, true)
        );
        assert_eq!(fit.time, curve.time);
        assert_eq!(fit.std_err.as_ref(), Some(&curve.std_err));
        assert_eq!(fit.std_chaz, fit.std_err);
        let cumhaz: Vec<f64> = curve.surv.iter().map(|s| -s.ln()).collect();
        assert_eq!(fit.cumhaz, cumhaz);

        let (time1, time2, status) = synthetic_interval();
        let group = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
        let grouped = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: None,
            group: Some(&group),
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: false,
            robust: None,
        })
        .unwrap();
        let sizes: Vec<usize> = grouped.curves.iter().map(|c| c.time.len()).collect();
        assert_eq!(grouped.fit.strata, Some(sizes));
        assert_eq!(grouped.fit.n, vec![5, 5]);
        assert!(grouped.curves.iter().all(|c| c.std_err.is_empty()));
        assert!(grouped.fit.std_err.is_none() && grouped.fit.std_chaz.is_none());
        assert!(grouped.fit.lower.is_none() && grouped.fit.upper.is_none());
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "{what}[{i}]: {a} != {e} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn a_jump_point_with_zero_mass_stays_on_the_curve() {
        let nan = f64::NAN;
        let time1 = [7.0, 12.0, 12.0, 12.0, 18.0, 24.0, 30.0];
        let time2 = [nan, nan, nan, nan, nan, 27.0, 33.0];
        let status = [1, 0, 2, 2, 2, 3, 3];
        let result = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        let curve = &result.curves[0];
        assert_eq!(curve.time, vec![7.0, 9.5, 12.0, 15.0, 25.5, 31.5]);
        assert_eq!(curve.n_censor, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert_close(
            &curve.n_risk,
            &[7.0, 3.0, 3.0, 2.0, 2.0, 1.0],
            1e-12,
            "n_risk",
        );
        assert_close(
            &curve.n_event,
            &[4.0, 2.914335439641035e-15, 0.0, 0.0, 1.0, 1.0],
            1e-12,
            "n_event",
        );
        let three_sevenths = 3.0 / 7.0;
        assert_close(
            &curve.surv,
            &[
                three_sevenths,
                three_sevenths,
                three_sevenths,
                three_sevenths,
                three_sevenths / 2.0,
                0.0,
            ],
            1e-12,
            "surv",
        );
        assert_close(
            &curve.std_err,
            &[
                0.2397416351932802,
                0.23974163519328,
                0.23974163519328,
                0.23974163519328,
                0.1932050635587907,
                0.0,
            ],
            1e-12,
            "std_err",
        );
        assert_close(
            &curve.lower,
            &[
                0.1431737823939145,
                0.1431737823939143,
                0.1431737823939143,
                0.1431737823939143,
                0.03660410511538853,
                0.0,
            ],
            1e-12,
            "lower",
        );
        assert_close(
            &curve.upper,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            1e-12,
            "upper",
        );
    }

    #[test]
    fn a_dead_jump_before_later_events_keeps_the_lookup_aligned() {
        let nan = f64::NAN;
        let time1 = [7.0, 12.0, 12.0, 12.0, 18.0, 24.0, 30.0, 40.0, 40.0, 40.0];
        let time2 = [nan, nan, nan, nan, nan, 27.0, 33.0, nan, nan, nan];
        let status = [1, 0, 2, 2, 2, 3, 3, 1, 1, 1];
        let result = turnbull(&TurnbullInput {
            time1: &time1,
            time2: &time2,
            status: &status,
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        })
        .unwrap();
        let curve = &result.curves[0];
        assert_eq!(curve.time, vec![7.0, 9.5, 12.0, 15.0, 25.5, 31.5, 40.0]);
        assert_eq!(curve.n_censor, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(curve.n_event[2], 0.0);
        assert_eq!(curve.n_event[3], 0.0);
        assert_close(
            &curve.n_risk,
            &[10.0, 6.000702202088914, 6.0, 5.0, 5.0, 4.0, 3.0],
            1e-3,
            "n_risk",
        );
        assert_close(
            &curve.n_event,
            &[
                3.999297797911086,
                0.0007022020889141,
                0.0,
                0.0,
                1.0,
                1.0,
                3.0,
            ],
            1e-3,
            "n_event",
        );
        assert_close(
            &curve.surv,
            &[0.6000702202088914, 0.6, 0.6, 0.6, 0.48, 0.36, 0.0],
            1e-3,
            "surv",
        );
        assert_close(
            &curve.std_err,
            &[
                0.2135178870554299,
                0.2135060560853284,
                0.2135060560853284,
                0.2135060560853284,
                0.2017282702807698,
                0.1835498323470771,
                0.0,
            ],
            1e-3,
            "std_err",
        );
        assert_close(
            &curve.lower,
            &[
                0.2987626224465689,
                0.2987148246437666,
                0.2987148246437666,
                0.2987148246437666,
                0.2106246142616751,
                0.1325282101178965,
                0.0,
            ],
            1e-3,
            "lower",
        );
        assert_close(
            &curve.upper,
            &[1.0, 1.0, 1.0, 1.0, 1.0, 0.9779050051661341, 0.0],
            1e-3,
            "upper",
        );
    }

    #[test]
    fn turnbull_validates_codes_and_intervals() {
        let bad_code = turnbull(&TurnbullInput {
            time1: &[1.0],
            time2: &[2.0],
            status: &[4],
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        });
        assert!(bad_code.is_err());
        let reversed = turnbull(&TurnbullInput {
            time1: &[3.0],
            time2: &[2.0],
            status: &[3],
            weights: None,
            group: None,
            conf_level: 0.95,
            conf_type: "log",
            timefix: true,
            se_fit: true,
            robust: None,
        });
        assert!(reversed.is_err());
    }
}
