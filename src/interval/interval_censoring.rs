//! Interval-censored data: a parametric regression and Turnbull's
//! nonparametric survival estimate (R survival `R/survfitTurnbull.R`).

use crate::data_prep::aeq_surv;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::statistical::erf;
use crate::internal::validation::{validate_finite, validate_length};
use crate::surv_analysis::{ConfType, SurvfitKMData, SurvfitKMOptions, SurvfitKMResult, survfitkm};
use pyo3::prelude::*;

type DistributionFn = fn(f64, f64, f64) -> f64;
type DistributionFns = (DistributionFn, DistributionFn);

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

/// Interval-censoring codes of R's `Surv(..., type = "interval")` status
/// column: 0 right-censored at `time1`, 1 exact at `time1`, 2 left-censored
/// at `time1`, 3 censored in `(time1, time2]`.
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

/// Inputs of [`turnbull`]: an interval-censored response (`time1`,
/// `time2`, `status` in R's `interval` coding; `time2` is ignored unless
/// `status == 3`), optional case weights and grouping.
#[derive(Debug, Clone)]
pub struct TurnbullInput<'a> {
    pub time1: &'a [f64],
    pub time2: &'a [f64],
    pub status: &'a [i32],
    pub weights: Option<&'a [f64]>,
    /// One curve per distinct label, in sorted label order.
    pub group: Option<&'a [i32]>,
    pub conf_level: f64,
    /// `"log"`, `"log-log"`, `"plain"`, `"logit"`, `"arcsin"` or `"none"`.
    pub conf_type: &'a str,
    /// Apply R's `aeqSurv` near-tie rounding to the times first.
    pub timefix: bool,
}

/// One curve of [`TurnbullResult`]: the fields of the `survfit` object.
/// The standard error is the robust (infinitesimal jackknife) one of
/// `surv`, like R's `std.err` for a `survfitKM` fit with `robust = TRUE`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
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

/// R's `survfit` object for interval-censored data (`type = "interval"`).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct TurnbullResult {
    pub curves: Vec<TurnbullCurve>,
    pub conf_type: String,
    pub conf_int: f64,
}

/// Convergence criterion of R's EM loop (`while (eps > .00005)`).
const TURNBULL_EPS: f64 = 0.00005;
/// R iterates without bound; this cap turns a stalled EM into an error.
const TURNBULL_MAX_ITER: usize = 100_000;

/// The jump points of the Turnbull estimate: every exact time plus the
/// midpoint of every `( `-or-`[` bracket immediately followed by a `]`
/// (R's `jtimes`), the left-censored observations below the smallest jump
/// being promoted to exact times.  Returns `(jtimes, mintime, status)`.
fn turnbull_jump_points(
    time1: &[f64],
    time2: &[f64],
    status: &mut [IntervalStatus],
) -> (Vec<f64>, f64) {
    // stat2: 0 = "[" (exact), 1 = "]" (left / interval end), 2 = "(" (right /
    // interval start); ties order as [, ], (.
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
    brackets.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut jtimes: Vec<f64> = brackets
        .iter()
        .filter(|(_, kind)| *kind == 0)
        .map(|(t, _)| *t)
        .collect();
    jtimes.extend(
        brackets
            .windows(2)
            .filter(|pair| pair[0].1 != 1 && pair[1].1 == 1)
            .map(|pair| 0.5 * (pair[0].0 + pair[1].0)),
    );
    let mintime = jtimes.iter().copied().fold(f64::INFINITY, f64::min);
    for i in 0..status.len() {
        if status[i] == IntervalStatus::Left && time1[i] < mintime {
            status[i] = IntervalStatus::Exact;
            jtimes.push(time1[i]);
        }
    }
    jtimes.sort_by(f64::total_cmp);
    jtimes.dedup();
    (jtimes, mintime)
}

/// The EM of R's `survfitTurnbull` `doit` function for one curve.
fn turnbull_curve(
    time1: &[f64],
    time2: &[f64],
    status: &[IntervalStatus],
    weights: &[f64],
    group: i32,
    conf_level: f64,
    conf_type: ConfType,
) -> SurvivalResult<TurnbullCurve> {
    let n = time1.len();
    let mut status = status.to_vec();
    let (jtimes, mintime) = turnbull_jump_points(time1, time2, &mut status);
    let njump = jtimes.len();
    if njump == 0 {
        return Err(SurvivalError::invalid_input(
            "no exact or interval-censored observations: the curve has no jump points",
        ));
    }
    // The real observations (exact and right-censored) precede the fake
    // observations standing in for the jump points.
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
    // survfitKM(..., robust = TRUE) without a cluster: every observation
    // is its own cluster for the infinitesimal-jackknife variance.
    let cluster: Vec<i64> = (0..fit_time.len() as i64).collect();
    let fit_data = |fit_weights: Vec<f64>, cluster: Option<Vec<i64>>| {
        SurvfitKMData::try_new(
            None,
            fit_time.clone(),
            fit_status.clone(),
            Some(fit_weights),
            None,
            None,
            cluster,
        )
    };

    // wtmat: which jump points each left/interval observation covers.
    let (wtmat, lwt): (Vec<Vec<f64>>, Vec<f64>) = if censored.is_empty() {
        (vec![vec![1.0; njump]], vec![1.0])
    } else {
        (
            censored
                .iter()
                .map(|&i| {
                    jtimes
                        .iter()
                        .map(|&t| {
                            let covered = match status[i] {
                                IntervalStatus::Left => t <= time1[i],
                                _ => t > time1[i] && t <= time2[i],
                            };
                            f64::from(covered)
                        })
                        .collect()
                })
                .collect(),
            censored.iter().map(|&i| weights[i]).collect(),
        )
    };

    // Starter curve: proportional to the number of intervals covering
    // each jump point.
    let column_sums: Vec<f64> = (0..njump)
        .map(|j| wtmat.iter().map(|row| row[j]).sum())
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
    let mut old = current_surv.clone();
    let mut last_weights: Option<Vec<f64>> = None;
    let mut iter = 0usize;
    let mut eps = 1.0;
    let mut aitken1 = vec![0.0; njump];
    let mut jump1 = vec![0.0; njump];
    let mut jump2 = vec![0.0; njump];
    let loglik = |jumps: &[f64]| -> f64 {
        wtmat
            .iter()
            .map(|row| row.iter().zip(jumps).map(|(w, j)| w * j).sum::<f64>().ln())
            .sum()
    };
    while eps > TURNBULL_EPS {
        iter += 1;
        if iter > TURNBULL_MAX_ITER {
            return Err(SurvivalError::not_converged(TURNBULL_MAX_ITER));
        }
        // KM jumps at the jump points
        let mut previous = 1.0;
        let mut jumps: Vec<f64> = current_surv
            .iter()
            .map(|&s| {
                let jump = previous - s;
                previous = s;
                jump
            })
            .collect();
        // Aitken acceleration every fifth iteration
        let aitken2 = aitken1.clone();
        aitken1 = jumps.iter().zip(&jump1).map(|(j, p)| j - p).collect();
        let jsave = jumps.clone();
        if iter.is_multiple_of(5) {
            let oldlik = loglik(&jumps);
            for j in 0..njump {
                let accelerated = jump2[j] - aitken2[j].powi(2) / (aitken1[j] - aitken2[j]);
                jumps[j] = if accelerated.is_nan() || accelerated <= 0.0 || accelerated >= 1.0 {
                    jsave[j]
                } else {
                    accelerated
                };
            }
            if loglik(&jumps) < oldlik {
                jumps = jsave.clone();
            }
        }
        jump2 = jump1;
        jump1 = jsave;

        // Partition each left/interval observation over the jumps.
        let mut wt2 = vec![0.0; njump];
        for (row, &weight) in wtmat.iter().zip(&lwt) {
            let denominator: f64 = row.iter().zip(&jumps).map(|(w, j)| w * j).sum();
            for j in 0..njump {
                wt2[j] += weight / denominator * row[j] * jumps[j];
            }
        }
        let mut fit_weights: Vec<f64> = real.iter().map(|&i| weights[i]).collect();
        fit_weights.extend_from_slice(&wt2);
        // R's doit builds tempy without aeqSurv, so times compare exactly.
        let km = survfitkm(
            &fit_data(fit_weights.clone(), None)?,
            &SurvfitKMOptions {
                se_fit: false,
                timefix: false,
                ..SurvfitKMOptions::default()
            },
        )?;
        let stemp: Vec<f64> = jtimes
            .iter()
            .map(|&t| step_survival(&km.time, &km.surv, t))
            .collect();
        eps = if iter % 5 < 2 {
            1.0
        } else {
            old.iter()
                .zip(&stemp)
                .map(|(o, s)| (o - s).abs())
                .fold(0.0, f64::max)
        };
        old = stemp.clone();
        current_surv = stemp;
        last_weights = Some(fit_weights);
    }
    let Some(fit_weights) = last_weights else {
        unreachable!("the EM runs at least once");
    };
    // Final curve with R's robust (infinitesimal jackknife) standard
    // errors: survfitTurnbull calls survfitKM with robust = TRUE.
    let km = survfitkm(
        &fit_data(fit_weights.clone(), Some(cluster))?,
        &SurvfitKMOptions {
            conf_int: conf_level,
            conf_type,
            robust: Some(true),
            timefix: false,
            ..SurvfitKMOptions::default()
        },
    )?;
    let mut curve = with_zero_weight_times(&km, &fit_time, &fit_weights);
    for (i, &t) in curve.time.iter().enumerate() {
        if t < mintime && curve.n_event[i] > 0.0 {
            curve.n_event[i] = 0.0;
        }
    }
    Ok(TurnbullCurve {
        group,
        n,
        iterations: iter,
        ..curve
    })
}

/// The Kaplan-Meier curve as a right-continuous step function: the
/// estimate at the last reported time `<= t`, or 1 before the first one.
/// R's `survfitKM` reports every unique time, so `doit` can use
/// `match(jtimes, tfit$time)`; the crate's engine reports no row for a
/// time whose observations all carry zero weight, which happens as soon
/// as the EM mass at a jump point rounds to 0.
fn step_survival(km_time: &[f64], km_surv: &[f64], t: f64) -> f64 {
    match km_time.partition_point(|&u| u <= t) {
        0 => 1.0,
        k => km_surv[k - 1],
    }
}

/// The fitted curve over every unique observation time, as R's
/// `survfitKM` reports it: a time the engine left out (all of its
/// observations have zero weight) becomes a row with `n.event = n.censor
/// = 0` that carries the survival, standard error and limits of the row
/// before it and the weight still at risk.  `group`, `n` and `iterations`
/// are left for the caller.
fn with_zero_weight_times(
    km: &SurvfitKMResult,
    fit_time: &[f64],
    fit_weights: &[f64],
) -> TurnbullCurve {
    let mut unique_times = fit_time.to_vec();
    unique_times.sort_by(f64::total_cmp);
    unique_times.dedup();
    let std_err = km.std_err_surv_scale().unwrap_or_default();
    let has_limits = km.lower.is_some() && km.upper.is_some();
    let (lower, upper) = (
        km.lower.clone().unwrap_or_default(),
        km.upper.clone().unwrap_or_default(),
    );
    let n_times = unique_times.len();
    let mut curve = TurnbullCurve {
        group: 0,
        n: 0,
        time: unique_times,
        n_risk: Vec::with_capacity(n_times),
        n_event: Vec::with_capacity(n_times),
        n_censor: Vec::with_capacity(n_times),
        surv: Vec::with_capacity(n_times),
        std_err: Vec::with_capacity(n_times),
        lower: Vec::with_capacity(if has_limits { n_times } else { 0 }),
        upper: Vec::with_capacity(if has_limits { n_times } else { 0 }),
        iterations: 0,
    };
    let mut at_risk: f64 = fit_weights.iter().sum();
    let mut next = 0;
    for &t in &curve.time {
        if next < km.time.len() && km.time[next] == t {
            curve.n_risk.push(km.n_risk[next]);
            curve.n_event.push(km.n_event[next]);
            curve.n_censor.push(km.n_censor[next]);
            curve.surv.push(km.surv[next]);
            curve.std_err.push(std_err[next]);
            if has_limits {
                curve.lower.push(lower[next]);
                curve.upper.push(upper[next]);
            }
            at_risk = km.n_risk[next] - km.n_event[next] - km.n_censor[next];
            next += 1;
        } else {
            // Nothing leaves the risk set here; before the first reported
            // time the curve is still at its origin.
            curve.n_risk.push(at_risk);
            curve.n_event.push(0.0);
            curve.n_censor.push(0.0);
            curve.surv.push(curve.surv.last().copied().unwrap_or(1.0));
            curve
                .std_err
                .push(curve.std_err.last().copied().unwrap_or(0.0));
            if has_limits {
                curve.lower.push(curve.lower.last().copied().unwrap_or(1.0));
                curve.upper.push(curve.upper.last().copied().unwrap_or(1.0));
            }
        }
    }
    curve
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

/// Turnbull's nonparametric estimate for interval-censored data, one curve
/// per group, as R's `survfit` computes it (`R/survfitTurnbull.R`): an EM
/// with Aitken acceleration over a Kaplan-Meier fit to the exact and
/// right-censored observations plus weighted pseudo-observations at the
/// jump points.
pub fn turnbull(input: &TurnbullInput<'_>) -> SurvivalResult<TurnbullResult> {
    let mut status = validate_turnbull(input)?;
    let n = input.time1.len();
    let conf_type = ConfType::parse(input.conf_type)?;
    let (time1, time2) = if input.timefix {
        // aeqSurv over both time columns; a missing time2 stays missing.
        let mut all: Vec<f64> = input.time1.to_vec();
        let interval_rows: Vec<usize> = (0..n)
            .filter(|&i| status[i] == IntervalStatus::Interval)
            .collect();
        all.extend(interval_rows.iter().map(|&i| input.time2[i]));
        let fixed = aeq_surv(&all, None, None)?.time;
        let mut time2 = input.time2.to_vec();
        for (k, &i) in interval_rows.iter().enumerate() {
            time2[i] = fixed[n + k];
        }
        (fixed[..n].to_vec(), time2)
    } else {
        (input.time1.to_vec(), input.time2.to_vec())
    };
    // An interval (x, x] is an exact observation.
    for (i, code) in status.iter_mut().enumerate() {
        if *code == IntervalStatus::Interval && time1[i] == time2[i] {
            *code = IntervalStatus::Exact;
        }
    }
    let weights: Vec<f64> = input.weights.map_or_else(|| vec![1.0; n], <[f64]>::to_vec);
    let mut labels: Vec<i32> = input.group.map_or_else(|| vec![1], <[i32]>::to_vec);
    labels.sort_unstable();
    labels.dedup();
    let curves = labels
        .iter()
        .map(|&label| {
            let rows: Vec<usize> = (0..n)
                .filter(|&i| input.group.is_none_or(|g| g[i] == label))
                .collect();
            turnbull_curve(
                &rows.iter().map(|&i| time1[i]).collect::<Vec<_>>(),
                &rows.iter().map(|&i| time2[i]).collect::<Vec<_>>(),
                &rows.iter().map(|&i| status[i]).collect::<Vec<_>>(),
                &rows.iter().map(|&i| weights[i]).collect::<Vec<_>>(),
                label,
                input.conf_level,
                conf_type,
            )
        })
        .collect::<SurvivalResult<Vec<_>>>()?;
    Ok(TurnbullResult {
        curves,
        conf_type: conf_type.as_str().to_string(),
        conf_int: input.conf_level,
    })
}

/// Python entry point: `turnbull(time1, time2, status, weights=None,
/// group=None, conf_level=0.95, conf_type="log", timefix=True)`; `status`
/// uses R's `interval` coding (0 right, 1 exact, 2 left, 3 interval).
#[pyfunction(name = "turnbull")]
#[pyo3(signature = (time1, time2, status, weights=None, group=None, conf_level=0.95, conf_type="log", timefix=true))]
#[allow(clippy::too_many_arguments)]
pub fn turnbull_py(
    py: Python<'_>,
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    weights: Option<Vec<f64>>,
    group: Option<Vec<i32>>,
    conf_level: f64,
    conf_type: &str,
    timefix: bool,
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
        })
    })?)
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
        // Surv(left, right, type = "interval2") on the fixture data:
        // left = 1 2 NA 4 5 3 6 NA 2 7, right = 3 4 2 6 5 NA 8 5 3 NA
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

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tolerance,
                "{what}[{i}]: {a} != {e} (tolerance {tolerance})"
            );
        }
    }

    /// R 4.5.3 / survival 3.8.11: `survfit(Surv(t1, t2, status,
    /// type = "interval") ~ 1)`.  The jump at 15 (midpoint of the
    /// right-censored 12 and the left-censored 18) loses all of its mass
    /// during the EM; R's `survfitKM` keeps the zero-weight row, the
    /// crate's engine drops it.
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

    /// The same data plus three exact times after the dead jump, so the
    /// dropped row is not the last one.  R (4.5.3 / survival 3.8.11) and
    /// the port reach the same maximum through different rounding of the
    /// Aitken steps on the flat likelihood, so the estimates only agree
    /// to the EM's stopping tolerance; the curve's shape is exact.
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
        });
        assert!(reversed.is_err());
    }
}
