//! Quantiles of survival curves: port of R survival `R/quantile.survfit.R`
//! (`findq`, `doquant`, `quantile.survfit`).  The quantile is where a
//! horizontal line at `p` meets the cumulative distribution `1 - surv`;
//! a flat stretch exactly at `p` is resolved by the midpoint rule, and the
//! limits come from the curve's own confidence bounds.

use super::{SurvfitCurve, survmean::split_curves};
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;

/// `quantile(fit, probs, conf.int)`: one row per curve, `NaN` for a
/// quantile the curve does not reach (R's `NA`).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object, get_all)]
pub struct SurvfitCurveQuantiles {
    pub probs: Vec<f64>,
    pub quantile: Vec<Vec<f64>>,
    pub lower: Option<Vec<Vec<f64>>>,
    pub upper: Option<Vec<Vec<f64>>>,
}

/// R's `approx(x, seq_along(x), v, method = "constant", f = 1)$y` for a
/// strictly increasing `x`: the (0-based) index of the knot at `v`, or of
/// the next knot above `v`; `None` outside the range (`rule = 1`).
fn approx_constant_right(x: &[f64], v: f64) -> Option<usize> {
    let n = x.len();
    if n == 0 || v.is_nan() || v < x[0] || v > x[n - 1] {
        return None;
    }
    let position = x.partition_point(|&knot| knot < v);
    Some(position)
}

/// R's `findq`: quantiles of the CDF `y` over `x` (both starting with
/// the time origin and 0).
fn findq(x: &[f64], y: &[f64], probs: &[f64], tol: f64) -> Vec<f64> {
    let y_max = y
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::MIN, f64::max);
    let p_min = probs.iter().copied().fold(f64::INFINITY, f64::min);
    if y_max < p_min {
        return vec![f64::NAN; probs.len()];
    }
    let xmax = x[x.len() - 1];
    // Drop duplicated y values (the censorings); R's duplicated() keeps
    // the first occurrence.
    let mut kept_x = Vec::with_capacity(x.len());
    let mut kept_y = Vec::with_capacity(y.len());
    for (&xi, &yi) in x.iter().zip(y) {
        let duplicate = kept_y
            .iter()
            .any(|&seen: &f64| seen == yi || (seen.is_nan() && yi.is_nan()));
        if !duplicate {
            kept_x.push(xi);
            kept_y.push(yi);
        }
    }
    let n = kept_y.len();
    // approx() drops NA knots (R's NA limits where the curve reaches 0).
    let knots_x: Vec<f64> = kept_x
        .iter()
        .zip(&kept_y)
        .filter(|(_, y)| !y.is_nan())
        .map(|(x, _)| *x)
        .collect();
    let knots_y: Vec<f64> = kept_y.iter().copied().filter(|y| !y.is_nan()).collect();
    let y_plus: Vec<f64> = knots_y.iter().map(|v| v + tol).collect();
    let y_minus: Vec<f64> = knots_y.iter().map(|v| v - tol).collect();
    probs
        .iter()
        .map(|&p| {
            if p == 0.0 {
                return kept_x[0];
            }
            let index1 = approx_constant_right(&y_plus, p);
            let index2 = approx_constant_right(&y_minus, p);
            let last = kept_y[n - 1];
            if !last.is_nan() && (p - last).abs() < tol {
                return match index1 {
                    Some(i1) => (knots_x[i1] + xmax) / 2.0,
                    None => f64::NAN,
                };
            }
            match (index1, index2) {
                (Some(i1), Some(i2)) => (knots_x[i1] + knots_x[i2]) / 2.0,
                _ => f64::NAN,
            }
        })
        .collect()
}

/// Quantiles of the lower and upper survival bounds of one curve.
type QuantileLimits = (Vec<f64>, Vec<f64>);

/// R's `doquant` for one curve: quantiles of the estimate and, when the
/// curve has limits, of the upper and lower bounds.
fn doquant(
    curve: &SurvfitCurve<'_>,
    probs: &[f64],
    first_x: f64,
    scale: f64,
    tol: f64,
) -> (Vec<f64>, Option<QuantileLimits>) {
    let mut x = Vec::with_capacity(curve.time.len() + 1);
    x.push(first_x);
    x.extend_from_slice(curve.time);
    let cdf = |values: &[f64]| {
        let mut y = Vec::with_capacity(values.len() + 1);
        y.push(0.0);
        y.extend(values.iter().map(|s| 1.0 - s));
        y
    };
    let scaled = |values: Vec<f64>| values.into_iter().map(|q| q / scale).collect::<Vec<_>>();
    let quantile = scaled(findq(&x, &cdf(curve.surv), probs, tol));
    let limits = match (curve.lower, curve.upper) {
        (Some(lower), Some(upper)) => Some((
            scaled(findq(&x, &cdf(lower), probs, tol)),
            scaled(findq(&x, &cdf(upper), probs, tol)),
        )),
        _ => None,
    };
    (quantile, limits)
}

/// Quantiles of every curve.  `first_x` is R's `x$start.time` (0 when
/// absent); `tolerance` defaults to `sqrt(.Machine$double.eps)`.
pub fn quantile_survfit(
    curves: &[SurvfitCurve<'_>],
    probs: &[f64],
    conf_int: bool,
    first_x: f64,
    scale: f64,
    tolerance: f64,
) -> SurvivalResult<SurvfitCurveQuantiles> {
    if curves.is_empty() {
        return Err(SurvivalError::invalid_input("no curves to summarise"));
    }
    if probs.iter().any(|p| p.is_nan() || !(0.0..=1.0).contains(p)) {
        return Err(SurvivalError::invalid_input("Invalid probability"));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(SurvivalError::invalid_input("scale must be positive"));
    }
    for curve in curves {
        curve.validate()?;
    }
    let conf_int = conf_int && curves.iter().all(|curve| curve.lower.is_some());
    let mut quantile = Vec::with_capacity(curves.len());
    let mut lower = Vec::with_capacity(curves.len());
    let mut upper = Vec::with_capacity(curves.len());
    for curve in curves {
        let (q, limits) = doquant(curve, probs, first_x, scale, tolerance);
        quantile.push(q);
        if let (true, Some((from_lower, from_upper))) = (conf_int, limits) {
            // R: the lower quantile limit comes from the lower survival
            // bound (it drops faster, so it reaches p sooner).
            lower.push(from_lower);
            upper.push(from_upper);
        }
    }
    Ok(SurvfitCurveQuantiles {
        probs: probs.to_vec(),
        quantile,
        lower: conf_int.then_some(lower),
        upper: conf_int.then_some(upper),
    })
}

/// Python entry point mirroring [`super::survmean_curves_py`]'s curve encoding;
/// `probs` defaults to the quartiles.
#[pyfunction(name = "quantile_survfit_curves")]
#[pyo3(signature = (time, surv, lower=None, upper=None, strata=None, probs=None, conf_int=true, start_time=0.0, scale=1.0, tolerance=None))]
#[allow(clippy::too_many_arguments)]
pub fn quantile_survfit_curves_py(
    time: Vec<f64>,
    surv: Vec<f64>,
    lower: Option<Vec<f64>>,
    upper: Option<Vec<f64>>,
    strata: Option<Vec<usize>>,
    probs: Option<Vec<f64>>,
    conf_int: bool,
    start_time: f64,
    scale: f64,
    tolerance: Option<f64>,
) -> PyResult<SurvfitCurveQuantiles> {
    let probs = probs.unwrap_or_else(|| vec![0.25, 0.5, 0.75]);
    let sizes = strata.unwrap_or_else(|| vec![time.len()]);
    let zeros = vec![0.0; time.len()];
    let curves = split_curves(
        &time,
        &surv,
        &zeros,
        &zeros,
        lower.as_deref(),
        upper.as_deref(),
        &sizes,
    )?;
    Ok(quantile_survfit(
        &curves,
        &probs,
        conf_int,
        start_time,
        scale,
        tolerance.unwrap_or_else(|| f64::EPSILON.sqrt()),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.490_116_119_384_765_6e-8;

    #[test]
    fn approx_constant_matches_r_semantics() {
        let x = [0.0, 0.2, 0.5, 1.0];
        assert_eq!(approx_constant_right(&x, 0.2), Some(1));
        assert_eq!(approx_constant_right(&x, 0.3), Some(2));
        assert_eq!(approx_constant_right(&x, 1.0), Some(3));
        assert_eq!(approx_constant_right(&x, 1.1), None);
        assert_eq!(approx_constant_right(&x, -0.1), None);
    }

    #[test]
    fn quantiles_use_the_midpoint_rule_on_flats() {
        // surv 0.75, 0.5, 0.5 (censor), 0.25 at times 1..4
        let time = [1.0, 2.0, 3.0, 4.0];
        let surv = [0.75, 0.5, 0.5, 0.25];
        let curve = SurvfitCurve {
            time: &time,
            surv: &surv,
            n_risk: &[4.0; 4],
            n_event: &[1.0, 1.0, 0.0, 1.0],
            lower: None,
            upper: None,
        };
        let result =
            quantile_survfit(&[curve], &[0.25, 0.5, 0.75, 0.9], true, 0.0, 1.0, TOL).unwrap();
        // p = .25 sits exactly on the first step: midpoint of 1 and 2
        assert_eq!(result.quantile[0][0], 1.5);
        // p = .5 sits on the flat from 2 to 4 (the censor at 3 is dropped)
        assert_eq!(result.quantile[0][1], 3.0);
        // p = .75 is the last value: midpoint of 4 and the last time 4
        assert_eq!(result.quantile[0][2], 4.0);
        assert!(result.quantile[0][3].is_nan());
        assert!(result.lower.is_none());
    }

    #[test]
    fn limits_come_from_the_confidence_bounds() {
        let time = [1.0, 2.0, 3.0];
        let surv = [0.6, 0.4, 0.2];
        let lower = [0.3, 0.1, 0.05];
        let upper = [0.9, 0.7, 0.5];
        let curve = SurvfitCurve {
            time: &time,
            surv: &surv,
            n_risk: &[3.0; 3],
            n_event: &[1.0; 3],
            lower: Some(&lower),
            upper: Some(&upper),
        };
        let result = quantile_survfit(&[curve], &[0.5], true, 0.0, 1.0, TOL).unwrap();
        assert_eq!(result.quantile[0][0], 2.0);
        // the lower bound crosses 0.5 at t=1, the upper bound at t=3
        assert_eq!(result.lower.as_ref().unwrap()[0][0], 1.0);
        assert_eq!(result.upper.as_ref().unwrap()[0][0], 3.0);
        assert_eq!(result.quantile.len(), 1);
    }

    #[test]
    fn a_curve_that_never_drops_gives_na() {
        let time = [1.0, 2.0];
        let surv = [0.9, 0.8];
        let curve = SurvfitCurve {
            time: &time,
            surv: &surv,
            n_risk: &[2.0; 2],
            n_event: &[1.0; 2],
            lower: None,
            upper: None,
        };
        let result = quantile_survfit(&[curve], &[0.5, 0.0], false, 0.0, 1.0, TOL).unwrap();
        assert!(result.quantile[0][0].is_nan());
        assert_eq!(result.quantile[0][1], 0.0);
        assert!(quantile_survfit(&[], &[0.5], false, 0.0, 1.0, TOL).is_err());
    }
}
