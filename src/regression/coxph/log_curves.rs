//! Event increments kept in log scale when active risk sets cannot be exponentiated.
//! Ordinary Breslow/Efron curves retain agsurv arithmetic. K-P survival uses
//! log increments at every scale, before its baseline can round to zero or one.
//! Interval integrals avoid subtracting two rounded cumulative hazards.

use super::CoxPHFit;
use super::scaling::{PredictorOrigin, add_predictor_parts, predictor_origin_scales};
use crate::error::SurvivalResult;
use crate::regression::log_risk_sweep::{LogMomentTree, LogMoments, LogRiskSweep, LogRiskTime};
use crate::regression::log_scale::LogScale;
use crate::surv_analysis::agsurv::{AgsurvCurve, CoxSurvCurve, CoxSurvType, IndividualInterval};
use ndarray::{Array2, ArrayView2};
use std::ops::Deref;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub(super) struct CachedCurves {
    ordinary: Vec<AgsurvCurve>,
    pub stable: bool,
    pub origins: Vec<PredictorOrigin>,
    pub reference_offset: f64,
    pub kp_logs: Option<Vec<Vec<f64>>>,
    logs: OnceLock<Vec<LogCurve>>,
}

impl Deref for CachedCurves {
    type Target = [AgsurvCurve];
    fn deref(&self) -> &Self::Target {
        &self.ordinary
    }
}

impl CachedCurves {
    pub fn ordinary(curves: Vec<AgsurvCurve>, origins: Vec<PredictorOrigin>) -> Self {
        Self {
            ordinary: curves,
            stable: false,
            reference_offset: origins.first().map_or(0.0, |origin| origin.offset),
            origins,
            kp_logs: None,
            logs: OnceLock::new(),
        }
    }

    pub fn ordinary_kp(
        curves: Vec<AgsurvCurve>,
        origins: Vec<PredictorOrigin>,
        increments: Vec<Vec<f64>>,
    ) -> Self {
        let mut result = Self::ordinary(curves, origins);
        result.kp_logs = Some(increments);
        result
    }

    pub fn logs(&self) -> &[LogCurve] {
        self.logs.get_or_init(|| {
            self.ordinary
                .iter()
                .enumerate()
                .map(|(i, curve)| {
                    let mut logs = LogCurve::from_ordinary(curve);
                    if let Some(kp) = &self.kp_logs {
                        logs.log_kp = Some(
                            kp[i]
                                .iter()
                                .map(|&value| LogScale::from_log(value))
                                .collect(),
                        );
                    }
                    logs
                })
                .collect()
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct LogCurve {
    log_hazard: Vec<LogScale>,
    log_variance: Vec<LogScale>,
    means: Array2<f64>,
    log_kp: Option<Vec<LogScale>>,
    pub log_cumulative: Vec<LogScale>,
    hazard_tree: OnceLock<LogMomentTree>,
    variance_tree: OnceLock<LogMomentTree>,
}

pub(super) fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}

fn product(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 { 0.0 } else { a * b }
}

fn quadratic(left: &[f64], var: &Array2<f64>, right: &[f64]) -> f64 {
    let mut result = 0.0;
    for (i, &a) in left.iter().enumerate() {
        for (j, &b) in right.iter().enumerate() {
            result += product(product(a, var[(i, j)]), b);
        }
    }
    result
}

fn log_quadratic(hazard: LogScale, quadratic: f64) -> LogScale {
    if quadratic == 0.0 || hazard.is_zero() {
        LogScale::EMPTY
    } else {
        LogScale::new_parts(
            hazard.origin,
            hazard.origin_correction,
            2.0 * hazard.log_mass + quadratic.ln(),
        )
    }
}

fn at_predictor(value: LogScale, predictor: LogScale, degree: f64) -> LogScale {
    if value.log_mass.is_infinite() {
        return LogScale::from_log(value.log_at_scale_degree(predictor, degree));
    }
    let result = add_predictor_parts(
        LogScale::new_parts(
            value.origin,
            value.origin_correction,
            value.log_mass / degree,
        ),
        predictor,
    );
    LogScale::new_parts(
        result.origin,
        result.origin_correction,
        degree * result.log_mass,
    )
}

fn cross_scale(left: LogScale, right: LogScale, factor: f64) -> LogScale {
    let middle = add_predictor_parts(
        LogScale::from_predictor_parts(left.origin / 2.0, left.origin_correction / 2.0),
        LogScale::from_predictor_parts(right.origin / 2.0, right.origin_correction / 2.0),
    );
    LogScale::new_parts(
        middle.origin,
        middle.origin_correction,
        left.log_mass + right.log_mass + factor.ln(),
    )
}

impl LogCurve {
    fn new(
        log_hazard: Vec<LogScale>,
        log_variance: Vec<LogScale>,
        means: Array2<f64>,
        log_kp: Option<Vec<LogScale>>,
    ) -> Self {
        let mut cumulative = LogScale::EMPTY;
        let log_cumulative = log_hazard
            .iter()
            .map(|&hazard| {
                cumulative = cumulative.mix(hazard).0;
                cumulative
            })
            .collect();
        Self {
            log_hazard,
            log_variance,
            means,
            log_kp,
            log_cumulative,
            hazard_tree: OnceLock::new(),
            variance_tree: OnceLock::new(),
        }
    }

    fn hazard_tree(&self) -> &LogMomentTree {
        self.hazard_tree.get_or_init(|| {
            let mut tree = LogMomentTree::new(self.log_hazard.len(), self.means.ncols());
            for (i, &hazard) in self.log_hazard.iter().enumerate() {
                tree.set(
                    i,
                    hazard,
                    self.means.row(i).as_slice().expect("contiguous mean"),
                );
            }
            tree
        })
    }

    fn variance_tree(&self) -> &LogMomentTree {
        self.variance_tree.get_or_init(|| {
            let mut tree = LogMomentTree::new_with_degree(self.log_variance.len(), 0, 2.0);
            for (i, &variance) in self.log_variance.iter().enumerate() {
                tree.set(i, variance, &[]);
            }
            tree
        })
    }

    fn from_ordinary(curve: &AgsurvCurve) -> Self {
        let mut means = curve.xbar.clone();
        for (g, &hazard) in curve.hazard.iter().enumerate() {
            for value in means.row_mut(g) {
                *value = if hazard == 0.0 { 0.0 } else { *value / hazard };
            }
        }
        Self::new(
            curve
                .hazard
                .iter()
                .map(|h| LogScale::from_log(h.ln()))
                .collect(),
            curve
                .varhaz
                .iter()
                .map(|v| LogScale::new(0.0, v.ln()))
                .collect(),
            means,
            curve.surv.as_ref().map(|s| {
                s.iter()
                    .map(|s| LogScale::from_log((-s.ln()).ln()))
                    .collect()
            }),
        )
    }

    pub fn range_log_hazard(&self, first: usize, end: usize, lp: LogScale) -> f64 {
        let hazard = if first == end {
            LogScale::EMPTY
        } else {
            self.hazard_tree().range(first, end).weight
        };
        hazard.log_at_scale_degree(lp, 1.0)
    }

    pub fn range_hazard(&self, first: usize, end: usize, lp: LogScale) -> f64 {
        self.range_log_hazard(first, end, lp).exp()
    }

    pub fn expected_se(
        &self,
        first: usize,
        end: usize,
        lp: LogScale,
        x: &[f64],
        var: &Array2<f64>,
    ) -> f64 {
        if first == end {
            return LogScale::EMPTY.exp_at_scale(lp);
        }
        let moments = self.hazard_tree().range(first, end);
        let delta: Vec<f64> = x
            .iter()
            .zip(&moments.mean)
            .map(|(&x, &mean)| x - mean)
            .collect();
        let q = quadratic(&delta, var, &delta);
        let mut variance = self
            .variance_tree()
            .range(first, end)
            .weight
            .mix_degree(log_quadratic(moments.weight, q), 2.0)
            .0;
        // R computes the difference of endpoint variances. Expand the
        // coefficient quadratic algebraically so a small late interval survives.
        if first > 0 && !delta.iter().all(|&d| d == 0.0) {
            let previous = self.hazard_tree().range(0, first);
            let prior: Vec<f64> = x
                .iter()
                .zip(&previous.mean)
                .map(|(&x, &mean)| x - mean)
                .collect();
            let cross = quadratic(&prior, var, &delta);
            if cross != 0.0 && !previous.weight.is_zero() && !moments.weight.is_zero() {
                let term = cross_scale(previous.weight, moments.weight, 2.0 * cross.abs());
                variance = if cross > 0.0 {
                    variance.mix_degree(term, 2.0).0
                } else {
                    let ratio = 2.0
                        * ((term.origin - variance.origin)
                            + (term.origin_correction - variance.origin_correction))
                        + (term.log_mass - variance.log_mass);
                    if ratio > 0.0 {
                        LogScale::from_log(f64::NAN)
                    } else {
                        variance.shifted((-ratio.exp()).ln_1p())
                    }
                };
            }
        }
        (variance.log_at_scale_degree(lp, 2.0) / 2.0).exp()
    }

    pub fn expand(
        &self,
        curve: &AgsurvCurve,
        x: ArrayView2<'_, f64>,
        log_risk: &[LogScale],
        var: Option<&Array2<f64>>,
    ) -> CoxSurvCurve {
        let mut surv = Array2::ones((curve.time.len(), x.nrows()));
        let mut cumhaz = Array2::zeros(surv.dim());
        let mut std_err = var.map(|_| Array2::zeros(surv.dim()));
        for (i, &lp) in log_risk.iter().enumerate() {
            let mut moments = LogMoments::new(x.ncols());
            let mut variance = LogScale::EMPTY;
            let mut survival_hazard = LogScale::EMPTY;
            for g in 0..curve.time.len() {
                moments.add(
                    self.log_hazard[g],
                    self.means.row(g).as_slice().expect("contiguous mean"),
                );
                variance = variance.mix_degree(self.log_variance[g], 2.0).0;
                survival_hazard = survival_hazard
                    .mix(self.log_kp.as_ref().map_or(self.log_hazard[g], |kp| kp[g]))
                    .0;
                let hazard = moments.weight.exp_at_scale(lp);
                cumhaz[(g, i)] = hazard;
                surv[(g, i)] = (-survival_hazard.exp_at_scale(lp)).exp();
                if let (Some(var), Some(se)) = (var, &mut std_err) {
                    let derivative: Vec<f64> = x
                        .row(i)
                        .iter()
                        .zip(&moments.mean)
                        .map(|(&x, &mean)| x - mean)
                        .collect();
                    let coefficient =
                        log_quadratic(moments.weight, quadratic(&derivative, var, &derivative));
                    se[(g, i)] = (variance
                        .mix_degree(coefficient, 2.0)
                        .0
                        .log_at_scale_degree(lp, 2.0)
                        / 2.0)
                        .exp();
                }
            }
        }
        CoxSurvCurve {
            n: curve.n,
            time: curve.time.clone(),
            n_risk: curve.n_risk.clone(),
            n_event: curve.n_event.clone(),
            n_censor: curve.n_censor.clone(),
            surv,
            cumhaz,
            std_err,
        }
    }
}

impl CoxPHFit {
    pub(super) fn compute_log_curves(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
        log_risk: &[LogScale],
        origins: &[PredictorOrigin],
    ) -> SurvivalResult<CachedCurves> {
        let p = self.nvar();
        let offset_mean = self.offset_mean();
        let origin_scales =
            predictor_origin_scales(origins, &self.means, &self.coefficients_or_zero());
        let scalar_risk: Vec<_> = log_risk.iter().map(|risk| risk.log_value()).collect();
        let mut ordinary = Vec::new();
        let mut logs = Vec::new();
        for (stratum, &(start, end)) in self.sorted.bounds.iter().enumerate() {
            let rows: Vec<usize> = self.sorted.order[start..end]
                .iter()
                .copied()
                .filter(|&i| start_time.is_none_or(|t| self.time[i] >= t))
                .collect();
            let mut curve = AgsurvCurve {
                n: rows.len(),
                time: vec![],
                n_event: vec![],
                n_risk: vec![],
                n_censor: vec![],
                hazard: vec![],
                cumhaz: vec![],
                varhaz: vec![],
                ndeath: vec![],
                xbar: Array2::zeros((0, p)),
                surv: (survtype == CoxSurvType::KalbfleischPrentice).then(Vec::new),
            };
            let mut log_hazard = Vec::new();
            let mut log_variance = Vec::new();
            let mut mean_rows = Vec::new();
            let mut log_kp = curve.surv.as_ref().map(|_| Vec::new());
            let mut cumulative = LogScale::EMPTY;
            LogRiskSweep {
                stop: &self.time,
                entry: self.entry.as_deref(),
                status: &self.status,
                x: self.x.view(),
                weights: &self.weights,
                log_risk: &scalar_risk,
                rows: &rows,
            }
            .for_each_time_scales(log_risk, |event| {
                let mut moments = LogMoments::new(p);
                let mut variance = LogScale::EMPTY;
                if !event.deaths.is_empty() {
                    if survtype == CoxSurvType::Efron {
                        let log_weight = event.log_death_weight - (event.deaths.len() as f64).ln();
                        let mut mean = vec![0.0; p];
                        for step in 0..event.deaths.len() {
                            let denom = event.efron_denom(step);
                            event.efron_mean(step, &mut mean);
                            for (value, &center) in mean.iter_mut().zip(&self.means) {
                                *value -= center;
                            }
                            moments.add(denom.reciprocal(log_weight), &mean);
                            variance = variance
                                .mix_degree(
                                    LogScale::new_parts(
                                        -denom.origin,
                                        -denom.origin_correction,
                                        log_weight - 2.0 * denom.log_mass,
                                    ),
                                    2.0,
                                )
                                .0;
                        }
                    } else {
                        let mean: Vec<f64> = event
                            .mean
                            .iter()
                            .zip(&self.means)
                            .map(|(m, c)| m - c)
                            .collect();
                        moments.add(event.denom.reciprocal(event.log_death_weight), &mean);
                        // Match agsurv's K-P variance definition, including its
                        // dependence on the baseline risk scale. The K-P survival
                        // estimate itself is invariant to a common risk shift.
                        let origin = origin_scales[stratum];
                        let reference = add_predictor_parts(
                            LogScale::from_log(offset_mean),
                            origin.reciprocal(0.0),
                        );
                        let relative_events = reference.shifted(event.log_death_weight);
                        let ratio = relative_events.ratio_log(event.denom);
                        let denom2 = if survtype == CoxSurvType::KalbfleischPrentice && ratio < 0.0
                        {
                            event.denom.shifted((-ratio.exp()).ln_1p())
                        } else {
                            event.denom
                        };
                        variance = LogScale::new_parts(
                            -event.denom.origin,
                            -event.denom.origin_correction,
                            event.log_death_weight - event.denom.log_mass - denom2.log_mass,
                        );
                    }
                }
                let kp = if event.deaths.is_empty() {
                    LogScale::EMPTY
                } else if log_kp.is_some() {
                    kp_log_hazard(&event, log_risk, &self.weights)
                } else {
                    LogScale::EMPTY
                };
                cumulative = cumulative.mix(moments.weight).0;
                curve.time.push(event.time);
                curve.n_event.push(event.death_weight);
                curve.n_risk.push(event.risk_weight);
                curve.n_censor.push(
                    event
                        .rows_at_time
                        .iter()
                        .filter(|&&i| self.status[i] == 0)
                        .map(|&i| self.weights[i])
                        .sum(),
                );
                curve.ndeath.push(event.deaths.len());
                curve.hazard.push(moments.weight.exp_at(0.0));
                curve.varhaz.push(variance.exp_at_degree(0.0, 2.0));
                curve.cumhaz.push(cumulative.exp_at(0.0));
                if let Some(surv) = &mut curve.surv {
                    surv.push((-kp.exp_at(0.0)).exp());
                }
                if let Some(values) = &mut log_kp {
                    values.push(kp);
                }
                log_hazard.push(moments.weight);
                log_variance.push(variance);
                mean_rows.extend_from_slice(&moments.mean);
                Ok(())
            })?;
            let means = Array2::from_shape_vec((curve.time.len(), p), mean_rows)
                .expect("one mean per time");
            curve.xbar = means.clone();
            for (i, &h) in curve.hazard.iter().enumerate() {
                for value in curve.xbar.row_mut(i) {
                    *value = product(*value, h);
                }
            }
            logs.push(LogCurve::new(log_hazard, log_variance, means, log_kp));
            ordinary.push(curve);
        }
        Ok(CachedCurves {
            ordinary,
            stable: true,
            origins: origins.to_vec(),
            reference_offset: offset_mean,
            kp_logs: None,
            logs: OnceLock::from(logs),
        })
    }
}

// Solve the K-P likelihood in q=-log(survival increment). In this form the
// death and nondeath risk masses never need to be subtracted or exponentiated.
fn kp_log_hazard(event: &LogRiskTime<'_>, log_risk: &[LogScale], weights: &[f64]) -> LogScale {
    let nondeath = event.nondeath_risk;
    if nondeath.is_zero() {
        return LogScale::from_log(f64::INFINITY);
    }
    let first_risk = log_risk[event.deaths[0]];
    if event
        .deaths
        .iter()
        .all(|&i| log_risk[i].ratio_log(first_risk) == 0.0)
    {
        // Equal-risk ties have a closed form: q*risk=log1p(death/nondeath).
        let ratio = event.death_risk.ratio_log(nondeath);
        if ratio < -36.0 {
            // log1p(death/nondeath) ~= death/nondeath; cancel the
            // equal death risk before a huge negative ratio can overflow.
            return nondeath.reciprocal(event.log_death_weight);
        }
        let log_softplus = if ratio > 36.0 {
            if ratio.is_finite() {
                ratio.ln()
            } else {
                // The log ratio can exceed f64 even though its logarithm is
                // finite. A finite surviving risk is never a terminal event.
                let a = event.death_risk;
                let scale = [
                    a.origin,
                    a.origin_correction,
                    a.log_mass,
                    nondeath.origin,
                    nondeath.origin_correction,
                    nondeath.log_mass,
                ]
                .into_iter()
                .map(f64::abs)
                .fold(0.0, f64::max);
                scale.ln()
                    + ((a.origin / scale - nondeath.origin / scale)
                        + (a.origin_correction / scale - nondeath.origin_correction / scale)
                        + (a.log_mass / scale - nondeath.log_mass / scale))
                        .ln()
            }
        } else {
            ratio.exp().ln_1p().ln()
        };
        return first_risk.reciprocal(log_softplus);
    }
    // Bracket and bisect paired log(q) values. A root can move from the
    // largest risk scale to a much smaller tied death's scale; a fixed origin
    // would move that cancellation back into one rounded scalar log mass.
    let score = |q: LogScale| {
        event.deaths.iter().fold(f64::NEG_INFINITY, |sum, &i| {
            let z = q.log_at_scale_degree(log_risk[i], 1.0);
            let term = if z < -36.0 {
                // expm1(q*risk) ~= q*risk. Cancel risk before subtraction.
                weights[i].ln() - q.log_at_scale_degree(nondeath, 1.0)
            } else {
                let log_expm1 = if z > 7.0 {
                    z.exp()
                } else {
                    z.exp().exp_m1().ln()
                };
                if log_expm1 == f64::INFINITY {
                    f64::NEG_INFINITY
                } else {
                    weights[i].ln() + log_risk[i].ratio_log(nondeath) - log_expm1
                }
            };
            log_add(sum, term)
        })
    };
    let midpoint = |a: LogScale, b: LogScale| {
        add_predictor_parts(
            LogScale::new_parts(a.origin / 2.0, a.origin_correction / 2.0, a.log_mass / 2.0),
            LogScale::new_parts(b.origin / 2.0, b.origin_correction / 2.0, b.log_mass / 2.0),
        )
    };
    // d / total risk <= q <= d / nondeath risk. These analytic bounds
    // avoid growing an arbitrary search step across the full exponent range.
    let mut lo = event.denom.reciprocal(event.log_death_weight);
    let mut hi = nondeath.reciprocal(event.log_death_weight);
    // A stratum can span all 1024 binary exponents before the final ~53
    // precision steps. Ordinary-scale roots converge in approximately 60.
    for _ in 0..2048 {
        let mid = midpoint(lo, hi);
        if mid.ratio_log(lo) == 0.0 || mid.ratio_log(hi) == 0.0 {
            break;
        }
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    midpoint(lo, hi)
}

pub(super) fn individual(
    curves: &CachedCurves,
    intervals: &[IndividualInterval<'_>],
    log_risk: &[LogScale],
    var: Option<&Array2<f64>>,
) -> CoxSurvCurve {
    let first = &intervals[0];
    let p = first.x2.len();
    let mut time = Vec::new();
    let mut n_risk = Vec::new();
    let mut n_event = Vec::new();
    let mut n_censor = Vec::new();
    let mut cumhaz = Vec::new();
    let mut surv = Vec::new();
    let mut std_err = var.map(|_| Vec::new());
    let mut moments = LogMoments::new(p);
    let mut variance = LogScale::EMPTY;
    let mut survival_hazard = LogScale::EMPTY;
    let mut toffset = 0.0;
    for (position, interval) in intervals.iter().enumerate() {
        if position > 0 {
            toffset += intervals[position - 1].stop - interval.start;
        }
        let curve = &curves[interval.stratum];
        let logs = &curves.logs()[interval.stratum];
        let lp = log_risk[position];
        let first = curve.time.partition_point(|&t| t <= interval.start);
        let end = curve.time.partition_point(|&t| t <= interval.stop);
        for g in first..end {
            let delta: Vec<f64> = interval
                .x2
                .iter()
                .zip(logs.means.row(g))
                .map(|(x, m)| x - m)
                .collect();
            moments.add(at_predictor(logs.log_hazard[g], lp, 1.0), &delta);
            variance = variance
                .mix_degree(at_predictor(logs.log_variance[g], lp, 2.0), 2.0)
                .0;
            survival_hazard = survival_hazard
                .mix(at_predictor(
                    logs.log_kp.as_ref().map_or(logs.log_hazard[g], |kp| kp[g]),
                    lp,
                    1.0,
                ))
                .0;
            time.push(toffset + curve.time[g]);
            n_risk.push(curve.n_risk[g]);
            n_event.push(curve.n_event[g]);
            n_censor.push(curve.n_censor[g]);
            let hazard = moments.weight.exp_at(0.0);
            cumhaz.push(hazard);
            surv.push((-survival_hazard.exp_at(0.0)).exp());
            if let (Some(var), Some(se)) = (var, &mut std_err) {
                let coefficient =
                    log_quadratic(moments.weight, quadratic(&moments.mean, var, &moments.mean));
                se.push(
                    (variance
                        .mix_degree(coefficient, 2.0)
                        .0
                        .log_at_scale_degree(LogScale::from_log(0.0), 2.0)
                        / 2.0)
                        .exp(),
                );
            }
        }
    }
    let column =
        |values| Array2::from_shape_vec((time.len(), 1), values).expect("one output per time");
    let surv = column(surv);
    let cumhaz = column(cumhaz);
    let std_err = std_err.map(column);
    CoxSurvCurve {
        n: curves[first.stratum].n,
        time,
        n_risk,
        n_event,
        n_censor,
        surv,
        cumhaz,
        std_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hazard_and_variance_ranges_cancel_origins_before_scaling() {
        for origin in [1e16, 1e308] {
            let logs = LogCurve::new(
                vec![
                    LogScale::new(-origin, 0.5_f64.ln()),
                    LogScale::new(origin, 0.5_f64.ln()),
                ],
                vec![
                    LogScale::new(-origin, 0.25_f64.ln()),
                    LogScale::new(origin, 0.25_f64.ln()),
                ],
                Array2::zeros((2, 0)),
                None,
            );
            for (first, predictor) in [(0, origin), (1, -origin)] {
                let risk = LogScale::new(predictor, 0.3);
                let expected = 0.5 * 0.3_f64.exp();
                assert!((logs.range_hazard(first, first + 1, risk) - expected).abs() < 1e-14);
                assert!(
                    (logs.expected_se(first, first + 1, risk, &[], &Array2::zeros((0, 0)))
                        - expected)
                        .abs()
                        < 1e-14
                );
            }
        }
    }

    #[test]
    fn heterogeneous_kp_ties_keep_small_effects_at_huge_common_offsets() {
        let stop = [1.0; 3];
        let status = [1, 1, 0];
        let weights = [1.0, 2.0, 3.0];
        let x = Array2::zeros((3, 0));
        let small = [0.2, -0.4, 0.8];
        let mut reference = 0.0;
        for origin in [0.0, 1e16, -1e16, 1e308, -1e308] {
            let risks = small.map(|value| LogScale::new(origin, value));
            let mut actual = 0.0;
            LogRiskSweep {
                stop: &stop,
                entry: None,
                status: &status,
                x: x.view(),
                weights: &weights,
                log_risk: &small,
                rows: &[0, 1, 2],
            }
            .for_each_time_scales(&risks, |event| {
                actual = kp_log_hazard(&event, &risks, &weights).exp_at_scale(risks[0]);
                Ok(())
            })
            .unwrap();
            if origin == 0.0 {
                reference = actual;
                // At q, the independent K-P score is zero.
                let q = actual / small[0].exp();
                let left: f64 = (0..2)
                    .map(|i| weights[i] * small[i].exp() / (q * small[i].exp()).exp_m1())
                    .sum();
                assert!((left - weights[2] * small[2].exp()).abs() < 1e-12);
            } else {
                assert!(
                    (actual - reference).abs() < 1e-14,
                    "origin {origin}: {actual} != {reference}"
                );
            }
        }
    }
}
