//! Survival curves after a Cox model: ports of R survival's `agsurv()`
//! (`R/agsurv.R` with the C helpers `src/agsurv4.c` and `src/agsurv5.c`) and
//! of the `coxsurv.fit()` expansion in `R/coxsurvfit.R`.
//!
//! [`agsurv`] computes, for one stratum, everything a Cox survival curve is
//! built from: the unique times, weighted event / censoring / at-risk
//! counts, the hazard increments and their variance, the weighted covariate
//! means `xbar` that carry the coefficient uncertainty, and (for the
//! Kalbfleisch-Prentice estimate) the per-time survival increments.
//! [`coxsurv_fit`] runs it per stratum and [`expand_curve`] /
//! [`individual_curve`] turn a stratum's pieces into curves for new
//! covariate rows (`survfit(fit, newdata)`), including the standard error
//! `sqrt(cumsum(varhaz) + dt' V dt) * risk2` on the cumulative-hazard scale.
//!
//! Everything here is plain Rust returning [`SurvivalResult`]; the Python
//! surface lives on `regression::coxph::CoxPHFit`.

use crate::error::{SurvivalError, SurvivalResult};
use ndarray::{Array1, Array2, ArrayView2};

/// R's `survtype` / `vartype` codes: `1` Kalbfleisch-Prentice, `2` Breslow
/// (Nelson-Aalen hazard), `3` Efron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoxSurvType {
    KalbfleischPrentice,
    Breslow,
    Efron,
}

impl CoxSurvType {
    /// `coxsurv.fit`'s `survtype <- if (stype==1) 1 else ctype+1`.
    pub fn from_stype_ctype(stype: u8, ctype: u8) -> SurvivalResult<Self> {
        match (stype, ctype) {
            (1, 1 | 2) => Ok(Self::KalbfleischPrentice),
            (2, 1) => Ok(Self::Breslow),
            (2, 2) => Ok(Self::Efron),
            _ => Err(SurvivalError::invalid_input(
                "stype must be 1 or 2 and ctype must be 1 or 2",
            )),
        }
    }
}

/// One stratum's data for [`agsurv`]: `start` is present for
/// (start, stop] data, `risk` is `exp(linear predictor)`.
#[derive(Clone, Copy)]
pub struct AgsurvData<'a> {
    pub start: Option<&'a [f64]>,
    pub stop: &'a [f64],
    pub status: &'a [i32],
    pub x: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    pub risk: &'a [f64],
}

impl AgsurvData<'_> {
    fn validate(&self) -> SurvivalResult<()> {
        let n = self.stop.len();
        if n == 0 {
            return Err(SurvivalError::invalid_input("agsurv: no observations"));
        }
        let check = |name: &str, len: usize| {
            if len != n {
                Err(SurvivalError::invalid_input(format!(
                    "agsurv: {name} has {len} rows but stop has {n}"
                )))
            } else {
                Ok(())
            }
        };
        if let Some(start) = self.start {
            check("start", start.len())?;
        }
        check("status", self.status.len())?;
        check("x", self.x.nrows())?;
        check("weights", self.weights.len())?;
        check("risk", self.risk.len())?;
        Ok(())
    }
}

/// The pieces of one stratum's curve (R's `agsurv()` list).
#[derive(Debug, Clone, PartialEq)]
pub struct AgsurvCurve {
    /// Number of observations in the stratum.
    pub n: usize,
    /// Sorted unique stop times (events and censorings).
    pub time: Vec<f64>,
    /// Weighted number of events at each time.
    pub n_event: Vec<f64>,
    /// Weighted number at risk at each time.
    pub n_risk: Vec<f64>,
    /// Weighted number censored at each time.
    pub n_censor: Vec<f64>,
    /// Hazard increment at each time.
    pub hazard: Vec<f64>,
    /// Cumulative hazard.
    pub cumhaz: Vec<f64>,
    /// Increment of the variance of the cumulative hazard at each time,
    /// the part that would remain if the coefficients were known.
    pub varhaz: Vec<f64>,
    /// Unweighted number of deaths at each time.
    pub ndeath: Vec<usize>,
    /// `ntime x nvar`: (weighted mean covariate of those at risk) times the
    /// hazard increment, the second part of the variance.
    pub xbar: Array2<f64>,
    /// Kalbfleisch-Prentice survival increments (`survtype == 1` only).
    pub surv: Option<Vec<f64>>,
}

/// `rev(cumsum(rev(x)))`: sum from the last element back to each position.
fn reverse_cumsum(values: &mut [f64]) {
    let mut total = 0.0;
    for value in values.iter_mut().rev() {
        total += *value;
        *value = total;
    }
}

/// `reverse_cumsum` applied to every column of a matrix.
fn reverse_cumsum_columns(values: &mut Array2<f64>) {
    for k in 0..values.ncols() {
        let mut total = 0.0;
        for g in (0..values.nrows()).rev() {
            total += values[(g, k)];
            values[(g, k)] = total;
        }
    }
}

/// Index of the group of `value` among the sorted unique `keys`.
fn group_index(keys: &[f64], value: f64) -> usize {
    keys.partition_point(|&key| key < value)
}

/// Sorted unique values (exact comparison, as R's `unique`).
fn sorted_unique(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.collect();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    sorted
}

/// Port of `src/agsurv4.c`: the Kalbfleisch-Prentice survival increment at
/// each unique time.  `risk` and `weights` are those of the deaths in time
/// order; `denom` is the weighted risk sum at each time.  A single death
/// solves the estimating equation in closed form, tied deaths by bisection.
fn agsurv4(ndeath: &[usize], risk: &[f64], weights: &[f64], denom: &[f64]) -> Vec<f64> {
    let mut km = vec![1.0; ndeath.len()];
    let mut j = 0;
    for (i, &deaths) in ndeath.iter().enumerate() {
        if deaths == 1 {
            km[i] = (1.0 - weights[j] * risk[j] / denom[i]).powf(1.0 / risk[j]);
        } else if deaths > 1 {
            let mut guess: f64 = 0.5;
            let mut inc = 0.25;
            for _ in 0..35 {
                let sum: f64 = (j..j + deaths)
                    .map(|k| weights[k] * risk[k] / (1.0 - guess.powf(risk[k])))
                    .sum();
                if sum < denom[i] {
                    guess += inc;
                } else {
                    guess -= inc;
                }
                inc /= 2.0;
            }
            km[i] = guess;
        }
        j += deaths;
    }
    km
}

/// Port of `src/agsurv5.c`: the Efron hazard sums.  For `d` tied deaths at
/// a time, `sum1 = mean_k 1/(nrisk - k/d erisk)`, `sum2` the same with the
/// square, and `xbar` the matching weighted covariate means.
struct Agsurv5 {
    sum1: Vec<f64>,
    sum2: Vec<f64>,
    xbar: Array2<f64>,
}

fn agsurv5(
    ndeath: &[usize],
    nrisk: &[f64],
    erisk: &[f64],
    xsum: &Array2<f64>,
    xsum2: &Array2<f64>,
) -> Agsurv5 {
    let ntime = ndeath.len();
    let nvar = xsum.ncols();
    let mut sum1 = vec![0.0; ntime];
    let mut sum2 = vec![0.0; ntime];
    let mut xbar = Array2::zeros((ntime, nvar));
    for i in 0..ntime {
        let d = ndeath[i];
        if d == 1 {
            let temp = 1.0 / nrisk[i];
            sum1[i] = temp;
            sum2[i] = temp * temp;
            for k in 0..nvar {
                xbar[(i, k)] = xsum[(i, k)] * temp * temp;
            }
        } else if d > 1 {
            let d_f = d as f64;
            for j in 0..d {
                let temp = 1.0 / (nrisk[i] - erisk[i] * j as f64 / d_f);
                sum1[i] += temp / d_f;
                sum2[i] += temp * temp / d_f;
                for k in 0..nvar {
                    xbar[(i, k)] +=
                        (xsum[(i, k)] - xsum2[(i, k)] * j as f64 / d_f) * temp * temp / d_f;
                }
            }
        }
    }
    Agsurv5 { sum1, sum2, xbar }
}

/// Port of `R/agsurv.R`: the survival-curve components of one stratum.
pub fn agsurv(
    data: &AgsurvData<'_>,
    survtype: CoxSurvType,
    vartype: CoxSurvType,
) -> SurvivalResult<AgsurvCurve> {
    data.validate()?;
    let n = data.stop.len();
    let nvar = data.x.ncols();
    let time = sorted_unique(data.stop.iter().copied());
    let ntime = time.len();

    let mut n_event = vec![0.0; ntime];
    let mut n_censor = vec![0.0; ntime];
    let mut nrisk = vec![0.0; ntime];
    let mut irisk = vec![0.0; ntime];
    let mut ndeath = vec![0usize; ntime];
    let mut xsum = Array2::zeros((ntime, nvar));
    let mut xsum2 = Array2::zeros((ntime, nvar));
    let mut erisk = vec![0.0; ntime];
    let wrisk: Vec<f64> = data
        .weights
        .iter()
        .zip(data.risk)
        .map(|(&w, &r)| w * r)
        .collect();
    for (i, &weighted_risk) in wrisk.iter().enumerate() {
        let g = group_index(&time, data.stop[i]);
        let death = data.status[i] == 1;
        if death {
            n_event[g] += data.weights[i];
            ndeath[g] += 1;
            erisk[g] += weighted_risk;
            for k in 0..nvar {
                xsum2[(g, k)] += weighted_risk * data.x[(i, k)];
            }
        } else {
            n_censor[g] += data.weights[i];
        }
        nrisk[g] += weighted_risk;
        irisk[g] += data.weights[i];
        for k in 0..nvar {
            xsum[(g, k)] += weighted_risk * data.x[(i, k)];
        }
    }
    reverse_cumsum(&mut nrisk);
    reverse_cumsum(&mut irisk);
    reverse_cumsum_columns(&mut xsum);

    if let Some(start) = data.start {
        // Subtract the rows that have not entered yet: those with
        // start >= t.  `etime` are the unique entry times; indx(t) points at
        // the first entry time >= t (R's approx(..., method = "constant",
        // f = 1, rule = 2)), or past the end when there is none.
        let etime = sorted_unique(start.iter().copied());
        let mut esum = vec![0.0; etime.len()];
        let mut ewt = vec![0.0; etime.len()];
        let mut xout = Array2::zeros((etime.len(), nvar));
        for i in 0..n {
            let g = group_index(&etime, start[i]);
            esum[g] += wrisk[i];
            ewt[g] += data.weights[i];
            for k in 0..nvar {
                xout[(g, k)] += wrisk[i] * data.x[(i, k)];
            }
        }
        reverse_cumsum(&mut esum);
        reverse_cumsum(&mut ewt);
        reverse_cumsum_columns(&mut xout);
        for (g, &t) in time.iter().enumerate() {
            let indx = group_index(&etime, t);
            if indx < etime.len() {
                nrisk[g] -= esum[indx];
                irisk[g] -= ewt[indx];
                for k in 0..nvar {
                    xsum[(g, k)] -= xout[(indx, k)];
                }
            }
        }
    }

    let surv = (survtype == CoxSurvType::KalbfleischPrentice).then(|| {
        let mut deaths: Vec<usize> = (0..n).filter(|&i| data.status[i] == 1).collect();
        deaths.sort_by(|&a, &b| data.stop[a].total_cmp(&data.stop[b]).then(a.cmp(&b)));
        let risk: Vec<f64> = deaths.iter().map(|&i| data.risk[i]).collect();
        let weights: Vec<f64> = deaths.iter().map(|&i| data.weights[i]).collect();
        agsurv4(&ndeath, &risk, &weights, &nrisk)
    });

    let efron = (survtype == CoxSurvType::Efron || vartype == CoxSurvType::Efron)
        .then(|| agsurv5(&ndeath, &nrisk, &erisk, &xsum, &xsum2));

    let hazard: Vec<f64> = (0..ntime)
        .map(|g| match (survtype, &efron) {
            (CoxSurvType::Efron, Some(tsum)) => n_event[g] * tsum.sum1[g],
            _ => n_event[g] / nrisk[g],
        })
        .collect();
    let varhaz: Vec<f64> = (0..ntime)
        .map(|g| match (vartype, &efron) {
            (CoxSurvType::KalbfleischPrentice, _) => {
                let denom = if n_event[g] >= nrisk[g] {
                    nrisk[g]
                } else {
                    nrisk[g] - n_event[g]
                };
                n_event[g] / (nrisk[g] * denom)
            }
            (CoxSurvType::Efron, Some(tsum)) => n_event[g] * tsum.sum2[g],
            _ => n_event[g] / (nrisk[g] * nrisk[g]),
        })
        .collect();
    let mut xbar = Array2::zeros((ntime, nvar));
    for g in 0..ntime {
        for k in 0..nvar {
            xbar[(g, k)] = match (vartype, &efron) {
                (CoxSurvType::Efron, Some(tsum)) => n_event[g] * tsum.xbar[(g, k)],
                _ => xsum[(g, k)] / nrisk[g] * hazard[g],
            };
        }
    }
    let mut cumhaz = hazard.clone();
    let mut running = 0.0;
    for value in cumhaz.iter_mut() {
        running += *value;
        *value = running;
    }

    Ok(AgsurvCurve {
        n,
        time,
        n_event,
        n_risk: irisk,
        n_censor,
        hazard,
        cumhaz,
        varhaz,
        ndeath,
        xbar,
        surv,
    })
}

/// A survival curve for one or more new covariate rows (the columns of the
/// matrices), R's `survfit(fit, newdata)` for one stratum.
#[derive(Debug, Clone, PartialEq)]
pub struct CoxSurvCurve {
    pub n: usize,
    pub time: Vec<f64>,
    pub n_risk: Vec<f64>,
    pub n_event: Vec<f64>,
    pub n_censor: Vec<f64>,
    /// `ntime x nrows(newdata)` survival probabilities.
    pub surv: Array2<f64>,
    /// `ntime x nrows(newdata)` cumulative hazards.
    pub cumhaz: Array2<f64>,
    /// Standard errors of the cumulative hazard (`std.err`, `logse = TRUE`).
    pub std_err: Option<Array2<f64>>,
}

/// Baseline survival of one stratum: `cumprod(surv)` for the
/// Kalbfleisch-Prentice estimate, `exp(-cumhaz)` otherwise.
fn baseline_survival(curve: &AgsurvCurve, survtype: CoxSurvType) -> Vec<f64> {
    match (survtype, &curve.surv) {
        (CoxSurvType::KalbfleischPrentice, Some(increments)) => {
            let mut running = 1.0;
            increments
                .iter()
                .map(|&value| {
                    running *= value;
                    running
                })
                .collect()
        }
        _ => curve.cumhaz.iter().map(|&h| (-h).exp()).collect(),
    }
}

/// `x' V x` for each row of `dt` (`rowSums((dt %*% varmat) * dt)`).
fn quadratic_forms(dt: &Array2<f64>, varmat: &Array2<f64>) -> Vec<f64> {
    dt.outer_iter()
        .map(|row| {
            let mut total = 0.0;
            for (i, &left) in row.iter().enumerate() {
                for (j, &right) in row.iter().enumerate() {
                    total += left * varmat[(i, j)] * right;
                }
            }
            total
        })
        .collect()
}

/// `coxsurv.fit`'s `expand`: curves of one stratum for the rows of `x2`
/// with relative risks `risk2`.  `varmat` requests the standard errors.
pub fn expand_curve(
    curve: &AgsurvCurve,
    survtype: CoxSurvType,
    x2: ArrayView2<'_, f64>,
    risk2: &[f64],
    varmat: Option<&Array2<f64>>,
) -> SurvivalResult<CoxSurvCurve> {
    let m = x2.nrows();
    let nvar = curve.xbar.ncols();
    if x2.ncols() != nvar {
        return Err(SurvivalError::invalid_input(format!(
            "newdata has {} columns but the curve has {nvar}",
            x2.ncols()
        )));
    }
    if risk2.len() != m {
        return Err(SurvivalError::invalid_input(
            "risk2 must have one value per newdata row",
        ));
    }
    let ntime = curve.time.len();
    let base_surv = baseline_survival(curve, survtype);
    let mut surv = Array2::zeros((ntime, m));
    let mut cumhaz = Array2::zeros((ntime, m));
    let mut std_err = varmat.map(|_| Array2::zeros((ntime, m)));
    let mut cum_varhaz = curve.varhaz.clone();
    let mut running = 0.0;
    for value in cum_varhaz.iter_mut() {
        running += *value;
        *value = running;
    }
    for i in 0..m {
        for g in 0..ntime {
            surv[(g, i)] = base_surv[g].powf(risk2[i]);
            cumhaz[(g, i)] = curve.cumhaz[g] * risk2[i];
        }
        if let (Some(varmat), Some(std_err)) = (varmat, std_err.as_mut()) {
            // dt = cumsum(hazard %o% x2[i,] - xbar)
            let mut dt = Array2::zeros((ntime, nvar));
            let mut running = vec![0.0; nvar];
            for g in 0..ntime {
                for k in 0..nvar {
                    running[k] += curve.hazard[g] * x2[(i, k)] - curve.xbar[(g, k)];
                    dt[(g, k)] = running[k];
                }
            }
            let term2 = quadratic_forms(&dt, varmat);
            for g in 0..ntime {
                std_err[(g, i)] = ((cum_varhaz[g] + term2[g]) * risk2[i] * risk2[i]).sqrt();
            }
        }
    }
    Ok(CoxSurvCurve {
        n: curve.n,
        time: curve.time.clone(),
        n_risk: curve.n_risk.clone(),
        n_event: curve.n_event.clone(),
        n_censor: curve.n_censor.clone(),
        surv,
        cumhaz,
        std_err,
    })
}

/// One (start, stop] interval of a time-dependent subject in
/// [`individual_curve`]: the stratum's curve pieces inside `(start, stop]`
/// are used with the row's covariates.
#[derive(Debug, Clone)]
pub struct IndividualInterval<'a> {
    pub start: f64,
    pub stop: f64,
    /// Index into the per-stratum curve list.
    pub stratum: usize,
    pub x2: &'a [f64],
    pub risk2: f64,
}

/// `coxsurv.fit`'s `onecurve`: stitches the curve of one subject whose
/// covariates (and possibly stratum) change over time, R's
/// `survfit(fit, newdata, id = )`.  The output time axis is shifted so
/// that the intervals abut (`toffset`).
pub fn individual_curve(
    curves: &[AgsurvCurve],
    survtype: CoxSurvType,
    intervals: &[IndividualInterval<'_>],
    varmat: Option<&Array2<f64>>,
) -> SurvivalResult<CoxSurvCurve> {
    let Some(first) = intervals.first() else {
        return Err(SurvivalError::invalid_input(
            "individual curve needs at least one interval",
        ));
    };
    let nvar = curves.first().map_or(0, |curve| curve.xbar.ncols());
    let mut time = Vec::new();
    let mut n_risk = Vec::new();
    let mut n_event = Vec::new();
    let mut n_censor = Vec::new();
    let mut hazard = Vec::new();
    let mut surv_increments = Vec::new();
    let mut varh1 = Vec::new();
    let mut dt_rows: Vec<Vec<f64>> = Vec::new();
    let mut toffset = 0.0;
    for (position, interval) in intervals.iter().enumerate() {
        if position > 0 {
            toffset += intervals[position - 1].stop - interval.start;
        }
        let curve = curves.get(interval.stratum).ok_or_else(|| {
            SurvivalError::invalid_input(format!(
                "interval stratum {} is not a fitted stratum",
                interval.stratum
            ))
        })?;
        if interval.x2.len() != nvar {
            return Err(SurvivalError::invalid_input(format!(
                "interval covariates have {} values but the model has {nvar}",
                interval.x2.len()
            )));
        }
        let base = baseline_survival(curve, survtype);
        for g in 0..curve.time.len() {
            let t = curve.time[g];
            if t <= interval.start || t > interval.stop {
                continue;
            }
            time.push(toffset + t);
            hazard.push(curve.hazard[g] * interval.risk2);
            surv_increments.push(if survtype == CoxSurvType::KalbfleischPrentice {
                curve
                    .surv
                    .as_ref()
                    .map_or(base[g], |s| s[g])
                    .powf(interval.risk2)
            } else {
                0.0
            });
            n_event.push(curve.n_event[g]);
            n_risk.push(curve.n_risk[g]);
            n_censor.push(curve.n_censor[g]);
            dt_rows.push(
                (0..nvar)
                    .map(|k| {
                        (curve.hazard[g] * interval.x2[k] - curve.xbar[(g, k)]) * interval.risk2
                    })
                    .collect(),
            );
            varh1.push(curve.varhaz[g] * interval.risk2 * interval.risk2);
        }
    }
    let ntime = time.len();
    let mut cumhaz = Array2::zeros((ntime, 1));
    let mut surv = Array2::zeros((ntime, 1));
    let mut running_hazard = 0.0;
    let mut running_surv = 1.0;
    for g in 0..ntime {
        running_hazard += hazard[g];
        cumhaz[(g, 0)] = running_hazard;
        surv[(g, 0)] = if survtype == CoxSurvType::KalbfleischPrentice {
            running_surv *= surv_increments[g];
            running_surv
        } else {
            (-running_hazard).exp()
        };
    }
    let std_err = varmat.map(|varmat| {
        let mut dt = Array2::zeros((ntime, nvar));
        let mut running = vec![0.0; nvar];
        for g in 0..ntime {
            for k in 0..nvar {
                running[k] += dt_rows[g][k];
                dt[(g, k)] = running[k];
            }
        }
        let term2 = quadratic_forms(&dt, varmat);
        let mut cum_varh1 = 0.0;
        let mut std_err = Array2::zeros((ntime, 1));
        for g in 0..ntime {
            cum_varh1 += varh1[g];
            std_err[(g, 0)] = (cum_varh1 + term2[g]).sqrt();
        }
        std_err
    });
    Ok(CoxSurvCurve {
        n: curves[first.stratum].n,
        time,
        n_risk,
        n_event,
        n_censor,
        surv,
        cumhaz,
        std_err,
    })
}

/// Runs [`agsurv`] once per stratum (`coxsurv.fit`'s first loop).  `strata`
/// are stratum codes; the curves come back in ascending code order together
/// with the codes.
pub fn coxsurv_fit(
    data: &AgsurvData<'_>,
    strata: Option<&[i32]>,
    survtype: CoxSurvType,
    vartype: CoxSurvType,
) -> SurvivalResult<(Vec<i32>, Vec<AgsurvCurve>)> {
    data.validate()?;
    let n = data.stop.len();
    let Some(strata) = strata else {
        return Ok((vec![0], vec![agsurv(data, survtype, vartype)?]));
    };
    if strata.len() != n {
        return Err(SurvivalError::invalid_input(format!(
            "strata has {} rows but stop has {n}",
            strata.len()
        )));
    }
    let mut codes = strata.to_vec();
    codes.sort_unstable();
    codes.dedup();
    let mut curves = Vec::with_capacity(codes.len());
    for &code in &codes {
        let rows: Vec<usize> = (0..n).filter(|&i| strata[i] == code).collect();
        let start = data
            .start
            .map(|start| rows.iter().map(|&i| start[i]).collect::<Vec<_>>());
        let stop: Vec<f64> = rows.iter().map(|&i| data.stop[i]).collect();
        let status: Vec<i32> = rows.iter().map(|&i| data.status[i]).collect();
        let weights: Vec<f64> = rows.iter().map(|&i| data.weights[i]).collect();
        let risk: Vec<f64> = rows.iter().map(|&i| data.risk[i]).collect();
        let mut x = Array2::zeros((rows.len(), data.x.ncols()));
        for (position, &i) in rows.iter().enumerate() {
            x.row_mut(position).assign(&data.x.row(i));
        }
        curves.push(agsurv(
            &AgsurvData {
                start: start.as_deref(),
                stop: &stop,
                status: &status,
                x: x.view(),
                weights: &weights,
                risk: &risk,
            },
            survtype,
            vartype,
        )?);
    }
    Ok((codes, curves))
}

/// Cumulative hazard of a curve just after `t`: `c(0, cumhaz)[findInterval(t, time) + 1]`.
pub fn cumhaz_at(curve: &AgsurvCurve, t: f64) -> f64 {
    step_at(&curve.time, &curve.cumhaz, t)
}

/// Value of a right-continuous step function (`c(0, values)[findInterval(t, times) + 1]`).
pub fn step_at(times: &[f64], values: &[f64], t: f64) -> f64 {
    let index = times.partition_point(|&time| time <= t);
    if index == 0 { 0.0 } else { values[index - 1] }
}

/// Cumulative sums of `varhaz` and of the rows of `xbar`, the two
/// integrated pieces `predict.coxph` needs for the standard error of an
/// expected count.
pub struct IntegratedCurve {
    pub cum_varhaz: Vec<f64>,
    pub cum_xbar: Array2<f64>,
}

pub fn integrate_curve(curve: &AgsurvCurve) -> IntegratedCurve {
    let mut cum_varhaz = curve.varhaz.clone();
    let mut running = 0.0;
    for value in cum_varhaz.iter_mut() {
        running += *value;
        *value = running;
    }
    let mut cum_xbar = curve.xbar.clone();
    for k in 0..cum_xbar.ncols() {
        let mut running = 0.0;
        for g in 0..cum_xbar.nrows() {
            running += cum_xbar[(g, k)];
            cum_xbar[(g, k)] = running;
        }
    }
    IntegratedCurve {
        cum_varhaz,
        cum_xbar,
    }
}

/// Row of `rbind(0, cum_xbar)[findInterval(t, time) + 1, ]`.
pub fn cum_xbar_at(curve: &AgsurvCurve, integrated: &IntegratedCurve, t: f64) -> Array1<f64> {
    let index = curve.time.partition_point(|&time| time <= t);
    if index == 0 {
        Array1::zeros(integrated.cum_xbar.ncols())
    } else {
        integrated.cum_xbar.row(index - 1).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn right_censored_breslow_matches_hand_computation() {
        // Times 1 (death), 2 (censor), 3 (two deaths, tied), risk all 1.
        let stop = [1.0, 2.0, 3.0, 3.0];
        let status = [1, 0, 1, 1];
        let x = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let weights = [1.0; 4];
        let risk = [1.0; 4];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let curve = agsurv(&data, CoxSurvType::Breslow, CoxSurvType::Breslow).unwrap();
        assert_eq!(curve.time, vec![1.0, 2.0, 3.0]);
        assert_eq!(curve.n_risk, vec![4.0, 3.0, 2.0]);
        assert_eq!(curve.n_event, vec![1.0, 0.0, 2.0]);
        assert_eq!(curve.n_censor, vec![0.0, 1.0, 0.0]);
        assert_eq!(curve.ndeath, vec![1, 0, 2]);
        assert_close(curve.hazard[0], 0.25);
        assert_close(curve.hazard[2], 1.0);
        assert_close(curve.cumhaz[2], 1.25);
        assert_close(curve.varhaz[0], 1.0 / 16.0);
        assert_close(curve.varhaz[2], 2.0 / 4.0);
        // xbar = (xsum / nrisk) * hazard: at t=1 xsum = 6, nrisk 4.
        assert_close(curve.xbar[(0, 0)], 6.0 / 4.0 * 0.25);
        assert_close(curve.xbar[(2, 0)], 5.0 / 2.0 * 1.0);
    }

    #[test]
    fn efron_hazard_averages_the_tied_denominators() {
        let stop = [1.0, 1.0, 2.0];
        let status = [1, 1, 0];
        let x = arr2(&[[0.0], [1.0], [2.0]]);
        let weights = [1.0; 3];
        let risk = [1.0, 2.0, 3.0];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let curve = agsurv(&data, CoxSurvType::Efron, CoxSurvType::Efron).unwrap();
        // nrisk = 6, erisk = 3, d = 2: sum1 = (1/6 + 1/(6 - 1.5)) / 2.
        let sum1 = (1.0 / 6.0 + 1.0 / 4.5) / 2.0;
        assert_close(curve.hazard[0], 2.0 * sum1);
        let sum2 = (1.0 / 36.0 + 1.0 / (4.5 * 4.5)) / 2.0;
        assert_close(curve.varhaz[0], 2.0 * sum2);
    }

    #[test]
    fn counting_process_risk_sets_respect_entry_times() {
        let start = [0.0, 0.0, 1.5, 0.0];
        let stop = [1.0, 2.0, 3.0, 3.0];
        let status = [1, 0, 1, 1];
        let x = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let weights = [1.0; 4];
        let risk = [1.0; 4];
        let data = AgsurvData {
            start: Some(&start),
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let curve = agsurv(&data, CoxSurvType::Breslow, CoxSurvType::Breslow).unwrap();
        // Row 2 enters at 1.5: not at risk at t = 1.
        assert_eq!(curve.n_risk, vec![3.0, 3.0, 2.0]);
        assert_close(curve.hazard[0], 1.0 / 3.0);
        assert_close(curve.xbar[(0, 0)], 4.0 / 3.0 / 3.0);
    }

    #[test]
    fn kalbfleisch_prentice_single_death_is_closed_form() {
        let stop = [1.0, 2.0];
        let status = [1, 0];
        let x = arr2(&[[0.0], [1.0]]);
        let weights = [1.0; 2];
        let risk = [2.0, 1.0];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let curve = agsurv(
            &data,
            CoxSurvType::KalbfleischPrentice,
            CoxSurvType::KalbfleischPrentice,
        )
        .unwrap();
        let surv = curve.surv.unwrap();
        assert_close(surv[0], (1.0 - 2.0 / 3.0_f64).powf(0.5));
        assert_eq!(surv[1], 1.0);
        assert_close(curve.varhaz[0], 1.0 / (3.0 * 2.0));
    }

    #[test]
    fn expansion_scales_by_the_relative_risk_and_reports_standard_errors() {
        let stop = [1.0, 2.0, 3.0];
        let status = [1, 1, 0];
        let x = arr2(&[[0.0], [1.0], [2.0]]);
        let weights = [1.0; 3];
        let risk = [1.0; 3];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let curve = agsurv(&data, CoxSurvType::Breslow, CoxSurvType::Breslow).unwrap();
        let x2 = arr2(&[[1.0], [2.0]]);
        let varmat = arr2(&[[0.5]]);
        let expanded = expand_curve(
            &curve,
            CoxSurvType::Breslow,
            x2.view(),
            &[1.0, 2.0],
            Some(&varmat),
        )
        .unwrap();
        assert_close(expanded.surv[(1, 0)], (-curve.cumhaz[1]).exp());
        assert_close(expanded.surv[(1, 1)], (-2.0 * curve.cumhaz[1]).exp());
        assert_close(expanded.cumhaz[(1, 1)], 2.0 * curve.cumhaz[1]);
        let std_err = expanded.std_err.unwrap();
        // dt at t=1 for row 0: hazard * 1 - xbar.
        let dt = curve.hazard[0] * 1.0 - curve.xbar[(0, 0)];
        assert_close(std_err[(0, 0)], (curve.varhaz[0] + dt * 0.5 * dt).sqrt());
    }

    #[test]
    fn strata_are_split_in_code_order() {
        let stop = [1.0, 2.0, 3.0, 4.0];
        let status = [1, 1, 1, 0];
        let x = arr2(&[[0.0], [1.0], [2.0], [3.0]]);
        let weights = [1.0; 4];
        let risk = [1.0; 4];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            weights: &weights,
            risk: &risk,
        };
        let (codes, curves) = coxsurv_fit(
            &data,
            Some(&[2, 1, 2, 1]),
            CoxSurvType::Breslow,
            CoxSurvType::Breslow,
        )
        .unwrap();
        assert_eq!(codes, vec![1, 2]);
        assert_eq!(curves[0].time, vec![2.0, 4.0]);
        assert_eq!(curves[1].time, vec![1.0, 3.0]);
        assert_eq!(cumhaz_at(&curves[1], 0.5), 0.0);
        assert_close(cumhaz_at(&curves[1], 3.5), 1.5);
    }
}
