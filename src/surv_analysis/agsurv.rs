//! Survival curves after a Cox model: ports of R survival's `agsurv()`
//! (`R/agsurv.R` with the C helpers `src/agsurv4.c` and `src/agsurv5.c`) and
//! of the `coxsurv.fit()` expansion in `R/coxsurvfit.R`.
//!
//! [`agsurv`] computes, for one stratum, everything a Cox survival curve is
//! built from: the unique times, weighted event / censoring / at-risk
//! counts, the hazard increments and their variance, the weighted covariate
//! means `xbar` that carry the coefficient uncertainty, and (for the
//! Kalbfleisch-Prentice estimate) the per-time survival increments.
//! [`agsurv_rows`] does the same for a subset of rows (a stratum of a fit)
//! without copying the data, and [`expand_curve`] / [`individual_curve`]
//! turn a stratum's pieces into curves for new covariate rows
//! (`survfit(fit, newdata)`), including the standard error
//! `sqrt(cumsum(varhaz) + dt' V dt) * risk2` on the cumulative-hazard scale.
//!
//! The kernels are plain Rust returning [`SurvivalResult`]; the Python
//! surface of a fitted model lives on `regression::coxph::CoxPHFit`. The
//! direct matrix driver lives in [`super::coxsurv_fit`]. Bindings here expose
//! [`cox_survfit_baseline`] for one stratum and [`step_values_at`] for reading
//! a curve at given times.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::matrix::matrix_rows;
use crate::internal::numpy_utils::{FloatMatrix, FloatVec};
use crate::internal::step::{find_interval, sort_unique, step_at};
use crate::internal::validation::{
    ValidationError, validate_binary_f64, validate_finite, validate_length, validate_non_negative,
    validate_positive, validate_sorted,
};
use ndarray::{Array1, Array2, ArrayView2};
use pyo3::prelude::*;

/// R's `survtype` / `vartype` codes: `1` Kalbfleisch-Prentice, `2` Breslow
/// (Nelson-Aalen hazard), `3` Efron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoxSurvType {
    KalbfleischPrentice,
    Breslow,
    Efron,
}

impl CoxSurvType {
    /// R's integer code: `1` Kalbfleisch-Prentice, `2` Breslow, `3` Efron.
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            1 => Some(Self::KalbfleischPrentice),
            2 => Some(Self::Breslow),
            3 => Some(Self::Efron),
            _ => None,
        }
    }

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

/// The data behind a set of Cox survival curves: `start` is present for
/// (start, stop] data, `risk` is `exp(linear predictor)`, and `means`, when
/// given, are subtracted from the columns of `x` on the fly (R's
/// `agsurv(y, x - means, ...)`), so the fitted design matrix is used as is.
#[derive(Clone, Copy)]
pub struct AgsurvData<'a> {
    pub start: Option<&'a [f64]>,
    pub stop: &'a [f64],
    pub status: &'a [i32],
    pub x: ArrayView2<'a, f64>,
    pub means: Option<&'a [f64]>,
    pub weights: &'a [f64],
    pub risk: &'a [f64],
}

impl AgsurvData<'_> {
    fn validate(&self) -> SurvivalResult<()> {
        let n = self.stop.len();
        if n == 0 {
            return Err(SurvivalError::invalid_input("agsurv: no observations"));
        }
        let check = |name: &str, len: usize, expected: usize| {
            if len != expected {
                Err(SurvivalError::invalid_input(format!(
                    "agsurv: {name} has {len} entries, expected {expected}"
                )))
            } else {
                Ok(())
            }
        };
        if let Some(start) = self.start {
            check("start", start.len(), n)?;
        }
        check("status", self.status.len(), n)?;
        check("x", self.x.nrows(), n)?;
        if let Some(means) = self.means {
            check("means", means.len(), self.x.ncols())?;
        }
        check("weights", self.weights.len(), n)?;
        check("risk", self.risk.len(), n)?;
        Ok(())
    }

    /// `x[i, k] - means[k]`.
    fn centered(&self, i: usize, k: usize) -> f64 {
        self.x[(i, k)] - self.means.map_or(0.0, |means| means[k])
    }
}

/// The pieces of one stratum's curve (R's `agsurv()` list).
#[pyclass(module = "survival._survival", skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct AgsurvCurve {
    /// Number of observations in the stratum.
    #[pyo3(get)]
    pub n: usize,
    /// Sorted unique stop times (events and censorings).
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Weighted number of events at each time.
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    /// Weighted number at risk at each time.
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    /// Weighted number censored at each time.
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    /// Hazard increment at each time.
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    /// Cumulative hazard.
    #[pyo3(get)]
    pub cumhaz: Vec<f64>,
    /// Increment of the variance of the cumulative hazard at each time,
    /// the part that would remain if the coefficients were known.
    #[pyo3(get)]
    pub varhaz: Vec<f64>,
    /// Unweighted number of deaths at each time.
    #[pyo3(get)]
    pub ndeath: Vec<usize>,
    /// `ntime x nvar`: (weighted mean covariate of those at risk) times the
    /// hazard increment, the second part of the variance.
    pub xbar: Array2<f64>,
    /// Kalbfleisch-Prentice survival increments (`survtype == 1` only).
    #[pyo3(get)]
    pub surv: Option<Vec<f64>>,
}

#[pymethods]
impl AgsurvCurve {
    /// `xbar`, one list per time.
    #[getter(xbar)]
    fn xbar_rows(&self) -> Vec<Vec<f64>> {
        matrix_rows(&self.xbar)
    }
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

/// Port of `src/agsurv4.c`: the Kalbfleisch-Prentice survival increment at
/// each unique time.  `risk` and `weights` are those of the deaths in time
/// order; `denom` is the weighted risk sum at each time.  A single death
/// solves the estimating equation in closed form, tied deaths by bisection.
fn agsurv4(ndeath: &[usize], risk: &[f64], weights: &[f64], denom: &[f64]) -> Vec<f64> {
    let mut km = vec![1.0; ndeath.len()];
    let mut j = 0;
    for (i, &deaths) in ndeath.iter().enumerate() {
        if deaths == 1 {
            // Subtracting entry risk sets can put a terminal death a few ulps
            // above its denominator. A fractional power of that negative
            // rounding residue is NaN; the survival increment is zero.
            km[i] = (1.0 - weights[j] * risk[j] / denom[i])
                .clamp(0.0, 1.0)
                .powf(1.0 / risk[j]);
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

/// Port of `R/agsurv.R`: the survival-curve components of one stratum,
/// all rows of `data`.
pub fn agsurv(
    data: &AgsurvData<'_>,
    survtype: CoxSurvType,
    vartype: CoxSurvType,
) -> SurvivalResult<AgsurvCurve> {
    data.validate()?;
    let rows: Vec<usize> = (0..data.stop.len()).collect();
    Ok(agsurv_of_rows(data, &rows, survtype, vartype))
}

/// [`agsurv`] for the stratum made of `rows` (indices into `data`), which
/// lets a stratified fit reuse its design matrix without copying.  A
/// stratum without rows has an empty curve (`n = 0`, no times), as
/// `survfit.coxph` keeps a stratum that `start.time` emptied.
pub fn agsurv_rows(
    data: &AgsurvData<'_>,
    rows: &[usize],
    survtype: CoxSurvType,
    vartype: CoxSurvType,
) -> SurvivalResult<AgsurvCurve> {
    data.validate()?;
    if let Some(&row) = rows.iter().find(|&&row| row >= data.stop.len()) {
        return Err(SurvivalError::invalid_input(format!(
            "agsurv: row {row} is out of range"
        )));
    }
    Ok(agsurv_of_rows(data, rows, survtype, vartype))
}

fn agsurv_of_rows(
    data: &AgsurvData<'_>,
    rows: &[usize],
    survtype: CoxSurvType,
    vartype: CoxSurvType,
) -> AgsurvCurve {
    let n = rows.len();
    let nvar = data.x.ncols();
    let time = sort_unique(rows.iter().map(|&i| data.stop[i]));
    let ntime = time.len();

    let mut n_event = vec![0.0; ntime];
    let mut n_censor = vec![0.0; ntime];
    let mut nrisk = vec![0.0; ntime];
    let mut irisk = vec![0.0; ntime];
    let mut ndeath = vec![0usize; ntime];
    let mut xsum = Array2::zeros((ntime, nvar));
    let mut xsum2 = Array2::zeros((ntime, nvar));
    let mut erisk = vec![0.0; ntime];
    for &i in rows {
        let weighted_risk = data.weights[i] * data.risk[i];
        let g = group_index(&time, data.stop[i]);
        let death = data.status[i] == 1;
        if death {
            n_event[g] += data.weights[i];
            ndeath[g] += 1;
            erisk[g] += weighted_risk;
            for k in 0..nvar {
                xsum2[(g, k)] += weighted_risk * data.centered(i, k);
            }
        } else {
            n_censor[g] += data.weights[i];
        }
        nrisk[g] += weighted_risk;
        irisk[g] += data.weights[i];
        for k in 0..nvar {
            xsum[(g, k)] += weighted_risk * data.centered(i, k);
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
        let etime = sort_unique(rows.iter().map(|&i| start[i]));
        let mut esum = vec![0.0; etime.len()];
        let mut ewt = vec![0.0; etime.len()];
        let mut xout = Array2::zeros((etime.len(), nvar));
        for &i in rows {
            let weighted_risk = data.weights[i] * data.risk[i];
            let g = group_index(&etime, start[i]);
            esum[g] += weighted_risk;
            ewt[g] += data.weights[i];
            for k in 0..nvar {
                xout[(g, k)] += weighted_risk * data.centered(i, k);
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
        let mut deaths: Vec<usize> = rows
            .iter()
            .copied()
            .filter(|&i| data.status[i] == 1)
            .collect();
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

    AgsurvCurve {
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
    }
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

/// The Kalbfleisch-Prentice survival increments of a curve (`agsurv`'s
/// `surv`) when `survtype` is that estimate, `None` for the others.
fn kp_increments(curve: &AgsurvCurve, survtype: CoxSurvType) -> SurvivalResult<Option<&[f64]>> {
    match (survtype, &curve.surv) {
        (CoxSurvType::KalbfleischPrentice, Some(increments)) => Ok(Some(increments)),
        (CoxSurvType::KalbfleischPrentice, None) => Err(SurvivalError::invalid_input(
            "Kalbfleisch-Prentice curves need survival increments (build them with survtype KP)",
        )),
        _ => Ok(None),
    }
}

/// Baseline survival of one stratum: `cumprod(surv)` for the
/// Kalbfleisch-Prentice estimate, `exp(-cumhaz)` otherwise.
fn baseline_survival(curve: &AgsurvCurve, survtype: CoxSurvType) -> SurvivalResult<Vec<f64>> {
    Ok(match kp_increments(curve, survtype)? {
        Some(increments) => {
            let mut running = 1.0;
            increments
                .iter()
                .map(|&value| {
                    running *= value;
                    running
                })
                .collect()
        }
        None => curve.cumhaz.iter().map(|&h| (-h).exp()).collect(),
    })
}

/// Running pieces of `cumsum(varhaz) + dt' V dt`. Only the current
/// covariate sums are needed; retaining one row per time costs O(ntime * nvar).
struct CurveVariance<'a> {
    varmat: &'a Array2<f64>,
    dt: Vec<f64>,
    baseline: f64,
}

impl<'a> CurveVariance<'a> {
    fn new(varmat: &'a Array2<f64>, nvar: usize) -> SurvivalResult<Self> {
        if varmat.dim() != (nvar, nvar) {
            return Err(SurvivalError::invalid_input(format!(
                "varmat must have shape ({nvar}, {nvar})"
            )));
        }
        Ok(Self {
            varmat,
            dt: vec![0.0; nvar],
            baseline: 0.0,
        })
    }

    fn reset(&mut self) {
        self.dt.fill(0.0);
        self.baseline = 0.0;
    }

    fn accumulate(&mut self, varhaz: f64, increments: impl Iterator<Item = f64>) -> f64 {
        self.baseline += varhaz;
        for (total, increment) in self.dt.iter_mut().zip(increments) {
            *total += increment;
        }
        // Preserve the original summation order, including off-diagonal terms.
        let mut coefficient = 0.0;
        for (i, &left) in self.dt.iter().enumerate() {
            for (j, &right) in self.dt.iter().enumerate() {
                coefficient += left * self.varmat[(i, j)] * right;
            }
        }
        self.baseline + coefficient
    }
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
    let base_surv = baseline_survival(curve, survtype)?;
    let mut surv = Array2::zeros((ntime, m));
    let mut cumhaz = Array2::zeros((ntime, m));
    let mut std_err = varmat.map(|_| Array2::zeros((ntime, m)));
    let mut variance = varmat.map(|v| CurveVariance::new(v, nvar)).transpose()?;
    for i in 0..m {
        for g in 0..ntime {
            surv[(g, i)] = base_surv[g].powf(risk2[i]);
            cumhaz[(g, i)] = curve.cumhaz[g] * risk2[i];
        }
        if let (Some(variance), Some(std_err)) = (variance.as_mut(), std_err.as_mut()) {
            variance.reset();
            for g in 0..ntime {
                let increments =
                    (0..nvar).map(|k| curve.hazard[g] * x2[(i, k)] - curve.xbar[(g, k)]);
                let var = variance.accumulate(curve.varhaz[g], increments);
                std_err[(g, i)] = (var * risk2[i] * risk2[i]).sqrt();
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
    let mut cumhaz = Vec::new();
    let mut surv = Vec::new();
    let mut std_err = varmat.map(|_| Vec::new());
    let mut variance = varmat.map(|v| CurveVariance::new(v, nvar)).transpose()?;
    let mut running_hazard = 0.0;
    let mut running_surv = 1.0;
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
        if curve.xbar.ncols() != nvar {
            return Err(SurvivalError::invalid_input(
                "all strata must have the same number of covariates",
            ));
        }
        if interval.x2.len() != nvar {
            return Err(SurvivalError::invalid_input(format!(
                "interval covariates have {} values but the model has {nvar}",
                interval.x2.len()
            )));
        }
        // onecurve's `slist$surv[indx]^risk2[i]`
        let increments = kp_increments(curve, survtype)?;
        // The baseline times are sorted. Look up (start, stop] once instead
        // of scanning the entire stratum for every interval of every subject.
        let first = curve.time.partition_point(|&t| t <= interval.start);
        let end = curve.time.partition_point(|&t| t <= interval.stop);
        for g in first..end {
            time.push(toffset + curve.time[g]);
            n_event.push(curve.n_event[g]);
            n_risk.push(curve.n_risk[g]);
            n_censor.push(curve.n_censor[g]);
            running_hazard += curve.hazard[g] * interval.risk2;
            cumhaz.push(running_hazard);
            surv.push(if let Some(increments) = increments {
                running_surv *= increments[g].powf(interval.risk2);
                running_surv
            } else {
                (-running_hazard).exp()
            });
            if let (Some(variance), Some(std_err)) = (variance.as_mut(), std_err.as_mut()) {
                let increments = (0..nvar).map(|k| {
                    (curve.hazard[g] * interval.x2[k] - curve.xbar[(g, k)]) * interval.risk2
                });
                let var = variance.accumulate(
                    curve.varhaz[g] * interval.risk2 * interval.risk2,
                    increments,
                );
                std_err.push(var.sqrt());
            }
        }
    }
    let ntime = time.len();
    let column = |values| Array2::from_shape_vec((ntime, 1), values).expect("one value per time");
    Ok(CoxSurvCurve {
        n: curves[first.stratum].n,
        time,
        n_risk,
        n_event,
        n_censor,
        surv: column(surv),
        cumhaz: column(cumhaz),
        std_err: std_err.map(column),
    })
}

/// Cumulative hazard of a curve just after `t`: `c(0, cumhaz)[findInterval(t, time) + 1]`.
pub fn cumhaz_at(curve: &AgsurvCurve, t: f64) -> f64 {
    step_at(&curve.time, &curve.cumhaz, t, 0.0)
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
    match find_interval(&curve.time, t, false) {
        0 => Array1::zeros(integrated.cum_xbar.ncols()),
        index => integrated.cum_xbar.row(index - 1).to_owned(),
    }
}

/// `agsurv(y, x, wt, risk, survtype, vartype)` on all rows as one stratum.
/// `y` has 2 or 3 columns ending in a 0/1 status (with start < stop), `x`
/// one row per observation; the weights must be finite and non-negative and
/// the risks finite and positive.
pub(super) fn prepare_baseline(
    y: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    weights: &[f64],
    risk: &[f64],
) -> SurvivalResult<PreparedBaseline> {
    let (n, ycols) = y.dim();
    if n == 0 {
        return Err(ValidationError::Empty {
            name: "y".to_string(),
        }
        .into());
    }
    if ycols != 2 && ycols != 3 {
        return Err(SurvivalError::invalid_input("y must have 2 or 3 columns"));
    }
    if let Some(((i, _), value)) = y.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "y row {i} contains non-finite value {value}"
        )));
    }
    let start: Option<Vec<f64>> = (ycols == 3).then(|| y.column(0).to_vec());
    let stop: Vec<f64> = y.column(ycols - 2).to_vec();
    let status: Vec<f64> = y.column(ycols - 1).to_vec();
    validate_binary_f64(&status, "y status")?;
    if let Some(i) = start
        .as_deref()
        .and_then(|start| (0..n).find(|&i| start[i] >= stop[i]))
    {
        return Err(SurvivalError::invalid_input(format!(
            "y start must be less than stop at row {i}"
        )));
    }
    validate_length(n, x.nrows(), "x")?;
    if let Some(((i, _), value)) = x.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(format!(
            "x row {i} contains non-finite value {value}"
        )));
    }
    validate_length(n, weights.len(), "weights")?;
    validate_finite(weights, "weights")?;
    validate_non_negative(weights, "weights")?;
    validate_length(n, risk.len(), "risk")?;
    validate_finite(risk, "risk")?;
    validate_positive(risk, "risk")?;
    Ok(PreparedBaseline {
        start,
        stop,
        status: status.iter().map(|&value| value as i32).collect(),
    })
}

pub(super) struct PreparedBaseline {
    pub start: Option<Vec<f64>>,
    pub stop: Vec<f64>,
    pub status: Vec<i32>,
}

impl PreparedBaseline {
    pub fn data<'a>(
        &'a self,
        x: ArrayView2<'a, f64>,
        weights: &'a [f64],
        risk: &'a [f64],
    ) -> AgsurvData<'a> {
        AgsurvData {
            start: self.start.as_deref(),
            stop: &self.stop,
            status: &self.status,
            x,
            means: None,
            weights,
            risk,
        }
    }
}

pub(super) fn check_baseline(curve: &AgsurvCurve) -> SurvivalResult<()> {
    if let Some(g) = curve.hazard.iter().position(|h| h.is_nan()) {
        return Err(SurvivalError::invalid_input(format!(
            "risk-set denominator must be positive at time {}",
            curve.time[g]
        )));
    }
    Ok(())
}

fn baseline_curve(
    y: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    weights: &[f64],
    risk: &[f64],
    survtype: i32,
    vartype: i32,
) -> SurvivalResult<AgsurvCurve> {
    let prepared = prepare_baseline(y, x, weights, risk)?;
    let code = |name: &str, code: i32| {
        CoxSurvType::from_code(code)
            .ok_or_else(|| SurvivalError::invalid_input(format!("{name} must be 1, 2, or 3")))
    };
    let survtype = code("survtype", survtype)?;
    let vartype = code("vartype", vartype)?;

    let curve = agsurv(&prepared.data(x, weights, risk), survtype, vartype)?;
    // Zero weights are allowed, but a zero-weight risk set has no defined hazard.
    check_baseline(&curve)?;
    Ok(curve)
}

/// `agsurv(y, x, wt, risk, survtype, vartype)` for one stratum, as
/// `coxsurv.fit` calls it: the unique times with their weighted counts, the
/// hazard, cumulative hazard and hazard variance increments, the number of
/// deaths, the `xbar` rows and (survtype 1) the Kalbfleisch-Prentice
/// increments.
#[pyfunction]
pub fn cox_survfit_baseline(
    py: Python<'_>,
    y: FloatMatrix,
    x: FloatMatrix,
    weights: FloatVec,
    risk: FloatVec,
    survtype: i32,
    vartype: i32,
) -> PyResult<AgsurvCurve> {
    Ok(py.detach(|| baseline_curve(y.view(), x.view(), &weights, &risk, survtype, vartype))?)
}

/// `step_at` at each of `requested_times`, the way `summary.survfit(fit,
/// times, extend = TRUE)` reads a curve (`initial` is the value before the
/// first time).
#[pyfunction]
pub fn step_values_at(
    times: Vec<f64>,
    values: Vec<f64>,
    requested_times: Vec<f64>,
    initial: f64,
) -> PyResult<Vec<f64>> {
    validate_length(times.len(), values.len(), "values")?;
    validate_finite(&times, "times")?;
    validate_sorted(&times, "times")?;
    validate_finite(&values, "values")?;
    validate_finite(&requested_times, "requested_times")?;
    validate_finite(&[initial], "initial")?;
    Ok(requested_times
        .iter()
        .map(|&t| step_at(&times, &values, t, initial))
        .collect())
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
            means: None,
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
            means: None,
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
            means: None,
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
            means: None,
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
    fn terminal_kp_death_is_zero_after_risk_set_roundoff() {
        let risk: f64 = 0.7;
        let rounded_below = f64::from_bits(risk.to_bits() - 1);
        assert_eq!(agsurv4(&[1], &[risk], &[1.0], &[rounded_below]), vec![0.0]);
        assert!(agsurv4(&[1], &[risk], &[1.0], &[f64::NAN])[0].is_nan());
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
            means: None,
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

    fn example_curves() -> [AgsurvCurve; 2] {
        [
            AgsurvCurve {
                n: 5,
                time: vec![1.0, 2.0, 4.0],
                n_event: vec![1.0, 2.0, 1.0],
                n_risk: vec![5.0, 4.0, 2.0],
                n_censor: vec![0.0, 0.0, 1.0],
                hazard: vec![0.1, 0.2, 0.3],
                cumhaz: vec![0.1, 0.3, 0.6],
                varhaz: vec![0.01, 0.04, 0.09],
                ndeath: vec![1, 2, 1],
                xbar: arr2(&[[0.01, 0.02], [0.01, 0.03], [0.04, 0.05]]),
                surv: Some(vec![0.9, 0.8, 0.7]),
            },
            AgsurvCurve {
                n: 8,
                time: vec![0.5, 2.0, 3.0, 6.0],
                n_event: vec![1.0, 2.0, 3.0, 1.0],
                n_risk: vec![8.0, 7.0, 5.0, 2.0],
                n_censor: vec![0.0, 0.0, 0.0, 1.0],
                hazard: vec![0.2, 0.3, 0.4, 0.5],
                cumhaz: vec![0.2, 0.5, 0.9, 1.4],
                varhaz: vec![0.04, 0.09, 0.16, 0.25],
                ndeath: vec![1, 2, 3, 1],
                xbar: arr2(&[[0.01, 0.02], [0.08, 0.02], [0.1, 0.06], [0.1, 0.2]]),
                surv: Some(vec![0.8, 0.7, 0.6, 0.5]),
            },
        ]
    }

    #[test]
    fn individual_intervals_keep_boundaries_offsets_and_covariance_across_strata() {
        let curves = example_curves();
        let varmat = arr2(&[[0.5, 0.1], [0.1, 0.25]]);
        let intervals = [
            IndividualInterval {
                start: 1.0,
                stop: 2.0,
                stratum: 0,
                x2: &[0.5, -1.0],
                risk2: 1.0,
            },
            // No baseline times here. It still changes the time offset but
            // contributes neither hazard nor coefficient uncertainty.
            IndividualInterval {
                start: 10.0,
                stop: 11.0,
                stratum: 1,
                x2: &[100.0, -100.0],
                risk2: 10.0,
            },
            IndividualInterval {
                start: 1.0,
                stop: 3.0,
                stratum: 1,
                x2: &[1.0, 0.5],
                risk2: 2.0,
            },
        ];
        for survtype in [
            CoxSurvType::KalbfleischPrentice,
            CoxSurvType::Breslow,
            CoxSurvType::Efron,
        ] {
            for se_fit in [false, true] {
                let result =
                    individual_curve(&curves, survtype, &intervals, se_fit.then_some(&varmat))
                        .unwrap();
                assert_eq!(result.n, 5);
                assert_eq!(result.time, vec![2.0, 4.0, 5.0]);
                assert_eq!(result.n_risk, vec![4.0, 7.0, 5.0]);
                assert_eq!(result.n_event, vec![2.0, 2.0, 3.0]);
                assert_eq!(result.n_censor, vec![0.0; 3]);
                let expected_dt = [(0.09, -0.23), (0.53, 0.03), (1.13, 0.31)];
                let baseline_variance = [0.04, 0.4, 1.04];
                let hazard: [f64; 3] = [0.2, 0.8, 1.6];
                let kp = [0.8, 0.8 * 0.49, 0.8 * 0.49 * 0.36];
                assert_eq!(result.std_err.is_some(), se_fit);
                for g in 0..3 {
                    assert_close(result.cumhaz[(g, 0)], hazard[g]);
                    assert_close(
                        result.surv[(g, 0)],
                        if survtype == CoxSurvType::KalbfleischPrentice {
                            kp[g]
                        } else {
                            (-hazard[g]).exp()
                        },
                    );
                    if let Some(se) = &result.std_err {
                        let (a, b) = expected_dt[g];
                        let var: f64 =
                            baseline_variance[g] + 0.5 * a * a + 0.2 * a * b + 0.25 * b * b;
                        assert_close(se[(g, 0)], var.sqrt());
                    }
                }
            }
        }
    }

    #[test]
    fn expansion_resets_coefficient_uncertainty_for_each_prediction_row() {
        let curve = &example_curves()[0];
        let x2 = arr2(&[[0.5, -1.0], [1.0, 0.5], [0.5, -1.0]]);
        let varmat = arr2(&[[0.5, 0.1], [0.1, 0.25]]);
        let result = expand_curve(
            curve,
            CoxSurvType::Efron,
            x2.view(),
            &[1.0, 2.0, 1.0],
            Some(&varmat),
        )
        .unwrap();
        let se = result.std_err.unwrap();
        // At the final time, H=.6, summed xbar=(.06,.10), sum(varhaz)=.14.
        // Thus dt=(.24,-.70) for row 0, (.54,.20) for row 1.
        for (col, a, b, risk) in [(0, 0.24, -0.70, 1.0), (1, 0.54, 0.20, 2.0)] {
            let variance: f64 = (0.14 + 0.5 * a * a + 0.2 * a * b + 0.25 * b * b) * risk * risk;
            assert_close(se[(2, col)], variance.sqrt());
        }
        assert_eq!(se.column(0), se.column(2));
    }

    #[test]
    fn curve_variance_supports_zero_covariates_and_empty_outputs() {
        let mut curves = example_curves();
        for curve in &mut curves {
            curve.xbar = Array2::zeros((curve.time.len(), 0));
        }
        let covariance = Array2::zeros((0, 0));
        let x2 = Array2::zeros((1, 0));
        let breslow = CoxSurvType::Breslow;
        let expanded =
            expand_curve(&curves[0], breslow, x2.view(), &[2.0], Some(&covariance)).unwrap();
        assert_close(expanded.std_err.unwrap()[(2, 0)], (0.14_f64 * 4.0).sqrt());
        let interval = IndividualInterval {
            start: 1.0,
            stop: 2.0,
            stratum: 1,
            x2: &[],
            risk2: 2.0,
        };
        let selected = individual_curve(
            &curves,
            breslow,
            std::slice::from_ref(&interval),
            Some(&covariance),
        )
        .unwrap();
        assert_eq!(selected.time, vec![2.0]);
        assert_close(selected.std_err.unwrap()[(0, 0)], 0.6);
        for (start, stop) in [(7.0, 8.0), (-2.0, -1.0), (3.0, 3.0)] {
            let empty = individual_curve(
                &curves,
                breslow,
                &[IndividualInterval {
                    start,
                    stop,
                    ..interval.clone()
                }],
                Some(&covariance),
            )
            .unwrap();
            assert_eq!(empty.n, 8);
            assert_eq!(empty.surv.dim(), (0, 1));
            assert_eq!(empty.cumhaz.dim(), (0, 1));
            assert_eq!(empty.std_err.unwrap().dim(), (0, 1));
        }
        let no_newdata = Array2::zeros((0, 0));
        let empty = expand_curve(
            &curves[0],
            breslow,
            no_newdata.view(),
            &[],
            Some(&covariance),
        )
        .unwrap();
        assert_eq!(empty.surv.dim(), (3, 0));
        assert_eq!(empty.std_err.unwrap().dim(), (3, 0));
    }

    #[test]
    fn curve_variance_rejects_wrong_dimensions_without_panicking() {
        let mut curves = example_curves();
        let breslow = CoxSurvType::Breslow;
        let bad = Array2::zeros((1, 2));
        let x2 = arr2(&[[0.0, 0.0]]);
        let interval = IndividualInterval {
            start: 0.0,
            stop: 6.0,
            stratum: 1,
            x2: &[0.0, 0.0],
            risk2: 1.0,
        };
        for error in [
            expand_curve(&curves[0], breslow, x2.view(), &[1.0], Some(&bad)).unwrap_err(),
            individual_curve(
                &curves,
                breslow,
                std::slice::from_ref(&interval),
                Some(&bad),
            )
            .unwrap_err(),
        ] {
            assert!(error.to_string().contains("varmat must have shape (2, 2)"));
        }
        curves[1].xbar = Array2::zeros((4, 1));
        assert!(
            individual_curve(&curves, breslow, &[interval], None)
                .unwrap_err()
                .to_string()
                .contains("all strata must have the same number of covariates")
        );
    }

    #[test]
    fn a_stratum_without_rows_has_an_empty_curve() {
        let stop = [1.0, 2.0];
        let status = [1, 0];
        let x = arr2(&[[0.0], [1.0]]);
        let weights = [1.0; 2];
        let risk = [1.0; 2];
        let data = AgsurvData {
            start: None,
            stop: &stop,
            status: &status,
            x: x.view(),
            means: None,
            weights: &weights,
            risk: &risk,
        };
        let kp = CoxSurvType::KalbfleischPrentice;
        let curve = agsurv_rows(&data, &[], kp, kp).unwrap();
        assert_eq!(curve.n, 0);
        assert!(curve.time.is_empty() && curve.cumhaz.is_empty());
        assert_eq!(curve.xbar.dim(), (0, 1));
        assert_eq!(curve.surv, Some(Vec::new()));
        let expanded = expand_curve(&curve, kp, x.view(), &[1.0, 2.0], None).unwrap();
        assert_eq!(expanded.surv.dim(), (0, 2));
        assert!(agsurv_rows(&data, &[2], kp, kp).is_err());
    }

    #[test]
    fn kalbfleisch_prentice_curves_need_their_increments() {
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
            means: None,
            weights: &weights,
            risk: &risk,
        };
        let kp = CoxSurvType::KalbfleischPrentice;
        let breslow = agsurv(&data, CoxSurvType::Breslow, CoxSurvType::Breslow).unwrap();
        let x2 = [0.5];
        let interval = IndividualInterval {
            start: 0.0,
            stop: 3.0,
            stratum: 0,
            x2: &x2,
            risk2: 2.0,
        };
        let message = "Kalbfleisch-Prentice curves need survival increments";
        let intervals = [interval];
        let error =
            individual_curve(std::slice::from_ref(&breslow), kp, &intervals, None).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        let x2_rows = arr2(&[[0.5]]);
        let error = expand_curve(&breslow, kp, x2_rows.view(), &[2.0], None).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");

        // with them, onecurve's running product of surv^risk2
        let curve = agsurv(&data, kp, kp).unwrap();
        let increments = curve.surv.clone().unwrap();
        let stitched = individual_curve(&[curve], kp, &intervals, None).unwrap();
        assert_close(stitched.surv[(0, 0)], increments[0].powf(2.0));
        assert_close(
            stitched.surv[(1, 0)],
            (increments[0] * increments[1]).powf(2.0),
        );
    }

    #[test]
    fn step_functions_are_right_continuous() {
        let times = [1.0, 3.0, 5.0];
        let values = [10.0, 30.0, 50.0];
        let read = |t| step_at(&times, &values, t, -1.0);
        assert_eq!(
            [read(0.5), read(1.0), read(4.0), read(6.0)],
            [-1.0, 10.0, 30.0, 50.0]
        );
        assert_eq!(
            step_values_at(times.to_vec(), values.to_vec(), vec![6.0, 0.5, 3.0], 1.0).unwrap(),
            vec![50.0, 1.0, 30.0]
        );
        assert!(step_values_at(vec![3.0, 1.0], vec![1.0, 2.0], vec![2.0], 0.0).is_err());
        assert!(step_values_at(vec![1.0], vec![1.0, 2.0], vec![2.0], 0.0).is_err());
        assert!(step_values_at(vec![1.0], vec![1.0], vec![f64::NAN], 0.0).is_err());
    }

    #[test]
    fn baseline_binding_matches_weighted_risk_sets() {
        let curve = baseline_curve(
            arr2(&[[1.0, 1.0], [2.0, 1.0], [2.0, 0.0], [3.0, 1.0]]).view(),
            arr2(&[[0.0], [1.0], [2.0], [3.0]]).view(),
            &[1.0, 2.0, 1.0, 1.0],
            &[1.0, 2.0, 1.0, 0.5],
            2,
            2,
        )
        .unwrap();
        assert_eq!(curve.time, vec![1.0, 2.0, 3.0]);
        assert_eq!(curve.n_event, vec![1.0, 2.0, 1.0]);
        assert_eq!(curve.n_censor, vec![0.0, 1.0, 0.0]);
        assert_eq!(curve.n_risk, vec![5.0, 4.0, 1.0]);
        assert_eq!(curve.ndeath, vec![1, 1, 1]);
        assert_close(curve.hazard[0], 1.0 / 6.5);
        assert_close(curve.hazard[1], 2.0 / 5.5);
        assert_close(curve.hazard[2], 2.0);
        assert_close(curve.xbar[(0, 0)], 7.5 / 6.5_f64.powi(2));

        let counting = baseline_curve(
            arr2(&[[0.0, 2.0, 1.0], [1.0, 3.0, 1.0], [2.0, 4.0, 0.0]]).view(),
            arr2(&[[0.0], [1.0], [2.0]]).view(),
            &[1.0; 3],
            &[1.0, 2.0, 4.0],
            3,
            3,
        )
        .unwrap();
        assert_eq!(counting.n_risk, vec![2.0, 2.0, 1.0]);
        assert_close(counting.hazard[0], 1.0 / 3.0);
        assert_close(counting.hazard[1], 1.0 / 6.0);
        assert_eq!(counting.hazard[2], 0.0);
    }

    #[test]
    fn baseline_binding_rejects_invalid_inputs() {
        let message = |y: &[f64], weights: &[f64], survtype: i32| {
            let y = ArrayView2::from_shape((1, y.len()), y).unwrap();
            baseline_curve(y, arr2(&[[0.0]]).view(), weights, &[1.0], survtype, 2)
                .unwrap_err()
                .to_string()
        };
        assert!(message(&[1.0, 1.0, 0.0], &[1.0], 2).contains("start must be less than stop"));
        assert!(message(&[1.0, 2.0], &[1.0], 2).contains("y status must contain only 0/1 values"));
        assert!(message(&[1.0, 1.0], &[1.0], 4).contains("survtype must be 1, 2, or 3"));
        assert!(message(&[1.0, 1.0], &[-1.0], 2).contains("weights contains negative"));
        assert!(message(&[1.0, 1.0], &[0.0], 2).contains("risk-set denominator must be positive"));
        assert!(message(&[1.0], &[1.0], 2).contains("y must have 2 or 3 columns"));
    }
}
