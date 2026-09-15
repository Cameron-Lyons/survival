//! Kaplan-Meier and Fleming-Harrington survival curves for right-censored
//! and counting-process data: the port of R's `survfitKM` (`R/survfitKM.R`)
//! and its C kernel `survfitkm` (`src/survfitkm.c`), survival 3.8-11/12.
//!
//! [`survfitkm`] is the only survival/cumulative-hazard engine in the crate:
//! the Nelson-Aalen facade, the pseudo-value and residual code, the
//! summary helpers, the G-rho weights of `survdiff`, the censoring
//! distribution of the Brier score, the Turnbull EM and the
//! `validation` summaries all read its [`SurvfitKMResult`].

use super::survfit_confint::{ConfLower, ConfType, survfit_confint, validate_conf_int};
use crate::constants::PARALLEL_THRESHOLD_LARGE;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_non_empty,
    validate_non_negative,
};
use ndarray::Array2;
use pyo3::prelude::*;
use rayon::prelude::*;

/// The `stype` argument of `survfit`: how the survival curve is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurvType {
    /// `stype = 1`: the Kaplan-Meier product limit.
    #[default]
    KaplanMeier,
    /// `stype = 2`: `exp(-cumulative hazard)`.
    ExpCumhaz,
}

impl SurvType {
    pub fn from_code(code: i32) -> SurvivalResult<Self> {
        match code {
            1 => Ok(Self::KaplanMeier),
            2 => Ok(Self::ExpCumhaz),
            _ => Err(SurvivalError::invalid_input("stype must be 1 or 2")),
        }
    }
}

/// The `ctype` argument of `survfit`: how tied events enter the hazard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HazardType {
    /// `ctype = 1`: Nelson-Aalen, one increment `d / n` per time.
    #[default]
    NelsonAalen,
    /// `ctype = 2`: Fleming-Harrington, `d` tied events enter one at a time.
    FlemingHarrington,
}

impl HazardType {
    pub fn from_code(code: i32) -> SurvivalResult<Self> {
        match code {
            1 => Ok(Self::NelsonAalen),
            2 => Ok(Self::FlemingHarrington),
            _ => Err(SurvivalError::invalid_input("ctype must be 1 or 2")),
        }
    }
}

/// The `influence` argument of `survfit`: which per-cluster influence
/// matrices to return (`1 * survival + 2 * cumulative hazard`, as R).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InfluenceRequest {
    #[default]
    None,
    Survival,
    Cumhaz,
    Both,
}

impl InfluenceRequest {
    pub fn from_code(code: i32) -> SurvivalResult<Self> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Survival),
            2 => Ok(Self::Cumhaz),
            3 => Ok(Self::Both),
            _ => Err(SurvivalError::invalid_input(
                "influence argument must be 0, 1, 2, or 3",
            )),
        }
    }

    fn survival(self) -> bool {
        matches!(self, Self::Survival | Self::Both)
    }

    fn cumhaz(self) -> bool {
        matches!(self, Self::Cumhaz | Self::Both)
    }
}

/// The arguments of `survfitKM` beyond the data.
#[derive(Debug, Clone)]
pub struct SurvfitKMOptions {
    pub stype: SurvType,
    pub ctype: HazardType,
    /// `se.fit`: compute standard errors (and hence confidence limits).
    pub se_fit: bool,
    /// `conf.int`: the confidence level.
    pub conf_int: f64,
    pub conf_type: ConfType,
    pub conf_lower: ConfLower,
    /// `start.time`: observations ending before it are dropped and the
    /// curves start there.
    pub start_time: Option<f64>,
    /// `robust`: `None` leaves the choice to R's rule (a cluster, non-integer
    /// weights, or an id with more than one event per subject turn it on).
    pub robust: Option<bool>,
    pub influence: InfluenceRequest,
    /// `entry`: report the number entering the risk set at each time
    /// (counting-process data with an `id` only).
    pub entry: bool,
    /// `timefix`: bin times that differ by less than `sqrt(.Machine$double.eps)`
    /// (`aeqSurv`) before anything else.
    pub timefix: bool,
    /// Estimate the censoring distribution instead (the `reverse` flag of
    /// `survfitkm.c`): censorings become the events and the deaths tied at
    /// the same time leave the risk set first.
    pub reverse: bool,
}

impl Default for SurvfitKMOptions {
    fn default() -> Self {
        Self {
            stype: SurvType::KaplanMeier,
            ctype: HazardType::NelsonAalen,
            se_fit: true,
            conf_int: 0.95,
            conf_type: ConfType::Log,
            conf_lower: ConfLower::Usual,
            start_time: None,
            robust: None,
            influence: InfluenceRequest::None,
            entry: false,
            timefix: true,
            reverse: false,
        }
    }
}

/// The data of a `survfit(Surv(...) ~ strata, weights, id, cluster)` call.
///
/// `strata` holds one integer code per observation; curves are produced for
/// the distinct codes in ascending order (R's factor levels).  `id` links
/// the rows of one subject in counting-process data and `cluster` groups
/// observations for the robust variance; both are arbitrary integer labels.
#[derive(Debug, Clone)]
pub struct SurvfitKMData {
    pub start: Option<Vec<f64>>,
    pub time: Vec<f64>,
    pub status: Vec<i32>,
    pub weights: Option<Vec<f64>>,
    pub strata: Option<Vec<i32>>,
    pub id: Option<Vec<i64>>,
    pub cluster: Option<Vec<i64>>,
}

impl SurvfitKMData {
    /// Right-censored data without weights or grouping.
    pub fn right_censored(time: Vec<f64>, status: Vec<i32>) -> SurvivalResult<Self> {
        Self::try_new(None, time, status, None, None, None, None)
    }

    pub fn try_new(
        start: Option<Vec<f64>>,
        time: Vec<f64>,
        status: Vec<i32>,
        weights: Option<Vec<f64>>,
        strata: Option<Vec<i32>>,
        id: Option<Vec<i64>>,
        cluster: Option<Vec<i64>>,
    ) -> SurvivalResult<Self> {
        validate_non_empty(&time, "time")?;
        validate_finite(&time, "time")?;
        validate_length(time.len(), status.len(), "status")?;
        validate_binary_i32(&status, "status")?;
        if let Some(start) = &start {
            validate_length(time.len(), start.len(), "start")?;
            validate_finite(start, "start")?;
            if let Some(index) = start.iter().zip(&time).position(|(s, t)| s >= t) {
                return Err(SurvivalError::invalid_input(format!(
                    "Stop time must be > start time (observation {index})"
                )));
            }
        }
        if let Some(weights) = &weights {
            validate_length(time.len(), weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            validate_non_negative(weights, "weights")?;
        }
        if let Some(strata) = &strata {
            validate_length(time.len(), strata.len(), "strata")?;
        }
        if let Some(id) = &id {
            validate_length(time.len(), id.len(), "id")?;
        }
        if let Some(cluster) = &cluster {
            validate_length(time.len(), cluster.len(), "cluster")?;
        }
        Ok(Self {
            start,
            time,
            status,
            weights,
            strata,
            id,
            cluster,
        })
    }

    fn n(&self) -> usize {
        self.time.len()
    }
}

/// Unweighted counts, reported alongside the weighted ones when case
/// weights are present (R's `counts` component).
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitCounts {
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<f64>>,
}

/// One curve's influence matrix: `values[k][t]` is the influence of cluster
/// `cluster[k]` on the estimate at the curve's `t`-th time.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitInfluence {
    #[pyo3(get)]
    pub cluster: Vec<i64>,
    #[pyo3(get)]
    pub values: Vec<Vec<f64>>,
}

/// A `survfit` object for single-endpoint survival, curves stacked one
/// after the other as in R.
///
/// `std_err` is the standard error of `log(surv)` when `logse` is true (the
/// Greenwood variance) and of `surv` itself otherwise (robust variance);
/// `std_chaz` is always the standard error of `cumhaz`.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SurvfitKMResult {
    /// Observations used by each curve.
    #[pyo3(get)]
    pub n: Vec<usize>,
    #[pyo3(get)]
    pub time: Vec<f64>,
    #[pyo3(get)]
    pub n_risk: Vec<f64>,
    #[pyo3(get)]
    pub n_event: Vec<f64>,
    #[pyo3(get)]
    pub n_censor: Vec<f64>,
    #[pyo3(get)]
    pub n_enter: Option<Vec<f64>>,
    #[pyo3(get)]
    pub counts: Option<SurvfitCounts>,
    #[pyo3(get)]
    pub surv: Vec<f64>,
    #[pyo3(get)]
    pub std_err: Option<Vec<f64>>,
    #[pyo3(get)]
    pub cumhaz: Vec<f64>,
    #[pyo3(get)]
    pub std_chaz: Option<Vec<f64>>,
    #[pyo3(get)]
    pub lower: Option<Vec<f64>>,
    #[pyo3(get)]
    pub upper: Option<Vec<f64>>,
    /// Rows per curve; `None` for a single curve without strata.
    #[pyo3(get)]
    pub strata: Option<Vec<usize>>,
    /// The strata code of each curve, in curve order.
    #[pyo3(get)]
    pub strata_codes: Option<Vec<i32>>,
    /// Subjects per curve when an `id` was given.
    #[pyo3(get)]
    pub n_id: Option<Vec<usize>>,
    #[pyo3(get)]
    pub logse: bool,
    #[pyo3(get)]
    pub conf_int: f64,
    #[pyo3(get)]
    pub conf_type: String,
    #[pyo3(get)]
    pub conf_lower: String,
    /// `"right"` or `"counting"`.
    #[pyo3(get, name = "type")]
    pub type_: String,
    /// The starting time of the curves.
    #[pyo3(get)]
    pub t0: f64,
    /// One entry per curve when requested.
    #[pyo3(get)]
    pub influence_surv: Option<Vec<SurvfitInfluence>>,
    #[pyo3(get)]
    pub influence_chaz: Option<Vec<SurvfitInfluence>>,
}

impl SurvfitKMResult {
    /// Number of curves.
    pub fn n_curves(&self) -> usize {
        self.strata.as_ref().map_or(1, Vec::len)
    }

    /// Row range of each curve in the stacked vectors.
    pub fn curve_ranges(&self) -> Vec<std::ops::Range<usize>> {
        match &self.strata {
            Some(strata) => {
                let mut start = 0;
                strata
                    .iter()
                    .map(|&count| {
                        let range = start..start + count;
                        start += count;
                        range
                    })
                    .collect()
            }
            None => {
                let whole = 0..self.time.len();
                vec![whole]
            }
        }
    }

    /// Standard error on the survival scale (`summary.survfit` reports it
    /// this way whatever `logse` is).
    pub fn std_err_surv_scale(&self) -> Option<Vec<f64>> {
        self.std_err.as_ref().map(|se| {
            if self.logse {
                se.iter().zip(&self.surv).map(|(s, p)| s * p).collect()
            } else {
                se.clone()
            }
        })
    }
}

// ---------------------------------------------------------------------------
// The C kernel: one curve
// ---------------------------------------------------------------------------

/// Rows of one curve, in the order `survfitkm.c` walks them.
struct CurveRows {
    /// Rows ordered by start time (counting-process data only).
    sort1: Option<Vec<usize>>,
    /// Rows ordered by stop time.
    sort2: Vec<usize>,
}

/// Everything `survfitkm.c` reads besides the row order.
struct KernelData<'a> {
    time1: Option<&'a [f64]>,
    time2: &'a [f64],
    status: &'a [i32],
    wt: &'a [f64],
    /// `1 * (first obs of a subject) + 2 * (last obs)`.
    position: &'a [u8],
    /// Cluster code (`0..nid`) of every row of the curve, with `nid`.
    cluster: Option<(&'a [usize], usize)>,
}

#[derive(Clone, Copy)]
struct KernelOptions {
    stype: SurvType,
    ctype: HazardType,
    influence: InfluenceRequest,
    reverse: bool,
    entry: bool,
}

/// Output of the kernel for one curve.
struct CurveFit {
    time: Vec<f64>,
    n_risk: Vec<f64>,
    n_event: Vec<f64>,
    n_censor: Vec<f64>,
    n_enter: Option<Vec<f64>>,
    wt_risk: Vec<f64>,
    wt_event: Vec<f64>,
    wt_censor: Vec<f64>,
    wt_enter: Option<Vec<f64>>,
    surv: Vec<f64>,
    cumhaz: Vec<f64>,
    std_surv: Vec<f64>,
    std_chaz: Vec<f64>,
    /// `nid x ntime`, when requested.
    influence_surv: Option<Array2<f64>>,
    influence_chaz: Option<Array2<f64>>,
}

/// The reporting times of one curve, as survival 3.8-11's `survfitkm.c`
/// computes them (the version the reference fixtures were generated with;
/// 3.8-12 moved the rule to R and dropped the unconditional first stop
/// time): the smallest stop time, every stop time that is an event or the
/// last interval of a subject and, with `entry`, every start time that is
/// the first interval of a subject.
fn kernel_unique_times(data: &KernelData<'_>, rows: &CurveRows, entry: bool) -> Vec<f64> {
    let sort2 = &rows.sort2;
    let mut dtime = Vec::new();
    match (entry, data.time1, &rows.sort1) {
        (true, Some(time1), Some(sort1)) => {
            let mut temp = time1[sort1[0]];
            dtime.push(temp);
            let mut j = 1;
            for &i2 in sort2 {
                while j < sort1.len() && time1[sort1[j]] < data.time2[i2] {
                    let i1 = sort1[j];
                    if time1[i1] != temp && data.position[i1] & 1 == 1 {
                        temp = time1[i1];
                        dtime.push(temp);
                    }
                    j += 1;
                }
                if data.time2[i2] != temp && (data.position[i2] > 1 || data.status[i2] > 0) {
                    temp = data.time2[i2];
                    dtime.push(temp);
                }
            }
        }
        _ => {
            let mut temp = data.time2[sort2[0]];
            dtime.push(temp);
            for &i2 in &sort2[1..] {
                if (data.position[i2] > 1 || data.status[i2] > 0) && data.time2[i2] != temp {
                    temp = data.time2[i2];
                    dtime.push(temp);
                }
            }
        }
    }
    dtime
}

/// Port of `survfitkm` (`src/survfitkm.c`) for one curve.
fn kernel(data: &KernelData<'_>, rows: &CurveRows, options: KernelOptions) -> CurveFit {
    let nused = rows.sort2.len();
    let sort2 = &rows.sort2;
    let dtime = kernel_unique_times(data, rows, options.entry);
    let ntime = dtime.len();
    let counting = data.time1.is_some();

    // -- the counts, walking backwards in time so risk sets accumulate
    let mut n_risk = vec![0.0; ntime];
    let mut n_event = vec![0.0; ntime];
    let mut n_censor = vec![0.0; ntime];
    let mut wt_risk = vec![0.0; ntime];
    let mut wt_event = vec![0.0; ntime];
    let mut wt_censor = vec![0.0; ntime];
    let mut n_enter = (options.entry && counting).then(|| vec![0.0; ntime]);
    let mut wt_enter = (options.entry && counting).then(|| vec![0.0; ntime]);
    {
        let mut person1 = nused;
        let mut person2 = nused;
        let mut n1 = 0.0;
        let mut wt1 = 0.0;
        for k in (0..ntime).rev() {
            let (mut n2, mut n3, mut wt2, mut wt3) = (0.0, 0.0, 0.0, 0.0);
            while person2 > 0 {
                let i2 = sort2[person2 - 1];
                if data.time2[i2] < dtime[k] {
                    break;
                }
                n1 += 1.0;
                wt1 += data.wt[i2];
                if data.status[i2] == 1 {
                    n2 += 1.0;
                    wt2 += data.wt[i2];
                } else if data.position[i2] & 2 != 0 {
                    // the last of a subject's (a,b](b,c]... string: a real censor
                    n3 += 1.0;
                    wt3 += data.wt[i2];
                }
                person2 -= 1;
            }
            if let (Some(time1), Some(sort1)) = (data.time1, &rows.sort1) {
                let (mut n4, mut wt4) = (0.0, 0.0);
                while person1 > 0 {
                    let i1 = sort1[person1 - 1];
                    if time1[i1] < dtime[k] {
                        break;
                    }
                    // entry is >= dtime, remove from the risk set
                    n1 -= 1.0;
                    wt1 -= data.wt[i1];
                    if options.entry && data.position[i1] & 1 != 0 && time1[i1] == dtime[k] {
                        n4 += 1.0;
                        wt4 += data.wt[i1];
                    }
                    person1 -= 1;
                }
                if let (Some(n_enter), Some(wt_enter)) = (&mut n_enter, &mut wt_enter) {
                    n_enter[k] = n4;
                    wt_enter[k] = wt4;
                }
            }
            n_risk[k] = n1;
            n_event[k] = n2;
            n_censor[k] = n3;
            wt_risk[k] = wt1;
            wt_event[k] = wt2;
            wt_censor[k] = wt3;
        }
    }

    // (unweighted events, weighted events, weighted at risk) at time i
    let event_terms = |i: usize| -> (f64, f64, f64) {
        if options.reverse {
            (n_censor[i], wt_censor[i], wt_risk[i] - wt_event[i])
        } else {
            (n_event[i], wt_event[i], wt_risk[i])
        }
    };

    // -- survival, cumulative hazard and the simple (Greenwood) variances
    let mut surv = vec![0.0; ntime];
    let mut cumhaz = vec![0.0; ntime];
    let mut std_surv = vec![0.0; ntime];
    let mut std_chaz = vec![0.0; ntime];
    {
        let mut nelson = 0.0;
        let mut km = 1.0;
        let mut v1 = 0.0;
        let mut v2 = 0.0;
        for i in 0..ntime {
            let (d0, d1, nrisk) = event_terms(i);
            match options.ctype {
                HazardType::NelsonAalen => {
                    if d0 > 0.0 && d1 > 0.0 {
                        nelson += d1 / nrisk;
                        v2 += d1 / (nrisk * nrisk);
                    }
                }
                HazardType::FlemingHarrington => {
                    let mut j = 0.0;
                    while j < d0 {
                        let dtemp = nrisk - j * d1 / d0;
                        nelson += d1 / (d0 * dtemp);
                        v2 += d1 / (d0 * dtemp * dtemp);
                        j += 1.0;
                    }
                }
            }
            cumhaz[i] = nelson;
            std_chaz[i] = v2.sqrt();
            match options.stype {
                SurvType::KaplanMeier => {
                    if d0 > 0.0 && d1 > 0.0 {
                        km *= (nrisk - d1) / nrisk;
                        v1 += d1 / (nrisk * (nrisk - d1)); // Greenwood
                    }
                    surv[i] = km;
                    std_surv[i] = v1.sqrt();
                }
                SurvType::ExpCumhaz => {
                    surv[i] = (-nelson).exp();
                    std_surv[i] = std_chaz[i];
                }
            }
        }
    }

    // -- infinitesimal jackknife (robust) variance and influence
    let mut influence_surv = None;
    let mut influence_chaz = None;
    if let Some((cluster, nid)) = data.cluster {
        let want_surv_matrix =
            options.influence.survival() && options.stype == SurvType::KaplanMeier;
        let want_chaz_matrix = options.influence.cumhaz()
            || (options.influence.survival() && options.stype == SurvType::ExpCumhaz);
        let mut imat1 = want_surv_matrix.then(|| Array2::zeros((nid, ntime)));
        let mut imat2 = want_chaz_matrix.then(|| Array2::zeros((nid, ntime)));
        let mut gcount = vec![0i64; nid];
        let mut gwt = vec![0.0; nid];
        let mut inf1 = vec![0.0; nid]; // survival influence
        let mut inf2 = vec![0.0; nid]; // cumulative hazard influence
        let mut person1 = 0;
        let mut person2 = 0;
        if !counting {
            // at the start everyone is at risk
            for &i2 in sort2 {
                gcount[cluster[i2]] += 1;
                gwt[cluster[i2]] += data.wt[i2];
            }
        }
        // Whether a row counts as an event for the curve being estimated.
        let is_event = |i2: usize| -> bool {
            if options.reverse {
                data.status[i2] == 0 && data.position[i2] & 2 != 0
            } else {
                data.status[i2] == 1
            }
        };
        let remove = |i2: usize, gcount: &mut [i64], gwt: &mut [f64]| {
            let g = cluster[i2];
            gcount[g] -= 1;
            if gcount[g] == 0 {
                gwt[g] = 0.0;
            } else {
                gwt[g] -= data.wt[i2];
            }
        };
        let mut km = 1.0; // lags one step behind the estimate
        let mut v1 = 0.0;
        let mut v2 = 0.0;
        for i in 0..ntime {
            // toss the outdated; with reverse the deaths tied at this time
            // leave before the censorings are treated as events
            while person2 < nused {
                let i2 = sort2[person2];
                let gone = data.time2[i2] < dtime[i]
                    || (options.reverse && data.time2[i2] == dtime[i] && data.status[i2] == 1);
                if !gone {
                    break;
                }
                remove(i2, &mut gcount, &mut gwt);
                person2 += 1;
            }
            if let (Some(time1), Some(sort1)) = (data.time1, &rows.sort1) {
                // add in new subjects
                while person1 < nused {
                    let i1 = sort1[person1];
                    if time1[i1] >= dtime[i] {
                        break;
                    }
                    gcount[cluster[i1]] += 1;
                    gwt[cluster[i1]] += data.wt[i1];
                    person1 += 1;
                }
            }
            let (d0, d1, nrisk) = event_terms(i);
            if d0 > 0.0 && d1 > 0.0 {
                let haz = d1 / nrisk;
                // per-event derivative terms of the cumulative hazard
                let (dn_term, risk_term) = match options.ctype {
                    HazardType::NelsonAalen => (1.0 / nrisk, haz / nrisk),
                    HazardType::FlemingHarrington => {
                        let mut dtemp = 0.0; // the working denominator
                        let mut dtemp2 = 0.0; // sum of squares
                        let mut dtemp3 = 0.0; // non-death derivative
                        let temp = nrisk - d1; // weights of the non-deaths
                        let mut k = d0.floor();
                        while k > 0.0 {
                            let frac = k / d0;
                            let btemp = 1.0 / (temp + frac * d1); // "b" in the math
                            dtemp += btemp;
                            dtemp2 += btemp * btemp * frac;
                            dtemp3 += btemp * btemp;
                            k -= 1.0;
                        }
                        dtemp /= d0; // average denominator
                        if d1 != d0 {
                            // case weights
                            dtemp2 *= d1 / d0;
                            dtemp3 *= d1 / d0;
                        }
                        (dtemp + dtemp3 - dtemp2, dtemp3)
                    }
                };
                for g in 0..nid {
                    if options.stype == SurvType::KaplanMeier {
                        inf1[g] = inf1[g] * (1.0 - haz) + gwt[g] * km * haz / nrisk;
                    }
                    if options.ctype == HazardType::NelsonAalen || gcount[g] > 0 {
                        inf2[g] -= gwt[g] * risk_term;
                    }
                }
                // catch the endpoints up to this event time
                while person2 < nused {
                    let i2 = sort2[person2];
                    if data.time2[i2] > dtime[i] {
                        break;
                    }
                    if is_event(i2) {
                        let g = cluster[i2];
                        if options.stype == SurvType::KaplanMeier {
                            inf1[g] -= km * data.wt[i2] / nrisk;
                        }
                        inf2[g] += data.wt[i2] * dn_term;
                    }
                    remove(i2, &mut gcount, &mut gwt);
                    person2 += 1;
                }
                km *= 1.0 - haz;
                v1 = inf1.iter().map(|v| v * v).sum();
                v2 = inf2.iter().map(|v| v * v).sum();
            }
            match options.stype {
                SurvType::KaplanMeier => {
                    std_surv[i] = v1.sqrt();
                    std_chaz[i] = v2.sqrt();
                }
                SurvType::ExpCumhaz => {
                    std_surv[i] = v2.sqrt();
                    std_chaz[i] = v2.sqrt();
                }
            }
            if let Some(imat1) = &mut imat1 {
                imat1.column_mut(i).assign(&ndarray::aview1(&inf1));
            }
            if let Some(imat2) = &mut imat2 {
                imat2.column_mut(i).assign(&ndarray::aview1(&inf2));
            }
        }
        influence_surv = imat1;
        influence_chaz = imat2;
    }

    CurveFit {
        time: dtime,
        n_risk,
        n_event,
        n_censor,
        n_enter,
        wt_risk,
        wt_event,
        wt_censor,
        wt_enter,
        surv,
        cumhaz,
        std_surv,
        std_chaz,
        influence_surv,
        influence_chaz,
    }
}

// ---------------------------------------------------------------------------
// The R-level driver
// ---------------------------------------------------------------------------

/// `factor(x, unique(x))`: integer codes in order of first appearance, with
/// the levels.
fn codes_by_first_appearance(values: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut levels = Vec::new();
    let mut lookup = std::collections::HashMap::new();
    let codes = values
        .iter()
        .map(|&value| {
            *lookup.entry(value).or_insert_with(|| {
                levels.push(value);
                levels.len() - 1
            })
        })
        .collect();
    (codes, levels)
}

/// Port of `survflag` (`R/xtras.R`): `1 * (first interval of a subject) +
/// 2 * (last interval)`, where a gap between consecutive intervals or a
/// change of curve also ends a sequence.
pub(crate) fn survflag(start: &[f64], stop: &[f64], id: &[usize], group: &[usize]) -> Vec<u8> {
    let n = stop.len();
    let mut indx: Vec<usize> = (0..n).collect();
    indx.sort_by(|&a, &b| {
        group[a]
            .cmp(&group[b])
            .then_with(|| id[a].cmp(&id[b]))
            .then_with(|| stop[a].total_cmp(&stop[b]))
            .then_with(|| a.cmp(&b))
    });
    let mut flag = vec![0u8; n];
    for (k, &row) in indx.iter().enumerate() {
        let breaks_before = k == 0 || {
            let prev = indx[k - 1];
            id[prev] != id[row] || group[prev] != group[row] || stop[prev] < start[row]
        };
        let breaks_after = k + 1 == n || {
            let next = indx[k + 1];
            id[next] != id[row] || group[next] != group[row] || stop[row] < start[next]
        };
        flag[row] = u8::from(breaks_before) + 2 * u8::from(breaks_after);
    }
    flag
}

/// `aeqSurv`: bin the time columns jointly so that near-ties become ties
/// (an interval that collapses to length 0 is an error there).
fn apply_timefix(
    start: Option<&[f64]>,
    time: &[f64],
) -> SurvivalResult<(Option<Vec<f64>>, Vec<f64>)> {
    let fixed = crate::data_prep::aeq_surv(time, start, None)?;
    Ok((fixed.time2, fixed.time))
}

/// `keep[order(values[keep])]`, ties in `keep` order.
///
/// Sorting `(value, row)` pairs rather than an index vector keeps the
/// keys next to each other in memory, which is several times faster than
/// an indirect comparison sort at a million rows.  `parallel` splits the
/// sort itself over threads; a caller sorting several curves at once
/// parallelises over the curves instead.
pub(crate) fn ordered_subset(keep: &[usize], values: &[f64], parallel: bool) -> Vec<usize> {
    let mut pairs: Vec<(f64, usize)> = keep.iter().map(|&i| (values[i], i)).collect();
    let order =
        |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1));
    if parallel && pairs.len() > PARALLEL_THRESHOLD_LARGE {
        pairs.par_sort_unstable_by(order);
    } else {
        pairs.sort_unstable_by(order);
    }
    pairs.into_iter().map(|(_, i)| i).collect()
}

/// The rows of each curve in data order: `split(seq_along(x), x)`.
pub(crate) fn rows_by_curve(x: &[usize], n_curves: usize) -> Vec<Vec<usize>> {
    let mut rows = vec![Vec::new(); n_curves];
    for (i, &curve) in x.iter().enumerate() {
        rows[curve].push(i);
    }
    rows
}

/// The rows of one curve in data order and in the orders the kernel walks
/// them.
struct CurveInput {
    keep: Vec<usize>,
    rows: CurveRows,
}

/// [`CurveInput`] of every curve: the sorts run in parallel over the
/// curves, or within the sort when there is a single curve.
fn curve_inputs(
    x: &[usize],
    n_curves: usize,
    start: Option<&[f64]>,
    time: &[f64],
    status: &[i32],
    reverse: bool,
) -> Vec<CurveInput> {
    let buckets = rows_by_curve(x, n_curves);
    let single = n_curves == 1;
    let prepare = |keep: Vec<usize>| {
        let sort1 = start.map(|start| ordered_subset(&keep, start, single));
        let sort2 = if reverse {
            // deaths first among ties so the kernel can drop them from the
            // risk set before the tied censorings are treated as events
            let mut sort2 = keep.clone();
            sort2.sort_by(|&a, &b| {
                time[a]
                    .total_cmp(&time[b])
                    .then_with(|| status[b].cmp(&status[a]))
            });
            sort2
        } else {
            ordered_subset(&keep, time, single)
        };
        CurveInput {
            keep,
            rows: CurveRows { sort1, sort2 },
        }
    };
    if single {
        buckets.into_iter().map(prepare).collect()
    } else {
        buckets.into_par_iter().map(prepare).collect()
    }
}

fn count_unique(values: impl Iterator<Item = usize>) -> usize {
    let mut seen: Vec<usize> = values.collect();
    seen.sort_unstable();
    seen.dedup();
    seen.len()
}

/// Port of `survfitKM` (`R/survfitKM.R`).
///
/// Every curve (one per distinct `strata` code, in ascending order) goes
/// through the C kernel once; the curves are then stacked and the
/// confidence limits added through [`survfit_confint`].  The rules for the
/// default `robust` choice, the cluster used by the robust variance, the
/// `conf.lower` adjustments and the derived survival influence for
/// `stype = 2` are R's.
pub fn survfitkm(
    data: &SurvfitKMData,
    options: &SurvfitKMOptions,
) -> SurvivalResult<SurvfitKMResult> {
    validate_conf_int(options.conf_int)?;
    let n_all = data.n();
    let counting = data.start.is_some();
    let (start, time) = if options.timefix {
        apply_timefix(data.start.as_deref(), &data.time)?
    } else {
        (data.start.clone(), data.time.clone())
    };

    // start.time: drop observations that end before it
    let t0 = match options.start_time {
        Some(start_time) => {
            if !start_time.is_finite() {
                return Err(SurvivalError::invalid_input(
                    "start.time must be a single numeric value",
                ));
            }
            start_time
        }
        None => start
            .iter()
            .flatten()
            .chain(&time)
            .fold(0.0_f64, |acc, &t| acc.min(t)),
    };
    let rows: Vec<usize> = (0..n_all).filter(|&i| time[i] >= t0).collect();
    if rows.is_empty() {
        return Err(SurvivalError::invalid_input(
            "all observations removed by start.time",
        ));
    }
    let n = rows.len();
    let pick = |values: &[f64]| -> Vec<f64> { rows.iter().map(|&i| values[i]).collect() };
    let time = pick(&time);
    let start = start.as_deref().map(pick);
    let status: Vec<i32> = rows.iter().map(|&i| data.status[i]).collect();
    let weights: Vec<f64> = match &data.weights {
        Some(w) => pick(w),
        None => vec![1.0; n],
    };
    let strata_levels: Vec<i32> = data.strata.as_ref().map_or_else(
        || vec![0],
        |strata| {
            // levels come from the full data, as in R, so a stratum that
            // start.time empties still gets an n of 0
            let mut levels = strata.clone();
            levels.sort_unstable();
            levels.dedup();
            levels
        },
    );
    let x: Vec<usize> = rows
        .iter()
        .map(|&i| {
            data.strata.as_ref().map_or(0, |strata| {
                strata_levels
                    .binary_search(&strata[i])
                    .expect("strata code is one of its own levels")
            })
        })
        .collect();

    // cluster / id / robust logic
    let has_cluster = data.cluster.is_some();
    let has_id = data.id.is_some();
    let has_rwt = weights.iter().any(|w| *w != w.floor());
    let has_robust = options.robust.is_some();
    let id_codes: Option<Vec<usize>> = data.id.as_ref().map(|id| {
        let subset: Vec<i64> = rows.iter().map(|&i| id[i]).collect();
        codes_by_first_appearance(&subset).0
    });
    let mut influence = options.influence;
    let mut entry = options.entry && has_id;
    let mut cluster_source: Option<Vec<i64>> = data
        .cluster
        .as_ref()
        .map(|cluster| rows.iter().map(|&i| cluster[i]).collect());
    let robust = match options.robust {
        Some(robust) => robust,
        None => {
            if influence != InfluenceRequest::None {
                if !(has_cluster || has_id) {
                    cluster_source = Some((0..n as i64).collect());
                }
                true
            } else {
                has_cluster
                    || has_rwt
                    || (has_id && {
                        let ids = id_codes.as_ref().expect("id present");
                        let events: Vec<usize> =
                            (0..n).filter(|&i| status[i] == 1).map(|i| ids[i]).collect();
                        count_unique(events.iter().copied()) < events.len()
                    })
            }
        }
    };
    // (cluster code per row, cluster labels); None = no robust variance
    let cluster: Option<(Vec<usize>, Vec<i64>)> = if let Some(source) = &cluster_source {
        // R warns "cluster specified with robust=FALSE, cluster ignored"
        robust.then(|| codes_by_first_appearance(source))
    } else if robust {
        if let Some(id) = &data.id {
            let subset: Vec<i64> = rows.iter().map(|&i| id[i]).collect();
            Some(codes_by_first_appearance(&subset))
        } else if !counting || !has_robust {
            Some(((0..n).collect(), (0..n as i64).collect()))
        } else {
            return Err(SurvivalError::invalid_input(
                "id or cluster option required",
            ));
        }
    } else {
        None
    };
    if !robust {
        // R warns "robust=FALSE implies influence=FALSE"
        influence = InfluenceRequest::None;
    }
    let cluster = if options.se_fit {
        cluster
    } else {
        influence = InfluenceRequest::None;
        None
    };
    if !counting {
        entry = false;
    }
    let position: Vec<u8> = match (&start, &id_codes) {
        (Some(start), Some(id)) => survflag(start, &time, id, &x),
        _ => vec![3; n],
    };

    // one kernel call per curve
    let kernel_options = KernelOptions {
        stype: options.stype,
        ctype: options.ctype,
        influence,
        reverse: options.reverse,
        entry,
    };
    let n_curves = strata_levels.len();
    let mut n_used = vec![0usize; n_curves];
    let mut n_id = has_id.then(|| vec![0usize; n_curves]);
    let mut fits: Vec<(usize, CurveFit, Vec<i64>)> = Vec::with_capacity(n_curves);
    let mut ctemp = vec![0usize; n];
    let inputs = curve_inputs(
        &x,
        n_curves,
        start.as_deref(),
        &time,
        &status,
        options.reverse,
    );
    for (curve, CurveInput { keep, rows }) in inputs.into_iter().enumerate() {
        n_used[curve] = keep.len();
        if keep.is_empty() {
            continue; // rare case where all are < start.time
        }
        if let (Some(n_id), Some(id)) = (&mut n_id, &id_codes) {
            n_id[curve] = count_unique(keep.iter().map(|&i| id[i]));
        }
        // clusters are renumbered 0, 1, 2, ... per curve in order of
        // appearance so each curve's influence matrix has only its own rows
        let (kernel_cluster, curve_clusters) = match &cluster {
            Some((codes, labels)) => {
                let subset: Vec<i64> = keep.iter().map(|&i| codes[i] as i64).collect();
                let (renumbered, unique) = codes_by_first_appearance(&subset);
                for (&i, code) in keep.iter().zip(renumbered) {
                    ctemp[i] = code;
                }
                let names = unique.iter().map(|&code| labels[code as usize]).collect();
                (Some((ctemp.as_slice(), unique.len())), names)
            }
            None => (None, Vec::new()),
        };
        let kernel_data = KernelData {
            time1: start.as_deref(),
            time2: &time,
            status: &status,
            wt: &weights,
            position: &position,
            cluster: kernel_cluster,
        };
        let fit = kernel(&kernel_data, &rows, kernel_options);
        fits.push((curve, fit, curve_clusters));
    }

    // stack the curves
    let total: usize = fits.iter().map(|(_, fit, _)| fit.time.len()).sum();
    let mut result = SurvfitKMResult {
        n: n_used,
        time: Vec::with_capacity(total),
        n_risk: Vec::with_capacity(total),
        n_event: Vec::with_capacity(total),
        n_censor: Vec::with_capacity(total),
        n_enter: entry.then(|| Vec::with_capacity(total)),
        counts: None,
        surv: Vec::with_capacity(total),
        std_err: options.se_fit.then(|| Vec::with_capacity(total)),
        cumhaz: Vec::with_capacity(total),
        std_chaz: options.se_fit.then(|| Vec::with_capacity(total)),
        lower: None,
        upper: None,
        strata: None,
        strata_codes: None,
        n_id,
        // R: se(log S) unless the robust variance was used; the C kernel's
        // Fleming-Harrington survival influence is nonetheless reported as
        // se(log S) by survfitKM.R, and this mirrors that source
        logse: cluster.is_none() || options.ctype == HazardType::FlemingHarrington,
        conf_int: options.conf_int,
        conf_type: options.conf_type.as_str().to_string(),
        conf_lower: options.conf_lower.as_str().to_string(),
        type_: if counting { "counting" } else { "right" }.to_string(),
        t0,
        influence_surv: None,
        influence_chaz: None,
    };
    let addcounts = weights.iter().any(|&w| w != 1.0);
    let mut counts = addcounts.then(|| SurvfitCounts {
        n_risk: Vec::with_capacity(total),
        n_event: Vec::with_capacity(total),
        n_censor: Vec::with_capacity(total),
        n_enter: entry.then(|| Vec::with_capacity(total)),
    });
    let mut strata_rows = Vec::with_capacity(fits.len());
    let mut strata_codes = Vec::with_capacity(fits.len());
    let mut influence_surv = influence.survival().then(Vec::new);
    let mut influence_chaz = influence.cumhaz().then(Vec::new);
    for (curve, fit, mut curve_clusters) in fits {
        strata_rows.push(fit.time.len());
        strata_codes.push(strata_levels[curve]);
        result.time.extend_from_slice(&fit.time);
        result.n_risk.extend_from_slice(&fit.wt_risk);
        result.n_event.extend_from_slice(&fit.wt_event);
        result.n_censor.extend_from_slice(&fit.wt_censor);
        if let (Some(n_enter), Some(wt_enter)) = (&mut result.n_enter, &fit.wt_enter) {
            n_enter.extend_from_slice(wt_enter);
        }
        if let Some(counts) = &mut counts {
            counts.n_risk.extend_from_slice(&fit.n_risk);
            counts.n_event.extend_from_slice(&fit.n_event);
            counts.n_censor.extend_from_slice(&fit.n_censor);
            if let (Some(n_enter), Some(fit_enter)) = (&mut counts.n_enter, &fit.n_enter) {
                n_enter.extend_from_slice(fit_enter);
            }
        }
        result.surv.extend_from_slice(&fit.surv);
        result.cumhaz.extend_from_slice(&fit.cumhaz);
        if let Some(std_err) = &mut result.std_err {
            std_err.extend_from_slice(&fit.std_surv);
        }
        if let Some(std_chaz) = &mut result.std_chaz {
            std_chaz.extend_from_slice(&fit.std_chaz);
        }
        let to_rows = |matrix: &Array2<f64>| -> Vec<Vec<f64>> {
            matrix.outer_iter().map(|row| row.to_vec()).collect()
        };
        if let Some(list) = &mut influence_surv {
            let values = match (&fit.influence_surv, &fit.influence_chaz) {
                (Some(matrix), _) => to_rows(matrix),
                // stype = 2: an obs that moves the cumulative hazard up
                // moves S down, influence.surv = -influence.chaz * S(t)
                (None, Some(matrix)) => matrix
                    .outer_iter()
                    .map(|row| {
                        row.iter()
                            .zip(&fit.surv)
                            .map(|(value, surv)| -value * surv)
                            .collect()
                    })
                    .collect(),
                (None, None) => Vec::new(),
            };
            list.push(SurvfitInfluence {
                cluster: if influence_chaz.is_some() {
                    curve_clusters.clone()
                } else {
                    std::mem::take(&mut curve_clusters)
                },
                values,
            });
        }
        if let Some(list) = &mut influence_chaz
            && let Some(matrix) = &fit.influence_chaz
        {
            list.push(SurvfitInfluence {
                cluster: curve_clusters,
                values: to_rows(matrix),
            });
        }
    }
    result.counts = counts;
    if n_curves > 1 {
        result.strata = Some(strata_rows);
        result.strata_codes = Some(strata_codes);
    }
    result.influence_surv = influence_surv;
    result.influence_chaz = influence_chaz;

    // confidence limits
    if options.se_fit && options.conf_type != ConfType::None {
        let std_err = result.std_err.as_deref().expect("se.fit keeps std.err");
        let std_low: Option<Vec<f64>> = match options.conf_lower {
            ConfLower::Usual => None,
            ConfLower::Peto => Some(
                result
                    .surv
                    .iter()
                    .zip(&result.n_risk)
                    .map(|(s, n)| ((1.0 - s) / n).sqrt())
                    .collect(),
            ),
            ConfLower::Modified => {
                // n.lag = the number at risk the last time there was an
                // event (or the first time of a stratum)
                let mut n_lag = vec![0.0; result.time.len()];
                for range in result.curve_ranges() {
                    let mut lag = f64::NAN;
                    for i in range.clone() {
                        if i == range.start || result.n_event[i] > 0.0 {
                            lag = result.n_risk[i];
                        }
                        n_lag[i] = lag;
                    }
                }
                Some(
                    std_err
                        .iter()
                        .zip(&n_lag)
                        .zip(&result.n_risk)
                        .map(|((se, lag), n)| se * (lag / n).sqrt())
                        .collect(),
                )
            }
        };
        let bands = survfit_confint(
            &result.surv,
            std_err,
            result.logse,
            options.conf_type,
            options.conf_int,
            std_low.as_deref(),
            true,
        )?;
        result.lower = Some(bands.lower);
        result.upper = Some(bands.upper);
    }
    Ok(result)
}

/// Python binding of [`survfitkm`]; the keyword arguments mirror
/// `survfit.formula` and `survfitKM`.
#[pyfunction(name = "survfitkm")]
#[pyo3(signature = (time, status, start=None, weights=None, strata=None, id=None, cluster=None, stype=1, ctype=1, se_fit=true, conf_int=0.95, conf_type="log", conf_lower="usual", start_time=None, robust=None, influence=0, entry=false, timefix=true, reverse=false))]
#[allow(clippy::too_many_arguments)]
pub fn survfitkm_py(
    time: Vec<f64>,
    status: Vec<i32>,
    start: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
    strata: Option<Vec<i32>>,
    id: Option<Vec<i64>>,
    cluster: Option<Vec<i64>>,
    stype: i32,
    ctype: i32,
    se_fit: bool,
    conf_int: f64,
    conf_type: &str,
    conf_lower: &str,
    start_time: Option<f64>,
    robust: Option<bool>,
    influence: i32,
    entry: bool,
    timefix: bool,
    reverse: bool,
) -> PyResult<SurvfitKMResult> {
    let data = SurvfitKMData::try_new(start, time, status, weights, strata, id, cluster)?;
    let options = SurvfitKMOptions {
        stype: SurvType::from_code(stype)?,
        ctype: HazardType::from_code(ctype)?,
        se_fit,
        conf_int,
        conf_type: ConfType::parse(conf_type)?,
        conf_lower: ConfLower::parse(conf_lower)?,
        start_time,
        robust,
        influence: InfluenceRequest::from_code(influence)?,
        entry,
        timefix,
        reverse,
    };
    Ok(survfitkm(&data, &options)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_approx(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&left, &right)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (left - right).abs() <= tol,
                "index {idx}: actual {left} differs from expected {right}"
            );
        }
    }

    fn fit(data: SurvfitKMData, options: SurvfitKMOptions) -> SurvfitKMResult {
        survfitkm(&data, &options).expect("survfitkm should succeed")
    }

    #[test]
    fn matches_r_aml_maintained() {
        // survfit(Surv(time, status) ~ 1, aml[aml$x == "Maintained", ])
        let time = vec![
            9.0, 13.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0,
        ];
        let status = vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0];
        let result = fit(
            SurvfitKMData::right_censored(time, status).unwrap(),
            SurvfitKMOptions::default(),
        );
        assert_eq!(result.n, vec![11]);
        assert_eq!(
            result.time,
            vec![9.0, 13.0, 18.0, 23.0, 28.0, 31.0, 34.0, 45.0, 48.0, 161.0]
        );
        assert_eq!(
            result.n_risk,
            vec![11.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        );
        assert_vec_approx(
            &result.surv[..6],
            &[
                0.909090909090909,
                0.818181818181818,
                0.715909090909091,
                0.613636363636364,
                0.613636363636364,
                0.490909090909091,
            ],
            1e-12,
        );
        assert!(result.logse);
        assert_vec_approx(
            &result.std_err.as_ref().unwrap()[..3],
            &[0.0953462589245592, 0.14213381090374, 0.195087577921207],
            1e-12,
        );
        assert_vec_approx(
            &result.lower.as_ref().unwrap()[..2],
            &[0.754133845081525, 0.619248987399364],
            1e-12,
        );
        assert_eq!(result.upper.as_ref().unwrap()[0], 1.0);
        assert!(result.strata.is_none());
        assert!(result.counts.is_none());
        assert_eq!(result.type_, "right");
    }

    #[test]
    fn robust_cluster_variance_matches_r() {
        // survfit(Surv(time, status) ~ 1, cluster = id) with id = c(1,1,2,2,3,3)
        let data = SurvfitKMData::try_new(
            None,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 0, 1, 0, 1],
            None,
            None,
            None,
            Some(vec![1, 1, 2, 2, 3, 3]),
        )
        .unwrap();
        let result = fit(data, SurvfitKMOptions::default());
        assert!(!result.logse);
        assert_vec_approx(
            result.std_err.as_ref().unwrap(),
            &[0.1360828, 0.2721655, 0.2721655, 0.2771598, 0.2771598, 0.0],
            1e-6,
        );
        assert_vec_approx(
            result.std_chaz.as_ref().unwrap(),
            &[
                0.1360828, 0.3320419, 0.3320419, 0.4571841, 0.4571841, 0.4571841,
            ],
            1e-6,
        );
    }

    #[test]
    fn robust_variance_follows_r_default_rule() {
        let data = SurvfitKMData::try_new(
            None,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 0, 1, 0, 1],
            Some(vec![1.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
            None,
            None,
            None,
        )
        .unwrap();
        // integer weights, no id: Greenwood
        let integer = fit(data.clone(), SurvfitKMOptions::default());
        assert!(integer.logse);
        assert!(integer.counts.is_some());
        // a non-integer weight switches to the infinitesimal jackknife
        let mut half = data.clone();
        half.weights = Some(vec![1.0, 2.0, 1.0, 1.0, 1.0, 0.5]);
        assert!(!fit(half, SurvfitKMOptions::default()).logse);
        // an id with more than one event per subject does too
        let mut repeated = data.clone();
        repeated.id = Some(vec![1, 1, 2, 2, 3, 3]);
        assert!(!fit(repeated, SurvfitKMOptions::default()).logse);
        // robust = FALSE overrides
        let mut forced = data;
        forced.cluster = Some(vec![1, 1, 2, 2, 3, 3]);
        let plain = fit(
            forced,
            SurvfitKMOptions {
                robust: Some(false),
                ..Default::default()
            },
        );
        assert!(plain.logse);
    }

    #[test]
    fn counting_data_with_id_matches_r() {
        // survfit(Surv(start, stop, status) ~ 1, id = id)
        let data = SurvfitKMData::try_new(
            Some(vec![0.0, 2.0, 0.0, 3.0, 0.0, 4.0]),
            vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0],
            vec![0, 1, 1, 0, 0, 1],
            None,
            None,
            Some(vec![1, 1, 2, 2, 3, 3]),
            None,
        )
        .unwrap();
        let result = fit(data, SurvfitKMOptions::default());
        assert_eq!(result.type_, "counting");
        assert_eq!(result.time, vec![2.0, 3.0, 5.0, 6.0, 7.0]);
        assert_eq!(result.n_risk, vec![3.0, 3.0, 3.0, 2.0, 1.0]);
        assert_eq!(result.n_censor, vec![0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_vec_approx(
            &result.surv,
            &[1.0, 2.0 / 3.0, 4.0 / 9.0, 4.0 / 9.0, 0.0],
            1e-12,
        );
        assert_eq!(result.n_id, Some(vec![3]));
        // id with one event per subject keeps the Greenwood variance
        assert!(result.logse);
    }

    #[test]
    fn entry_counts_match_r() {
        // survfit(Surv(start, stop, status) ~ 1, id = id, entry = TRUE, influence = TRUE)
        let data = SurvfitKMData::try_new(
            Some(vec![0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 2.0]),
            vec![2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 9.0],
            vec![0, 1, 1, 0, 0, 1, 1],
            None,
            None,
            Some(vec![1, 1, 2, 2, 3, 3, 4]),
            None,
        )
        .unwrap();
        let result = fit(
            data,
            SurvfitKMOptions {
                entry: true,
                influence: InfluenceRequest::Survival,
                ..Default::default()
            },
        );
        assert_eq!(result.time, vec![0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0]);
        assert_eq!(result.n_risk, vec![0.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(result.n_event, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(
            result.n_censor,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            result.n_enter,
            Some(vec![2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );
        assert_vec_approx(
            &result.surv,
            &[1.0, 1.0, 1.0, 0.75, 0.5625, 0.5625, 0.28125, 0.0],
            1e-12,
        );
        assert_eq!(result.n_id, Some(vec![4]));
        // influence = TRUE forces the robust variance, clustered on the id
        assert!(!result.logse);
        let influence = &result.influence_surv.as_ref().unwrap()[0];
        assert_eq!(influence.cluster, vec![1, 2, 3, 4]);
        assert_vec_approx(
            &influence.values[0],
            &[0.0, 0.0, 0.0, 0.0625, -0.09375, -0.09375, -0.046875, 0.0],
            1e-12,
        );
        assert_vec_approx(
            &influence.values[1],
            &[0.0, 0.0, 0.0, -0.1875, -0.09375, -0.09375, -0.046875, 0.0],
            1e-12,
        );
        assert_vec_approx(
            &influence.values[3],
            &[0.0, 0.0, 0.0, 0.0625, 0.09375, 0.09375, 0.1875, 0.0],
            1e-12,
        );
    }

    #[test]
    fn strata_are_stacked_in_level_order() {
        let data = SurvfitKMData::try_new(
            None,
            vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            vec![1, 0, 0, 1, 1, 1, 0, 0],
            None,
            Some(vec![4, 2, 4, 2, 4, 2, 4, 2]),
            None,
            None,
        )
        .unwrap();
        let result = fit(data.clone(), SurvfitKMOptions::default());
        assert_eq!(result.strata, Some(vec![4, 4]));
        assert_eq!(result.strata_codes, Some(vec![2, 4]));
        assert_eq!(result.n, vec![4, 4]);
        let ranges = result.curve_ranges();
        assert_eq!(ranges, vec![0..4, 4..8]);
        // each curve equals the fit of its own rows
        let mut own = data;
        own.strata = None;
        own.time = vec![1.5, 2.5, 3.5, 4.5];
        own.status = vec![0, 1, 1, 0];
        let single = fit(own, SurvfitKMOptions::default());
        assert_eq!(&result.time[0..4], single.time.as_slice());
        assert_eq!(&result.surv[0..4], single.surv.as_slice());
    }

    #[test]
    fn start_time_drops_early_observations() {
        let data =
            SurvfitKMData::right_censored(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 1, 0, 1])
                .unwrap();
        let options = SurvfitKMOptions {
            start_time: Some(2.5),
            ..Default::default()
        };
        let result = fit(data.clone(), options);
        assert_eq!(result.n, vec![3]);
        assert_eq!(result.time, vec![3.0, 4.0, 5.0]);
        assert_eq!(result.t0, 2.5);
        let err = survfitkm(
            &data,
            &SurvfitKMOptions {
                start_time: Some(10.0),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("start.time"));
    }

    #[test]
    fn reverse_estimates_the_censoring_distribution() {
        // deaths tied with a censoring leave the risk set first
        let data =
            SurvfitKMData::right_censored(vec![1.0, 2.0, 2.0, 3.0, 4.0], vec![1, 1, 0, 0, 1])
                .unwrap();
        let options = SurvfitKMOptions {
            reverse: true,
            ..Default::default()
        };
        let result = fit(data, options);
        assert_eq!(result.time, vec![1.0, 2.0, 3.0, 4.0]);
        // G(2) = 1 - 1/(4 - 1): the death at 2 is not in the risk set
        assert_vec_approx(&result.surv, &[1.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-12);
    }

    #[test]
    fn conf_lower_options_widen_only_the_lower_limit() {
        let data = SurvfitKMData::right_censored(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 0, 1, 0, 1, 0],
        )
        .unwrap();
        let usual = fit(data.clone(), SurvfitKMOptions::default());
        for conf_lower in [ConfLower::Peto, ConfLower::Modified] {
            let widened = fit(
                data.clone(),
                SurvfitKMOptions {
                    conf_lower,
                    ..Default::default()
                },
            );
            assert_eq!(widened.upper, usual.upper);
            assert_eq!(widened.conf_lower, conf_lower.as_str());
            assert_ne!(widened.lower, usual.lower);
        }
    }

    #[test]
    fn se_fit_false_and_conf_type_none_drop_components() {
        let data = SurvfitKMData::right_censored(vec![1.0, 2.0, 3.0], vec![1, 1, 0]).unwrap();
        let no_se = fit(
            data.clone(),
            SurvfitKMOptions {
                se_fit: false,
                ..Default::default()
            },
        );
        assert!(no_se.std_err.is_none() && no_se.lower.is_none());
        let no_ci = fit(
            data,
            SurvfitKMOptions {
                conf_type: ConfType::None,
                ..Default::default()
            },
        );
        assert!(no_ci.std_err.is_some() && no_ci.lower.is_none());
    }

    #[test]
    fn influence_matrices_reproduce_the_robust_variance() {
        let data = SurvfitKMData::try_new(
            None,
            vec![1.0, 2.0, 2.0, 3.0, 4.0, 5.0],
            vec![1, 1, 1, 0, 1, 0],
            Some(vec![1.0, 0.5, 1.5, 1.0, 2.0, 0.75]),
            None,
            None,
            Some(vec![1, 1, 2, 2, 3, 3]),
        )
        .unwrap();
        for (stype, ctype) in [
            (SurvType::KaplanMeier, HazardType::NelsonAalen),
            (SurvType::KaplanMeier, HazardType::FlemingHarrington),
            (SurvType::ExpCumhaz, HazardType::NelsonAalen),
            (SurvType::ExpCumhaz, HazardType::FlemingHarrington),
        ] {
            let result = fit(
                data.clone(),
                SurvfitKMOptions {
                    stype,
                    ctype,
                    influence: InfluenceRequest::Both,
                    ..Default::default()
                },
            );
            let surv = &result.influence_surv.as_ref().unwrap()[0];
            let chaz = &result.influence_chaz.as_ref().unwrap()[0];
            assert_eq!(surv.cluster, vec![1, 2, 3]);
            let column_norm = |matrix: &[Vec<f64>], col: usize| -> f64 {
                matrix
                    .iter()
                    .map(|row| row[col] * row[col])
                    .sum::<f64>()
                    .sqrt()
            };
            let std_chaz = result.std_chaz.as_ref().unwrap();
            for (col, expected) in std_chaz.iter().enumerate() {
                assert!((column_norm(&chaz.values, col) - expected).abs() < 1e-12);
            }
            if stype == SurvType::KaplanMeier {
                let std_err = result.std_err.as_ref().unwrap();
                for (col, expected) in std_err.iter().enumerate() {
                    assert!((column_norm(&surv.values, col) - expected).abs() < 1e-12);
                }
            } else {
                for (row, chaz_row) in surv.values.iter().zip(&chaz.values) {
                    for (col, value) in row.iter().enumerate() {
                        assert!((value + chaz_row[col] * result.surv[col]).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn timefix_bins_near_ties() {
        let data =
            SurvfitKMData::right_censored(vec![1.0, 1.0 + 5e-10, 2.0], vec![1, 1, 0]).unwrap();
        let fixed = fit(data.clone(), SurvfitKMOptions::default());
        assert_eq!(fixed.time, vec![1.0, 2.0]);
        assert_eq!(fixed.n_event, vec![2.0, 0.0]);
        let exact = fit(
            data,
            SurvfitKMOptions {
                timefix: false,
                ..Default::default()
            },
        );
        assert_eq!(exact.time.len(), 3);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(SurvfitKMData::right_censored(vec![], vec![]).is_err());
        assert!(SurvfitKMData::right_censored(vec![1.0], vec![2]).is_err());
        assert!(
            SurvfitKMData::try_new(Some(vec![1.0]), vec![1.0], vec![1], None, None, None, None)
                .is_err()
        );
        assert!(SurvType::from_code(3).is_err());
        assert!(HazardType::from_code(0).is_err());
        assert!(InfluenceRequest::from_code(4).is_err());
        let data = SurvfitKMData::right_censored(vec![1.0], vec![1]).unwrap();
        assert!(
            survfitkm(
                &data,
                &SurvfitKMOptions {
                    conf_int: 1.5,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
