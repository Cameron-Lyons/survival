//! R's `concordancefit` (survival 3.8-12, `concordance.R`).
//!
//! The statistic is `(concordant + tied.x / 2) / (concordant + discordant +
//! tied.x)`; its variance comes from the per-observation influence values
//! (`var`, summed within clusters when a cluster is given) and, for
//! comparison, from the Cox-model formula (`cvar`).  Each stratum and each
//! predictor column is one `concordance_sweep` (`kernels.rs`)
//! over the data ordered by decreasing time.
//!
//! Two R behaviours are not reproduced: the time-shift trick R uses to fold
//! more than ten strata into one sweep (`timewt = "n"`/`"I"` only) is a
//! speed optimisation with identical results, so the strata are always
//! looped over; and with several predictors and strata R stops with an
//! error, whereas here the counts are pooled over strata per predictor, as
//! for a single predictor.

use crate::concordance::kernels::{
    FastKm, SweepInput, SweepOutput, btree, concordance_sweep, fastkm,
};
use crate::core::strata_order::{SurvResponse, stratum_groups, validate_intervals};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{
    validate_binary_i32, validate_finite, validate_length, validate_non_negative,
};
use ndarray::{Array2, ArrayView2};
use pyo3::prelude::*;

/// R's `timewt` argument: the weight given to each event time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeWeight {
    /// `"n"`: every event time counts once (Harrell's C).
    N,
    /// `"S"`: `sum(w) S(t-) / n(t)` (Uno's weighting for a single curve).
    S,
    /// `"S/G"`: `sum(w) S(t-) / (G(t-) n(t))`.
    SOverG,
    /// `"n/G2"`: `1 / G(t-)^2` (Uno's C).
    NOverG2,
    /// `"I"`: `1 / n(t)`.
    I,
}

impl TimeWeight {
    /// Parses R's spelling of the option.
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "n" => Ok(Self::N),
            "S" => Ok(Self::S),
            "S/G" => Ok(Self::SOverG),
            "n/G2" => Ok(Self::NOverG2),
            "I" => Ok(Self::I),
            other => Err(SurvivalError::invalid_input(format!(
                "timewt must be one of \"n\", \"S\", \"S/G\", \"n/G2\", \"I\", got {other:?}"
            ))),
        }
    }
}

/// The arguments of `concordancefit` beyond the data.
#[derive(Clone, Debug)]
pub struct ConcordanceOptions {
    pub timewt: TimeWeight,
    /// Event times below `ymin` are moved up to it.
    pub ymin: Option<f64>,
    /// Event times above `ymax` get time weight zero.
    pub ymax: Option<f64>,
    /// 0: nothing extra; 1: `dfbeta`; 2: the influence matrix; 3: both.
    pub influence: u8,
    /// Return the per-event ranks (R's `ranks = TRUE`).
    pub ranks: bool,
    /// Treat larger `x` as predicting shorter times (a risk score).
    pub reverse: bool,
    /// Round nearly tied times together first (`aeqSurv`).
    pub timefix: bool,
    /// Keep per-stratum counts when there are at most this many strata.
    pub keepstrata: usize,
    /// Compute the variances (and, with them, influence and ranks).
    pub std_err: bool,
}

impl Default for ConcordanceOptions {
    fn default() -> Self {
        Self {
            timewt: TimeWeight::N,
            ymin: None,
            ymax: None,
            influence: 0,
            ranks: false,
            reverse: false,
            timefix: true,
            keepstrata: 10,
            std_err: true,
        }
    }
}

/// The five pair counts of one predictor (or one stratum).
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub struct ConcordanceCounts {
    #[pyo3(get)]
    pub concordant: f64,
    #[pyo3(get)]
    pub discordant: f64,
    #[pyo3(get)]
    pub tied_x: f64,
    #[pyo3(get)]
    pub tied_y: f64,
    #[pyo3(get)]
    pub tied_xy: f64,
}

impl ConcordanceCounts {
    fn from_sweep(count: &[f64; 6]) -> Self {
        Self {
            concordant: count[0],
            discordant: count[1],
            tied_x: count[2],
            tied_y: count[3],
            tied_xy: count[4],
        }
    }

    fn swap_concordant_discordant(&mut self) {
        std::mem::swap(&mut self.concordant, &mut self.discordant);
    }
}

/// R's `ranks` data frame: one row per event whose time weight is positive,
/// in ascending time order.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct ConcordanceRanks {
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// `(weight above - weight below) / total` in the risk set, in `[-1, 1]`.
    #[pyo3(get)]
    pub rank: Vec<f64>,
    /// Risk-set weight times the time weight.
    #[pyo3(get)]
    pub timewt: Vec<f64>,
    #[pyo3(get)]
    pub casewt: Vec<f64>,
}

/// The `concordance` object.  Vectors indexed by predictor column have
/// length one for a single predictor.
#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct ConcordanceFit {
    /// Concordance per predictor column.
    #[pyo3(get)]
    pub concordance: Vec<f64>,
    /// Number of observations.
    #[pyo3(get)]
    pub n: usize,
    /// Pair counts: one row per predictor, or one per stratum when
    /// `count_strata` is set (R's `keepstrata`).
    #[pyo3(get)]
    pub count: Vec<ConcordanceCounts>,
    /// The stratum label of each `count` row when the counts are kept per
    /// stratum.
    #[pyo3(get)]
    pub count_strata: Option<Vec<i32>>,
    /// Influence-based variance matrix (`nvar x nvar`).
    #[pyo3(get)]
    pub var: Option<Vec<Vec<f64>>>,
    /// Cox-model variance per predictor.
    #[pyo3(get)]
    pub cvar: Option<Vec<f64>>,
    /// Influence of each observation (or cluster, ascending label) on the
    /// concordance, one column per predictor.
    #[pyo3(get)]
    pub dfbeta: Option<Vec<Vec<f64>>>,
    /// Per predictor, the `n x 5` influence of each observation on the five
    /// counts.
    #[pyo3(get)]
    pub influence: Option<Vec<Vec<Vec<f64>>>>,
    /// Per predictor, the per-event ranks.
    #[pyo3(get)]
    pub ranks: Option<Vec<ConcordanceRanks>>,
}

struct SurvTimes {
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    status: Vec<i32>,
}

/// The concordance of `x` (one predictor per column) with the outcome.
pub fn concordancefit(
    response: SurvResponse<'_>,
    x: ArrayView2<'_, f64>,
    weights: Option<&[f64]>,
    strata: Option<&[i32]>,
    cluster: Option<&[i32]>,
    options: &ConcordanceOptions,
) -> SurvivalResult<ConcordanceFit> {
    let n = x.nrows();
    let nvar = x.ncols();
    if n == 0 || nvar == 0 {
        return Err(SurvivalError::invalid_input(
            "x must have at least one row and one column",
        ));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input("x contains missing values"));
    }
    if options.influence > 3 {
        return Err(SurvivalError::invalid_input(
            "influence must be 0, 1, 2 or 3",
        ));
    }
    let mut times = match response {
        SurvResponse::Right(data) => SurvTimes {
            start: None,
            stop: data.time.clone(),
            status: data.status.clone(),
        },
        SurvResponse::Counting(data) => {
            validate_intervals(&data.start, &data.stop)?;
            if matches!(options.timewt, TimeWeight::SOverG | TimeWeight::NOverG2) {
                return Err(SurvivalError::invalid_input(
                    "S/G and n/G2 timewt options are not supported for (time1, time2) data",
                ));
            }
            SurvTimes {
                start: Some(data.start.clone()),
                stop: data.stop.clone(),
                status: data.event.clone(),
            }
        }
    };
    validate_length(n, times.stop.len(), "y")?;
    validate_binary_i32(&times.status, "status")?;
    let unit = vec![1.0; n];
    let weights = match weights {
        Some(weights) => {
            validate_length(n, weights.len(), "weights")?;
            validate_finite(weights, "weights")?;
            validate_non_negative(weights, "weights")?;
            weights
        }
        None => &unit,
    };
    let ones = vec![1; n];
    let strata = match strata {
        Some(strata) => {
            validate_length(n, strata.len(), "strata")?;
            strata
        }
        None => &ones,
    };
    if let Some(cluster) = cluster {
        validate_length(n, cluster.len(), "cluster")?;
    }
    if options.timefix {
        timefix(&mut times)?;
    }
    if let Some(ymin) = options.ymin {
        for stop in &mut times.stop {
            *stop = stop.max(ymin);
        }
    }
    // Without standard errors R also drops the ranks and the influence.
    let std_err = options.std_err;
    let ranks = options.ranks && std_err;
    let influence = if std_err { options.influence } else { 0 };

    let groups = stratum_groups(strata);
    let nstrat = groups.len();
    let keepstrata = nstrat <= options.keepstrata && nvar == 1 && nstrat > 1;

    // One sweep per stratum and predictor.
    let mut per_stratum: Vec<Vec<SweepOutput>> = Vec::with_capacity(nstrat);
    let mut rank_tables: Vec<ConcordanceRanks> = (0..nvar).map(|_| empty_ranks()).collect();
    for (_, rows) in &groups {
        let mut per_x = Vec::with_capacity(nvar);
        for column in 0..nvar {
            let risk: Vec<f64> = rows.iter().map(|&i| x[[i, column]]).collect();
            let sweep = docount(&times, rows, &risk, weights, options, std_err, ranks)?;
            if ranks {
                append_ranks(&mut rank_tables[column], &times, rows, &sweep);
            }
            per_x.push(sweep);
        }
        per_stratum.push(per_x);
    }

    // Pooled counts per predictor.
    let pooled: Vec<[f64; 6]> = (0..nvar)
        .map(|column| {
            let mut total = [0.0; 6];
            for per_x in &per_stratum {
                for (sum, value) in total.iter_mut().zip(per_x[column].count) {
                    *sum += value;
                }
            }
            total
        })
        .collect();
    let npair: Vec<f64> = pooled.iter().map(|c| c[0] + c[1] + c[2]).collect();
    let somer: Vec<f64> = pooled
        .iter()
        .zip(&npair)
        .map(|(c, npair)| (c[0] - c[1]) / npair)
        .collect();
    let mut concordance: Vec<f64> = somer.iter().map(|s| (s + 1.0) / 2.0).collect();
    let mut count: Vec<ConcordanceCounts> = if keepstrata {
        per_stratum
            .iter()
            .map(|per_x| ConcordanceCounts::from_sweep(&per_x[0].count))
            .collect()
    } else {
        pooled.iter().map(ConcordanceCounts::from_sweep).collect()
    };
    let count_strata = keepstrata.then(|| groups.iter().map(|(label, _)| *label).collect());

    let mut var: Option<Vec<Vec<f64>>> = None;
    let mut cvar: Option<Vec<f64>> = None;
    let mut dfbeta: Option<Vec<Vec<f64>>> = None;
    let mut influence_out: Option<Vec<Vec<Vec<f64>>>> = None;
    if std_err {
        cvar = Some(
            pooled
                .iter()
                .zip(&npair)
                .map(|(c, npair)| c[5] / (4.0 * npair * npair))
                .collect(),
        );
        // Influence per predictor, rows back in input order.
        let mut infl: Vec<Vec<[f64; 5]>> = vec![vec![[0.0; 5]; n]; nvar];
        for ((_, rows), per_x) in groups.iter().zip(&per_stratum) {
            for (column, sweep) in per_x.iter().enumerate() {
                for (k, &row) in rows.iter().enumerate() {
                    infl[column][row] = sweep.influence[k];
                }
            }
        }
        // d(A/B) = (dA - dB A/B) / B with A = C - D and B = C + D + Tx.
        let mut df = Array2::zeros((n, nvar));
        for column in 0..nvar {
            for row in 0..n {
                let inf = infl[column][row];
                df[[row, column]] =
                    ((inf[0] - inf[1]) - (inf[0] + inf[1] + inf[2]) * somer[column]) * weights[row]
                        / (2.0 * npair[column]);
            }
        }
        let df = match cluster {
            Some(cluster) => rowsum(&df, cluster),
            None => df,
        };
        var = Some(
            df.t()
                .dot(&df)
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
        );
        if influence == 1 || influence == 3 {
            dfbeta = Some(df.outer_iter().map(|row| row.to_vec()).collect());
        }
        if influence >= 2 {
            influence_out = Some(
                infl.iter()
                    .map(|rows| rows.iter().map(|row| row.to_vec()).collect())
                    .collect(),
            );
        }
    }

    if options.reverse {
        for c in &mut concordance {
            *c = 1.0 - *c;
        }
        for row in &mut count {
            row.swap_concordant_discordant();
        }
        if let Some(dfbeta) = &mut dfbeta {
            for row in dfbeta.iter_mut() {
                for value in row.iter_mut() {
                    *value = -*value;
                }
            }
        }
        if let Some(influence) = &mut influence_out {
            for rows in influence.iter_mut() {
                for row in rows.iter_mut() {
                    row.swap(0, 1);
                }
            }
        }
        for table in &mut rank_tables {
            for rank in &mut table.rank {
                *rank = -*rank;
            }
        }
    }

    Ok(ConcordanceFit {
        concordance,
        n,
        count,
        count_strata,
        var,
        cvar,
        dfbeta,
        influence: influence_out,
        ranks: ranks.then_some(rank_tables),
    })
}

/// `aeqSurv`: bin times that differ by less than `sqrt(.Machine$double.eps)`
/// (absolutely or relative to the mean absolute time).
fn timefix(times: &mut SurvTimes) -> SurvivalResult<()> {
    let fixed = crate::data_prep::aeq_surv(&times.stop, times.start.as_deref(), None)?;
    if let (Some(start), Some(fixed_start)) = (&mut times.start, fixed.time2) {
        start.copy_from_slice(&fixed_start);
    }
    times.stop = fixed.time;
    Ok(())
}

/// R's `docount`: one predictor within one stratum.
fn docount(
    times: &SurvTimes,
    rows: &[usize],
    risk: &[f64],
    weights: &[f64],
    options: &ConcordanceOptions,
    std_err: bool,
    ranks: bool,
) -> SurvivalResult<SweepOutput> {
    let n = rows.len();
    let stop: Vec<f64> = rows.iter().map(|&i| times.stop[i]).collect();
    let status: Vec<i32> = rows.iter().map(|&i| times.status[i]).collect();
    let start: Option<Vec<f64>> = times
        .start
        .as_ref()
        .map(|start| rows.iter().map(|&i| start[i]).collect());
    let wts: Vec<f64> = rows.iter().map(|&i| weights[i]).collect();
    let nevent = status.iter().filter(|&&s| s == 1).count();
    if nevent == 0 {
        // A stratum without events contributes nothing.
        return Ok(SweepOutput {
            count: [0.0; 6],
            influence: vec![[0.0; 5]; if std_err { n } else { 0 }],
            resid: Vec::new(),
        });
    }
    // With a single event every time weighting is the same.
    let timeopt = if nevent < 2 {
        TimeWeight::N
    } else {
        options.timewt
    };

    // Reverse time, censored before deaths, then by predictor.
    let mut sort_stop: Vec<usize> = (0..n).collect();
    sort_stop.sort_by(|&a, &b| {
        stop[b]
            .total_cmp(&stop[a])
            .then_with(|| status[a].cmp(&status[b]))
            .then_with(|| risk[a].total_cmp(&risk[b]))
    });
    let sort_start: Option<Vec<usize>> = start.as_ref().map(|start| {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| start[b].total_cmp(&start[a]));
        order
    });

    let (etime, timewt): (Vec<f64>, Vec<f64>) = if timeopt == TimeWeight::N {
        let mut etime: Vec<f64> = (0..n)
            .filter(|&i| status[i] == 1)
            .map(|i| stop[i])
            .collect();
        etime.sort_by(f64::total_cmp);
        etime.dedup();
        let weights = vec![1.0; etime.len()];
        (etime, weights)
    } else {
        let FastKm {
            etime,
            surv,
            censor,
            nrisk,
        } = fastkm(
            start.as_deref(),
            &stop,
            &status,
            &wts,
            sort_start.as_deref(),
            &sort_stop,
        );
        let total: f64 = wts.iter().sum();
        // A death time at which no case weight is at risk (only zero-weight
        // observations remain) can weight nothing: every term it would
        // scale is already 0, so give it weight 0 instead of `1/0`.
        let timewt: Vec<f64> = (0..etime.len())
            .map(|k| match timeopt {
                _ if nrisk[k] <= 0.0 => 0.0,
                TimeWeight::S => total * surv[k] / nrisk[k],
                TimeWeight::SOverG => total * surv[k] / (censor[k] * nrisk[k]),
                TimeWeight::NOverG2 => 1.0 / (censor[k] * censor[k]),
                TimeWeight::I => 1.0 / nrisk[k],
                TimeWeight::N => 1.0,
            })
            .collect();
        (etime, timewt)
    };
    if timewt.iter().any(|w| !w.is_finite()) {
        return Err(SurvivalError::computation(
            "concordance: non-finite time weight (program error, notify author)",
        ));
    }
    let mut timewt = timewt;
    if let Some(ymax) = options.ymax {
        for (weight, &time) in timewt.iter_mut().zip(&etime) {
            if time > ymax {
                *weight = 0.0;
            }
        }
    }
    timewt.reverse();

    // Tree node of each predictor value among the sorted unique values.
    let mut levels = risk.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let tree = btree(levels.len());
    let node: Vec<usize> = risk
        .iter()
        .map(|value| tree[levels.partition_point(|level| level < value)])
        .collect();

    Ok(concordance_sweep(
        &SweepInput {
            start: start.as_deref(),
            stop: &stop,
            status: &status,
            node: &node,
            weight: &wts,
            timewt: &timewt,
            sort_start: sort_start.as_deref(),
            sort_stop: &sort_stop,
        },
        std_err,
        ranks,
    ))
}

fn empty_ranks() -> ConcordanceRanks {
    ConcordanceRanks {
        time: Vec::new(),
        rank: Vec::new(),
        timewt: Vec::new(),
        casewt: Vec::new(),
    }
}

/// Appends a stratum's rank rows (ascending event time, rows with zero
/// time weight dropped, as R does).
fn append_ranks(
    table: &mut ConcordanceRanks,
    times: &SurvTimes,
    rows: &[usize],
    sweep: &SweepOutput,
) {
    let mut death_times: Vec<f64> = rows
        .iter()
        .filter(|&&i| times.status[i] == 1)
        .map(|&i| times.stop[i])
        .collect();
    death_times.sort_by(f64::total_cmp);
    for (time, resid) in death_times.iter().zip(&sweep.resid) {
        if resid[1] > 0.0 {
            table.time.push(*time);
            table.rank.push(resid[0]);
            table.timewt.push(resid[1]);
            table.casewt.push(resid[2]);
        }
    }
}

/// R's `rowsum(x, group)`: column sums within each group, groups in
/// ascending label order.
fn rowsum(values: &Array2<f64>, group: &[i32]) -> Array2<f64> {
    let groups = stratum_groups(group);
    let mut out = Array2::zeros((groups.len(), values.ncols()));
    for (g, (_, rows)) in groups.iter().enumerate() {
        for &row in rows {
            for column in 0..values.ncols() {
                out[[g, column]] += values[[row, column]];
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    //! Reference values from R 4.5.3 / survival 3.8.11 on the data below:
    //! `concordance(Surv(time, status) ~ x, d, ...)`.
    use super::*;
    use crate::internal::typed_inputs::{CountingProcessData, SurvivalData};
    use ndarray::Array2;

    fn right(time: &[f64], status: &[i32]) -> SurvivalData {
        SurvivalData::try_new(time.to_vec(), status.to_vec()).unwrap()
    }

    fn column(values: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap()
    }

    fn fit(
        time: &[f64],
        status: &[i32],
        x: &[f64],
        options: &ConcordanceOptions,
    ) -> ConcordanceFit {
        let data = right(time, status);
        concordancefit(
            SurvResponse::Right(&data),
            column(x).view(),
            None,
            None,
            None,
            options,
        )
        .unwrap()
    }

    fn assert_counts(actual: &ConcordanceCounts, expected: [f64; 5]) {
        let actual = [
            actual.concordant,
            actual.discordant,
            actual.tied_x,
            actual.tied_y,
            actual.tied_xy,
        ];
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "{actual:?} != {expected:?}");
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    const TIME: [f64; 8] = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
    const STATUS: [i32; 8] = [1, 1, 0, 1, 1, 1, 0, 1];
    const X: [f64; 8] = [0.5, 0.2, 0.5, 0.9, 0.2, 0.7, 0.1, 0.9];

    #[test]
    fn counts_variances_influence_and_ranks_match_r() {
        let out = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                influence: 3,
                ranks: true,
                ..ConcordanceOptions::default()
            },
        );
        assert_eq!(out.n, 8);
        assert_counts(&out.count[0], [9.0, 9.0, 3.0, 1.0, 0.0]);
        assert_close(out.concordance[0], 0.5);
        assert_close(out.var.as_ref().unwrap()[0][0], 0.029478458049886618);
        assert_close(out.cvar.as_ref().unwrap()[0], 0.03020327178490444);
        let dfbeta = out.dfbeta.as_ref().unwrap();
        let expected = [
            0.0,
            0.047619047619047616,
            0.023809523809523808,
            -0.023809523809523808,
            -0.047619047619047616,
            0.023809523809523808,
            -0.119047619047619041,
            0.095238095238095233,
        ];
        for (row, expected) in dfbeta.iter().zip(expected) {
            assert_close(row[0], expected);
        }
        let influence = &out.influence.as_ref().unwrap()[0];
        let expected = [
            [3.0, 3.0, 1.0, 0.0, 0.0],
            [4.0, 2.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 1.0, 0.0],
            [3.0, 2.0, 0.0, 1.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 1.0, 0.0, 0.0],
        ];
        for (row, expected) in influence.iter().zip(expected) {
            assert_eq!(row.as_slice(), &expected);
        }
        let ranks = &out.ranks.as_ref().unwrap()[0];
        assert_eq!(ranks.time, vec![1.0, 2.0, 3.0, 4.0, 4.0, 6.0]);
        let expected = [0.0, 0.42857142857142855, -0.6, -0.25, 0.25, 0.0];
        for (actual, expected) in ranks.rank.iter().zip(expected) {
            assert_close(*actual, expected);
        }
        assert_eq!(ranks.timewt, vec![8.0, 7.0, 5.0, 4.0, 4.0, 1.0]);
        assert_eq!(ranks.casewt, vec![1.0; 6]);
    }

    #[test]
    fn reverse_flips_the_statistic_and_the_counts() {
        let plain = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                influence: 3,
                ranks: true,
                ..ConcordanceOptions::default()
            },
        );
        let reversed = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                reverse: true,
                influence: 3,
                ranks: true,
                ..ConcordanceOptions::default()
            },
        );
        assert_close(reversed.concordance[0], 1.0 - plain.concordance[0]);
        assert_eq!(reversed.count[0].concordant, plain.count[0].discordant);
        assert_eq!(reversed.count[0].discordant, plain.count[0].concordant);
        assert_eq!(reversed.count[0].tied_x, plain.count[0].tied_x);
        assert_eq!(reversed.var, plain.var);
        assert_eq!(reversed.cvar, plain.cvar);
        let (plain_df, reversed_df) = (plain.dfbeta.unwrap(), reversed.dfbeta.unwrap());
        for (a, b) in plain_df.iter().zip(&reversed_df) {
            assert_eq!(a[0], -b[0]);
        }
        let (plain_inf, reversed_inf) = (plain.influence.unwrap(), reversed.influence.unwrap());
        for (a, b) in plain_inf[0].iter().zip(&reversed_inf[0]) {
            assert_eq!(vec![a[1], a[0], a[2], a[3], a[4]], *b);
        }
        let (plain_ranks, reversed_ranks) = (plain.ranks.unwrap(), reversed.ranks.unwrap());
        for (a, b) in plain_ranks[0].rank.iter().zip(&reversed_ranks[0].rank) {
            assert_eq!(*a, -b);
        }
    }

    #[test]
    fn ymax_zeroes_late_event_times() {
        let out = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                ymax: Some(3.0),
                ..ConcordanceOptions::default()
            },
        );
        assert_counts(&out.count[0], [7.0, 7.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn strata_keep_per_stratum_counts_and_pool_the_statistic() {
        let data = right(&TIME, &STATUS);
        let strata = [1, 1, 1, 1, 2, 2, 2, 2];
        let out = concordancefit(
            SurvResponse::Right(&data),
            column(&X).view(),
            None,
            Some(&strata),
            None,
            &ConcordanceOptions::default(),
        )
        .unwrap();
        assert_eq!(out.count.len(), 2);
        assert_eq!(out.count_strata, Some(vec![1, 2]));
        assert_counts(&out.count[0], [3.0, 1.0, 1.0, 0.0, 0.0]);
        assert_counts(&out.count[1], [2.0, 2.0, 0.0, 1.0, 0.0]);
        assert_close(out.concordance[0], 0.61111111111111116);
        assert_close(out.var.as_ref().unwrap()[0][0], 0.037265660722450848);
        // keepstrata = 0 collapses the counts.
        let collapsed = concordancefit(
            SurvResponse::Right(&data),
            column(&X).view(),
            None,
            Some(&strata),
            None,
            &ConcordanceOptions {
                keepstrata: 0,
                ..ConcordanceOptions::default()
            },
        )
        .unwrap();
        assert_eq!(collapsed.count.len(), 1);
        assert!(collapsed.count_strata.is_none());
        assert_counts(&collapsed.count[0], [5.0, 3.0, 1.0, 1.0, 0.0]);
        assert_eq!(collapsed.var, out.var);
    }

    #[test]
    fn time_weights_match_r() {
        let cases = [
            (
                TimeWeight::S,
                [9.4, 10.0, 3.2, 1.2, 0.0],
                0.031057472926532059,
            ),
            (
                TimeWeight::SOverG,
                [9.88, 11.2, 3.44, 1.44, 0.0],
                0.032793061463206923,
            ),
            (
                TimeWeight::NOverG2,
                [9.88, 11.2, 3.44, 1.44, 0.0],
                0.032793061463206943,
            ),
            (
                TimeWeight::I,
                [
                    1.4464285714285714,
                    1.6178571428571429,
                    0.46785714285714286,
                    0.25,
                    0.0,
                ],
                0.036680341889530892,
            ),
        ];
        for (timewt, counts, var) in cases {
            let out = fit(
                &TIME,
                &STATUS,
                &X,
                &ConcordanceOptions {
                    timewt,
                    ..ConcordanceOptions::default()
                },
            );
            assert_counts(&out.count[0], counts);
            assert!(
                (out.var.as_ref().unwrap()[0][0] - var).abs() < 1e-12,
                "{timewt:?} var"
            );
        }
        let out = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                timewt: TimeWeight::S,
                ..ConcordanceOptions::default()
            },
        );
        assert_close(out.cvar.as_ref().unwrap()[0], 0.027801008021659596);
    }

    #[test]
    fn case_weights_and_clusters_match_r() {
        let data = right(&TIME, &STATUS);
        let weights = [1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 0.5, 1.5];
        let options = ConcordanceOptions {
            influence: 1,
            ..ConcordanceOptions::default()
        };
        let weighted = concordancefit(
            SurvResponse::Right(&data),
            column(&X).view(),
            Some(&weights),
            None,
            None,
            &options,
        )
        .unwrap();
        assert_counts(&weighted.count[0], [20.5, 11.25, 4.75, 2.0, 0.0]);
        assert_close(weighted.var.as_ref().unwrap()[0][0], 0.019449795103506584);
        let expected = [
            -0.0106961906549071124,
            0.0540439106774254080,
            0.0084912741602552067,
            -0.0519328204165884835,
            -0.0431131544379808607,
            0.0233627322199286905,
            -0.0643882529555263594,
            0.0842325014073934952,
        ];
        for (row, expected) in weighted.dfbeta.unwrap().iter().zip(expected) {
            assert_close(row[0], expected);
        }

        let cluster = [1, 1, 2, 2, 3, 3, 4, 4];
        let clustered = concordancefit(
            SurvResponse::Right(&data),
            column(&X).view(),
            None,
            None,
            Some(&cluster),
            &options,
        )
        .unwrap();
        assert_close(clustered.var.as_ref().unwrap()[0][0], 0.0034013605442176865);
        let expected = [
            0.047619047619047616,
            0.0,
            -0.023809523809523808,
            -0.023809523809523808,
        ];
        let dfbeta = clustered.dfbeta.unwrap();
        assert_eq!(dfbeta.len(), 4);
        for (row, expected) in dfbeta.iter().zip(expected) {
            assert_close(row[0], expected);
        }
    }

    #[test]
    fn zero_weight_deaths_are_no_ops_for_every_time_weight() {
        // A death with case weight 0 contributes nothing, so the fit must
        // equal the fit without that observation (R indexes past its time
        // weight vector here); the extreme case is a zero-weight death at
        // the largest time, where nothing with positive weight is at risk.
        let all = [
            TimeWeight::N,
            TimeWeight::S,
            TimeWeight::SOverG,
            TimeWeight::NOverG2,
            TimeWeight::I,
        ];
        let weights = [1.0, 2.0, 0.5, 1.5, 1.0, 2.0, 0.5, 1.5];
        for dropped in [3usize, 7] {
            let mut zeroed = weights;
            zeroed[dropped] = 0.0;
            let keep: Vec<usize> = (0..8).filter(|&i| i != dropped).collect();
            let time: Vec<f64> = keep.iter().map(|&i| TIME[i]).collect();
            let status: Vec<i32> = keep.iter().map(|&i| STATUS[i]).collect();
            let x: Vec<f64> = keep.iter().map(|&i| X[i]).collect();
            let wt: Vec<f64> = keep.iter().map(|&i| weights[i]).collect();
            let full = right(&TIME, &STATUS);
            let reduced = right(&time, &status);
            for timewt in all {
                let options = ConcordanceOptions {
                    timewt,
                    influence: 1,
                    ..ConcordanceOptions::default()
                };
                let with_zero = concordancefit(
                    SurvResponse::Right(&full),
                    column(&X).view(),
                    Some(&zeroed),
                    None,
                    None,
                    &options,
                )
                .unwrap();
                let without = concordancefit(
                    SurvResponse::Right(&reduced),
                    column(&x).view(),
                    Some(&wt),
                    None,
                    None,
                    &options,
                )
                .unwrap();
                assert_eq!(with_zero.n, 8);
                assert_eq!(
                    with_zero.count, without.count,
                    "{timewt:?} dropped {dropped}"
                );
                assert_close(with_zero.concordance[0], without.concordance[0]);
                assert_close(
                    with_zero.var.as_ref().unwrap()[0][0],
                    without.var.as_ref().unwrap()[0][0],
                );
                let dfbeta = with_zero.dfbeta.as_ref().unwrap();
                assert_eq!(dfbeta[dropped][0], 0.0);
                for (&i, row) in keep.iter().zip(without.dfbeta.as_ref().unwrap()) {
                    assert_close(dfbeta[i][0], row[0]);
                }
                if dropped == 3 {
                    assert_close(
                        with_zero.cvar.as_ref().unwrap()[0],
                        without.cvar.as_ref().unwrap()[0],
                    );
                }
            }
            // (start, stop] data takes the same path through fastkm2.
            let start = vec![0.0; 8];
            let counting =
                CountingProcessData::try_new(start, TIME.to_vec(), STATUS.to_vec()).unwrap();
            for timewt in [TimeWeight::S, TimeWeight::I] {
                let options = ConcordanceOptions {
                    timewt,
                    ..ConcordanceOptions::default()
                };
                let with_zero = concordancefit(
                    SurvResponse::Counting(&counting),
                    column(&X).view(),
                    Some(&zeroed),
                    None,
                    None,
                    &options,
                )
                .unwrap();
                let without = concordancefit(
                    SurvResponse::Right(&reduced),
                    column(&x).view(),
                    Some(&wt),
                    None,
                    None,
                    &options,
                )
                .unwrap();
                assert_eq!(
                    with_zero.count, without.count,
                    "{timewt:?} dropped {dropped}"
                );
                assert_close(
                    with_zero.var.as_ref().unwrap()[0][0],
                    without.var.as_ref().unwrap()[0][0],
                );
            }
        }
    }

    #[test]
    fn several_predictors_give_a_covariance_matrix() {
        let data = right(&TIME, &STATUS);
        let y: Vec<f64> = X.iter().map(|v| 1.0 - v).collect();
        let mut x = Array2::zeros((8, 2));
        for i in 0..8 {
            x[[i, 0]] = X[i];
            x[[i, 1]] = y[i];
        }
        let out = concordancefit(
            SurvResponse::Right(&data),
            x.view(),
            None,
            None,
            None,
            &ConcordanceOptions {
                influence: 1,
                ..ConcordanceOptions::default()
            },
        )
        .unwrap();
        let single = fit(&TIME, &STATUS, &X, &ConcordanceOptions::default());
        assert_eq!(out.concordance.len(), 2);
        assert_close(out.concordance[0], single.concordance[0]);
        assert_close(out.concordance[1], 1.0 - single.concordance[0]);
        let var = out.var.as_ref().unwrap();
        assert_eq!(var.len(), 2);
        assert_close(var[0][0], single.var.as_ref().unwrap()[0][0]);
        assert_close(var[0][1], -var[0][0]);
        assert_eq!(out.dfbeta.as_ref().unwrap()[0].len(), 2);
        assert_eq!(out.cvar.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn counting_process_data_with_zero_starts_matches_right_censored() {
        let start = vec![0.0; 8];
        let counting = CountingProcessData::try_new(start, TIME.to_vec(), STATUS.to_vec()).unwrap();
        let options = ConcordanceOptions {
            influence: 3,
            ranks: true,
            timewt: TimeWeight::S,
            ..ConcordanceOptions::default()
        };
        let plain = fit(&TIME, &STATUS, &X, &options);
        let counting = concordancefit(
            SurvResponse::Counting(&counting),
            column(&X).view(),
            None,
            None,
            None,
            &options,
        )
        .unwrap();
        assert_close(plain.concordance[0], counting.concordance[0]);
        assert_eq!(plain.count, counting.count);
        assert_close(plain.var.unwrap()[0][0], counting.var.unwrap()[0][0]);
        assert_eq!(plain.influence, counting.influence);
        assert_eq!(plain.ranks, counting.ranks);
    }

    #[test]
    fn std_err_false_returns_counts_only() {
        let out = fit(
            &TIME,
            &STATUS,
            &X,
            &ConcordanceOptions {
                std_err: false,
                influence: 3,
                ranks: true,
                ..ConcordanceOptions::default()
            },
        );
        let full = fit(&TIME, &STATUS, &X, &ConcordanceOptions::default());
        assert_eq!(out.count, full.count);
        assert_eq!(out.concordance, full.concordance);
        assert!(out.var.is_none() && out.cvar.is_none());
        assert!(out.dfbeta.is_none() && out.influence.is_none() && out.ranks.is_none());
    }

    #[test]
    fn rejects_invalid_arguments() {
        let data = right(&TIME, &STATUS);
        let x = column(&X);
        let bad = concordancefit(
            SurvResponse::Right(&data),
            x.view(),
            Some(&[1.0; 7]),
            None,
            None,
            &ConcordanceOptions::default(),
        );
        assert!(bad.is_err());
        let counting =
            CountingProcessData::try_new(vec![0.0; 8], TIME.to_vec(), STATUS.to_vec()).unwrap();
        let bad = concordancefit(
            SurvResponse::Counting(&counting),
            x.view(),
            None,
            None,
            None,
            &ConcordanceOptions {
                timewt: TimeWeight::NOverG2,
                ..ConcordanceOptions::default()
            },
        );
        assert!(bad.unwrap_err().to_string().contains("not supported"));
        assert!(TimeWeight::parse("G").is_err());
        assert_eq!(TimeWeight::parse("S/G").unwrap(), TimeWeight::SOverG);
    }
}
