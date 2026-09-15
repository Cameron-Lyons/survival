//! Case-cohort analysis: R survival's `cch()` (`R/cch.R`) with the
//! Prentice, Self-Prentice, Lin-Ying and Borgan I / II estimators.
//!
//! Each estimator is a Cox fit on a rearranged data set (cases entered
//! just before their failure, offsets of `-100` for the pseudo-cases,
//! sampling weights for Borgan) followed by a phase-two variance built from
//! the dfbeta or score residuals of that fit; everything Cox-related goes
//! through [`CoxPHFit`].

use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{CoxPHFit, CoxphData, CoxphOptions};
use crate::regression::coxph_diagnostics::{ResidualType, Residuals};
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnstratifiedMethod {
    Prentice,
    SelfPrentice,
    LinYing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BorganMethod {
    I,
    II,
}

fn normalized(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

impl UnstratifiedMethod {
    fn parse(value: &str) -> SurvivalResult<Self> {
        match normalized(value).as_str() {
            "prentice" => Ok(Self::Prentice),
            "selfprentice" => Ok(Self::SelfPrentice),
            "linying" => Ok(Self::LinYing),
            _ => Err(SurvivalError::invalid_input(
                "method must be 'Prentice', 'SelfPrentice', or 'LinYing'",
            )),
        }
    }

    fn r_name(self) -> &'static str {
        match self {
            Self::Prentice => "Prentice",
            Self::SelfPrentice => "SelfPrentice",
            Self::LinYing => "LinYing",
        }
    }
}

impl BorganMethod {
    fn parse(value: &str) -> SurvivalResult<Self> {
        match normalized(value).as_str() {
            "iborgan" => Ok(Self::I),
            "iiborgan" => Ok(Self::II),
            _ => Err(SurvivalError::invalid_input(
                "method must be 'I.Borgan' or 'II.Borgan'",
            )),
        }
    }

    fn r_name(self) -> &'static str {
        match self {
            Self::I => "I.Borgan",
            Self::II => "II.Borgan",
        }
    }
}

/// R's `cch` object: the underlying Cox fit plus the case-cohort variance
/// pieces.
#[derive(Debug, Clone)]
#[pyclass(skip_from_py_object)]
pub struct CchFitResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    /// `var`: the case-cohort variance (`naive_var`, or the robust
    /// Lin-Ying variance).
    #[pyo3(get)]
    pub var: Vec<Vec<f64>>,
    /// Model-based variance of the Cox fit plus `phase2var`.
    #[pyo3(get)]
    pub naive_var: Vec<Vec<f64>>,
    /// Variance from the phase-two (subcohort) sampling.
    #[pyo3(get)]
    pub phase2var: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub method: String,
    /// Cohort size per stratum (one entry when unstratified).
    #[pyo3(get)]
    pub cohort_size: Vec<usize>,
    /// Subcohort size per stratum.
    #[pyo3(get)]
    pub subcohort_size: Vec<usize>,
    #[pyo3(get)]
    pub stratified: bool,
    /// Stratum of each input row (Borgan methods).
    #[pyo3(get)]
    pub stratum: Option<Vec<usize>>,
    #[pyo3(get)]
    pub robust: bool,
    /// Borgan's per-stratum optimal allocation fractions (`opt`).
    #[pyo3(get)]
    pub opt: Option<Vec<Vec<f64>>>,
    /// Borgan's phase-two score matrix (`delta`).
    #[pyo3(get)]
    pub delta: Option<Vec<Vec<f64>>>,
    /// Borgan's weighted score residuals collapsed by `id` (`sc`).
    #[pyo3(get)]
    pub sc: Option<Vec<Vec<f64>>>,
    /// The Cox fit the estimator is built on (Prentice: the augmented
    /// data set; its coefficients are replaced by `coefficients`).
    #[pyo3(get)]
    pub fit: CoxPHFit,
}

fn matrix_rows(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.outer_iter().map(|row| row.to_vec()).collect()
}

fn matrix_from_rows(rows: &[Vec<f64>], ncols: usize) -> Array2<f64> {
    Array2::from_shape_vec(
        (rows.len(), ncols),
        rows.iter().flatten().copied().collect(),
    )
    .expect("rectangular rows")
}

fn validate_cch_inputs(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    subcohort: &[i32],
    id: &[i64],
    cohort_size: usize,
) -> SurvivalResult<usize> {
    let n = stop.len();
    if n == 0 {
        return Err(SurvivalError::invalid_input("stop must not be empty"));
    }
    for (name, len) in [
        ("status", status.len()),
        ("covariates", covariates.len()),
        ("subcohort", subcohort.len()),
        ("id", id.len()),
    ] {
        if len != n {
            return Err(SurvivalError::invalid_input(format!(
                "{name} has {len} rows but stop has {n}"
            )));
        }
    }
    if let Some(values) = start
        && values.len() != n
    {
        return Err(SurvivalError::invalid_input(format!(
            "start has {} rows but stop has {n}",
            values.len()
        )));
    }
    if cohort_size < n {
        return Err(SurvivalError::invalid_input(
            "Number of records greater than cohort size",
        ));
    }
    if status.iter().any(|&value| value != 0 && value != 1) {
        return Err(SurvivalError::invalid_input(
            "status must contain only 0/1 values",
        ));
    }
    if subcohort.iter().any(|&value| value != 0 && value != 1) {
        return Err(SurvivalError::invalid_input(
            "Permissible values for subcohort indicator are 0/1 or TRUE/FALSE",
        ));
    }
    let outside_censored = status
        .iter()
        .zip(subcohort)
        .filter(|&(&event, &sampled)| event == 0 && sampled == 0)
        .count();
    if outside_censored > 0 {
        return Err(SurvivalError::invalid_input(format!(
            "{outside_censored} censored observations not in subcohort"
        )));
    }
    if id.iter().copied().collect::<HashSet<_>>().len() != n {
        return Err(SurvivalError::invalid_input(
            "Multiple records per id not allowed",
        ));
    }
    if stop.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "stop must contain only finite values",
        ));
    }
    if let Some(values) = start {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SurvivalError::invalid_input(
                "start must contain only finite values",
            ));
        }
        if values.iter().zip(stop).any(|(&entry, &exit)| entry >= exit) {
            return Err(SurvivalError::invalid_input(
                "every start value must be less than stop",
            ));
        }
    }
    let width = covariates.first().map_or(0, Vec::len);
    if width == 0 {
        return Err(SurvivalError::invalid_input(
            "covariates must contain at least one column",
        ));
    }
    if covariates.iter().any(|row| row.len() != width) {
        return Err(SurvivalError::invalid_input(
            "covariates must be rectangular",
        ));
    }
    if covariates.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SurvivalError::invalid_input(
            "covariates must contain only finite values",
        ));
    }
    Ok(width)
}

#[allow(clippy::too_many_arguments)]
fn validate_borgan_inputs(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    subcohort: &[i32],
    id: &[i64],
    stratum: &[usize],
    cohort_sizes: &[usize],
) -> SurvivalResult<usize> {
    let cohort_size = cohort_sizes.iter().try_fold(0usize, |total, &size| {
        total
            .checked_add(size)
            .ok_or_else(|| SurvivalError::invalid_input("cohort_sizes sum is too large"))
    })?;
    let width = validate_cch_inputs(stop, status, covariates, start, subcohort, id, cohort_size)?;
    if cohort_sizes.is_empty() {
        return Err(SurvivalError::invalid_input(
            "cohort_sizes must not be empty",
        ));
    }
    if cohort_sizes.contains(&0) {
        return Err(SurvivalError::invalid_input(
            "cohort_sizes must contain only positive values",
        ));
    }
    if stratum.len() != stop.len() {
        return Err(SurvivalError::invalid_input(format!(
            "stratum has {} rows but stop has {}",
            stratum.len(),
            stop.len()
        )));
    }
    if stratum.iter().any(|&value| value >= cohort_sizes.len()) {
        return Err(SurvivalError::invalid_input(
            "stratum codes must index every value in cohort_sizes",
        ));
    }
    let mut observed_sizes = vec![0usize; cohort_sizes.len()];
    for &code in stratum {
        observed_sizes[code] += 1;
    }
    if observed_sizes.contains(&0) {
        return Err(SurvivalError::invalid_input(
            "cohort.size and stratum do not match",
        ));
    }
    if observed_sizes
        .iter()
        .zip(cohort_sizes)
        .any(|(&observed, &population)| observed > population)
    {
        return Err(SurvivalError::invalid_input(
            "Population smaller than sample in some strata",
        ));
    }
    Ok(width)
}

/// Half the smallest gap between distinct event times (1 with a single
/// event time): the artificial entry offset of the Prentice pseudo-cases.
fn event_time_delta(stop: &[f64], status: &[i32]) -> f64 {
    let mut times: Vec<f64> = stop
        .iter()
        .zip(status)
        .filter_map(|(&time, &event)| (event == 1).then_some(time))
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    if times.len() <= 1 {
        return 1.0;
    }
    times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(f64::INFINITY, f64::min)
        / 2.0
}

struct CoxInput {
    stop: Vec<f64>,
    status: Vec<i32>,
    x: Vec<Vec<f64>>,
    start: Vec<f64>,
    offset: Vec<f64>,
    weights: Option<Vec<f64>>,
}

/// `coxph(Surv(start, stop, status) ~ x + offset, weights, init, iter.max)`
/// with `coxph`'s defaults (Efron ties, `nocenter = c(-1, 0, 1)`).
fn fit_cox(input: CoxInput, init: Option<Vec<f64>>, iter_max: usize) -> SurvivalResult<CoxPHFit> {
    let nvar = input.x.first().map_or(0, Vec::len);
    let data = CoxphData::try_new(
        input.stop,
        Some(input.start),
        input.status,
        matrix_from_rows(&input.x, nvar),
        input.weights,
        None,
        Some(input.offset),
    )?;
    let options = CoxphOptions {
        method: TieMethod::Efron,
        init,
        iter_max,
        ..CoxphOptions::default()
    };
    CoxPHFit::fit(data, options)
}

fn residual_matrix(
    fit: &CoxPHFit,
    kind: ResidualType,
    weighted: bool,
    collapse: Option<&[i32]>,
) -> SurvivalResult<Array2<f64>> {
    match fit.residuals(kind, Some(weighted), collapse, None)? {
        Residuals::Matrix(values) => Ok(values),
        Residuals::Vector(_) => unreachable!("dfbeta and score residuals are matrices"),
    }
}

struct CchComputation {
    fit: CoxPHFit,
    coefficients: Vec<f64>,
    phase2var: Array2<f64>,
    naive_var: Array2<f64>,
    var: Array2<f64>,
    robust: bool,
}

/// The Prentice (`prentice = true`) and Self-Prentice estimators: cases
/// enter the risk set just before their failure with offset `-100`, the
/// subcohort is appended censored.
fn augmented_fit(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: &[f64],
    subcohort: &[i32],
    cohort_size: usize,
    prentice: bool,
) -> SurvivalResult<CchComputation> {
    let case_indices: Vec<usize> = (0..stop.len()).filter(|&i| status[i] == 1).collect();
    let subcohort_indices: Vec<usize> = (0..stop.len()).filter(|&i| subcohort[i] == 1).collect();

    // Prentice's point estimate: the cases outside the subcohort enter
    // `delta` before their failure.
    let initial_coefficients = if prentice {
        let delta = event_time_delta(stop, status);
        let mut entry = start.to_vec();
        for idx in 0..stop.len() {
            if status[idx] == 1 && subcohort[idx] == 0 {
                let candidate = stop[idx] - delta;
                // Float-near ties can make stop - delta == stop; keep a strict entry time.
                entry[idx] = if candidate < stop[idx] {
                    candidate
                } else {
                    stop[idx].next_down()
                };
            }
        }
        let fit = fit_cox(
            CoxInput {
                stop: stop.to_vec(),
                status: status.to_vec(),
                x: covariates.to_vec(),
                start: entry,
                offset: vec![0.0; stop.len()],
                weights: None,
            },
            None,
            20,
        )?;
        Some(fit.coefficients)
    } else {
        None
    };

    let mut input = CoxInput {
        stop: Vec::new(),
        status: Vec::new(),
        x: Vec::new(),
        start: Vec::new(),
        offset: Vec::new(),
        weights: None,
    };
    for &idx in &case_indices {
        input.stop.push(stop[idx]);
        input.status.push(1);
        input.x.push(covariates[idx].clone());
        input.start.push(start[idx]);
        input.offset.push(-100.0);
    }
    for &idx in &subcohort_indices {
        input.stop.push(stop[idx]);
        input.status.push(0);
        input.x.push(covariates[idx].clone());
        input.start.push(start[idx]);
        input.offset.push(0.0);
    }
    let fit = fit_cox(
        input,
        initial_coefficients.clone(),
        if prentice { 35 } else { 20 },
    )?;
    let dfbeta = residual_matrix(&fit, ResidualType::Dfbeta, true, None)?;
    let phase2_rows = dfbeta
        .slice(ndarray::s![case_indices.len().., ..])
        .to_owned();
    let phase2_scale = 1.0 - subcohort_indices.len() as f64 / cohort_size as f64;
    let phase2var = phase2_rows.t().dot(&phase2_rows) * phase2_scale;
    let naive_var = &fit.var + &phase2var;
    let coefficients = initial_coefficients.unwrap_or_else(|| fit.coefficients.clone());
    Ok(CchComputation {
        fit,
        coefficients,
        phase2var,
        var: naive_var.clone(),
        naive_var,
        robust: false,
    })
}

/// The Lin-Ying estimator: a Cox fit with the offset
/// `log((ntot - nd) / (nc - ncd))` for the non-cases.
fn lin_ying_fit(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: &[f64],
    subcohort: &[i32],
    cohort_size: usize,
    robust: bool,
) -> SurvivalResult<CchComputation> {
    let n_events = status.iter().filter(|&&event| event == 1).count();
    let subcohort_size = subcohort.iter().filter(|&&sampled| sampled == 1).count();
    let subcohort_events = (0..stop.len())
        .filter(|&i| status[i] == 1 && subcohort[i] == 1)
        .count();
    let sampled_noncases = subcohort_size - subcohort_events;
    let cohort_noncases = cohort_size - n_events;
    if sampled_noncases == 0 || cohort_noncases == 0 {
        return Err(SurvivalError::invalid_input(
            "LinYing requires at least one sampled noncase and one cohort noncase",
        ));
    }
    let sampling_inverse = cohort_noncases as f64 / sampled_noncases as f64;
    let offsets: Vec<f64> = status
        .iter()
        .map(|&event| {
            if event == 1 {
                0.0
            } else {
                sampling_inverse.ln()
            }
        })
        .collect();
    let fit = fit_cox(
        CoxInput {
            stop: stop.to_vec(),
            status: status.to_vec(),
            x: covariates.to_vec(),
            start: start.to_vec(),
            offset: offsets,
            weights: None,
        },
        None,
        20,
    )?;
    let dfbeta = residual_matrix(&fit, ResidualType::Dfbeta, true, None)?;
    let noncase_rows: Vec<usize> = (0..stop.len()).filter(|&i| status[i] == 0).collect();
    let mut db0 = Array2::zeros((noncase_rows.len(), dfbeta.ncols()));
    for (position, &row) in noncase_rows.iter().enumerate() {
        db0.row_mut(position).assign(&dfbeta.row(row));
    }
    let means = db0
        .mean_axis(ndarray::Axis(0))
        .expect("at least one noncase");
    let centered = &db0 - &means;
    let phase2_scale = 1.0 - sampled_noncases as f64 / cohort_noncases as f64;
    let phase2var = centered.t().dot(&centered) * phase2_scale;
    let naive_var = &fit.var + &phase2var;
    let var = if robust {
        // crossprod(db, db / offs): the cases keep weight 1.
        let mut scaled = dfbeta.clone();
        for (i, mut row) in scaled.outer_iter_mut().enumerate() {
            if status[i] == 0 {
                row.mapv_inplace(|value| value / sampling_inverse);
            }
        }
        dfbeta.t().dot(&scaled) + &phase2var
    } else {
        naive_var.clone()
    };
    let coefficients = fit.coefficients.clone();
    Ok(CchComputation {
        fit,
        coefficients,
        phase2var,
        naive_var,
        var,
        robust,
    })
}

struct BorganPhaseTwo {
    variance: Array2<f64>,
    delta: Array2<f64>,
    opt: Vec<Vec<f64>>,
}

/// Borgan's equations (17) and (19): the within-stratum covariance of the
/// phase-two score residuals, scaled by the sampling fractions.
fn borgan_phase_two(
    score_rows: &Array2<f64>,
    row_strata: &[usize],
    sample_sizes: &[usize],
    population_sizes: &[usize],
    sampling_inverse: &[f64],
    model_var: &Array2<f64>,
) -> BorganPhaseTwo {
    let stratum_count = sample_sizes.len();
    let width = model_var.nrows();
    let mut means = vec![vec![0.0; width]; stratum_count];
    for (row, &stratum) in score_rows.outer_iter().zip(row_strata) {
        for (mean, &value) in means[stratum].iter_mut().zip(row.iter()) {
            *mean += value;
        }
    }
    for (mean, &sample_size) in means.iter_mut().zip(sample_sizes) {
        for value in mean.iter_mut() {
            *value /= sample_size as f64;
        }
    }
    let mut delta = Array2::zeros((width, width));
    let mut opt = vec![vec![0.0; width]; stratum_count];
    for stratum in 0..stratum_count {
        let mut crossproduct = Array2::zeros((width, width));
        for (row, &s) in score_rows.outer_iter().zip(row_strata) {
            if s != stratum {
                continue;
            }
            for i in 0..width {
                for j in 0..width {
                    crossproduct[(i, j)] +=
                        (row[i] - means[stratum][i]) * (row[j] - means[stratum][j]);
                }
            }
        }
        crossproduct /= (sample_sizes[stratum] - 1) as f64;
        let scale = (sampling_inverse[stratum] - 1.0) * population_sizes[stratum] as f64;
        delta = delta + &crossproduct * scale;
        let stratum_variance = model_var.dot(&crossproduct).dot(model_var);
        for column in 0..width {
            opt[stratum][column] = population_sizes[stratum] as f64
                * stratum_variance[(column, column)].max(0.0).sqrt();
        }
    }
    for column in 0..width {
        let total: f64 = opt.iter().map(|row| row[column]).sum();
        if total > 0.0 {
            for row in opt.iter_mut() {
                row[column] /= total;
            }
        }
    }
    BorganPhaseTwo {
        variance: model_var.dot(&delta).dot(model_var),
        delta,
        opt,
    }
}

struct BorganComputation {
    computation: CchComputation,
    opt: Vec<Vec<f64>>,
    delta: Array2<f64>,
    sc: Array2<f64>,
}

#[allow(clippy::too_many_arguments)]
fn borgan_fit(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: &[f64],
    subcohort: &[i32],
    stratum: &[usize],
    cohort_sizes: &[usize],
    method: BorganMethod,
) -> SurvivalResult<BorganComputation> {
    let observed_n = stop.len();
    let stratum_count = cohort_sizes.len();
    let mut event_counts = vec![0usize; stratum_count];
    let mut sampled_counts = vec![0usize; stratum_count];
    let mut sampled_noncase_counts = vec![0usize; stratum_count];
    for idx in 0..observed_n {
        let s = stratum[idx];
        if status[idx] == 1 {
            event_counts[s] += 1;
        }
        if subcohort[idx] == 1 {
            sampled_counts[s] += 1;
            if status[idx] == 0 {
                sampled_noncase_counts[s] += 1;
            }
        }
    }
    let (sample_sizes, population_sizes) = match method {
        BorganMethod::I => (sampled_counts.clone(), cohort_sizes.to_vec()),
        BorganMethod::II => {
            let noncase_population = cohort_sizes
                .iter()
                .zip(&event_counts)
                .map(|(&population, &events)| population.checked_sub(events))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    SurvivalError::invalid_input("a stratum has more events than cohort members")
                })?;
            (sampled_noncase_counts.clone(), noncase_population)
        }
    };
    if sample_sizes.iter().any(|&size| size < 2) {
        return Err(SurvivalError::invalid_input(
            "each Borgan sampling stratum requires at least two phase-two rows",
        ));
    }
    if sample_sizes
        .iter()
        .zip(&population_sizes)
        .any(|(&sample, &population)| sample > population)
    {
        return Err(SurvivalError::invalid_input(
            "Population smaller than sample in some strata",
        ));
    }
    let sampling_inverse: Vec<f64> = population_sizes
        .iter()
        .zip(&sample_sizes)
        .map(|(&population, &sample)| population as f64 / sample as f64)
        .collect();

    // Rows of the Cox fit and the input row each comes from.
    let mut source_indices = Vec::new();
    let mut input = CoxInput {
        stop: Vec::new(),
        status: Vec::new(),
        x: Vec::new(),
        start: Vec::new(),
        offset: Vec::new(),
        weights: Some(Vec::new()),
    };
    let weights = input.weights.as_mut().expect("weights are present");
    let mut phase2_start = 0usize;
    match method {
        BorganMethod::I => {
            let case_indices: Vec<usize> = (0..observed_n).filter(|&i| status[i] == 1).collect();
            let subcohort_indices: Vec<usize> =
                (0..observed_n).filter(|&i| subcohort[i] == 1).collect();
            phase2_start = case_indices.len();
            for idx in case_indices {
                source_indices.push(idx);
                input.stop.push(stop[idx]);
                input.status.push(1);
                input.x.push(covariates[idx].clone());
                input.start.push(start[idx]);
                input.offset.push(-100.0);
                weights.push(1.0);
            }
            for idx in subcohort_indices {
                source_indices.push(idx);
                input.stop.push(stop[idx]);
                input.status.push(0);
                input.x.push(covariates[idx].clone());
                input.start.push(start[idx]);
                input.offset.push(0.0);
                weights.push(sampling_inverse[stratum[idx]]);
            }
        }
        BorganMethod::II => {
            source_indices.extend(0..observed_n);
            input.stop.extend_from_slice(stop);
            input.status.extend_from_slice(status);
            input.x.extend_from_slice(covariates);
            input.start.extend_from_slice(start);
            input.offset.resize(observed_n, 0.0);
            weights.extend((0..observed_n).map(|idx| {
                if status[idx] == 1 {
                    1.0
                } else {
                    sampling_inverse[stratum[idx]]
                }
            }));
        }
    }

    let fit = fit_cox(input, None, 25)?;
    let score_rows = residual_matrix(&fit, ResidualType::Score, false, None)?;
    let phase2_rows: Vec<usize> = match method {
        BorganMethod::I => (phase2_start..score_rows.nrows()).collect(),
        BorganMethod::II => (0..observed_n).filter(|&i| status[i] == 0).collect(),
    };
    let mut phase2_scores = Array2::zeros((phase2_rows.len(), score_rows.ncols()));
    for (position, &row) in phase2_rows.iter().enumerate() {
        phase2_scores.row_mut(position).assign(&score_rows.row(row));
    }
    let phase2_strata: Vec<usize> = phase2_rows
        .iter()
        .map(|&row| stratum[source_indices[row]])
        .collect();
    let phase_two = borgan_phase_two(
        &phase2_scores,
        &phase2_strata,
        &sample_sizes,
        &population_sizes,
        &sampling_inverse,
        &fit.var,
    );
    let naive_var = &fit.var + &phase_two.variance;
    // resid(fit, type = "score", collapse = id, weighted = TRUE): one row per
    // input record.
    let id: Vec<i32> = source_indices.iter().map(|&idx| idx as i32).collect();
    let sc = residual_matrix(&fit, ResidualType::Score, true, Some(&id))?;
    let coefficients = fit.coefficients.clone();
    Ok(BorganComputation {
        computation: CchComputation {
            fit,
            coefficients,
            phase2var: phase_two.variance,
            var: naive_var.clone(),
            naive_var,
            robust: false,
        },
        opt: phase_two.opt,
        delta: phase_two.delta,
        sc,
    })
}

struct CchMetadata {
    method: String,
    subcohort_size: Vec<usize>,
    cohort_size: Vec<usize>,
    stratum: Option<Vec<usize>>,
    opt: Option<Vec<Vec<f64>>>,
    delta: Option<Array2<f64>>,
    sc: Option<Array2<f64>>,
}

fn finish(computation: CchComputation, metadata: CchMetadata) -> CchFitResult {
    CchFitResult {
        coefficients: computation.coefficients,
        var: matrix_rows(&computation.var),
        naive_var: matrix_rows(&computation.naive_var),
        phase2var: matrix_rows(&computation.phase2var),
        method: metadata.method,
        cohort_size: metadata.cohort_size,
        subcohort_size: metadata.subcohort_size,
        stratified: metadata.stratum.is_some(),
        stratum: metadata.stratum,
        robust: computation.robust,
        opt: metadata.opt,
        delta: metadata.delta.as_ref().map(matrix_rows),
        sc: metadata.sc.as_ref().map(matrix_rows),
        fit: computation.fit,
    }
}

/// `cch(Surv(start, stop, status) ~ x, subcoh, id, cohort.size, method, robust)`
/// for the unstratified estimators.
#[pyfunction]
#[pyo3(signature = (stop, status, covariates, subcohort, id, cohort_size, start=None, method="Prentice", robust=false))]
#[allow(clippy::too_many_arguments)]
pub fn cch_fit(
    stop: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    subcohort: Vec<i32>,
    id: Vec<i64>,
    cohort_size: usize,
    start: Option<Vec<f64>>,
    method: &str,
    robust: bool,
) -> PyResult<CchFitResult> {
    validate_cch_inputs(
        &stop,
        &status,
        &covariates,
        start.as_deref(),
        &subcohort,
        &id,
        cohort_size,
    )?;
    let method = UnstratifiedMethod::parse(method)?;
    let entry = start.unwrap_or_else(|| vec![0.0; stop.len()]);
    let subcohort_size = subcohort.iter().filter(|&&sampled| sampled == 1).count();
    let computation = match method {
        UnstratifiedMethod::Prentice => augmented_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            true,
        )?,
        UnstratifiedMethod::SelfPrentice => augmented_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            false,
        )?,
        UnstratifiedMethod::LinYing => lin_ying_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            robust,
        )?,
    };
    Ok(finish(
        computation,
        CchMetadata {
            method: method.r_name().to_string(),
            subcohort_size: vec![subcohort_size],
            cohort_size: vec![cohort_size],
            stratum: None,
            opt: None,
            delta: None,
            sc: None,
        },
    ))
}

/// `cch(..., stratum, cohort.size = per-stratum sizes, method = "I.Borgan" | "II.Borgan")`.
#[pyfunction]
#[pyo3(signature = (stop, status, covariates, subcohort, id, stratum, cohort_sizes, start=None, method="I.Borgan"))]
#[allow(clippy::too_many_arguments)]
pub fn cch_borgan_fit(
    stop: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    subcohort: Vec<i32>,
    id: Vec<i64>,
    stratum: Vec<usize>,
    cohort_sizes: Vec<usize>,
    start: Option<Vec<f64>>,
    method: &str,
) -> PyResult<CchFitResult> {
    validate_borgan_inputs(
        &stop,
        &status,
        &covariates,
        start.as_deref(),
        &subcohort,
        &id,
        &stratum,
        &cohort_sizes,
    )?;
    let method = BorganMethod::parse(method)?;
    let entry = start.unwrap_or_else(|| vec![0.0; stop.len()]);
    // R: subcohort.sizes <- table(stratum), every sampled record.
    let mut subcohort_size = vec![0usize; cohort_sizes.len()];
    for &s in &stratum {
        subcohort_size[s] += 1;
    }
    let borgan = borgan_fit(
        &stop,
        &status,
        &covariates,
        &entry,
        &subcohort,
        &stratum,
        &cohort_sizes,
        method,
    )?;
    Ok(finish(
        borgan.computation,
        CchMetadata {
            method: method.r_name().to_string(),
            subcohort_size,
            cohort_size: cohort_sizes,
            stratum: Some(stratum),
            opt: Some(borgan.opt),
            delta: Some(borgan.delta),
            sc: Some(borgan.sc),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    type CchFixture = (Vec<f64>, Vec<i32>, Vec<Vec<f64>>, Vec<i32>, Vec<i64>);
    type CountingCchFixture = (
        Vec<f64>,
        Vec<f64>,
        Vec<i32>,
        Vec<Vec<f64>>,
        Vec<i32>,
        Vec<i64>,
    );

    fn fixture() -> CchFixture {
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 0, 1, 0, 1, 0, 1, 1],
            vec![
                vec![-0.8],
                vec![-0.2],
                vec![0.3],
                vec![0.9],
                vec![-0.5],
                vec![0.6],
                vec![1.2],
                vec![-1.0],
            ],
            vec![1, 1, 1, 1, 1, 1, 0, 0],
            (1..=8).collect(),
        )
    }

    fn r_parity_fixture() -> CountingCchFixture {
        let start = vec![
            0.0, 2.0, 1.0, 5.0, 4.0, 0.0, 10.0, 3.0, 12.0, 1.0, 5.0, 9.0, 0.0, 6.0, 2.0, 4.0, 7.0,
            2.0, 11.0, 13.0,
        ];
        let stop = vec![
            5.0, 12.0, 3.0, 18.0, 9.0, 1.0, 15.0, 7.0, 20.0, 4.0, 11.0, 16.0, 2.0, 14.0, 6.0, 10.0,
            13.0, 8.0, 17.0, 19.0,
        ];
        let status = vec![1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1];
        let x = [
            -1.2, 0.4, 0.9, -0.3, 1.4, -0.8, 0.2, 1.1, -0.5, 0.7, -1.0, 0.1, 1.7, -0.6, 0.5, -1.5,
            1.0, -0.1, 0.8, -0.9,
        ];
        let z = [
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
        ];
        let covariates = x
            .into_iter()
            .zip(z)
            .map(|(left, right)| vec![left, right])
            .collect();
        let subcohort = (0..20).map(|idx| i32::from(idx < 14)).collect();
        (
            start,
            stop,
            status,
            covariates,
            subcohort,
            (1..=20).collect(),
        )
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected:.17}, got {actual:.17}"
            );
        }
    }

    fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_close(actual_row, expected_row);
        }
    }

    #[test]
    fn unstratified_methods_fit_real_survival_times() {
        let (stop, status, covariates, subcohort, id) = fixture();
        for method in ["Prentice", "SelfPrentice", "LinYing"] {
            let result = cch_fit(
                stop.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                20,
                None,
                method,
                method == "LinYing",
            )
            .expect("case-cohort fit should succeed");
            assert_eq!(result.coefficients.len(), 1);
            assert!(result.coefficients[0].is_finite());
            assert!(result.var[0][0].is_finite());
            assert!(result.var[0][0] >= 0.0);
            assert_eq!(result.subcohort_size, vec![6]);
            assert_eq!(result.cohort_size, vec![20]);
            assert_eq!(result.method, method);
            assert!(!result.stratified);
        }
    }

    #[test]
    fn right_censored_results_match_r_survival() {
        let (_start, stop, status, covariates, subcohort, id) = r_parity_fixture();
        let expected = [
            (
                "Prentice",
                vec![-0.750_094_296_490_168, 0.832_850_534_909_300_8],
                vec![
                    vec![0.522_605_963_504_727_5, -0.202_211_481_196_043_4],
                    vec![-0.202_211_481_196_043_4, 1.276_498_083_276_546_7],
                ],
            ),
            (
                "SelfPrentice",
                vec![-0.763_491_690_039_069_1, 1.399_231_426_827_849],
                vec![
                    vec![0.522_605_963_398_621_8, -0.202_211_481_261_028_25],
                    vec![-0.202_211_481_261_028_25, 1.276_498_083_557_965],
                ],
            ),
            (
                "LinYing",
                vec![-1.351_125_060_104_277_7, 0.008_608_309_135_789_173],
                vec![
                    vec![0.350_099_414_059_607_9, 0.067_152_079_586_968_41],
                    vec![0.067_152_079_586_968_41, 0.631_459_031_713_116_3],
                ],
            ),
        ];
        for (method, expected_coefficients, expected_variance) in expected {
            let result = cch_fit(
                stop.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                80,
                None,
                method,
                method == "LinYing",
            )
            .expect("R parity fit should succeed");
            assert_close(&result.coefficients, &expected_coefficients);
            assert_matrix_close(&result.var, &expected_variance);
        }
    }

    #[test]
    fn counting_process_results_match_r_survival() {
        let (start, stop, status, covariates, subcohort, id) = r_parity_fixture();
        let expected = [
            (
                "Prentice",
                vec![-0.681_977_258_256_422_5, 0.629_799_094_589_367_8],
                vec![
                    vec![0.447_042_884_028_085_7, -0.395_965_103_654_998_26],
                    vec![-0.395_965_103_654_998_26, 1.771_738_748_947_702_6],
                ],
            ),
            (
                "SelfPrentice",
                vec![-0.787_643_185_183_800_9, 1.285_919_285_286_706],
                vec![
                    vec![0.447_042_884_333_092_25, -0.395_965_108_933_706_8],
                    vec![-0.395_965_108_933_706_8, 1.771_738_774_081_835_6],
                ],
            ),
            (
                "LinYing",
                vec![-1.166_298_764_457_855_3, -0.042_048_877_306_928_675],
                vec![
                    vec![0.191_775_299_232_715_6, -0.165_361_540_520_821_66],
                    vec![-0.165_361_540_520_821_66, 0.671_840_729_765_241_5],
                ],
            ),
        ];
        for (method, expected_coefficients, expected_variance) in expected {
            let result = cch_fit(
                stop.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                80,
                Some(start.clone()),
                method,
                method == "LinYing",
            )
            .expect("R parity fit should succeed");
            assert_close(&result.coefficients, &expected_coefficients);
            assert_matrix_close(&result.var, &expected_variance);
        }
    }

    #[test]
    fn stratified_borgan_results_match_r_survival() {
        let (start, stop, status, covariates, subcohort, id) = r_parity_fixture();
        let stratum = (0..stop.len()).map(|idx| idx % 2).collect::<Vec<_>>();
        let expected = [
            (
                "I.Borgan",
                vec![-0.763_491_690_039_068, 1.399_231_426_827_85],
                vec![
                    vec![0.532_806_143_623_19, -0.207_366_276_403_962],
                    vec![-0.207_366_276_403_962, 1.339_426_794_016_54],
                ],
                vec![
                    vec![0.697_261_201_213_853, 0.649_936_772_854_918],
                    vec![0.302_738_798_786_147, 0.350_063_227_145_082],
                ],
            ),
            (
                "II.Borgan",
                vec![-1.351_125_060_104_28, 0.008_608_309_135_789_29],
                vec![
                    vec![0.282_233_396_842_156, 0.001_531_832_828_161_97],
                    vec![0.001_531_832_828_161_97, 0.542_554_720_451_637],
                ],
                vec![
                    vec![0.524_014_328_275_41, 0.356_352_324_777_529],
                    vec![0.475_985_671_724_59, 0.643_647_675_222_471],
                ],
            ),
        ];
        for (method, expected_coefficients, expected_variance, expected_opt) in expected {
            let result = cch_borgan_fit(
                stop.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                stratum.clone(),
                vec![40, 40],
                None,
                method,
            )
            .expect("right-censored Borgan fit should succeed");
            assert_close(&result.coefficients, &expected_coefficients);
            assert_matrix_close(&result.var, &expected_variance);
            assert_matrix_close(
                result.opt.as_ref().expect("allocation fractions"),
                &expected_opt,
            );
            assert!(result.stratified);
            assert_eq!(result.cohort_size, vec![40, 40]);
            assert_eq!(result.subcohort_size, vec![10, 10]);
            assert_eq!(result.sc.as_ref().unwrap().len(), 20);
        }

        let expected = [
            (
                "I.Borgan",
                vec![-0.787_643_185_183_801, 1.285_919_285_286_71],
                vec![
                    vec![0.446_615_040_566_727, -0.379_274_338_369_315],
                    vec![-0.379_274_338_369_315, 1.792_883_479_688_41],
                ],
            ),
            (
                "II.Borgan",
                vec![-1.166_298_764_457_86, -0.042_048_877_306_928_8],
                vec![
                    vec![0.220_569_751_476_77, -0.100_625_898_088_698],
                    vec![-0.100_625_898_088_698, 0.619_445_707_012_408],
                ],
            ),
        ];
        for (method, expected_coefficients, expected_variance) in expected {
            let result = cch_borgan_fit(
                stop.clone(),
                status.clone(),
                covariates.clone(),
                subcohort.clone(),
                id.clone(),
                stratum.clone(),
                vec![40, 40],
                Some(start.clone()),
                method,
            )
            .expect("counting-process Borgan fit should succeed");
            assert_close(&result.coefficients, &expected_coefficients);
            assert_matrix_close(&result.var, &expected_variance);
        }
    }

    #[test]
    fn rejects_censored_rows_outside_subcohort() {
        let (stop, status, covariates, mut subcohort, id) = fixture();
        subcohort[1] = 0;
        let error = cch_fit(
            stop, status, covariates, subcohort, id, 20, None, "Prentice", false,
        )
        .expect_err("invalid sampling should fail");
        assert!(error.to_string().contains("censored observations"));
    }

    #[test]
    fn accepts_benchmark_style_case_cohort_sample() {
        let p = 4;
        for n in [100usize, 5000usize] {
            // Intentionally reuse the float-colliding tied generator so Prentice's
            // entry adjustment must keep a strict entry time at large n.
            let stop = (0..n)
                .map(|i| 1.0 + (i % 80) as f64 * 0.25 + (i / 80) as f64 * 0.01)
                .collect::<Vec<_>>();
            let mut status = (0..n)
                .map(|i| if i % 4 == 0 { 0 } else { 1 })
                .collect::<Vec<_>>();
            let subcohort = (0..n)
                .map(|idx| i32::from(idx % 5 != 0))
                .collect::<Vec<_>>();
            for idx in 0..n {
                if subcohort[idx] == 0 {
                    status[idx] = 1;
                }
            }
            let covariates = (0..n)
                .map(|i| {
                    (0..p)
                        .map(|j| {
                            let centered_i = (i % 17) as f64 - 8.0;
                            let centered_j = (j % 5) as f64 - 2.0;
                            centered_i * 0.03
                                + centered_j * 0.1
                                + ((i * (j + 3)) % 11) as f64 * 0.01
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let id = (0..n).map(|idx| idx as i64).collect::<Vec<_>>();
            for method in ["Prentice", "LinYing"] {
                cch_fit(
                    stop.clone(),
                    status.clone(),
                    covariates.clone(),
                    subcohort.clone(),
                    id.clone(),
                    n * 4,
                    None,
                    method,
                    method == "LinYing",
                )
                .unwrap_or_else(|err| {
                    panic!("{method} benchmark-style fit failed for n={n}: {err}")
                });
            }
        }
    }
}
