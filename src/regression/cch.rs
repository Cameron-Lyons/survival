use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN};
use crate::regression::coxph::{
    CoxPHFit, CoxPHModel, Subject, coxph_fit_with_counting_roundoff_compatibility,
};
use pyo3::prelude::*;
use std::collections::HashSet;

fn index_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyIndexError::new_err(message.into())
}

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeCchMethod {
    Prentice,
    SelfPrentice,
    LinYing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeBorganMethod {
    I,
    II,
}

impl NativeBorganMethod {
    fn parse(value: &str) -> PyResult<Self> {
        let normalized = value
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match normalized.as_str() {
            "iborgan" => Ok(Self::I),
            "iiborgan" => Ok(Self::II),
            _ => Err(value_error("method must be 'I.Borgan' or 'II.Borgan'")),
        }
    }

    fn as_r_name(self) -> &'static str {
        match self {
            Self::I => "I.Borgan",
            Self::II => "II.Borgan",
        }
    }
}

impl NativeCchMethod {
    fn parse(value: &str) -> PyResult<Self> {
        let normalized = value
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match normalized.as_str() {
            "prentice" => Ok(Self::Prentice),
            "selfprentice" => Ok(Self::SelfPrentice),
            "linying" => Ok(Self::LinYing),
            _ => Err(value_error(
                "method must be 'Prentice', 'SelfPrentice', or 'LinYing'",
            )),
        }
    }

    fn as_r_name(self) -> &'static str {
        match self {
            Self::Prentice => "Prentice",
            Self::SelfPrentice => "SelfPrentice",
            Self::LinYing => "LinYing",
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(from_py_object)]
pub enum CchMethod {
    Prentice,
    SelfPrentice,
    LinYing,
    IBorgan,
    IIBorgan,
}

#[pyclass]
pub struct CohortData {
    subjects: Vec<Subject>,
}

impl Default for CohortData {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl CohortData {
    #[staticmethod]
    pub fn new() -> CohortData {
        CohortData {
            subjects: Vec::new(),
        }
    }

    pub fn add_subject(&mut self, subject: Subject) {
        self.subjects.push(subject);
    }

    pub fn get_subject(&self, index: usize) -> PyResult<Subject> {
        self.subjects.get(index).cloned().ok_or_else(|| {
            index_error(format!(
                "subject index {index} out of range for cohort of size {}",
                self.subjects.len()
            ))
        })
    }

    pub fn __len__(&self) -> usize {
        self.subjects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.subjects.is_empty()
    }

    #[pyo3(signature = (method, max_iter=100))]
    pub fn fit(&self, method: CchMethod, max_iter: u16) -> PyResult<CoxPHModel> {
        if !matches!(method, CchMethod::Prentice) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "CchMethod::{method:?} is not implemented by the legacy synthetic-time API; use cch_fit with real survival times"
            )));
        }

        let mut model = CoxPHModel::new();
        for subject in &self.subjects {
            if subject.is_subcohort || subject.is_case {
                model.add_subject(subject)?;
            }
        }
        model.fit(max_iter)?;
        Ok(model)
    }
}

#[derive(Debug)]
#[pyclass(skip_from_py_object)]
pub struct CchFitResult {
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub naive_information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub model_information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub phase2_variance: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: Vec<f64>,
    #[pyo3(get)]
    pub score_vector: Vec<f64>,
    #[pyo3(get)]
    pub score_test: f64,
    #[pyo3(get)]
    pub convergence_flag: i32,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub risk_scores: Vec<f64>,
    #[pyo3(get)]
    pub event_times: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub entry_times: Option<Vec<f64>>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub offsets: Vec<f64>,
    #[pyo3(get)]
    pub residuals: Vec<f64>,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub nevent: usize,
    #[pyo3(get)]
    pub observed_n: usize,
    #[pyo3(get)]
    pub subcohort_size: usize,
    #[pyo3(get)]
    pub cohort_size: usize,
    #[pyo3(get)]
    pub robust: bool,
    #[pyo3(get)]
    pub stratified: bool,
    #[pyo3(get)]
    pub stratum: Option<Vec<usize>>,
    #[pyo3(get)]
    pub cohort_sizes: Vec<usize>,
    #[pyo3(get)]
    pub subcohort_sizes: Vec<usize>,
    #[pyo3(get)]
    pub optimization_fraction: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub phase2_score_matrix: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub collapsed_score_rows: Option<Vec<Vec<f64>>>,
    score_residual_rows: Vec<Vec<f64>>,
    dfbeta_rows: Vec<Vec<f64>>,
}

#[pymethods]
impl CchFitResult {
    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let coefficients = &self.coefficients[0];
        covariates
            .iter()
            .enumerate()
            .map(|(row_idx, row)| {
                if row.len() != coefficients.len() {
                    return Err(value_error(format!(
                        "covariates[{row_idx}] has {} columns but the fit expects {}",
                        row.len(),
                        coefficients.len()
                    )));
                }
                if row.iter().any(|value| !value.is_finite()) {
                    return Err(value_error(format!(
                        "covariates[{row_idx}] must contain only finite values"
                    )));
                }
                Ok(row
                    .iter()
                    .zip(coefficients)
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum())
            })
            .collect()
    }

    pub fn martingale_residuals(&self) -> Vec<f64> {
        self.residuals.clone()
    }

    pub fn deviance_residuals(&self) -> Vec<f64> {
        self.residuals
            .iter()
            .zip(&self.status)
            .map(|(&residual, &status)| {
                let status = f64::from(status);
                let log_term = if status > 0.0 {
                    status * (status - residual).max(f64::MIN_POSITIVE).ln()
                } else {
                    0.0
                };
                let magnitude = (-2.0 * (residual + log_term)).max(0.0).sqrt();
                if residual >= 0.0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect()
    }

    pub fn score_residuals(&self) -> Vec<Vec<f64>> {
        self.score_residual_rows.clone()
    }

    pub fn dfbeta(&self) -> Vec<Vec<f64>> {
        self.dfbeta_rows.clone()
    }
}

fn add_matrices(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    left.iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            left_row
                .iter()
                .zip(right_row)
                .map(|(left_value, right_value)| left_value + right_value)
                .collect()
        })
        .collect()
}

fn mirror_lower_triangle(matrix: &mut [Vec<f64>]) {
    for outer in 0..matrix.len() {
        let (prior_rows, current_and_later) = matrix.split_at_mut(outer);
        let current_row = &current_and_later[0];
        for (inner_row, &value) in prior_rows.iter_mut().zip(current_row) {
            inner_row[outer] = value;
        }
    }
}

fn scaled_crossproduct(rows: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    let width = rows.first().map_or(0, Vec::len);
    let mut result = vec![vec![0.0; width]; width];
    for row in rows {
        for outer in 0..width {
            for inner in 0..=outer {
                result[outer][inner] += scale * row[outer] * row[inner];
            }
        }
    }
    mirror_lower_triangle(&mut result);
    result
}

fn weighted_crossproduct(rows: &[Vec<f64>], divisors: &[f64]) -> Vec<Vec<f64>> {
    let width = rows.first().map_or(0, Vec::len);
    let mut result = vec![vec![0.0; width]; width];
    for (row, &divisor) in rows.iter().zip(divisors) {
        let scale = 1.0 / divisor;
        for outer in 0..width {
            for inner in 0..=outer {
                result[outer][inner] += scale * row[outer] * row[inner];
            }
        }
    }
    mirror_lower_triangle(&mut result);
    result
}

fn centered_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let width = rows[0].len();
    let mut means = vec![0.0; width];
    for row in rows {
        for (mean, &value) in means.iter_mut().zip(row) {
            *mean += value;
        }
    }
    for mean in &mut means {
        *mean /= rows.len() as f64;
    }
    rows.iter()
        .map(|row| {
            row.iter()
                .zip(&means)
                .map(|(&value, &mean)| value - mean)
                .collect()
        })
        .collect()
}

fn square_matrix_product(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let width = left.len();
    let mut result = vec![vec![0.0; width]; width];
    for (row_idx, left_row) in left.iter().enumerate() {
        for (shared_idx, &left_value) in left_row.iter().enumerate() {
            for (column_idx, result_value) in result[row_idx].iter_mut().enumerate() {
                *result_value += left_value * right[shared_idx][column_idx];
            }
        }
    }
    result
}

fn sandwich_product(variance: &[Vec<f64>], middle: &[Vec<f64>]) -> Vec<Vec<f64>> {
    square_matrix_product(&square_matrix_product(variance, middle), variance)
}

fn collapse_weighted_score_rows(
    rows: &[Vec<f64>],
    weights: &[f64],
    source_indices: &[usize],
    observed_n: usize,
) -> Vec<Vec<f64>> {
    let width = rows.first().map_or(0, Vec::len);
    let mut collapsed = vec![vec![0.0; width]; observed_n];
    for ((row, &weight), &source_idx) in rows.iter().zip(weights).zip(source_indices) {
        for (target, &value) in collapsed[source_idx].iter_mut().zip(row) {
            *target += weight * value;
        }
    }
    collapsed
}

fn validate_cch_inputs(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: Option<&[f64]>,
    subcohort: &[i32],
    id: &[i64],
    cohort_size: usize,
) -> PyResult<usize> {
    let n = stop.len();
    if n == 0 {
        return Err(value_error("stop must not be empty"));
    }
    for (name, len) in [
        ("status", status.len()),
        ("covariates", covariates.len()),
        ("subcohort", subcohort.len()),
        ("id", id.len()),
    ] {
        if len != n {
            return Err(value_error(format!(
                "{name} has {len} rows but stop has {n}"
            )));
        }
    }
    if let Some(values) = start
        && values.len() != n
    {
        return Err(value_error(format!(
            "start has {} rows but stop has {n}",
            values.len()
        )));
    }
    if cohort_size < n {
        return Err(value_error("number of records is greater than cohort_size"));
    }
    if status.iter().any(|&value| value != 0 && value != 1) {
        return Err(value_error("status must contain only 0/1 values"));
    }
    if subcohort.iter().any(|&value| value != 0 && value != 1) {
        return Err(value_error(
            "subcohort must contain only 0/1 or boolean values",
        ));
    }
    let outside_censored = status
        .iter()
        .zip(subcohort)
        .filter(|&(&event, &sampled)| event == 0 && sampled == 0)
        .count();
    if outside_censored > 0 {
        return Err(value_error(format!(
            "{outside_censored} censored observations are not in the subcohort"
        )));
    }
    let unique_ids = id.iter().copied().collect::<HashSet<_>>();
    if unique_ids.len() != n {
        return Err(value_error("multiple records per id are not allowed"));
    }
    if stop.iter().any(|value| !value.is_finite()) {
        return Err(value_error("stop must contain only finite values"));
    }
    if let Some(values) = start {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(value_error("start must contain only finite values"));
        }
        if values.iter().zip(stop).any(|(&entry, &exit)| entry >= exit) {
            return Err(value_error("every start value must be less than stop"));
        }
    } else if stop.iter().any(|&value| value <= 0.0) {
        return Err(value_error(
            "right-censored case-cohort times must be positive",
        ));
    }
    let width = covariates.first().map_or(0, Vec::len);
    if width == 0 {
        return Err(value_error("covariates must contain at least one column"));
    }
    if covariates.iter().any(|row| row.len() != width) {
        return Err(value_error("covariates must be rectangular"));
    }
    if covariates.iter().flatten().any(|value| !value.is_finite()) {
        return Err(value_error("covariates must contain only finite values"));
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
) -> PyResult<usize> {
    let cohort_size = cohort_sizes.iter().try_fold(0usize, |total, &size| {
        total
            .checked_add(size)
            .ok_or_else(|| value_error("cohort_sizes sum is too large"))
    })?;
    let width = validate_cch_inputs(stop, status, covariates, start, subcohort, id, cohort_size)?;
    if cohort_sizes.is_empty() {
        return Err(value_error("cohort_sizes must not be empty"));
    }
    if cohort_sizes.contains(&0) {
        return Err(value_error(
            "cohort_sizes must contain only positive values",
        ));
    }
    if stratum.len() != stop.len() {
        return Err(value_error(format!(
            "stratum has {} rows but stop has {}",
            stratum.len(),
            stop.len()
        )));
    }
    if stratum.iter().any(|&value| value >= cohort_sizes.len()) {
        return Err(value_error(
            "stratum codes must index every value in cohort_sizes",
        ));
    }
    let mut observed_sizes = vec![0usize; cohort_sizes.len()];
    for &code in stratum {
        observed_sizes[code] += 1;
    }
    if observed_sizes.contains(&0) {
        return Err(value_error(
            "each cohort_sizes entry must have an observed stratum",
        ));
    }
    if observed_sizes
        .iter()
        .zip(cohort_sizes)
        .any(|(&observed, &population)| observed > population)
    {
        return Err(value_error(
            "population is smaller than the sample in a stratum",
        ));
    }
    Ok(width)
}

fn event_time_delta(stop: &[f64], status: &[i32]) -> f64 {
    let mut times = stop
        .iter()
        .zip(status)
        .filter_map(|(&time, &event)| (event == 1).then_some(time))
        .collect::<Vec<_>>();
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

fn fit_cox(
    stop: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    start: Vec<f64>,
    offset: Vec<f64>,
    initial_beta: Option<Vec<f64>>,
    max_iter: usize,
) -> PyResult<CoxPHFit> {
    fit_weighted_cox(
        stop,
        status,
        covariates,
        start,
        offset,
        None,
        initial_beta,
        max_iter,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_weighted_cox(
    stop: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    start: Vec<f64>,
    offset: Vec<f64>,
    weights: Option<Vec<f64>>,
    initial_beta: Option<Vec<f64>>,
    max_iter: usize,
) -> PyResult<CoxPHFit> {
    coxph_fit_with_counting_roundoff_compatibility(
        stop,
        status,
        covariates,
        None,
        weights,
        Some(offset),
        initial_beta,
        Some(max_iter),
        None,
        None,
        Some("efron"),
        Some(start),
        Some(vec![-1.0, 0.0, 1.0]),
        None,
        None,
    )
}

struct CchComputation {
    fit: CoxPHFit,
    coefficients: Vec<f64>,
    phase2_variance: Vec<Vec<f64>>,
    naive_variance: Vec<Vec<f64>>,
    variance: Vec<Vec<f64>>,
    offsets: Vec<f64>,
    robust: bool,
}

fn two_pass_mean(values: &[f64]) -> f64 {
    debug_assert!(!values.is_empty());
    let count = values.len() as f64;
    let mut mean = values.iter().sum::<f64>() / count;
    if mean.is_finite() {
        mean += values.iter().map(|&value| value - mean).sum::<f64>() / count;
    }
    mean
}

fn augmented_fit(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: &[f64],
    subcohort: &[i32],
    cohort_size: usize,
    prentice: bool,
) -> PyResult<CchComputation> {
    let case_indices = status
        .iter()
        .enumerate()
        .filter_map(|(idx, &event)| (event == 1).then_some(idx))
        .collect::<Vec<_>>();
    let subcohort_indices = subcohort
        .iter()
        .enumerate()
        .filter_map(|(idx, &sampled)| (sampled == 1).then_some(idx))
        .collect::<Vec<_>>();

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
            stop.to_vec(),
            status.to_vec(),
            covariates.to_vec(),
            entry,
            vec![0.0; stop.len()],
            None,
            20,
        )?;
        Some(fit.coefficients[0].clone())
    } else {
        None
    };

    let augmented_n = case_indices.len() + subcohort_indices.len();
    let mut augmented_stop = Vec::with_capacity(augmented_n);
    let mut augmented_status = Vec::with_capacity(augmented_n);
    let mut augmented_covariates = Vec::with_capacity(augmented_n);
    let mut augmented_start = Vec::with_capacity(augmented_n);
    let mut offsets = Vec::with_capacity(augmented_n);
    for &idx in &case_indices {
        augmented_stop.push(stop[idx]);
        augmented_status.push(1);
        augmented_covariates.push(covariates[idx].clone());
        augmented_start.push(start[idx]);
        offsets.push(-100.0);
    }
    for &idx in &subcohort_indices {
        augmented_stop.push(stop[idx]);
        augmented_status.push(0);
        augmented_covariates.push(covariates[idx].clone());
        augmented_start.push(start[idx]);
        offsets.push(0.0);
    }

    let offset_mean = two_pass_mean(&offsets);
    let centered_offsets = offsets
        .iter()
        .map(|&offset| offset - offset_mean)
        .collect::<Vec<_>>();
    // The formula-level fit reports unweighted design means and leaves
    // indicator columns uncentered, independently of optimizer scaling.
    let reported_means = (0..augmented_covariates[0].len())
        .map(|column| {
            if augmented_covariates
                .iter()
                .all(|row| matches!(row[column], -1.0 | 0.0 | 1.0))
            {
                0.0
            } else {
                augmented_covariates
                    .iter()
                    .map(|row| row[column])
                    .sum::<f64>()
                    / augmented_n as f64
            }
        })
        .collect::<Vec<_>>();
    let mut fit = fit_cox(
        augmented_stop,
        augmented_status,
        augmented_covariates,
        augmented_start,
        centered_offsets.clone(),
        initial_coefficients.clone(),
        if prentice { 35 } else { 20 },
    )?;
    let covariate_center = fit.coefficients[0]
        .iter()
        .zip(&reported_means)
        .map(|(&coefficient, &mean)| coefficient * mean)
        .sum::<f64>();
    for (linear_predictor, risk_score) in fit
        .linear_predictors
        .iter_mut()
        .zip(fit.risk_scores.iter_mut())
    {
        *linear_predictor -= covariate_center;
        *linear_predictor += offset_mean;
        *risk_score = linear_predictor.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp();
    }
    fit.means = reported_means;
    let dfbeta = fit.dfbeta()?;
    let phase2_rows = dfbeta[case_indices.len()..].to_vec();
    let phase2_scale = 1.0 - subcohort_indices.len() as f64 / cohort_size as f64;
    let phase2_variance = scaled_crossproduct(&phase2_rows, phase2_scale);
    let model_variance = fit.information_matrix.clone();
    let naive_variance = add_matrices(&model_variance, &phase2_variance);
    let coefficients = initial_coefficients.unwrap_or_else(|| fit.coefficients[0].clone());

    Ok(CchComputation {
        fit,
        coefficients,
        phase2_variance,
        variance: naive_variance.clone(),
        naive_variance,
        offsets: centered_offsets,
        robust: false,
    })
}

fn lin_ying_fit(
    stop: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    start: &[f64],
    subcohort: &[i32],
    cohort_size: usize,
    robust: bool,
) -> PyResult<CchComputation> {
    let n_events = status.iter().filter(|&&event| event == 1).count();
    let subcohort_size = subcohort.iter().filter(|&&sampled| sampled == 1).count();
    let subcohort_events = status
        .iter()
        .zip(subcohort)
        .filter(|&(&event, &sampled)| event == 1 && sampled == 1)
        .count();
    let sampled_noncases = subcohort_size - subcohort_events;
    let cohort_noncases = cohort_size - n_events;
    if sampled_noncases == 0 || cohort_noncases == 0 {
        return Err(value_error(
            "LinYing requires at least one sampled noncase and one cohort noncase",
        ));
    }
    let sampling_inverse = cohort_noncases as f64 / sampled_noncases as f64;
    let offsets = status
        .iter()
        .map(|&event| {
            if event == 1 {
                0.0
            } else {
                sampling_inverse.ln()
            }
        })
        .collect::<Vec<_>>();
    let fit = fit_cox(
        stop.to_vec(),
        status.to_vec(),
        covariates.to_vec(),
        start.to_vec(),
        offsets.clone(),
        None,
        20,
    )?;
    let dfbeta = fit.dfbeta()?;
    let noncase_dfbeta = dfbeta
        .iter()
        .zip(status)
        .filter_map(|(row, &event)| (event == 0).then_some(row.clone()))
        .collect::<Vec<_>>();
    let phase2_scale = 1.0 - sampled_noncases as f64 / cohort_noncases as f64;
    let phase2_variance = scaled_crossproduct(&centered_rows(&noncase_dfbeta), phase2_scale);
    let model_variance = fit.information_matrix.clone();
    let naive_variance = add_matrices(&model_variance, &phase2_variance);
    let variance = if robust {
        let inverse_sampling = status
            .iter()
            .map(|&event| if event == 1 { 1.0 } else { sampling_inverse })
            .collect::<Vec<_>>();
        add_matrices(
            &weighted_crossproduct(&dfbeta, &inverse_sampling),
            &phase2_variance,
        )
    } else {
        naive_variance.clone()
    };
    let coefficients = fit.coefficients[0].clone();
    Ok(CchComputation {
        fit,
        coefficients,
        phase2_variance,
        naive_variance,
        variance,
        offsets,
        robust,
    })
}

struct BorganPhaseTwo {
    variance: Vec<Vec<f64>>,
    score_matrix: Vec<Vec<f64>>,
    optimization_fraction: Vec<Vec<f64>>,
}

fn borgan_phase_two(
    score_rows: &[Vec<f64>],
    row_strata: &[usize],
    sample_sizes: &[usize],
    population_sizes: &[usize],
    sampling_inverse: &[f64],
    model_variance: &[Vec<f64>],
) -> BorganPhaseTwo {
    let stratum_count = sample_sizes.len();
    let width = model_variance.len();
    let mut means = vec![vec![0.0; width]; stratum_count];
    for (row, &stratum_idx) in score_rows.iter().zip(row_strata) {
        for (mean, &value) in means[stratum_idx].iter_mut().zip(row) {
            *mean += value;
        }
    }
    for (mean, &sample_size) in means.iter_mut().zip(sample_sizes) {
        for value in mean {
            *value /= sample_size as f64;
        }
    }

    let mut crossproducts = vec![vec![vec![0.0; width]; width]; stratum_count];
    for (row, &stratum_idx) in score_rows.iter().zip(row_strata) {
        for outer in 0..width {
            let outer_value = row[outer] - means[stratum_idx][outer];
            for inner in 0..=outer {
                crossproducts[stratum_idx][outer][inner] +=
                    outer_value * (row[inner] - means[stratum_idx][inner]);
            }
        }
    }

    let mut score_matrix = vec![vec![0.0; width]; width];
    let mut optimization_fraction = vec![vec![0.0; width]; stratum_count];
    for stratum_idx in 0..stratum_count {
        mirror_lower_triangle(&mut crossproducts[stratum_idx]);
        let denominator = (sample_sizes[stratum_idx] - 1) as f64;
        for row in &mut crossproducts[stratum_idx] {
            for value in row {
                *value /= denominator;
            }
        }
        let delta_scale =
            (sampling_inverse[stratum_idx] - 1.0) * population_sizes[stratum_idx] as f64;
        for outer in 0..width {
            for inner in 0..width {
                score_matrix[outer][inner] +=
                    delta_scale * crossproducts[stratum_idx][outer][inner];
            }
        }
        let stratum_variance = sandwich_product(model_variance, &crossproducts[stratum_idx]);
        for column_idx in 0..width {
            optimization_fraction[stratum_idx][column_idx] = population_sizes[stratum_idx] as f64
                * stratum_variance[column_idx][column_idx].max(0.0).sqrt();
        }
    }
    for column_idx in 0..width {
        let total = optimization_fraction
            .iter()
            .map(|row| row[column_idx])
            .sum::<f64>();
        if total > 0.0 {
            for row in &mut optimization_fraction {
                row[column_idx] /= total;
            }
        }
    }

    BorganPhaseTwo {
        variance: sandwich_product(model_variance, &score_matrix),
        score_matrix,
        optimization_fraction,
    }
}

struct BorganComputation {
    computation: CchComputation,
    optimization_fraction: Vec<Vec<f64>>,
    phase2_score_matrix: Vec<Vec<f64>>,
    collapsed_score_rows: Vec<Vec<f64>>,
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
    method: NativeBorganMethod,
) -> PyResult<BorganComputation> {
    let observed_n = stop.len();
    let stratum_count = cohort_sizes.len();
    let mut event_counts = vec![0usize; stratum_count];
    let mut sampled_counts = vec![0usize; stratum_count];
    let mut sampled_noncase_counts = vec![0usize; stratum_count];
    for idx in 0..observed_n {
        let stratum_idx = stratum[idx];
        if status[idx] == 1 {
            event_counts[stratum_idx] += 1;
        }
        if subcohort[idx] == 1 {
            sampled_counts[stratum_idx] += 1;
            if status[idx] == 0 {
                sampled_noncase_counts[stratum_idx] += 1;
            }
        }
    }

    let (sample_sizes, population_sizes) = match method {
        NativeBorganMethod::I => (sampled_counts.clone(), cohort_sizes.to_vec()),
        NativeBorganMethod::II => {
            let noncase_population = cohort_sizes
                .iter()
                .zip(&event_counts)
                .map(|(&population, &events)| population.checked_sub(events))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| value_error("a stratum has more events than cohort members"))?;
            (sampled_noncase_counts.clone(), noncase_population)
        }
    };
    if sample_sizes.iter().any(|&size| size < 2) {
        return Err(value_error(
            "each Borgan sampling stratum requires at least two phase-two rows",
        ));
    }
    if sample_sizes
        .iter()
        .zip(&population_sizes)
        .any(|(&sample, &population)| sample > population)
    {
        return Err(value_error(
            "population is smaller than the sample in a stratum",
        ));
    }
    let sampling_inverse = population_sizes
        .iter()
        .zip(&sample_sizes)
        .map(|(&population, &sample)| population as f64 / sample as f64)
        .collect::<Vec<_>>();

    let mut source_indices = Vec::new();
    let mut fit_stop = Vec::new();
    let mut fit_status = Vec::new();
    let mut fit_covariates = Vec::new();
    let mut fit_start = Vec::new();
    let mut offsets = Vec::new();
    let mut weights = Vec::new();
    let mut phase2_start = 0usize;
    match method {
        NativeBorganMethod::I => {
            let case_indices = status
                .iter()
                .enumerate()
                .filter_map(|(idx, &event)| (event == 1).then_some(idx))
                .collect::<Vec<_>>();
            let subcohort_indices = subcohort
                .iter()
                .enumerate()
                .filter_map(|(idx, &sampled)| (sampled == 1).then_some(idx))
                .collect::<Vec<_>>();
            phase2_start = case_indices.len();
            for idx in case_indices {
                source_indices.push(idx);
                fit_stop.push(stop[idx]);
                fit_status.push(1);
                fit_covariates.push(covariates[idx].clone());
                fit_start.push(start[idx]);
                offsets.push(-100.0);
                weights.push(1.0);
            }
            for idx in subcohort_indices {
                source_indices.push(idx);
                fit_stop.push(stop[idx]);
                fit_status.push(0);
                fit_covariates.push(covariates[idx].clone());
                fit_start.push(start[idx]);
                offsets.push(0.0);
                weights.push(sampling_inverse[stratum[idx]]);
            }
        }
        NativeBorganMethod::II => {
            source_indices.extend(0..observed_n);
            fit_stop.extend_from_slice(stop);
            fit_status.extend_from_slice(status);
            fit_covariates.extend_from_slice(covariates);
            fit_start.extend_from_slice(start);
            offsets.resize(observed_n, 0.0);
            weights.extend((0..observed_n).map(|idx| {
                if status[idx] == 1 {
                    1.0
                } else {
                    sampling_inverse[stratum[idx]]
                }
            }));
        }
    }

    let fit = fit_weighted_cox(
        fit_stop,
        fit_status,
        fit_covariates,
        fit_start,
        offsets.clone(),
        Some(weights),
        None,
        25,
    )?;
    let score_rows = fit.score_residuals()?;
    let phase2_source_indices = match method {
        NativeBorganMethod::I => (phase2_start..score_rows.len()).collect::<Vec<_>>(),
        NativeBorganMethod::II => status
            .iter()
            .enumerate()
            .filter_map(|(idx, &event)| (event == 0).then_some(idx))
            .collect(),
    };
    let phase2_rows = phase2_source_indices
        .iter()
        .map(|&idx| score_rows[idx].clone())
        .collect::<Vec<_>>();
    let phase2_strata = phase2_source_indices
        .iter()
        .map(|&idx| stratum[source_indices[idx]])
        .collect::<Vec<_>>();
    let model_variance = fit.information_matrix.clone();
    let phase_two = borgan_phase_two(
        &phase2_rows,
        &phase2_strata,
        &sample_sizes,
        &population_sizes,
        &sampling_inverse,
        &model_variance,
    );
    let naive_variance = add_matrices(&model_variance, &phase_two.variance);
    let collapsed_score_rows =
        collapse_weighted_score_rows(&score_rows, &fit.weights, &source_indices, observed_n);
    let coefficients = fit.coefficients[0].clone();
    Ok(BorganComputation {
        computation: CchComputation {
            fit,
            coefficients,
            phase2_variance: phase_two.variance,
            variance: naive_variance.clone(),
            naive_variance,
            offsets,
            robust: false,
        },
        optimization_fraction: phase_two.optimization_fraction,
        phase2_score_matrix: phase_two.score_matrix,
        collapsed_score_rows,
    })
}

struct CchResultMetadata {
    method: String,
    nevent: usize,
    observed_n: usize,
    subcohort_sizes: Vec<usize>,
    cohort_sizes: Vec<usize>,
    stratum: Option<Vec<usize>>,
    optimization_fraction: Option<Vec<Vec<f64>>>,
    phase2_score_matrix: Option<Vec<Vec<f64>>>,
    collapsed_score_rows: Option<Vec<Vec<f64>>>,
}

fn finish_cch_result(
    computation: CchComputation,
    metadata: CchResultMetadata,
) -> PyResult<CchFitResult> {
    let score_residual_rows = computation.fit.score_residuals()?;
    let dfbeta_rows = computation.fit.dfbeta()?;
    let residuals = computation.fit.martingale_residuals()?;
    let CchComputation {
        fit,
        coefficients,
        phase2_variance,
        naive_variance,
        variance,
        offsets,
        robust,
    } = computation;
    let n = fit.event_times.len();
    let cohort_size = metadata.cohort_sizes.iter().sum();
    let subcohort_size = metadata.subcohort_sizes.iter().sum();
    let stratified = metadata.stratum.is_some();

    Ok(CchFitResult {
        coefficients: vec![coefficients],
        information_matrix: variance,
        naive_information_matrix: naive_variance,
        model_information_matrix: fit.information_matrix,
        phase2_variance,
        log_likelihood: fit.log_likelihood,
        score_vector: fit.score_vector,
        score_test: fit.score_test,
        convergence_flag: fit.convergence_flag,
        iterations: fit.iterations,
        risk_scores: fit.risk_scores,
        event_times: fit.event_times,
        status: fit.status,
        linear_predictors: fit.linear_predictors,
        entry_times: fit.entry_times,
        weights: fit.weights,
        covariates: fit.covariates,
        means: fit.means,
        offsets,
        residuals,
        method: metadata.method,
        n,
        nevent: metadata.nevent,
        observed_n: metadata.observed_n,
        subcohort_size,
        cohort_size,
        robust,
        stratified,
        stratum: metadata.stratum,
        cohort_sizes: metadata.cohort_sizes,
        subcohort_sizes: metadata.subcohort_sizes,
        optimization_fraction: metadata.optimization_fraction,
        phase2_score_matrix: metadata.phase2_score_matrix,
        collapsed_score_rows: metadata.collapsed_score_rows,
        score_residual_rows,
        dfbeta_rows,
    })
}

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
    let method = NativeCchMethod::parse(method)?;
    let entry = start.clone().unwrap_or_else(|| vec![0.0; stop.len()]);
    let observed_n = stop.len();
    let nevent = status.iter().filter(|&&event| event == 1).count();
    let subcohort_size = subcohort.iter().filter(|&&sampled| sampled == 1).count();
    let computation = match method {
        NativeCchMethod::Prentice => augmented_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            true,
        )?,
        NativeCchMethod::SelfPrentice => augmented_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            false,
        )?,
        NativeCchMethod::LinYing => lin_ying_fit(
            &stop,
            &status,
            &covariates,
            &entry,
            &subcohort,
            cohort_size,
            robust,
        )?,
    };
    finish_cch_result(
        computation,
        CchResultMetadata {
            method: method.as_r_name().to_string(),
            nevent,
            observed_n,
            subcohort_sizes: vec![subcohort_size],
            cohort_sizes: vec![cohort_size],
            stratum: None,
            optimization_fraction: None,
            phase2_score_matrix: None,
            collapsed_score_rows: None,
        },
    )
}

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
    let method = NativeBorganMethod::parse(method)?;
    let entry = start.unwrap_or_else(|| vec![0.0; stop.len()]);
    let observed_n = stop.len();
    let nevent = status.iter().filter(|&&event| event == 1).count();
    let mut subcohort_sizes = vec![0usize; cohort_sizes.len()];
    for (&sampled, &stratum_idx) in subcohort.iter().zip(&stratum) {
        subcohort_sizes[stratum_idx] += usize::from(sampled == 1);
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
    finish_cch_result(
        borgan.computation,
        CchResultMetadata {
            method: method.as_r_name().to_string(),
            nevent,
            observed_n,
            subcohort_sizes,
            cohort_sizes,
            stratum: Some(stratum),
            optimization_fraction: Some(borgan.optimization_fraction),
            phase2_score_matrix: Some(borgan.phase2_score_matrix),
            collapsed_score_rows: Some(borgan.collapsed_score_rows),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::common::initialize_python;

    type CchFixture = (Vec<f64>, Vec<i32>, Vec<Vec<f64>>, Vec<i32>, Vec<i64>);
    type CountingCchFixture = (
        Vec<f64>,
        Vec<f64>,
        Vec<i32>,
        Vec<Vec<f64>>,
        Vec<i32>,
        Vec<i64>,
    );

    fn subject(id: usize) -> Subject {
        Subject::new(id, vec![id as f64], true, true, 0)
    }

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
                (actual - expected).abs() < 1e-11,
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
    fn cohort_data_len_and_get_subject_are_safe() {
        let mut cohort = CohortData::new();
        assert_eq!(cohort.__len__(), 0);
        assert!(cohort.is_empty());

        cohort.add_subject(subject(7));
        assert_eq!(cohort.__len__(), 1);
        assert!(!cohort.is_empty());
        assert_eq!(
            cohort
                .get_subject(0)
                .expect("subject at index 0 should exist")
                .id,
            7
        );
        assert!(cohort.get_subject(1).is_err());
    }

    #[test]
    fn native_unstratified_methods_fit_real_survival_times() {
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
            assert_eq!(result.coefficients[0].len(), 1);
            assert!(result.coefficients[0][0].is_finite());
            assert!(result.information_matrix[0][0].is_finite());
            assert!(result.information_matrix[0][0] >= 0.0);
            assert_eq!(result.observed_n, stop.len());
            assert_eq!(result.subcohort_size, 6);
            assert_eq!(result.cohort_size, 20);
        }
    }

    #[test]
    fn native_right_censored_results_match_r_survival() {
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
            assert_close(&result.coefficients[0], &expected_coefficients);
            assert_matrix_close(&result.information_matrix, &expected_variance);
        }
    }

    #[test]
    fn native_counting_process_results_match_r_survival() {
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
            assert_close(&result.coefficients[0], &expected_coefficients);
            assert_matrix_close(&result.information_matrix, &expected_variance);
        }
    }

    #[test]
    fn native_counting_process_prentice_preserves_small_offset_risk() {
        let stop = vec![
            3.0, 13.0, 11.0, 15.0, 5.0, 16.0, 9.0, 1.0, 12.0, 14.0, 19.0, 16.0, 9.0, 3.0, 15.0,
            18.0, 7.0, 9.0, 10.0, 15.0, 10.0,
        ];
        let status = vec![
            0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        ];
        let x = [
            -2.366_019_802_897_81,
            -0.939_726_558_703_197,
            0.672_805_414_806_265,
            -0.476_183_125_795_317,
            -0.636_546_918_038_443,
            -0.687_008_929_997_081,
            0.535_019_844_914_994,
            -0.210_862_903_347_529,
            0.705_276_653_609_758,
            -0.678_855_799_561_129,
            -0.832_078_189_498_332,
            -0.956_832_544_488_333,
            -0.230_958_721_600_656,
            -0.542_591_235_128_462,
            -1.206_226_329_830_76,
            1.486_831_793_410_71,
            1.289_638_472_695_21,
            0.271_588_841_450_844,
            -1.635_910_435_825_59,
            -0.831_208_786_158_255,
            -0.890_202_805_534_755,
        ];
        let covariates = x.into_iter().map(|value| vec![value]).collect();
        let subcohort = vec![
            1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0,
        ];
        let start = vec![
            1.0, 8.0, 6.0, 13.0, 3.0, 14.0, 4.0, 0.0, 10.0, 12.0, 16.0, 11.0, 4.0, 1.0, 13.0, 14.0,
            6.0, 8.0, 9.0, 13.0, 4.0,
        ];
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort,
            (1..=21).collect(),
            63,
            Some(start),
            "Prentice",
            false,
        )
        .expect("counting-process Prentice fit should succeed");

        assert_close(&result.coefficients[0], &[0.291_262_177_689_472]);
        assert_matrix_close(
            &result.model_information_matrix,
            &[vec![0.193_853_550_610_786]],
        );
        assert_matrix_close(&result.phase2_variance, &[vec![0.150_259_455_416_288]]);
        assert_matrix_close(&result.information_matrix, &[vec![0.344_113_006_027_074]]);
        assert_close(
            &result.log_likelihood,
            &[-1_207.817_811_028_63, -1_207.811_906_389_63],
        );
        assert!((result.score_test - 0.011_850_593_966_545_8).abs() < 1e-11);
        assert_eq!(result.iterations, 2);
    }

    #[test]
    fn native_counting_process_prentice_matches_factor_phase_two_roundoff() {
        let stop = vec![
            7.0, 16.0, 7.0, 1.0, 3.0, 1.0, 19.0, 1.0, 4.0, 13.0, 16.0, 2.0, 15.0, 4.0, 6.0, 2.0,
            2.0, 2.0, 8.0, 15.0, 19.0, 12.0, 11.0, 18.0, 6.0, 17.0,
        ];
        let status = vec![
            1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
        ];
        let groups = [
            1, 2, 2, 2, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 3, 1, 2, 3, 1, 2, 3, 1, 1, 3, 3, 1,
        ];
        let covariates = groups
            .into_iter()
            .map(|group| vec![f64::from(group == 2), f64::from(group == 3)])
            .collect();
        let subcohort = vec![
            0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
        ];
        let start = vec![
            1.0, 11.0, 4.0, 0.0, 2.0, 0.0, 13.0, 0.0, 0.0, 12.0, 15.0, 0.0, 14.0, 0.0, 3.0, 0.0,
            1.0, 0.0, 2.0, 12.0, 17.0, 9.0, 8.0, 15.0, 4.0, 11.0,
        ];
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort,
            (1..=26).collect(),
            78,
            Some(start),
            "Prentice",
            false,
        )
        .expect("factor Prentice fit should succeed");

        assert_close(
            &result.coefficients[0],
            &[0.079_030_942_884_838_3, -0.661_618_805_430_928],
        );
        assert_matrix_close(
            &result.model_information_matrix,
            &[
                vec![0.536_078_739_349_966, 0.234_044_458_790_259],
                vec![0.234_044_458_790_259, 0.817_122_606_277_822],
            ],
        );
        let expected_phase_two = [
            [4.554_908_571_063_97e54, 5.384_366_515_507_59e53],
            [5.384_366_515_507_59e53, 1.694_513_462_088_3e53],
        ];
        for (actual_row, expected_row) in result.phase2_variance.iter().zip(expected_phase_two) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual / expected - 1.0).abs() < 1e-11,
                    "expected {expected:.17e}, got {actual:.17e}"
                );
            }
        }
        assert_close(
            &result.log_likelihood,
            &[-1_710.608_339_894_93, -1_710.187_760_623_87],
        );
        assert!((result.score_test - 0.846_482_519_571_698).abs() < 1e-11);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn native_self_prentice_matches_right_censored_phase_two_roundoff() {
        let stop = vec![
            9.0, 2.0, 11.0, 17.0, 13.0, 16.0, 9.0, 17.0, 2.0, 14.0, 4.0, 5.0, 9.0, 3.0, 5.0, 14.0,
            13.0, 11.0, 12.0, 13.0, 19.0, 5.0, 17.0, 7.0, 11.0, 9.0, 17.0, 13.0, 19.0, 2.0, 7.0,
            17.0, 11.0, 6.0, 17.0, 20.0,
        ];
        let status = vec![
            1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 0, 1, 0, 0, 1,
        ];
        let x = vec![
            -0.268_496_659_511_314_63,
            0.335_513_465_200_485_75,
            1.543_538_272_789_444,
            0.736_611_278_363_456_6,
            -1.293_128_439_900_177_5,
            0.200_945_683_174_392_68,
            1.008_253_155_630_741_5,
            0.366_515_487_283_751_7,
            -0.905_546_023_312_398_8,
            -0.668_610_076_688_068_2,
            -1.450_888_541_502_198_8,
            1.112_470_096_940_035_8,
            0.206_911_965_429_309_5,
            0.767_274_338_629_228_6,
            0.403_762_541_077_289_37,
            -2.084_153_589_625_603_6,
            -1.641_738_523_322_735_3,
            0.752_711_012_030_994,
            1.924_748_362_203_215_3,
            0.515_767_051_110_402_4,
            -0.286_366_660_299_187_4,
            0.945_700_584_816_985_9,
            -0.591_227_269_430_436_1,
            1.169_159_231_896_013_8,
            1.029_779_744_428_501_7,
            0.220_287_103_277_879_04,
            -0.223_255_527_834_578_4,
            -0.106_245_341_765_360_09,
            1.087_940_890_808_079_4,
            -2.451_962_606_343_388,
            -1.031_058_975_566_138,
            0.388_790_816_358_851_04,
            -0.605_095_620_571_917_8,
            0.494_756_754_522_265_14,
            0.198_096_233_886_051_08,
            0.152_754_316_498_647_65,
        ];
        let subcohort = vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 0,
        ];
        let result = cch_fit(
            stop,
            status,
            x.into_iter().map(|value| vec![value]).collect(),
            subcohort,
            (1..=36).collect(),
            108,
            None,
            "SelfPrentice",
            false,
        )
        .expect("SelfPrentice fit should succeed");

        assert_close(&result.coefficients[0], &[-0.831_949_191_237_055_9]);
        assert!(
            (result.phase2_variance[0][0] / 4.218_812_001_681_335e53 - 1.0).abs() < 1e-11,
            "expected reference phase-two variance, got {:.17e}",
            result.phase2_variance[0][0]
        );
        assert!((result.means[0] - 0.113_479_457_821_405_05).abs() < 1e-15);
        assert_close(
            &result.log_likelihood,
            &[-2_565.281_673_644_15, -2_560.966_586_506_64],
        );
        assert!((result.score_test - 8.884_185_983_358_03).abs() < 1e-11);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn native_self_prentice_matches_two_covariate_phase_two_roundoff() {
        let stop = vec![
            19.0, 8.0, 8.0, 8.0, 7.0, 13.0, 13.0, 2.0, 20.0, 14.0, 4.0, 8.0, 19.0, 5.0, 18.0, 6.0,
            15.0, 9.0, 1.0, 12.0, 12.0, 18.0, 2.0, 20.0, 15.0, 10.0, 6.0, 15.0, 1.0, 13.0, 6.0,
            11.0, 1.0,
        ];
        let status = vec![
            0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
            1, 1, 1, 1,
        ];
        let x = [
            -1.018_810_816_057_810_2,
            -0.060_063_439_972_764_65,
            0.075_588_774_970_652_22,
            0.895_060_839_637_629_3,
            0.191_100_219_778_663_68,
            -0.206_239_320_623_294_4,
            -1.674_124_020_681_451,
            -1.407_197_205_015_605_9,
            0.523_007_846_505_987_3,
            0.425_929_482_546_762_2,
            -0.697_740_417_759_125_7,
            1.059_327_848_858_200_5,
            -1.965_102_054_428_597_5,
            1.525_816_958_024_978,
            -0.475_422_350_543_277_64,
            0.991_666_353_494_522_3,
            0.262_242_050_913_803_1,
            0.479_444_569_078_616_66,
            -1.471_357_192_336_171_5,
            -0.274_388_189_768_790_15,
            0.339_838_022_448_828_27,
            -1.823_889_130_374_678_7,
            0.383_325_985_019_290_44,
            -0.462_364_882_288_566_87,
            0.766_241_564_317_813_3,
            0.323_476_797_723_082_1,
            -0.649_392_766_559_504_4,
            0.961_867_347_018_554_4,
            -1.247_693_827_350_25,
            -1.360_181_660_257_599,
            1.194_023_396_986_788_4,
            1.331_451_562_678_206_9,
            -1.801_424_064_123_098_3,
        ];
        let z = [
            1.815_829_850_791_582_4,
            -0.811_701_727_992_517_2,
            -0.243_778_672_694_800_1,
            -0.840_255_544_808_328_9,
            0.466_994_239_137_976_4,
            -0.975_564_366_102_540_2,
            -1.078_449_993_730_198_6,
            -1.060_104_757_726_408_9,
            -0.431_923_532_830_114_14,
            -0.679_131_876_481_877_3,
            -0.328_913_099_602_879_7,
            0.665_434_712_070_193_8,
            -0.051_742_381_155_122_856,
            -1.371_400_790_442_056,
            0.092_725_713_344_919_97,
            -2.145_492_149_155_372_5,
            -0.346_026_428_766_257_94,
            0.569_751_278_633_426_8,
            1.425_947_117_453_357,
            1.258_477_415_880_124_6,
            -1.384_031_512_663_769_4,
            0.868_933_515_000_480_4,
            0.482_959_727_315_855_3,
            2.001_407_933_595_794,
            0.804_847_329_002_513_8,
            1.154_806_444_507_626_7,
            -0.986_764_657_864_526_2,
            1.366_544_733_896_638_6,
            -0.398_833_678_682_121_9,
            -1.056_183_440_773_958_3,
            -0.825_031_105_029_923_4,
            0.042_151_325_226_441_716,
            -0.230_116_164_739_260_3,
        ];
        let covariates = x
            .into_iter()
            .zip(z)
            .map(|(first, second)| vec![first, second])
            .collect();
        let subcohort = vec![
            1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0,
            0, 1, 1, 1,
        ];
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort,
            (1..=33).collect(),
            99,
            None,
            "SelfPrentice",
            false,
        )
        .expect("two-covariate SelfPrentice fit should succeed");

        assert_close(
            &result.coefficients[0],
            &[-0.437_468_131_611_164_7, -0.549_041_700_616_958_8],
        );
        let expected_phase_two = [
            [2.929_358_030_607_271_5e54, 1.007_943_384_337_772_2e54],
            [1.007_943_384_337_772_2e54, 2.324_626_112_202_456_6e54],
        ];
        for (actual_row, expected_row) in result.phase2_variance.iter().zip(expected_phase_two) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual / expected - 1.0).abs() < 1e-11,
                    "expected {expected:.17e}, got {actual:.17e}"
                );
            }
        }
        assert_close(
            &result.log_likelihood,
            &[-1_436.230_305_670_821, -1_433.658_049_282_212_5],
        );
        assert!((result.score_test - 4.956_125_752_130_163).abs() < 1e-11);
        assert_eq!(result.iterations, 4);
    }

    #[test]
    fn native_self_prentice_matches_scalar_counting_offset_mean_roundoff() {
        let stop = vec![
            12.0, 13.0, 6.0, 3.0, 14.0, 17.0, 6.0, 13.0, 19.0, 4.0, 6.0, 5.0, 4.0, 13.0, 7.0, 18.0,
            20.0, 12.0, 16.0, 11.0, 10.0, 12.0, 7.0, 10.0, 19.0, 14.0, 11.0,
        ];
        let status = vec![
            1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1,
        ];
        let start = vec![
            9.0, 10.0, 5.0, 1.0, 8.0, 16.0, 3.0, 10.0, 13.0, 0.0, 0.0, 3.0, 0.0, 9.0, 5.0, 13.0,
            16.0, 11.0, 10.0, 10.0, 7.0, 10.0, 4.0, 5.0, 18.0, 11.0, 5.0,
        ];
        let x = [
            8.926_919_521_119_415e-3,
            9.060_546_400_253_241e-1,
            1.832_220_688_126_756_7,
            -5.487_920_544_142_219e-1,
            -3.853_815_727_983_842_5e-1,
            -2.742_385_422_072_713_4e-1,
            -1.257_580_006_066_175_4e-1,
            2.825_336_789_054_21e-2,
            -1.423_320_756_314_914_7e-1,
            -1.948_042_508_389_166_2e-1,
            -1.273_056_905_241_701,
            -3.860_241_881_622_384e-1,
            -2.341_092_420_046_039,
            -2.414_972_829_893_251e-1,
            -4.866_218_287_735_713_5e-1,
            5.613_828_624_512_708e-1,
            -5.624_384_123_439_998e-1,
            -1.068_133_742_345_954_7,
            3.188_303_397_678_311_5e-1,
            -8.352_213_781_406_432e-1,
            -9.302_242_429_820_97e-3,
            -5.042_953_637_548_548e-1,
            -6.629_703_130_264_287e-1,
            -1.018_832_259_782_403_3,
            -4.171_629_787_202_551_5e-1,
            2.874_493_651_453_408_8e-2,
            8.739_803_580_994_467e-1,
        ];
        let covariates: Vec<Vec<f64>> = x.into_iter().map(|value| vec![value]).collect();
        let subcohort = vec![
            1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
        ];
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort,
            (1..=27).collect(),
            81,
            Some(start),
            "SelfPrentice",
            false,
        )
        .expect("scalar counting-process SelfPrentice fit should succeed");

        assert_close(&result.coefficients[0], &[0.977_201_773_860_779_8]);
        assert_close(
            &result.model_information_matrix[0],
            &[0.416_249_693_737_983_7],
        );
        assert!(
            (result.phase2_variance[0][0] / 6.353_402_100_415_902e54 - 1.0).abs() < 1e-11,
            "expected scalar reference phase-two variance, got {:.17e}",
            result.phase2_variance[0][0]
        );
        assert_close(
            &result.log_likelihood,
            &[-1_314.816_730_500_623_2, -1_313.498_082_563_651_5],
        );
        assert!((result.score_test - 2.634_447_115_612_323_5).abs() < 1e-11);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn native_prentice_matches_factor_counting_phase_two_roundoff() {
        let stop = vec![
            9.0, 1.0, 11.0, 20.0, 9.0, 1.0, 11.0, 9.0, 7.0, 14.0, 8.0, 2.0, 18.0, 9.0, 4.0, 16.0,
            10.0, 17.0, 16.0, 14.0, 4.0, 15.0, 1.0, 5.0, 7.0,
        ];
        let status = vec![
            0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
        ];
        let start = vec![
            5.0, 0.0, 6.0, 17.0, 7.0, 0.0, 9.0, 6.0, 4.0, 10.0, 5.0, 0.0, 12.0, 4.0, 3.0, 14.0,
            9.0, 15.0, 15.0, 13.0, 0.0, 14.0, 0.0, 2.0, 2.0,
        ];
        let groups = [
            'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'b', 'a', 'b', 'b', 'b',
            'a', 'a', 'c', 'b', 'a', 'b', 'a', 'a',
        ];
        let subcohort = [
            1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        ];
        let covariates = groups
            .iter()
            .map(|&group| vec![f64::from(group == 'b'), f64::from(group == 'c')])
            .collect::<Vec<_>>();
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort.to_vec(),
            (1..=25).collect(),
            75,
            Some(start),
            "Prentice",
            false,
        )
        .expect("factor counting-process Prentice fit should succeed");

        assert_close(
            &result.coefficients[0],
            &[-0.197_669_724_620_924_2, 0.170_940_779_052_008_94],
        );
        let expected_model_variance = [
            [0.356_230_635_545_144_2, 0.329_963_928_455_902_56],
            [0.329_963_928_455_902_56, 1.840_590_246_899_576_5],
        ];
        for (actual_row, expected_row) in result
            .model_information_matrix
            .iter()
            .zip(expected_model_variance)
        {
            assert_close(actual_row, &expected_row);
        }
        let expected_phase_two = [
            [1.119_153_148_549_493_4e51, -6.436_369_031_047_727e52],
            [-6.436_369_031_047_727e52, 3.701_624_425_354_347e54],
        ];
        for (actual_row, expected_row) in result.phase2_variance.iter().zip(expected_phase_two) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual / expected - 1.0).abs() < 1e-11,
                    "expected {expected:.17e}, got {actual:.17e}"
                );
            }
        }
        assert_close(
            &result.log_likelihood,
            &[-1_518.566_995_987_751_2, -1_518.546_609_073_264],
        );
        assert!((result.score_test - 0.040_572_557_274_083_286).abs() < 1e-11);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn native_prentice_matches_delayed_entry_factor_roundoff() {
        let stop = vec![
            12.0, 3.0, 2.0, 8.0, 2.0, 17.0, 10.0, 6.0, 3.0, 3.0, 8.0, 5.0, 8.0, 11.0, 1.0, 8.0,
            11.0, 5.0, 9.0, 1.0,
        ];
        let status = vec![0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let groups = [1, 1, 2, 2, 3, 1, 3, 1, 1, 2, 3, 2, 3, 2, 3, 1, 2, 2, 2, 2];
        let covariates = groups
            .into_iter()
            .map(|group| vec![f64::from(group == 2), f64::from(group == 3)])
            .collect();
        let subcohort = vec![1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1];
        let start = vec![
            6.0, 0.0, 0.0, 5.0, 0.0, 14.0, 5.0, 1.0, 2.0, 2.0, 7.0, 1.0, 3.0, 7.0, 0.0, 5.0, 9.0,
            0.0, 4.0, 0.0,
        ];
        let result = cch_fit(
            stop,
            status,
            covariates,
            subcohort,
            (1..=20).collect(),
            60,
            Some(start),
            "Prentice",
            false,
        )
        .expect("delayed-entry Prentice fit should succeed");

        assert_close(
            &result.coefficients[0],
            &[1.090_919_321_048_074_5, 1.932_782_521_394_574],
        );
        let expected_phase_two = [
            [7.186_983_559_093_221e55, 1.955_659_165_571_378_2e56],
            [1.955_659_165_571_378_2e56, 5.902_264_325_547_351e56],
        ];
        for (actual_row, expected_row) in result.phase2_variance.iter().zip(expected_phase_two) {
            for (&actual, expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual / expected - 1.0).abs() < 1e-11,
                    "expected {expected:.17e}, got {actual:.17e}"
                );
            }
        }
        assert_close(
            &result.log_likelihood,
            &[-1_620.065_518_335, -1_618.895_358_472_12],
        );
        assert!((result.score_test - 2.109_158_956_458_69).abs() < 1e-11);
        assert_eq!(result.iterations, 4);
    }

    #[test]
    fn native_stratified_borgan_results_match_r_survival() {
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
            assert_close(&result.coefficients[0], &expected_coefficients);
            assert_matrix_close(&result.information_matrix, &expected_variance);
            assert_matrix_close(
                result
                    .optimization_fraction
                    .as_ref()
                    .expect("Borgan fit should report allocation fractions"),
                &expected_opt,
            );
            assert!(result.stratified);
            assert_eq!(result.cohort_sizes, vec![40, 40]);
            assert_eq!(result.subcohort_sizes, vec![7, 7]);
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
            assert_close(&result.coefficients[0], &expected_coefficients);
            assert_matrix_close(&result.information_matrix, &expected_variance);
        }
    }

    #[test]
    fn native_fit_rejects_censored_rows_outside_subcohort() {
        initialize_python();
        let (stop, status, covariates, mut subcohort, id) = fixture();
        subcohort[1] = 0;
        let error = cch_fit(
            stop, status, covariates, subcohort, id, 20, None, "Prentice", false,
        )
        .expect_err("invalid sampling should fail");
        assert!(error.to_string().contains("censored observations"));
    }

    #[test]
    fn native_fit_accepts_benchmark_style_case_cohort_sample() {
        initialize_python();
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
