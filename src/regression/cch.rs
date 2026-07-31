use crate::regression::coxph::{CoxPHFit, CoxPHModel, Subject, coxph_fit};
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
    coxph_fit(
        stop,
        status,
        covariates,
        None,
        None,
        Some(offset),
        initial_beta,
        Some(max_iter),
        None,
        None,
        Some("efron"),
        Some(start),
        Some(vec![-1.0, 0.0, 1.0]),
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

    let fit = fit_cox(
        augmented_stop,
        augmented_status,
        augmented_covariates,
        augmented_start,
        offsets.clone(),
        initial_coefficients.clone(),
        if prentice { 35 } else { 20 },
    )?;
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
        offsets,
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
        method: method.as_r_name().to_string(),
        n,
        nevent,
        observed_n,
        subcohort_size,
        cohort_size,
        robust,
        score_residual_rows,
        dfbeta_rows,
    })
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
