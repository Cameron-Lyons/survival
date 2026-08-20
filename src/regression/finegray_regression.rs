use pyo3::prelude::*;
use std::fmt;

use crate::constants::{
    IPCW_SURVIVAL_FLOOR, clamped_normal_ci_bounds, exp_clamped, normal_ci_bounds_95, same_time,
};
use crate::internal::matrix::invert_matrix;
#[cfg(test)]
use crate::internal::statistical::compute_censoring_km;
use crate::internal::statistical::{normal_cdf, normal_inverse_cdf};

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct FineGrayResult {
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_scores: Vec<f64>,
    pub p_values: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub variance_matrix: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub log_likelihood_null: f64,
    pub n_events: usize,
    pub n_competing: usize,
    pub n_censored: usize,
    pub n_observations: usize,
    pub event_type: i32,
    pub convergence: bool,
    pub iterations: usize,
}

impl fmt::Display for FineGrayResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FineGrayResult(coef={:?}, n_events={}, converged={})",
            self.coefficients, self.n_events, self.convergence
        )
    }
}

#[pymethods]
impl FineGrayResult {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coefficients: Vec<f64>,
        std_errors: Vec<f64>,
        z_scores: Vec<f64>,
        p_values: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        variance_matrix: Vec<Vec<f64>>,
        log_likelihood: f64,
        log_likelihood_null: f64,
        n_events: usize,
        n_competing: usize,
        n_censored: usize,
        n_observations: usize,
        event_type: i32,
        convergence: bool,
        iterations: usize,
    ) -> Self {
        Self {
            coefficients,
            std_errors,
            z_scores,
            p_values,
            ci_lower,
            ci_upper,
            variance_matrix,
            log_likelihood,
            log_likelihood_null,
            n_events,
            n_competing,
            n_censored,
            n_observations,
            event_type,
            convergence,
            iterations,
        }
    }

    fn hazard_ratio(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|&coefficient| exp_clamped(coefficient))
            .collect()
    }

    fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Fine-Gray Subdistribution Hazard Model\n");
        s.push_str("======================================\n\n");
        s.push_str(&format!(
            "N={}, Events={}, Competing={}, Censored={}\n",
            self.n_observations, self.n_events, self.n_competing, self.n_censored
        ));
        s.push_str(&format!("Event type: {}\n\n", self.event_type));
        s.push_str("Coefficients:\n");
        s.push_str("  coef      exp(coef)  se(coef)   z        p\n");
        for i in 0..self.coefficients.len() {
            s.push_str(&format!(
                "  {:.4}    {:.4}     {:.4}     {:.3}    {:.4}\n",
                self.coefficients[i],
                exp_clamped(self.coefficients[i]),
                self.std_errors[i],
                self.z_scores[i],
                self.p_values[i]
            ));
        }
        s.push_str(&format!(
            "\nLog-likelihood: {:.4} (null: {:.4})\n",
            self.log_likelihood, self.log_likelihood_null
        ));
        s.push_str(&format!("Converged: {}\n", self.convergence));
        s
    }
}

#[derive(Debug, Clone)]
#[pyclass(str, get_all, from_py_object)]
pub struct CompetingRisksCIF {
    pub times: Vec<f64>,
    pub cif: Vec<f64>,
    pub variance: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
    pub n_risk: Vec<usize>,
    pub n_events: Vec<usize>,
    pub event_type: i32,
}

impl fmt::Display for CompetingRisksCIF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompetingRisksCIF(event_type={}, n_times={})",
            self.event_type,
            self.times.len()
        )
    }
}

#[pymethods]
impl CompetingRisksCIF {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        times: Vec<f64>,
        cif: Vec<f64>,
        variance: Vec<f64>,
        ci_lower: Vec<f64>,
        ci_upper: Vec<f64>,
        n_risk: Vec<usize>,
        n_events: Vec<usize>,
        event_type: i32,
    ) -> Self {
        Self {
            times,
            cif,
            variance,
            ci_lower,
            ci_upper,
            n_risk,
            n_events,
            event_type,
        }
    }
}

pub(crate) fn finegray_regression_core(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> FineGrayResult {
    let n = time.len();
    let p = if n > 0 && !covariates.is_empty() {
        covariates[0].len()
    } else {
        0
    };

    if n == 0 || p == 0 {
        return FineGrayResult {
            coefficients: vec![],
            std_errors: vec![],
            z_scores: vec![],
            p_values: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            variance_matrix: vec![],
            log_likelihood: 0.0,
            log_likelihood_null: 0.0,
            n_events: 0,
            n_competing: 0,
            n_censored: 0,
            n_observations: 0,
            event_type,
            convergence: false,
            iterations: 0,
        };
    }

    let n_events = status.iter().filter(|&&s| s == event_type).count();
    let n_competing = status
        .iter()
        .filter(|&&s| s != 0 && s != event_type)
        .count();
    let n_censored = status.iter().filter(|&&s| s == 0).count();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
    let event_groups = target_event_groups(&indices, time, status, event_type);
    let censoring_survival = censoring_survival_before(&indices, time, status);

    let mut beta = vec![0.0; p];
    let mut converged = false;
    let mut iterations = 0;

    let (_, _, log_likelihood_null) = compute_gradient_hessian(
        &event_groups,
        &vec![0.0; p],
        time,
        status,
        covariates,
        event_type,
        &indices,
        &censoring_survival,
    );

    for iter in 0..max_iter {
        iterations = iter + 1;

        let (gradient, hessian, _ll) = compute_gradient_hessian(
            &event_groups,
            &beta,
            time,
            status,
            covariates,
            event_type,
            &indices,
            &censoring_survival,
        );

        let neg_hessian: Vec<Vec<f64>> = hessian
            .iter()
            .map(|row| row.iter().map(|&x| -x).collect())
            .collect();

        let hess_inv = match invert_matrix(&neg_hessian) {
            Some(inv) => inv,
            None => break,
        };

        let mut delta = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                delta[i] += hess_inv[i][j] * gradient[j];
            }
        }

        let max_delta = delta.iter().map(|&d| d.abs()).fold(0.0, f64::max);

        for i in 0..p {
            beta[i] += delta[i];
        }

        if max_delta < eps {
            converged = true;
            break;
        }
    }

    let (_, hessian, log_likelihood) = compute_gradient_hessian(
        &event_groups,
        &beta,
        time,
        status,
        covariates,
        event_type,
        &indices,
        &censoring_survival,
    );

    let neg_hessian: Vec<Vec<f64>> = hessian
        .iter()
        .map(|row| row.iter().map(|&x| -x).collect())
        .collect();

    let variance_matrix = invert_matrix(&neg_hessian).unwrap_or_else(|| vec![vec![0.0; p]; p]);

    let std_errors: Vec<f64> = (0..p)
        .map(|i| variance_matrix[i][i].max(0.0).sqrt())
        .collect();

    let z_scores: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se > crate::constants::DIVISION_FLOOR {
                b / se
            } else {
                0.0
            }
        })
        .collect();

    let p_values: Vec<f64> = z_scores
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let (ci_lower, ci_upper) = normal_ci_bounds_95(&beta, &std_errors);

    FineGrayResult {
        coefficients: beta,
        std_errors,
        z_scores,
        p_values,
        ci_lower,
        ci_upper,
        variance_matrix,
        log_likelihood,
        log_likelihood_null,
        n_events,
        n_competing,
        n_censored,
        n_observations: n,
        event_type,
        convergence: converged,
        iterations,
    }
}

fn target_event_groups(
    sorted_indices: &[usize],
    time: &[f64],
    status: &[i32],
    event_type: i32,
) -> Vec<Vec<usize>> {
    let event_indices = sorted_indices
        .iter()
        .copied()
        .filter(|&idx| status[idx] == event_type)
        .collect::<Vec<_>>();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for idx in event_indices {
        if groups
            .last()
            .is_some_and(|group| same_time(time[group[0]], time[idx]))
        {
            groups.last_mut().expect("event group exists").push(idx);
        } else {
            groups.push(vec![idx]);
        }
    }
    groups
}

fn censoring_survival_before(sorted_indices: &[usize], time: &[f64], status: &[i32]) -> Vec<f64> {
    let mut survival_before = vec![1.0; time.len()];
    let mut survival = 1.0;
    let mut at_risk = time.len();
    let mut group_start = 0;

    while group_start < sorted_indices.len() {
        let group_time = time[sorted_indices[group_start]];
        let mut group_end = group_start + 1;
        while group_end < sorted_indices.len()
            && same_time(time[sorted_indices[group_end]], group_time)
        {
            group_end += 1;
        }

        let group = &sorted_indices[group_start..group_end];
        for &idx in group {
            survival_before[idx] = survival;
        }
        let censored = group.iter().filter(|&&idx| status[idx] == 0).count();
        let failures = group.len() - censored;
        let censoring_risk = at_risk.saturating_sub(failures);
        if censored > 0 && censoring_risk > 0 {
            survival *= 1.0 - censored as f64 / censoring_risk as f64;
        }
        at_risk -= group.len();
        group_start = group_end;
    }

    survival_before
}

#[derive(Clone)]
struct RiskMoments {
    scalar: f64,
    first: Vec<f64>,
    second: Vec<Vec<f64>>,
}

impl RiskMoments {
    fn new(n_vars: usize) -> Self {
        Self {
            scalar: 0.0,
            first: vec![0.0; n_vars],
            second: vec![vec![0.0; n_vars]; n_vars],
        }
    }

    fn add(&mut self, covariates: &[f64], weighted_risk: f64) {
        self.scalar += weighted_risk;
        for (column, &value) in covariates.iter().enumerate() {
            self.first[column] += weighted_risk * value;
            for (other_column, &other_value) in covariates.iter().enumerate() {
                self.second[column][other_column] += weighted_risk * value * other_value;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_gradient_hessian(
    event_groups: &[Vec<usize>],
    beta: &[f64],
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    sorted_indices: &[usize],
    censoring_survival: &[f64],
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n_vars = beta.len();
    let linear_predictors = covariates
        .iter()
        .map(|row| {
            row.iter()
                .zip(beta)
                .map(|(&value, &coefficient)| value * coefficient)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let risk_scores = linear_predictors
        .iter()
        .map(|&value| exp_clamped(value))
        .collect::<Vec<_>>();

    let mut competing_prefixes = Vec::with_capacity(event_groups.len());
    let mut competing_prefix = RiskMoments::new(n_vars);
    let mut row_cursor = 0;
    for group in event_groups {
        let event_time = time[group[0]];
        while row_cursor < sorted_indices.len()
            && time[sorted_indices[row_cursor]] < event_time
            && !same_time(time[sorted_indices[row_cursor]], event_time)
        {
            let idx = sorted_indices[row_cursor];
            if status[idx] != 0 && status[idx] != event_type {
                let normalized_risk =
                    risk_scores[idx] / censoring_survival[idx].max(IPCW_SURVIVAL_FLOOR);
                competing_prefix.add(&covariates[idx], normalized_risk);
            }
            row_cursor += 1;
        }
        competing_prefixes.push(competing_prefix.clone());
    }

    let mut gradient = vec![0.0; n_vars];
    let mut hessian = vec![vec![0.0; n_vars]; n_vars];
    let mut log_likelihood = 0.0;
    let mut ordinary_risk = RiskMoments::new(n_vars);
    let mut reverse_cursor = sorted_indices.len();

    for (group_idx, group) in event_groups.iter().enumerate().rev() {
        let event_time = time[group[0]];
        while reverse_cursor > 0 {
            let idx = sorted_indices[reverse_cursor - 1];
            if time[idx] < event_time && !same_time(time[idx], event_time) {
                break;
            }
            ordinary_risk.add(&covariates[idx], risk_scores[idx]);
            reverse_cursor -= 1;
        }

        let censoring_weight = censoring_survival[group[0]].max(IPCW_SURVIVAL_FLOOR);
        let prefix = &competing_prefixes[group_idx];
        let risk_sum =
            (ordinary_risk.scalar + censoring_weight * prefix.scalar).max(IPCW_SURVIVAL_FLOOR);
        let event_count = group.len() as f64;
        let mut means = vec![0.0; n_vars];

        for column in 0..n_vars {
            let first = ordinary_risk.first[column] + censoring_weight * prefix.first[column];
            means[column] = first / risk_sum;
            let event_covariate_sum = group
                .iter()
                .map(|&idx| covariates[idx][column])
                .sum::<f64>();
            gradient[column] += event_covariate_sum - event_count * means[column];
        }

        for column in 0..n_vars {
            for other_column in 0..n_vars {
                let second = ordinary_risk.second[column][other_column]
                    + censoring_weight * prefix.second[column][other_column];
                hessian[column][other_column] -=
                    event_count * (second / risk_sum - means[column] * means[other_column]);
            }
        }

        log_likelihood += group.iter().map(|&idx| linear_predictors[idx]).sum::<f64>()
            - event_count * risk_sum.ln();
    }

    (gradient, hessian, log_likelihood)
}

pub(crate) fn competing_risks_cif_core(
    time: &[f64],
    status: &[i32],
    event_type: i32,
    confidence_level: f64,
) -> CompetingRisksCIF {
    let n = time.len();

    if n == 0 {
        return CompetingRisksCIF {
            times: vec![],
            cif: vec![],
            variance: vec![],
            ci_lower: vec![],
            ci_upper: vec![],
            n_risk: vec![],
            n_events: vec![],
            event_type,
        };
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    let mut unique_times = Vec::new();
    let mut cif_values = Vec::new();
    let mut variance_values = Vec::new();
    let mut n_risk_values = Vec::new();
    let mut n_events_values = Vec::new();

    let mut cause_codes = status
        .iter()
        .copied()
        .filter(|&value| value > 0)
        .collect::<Vec<_>>();
    cause_codes.push(event_type);
    cause_codes.sort_unstable();
    cause_codes.dedup();
    let n_causes = cause_codes.len();
    let target_cause_idx = cause_codes
        .binary_search(&event_type)
        .expect("event type was inserted into cause codes");
    let mut state_probabilities = vec![0.0; n_causes + 1];
    state_probabilities[0] = 1.0;
    let mut state_covariance = vec![vec![0.0; n_causes + 1]; n_causes + 1];
    let mut at_risk = n;

    let mut i = 0;
    while i < n {
        let current_time = time[indices[i]];
        let mut event_counts = vec![0usize; n_causes];
        let mut total_at_time = 0;

        while i < n && same_time(time[indices[i]], current_time) {
            let s = status[indices[i]];
            if s > 0 {
                let cause_idx = cause_codes
                    .binary_search(&s)
                    .expect("observed event type exists in cause codes");
                event_counts[cause_idx] += 1;
            }
            total_at_time += 1;
            i += 1;
        }

        let hazards = event_counts
            .iter()
            .map(|&count| count as f64 / at_risk as f64)
            .collect::<Vec<_>>();
        let survival_before = state_probabilities[0];
        state_covariance = updated_aj_covariance(
            &state_covariance,
            survival_before,
            &hazards,
            &event_counts,
            at_risk,
        );
        for (cause_idx, &hazard) in hazards.iter().enumerate() {
            state_probabilities[cause_idx + 1] += survival_before * hazard;
        }
        state_probabilities[0] *= 1.0 - hazards.iter().sum::<f64>();

        let n_event_type = event_counts[target_cause_idx];
        let target_state = target_cause_idx + 1;
        let cum_inc = state_probabilities[target_state];
        let variance = state_covariance[target_state][target_state].max(0.0);

        unique_times.push(current_time);
        cif_values.push(cum_inc);
        variance_values.push(variance);
        n_risk_values.push(at_risk);
        n_events_values.push(n_event_type);

        at_risk -= total_at_time;
    }

    let z = normal_inverse_cdf(0.5 + confidence_level / 2.0);

    let cif_se: Vec<f64> = variance_values.iter().map(|&v| v.sqrt()).collect();
    let (ci_lower, ci_upper) = clamped_normal_ci_bounds(&cif_values, &cif_se, z, 0.0, 1.0);

    CompetingRisksCIF {
        times: unique_times,
        cif: cif_values,
        variance: variance_values,
        ci_lower,
        ci_upper,
        n_risk: n_risk_values,
        n_events: n_events_values,
        event_type,
    }
}

fn updated_aj_covariance(
    covariance: &[Vec<f64>],
    survival: f64,
    hazards: &[f64],
    event_counts: &[usize],
    at_risk: usize,
) -> Vec<Vec<f64>> {
    let n_states = hazards.len() + 1;
    let survival_scale = 1.0 - hazards.iter().sum::<f64>();
    let mut left_product = vec![vec![0.0; n_states]; n_states];
    for state in 0..n_states {
        for column in 0..n_states {
            left_product[state][column] = if state == 0 {
                survival_scale * covariance[0][column]
            } else {
                covariance[state][column] + hazards[state - 1] * covariance[0][column]
            };
        }
    }

    let mut updated = vec![vec![0.0; n_states]; n_states];
    for state in 0..n_states {
        updated[state][0] = survival_scale * left_product[state][0];
        for column in 1..n_states {
            updated[state][column] =
                left_product[state][column] + hazards[column - 1] * left_product[state][0];
        }
    }

    let risk = at_risk as f64;
    let risk_squared = risk * risk;
    let risk_cubed = risk_squared * risk;
    let survival_squared = survival * survival;
    for (cause, &cause_events) in event_counts.iter().enumerate() {
        for (other_cause, &other_events) in event_counts.iter().enumerate() {
            let mut hazard_covariance = -(cause_events as f64) * (other_events as f64) / risk_cubed;
            if cause == other_cause {
                hazard_covariance += cause_events as f64 / risk_squared;
            }
            let contribution = survival_squared * hazard_covariance;
            updated[0][0] += contribution;
            updated[0][other_cause + 1] -= contribution;
            updated[cause + 1][0] -= contribution;
            updated[cause + 1][other_cause + 1] += contribution;
        }
    }

    updated
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, event_type, max_iter=25, eps=1e-9))]
pub fn finegray_regression(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> PyResult<FineGrayResult> {
    validate_finegray_regression_input(&time, &status, &covariates, event_type, max_iter, eps)?;

    Ok(finegray_regression_core(
        &time,
        &status,
        &covariates,
        event_type,
        max_iter,
        eps,
    ))
}

#[pyfunction]
#[pyo3(signature = (time, status, event_type, confidence_level=0.95))]
pub fn competing_risks_cif(
    time: Vec<f64>,
    status: Vec<i32>,
    event_type: i32,
    confidence_level: f64,
) -> PyResult<CompetingRisksCIF> {
    validate_competing_risks_input(&time, &status, event_type, confidence_level)?;

    Ok(competing_risks_cif_core(
        &time,
        &status,
        event_type,
        confidence_level,
    ))
}

fn validate_survival_outcome(time: &[f64], status: &[i32], event_type: i32) -> PyResult<()> {
    if time.is_empty() {
        return Err(value_error("time must not be empty"));
    }
    if time.len() != status.len() {
        return Err(value_error("time and status must have the same length"));
    }
    if event_type <= 0 {
        return Err(value_error("event_type must be positive"));
    }
    for (idx, &value) in time.iter().enumerate() {
        if !value.is_finite() {
            return Err(value_error(format!(
                "time contains non-finite value at index {}",
                idx
            )));
        }
        if value < 0.0 {
            return Err(value_error(format!(
                "time contains negative value {} at index {}",
                value, idx
            )));
        }
    }
    for (idx, &value) in status.iter().enumerate() {
        if value < 0 {
            return Err(value_error(format!(
                "status contains negative value {} at index {}",
                value, idx
            )));
        }
    }
    Ok(())
}

fn validate_finegray_regression_input(
    time: &[f64],
    status: &[i32],
    covariates: &[Vec<f64>],
    event_type: i32,
    max_iter: usize,
    eps: f64,
) -> PyResult<()> {
    validate_survival_outcome(time, status, event_type)?;
    if time.len() != covariates.len() {
        return Err(value_error("time and covariates must have the same length"));
    }
    if covariates.is_empty() || covariates[0].is_empty() {
        return Err(value_error("covariates must not be empty"));
    }
    let p = covariates[0].len();
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != p {
            return Err(value_error(format!(
                "all covariate rows must have the same length (row {} has {} instead of {})",
                row_idx,
                row.len(),
                p
            )));
        }
        for (col_idx, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(value_error(format!(
                    "covariates contains non-finite value at row {}, column {}",
                    row_idx, col_idx
                )));
            }
        }
    }
    if max_iter == 0 {
        return Err(value_error("max_iter must be positive"));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(value_error("eps must be a positive finite value"));
    }
    Ok(())
}

fn validate_competing_risks_input(
    time: &[f64],
    status: &[i32],
    event_type: i32,
    confidence_level: f64,
) -> PyResult<()> {
    validate_survival_outcome(time, status, event_type)?;
    if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
        return Err(value_error(
            "confidence_level must be a finite value between 0 and 1",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finegray_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let status = vec![1, 2, 1, 0, 2, 1, 0, 1, 2, 1];
        let covariates: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64 * 0.1, (10 - i) as f64 * 0.1])
            .collect();

        let result = finegray_regression_core(&time, &status, &covariates, 1, 25, 1e-9);

        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.std_errors.len(), 2);
        assert!(result.n_events > 0);
        assert!(result.n_competing > 0);
    }

    #[test]
    fn censoring_weights_order_tied_failures_before_censoring() {
        let time: Vec<f64> = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let status = vec![1, 1, 0, 0, 2];
        let mut indices = (0..time.len()).collect::<Vec<_>>();
        indices.sort_by(|&left, &right| time[left].total_cmp(&time[right]));

        let survival = censoring_survival_before(&indices, &time, &status);

        assert_eq!(survival[0], 1.0);
        assert_eq!(survival[1], 1.0);
        assert_eq!(survival[2], 1.0);
        assert!((survival[3] - 2.0 / 3.0).abs() < 1e-12);
        assert!((survival[4] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_finegray_no_competing() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 1, 0, 1, 0];
        let covariates: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.2]).collect();

        let result = finegray_regression_core(&time, &status, &covariates, 1, 25, 1e-9);

        assert_eq!(result.n_competing, 0);
        assert!(result.n_events > 0);
    }

    #[test]
    fn test_competing_risks_cif_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let status = vec![1, 2, 1, 0, 2, 1, 0, 1];

        let result = competing_risks_cif_core(&time, &status, 1, 0.95);

        assert!(!result.times.is_empty());
        assert_eq!(result.times.len(), result.cif.len());
        for &c in &result.cif {
            assert!((0.0..=1.0).contains(&c));
        }
        for i in 1..result.cif.len() {
            assert!(result.cif[i] >= result.cif[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_competing_risks_cif_multiple_types() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 2, 1, 2, 0];

        let cif1 = competing_risks_cif_core(&time, &status, 1, 0.95);
        let cif2 = competing_risks_cif_core(&time, &status, 2, 0.95);

        assert!(cif1.cif.last().unwrap_or(&0.0) > &0.0);
        assert!(cif2.cif.last().unwrap_or(&0.0) > &0.0);

        let total_cif = cif1.cif.last().unwrap_or(&0.0) + cif2.cif.last().unwrap_or(&0.0);
        assert!(total_cif <= 1.0 + 1e-10);
    }

    #[test]
    fn test_competing_risks_cif_empty() {
        let result = competing_risks_cif_core(&[], &[], 1, 0.95);
        assert!(result.times.is_empty());
        assert!(result.cif.is_empty());
    }

    #[test]
    fn test_censoring_km() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let status = vec![1, 0, 1, 0, 1];

        let (km_times, km_values) = compute_censoring_km(&time, &status);

        assert!(!km_times.is_empty());
        assert_eq!(km_times.len(), km_values.len());

        for &v in &km_values {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_finegray_public_api_rejects_malformed_inputs() {
        pyo3::Python::initialize();
        assert!(
            finegray_regression(vec![], vec![], vec![], 1, 25, 1e-9)
                .unwrap_err()
                .to_string()
                .contains("time must not be empty")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![f64::NAN]], 1, 25, 1e-9,)
                .unwrap_err()
                .to_string()
                .contains("covariates contains non-finite")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![0.0]], 1, 0, 1e-9)
                .unwrap_err()
                .to_string()
                .contains("max_iter must be positive")
        );
        assert!(
            finegray_regression(vec![1.0], vec![1], vec![vec![0.0]], 1, 25, f64::INFINITY)
                .unwrap_err()
                .to_string()
                .contains("eps must be")
        );
    }

    #[test]
    fn test_competing_risks_cif_public_api_rejects_malformed_inputs() {
        pyo3::Python::initialize();
        assert!(
            competing_risks_cif(vec![1.0], vec![], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("same length")
        );
        assert!(
            competing_risks_cif(vec![f64::INFINITY], vec![1], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("time contains non-finite")
        );
        assert!(
            competing_risks_cif(vec![1.0], vec![-1], 1, 0.95)
                .unwrap_err()
                .to_string()
                .contains("status contains negative")
        );
        assert!(
            competing_risks_cif(vec![1.0], vec![1], 1, 1.0)
                .unwrap_err()
                .to_string()
                .contains("confidence_level")
        );
    }

    #[test]
    fn test_hazard_ratios_are_clamped_for_large_coefficients() {
        let result = FineGrayResult::new(
            vec![1_000.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![999.0],
            vec![1001.0],
            vec![vec![1.0]],
            0.0,
            0.0,
            1,
            0,
            0,
            1,
            1,
            true,
            1,
        );

        assert!(result.hazard_ratio()[0].is_finite());
        assert!(result.summary().contains("Fine-Gray"));
    }
}
