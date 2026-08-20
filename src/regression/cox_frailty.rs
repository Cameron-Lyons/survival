use crate::constants::{
    CONVERGENCE_FLAG, COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE, EXP_CLAMP_MAX,
    EXP_CLAMP_MIN,
};
use ndarray::Array2;
use pyo3::prelude::*;

#[derive(Clone, Copy)]
enum Ties {
    Breslow,
    Efron,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct CoxPHFrailtyFit {
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub frailty: Vec<f64>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub score_vector: Vec<f64>,
    #[pyo3(get)]
    pub information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub naive_information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub frailty_variance: Vec<f64>,
    #[pyo3(get)]
    pub covariate_degrees_of_freedom: Vec<f64>,
    #[pyo3(get)]
    pub frailty_degrees_of_freedom: f64,
    #[pyo3(get)]
    pub degrees_of_freedom: f64,
    #[pyo3(get)]
    pub log_likelihood: Vec<f64>,
    #[pyo3(get)]
    pub penalized_log_likelihood: f64,
    #[pyo3(get)]
    pub score_test: f64,
    #[pyo3(get)]
    pub convergence_flag: i32,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub risk_scores: Vec<f64>,
    #[pyo3(get)]
    pub linear_predictors: Vec<f64>,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub event_times: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub covariates: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub strata: Vec<i32>,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub nocenter: Vec<f64>,
    #[pyo3(get)]
    pub offset: Vec<f64>,
}

#[pymethods]
impl CoxPHFrailtyFit {
    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        covariates
            .into_iter()
            .map(|row| {
                if row.len() != beta.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "covariate row has {} columns but model expects {}",
                        row.len(),
                        beta.len()
                    )));
                }
                validate_finite("covariates row", &row)?;
                Ok(row
                    .iter()
                    .zip(beta)
                    .map(|(value, coefficient)| value * coefficient)
                    .sum())
            })
            .collect()
    }
}

struct IterationState {
    score: Vec<f64>,
    group_diagonal: Vec<f64>,
    information: Array2<f64>,
    penalized_log_likelihood: f64,
}

struct SparseFrailtySolver {
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Array2<f64>,
    groups: Vec<usize>,
    strata_end: Vec<bool>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    theta: f64,
    ordinary_penalty: Array2<f64>,
    ties: Ties,
    max_iter: usize,
    eps: f64,
    tolerance: f64,
}

struct SolverResult {
    beta: Vec<f64>,
    frailty: Vec<f64>,
    means: Vec<f64>,
    score: Vec<f64>,
    covariance: Array2<f64>,
    naive_covariance: Array2<f64>,
    frailty_variance: Vec<f64>,
    covariate_df: Vec<f64>,
    frailty_df: f64,
    initial_log_likelihood: f64,
    final_log_likelihood: f64,
    penalized_log_likelihood: f64,
    score_test: f64,
    flag: i32,
    iterations: usize,
}

impl SparseFrailtySolver {
    fn partial_log_likelihood(&self, beta: &[f64], frailty: &[f64]) -> f64 {
        self.iteration(beta, frailty).penalized_log_likelihood + self.penalty_value(beta, frailty)
    }

    fn penalty_value(&self, beta: &[f64], frailty: &[f64]) -> f64 {
        let frailty_penalty = 0.5
            * (frailty
                .iter()
                .map(|value| value * value / self.theta)
                .sum::<f64>()
                + frailty.len() as f64 * (2.0 * std::f64::consts::PI * self.theta).ln());
        let ordinary_penalty = 0.5
            * beta
                .iter()
                .enumerate()
                .map(|(row, &coefficient)| {
                    coefficient
                        * beta
                            .iter()
                            .enumerate()
                            .map(|(column, &other)| self.ordinary_penalty[(row, column)] * other)
                            .sum::<f64>()
                })
                .sum::<f64>();
        frailty_penalty + ordinary_penalty
    }

    fn iteration(&self, beta: &[f64], frailty: &[f64]) -> IterationState {
        let nvar = beta.len();
        let nfrail = frailty.len();
        let width = nfrail + nvar;
        let mut score = vec![0.0; width];
        let mut group_diagonal = vec![0.0; nfrail];
        let mut information = Array2::zeros((nvar, width));
        let mut risk_sum = vec![0.0; width];
        let mut death_risk_sum = vec![0.0; width];
        let mut cross_sum = Array2::<f64>::zeros((nvar, width));
        let mut death_cross_sum = Array2::<f64>::zeros((nvar, width));
        let mut risk_means = vec![0.0; width];
        let mut log_likelihood = 0.0;
        let mut stratum_start = 0usize;

        while stratum_start < self.time.len() {
            let mut stratum_end = stratum_start;
            while !self.strata_end[stratum_end] {
                stratum_end += 1;
            }
            let mut denominator = 0.0;
            risk_sum.fill(0.0);
            cross_sum.fill(0.0);

            let mut time_start = stratum_start;
            while time_start <= stratum_end {
                let event_time = self.time[time_start];
                let mut time_end = time_start;
                while time_end < stratum_end && self.time[time_end + 1] == event_time {
                    time_end += 1;
                }

                death_risk_sum.fill(0.0);
                death_cross_sum.fill(0.0);
                let mut death_count = 0usize;
                let mut death_weight = 0.0;
                let mut death_risk = 0.0;

                for person in time_start..=time_end {
                    let group = self.groups[person];
                    let mut linear_predictor = self.offset[person] + frailty[group];
                    for (variable, &coefficient) in beta.iter().enumerate() {
                        linear_predictor += coefficient * self.covariates[(person, variable)];
                    }
                    let risk = linear_predictor.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp()
                        * self.weights[person];
                    denominator += risk;
                    risk_sum[group] += risk;
                    for variable in 0..nvar {
                        let value = self.covariates[(person, variable)];
                        let weighted = risk * value;
                        risk_sum[nfrail + variable] += weighted;
                        cross_sum[(variable, group)] += weighted;
                        for other in 0..=variable {
                            cross_sum[(variable, nfrail + other)] +=
                                weighted * self.covariates[(person, other)];
                        }
                    }

                    if self.status[person] != 0 {
                        death_count += 1;
                        death_weight += self.weights[person];
                        death_risk += risk;
                        log_likelihood += self.weights[person] * linear_predictor;
                        score[group] += self.weights[person];
                        death_risk_sum[group] += risk;
                        for variable in 0..nvar {
                            let value = self.covariates[(person, variable)];
                            let weighted = risk * value;
                            score[nfrail + variable] += self.weights[person] * value;
                            death_risk_sum[nfrail + variable] += weighted;
                            death_cross_sum[(variable, group)] += weighted;
                            for other in 0..=variable {
                                death_cross_sum[(variable, nfrail + other)] +=
                                    weighted * self.covariates[(person, other)];
                            }
                        }
                    }
                }

                if death_count > 0 {
                    let average_weight = death_weight / death_count as f64;
                    for step in 0..death_count {
                        let fraction = match self.ties {
                            Ties::Breslow => 0.0,
                            Ties::Efron => step as f64 / death_count as f64,
                        };
                        let adjusted_denominator = denominator - fraction * death_risk;
                        log_likelihood -= average_weight * adjusted_denominator.ln();
                        for index in 0..width {
                            let mean = (risk_sum[index] - fraction * death_risk_sum[index])
                                / adjusted_denominator;
                            risk_means[index] = mean;
                            score[index] -= average_weight * mean;
                            if index < nfrail {
                                group_diagonal[index] += mean * (1.0 - mean);
                            } else {
                                let variable = index - nfrail;
                                for other in 0..=index {
                                    information[(variable, other)] += average_weight
                                        * ((cross_sum[(variable, other)]
                                            - fraction * death_cross_sum[(variable, other)])
                                            / adjusted_denominator
                                            - mean * risk_means[other]);
                                }
                            }
                        }
                    }
                }

                time_start = time_end + 1;
            }
            stratum_start = stratum_end + 1;
        }

        let precision = 1.0 / self.theta;
        let center = frailty.iter().sum::<f64>() / nfrail as f64;
        let centered_frailty = frailty
            .iter()
            .map(|value| value - center)
            .collect::<Vec<_>>();
        for group in 0..nfrail {
            score[group] -= centered_frailty[group] * precision;
            group_diagonal[group] += precision;
        }
        log_likelihood -= 0.5
            * (centered_frailty
                .iter()
                .map(|value| value * value * precision)
                .sum::<f64>()
                + nfrail as f64 * (2.0 * std::f64::consts::PI * self.theta).ln());

        for row in 0..nvar {
            let penalty_score = beta
                .iter()
                .enumerate()
                .map(|(column, &other)| self.ordinary_penalty[(row, column)] * other)
                .sum::<f64>();
            score[nfrail + row] -= penalty_score;
            for column in 0..=row {
                information[(row, nfrail + column)] += self.ordinary_penalty[(row, column)];
            }
            log_likelihood -= 0.5 * beta[row] * penalty_score;
        }

        IterationState {
            score,
            group_diagonal,
            information,
            penalized_log_likelihood: log_likelihood,
        }
    }

    fn solve(self, means: Vec<f64>, mut beta: Vec<f64>, mut frailty: Vec<f64>) -> SolverResult {
        let nfrail = frailty.len();
        let nvar = beta.len();
        let initial_beta = beta.clone();
        let initial_frailty = vec![0.0; nfrail];
        let initial_log_likelihood = self.partial_log_likelihood(&initial_beta, &initial_frailty);
        let mut accepted_beta = beta.clone();
        let mut accepted_frailty = frailty.clone();
        let mut accepted_log_likelihood = initial_log_likelihood;
        let mut halving = false;
        let mut score_test = 0.0;
        let mut flag = 0;
        let mut iterations = 0;
        for iteration in 0..=self.max_iter {
            iterations = iteration;
            let center = frailty.iter().sum::<f64>() / nfrail as f64;
            for value in &mut frailty {
                *value -= center;
            }
            let state = self.iteration(&beta, &frailty);
            let new_log_likelihood = state.penalized_log_likelihood;
            let mut factor = state.information.clone();
            let diagonal = state.group_diagonal.clone();
            flag = cholesky3(&mut factor, nfrail, &diagonal, self.tolerance);

            if new_log_likelihood.abs() < self.eps
                || ((1.0 - accepted_log_likelihood / new_log_likelihood).abs() <= self.eps
                    && !halving)
            {
                break;
            }
            if iteration == self.max_iter {
                flag = CONVERGENCE_FLAG;
                break;
            }
            if iteration > 0 && new_log_likelihood < accepted_log_likelihood {
                halving = true;
                for (value, accepted) in beta.iter_mut().zip(&accepted_beta) {
                    *value = (*value + accepted) / 2.0;
                }
                for (value, accepted) in frailty.iter_mut().zip(&accepted_frailty) {
                    *value = (*value + accepted) / 2.0;
                }
                continue;
            }

            halving = false;
            accepted_log_likelihood = new_log_likelihood;
            accepted_beta.clone_from(&beta);
            accepted_frailty.clone_from(&frailty);
            let mut step = state.score.clone();
            chsolve3(&factor, nfrail, &diagonal, &mut step);
            if iteration == 0 {
                score_test = step.iter().zip(&state.score).map(|(a, b)| a * b).sum();
            }
            for variable in 0..nvar {
                beta[variable] += step[nfrail + variable];
            }
            for group in 0..nfrail {
                frailty[group] += step[group];
            }
        }

        let center = frailty.iter().sum::<f64>() / nfrail as f64;
        for value in &mut frailty {
            *value -= center;
        }
        let state = self.iteration(&beta, &frailty);
        let penalized_log_likelihood = state.penalized_log_likelihood;
        let final_log_likelihood = penalized_log_likelihood + self.penalty_value(&beta, &frailty);
        let mut factor = state.information.clone();
        let diagonal = state.group_diagonal.clone();
        flag = if flag == CONVERGENCE_FLAG {
            flag
        } else {
            cholesky3(&mut factor, nfrail, &diagonal, self.tolerance)
        };
        let hmat = normalized_factor(&factor, nfrail);
        let (inverse_factor, inverse_diagonal) = chinv3(factor, nfrail, diagonal);
        let covariance_parts = covariance_parts(
            &hmat,
            &inverse_factor,
            &inverse_diagonal,
            nfrail,
            &self.ordinary_penalty,
            1.0 / self.theta,
        );

        let covariate_df = (0..nvar)
            .map(|variable| {
                let variance = covariance_parts.covariance[(variable, variable)];
                if variance > 0.0 {
                    (covariance_parts.naive_covariance[(variable, variable)] / variance)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();

        SolverResult {
            beta,
            frailty,
            means,
            score: state.score[nfrail..].to_vec(),
            covariance: covariance_parts.covariance,
            naive_covariance: covariance_parts.naive_covariance,
            frailty_variance: covariance_parts.frailty_variance,
            covariate_df,
            frailty_df: covariance_parts.frailty_df,
            initial_log_likelihood,
            final_log_likelihood,
            penalized_log_likelihood,
            score_test,
            flag,
            iterations,
        }
    }
}

struct CovarianceParts {
    covariance: Array2<f64>,
    naive_covariance: Array2<f64>,
    frailty_variance: Vec<f64>,
    frailty_df: f64,
}

fn normalized_factor(factor: &Array2<f64>, nfrail: usize) -> Array2<f64> {
    let mut result = factor.clone();
    for variable in 0..factor.nrows() {
        result[(variable, nfrail + variable)] = 1.0;
        for column in (nfrail + variable + 1)..factor.ncols() {
            result[(variable, column)] = 0.0;
        }
    }
    result
}

fn covariance_parts(
    hmat: &Array2<f64>,
    inverse_factor: &Array2<f64>,
    inverse_diagonal: &[f64],
    nfrail: usize,
    ordinary_penalty: &Array2<f64>,
    frailty_precision: f64,
) -> CovarianceParts {
    let nvar = inverse_factor.nrows();
    let dense_diagonal = &inverse_diagonal[nfrail..];
    let mut covariance = Array2::<f64>::zeros((nvar, nvar));
    let mut frailty_variance = inverse_diagonal[..nfrail].to_vec();
    let mut cross = Array2::<f64>::zeros((nfrail, nvar));

    for group in 0..nfrail {
        for component in 0..nvar {
            let value = inverse_factor[(component, group)];
            frailty_variance[group] += value * value * dense_diagonal[component];
        }
    }
    for row in 0..nvar {
        for column in 0..nvar {
            covariance[(row, column)] = (0..nvar)
                .map(|component| {
                    inverse_factor[(component, nfrail + row)]
                        * dense_diagonal[component]
                        * inverse_factor[(component, nfrail + column)]
                })
                .sum();
        }
    }
    for group in 0..nfrail {
        for column in 0..nvar {
            cross[(group, column)] = (0..nvar)
                .map(|component| {
                    inverse_factor[(component, group)]
                        * dense_diagonal[component]
                        * inverse_factor[(component, nfrail + column)]
                })
                .sum();
        }
    }

    let mut naive_covariance = covariance.clone();
    for row in 0..nvar {
        for column in 0..nvar {
            naive_covariance[(row, column)] -= frailty_precision
                * (0..nfrail)
                    .map(|group| cross[(group, row)] * cross[(group, column)])
                    .sum::<f64>();
            let mut ordinary_adjustment = 0.0;
            for left in 0..nvar {
                for right in 0..nvar {
                    ordinary_adjustment += covariance[(row, left)]
                        * ordinary_penalty[(left, right)]
                        * covariance[(right, column)];
                }
            }
            naive_covariance[(row, column)] -= ordinary_adjustment;
        }
    }

    let mut frailty_df = nfrail as f64 - frailty_precision * frailty_variance.iter().sum::<f64>();
    if ordinary_penalty.iter().any(|value| *value != 0.0) && nvar > 0 {
        let mut h22 = Array2::<f64>::zeros((nvar, nvar));
        for row in 0..nvar {
            for column in 0..nvar {
                h22[(row, column)] = (0..(nfrail + nvar))
                    .filter_map(|component| {
                        let diagonal = inverse_diagonal[component];
                        (diagonal != 0.0)
                            .then(|| hmat[(row, component)] * hmat[(column, component)] / diagonal)
                    })
                    .sum();
            }
        }
        if let Some(h22_inverse) = invert_symmetric(&h22) {
            for variable in 0..nvar {
                frailty_df -= (covariance[(variable, variable)]
                    - h22_inverse[(variable, variable)])
                    * ordinary_penalty[(variable, variable)];
            }
        }
    }

    CovarianceParts {
        covariance,
        naive_covariance,
        frailty_variance,
        frailty_df: frailty_df.clamp(0.0, nfrail as f64),
    }
}

fn cholesky3(matrix: &mut Array2<f64>, nfrail: usize, diagonal: &[f64], tolerance: f64) -> i32 {
    let nvar = matrix.nrows();
    let mut epsilon = 0.0_f64;
    for &value in diagonal {
        if value < epsilon {
            epsilon = value;
        }
    }
    for variable in 0..nvar {
        if matrix[(variable, nfrail + variable)] < epsilon {
            epsilon = matrix[(variable, nfrail + variable)];
        }
    }
    epsilon = if epsilon == 0.0 {
        tolerance
    } else {
        epsilon * tolerance
    };
    let mut rank = 0_i32;
    let mut nonnegative = 1_i32;

    for group in 0..nfrail {
        let pivot = diagonal[group];
        if !pivot.is_finite() || pivot < epsilon {
            for variable in 0..nvar {
                matrix[(variable, group)] = 0.0;
            }
            if pivot < -8.0 * epsilon {
                nonnegative = -1;
            }
            continue;
        }
        rank += 1;
        for row in 0..nvar {
            let value = matrix[(row, group)] / pivot;
            matrix[(row, group)] = value;
            matrix[(row, nfrail + row)] -= value * value * pivot;
            for column in (row + 1)..nvar {
                matrix[(column, nfrail + row)] -= value * matrix[(column, group)];
            }
        }
    }

    for row in 0..nvar {
        let pivot = matrix[(row, nfrail + row)];
        if !pivot.is_finite() || pivot < epsilon {
            for column in row..nvar {
                matrix[(column, nfrail + row)] = 0.0;
            }
            if pivot < -8.0 * epsilon {
                nonnegative = -1;
            }
            continue;
        }
        rank += 1;
        for column in (row + 1)..nvar {
            let value = matrix[(column, nfrail + row)] / pivot;
            matrix[(column, nfrail + row)] = value;
            matrix[(column, nfrail + column)] -= value * value * pivot;
            for next in (column + 1)..nvar {
                matrix[(next, nfrail + column)] -= value * matrix[(next, nfrail + row)];
            }
        }
    }
    rank * nonnegative
}

fn chsolve3(matrix: &Array2<f64>, nfrail: usize, diagonal: &[f64], values: &mut [f64]) {
    let nvar = matrix.nrows();
    for row in 0..nvar {
        let mut value = values[nfrail + row];
        for group in 0..nfrail {
            value -= values[group] * matrix[(row, group)];
        }
        for prior in 0..row {
            value -= values[nfrail + prior] * matrix[(row, nfrail + prior)];
        }
        values[nfrail + row] = value;
    }
    for row in (0..nvar).rev() {
        let pivot = matrix[(row, nfrail + row)];
        if pivot == 0.0 {
            values[nfrail + row] = 0.0;
        } else {
            let mut value = values[nfrail + row] / pivot;
            for next in (row + 1)..nvar {
                value -= values[nfrail + next] * matrix[(next, nfrail + row)];
            }
            values[nfrail + row] = value;
        }
    }
    for group in (0..nfrail).rev() {
        if diagonal[group] == 0.0 {
            values[group] = 0.0;
        } else {
            let mut value = values[group] / diagonal[group];
            for row in 0..nvar {
                value -= values[nfrail + row] * matrix[(row, group)];
            }
            values[group] = value;
        }
    }
}

fn chinv3(
    mut matrix: Array2<f64>,
    nfrail: usize,
    mut diagonal: Vec<f64>,
) -> (Array2<f64>, Vec<f64>) {
    let nvar = matrix.nrows();
    for group in 0..nfrail {
        if diagonal[group] > 0.0 {
            diagonal[group] = 1.0 / diagonal[group];
            for row in 0..nvar {
                matrix[(row, group)] = -matrix[(row, group)];
            }
        }
    }
    for row in 0..nvar {
        let diagonal_column = nfrail + row;
        if matrix[(row, diagonal_column)] > 0.0 {
            matrix[(row, diagonal_column)] = 1.0 / matrix[(row, diagonal_column)];
            for next in (row + 1)..nvar {
                matrix[(next, diagonal_column)] = -matrix[(next, diagonal_column)];
                for column in 0..diagonal_column {
                    matrix[(next, column)] +=
                        matrix[(next, diagonal_column)] * matrix[(row, column)];
                }
            }
        }
    }
    for row in 0..nvar {
        diagonal.push(matrix[(row, nfrail + row)]);
        matrix[(row, nfrail + row)] = 1.0;
        for column in (nfrail + row + 1)..(nfrail + nvar) {
            matrix[(row, column)] = 0.0;
        }
    }
    (matrix, diagonal)
}

fn invert_symmetric(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let n = matrix.nrows();
    let mut augmented = Array2::<f64>::zeros((n, 2 * n));
    for row in 0..n {
        for column in 0..n {
            augmented[(row, column)] = matrix[(row, column)];
        }
        augmented[(row, n + row)] = 1.0;
    }
    for pivot_column in 0..n {
        let pivot_row = (pivot_column..n).max_by(|&left, &right| {
            augmented[(left, pivot_column)]
                .abs()
                .total_cmp(&augmented[(right, pivot_column)].abs())
        })?;
        let pivot = augmented[(pivot_row, pivot_column)];
        if !pivot.is_finite() || pivot.abs() <= f64::EPSILON {
            return None;
        }
        if pivot_row != pivot_column {
            for column in 0..2 * n {
                augmented.swap((pivot_row, column), (pivot_column, column));
            }
        }
        let pivot = augmented[(pivot_column, pivot_column)];
        for column in 0..2 * n {
            augmented[(pivot_column, column)] /= pivot;
        }
        for row in 0..n {
            if row == pivot_column {
                continue;
            }
            let multiple = augmented[(row, pivot_column)];
            for column in 0..2 * n {
                augmented[(row, column)] -= multiple * augmented[(pivot_column, column)];
            }
        }
    }
    Some(Array2::from_shape_fn((n, n), |(row, column)| {
        augmented[(row, n + column)]
    }))
}

fn validate_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name}[{index}] must be finite"
        )));
    }
    Ok(())
}

fn validate_penalty(values: Option<Vec<Vec<f64>>>, nvar: usize) -> PyResult<Array2<f64>> {
    let Some(values) = values else {
        return Ok(Array2::zeros((nvar, nvar)));
    };
    if values.len() != nvar || values.iter().any(|row| row.len() != nvar) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "penalty_matrix must have shape {nvar} x {nvar}"
        )));
    }
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    validate_finite("penalty_matrix", &flat)?;
    let mut matrix = Array2::from_shape_vec((nvar, nvar), flat).expect("validated penalty shape");
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let tolerance = 1e-10 * scale.max(f64::MIN_POSITIVE);
    for row in 0..nvar {
        for column in 0..row {
            let left = matrix[(row, column)];
            let right = matrix[(column, row)];
            if (left - right).abs() > tolerance {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "penalty_matrix must be symmetric",
                ));
            }
            matrix[(row, column)] = 0.5 * (left + right);
            matrix[(column, row)] = matrix[(row, column)];
        }
    }
    let mut factor = matrix.clone();
    for pivot_index in 0..nvar {
        let largest_diagonal = (pivot_index..nvar)
            .max_by(|&left, &right| factor[(left, left)].total_cmp(&factor[(right, right)]))
            .unwrap_or(pivot_index);
        if largest_diagonal != pivot_index {
            for column in 0..nvar {
                factor.swap((pivot_index, column), (largest_diagonal, column));
            }
            for row in 0..nvar {
                factor.swap((row, pivot_index), (row, largest_diagonal));
            }
        }
        let value = factor[(pivot_index, pivot_index)];
        if value < -tolerance || !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "penalty_matrix must be positive semidefinite",
            ));
        }
        if value <= tolerance {
            if (pivot_index..nvar).any(|row| {
                (pivot_index..nvar).any(|column| factor[(row, column)].abs() > tolerance)
            }) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "penalty_matrix must be positive semidefinite",
                ));
            }
            continue;
        }
        for row in (pivot_index + 1)..nvar {
            for column in row..nvar {
                let updated = factor[(row, column)]
                    - factor[(row, pivot_index)] * factor[(column, pivot_index)] / value;
                factor[(row, column)] = updated;
                factor[(column, row)] = updated;
            }
        }
    }
    Ok(matrix)
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, groups, theta, strata=None, weights=None, offset=None, initial_beta=None, initial_frailty=None, max_iter=None, eps=None, toler=None, method=None, nocenter=None, penalty_matrix=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_frailty_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    groups: Vec<usize>,
    theta: f64,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    initial_beta: Option<Vec<f64>>,
    initial_frailty: Option<Vec<f64>>,
    max_iter: Option<usize>,
    eps: Option<f64>,
    toler: Option<f64>,
    method: Option<&str>,
    nocenter: Option<Vec<f64>>,
    penalty_matrix: Option<Vec<Vec<f64>>>,
) -> PyResult<CoxPHFrailtyFit> {
    let n = time.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time must not be empty",
        ));
    }
    for (name, length) in [
        ("status", status.len()),
        ("covariates", covariates.len()),
        ("groups", groups.len()),
    ] {
        if length != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} has {length} rows but time has {n}"
            )));
        }
    }
    validate_finite("time", &time)?;
    if status.iter().any(|value| *value != 0 && *value != 1) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "status must contain only 0 and 1",
        ));
    }
    let nvar = covariates.first().map_or(0, Vec::len);
    if covariates.iter().any(|row| row.len() != nvar) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covariates must be rectangular",
        ));
    }
    validate_finite(
        "covariates",
        &covariates.iter().flatten().copied().collect::<Vec<_>>(),
    )?;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "theta must be a finite positive value",
        ));
    }
    let nfrail = groups.iter().copied().max().map_or(0, |value| value + 1);
    let mut observed_groups = vec![false; nfrail];
    for &group in &groups {
        observed_groups[group] = true;
    }
    if nfrail == 0 || observed_groups.iter().any(|observed| !observed) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "groups must be contiguous zero-based indices",
        ));
    }
    let check_optional = |name: &str, length: Option<usize>| -> PyResult<()> {
        if let Some(length) = length
            && length != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} has {length} rows but time has {n}"
            )));
        }
        Ok(())
    };
    check_optional("strata", strata.as_ref().map(Vec::len))?;
    check_optional("weights", weights.as_ref().map(Vec::len))?;
    check_optional("offset", offset.as_ref().map(Vec::len))?;
    if let Some(values) = weights.as_ref() {
        validate_finite("weights", values)?;
        if values.iter().any(|value| *value <= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weights must contain positive values",
            ));
        }
    }
    if let Some(values) = offset.as_ref() {
        validate_finite("offset", values)?;
    }
    let beta = initial_beta.unwrap_or_else(|| vec![0.0; nvar]);
    if beta.len() != nvar {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_beta has {} values but covariates has {nvar} columns",
            beta.len()
        )));
    }
    validate_finite("initial_beta", &beta)?;
    let frailty = initial_frailty.unwrap_or_else(|| vec![0.0; nfrail]);
    if frailty.len() != nfrail {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_frailty has {} values but groups contains {nfrail} levels",
            frailty.len()
        )));
    }
    validate_finite("initial_frailty", &frailty)?;
    let eps = eps.unwrap_or(COX_CONVERGENCE_TOLERANCE);
    let toler = toler.unwrap_or(COX_RANK_TOLERANCE);
    if !eps.is_finite() || eps <= 0.0 || !toler.is_finite() || toler <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eps and toler must be finite positive values",
        ));
    }
    let ties = match method.unwrap_or("efron").to_ascii_lowercase().as_str() {
        "breslow" => Ties::Breslow,
        "efron" => Ties::Efron,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'breslow' or 'efron'",
            ));
        }
    };
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let offset = offset.unwrap_or_else(|| vec![0.0; n]);
    let nocenter = nocenter.unwrap_or_else(|| vec![-1.0, 0.0, 1.0]);
    validate_finite("nocenter", &nocenter)?;
    let ordinary_penalty = validate_penalty(penalty_matrix, nvar)?;

    let doscale = (0..nvar)
        .map(|column| {
            !covariates
                .iter()
                .all(|row| nocenter.iter().any(|value| row[column] == *value))
        })
        .collect::<Vec<_>>();
    let means = (0..nvar)
        .map(|column| {
            if doscale[column] {
                covariates.iter().map(|row| row[column]).sum::<f64>() / n as f64
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        strata[left]
            .cmp(&strata[right])
            .then_with(|| time[right].total_cmp(&time[left]))
            .then_with(|| status[left].cmp(&status[right]))
            .then_with(|| left.cmp(&right))
    });
    let sorted_time = order.iter().map(|&index| time[index]).collect::<Vec<_>>();
    let sorted_status = order.iter().map(|&index| status[index]).collect::<Vec<_>>();
    let sorted_groups = order.iter().map(|&index| groups[index]).collect::<Vec<_>>();
    let sorted_offset = order.iter().map(|&index| offset[index]).collect::<Vec<_>>();
    let sorted_weights = order
        .iter()
        .map(|&index| weights[index])
        .collect::<Vec<_>>();
    let mut strata_end = vec![false; n];
    for sorted in 0..n {
        strata_end[sorted] = sorted + 1 == n || strata[order[sorted + 1]] != strata[order[sorted]];
    }
    let sorted_covariates = Array2::from_shape_fn((n, nvar), |(row, column)| {
        covariates[order[row]][column] - means[column]
    });
    let solver = SparseFrailtySolver {
        time: sorted_time,
        status: sorted_status,
        covariates: sorted_covariates,
        groups: sorted_groups,
        strata_end,
        offset: sorted_offset,
        weights: sorted_weights,
        theta,
        ordinary_penalty,
        ties,
        max_iter: max_iter.unwrap_or(COX_MAX_ITER),
        eps,
        tolerance: toler,
    };
    let result = solver.solve(means, beta, frailty);
    let mut linear_predictors = Vec::with_capacity(n);
    let mut risk_scores = Vec::with_capacity(n);
    for row in 0..n {
        let linear_predictor = offset[row]
            + result.frailty[groups[row]]
            + covariates[row]
                .iter()
                .zip(&result.means)
                .zip(&result.beta)
                .map(|((&value, &mean), &coefficient)| (value - mean) * coefficient)
                .sum::<f64>();
        linear_predictors.push(linear_predictor);
        risk_scores.push(linear_predictor.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp());
    }
    let covariance = result
        .covariance
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let naive_covariance = result
        .naive_covariance
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let degrees_of_freedom = result.covariate_df.iter().sum::<f64>() + result.frailty_df;

    Ok(CoxPHFrailtyFit {
        coefficients: vec![result.beta],
        frailty: result.frailty,
        means: result.means,
        score_vector: result.score,
        information_matrix: covariance,
        naive_information_matrix: naive_covariance,
        frailty_variance: result.frailty_variance,
        covariate_degrees_of_freedom: result.covariate_df,
        frailty_degrees_of_freedom: result.frailty_df,
        degrees_of_freedom,
        log_likelihood: vec![result.initial_log_likelihood, result.final_log_likelihood],
        penalized_log_likelihood: result.penalized_log_likelihood,
        score_test: result.score_test,
        convergence_flag: result.flag,
        iterations: result.iterations,
        risk_scores,
        linear_predictors,
        theta,
        event_times: time,
        status,
        weights,
        covariates,
        strata,
        method: match ties {
            Ties::Breslow => "breslow",
            Ties::Efron => "efron",
        }
        .to_string(),
        nocenter,
        offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specialized_factorization_solves_diagonal_dense_system() {
        let nfrail = 2;
        let diagonal = vec![3.0, 4.0];
        let mut matrix =
            Array2::from_shape_vec((2, 4), vec![0.2, 0.3, 2.0, 0.0, -0.1, 0.4, 0.5, 1.5]).unwrap();
        let mut rhs = vec![1.0, -0.5, 0.25, 2.0];
        assert_eq!(cholesky3(&mut matrix, nfrail, &diagonal, 1e-12), 4);
        chsolve3(&matrix, nfrail, &diagonal, &mut rhs);
        let full = [
            [3.0, 0.0, 0.2, -0.1],
            [0.0, 4.0, 0.3, 0.4],
            [0.2, 0.3, 2.0, 0.5],
            [-0.1, 0.4, 0.5, 1.5],
        ];
        let target = [1.0, -0.5, 0.25, 2.0];
        for row in 0..4 {
            let actual = (0..4)
                .map(|column| full[row][column] * rhs[column])
                .sum::<f64>();
            assert!((actual - target[row]).abs() < 1e-10);
        }
    }

    #[test]
    fn penalty_validation_rejects_indefinite_zero_diagonal_matrix() {
        assert!(validate_penalty(Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]), 2).is_err());
    }
}
