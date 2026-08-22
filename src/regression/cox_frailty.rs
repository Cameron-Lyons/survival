use crate::constants::{
    CONVERGENCE_FLAG, COX_CONVERGENCE_TOLERANCE, COX_MAX_ITER, COX_RANK_TOLERANCE, EXP_CLAMP_MAX,
    EXP_CLAMP_MIN,
};
use crate::internal::statistical::ln_gamma;
use crate::regression::cox_optimizer::{CoxFit, CoxFrailtyPenalty, Method as CoxMethod};
use crate::regression::coxph::CoxPHFit;
use crate::regression::coxph_detail_module::CoxphDetail;
use crate::regression::coxph_support::StratifiedBaselineLookup;
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

#[derive(Clone, Copy)]
enum Ties {
    Breslow,
    Efron,
}

#[derive(Clone, Copy)]
enum FrailtyDistribution {
    Gaussian,
    Gamma,
    StudentT(f64),
}

fn student_t_location_terms(value: f64, denominator: f64) -> (f64, f64) {
    let scaled_square = value * value / denominator;
    let temp = 1.0 + scaled_square;
    (
        value / temp,
        1.0 / temp - 2.0 * scaled_square / (temp * temp),
    )
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct CoxPHFrailtyFit {
    #[pyo3(get)]
    pub frailty: Vec<f64>,
    #[pyo3(get)]
    pub naive_information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub frailty_variance: Vec<f64>,
    #[pyo3(get)]
    pub covariate_degrees_of_freedom: Vec<f64>,
    #[pyo3(get)]
    pub frailty_degrees_of_freedom: f64,
    #[pyo3(get)]
    pub penalized_log_likelihood: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub distribution: String,
    #[pyo3(get)]
    pub tdf: Option<f64>,
    #[pyo3(get)]
    pub penalty_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dense: bool,
    #[pyo3(get)]
    pub frailty_columns: Vec<usize>,
    #[pyo3(get)]
    pub offset: Vec<f64>,
    diagnostic_fit: CoxPHFit,
}

#[pymethods]
impl CoxPHFrailtyFit {
    #[getter]
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.diagnostic_fit.coefficients.clone()
    }

    #[getter]
    pub fn means(&self) -> Vec<f64> {
        self.diagnostic_fit.means.clone()
    }

    #[getter]
    pub fn score_vector(&self) -> Vec<f64> {
        self.diagnostic_fit.score_vector.clone()
    }

    #[getter]
    pub fn information_matrix(&self) -> Vec<Vec<f64>> {
        self.diagnostic_fit.information_matrix.clone()
    }

    #[getter]
    pub fn degrees_of_freedom(&self) -> f64 {
        self.diagnostic_fit.degrees_of_freedom
    }

    #[getter]
    pub fn log_likelihood(&self) -> Vec<f64> {
        self.diagnostic_fit.log_likelihood.clone()
    }

    #[getter]
    pub fn score_test(&self) -> f64 {
        self.diagnostic_fit.score_test
    }

    #[getter]
    pub fn convergence_flag(&self) -> i32 {
        self.diagnostic_fit.convergence_flag
    }

    #[getter]
    pub fn iterations(&self) -> usize {
        self.diagnostic_fit.iterations
    }

    #[getter]
    pub fn linear_predictors(&self) -> Vec<f64> {
        let center = self.ordinary_linear_predictor_center();
        self.diagnostic_fit
            .linear_predictors
            .iter()
            .map(|value| value - center)
            .collect()
    }

    #[getter]
    pub fn risk_scores(&self) -> Vec<f64> {
        self.linear_predictors()
            .into_iter()
            .map(|value| value.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp())
            .collect()
    }

    #[getter]
    pub fn event_times(&self) -> Vec<f64> {
        self.diagnostic_fit.event_times.clone()
    }

    #[getter]
    pub fn status(&self) -> Vec<i32> {
        self.diagnostic_fit.status.clone()
    }

    #[getter]
    pub fn entry_times(&self) -> Option<Vec<f64>> {
        self.diagnostic_fit.entry_times.clone()
    }

    #[getter]
    pub fn weights(&self) -> Vec<f64> {
        self.diagnostic_fit.weights.clone()
    }

    #[getter]
    pub fn covariates(&self) -> Vec<Vec<f64>> {
        self.diagnostic_fit.covariates.clone()
    }

    #[getter]
    pub fn strata(&self) -> Vec<i32> {
        self.diagnostic_fit.strata.clone()
    }

    #[getter]
    pub fn method(&self) -> String {
        self.diagnostic_fit.method.clone()
    }

    #[getter]
    pub fn nocenter(&self) -> Vec<f64> {
        self.diagnostic_fit.nocenter.clone()
    }

    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.diagnostic_fit.predict(covariates)
    }

    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.diagnostic_fit.hazard_ratios()
    }

    #[pyo3(signature = (centered = true))]
    pub fn basehaz(&self, centered: bool) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (times, hazards, _) = self.basehaz_with_strata(centered)?;
        Ok((times, hazards))
    }

    #[pyo3(signature = (centered = true))]
    pub fn basehaz_with_strata(&self, centered: bool) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
        self.baseline_with_strata(centered, true)
    }

    #[pyo3(signature = (covariates = None, centered = true))]
    pub fn survival_curve(
        &self,
        covariates: Option<Vec<Vec<f64>>>,
        centered: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let beta = self.ordinary_coefficients()?;
        let rows = match covariates {
            Some(rows) => rows,
            None => {
                let strata = self.unique_strata();
                return self.survival_curves_with_strata(
                    std::slice::from_ref(&self.diagnostic_fit.means),
                    &strata,
                    centered,
                    true,
                );
            }
        };
        self.validate_prediction_rows(&rows, beta.len())?;
        let center = self.baseline_center(centered);
        let (times, hazards, _) = self.baseline_with_strata(centered, false)?;
        let curves = rows
            .iter()
            .map(|row| {
                let risk = (Self::row_linear_predictor(row, beta) - center)
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                hazards
                    .iter()
                    .map(|hazard| (-(hazard * risk)).exp().clamp(0.0, 1.0))
                    .collect()
            })
            .collect();
        Ok((times, curves))
    }

    #[pyo3(signature = (covariates, strata, centered = true))]
    pub fn survival_curve_with_strata(
        &self,
        covariates: Vec<Vec<f64>>,
        strata: Vec<i32>,
        centered: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.survival_curves_with_strata(&covariates, &strata, centered, false)
    }

    pub fn expected_events(&self) -> PyResult<Vec<f64>> {
        self.diagnostic_fit.expected_events()
    }

    pub fn expected_basehaz_with_strata(&self) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
        self.diagnostic_fit.basehaz_with_strata(false)
    }

    pub fn martingale_residuals(&self) -> PyResult<Vec<f64>> {
        self.diagnostic_fit.martingale_residuals()
    }

    pub fn deviance_residuals(&self) -> PyResult<Vec<f64>> {
        self.diagnostic_fit.deviance_residuals()
    }

    pub fn schoenfeld_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.schoenfeld_residuals()
    }

    pub fn scaled_schoenfeld_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.scaled_schoenfeld_residuals()
    }

    pub fn scaled_schoenfeld_residuals_with_variance(
        &self,
        information_matrix: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit
            .scaled_schoenfeld_residuals_with_variance(information_matrix)
    }

    pub fn coxph_detail(&self, riskmat: bool) -> PyResult<CoxphDetail> {
        self.diagnostic_fit.coxph_detail(riskmat)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transformed_events, active_columns, groups, information_matrix, single_df, global_test, penalty_matrix=None))]
    pub fn cox_zph_diagnostics(
        &self,
        transformed_events: Vec<f64>,
        active_columns: Vec<usize>,
        groups: Vec<Vec<usize>>,
        information_matrix: Vec<Vec<f64>>,
        single_df: bool,
        global_test: bool,
        penalty_matrix: Option<Vec<Vec<f64>>>,
    ) -> PyResult<(Vec<Vec<f64>>, crate::validation::ProportionalityTest)> {
        self.diagnostic_fit.cox_zph_diagnostics(
            transformed_events,
            active_columns,
            groups,
            information_matrix,
            single_df,
            global_test,
            penalty_matrix,
        )
    }

    pub fn partial_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.partial_residuals()
    }

    pub fn score_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.score_residuals()
    }

    pub fn dfbeta(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.dfbeta()
    }

    pub fn dfbetas(&self) -> PyResult<Vec<Vec<f64>>> {
        self.diagnostic_fit.dfbetas()
    }
}

impl CoxPHFrailtyFit {
    fn ordinary_coefficients(&self) -> PyResult<&[f64]> {
        self.diagnostic_fit
            .coefficients
            .first()
            .map(Vec::as_slice)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
            })
    }

    fn ordinary_linear_predictor_center(&self) -> f64 {
        self.diagnostic_fit
            .means
            .iter()
            .zip(
                self.diagnostic_fit
                    .coefficients
                    .first()
                    .into_iter()
                    .flatten(),
            )
            .map(|(mean, coefficient)| mean * coefficient)
            .sum()
    }

    fn baseline_center(&self, centered: bool) -> f64 {
        if !centered {
            return 0.0;
        }
        let weight_sum = self.diagnostic_fit.weights.iter().sum::<f64>();
        let offset_center = if weight_sum > 0.0 {
            self.offset
                .iter()
                .zip(&self.diagnostic_fit.weights)
                .map(|(offset, weight)| offset * weight)
                .sum::<f64>()
                / weight_sum
        } else {
            0.0
        };
        self.ordinary_linear_predictor_center() + offset_center
    }

    fn baseline_linear_predictors(&self) -> PyResult<Vec<f64>> {
        let beta = self.ordinary_coefficients()?;
        Ok(self
            .diagnostic_fit
            .covariates
            .iter()
            .zip(&self.offset)
            .map(|(row, offset)| Self::row_linear_predictor(row, beta) + offset)
            .collect())
    }

    fn baseline_with_strata(
        &self,
        centered: bool,
        include_censor_times: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
        self.diagnostic_fit.compute_basehaz_with_predictors(
            &self.baseline_linear_predictors()?,
            self.baseline_center(centered),
            include_censor_times,
        )
    }

    fn unique_strata(&self) -> Vec<i32> {
        let mut strata = self.diagnostic_fit.strata.clone();
        strata.sort_unstable();
        strata.dedup();
        if strata.is_empty() { vec![0] } else { strata }
    }

    fn validate_prediction_rows(&self, rows: &[Vec<f64>], width: usize) -> PyResult<()> {
        for (index, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "covariates must have {width} columns"
                )));
            }
            validate_finite(&format!("covariates[{index}]"), row)?;
        }
        Ok(())
    }

    fn row_linear_predictor(row: &[f64], beta: &[f64]) -> f64 {
        row.iter()
            .zip(beta)
            .map(|(value, coefficient)| value * coefficient)
            .sum()
    }

    fn survival_curves_with_strata(
        &self,
        covariates: &[Vec<f64>],
        strata: &[i32],
        centered: bool,
        shared_row: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let beta = self.ordinary_coefficients()?;
        if (!shared_row && covariates.len() != strata.len())
            || (shared_row && covariates.len() != 1)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "strata must have one entry per covariate row",
            ));
        }
        self.validate_prediction_rows(covariates, beta.len())?;
        let center = self.baseline_center(centered);
        let (base_times, base_hazards, base_strata) = self.baseline_with_strata(centered, false)?;
        let baseline =
            StratifiedBaselineLookup::from_components(&base_times, &base_hazards, &base_strata);
        let times = baseline.times_for_strata(strata);
        let curves = strata
            .iter()
            .enumerate()
            .map(|(index, &stratum)| {
                let row = if shared_row {
                    &covariates[0]
                } else {
                    &covariates[index]
                };
                let risk = (Self::row_linear_predictor(row, beta) - center)
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                times
                    .iter()
                    .map(|&time| {
                        let hazard = baseline.cumulative_hazard_at(stratum, time);
                        (-(hazard * risk)).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();
        Ok((times, curves))
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
    entry_times: Option<Vec<f64>>,
    entry_order: Option<Vec<usize>>,
    covariates: Array2<f64>,
    groups: Vec<usize>,
    strata_end: Vec<bool>,
    offset: Vec<f64>,
    weights: Vec<f64>,
    theta: f64,
    distribution: FrailtyDistribution,
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
    fn recenter_frailty(&self, frailty: &mut [f64]) {
        let center = match self.distribution {
            FrailtyDistribution::Gaussian => frailty.iter().sum::<f64>() / frailty.len() as f64,
            FrailtyDistribution::Gamma => {
                let maximum = frailty.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                maximum
                    + (frailty
                        .iter()
                        .map(|value| (value - maximum).exp())
                        .sum::<f64>()
                        / frailty.len() as f64)
                        .ln()
            }
            FrailtyDistribution::StudentT(degrees_of_freedom) => {
                let denominator = self.theta * (degrees_of_freedom - 2.0);
                let (first_sum, second_sum) =
                    frailty
                        .iter()
                        .fold((0.0, 0.0), |(first_sum, second_sum), &value| {
                            let (first, second) = student_t_location_terms(value, denominator);
                            (first_sum + first, second_sum + second)
                        });
                first_sum / second_sum
            }
        };
        for value in frailty {
            *value -= center;
        }
    }

    fn frailty_penalty_second(&self, frailty: &[f64]) -> Vec<f64> {
        let precision = 1.0 / self.theta;
        match self.distribution {
            FrailtyDistribution::Gaussian => vec![precision; frailty.len()],
            FrailtyDistribution::Gamma => frailty
                .iter()
                .map(|value| value.exp() * precision)
                .collect(),
            FrailtyDistribution::StudentT(degrees_of_freedom) => {
                let denominator = self.theta * (degrees_of_freedom - 2.0);
                let scale = (degrees_of_freedom + 1.0) / denominator;
                frailty
                    .iter()
                    .map(|value| {
                        let (_, second) = student_t_location_terms(*value, denominator);
                        scale * second
                    })
                    .collect()
            }
        }
    }

    fn update_risk_moments(
        &self,
        person: usize,
        risk: f64,
        direction: f64,
        denominator: &mut f64,
        risk_sum: &mut [f64],
        cross_sum: &mut Array2<f64>,
    ) {
        let nfrail = risk_sum.len() - self.covariates.ncols();
        let adjusted_risk = direction * risk;
        let group = self.groups[person];
        *denominator += adjusted_risk;
        risk_sum[group] += adjusted_risk;
        for variable in 0..self.covariates.ncols() {
            let value = self.covariates[(person, variable)];
            let weighted = adjusted_risk * value;
            risk_sum[nfrail + variable] += weighted;
            cross_sum[(variable, group)] += weighted;
            for other in 0..=variable {
                cross_sum[(variable, nfrail + other)] +=
                    weighted * self.covariates[(person, other)];
            }
        }
    }

    fn partial_log_likelihood(&self, beta: &[f64], frailty: &[f64]) -> f64 {
        self.iteration(beta, frailty).penalized_log_likelihood + self.penalty_value(beta, frailty)
    }

    fn frailty_penalty_value(&self, frailty: &[f64]) -> f64 {
        match self.distribution {
            FrailtyDistribution::Gaussian => {
                0.5 * frailty
                    .iter()
                    .map(|value| {
                        value * value / self.theta + (2.0 * std::f64::consts::PI * self.theta).ln()
                    })
                    .sum::<f64>()
            }
            FrailtyDistribution::Gamma => -frailty.iter().sum::<f64>() / self.theta,
            FrailtyDistribution::StudentT(degrees_of_freedom) => {
                let denominator = self.theta * (degrees_of_freedom - 2.0);
                let constant = 0.5 * (std::f64::consts::PI * denominator).ln()
                    + ln_gamma(degrees_of_freedom / 2.0)
                    - ln_gamma((degrees_of_freedom + 1.0) / 2.0);
                frailty
                    .iter()
                    .map(|value| {
                        constant
                            + 0.5
                                * (degrees_of_freedom + 1.0)
                                * (1.0 + value * value / denominator).ln()
                    })
                    .sum()
            }
        }
    }

    fn penalty_value(&self, beta: &[f64], frailty: &[f64]) -> f64 {
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
        self.frailty_penalty_value(frailty) + ordinary_penalty
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
        let linear_predictors = (0..self.time.len())
            .map(|person| {
                let mut value = self.offset[person] + frailty[self.groups[person]];
                for (variable, coefficient) in beta.iter().enumerate() {
                    value += coefficient * self.covariates[(person, variable)];
                }
                value
            })
            .collect::<Vec<_>>();
        let risks = linear_predictors
            .iter()
            .zip(&self.weights)
            .map(|(linear_predictor, weight)| {
                linear_predictor.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp() * weight
            })
            .collect::<Vec<_>>();

        while stratum_start < self.time.len() {
            let mut stratum_end = stratum_start;
            while !self.strata_end[stratum_end] {
                stratum_end += 1;
            }
            let mut denominator = 0.0;
            risk_sum.fill(0.0);
            cross_sum.fill(0.0);
            let mut entry_ptr = stratum_start;

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
                    let linear_predictor = linear_predictors[person];
                    let risk = risks[person];
                    self.update_risk_moments(
                        person,
                        risk,
                        1.0,
                        &mut denominator,
                        &mut risk_sum,
                        &mut cross_sum,
                    );

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
                if let (Some(entry_times), Some(entry_order)) =
                    (self.entry_times.as_ref(), self.entry_order.as_ref())
                {
                    while entry_ptr <= stratum_end
                        && entry_times[entry_order[entry_ptr]] >= event_time
                    {
                        let person = entry_order[entry_ptr];
                        self.update_risk_moments(
                            person,
                            risks[person],
                            -1.0,
                            &mut denominator,
                            &mut risk_sum,
                            &mut cross_sum,
                        );
                        entry_ptr += 1;
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
        match self.distribution {
            FrailtyDistribution::Gaussian => {
                for group in 0..nfrail {
                    score[group] -= frailty[group] * precision;
                    group_diagonal[group] += precision;
                }
            }
            FrailtyDistribution::Gamma => {
                for group in 0..nfrail {
                    let relative_risk = frailty[group].exp();
                    score[group] -= (relative_risk - 1.0) * precision;
                    group_diagonal[group] += relative_risk * precision;
                }
            }
            FrailtyDistribution::StudentT(degrees_of_freedom) => {
                let denominator = self.theta * (degrees_of_freedom - 2.0);
                let scale = (degrees_of_freedom + 1.0) / denominator;
                for group in 0..nfrail {
                    let (first, second) = student_t_location_terms(frailty[group], denominator);
                    score[group] -= scale * first;
                    group_diagonal[group] += scale * second;
                }
            }
        }
        log_likelihood -= self.frailty_penalty_value(frailty);

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
            self.recenter_frailty(&mut frailty);
            let state = self.iteration(&beta, &frailty);
            let new_log_likelihood = state.penalized_log_likelihood;
            let mut factor = state.information.clone();
            let diagonal = state.group_diagonal.clone();
            flag = cholesky3(&mut factor, nfrail, &diagonal, self.tolerance);

            if iteration > 0
                && (new_log_likelihood.abs() < self.eps
                    || ((1.0 - accepted_log_likelihood / new_log_likelihood).abs() <= self.eps
                        && !halving))
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

        self.recenter_frailty(&mut frailty);
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
            &self.frailty_penalty_second(&frailty),
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
    frailty_penalty: &[f64],
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
            naive_covariance[(row, column)] -= (0..nfrail)
                .map(|group| frailty_penalty[group] * cross[(group, row)] * cross[(group, column)])
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

    let mut frailty_df = nfrail as f64
        - frailty_penalty
            .iter()
            .zip(&frailty_variance)
            .map(|(penalty, variance)| penalty * variance)
            .sum::<f64>();
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

fn dense_term_degrees_of_freedom(
    covariance: &Array2<f64>,
    naive_covariance: &Array2<f64>,
    columns: &[usize],
) -> f64 {
    if columns.is_empty() {
        return 0.0;
    }
    let width = columns.len();
    let term_covariance = Array2::from_shape_fn((width, width), |(row, column)| {
        covariance[(columns[row], columns[column])]
    });
    let term_naive = Array2::from_shape_fn((width, width), |(row, column)| {
        naive_covariance[(columns[row], columns[column])]
    });
    let Some(inverse) = invert_symmetric(&term_covariance) else {
        return 0.0;
    };
    (0..width)
        .map(|row| {
            (0..width)
                .map(|column| inverse[(row, column)] * term_naive[(column, row)])
                .sum::<f64>()
        })
        .sum::<f64>()
        .clamp(0.0, width as f64)
}

#[allow(clippy::too_many_arguments)]
fn dense_coxph_frailty_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    groups: Vec<usize>,
    theta: f64,
    strata: Vec<i32>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    initial_beta: Vec<f64>,
    max_iter: usize,
    eps: f64,
    toler: f64,
    ties: Ties,
    nocenter: Vec<f64>,
    ordinary_penalty: Array2<f64>,
    entry_times: Option<Vec<f64>>,
    distribution: FrailtyDistribution,
    frailty_columns: Vec<usize>,
) -> PyResult<CoxPHFrailtyFit> {
    let n = time.len();
    let nvar = covariates.first().map_or(0, Vec::len);
    let mut seen_columns = vec![false; nvar];
    for (group, &column) in frailty_columns.iter().enumerate() {
        if column >= nvar || seen_columns[column] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frailty_columns must contain unique valid covariate indices",
            ));
        }
        seen_columns[column] = true;
        for (row, values) in covariates.iter().enumerate() {
            let expected = usize::from(groups[row] == group) as f64;
            if values[column] != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "dense frailty covariates must be full one-hot group indicators",
                ));
            }
        }
    }
    let penalty_distribution = match distribution {
        FrailtyDistribution::Gamma => CoxFrailtyPenalty::Gamma,
        FrailtyDistribution::StudentT(degrees_of_freedom) => {
            CoxFrailtyPenalty::StudentT(degrees_of_freedom)
        }
        FrailtyDistribution::Gaussian => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dense grouped fitting is only needed for gamma and Student-t frailty",
            ));
        }
    };

    let mut order = (0..n).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        strata[left]
            .cmp(&strata[right])
            .then_with(|| time[left].total_cmp(&time[right]))
            .then_with(|| left.cmp(&right))
    });
    let sorted_time = order.iter().map(|&index| time[index]).collect::<Vec<_>>();
    let sorted_status = order.iter().map(|&index| status[index]).collect::<Vec<_>>();
    let sorted_offset = order.iter().map(|&index| offset[index]).collect::<Vec<_>>();
    let sorted_weights = order
        .iter()
        .map(|&index| weights[index])
        .collect::<Vec<_>>();
    let sorted_entry_times = entry_times
        .as_ref()
        .map(|values| order.iter().map(|&index| values[index]).collect::<Vec<_>>());
    let mut strata_boundaries = vec![0; n];
    for sorted in 0..n {
        if sorted + 1 == n || strata[order[sorted + 1]] != strata[order[sorted]] {
            strata_boundaries[sorted] = 1;
        }
    }
    let sorted_covariates =
        Array2::from_shape_fn((n, nvar), |(row, column)| covariates[order[row]][column]);
    let doscale = (0..nvar)
        .map(|column| {
            !covariates
                .iter()
                .all(|row| nocenter.iter().any(|value| row[column] == *value))
        })
        .collect::<Vec<_>>();
    let method = match ties {
        Ties::Breslow => CoxMethod::Breslow,
        Ties::Efron => CoxMethod::Efron,
    };
    let mut cox_fit = CoxFit::new_with_entry_times(
        Array1::from_vec(sorted_time),
        Array1::from_vec(sorted_status),
        sorted_covariates,
        sorted_entry_times.map(Array1::from_vec),
        Array1::from_vec(strata_boundaries),
        Array1::from_vec(sorted_offset),
        Array1::from_vec(sorted_weights),
        method,
        max_iter,
        eps,
        toler,
        doscale,
        initial_beta,
    )
    .map_err(|error| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "dense frailty fit initialization failed: {error}"
        ))
    })?;
    cox_fit.set_frailty_penalty(
        &ordinary_penalty,
        frailty_columns.clone(),
        theta,
        penalty_distribution,
    );
    cox_fit.fit().map_err(|error| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("dense frailty fit failed: {error}"))
    })?;
    let penalized_log_likelihood = cox_fit.penalized_log_likelihood();
    let penalty = cox_fit.penalty_hessian();
    let (beta, means, score, covariance, log_likelihood, score_test, flag, iterations) =
        cox_fit.results();
    let covariance_penalty = covariance.dot(&penalty);
    let naive_covariance = &covariance - &covariance_penalty.dot(&covariance);
    let frailty_degrees_of_freedom =
        dense_term_degrees_of_freedom(&covariance, &naive_covariance, &frailty_columns);
    let covariate_degrees_of_freedom = (0..nvar)
        .map(|column| {
            if seen_columns[column] || covariance[(column, column)] <= 0.0 {
                0.0
            } else {
                (naive_covariance[(column, column)] / covariance[(column, column)]).clamp(0.0, 1.0)
            }
        })
        .collect::<Vec<_>>();
    let degrees_of_freedom =
        covariate_degrees_of_freedom.iter().sum::<f64>() + frailty_degrees_of_freedom;
    let frailty = frailty_columns
        .iter()
        .map(|&column| beta[column])
        .collect::<Vec<_>>();
    let frailty_variance = frailty_columns
        .iter()
        .map(|&column| covariance[(column, column)])
        .collect::<Vec<_>>();
    let linear_predictors = covariates
        .iter()
        .zip(&offset)
        .map(|(row, offset)| {
            offset
                + row
                    .iter()
                    .zip(&beta)
                    .map(|(value, coefficient)| value * coefficient)
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let covariance = covariance
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let naive_information_matrix = naive_covariance
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let penalty_matrix = penalty
        .outer_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let method = match ties {
        Ties::Breslow => "breslow",
        Ties::Efron => "efron",
    }
    .to_string();
    let diagnostic_fit = CoxPHFit {
        coefficients: vec![beta],
        means,
        score_vector: score,
        information_matrix: covariance,
        degrees_of_freedom,
        log_likelihood: log_likelihood.to_vec(),
        score_test,
        convergence_flag: flag,
        iterations,
        risk_scores: Vec::new(),
        event_times: time,
        status,
        linear_predictors,
        entry_times,
        weights,
        covariates,
        strata,
        method,
        nocenter,
    };
    Ok(CoxPHFrailtyFit {
        frailty,
        naive_information_matrix,
        frailty_variance,
        covariate_degrees_of_freedom,
        frailty_degrees_of_freedom,
        penalized_log_likelihood,
        theta,
        distribution: match distribution {
            FrailtyDistribution::Gamma => "gamma",
            FrailtyDistribution::StudentT(_) => "t",
            FrailtyDistribution::Gaussian => unreachable!(),
        }
        .to_string(),
        tdf: match distribution {
            FrailtyDistribution::StudentT(degrees_of_freedom) => Some(degrees_of_freedom),
            FrailtyDistribution::Gamma => None,
            FrailtyDistribution::Gaussian => unreachable!(),
        },
        penalty_matrix,
        dense: true,
        frailty_columns,
        offset,
        diagnostic_fit,
    })
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, groups, theta, strata=None, weights=None, offset=None, initial_beta=None, initial_frailty=None, max_iter=None, eps=None, toler=None, method=None, nocenter=None, penalty_matrix=None, entry_times=None, distribution=None, tdf=None, dense=false, frailty_columns=None))]
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
    entry_times: Option<Vec<f64>>,
    distribution: Option<&str>,
    tdf: Option<f64>,
    dense: bool,
    frailty_columns: Option<Vec<usize>>,
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
    check_optional("entry_times", entry_times.as_ref().map(Vec::len))?;
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
    if let Some(values) = entry_times.as_ref() {
        validate_finite("entry_times", values)?;
        for (index, (&start, &stop)) in values.iter().zip(&time).enumerate() {
            if start >= stop {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "entry_times[{index}] must be less than time[{index}]"
                )));
            }
        }
    }
    let beta = initial_beta.unwrap_or_else(|| vec![0.0; nvar]);
    if beta.len() != nvar {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_beta has {} values but covariates has {nvar} columns",
            beta.len()
        )));
    }
    validate_finite("initial_beta", &beta)?;
    let has_initial_frailty = initial_frailty.is_some();
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
    let distribution = match distribution
        .unwrap_or("gaussian")
        .to_ascii_lowercase()
        .as_str()
    {
        "gaussian" => {
            if tdf.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tdf is only valid for Student-t frailty",
                ));
            }
            FrailtyDistribution::Gaussian
        }
        "gamma" => {
            if tdf.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tdf is only valid for Student-t frailty",
                ));
            }
            FrailtyDistribution::Gamma
        }
        "t" => {
            let degrees_of_freedom = tdf.unwrap_or(5.0);
            if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 2.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "tdf must be a finite value greater than 2",
                ));
            }
            FrailtyDistribution::StudentT(degrees_of_freedom)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distribution must be 'gaussian', 'gamma', or 't'",
            ));
        }
    };
    let strata = strata.unwrap_or_else(|| vec![0; n]);
    let weights = weights.unwrap_or_else(|| vec![1.0; n]);
    let offset = offset.unwrap_or_else(|| vec![0.0; n]);
    let nocenter = nocenter.unwrap_or_else(|| vec![-1.0, 0.0, 1.0]);
    validate_finite("nocenter", &nocenter)?;
    let ordinary_penalty = validate_penalty(penalty_matrix, nvar)?;

    if dense {
        if has_initial_frailty {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_frailty is not used for a dense frailty fit; include effects in initial_beta",
            ));
        }
        let frailty_columns = frailty_columns.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "frailty_columns is required for a dense frailty fit",
            )
        })?;
        if frailty_columns.len() != nfrail {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frailty_columns must contain one column per observed group",
            ));
        }
        return dense_coxph_frailty_fit(
            time,
            status,
            covariates,
            groups,
            theta,
            strata,
            weights,
            offset,
            beta,
            max_iter.unwrap_or(COX_MAX_ITER),
            eps,
            toler,
            ties,
            nocenter,
            ordinary_penalty,
            entry_times,
            distribution,
            frailty_columns,
        );
    }
    if frailty_columns.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "frailty_columns is only valid for a dense frailty fit",
        ));
    }

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
    let sorted_entry_times = entry_times
        .as_ref()
        .map(|values| order.iter().map(|&index| values[index]).collect::<Vec<_>>());
    let entry_order = sorted_entry_times.as_ref().map(|values| {
        let mut positions = (0..n).collect::<Vec<_>>();
        positions.sort_by(|&left, &right| {
            strata[order[left]]
                .cmp(&strata[order[right]])
                .then_with(|| values[right].total_cmp(&values[left]))
                .then_with(|| left.cmp(&right))
        });
        positions
    });
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
        entry_times: sorted_entry_times,
        entry_order,
        covariates: sorted_covariates,
        groups: sorted_groups,
        strata_end,
        offset: sorted_offset,
        weights: sorted_weights,
        theta,
        distribution,
        ordinary_penalty,
        ties,
        max_iter: max_iter.unwrap_or(COX_MAX_ITER),
        eps,
        tolerance: toler,
    };
    let result = solver.solve(means, beta, frailty);
    let mut centered_linear_predictors = Vec::with_capacity(n);
    for row in 0..n {
        let linear_predictor = offset[row]
            + result.frailty[groups[row]]
            + covariates[row]
                .iter()
                .zip(&result.means)
                .zip(&result.beta)
                .map(|((&value, &mean), &coefficient)| (value - mean) * coefficient)
                .sum::<f64>();
        centered_linear_predictors.push(linear_predictor);
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
    let ordinary_center = result
        .means
        .iter()
        .zip(&result.beta)
        .map(|(mean, coefficient)| mean * coefficient)
        .sum::<f64>();
    let linear_predictors = centered_linear_predictors
        .iter()
        .map(|value| value + ordinary_center)
        .collect::<Vec<_>>();
    let method = match ties {
        Ties::Breslow => "breslow",
        Ties::Efron => "efron",
    }
    .to_string();
    let diagnostic_fit = CoxPHFit {
        coefficients: vec![result.beta],
        means: result.means,
        score_vector: result.score,
        information_matrix: covariance,
        degrees_of_freedom,
        log_likelihood: vec![result.initial_log_likelihood, result.final_log_likelihood],
        score_test: result.score_test,
        convergence_flag: result.flag,
        iterations: result.iterations,
        risk_scores: Vec::new(),
        event_times: time,
        status,
        linear_predictors,
        entry_times,
        weights,
        covariates,
        strata,
        method,
        nocenter,
    };

    Ok(CoxPHFrailtyFit {
        frailty: result.frailty,
        naive_information_matrix: naive_covariance,
        frailty_variance: result.frailty_variance,
        covariate_degrees_of_freedom: result.covariate_df,
        frailty_degrees_of_freedom: result.frailty_df,
        penalized_log_likelihood: result.penalized_log_likelihood,
        theta,
        distribution: match distribution {
            FrailtyDistribution::Gaussian => "gaussian",
            FrailtyDistribution::Gamma => "gamma",
            FrailtyDistribution::StudentT(_) => "t",
        }
        .to_string(),
        tdf: match distribution {
            FrailtyDistribution::StudentT(degrees_of_freedom) => Some(degrees_of_freedom),
            FrailtyDistribution::Gaussian | FrailtyDistribution::Gamma => None,
        },
        penalty_matrix: Vec::new(),
        dense: false,
        frailty_columns: Vec::new(),
        offset,
        diagnostic_fit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_penalty_derivatives_match_finite_differences() {
        let theta = 0.7;
        let degrees_of_freedom = 5.0;
        let denominator = theta * (degrees_of_freedom - 2.0);
        let scale = (degrees_of_freedom + 1.0) / denominator;
        let value = 0.4;
        let first_step = 1e-5;
        let second_step = 1e-4;
        let penalty = |point: f64| {
            0.5 * (degrees_of_freedom + 1.0) * (1.0 + point * point / denominator).ln()
        };
        let (first, second) = student_t_location_terms(value, denominator);
        let numeric_first =
            (penalty(value + first_step) - penalty(value - first_step)) / (2.0 * first_step);
        let numeric_second = (penalty(value + second_step) - 2.0 * penalty(value)
            + penalty(value - second_step))
            / (second_step * second_step);

        assert!((scale * first - numeric_first).abs() < 1e-10);
        assert!((scale * second - numeric_second).abs() < 1e-6);
    }

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

    #[test]
    fn counting_process_fit_matches_reference() {
        let fit = coxph_frailty_fit(
            (1..=18).map(f64::from).collect(),
            vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [
                1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1.0, 0.6, 1.7, 0.1, 1.3,
                0.8, 1.6,
            ]
            .into_iter()
            .map(|value| vec![value])
            .collect(),
            (0..18).map(|value| value % 6).collect(),
            0.5,
            None,
            None,
            None,
            None,
            None,
            Some(50),
            Some(1e-12),
            Some(1e-13),
            Some("breslow"),
            None,
            None,
            Some(vec![
                0.0, 0.0, 0.5, 1.0, 0.0, 2.0, 3.0, 1.0, 4.0, 2.0, 6.0, 5.0, 7.0, 8.0, 6.0, 10.0,
                11.0, 9.0,
            ]),
            None,
            None,
            false,
            None,
        )
        .expect("counting-process frailty fit should compute");

        assert!((fit.coefficients()[0][0] - -0.736602816003447).abs() < 1e-12);
        assert!((fit.frailty_degrees_of_freedom - 2.25629339890398).abs() < 1e-12);
        assert!((fit.log_likelihood()[1] - -17.1661166752336).abs() < 1e-12);
    }

    #[test]
    fn fixed_gamma_fit_matches_reference() {
        let fit = coxph_frailty_fit(
            (1..=18).map(f64::from).collect(),
            vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            [
                1.2, 0.7, 1.5, 0.2, 1.1, 0.4, 1.8, 0.9, 0.5, 1.4, 0.3, 1.0, 0.6, 1.7, 0.1, 1.3,
                0.8, 1.6,
            ]
            .into_iter()
            .map(|value| vec![value])
            .collect(),
            (0..18).map(|value| value % 6).collect(),
            0.5,
            None,
            None,
            None,
            None,
            None,
            Some(50),
            Some(1e-12),
            Some(1e-13),
            Some("breslow"),
            None,
            None,
            None,
            Some("gamma"),
            None,
            false,
            None,
        )
        .expect("fixed gamma frailty fit should compute");

        assert!(
            (fit.coefficients()[0][0] - -0.772237352873988).abs() < 1e-12,
            "{:?} {:?} {:?} {:?}",
            fit.coefficients(),
            fit.frailty,
            fit.frailty_degrees_of_freedom,
            fit.log_likelihood()
        );
        assert!((fit.frailty[1] - 0.320013468117002).abs() < 1e-12);
        assert!((fit.frailty_degrees_of_freedom - 2.1898681329284).abs() < 1e-12);
        assert!((fit.log_likelihood()[1] - -22.045263541983).abs() < 1e-12);
    }

    #[test]
    fn dense_gamma_fit_matches_full_coefficient_reference() {
        let groups = (0..6)
            .flat_map(|group| std::iter::repeat_n(group, 3))
            .collect::<Vec<_>>();
        let x = [
            -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, -1.4, -0.9,
            0.1, 0.9,
        ];
        let covariates = x
            .into_iter()
            .enumerate()
            .map(|(row, value)| {
                let mut values = vec![value];
                values.extend((0..6).map(|group| if groups[row] == group { 1.0 } else { 0.0 }));
                values
            })
            .collect::<Vec<_>>();
        let fit = coxph_frailty_fit(
            (2..=19).map(f64::from).collect(),
            vec![1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
            covariates,
            groups,
            0.5,
            None,
            None,
            None,
            None,
            None,
            Some(50),
            Some(1e-10),
            Some(1e-13),
            Some("breslow"),
            None,
            None,
            None,
            Some("gamma"),
            None,
            true,
            Some((1..7).collect()),
        )
        .expect("dense gamma frailty fit should compute");

        assert!(fit.dense);
        assert_eq!(fit.frailty_columns, (1..7).collect::<Vec<_>>());
        assert_eq!(fit.coefficients()[0].len(), 7);
        assert!((fit.coefficients()[0][0] - -0.180646448984096).abs() < 1e-12);
        assert!((fit.coefficients()[0][1] - 0.543042196697236).abs() < 1e-12);
        assert!((fit.information_matrix()[0][0] - 0.131919318665047).abs() < 1e-12);
        assert!((fit.frailty_degrees_of_freedom - 2.15598303424186).abs() < 1e-12);
        assert!((fit.log_likelihood()[1] - -17.9422229283244).abs() < 1e-12);
    }
}
