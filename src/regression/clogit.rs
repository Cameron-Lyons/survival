use std::collections::BTreeMap;

use crate::constants::exp_clamped;
use crate::internal::matrix::invert_matrix;
use crate::regression::exact_ties::exact_tied_moments;
use ndarray::Array2;
use pyo3::prelude::*;

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct ClogitDataSet {
    case_control_status: Vec<u8>,
    strata: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}
impl Default for ClogitDataSet {
    fn default() -> Self {
        Self::new()
    }
}
#[pymethods]
impl ClogitDataSet {
    #[new]
    pub fn new() -> ClogitDataSet {
        ClogitDataSet {
            case_control_status: Vec::new(),
            strata: Vec::new(),
            covariates: Vec::new(),
        }
    }
    pub fn add_observation(
        &mut self,
        case_control_status: u8,
        stratum: u8,
        covariates: Vec<f64>,
    ) -> PyResult<()> {
        if case_control_status > 1 {
            return Err(value_error("case_control_status values must be 0 or 1"));
        }

        if let Some(expected_covariates) = self.covariates.first().map(Vec::len)
            && covariates.len() != expected_covariates
        {
            return Err(value_error(
                "all observations must have the same number of covariates",
            ));
        }

        if covariates.iter().any(|value| !value.is_finite()) {
            return Err(value_error("covariates must contain only finite values"));
        }

        self.case_control_status.push(case_control_status);
        self.strata.push(stratum);
        self.covariates.push(covariates);
        Ok(())
    }
    pub fn get_num_observations(&self) -> usize {
        self.case_control_status.len()
    }
    pub fn get_num_covariates(&self) -> usize {
        self.covariates.first().map_or(0, Vec::len)
    }
    pub fn __len__(&self) -> usize {
        self.get_num_observations()
    }
    pub fn is_empty(&self) -> bool {
        self.case_control_status.is_empty()
    }
}
impl ClogitDataSet {
    pub(crate) fn get_case_control_status(&self, id: usize) -> u8 {
        self.case_control_status[id]
    }

    pub(crate) fn get_stratum(&self, id: usize) -> u8 {
        self.strata[id]
    }

    pub(crate) fn get_covariates(&self, id: usize) -> &Vec<f64> {
        &self.covariates[id]
    }

    fn validate(&self) -> PyResult<BTreeMap<u8, Vec<usize>>> {
        if self.case_control_status.is_empty() {
            return Err(value_error("at least one observation is required"));
        }

        if self.case_control_status.len() != self.strata.len()
            || self.case_control_status.len() != self.covariates.len()
        {
            return Err(value_error(
                "case_control_status, strata, and covariates must have the same length",
            ));
        }

        let num_covariates = self.get_num_covariates();
        let mut strata_groups: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for observation in 0..self.get_num_observations() {
            let case_status = self.get_case_control_status(observation);
            if case_status > 1 {
                return Err(value_error("case_control_status values must be 0 or 1"));
            }

            let covariates = self.get_covariates(observation);
            if covariates.len() != num_covariates {
                return Err(value_error(
                    "all observations must have the same number of covariates",
                ));
            }
            if covariates.iter().any(|value| !value.is_finite()) {
                return Err(value_error("covariates must contain only finite values"));
            }

            strata_groups
                .entry(self.get_stratum(observation))
                .or_default()
                .push(observation);
        }

        for indices in strata_groups.values() {
            if indices.len() < 2 {
                return Err(value_error(
                    "each stratum must contain at least two observations",
                ));
            }

            let case_count = indices
                .iter()
                .map(|&idx| self.get_case_control_status(idx) as usize)
                .sum::<usize>();
            if case_count == 0 || case_count == indices.len() {
                return Err(value_error(
                    "each stratum must contain at least one case and one control",
                ));
            }
        }

        Ok(strata_groups)
    }
}
#[pyclass]
pub struct ConditionalLogisticRegression {
    data: ClogitDataSet,
    #[pyo3(get)]
    coefficients: Vec<f64>,
    #[pyo3(get, set)]
    max_iter: u32,
    #[pyo3(get, set)]
    tol: f64,
    #[pyo3(get)]
    iterations: u32,
    #[pyo3(get)]
    converged: bool,
}
#[pymethods]
impl ConditionalLogisticRegression {
    #[new]
    #[pyo3(signature = (data, max_iter=100, tol=crate::constants::CLOGIT_TOLERANCE))]
    pub fn new(data: ClogitDataSet, max_iter: u32, tol: f64) -> PyResult<Self> {
        validate_solver_controls(max_iter, tol)?;
        Ok(ConditionalLogisticRegression {
            data,
            coefficients: Vec::new(),
            max_iter,
            tol,
            iterations: 0,
            converged: false,
        })
    }
    pub fn fit(&mut self) -> PyResult<()> {
        validate_solver_controls(self.max_iter, self.tol)?;
        let num_covariates = self.data.get_num_covariates();
        let strata_groups = self.data.validate()?;
        if num_covariates == 0 {
            self.coefficients.clear();
            self.iterations = 0;
            self.converged = true;
            return Ok(());
        }

        let n = self.data.get_num_observations();
        let covariates = Array2::from_shape_vec(
            (n, num_covariates),
            self.data.covariates.iter().flatten().copied().collect(),
        )
        .map_err(|error| value_error(format!("invalid covariate matrix: {error}")))?;
        let groups = strata_groups.into_values().collect::<Vec<_>>();
        let mut coefficients = vec![0.0; num_covariates];
        let (mut log_likelihood, mut score, mut information) = conditional_likelihood_moments(
            &coefficients,
            &covariates,
            &self.data.case_control_status,
            &groups,
        )?;
        self.iterations = 0;
        self.converged = false;

        for iteration in 1..=self.max_iter {
            let Some(inverse) = invert_matrix(&information) else {
                break;
            };
            let update = inverse
                .iter()
                .map(|row| {
                    row.iter()
                        .zip(&score)
                        .map(|(&value, &score_value)| value * score_value)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let mut candidate = coefficients
                .iter()
                .zip(&update)
                .map(|(&coefficient, &delta)| coefficient + delta)
                .collect::<Vec<_>>();
            let (mut candidate_ll, mut candidate_score, mut candidate_information) =
                conditional_likelihood_moments(
                    &candidate,
                    &covariates,
                    &self.data.case_control_status,
                    &groups,
                )?;
            let mut halvings = 0;
            while candidate_ll < log_likelihood && halvings < 20 {
                for (value, &coefficient) in candidate.iter_mut().zip(&coefficients) {
                    *value = (*value + coefficient) * 0.5;
                }
                (candidate_ll, candidate_score, candidate_information) =
                    conditional_likelihood_moments(
                        &candidate,
                        &covariates,
                        &self.data.case_control_status,
                        &groups,
                    )?;
                halvings += 1;
            }
            if candidate_ll < log_likelihood {
                break;
            }
            let relative_change = (1.0 - log_likelihood / candidate_ll).abs();
            coefficients = candidate;
            log_likelihood = candidate_ll;
            score = candidate_score;
            information = candidate_information;
            self.iterations = iteration;
            if relative_change <= self.tol {
                self.converged = true;
                break;
            }
        }
        self.coefficients = coefficients;

        Ok(())
    }
    pub fn predict(&self, covariates: Vec<f64>) -> PyResult<f64> {
        self.validate_prediction_row(&covariates)?;
        let exp_sum: f64 = self
            .coefficients
            .iter()
            .zip(covariates.iter())
            .map(|(coef, cov)| coef * cov)
            .sum();
        Ok(exp_clamped(exp_sum))
    }
    pub fn odds_ratios(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|coefficient| exp_clamped(*coefficient))
            .collect()
    }
}

impl ConditionalLogisticRegression {
    fn validate_prediction_row(&self, covariates: &[f64]) -> PyResult<()> {
        if self.coefficients.is_empty() && self.iterations == 0 && !self.converged {
            return Err(value_error("model must be fit before prediction"));
        }
        if covariates.len() != self.coefficients.len() {
            return Err(value_error(format!(
                "prediction row has {} covariates, expected {}",
                covariates.len(),
                self.coefficients.len()
            )));
        }
        if covariates.iter().any(|value| !value.is_finite()) {
            return Err(value_error("prediction covariates must be finite"));
        }
        Ok(())
    }
}

fn validate_solver_controls(max_iter: u32, tol: f64) -> PyResult<()> {
    if max_iter == 0 {
        return Err(value_error("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(value_error("tol must be a positive finite value"));
    }
    Ok(())
}

fn conditional_likelihood_moments(
    coefficients: &[f64],
    covariates: &Array2<f64>,
    case_status: &[u8],
    strata_groups: &[Vec<usize>],
) -> PyResult<(f64, Vec<f64>, Vec<Vec<f64>>)> {
    let nvar = coefficients.len();
    let linear_predictors = covariates
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(coefficients)
                .map(|(&value, &coefficient)| value * coefficient)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let mut log_likelihood = 0.0;
    let mut score = vec![0.0; nvar];
    let mut information = vec![vec![0.0; nvar]; nvar];

    for indices in strata_groups {
        let deaths = indices.iter().filter(|&&idx| case_status[idx] == 1).count();
        let moments = exact_tied_moments(indices, deaths, &linear_predictors, covariates);
        if !moments.log_denom.is_finite()
            || moments.mean.iter().any(|value| !value.is_finite())
            || moments.covariance.iter().any(|value| !value.is_finite())
        {
            return Err(value_error(
                "conditional likelihood moments are numerically unstable",
            ));
        }
        for &idx in indices {
            if case_status[idx] == 0 {
                continue;
            }
            log_likelihood += linear_predictors[idx];
            for variable in 0..nvar {
                score[variable] += covariates[(idx, variable)];
            }
        }
        log_likelihood -= moments.log_denom;
        for (variable, information_row) in information.iter_mut().enumerate() {
            score[variable] -= moments.mean[variable];
            for (other, value) in information_row.iter_mut().enumerate() {
                *value += moments.covariance[(variable, other)];
            }
        }
    }

    Ok((log_likelihood, score, information))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matched_dataset() -> ClogitDataSet {
        let mut dataset = ClogitDataSet::new();
        for (case_status, stratum, covariates) in [
            (1, 0, vec![2.0]),
            (0, 0, vec![1.0]),
            (1, 1, vec![3.0]),
            (0, 1, vec![1.0]),
        ] {
            dataset
                .add_observation(case_status, stratum, covariates)
                .expect("valid matched observation");
        }
        dataset
    }

    #[test]
    fn dataset_rejects_invalid_rows() {
        let mut dataset = ClogitDataSet::new();

        assert!(dataset.add_observation(2, 0, vec![1.0]).is_err());
        assert!(dataset.add_observation(1, 0, vec![f64::NAN]).is_err());

        dataset.add_observation(1, 0, vec![1.0]).unwrap();
        assert!(dataset.add_observation(0, 0, vec![1.0, 2.0]).is_err());
        assert_eq!(dataset.get_num_observations(), 1);
    }

    #[test]
    fn solver_controls_are_validated() {
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 0, 1e-6).is_err());
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 10, 0.0).is_err());
        assert!(ConditionalLogisticRegression::new(matched_dataset(), 10, f64::INFINITY).is_err());
    }

    #[test]
    fn fit_validates_data_before_null_model_short_circuit() {
        let mut dataset = ClogitDataSet::new();
        dataset.add_observation(1, 0, Vec::new()).unwrap();

        let mut model = ConditionalLogisticRegression::new(dataset, 10, 1e-6).unwrap();
        assert!(model.fit().is_err());
    }

    #[test]
    fn prediction_requires_fitted_model_and_matching_finite_row() {
        let mut model = ConditionalLogisticRegression::new(matched_dataset(), 10, 1e-9).unwrap();
        assert!(model.predict(vec![1.0]).is_err());

        model.fit().unwrap();

        assert!(model.predict(Vec::new()).is_err());
        assert!(model.predict(vec![f64::NAN]).is_err());
        assert!(model.predict(vec![1.0]).unwrap().is_finite());
    }

    #[test]
    fn exponentials_are_clamped_for_public_outputs() {
        let mut model = ConditionalLogisticRegression::new(matched_dataset(), 1, 1e-9).unwrap();
        model.coefficients = vec![1_000.0, -1_000.0];
        model.iterations = 1;

        let ratios = model.odds_ratios();

        assert_eq!(ratios.len(), 2);
        assert!(ratios.iter().all(|ratio| ratio.is_finite()));
        assert!(model.predict(vec![1.0, 1.0]).unwrap().is_finite());
    }

    #[test]
    fn exact_fit_matches_multiple_case_reference() {
        let mut dataset = ClogitDataSet::new();
        let case = [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0];
        let strata = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4];
        let x = [
            0.2, 0.8, 1.1, 0.4, 0.5, 1.2, 0.9, 0.1, 0.7, 1.0, 0.3, 1.3, 0.6, 0.2, 1.4, 0.9,
        ];
        let z = [
            1.0, 0.3, 0.8, 1.4, 0.2, 1.1, 0.6, 0.9, 0.5, 1.2, 0.1, 0.7, 1.3, 0.4, 0.8, 0.2,
        ];
        for idx in 0..case.len() {
            dataset
                .add_observation(case[idx], strata[idx], vec![x[idx], z[idx]])
                .unwrap();
        }

        let mut model = ConditionalLogisticRegression::new(dataset, 50, 1e-9).unwrap();
        model.fit().unwrap();

        assert!((model.coefficients[0] - (-0.845_780_876_118_187_9)).abs() <= 1e-10);
        assert!((model.coefficients[1] - 1.476_653_196_461_022_3).abs() <= 1e-10);
        assert_eq!(model.iterations, 3);
        assert!(model.converged);
    }
}
