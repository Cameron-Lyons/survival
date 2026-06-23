use std::collections::BTreeMap;

use crate::constants::{DIVISION_FLOOR, exp_clamped};
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

        let n_strata = strata_groups.len() as f64;

        self.coefficients = vec![0.0; num_covariates];
        self.iterations = 0;
        self.converged = false;

        // Use a stable gradient ascent step on the conditional likelihood
        // within each stratum so matched-set membership actually affects the fit.
        let learning_rate = 0.1;
        while self.iterations < self.max_iter {
            let mut gradient = vec![0.0; num_covariates];

            for indices in strata_groups.values() {
                let mut linear_predictors = Vec::with_capacity(indices.len());
                let mut covariate_rows = Vec::with_capacity(indices.len());
                let mut case_count = 0_usize;
                let mut case_sums = vec![0.0; num_covariates];

                for &observation in indices {
                    let covariates = self.data.get_covariates(observation);
                    let case_status = self.data.get_case_control_status(observation) as usize;
                    case_count += case_status;

                    for (sum, value) in case_sums.iter_mut().zip(covariates.iter()) {
                        *sum += case_status as f64 * value;
                    }

                    let linear_predictor: f64 = self
                        .coefficients
                        .iter()
                        .zip(covariates.iter())
                        .map(|(coef, cov)| coef * cov)
                        .sum();
                    linear_predictors.push(linear_predictor);
                    covariate_rows.push(covariates.as_slice());
                }

                let expected_sums = exact_subset_expected_sums(
                    &covariate_rows,
                    &linear_predictors,
                    case_count,
                    num_covariates,
                )?;

                for covariate_idx in 0..num_covariates {
                    gradient[covariate_idx] +=
                        case_sums[covariate_idx] - expected_sums[covariate_idx];
                }
            }

            let mut max_change = 0.0_f64;
            for (coefficient, gradient_component) in self.coefficients.iter_mut().zip(gradient) {
                let delta = (learning_rate * gradient_component / n_strata).clamp(-0.5, 0.5);
                *coefficient += delta;
                max_change = max_change.max(delta.abs());
            }

            self.iterations += 1;
            if max_change < self.tol {
                self.converged = true;
                break;
            }
        }

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

fn exact_subset_expected_sums(
    covariates: &[&[f64]],
    linear_predictors: &[f64],
    selected_count: usize,
    num_covariates: usize,
) -> PyResult<Vec<f64>> {
    debug_assert_eq!(covariates.len(), linear_predictors.len());
    debug_assert!(selected_count > 0);
    debug_assert!(selected_count < covariates.len());

    let max_linear_predictor = linear_predictors
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut subset_weights = vec![0.0; selected_count + 1];
    let mut weighted_sums = vec![0.0; (selected_count + 1) * num_covariates];
    subset_weights[0] = 1.0;

    for (observation_idx, covariate_row) in covariates.iter().enumerate() {
        let weight = exp_clamped(linear_predictors[observation_idx] - max_linear_predictor);
        let max_k = selected_count.min(observation_idx + 1);

        for k in (1..=max_k).rev() {
            let previous_weight = subset_weights[k - 1];
            if previous_weight <= 0.0 {
                continue;
            }

            let added_weight = previous_weight * weight;
            subset_weights[k] += added_weight;

            let current_offset = k * num_covariates;
            let previous_offset = (k - 1) * num_covariates;
            for covariate_idx in 0..num_covariates {
                weighted_sums[current_offset + covariate_idx] += weight
                    * (weighted_sums[previous_offset + covariate_idx]
                        + previous_weight * covariate_row[covariate_idx]);
            }
        }

        normalize_conditional_dp(
            &mut subset_weights,
            &mut weighted_sums,
            max_k,
            num_covariates,
        )?;
    }

    let denominator = subset_weights[selected_count];
    if !denominator.is_finite() || denominator <= DIVISION_FLOOR {
        return Err(value_error(
            "conditional likelihood denominator is numerically unstable",
        ));
    }

    let expected_offset = selected_count * num_covariates;
    let expected_sums = weighted_sums[expected_offset..expected_offset + num_covariates]
        .iter()
        .map(|value| value / denominator)
        .collect::<Vec<_>>();
    if expected_sums.iter().any(|value| !value.is_finite()) {
        return Err(value_error(
            "conditional likelihood expectation is numerically unstable",
        ));
    }

    Ok(expected_sums)
}

fn normalize_conditional_dp(
    subset_weights: &mut [f64],
    weighted_sums: &mut [f64],
    max_k: usize,
    num_covariates: usize,
) -> PyResult<()> {
    let mut max_abs = 0.0_f64;
    for value in subset_weights.iter().take(max_k + 1) {
        if !value.is_finite() {
            return Err(value_error(
                "conditional likelihood weights are numerically unstable",
            ));
        }
        max_abs = max_abs.max(value.abs());
    }

    for k in 0..=max_k {
        let offset = k * num_covariates;
        for value in &weighted_sums[offset..offset + num_covariates] {
            if !value.is_finite() {
                return Err(value_error(
                    "conditional likelihood sums are numerically unstable",
                ));
            }
            max_abs = max_abs.max(value.abs());
        }
    }

    if max_abs > 1e100 {
        for value in subset_weights.iter_mut().take(max_k + 1) {
            *value /= max_abs;
        }
        for k in 0..=max_k {
            let offset = k * num_covariates;
            for value in &mut weighted_sums[offset..offset + num_covariates] {
                *value /= max_abs;
            }
        }
    }

    Ok(())
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
    fn exact_subset_expectation_handles_multiple_cases() {
        let rows = [vec![2.0], vec![3.0], vec![0.0]];
        let covariates = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let linear_predictors = vec![2.0, 3.0, 0.0];

        let expected = exact_subset_expected_sums(&covariates, &linear_predictors, 2, 1).unwrap();

        let w01 = (2.0_f64 + 3.0).exp();
        let w02 = 2.0_f64.exp();
        let w12 = 3.0_f64.exp();
        let denominator = w01 + w02 + w12;
        let manual = (5.0 * w01 + 2.0 * w02 + 3.0 * w12) / denominator;
        let softmax_approximation = 2.0 * (2.0 * 2.0_f64.exp() + 3.0 * 3.0_f64.exp())
            / (2.0_f64.exp() + 3.0_f64.exp() + 1.0);

        assert!((expected[0] - manual).abs() < 1e-12);
        assert!((expected[0] - softmax_approximation).abs() > 1e-2);
    }
}
