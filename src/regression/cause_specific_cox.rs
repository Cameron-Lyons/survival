use crate::constants::{exp_ci_bounds_95, exp_clamped};
use crate::internal::validation::{
    validate_finite, validate_no_nan, validate_non_empty, validate_non_negative,
};
use crate::regression::coxph::coxph_fit;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum CensoringType {
    Censored,
    Competing,
}

#[pymethods]
impl CensoringType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "censored" => Ok(CensoringType::Censored),
            "competing" => Ok(CensoringType::Competing),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown censoring type. Use 'censored' or 'competing'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CauseSpecificCoxConfig {
    #[pyo3(get, set)]
    pub cause_of_interest: i32,
    #[pyo3(get, set)]
    pub treat_other_causes_as: CensoringType,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub ties: String,
}

#[pymethods]
impl CauseSpecificCoxConfig {
    #[new]
    #[pyo3(signature = (
        cause_of_interest=1,
        treat_other_causes_as=CensoringType::Censored,
        max_iter=100,
        tol=1e-9,
        ties="breslow"
    ))]
    pub fn new(
        cause_of_interest: i32,
        treat_other_causes_as: CensoringType,
        max_iter: usize,
        tol: f64,
        ties: &str,
    ) -> PyResult<Self> {
        if cause_of_interest < 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cause_of_interest must be >= 1",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        let ties_lower = ties.to_lowercase();
        if ties_lower != "breslow" && ties_lower != "efron" {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ties must be 'breslow' or 'efron'",
            ));
        }

        Ok(CauseSpecificCoxConfig {
            cause_of_interest,
            treat_other_causes_as,
            max_iter,
            tol,
            ties: ties_lower,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CauseSpecificCoxResult {
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_lower: Vec<f64>,
    #[pyo3(get)]
    pub hr_ci_upper: Vec<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub n_events: usize,
    #[pyo3(get)]
    pub n_at_risk: usize,
    #[pyo3(get)]
    pub n_competing: usize,
    #[pyo3(get)]
    pub n_censored: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub cause_of_interest: i32,
    #[pyo3(get)]
    pub baseline_hazard_times: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_baseline_hazard: Vec<f64>,
}

#[pymethods]
impl CauseSpecificCoxResult {
    fn __repr__(&self) -> String {
        format!(
            "CauseSpecificCoxResult(cause={}, n_events={}, converged={})",
            self.cause_of_interest, self.n_events, self.converged
        )
    }

    fn predict_cumulative_hazard(&self, x: Vec<f64>, n_obs: usize) -> PyResult<Vec<Vec<f64>>> {
        let risk_scores = self.prediction_risk_scores(&x, n_obs)?;

        Ok(risk_scores
            .into_par_iter()
            .map(|risk_score| {
                self.cumulative_baseline_hazard
                    .iter()
                    .map(|&hazard| hazard * risk_score)
                    .collect()
            })
            .collect())
    }

    fn predict_survival(&self, x: Vec<f64>, n_obs: usize) -> PyResult<Vec<Vec<f64>>> {
        let cumulative_hazard = self.predict_cumulative_hazard(x, n_obs)?;
        Ok(cumulative_hazard
            .into_par_iter()
            .map(|hazards| {
                hazards
                    .into_iter()
                    .map(|hazard| (-hazard).exp().clamp(0.0, 1.0))
                    .collect()
            })
            .collect())
    }

    fn predict_cif(&self, x: Vec<f64>, n_obs: usize) -> PyResult<Vec<Vec<f64>>> {
        let cumulative_hazard = self.predict_cumulative_hazard(x, n_obs)?;
        Ok(cumulative_hazard
            .into_par_iter()
            .map(|hazards| {
                hazards
                    .into_iter()
                    .map(|hazard| (-(-hazard).exp_m1()).clamp(0.0, 1.0))
                    .collect()
            })
            .collect())
    }
}

impl CauseSpecificCoxResult {
    fn prediction_risk_scores(&self, x: &[f64], n_obs: usize) -> PyResult<Vec<f64>> {
        let n_vars = self.coefficients.len();
        if n_vars == 0 {
            return Err(PyValueError::new_err(
                "cannot predict with a model that has no coefficients",
            ));
        }

        let expected_len = n_obs.checked_mul(n_vars).ok_or_else(|| {
            PyValueError::new_err("n_obs * n_vars overflowed while validating x length")
        })?;
        if x.len() != expected_len {
            return Err(PyValueError::new_err("x length must equal n_obs * n_vars"));
        }
        validate_no_nan(x, "x")?;
        validate_finite(x, "x")?;

        x.par_chunks_exact(n_vars)
            .map(|row| {
                let linear_predictor = row
                    .iter()
                    .zip(&self.coefficients)
                    .map(|(&value, &coefficient)| value * coefficient)
                    .sum::<f64>();
                (!linear_predictor.is_nan()).then(|| exp_clamped(linear_predictor))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| PyValueError::new_err("x produced an undefined linear predictor"))
    }
}

fn validate_cause_specific_config(config: &CauseSpecificCoxConfig) -> PyResult<()> {
    if config.cause_of_interest < 1 {
        return Err(PyValueError::new_err("cause_of_interest must be >= 1"));
    }
    if config.max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    if !config.ties.eq_ignore_ascii_case("breslow") && !config.ties.eq_ignore_ascii_case("efron") {
        return Err(PyValueError::new_err("ties must be 'breslow' or 'efron'"));
    }
    Ok(())
}

pub(crate) fn validate_cause_specific_inputs(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    cause: &[i32],
    weights: Option<&[f64]>,
) -> PyResult<()> {
    if n_obs == 0 {
        return Err(PyValueError::new_err("n_obs must be positive"));
    }
    if n_vars == 0 {
        return Err(PyValueError::new_err("n_vars must be positive"));
    }

    let expected_x_len = n_obs.checked_mul(n_vars).ok_or_else(|| {
        PyValueError::new_err("n_obs * n_vars overflowed while validating x length")
    })?;
    if x.len() != expected_x_len {
        return Err(PyValueError::new_err("x length must equal n_obs * n_vars"));
    }
    if time.len() != n_obs || cause.len() != n_obs {
        return Err(PyValueError::new_err(
            "time and cause must have length n_obs",
        ));
    }

    validate_no_nan(x, "x")?;
    validate_finite(x, "x")?;
    validate_no_nan(time, "time")?;
    validate_finite(time, "time")?;
    validate_non_negative(time, "time")?;
    for (idx, &value) in cause.iter().enumerate() {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "cause must contain non-negative values; got {value} at index {idx}"
            )));
        }
    }

    if let Some(weights) = weights {
        validate_non_empty(weights, "weights")?;
        if weights.len() != n_obs {
            return Err(PyValueError::new_err("weights must have length n_obs"));
        }
        validate_no_nan(weights, "weights")?;
        validate_finite(weights, "weights")?;
        validate_non_negative(weights, "weights")?;
        if weights.iter().all(|&weight| weight == 0.0) {
            return Err(PyValueError::new_err(
                "weights must include at least one positive value",
            ));
        }
    }

    Ok(())
}

#[inline]
fn observation_weight(weights: Option<&[f64]>, idx: usize) -> f64 {
    weights.map_or(1.0, |values| values[idx])
}

fn cause_specific_case_weights(
    cause: &[i32],
    weights: Option<&[f64]>,
    cause_of_interest: i32,
    treat_other_as: CensoringType,
) -> Vec<f64> {
    cause
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            let weight = observation_weight(weights, idx);
            if treat_other_as == CensoringType::Competing && value > 0 && value != cause_of_interest
            {
                0.0
            } else {
                weight
            }
        })
        .collect()
}

fn row_major_covariates(x: &[f64], n_vars: usize) -> Vec<Vec<f64>> {
    x.chunks_exact(n_vars).map(<[f64]>::to_vec).collect()
}

fn hazard_increments(cumulative: &[f64]) -> Vec<f64> {
    let mut previous = 0.0;
    cumulative
        .iter()
        .map(|&value| {
            let increment = value - previous;
            previous = value;
            increment
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, config, weights=None))]
pub fn cause_specific_cox(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    config: &CauseSpecificCoxConfig,
    weights: Option<Vec<f64>>,
) -> PyResult<CauseSpecificCoxResult> {
    validate_cause_specific_inputs(&x, n_obs, n_vars, &time, &cause, weights.as_deref())?;
    validate_cause_specific_config(config)?;

    cause_specific_cox_fit(&x, n_obs, n_vars, &time, &cause, config, weights.as_deref())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cause_specific_cox_fit(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    cause: &[i32],
    config: &CauseSpecificCoxConfig,
    weights: Option<&[f64]>,
) -> PyResult<CauseSpecificCoxResult> {
    debug_assert_eq!(x.len(), n_obs * n_vars);
    debug_assert_eq!(time.len(), n_obs);
    debug_assert_eq!(cause.len(), n_obs);
    debug_assert!(weights.is_none_or(|values| values.len() == n_obs));

    let n_events = cause
        .iter()
        .filter(|&&c| c == config.cause_of_interest)
        .count();
    let n_competing = cause
        .iter()
        .filter(|&&c| c > 0 && c != config.cause_of_interest)
        .count();
    let n_censored = cause.iter().filter(|&&c| c == 0).count();

    let case_weights = cause_specific_case_weights(
        cause,
        weights,
        config.cause_of_interest,
        config.treat_other_causes_as,
    );
    if n_events == 0 || case_weights.iter().all(|&weight| weight == 0.0) {
        let coefficients = vec![0.0; n_vars];
        let std_errors = vec![crate::constants::DIVISION_FLOOR; n_vars];
        let hazard_ratios = vec![1.0; n_vars];
        let (hr_ci_lower, hr_ci_upper) = exp_ci_bounds_95(&coefficients, &std_errors);
        return Ok(CauseSpecificCoxResult {
            coefficients,
            std_errors,
            hazard_ratios,
            hr_ci_lower,
            hr_ci_upper,
            log_likelihood: 0.0,
            n_events,
            n_at_risk: n_obs,
            n_competing,
            n_censored,
            n_iter: 0,
            converged: false,
            cause_of_interest: config.cause_of_interest,
            baseline_hazard_times: Vec::new(),
            baseline_hazard: Vec::new(),
            cumulative_baseline_hazard: Vec::new(),
        });
    }

    let status = cause
        .iter()
        .map(|&value| i32::from(value == config.cause_of_interest))
        .collect();
    let fit = coxph_fit(
        time.to_vec(),
        status,
        row_major_covariates(x, n_vars),
        None,
        Some(case_weights),
        None,
        None,
        Some(config.max_iter),
        Some(config.tol),
        None,
        Some(&config.ties),
        None,
        None,
        None,
        None,
    )?;
    let beta = fit
        .coefficients
        .first()
        .cloned()
        .unwrap_or_else(|| vec![0.0; n_vars]);
    let std_errors = (0..n_vars)
        .map(|column| {
            fit.information_matrix
                .get(column)
                .and_then(|row| row.get(column))
                .copied()
                .unwrap_or(0.0)
                .abs()
                .sqrt()
                .max(crate::constants::DIVISION_FLOOR)
        })
        .collect::<Vec<_>>();
    let hazard_ratios = beta
        .iter()
        .map(|&value| exp_clamped(value))
        .collect::<Vec<_>>();
    let (hr_ci_lower, hr_ci_upper) = exp_ci_bounds_95(&beta, &std_errors);
    let (baseline_times, cum_baseline_hazard) = fit.basehaz(false)?;
    let baseline_hazard = hazard_increments(&cum_baseline_hazard);
    let loglik = fit.log_likelihood.last().copied().unwrap_or(0.0);
    let converged = fit.iterations < config.max_iter && loglik.is_finite();

    Ok(CauseSpecificCoxResult {
        coefficients: beta,
        std_errors,
        hazard_ratios,
        hr_ci_lower,
        hr_ci_upper,
        log_likelihood: loglik,
        n_events,
        n_at_risk: n_obs,
        n_competing,
        n_censored,
        n_iter: fit.iterations,
        converged,
        cause_of_interest: config.cause_of_interest,
        baseline_hazard_times: baseline_times,
        baseline_hazard,
        cumulative_baseline_hazard: cum_baseline_hazard,
    })
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, max_cause, weights=None, max_iter=100, tol=1e-9))]
#[allow(clippy::too_many_arguments)]
pub fn cause_specific_cox_all(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    max_cause: i32,
    weights: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<CauseSpecificCoxResult>> {
    if max_cause < 1 {
        return Err(PyValueError::new_err("max_cause must be >= 1"));
    }
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    validate_cause_specific_inputs(&x, n_obs, n_vars, &time, &cause, weights.as_deref())?;

    let mut results = Vec::with_capacity(max_cause as usize);
    let weights = weights.as_deref();

    for c in 1..=max_cause {
        let config =
            CauseSpecificCoxConfig::new(c, CensoringType::Censored, max_iter, tol, "breslow")?;

        let result = cause_specific_cox_fit(&x, n_obs, n_vars, &time, &cause, &config, weights)?;

        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        if left == right {
            return;
        }

        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to equal {right}"
        );
    }

    fn assert_vec_close(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (&left_value, &right_value) in left.iter().zip(right) {
            assert_close(left_value, right_value);
        }
    }

    fn assert_result_close(left: &CauseSpecificCoxResult, right: &CauseSpecificCoxResult) {
        assert_vec_close(&left.coefficients, &right.coefficients);
        assert_vec_close(&left.std_errors, &right.std_errors);
        assert_vec_close(&left.hazard_ratios, &right.hazard_ratios);
        assert_vec_close(&left.hr_ci_lower, &right.hr_ci_lower);
        assert_vec_close(&left.hr_ci_upper, &right.hr_ci_upper);
        assert_close(left.log_likelihood, right.log_likelihood);
        assert_eq!(left.n_events, right.n_events);
        assert_eq!(left.n_at_risk, right.n_at_risk);
        assert_eq!(left.n_competing, right.n_competing);
        assert_eq!(left.n_censored, right.n_censored);
        assert_eq!(left.n_iter, right.n_iter);
        assert_eq!(left.converged, right.converged);
        assert_eq!(left.cause_of_interest, right.cause_of_interest);
        assert_vec_close(&left.baseline_hazard_times, &right.baseline_hazard_times);
        assert_vec_close(&left.baseline_hazard, &right.baseline_hazard);
        assert_vec_close(
            &left.cumulative_baseline_hazard,
            &right.cumulative_baseline_hazard,
        );
    }

    fn reference_fit(ties: &str) -> CauseSpecificCoxResult {
        let x1 = [
            0.2, -1.0, 0.7, 0.0, -0.6, 1.1, -0.2, 0.5, 1.2, -0.8, 0.3, -1.1, 0.9, -0.4,
        ];
        let x2 = [
            1.0, 0.4, -0.9, 0.2, 0.8, -0.5, 1.1, -0.3, 0.6, 1.3, -0.7, 0.5, -1.2, 0.1,
        ];
        let x = x1
            .iter()
            .zip(x2.iter())
            .flat_map(|(&left, &right)| [left, right])
            .collect();
        let time = vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 8.0, 9.0,
        ];
        let cause = vec![1, 2, 1, 0, 1, 1, 1, 0, 1, 2, 1, 0, 2, 1];
        let weights = vec![
            1.0, 1.2, 0.8, 1.5, 0.9, 1.1, 0.75, 1.3, 1.0, 0.85, 1.25, 0.95, 1.4, 1.05,
        ];
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-9, ties).unwrap();

        cause_specific_cox(x, time.len(), 2, time, cause, &config, Some(weights)).unwrap()
    }

    fn assert_reference_values(
        result: &CauseSpecificCoxResult,
        coefficients: &[f64],
        std_errors: &[f64],
        log_likelihood: f64,
        cumulative_baseline_hazard: &[f64],
    ) {
        let tolerance = 1e-9;
        for (actual, expected) in result.coefficients.iter().zip(coefficients) {
            assert!((actual - expected).abs() <= tolerance);
        }
        for (actual, expected) in result.std_errors.iter().zip(std_errors) {
            assert!((actual - expected).abs() <= tolerance);
        }
        assert!((result.log_likelihood - log_likelihood).abs() <= tolerance);
        assert_eq!(
            result.baseline_hazard_times,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        for (actual, expected) in result
            .cumulative_baseline_hazard
            .iter()
            .zip(cumulative_baseline_hazard)
        {
            assert!((actual - expected).abs() <= tolerance);
        }
        assert!(result.converged);
        assert_eq!(result.n_iter, 4);
    }

    #[test]
    fn native_tied_fits_match_reference_cox_models() {
        assert_reference_values(
            &reference_fit("breslow"),
            &[1.0046774476980949, 0.8108994771525215],
            &[0.5873650496929306, 0.559619090488565],
            -13.884_743_933_880_37,
            &[
                0.04492149122855391,
                0.085_925_229_914_249_8,
                0.13558418931321614,
                0.25557984346508395,
                0.3410067911069257,
                0.613319626711523,
                0.613319626711523,
                0.613319626711523,
                1.9915233911638635,
            ],
        );
        assert_reference_values(
            &reference_fit("efron"),
            &[1.0029646789269757, 0.8090613798416071],
            &[0.5826568579896737, 0.5568223789788628],
            -13.766109932025635,
            &[
                0.04497616447743583,
                0.08602423033317065,
                0.1357435932796217,
                0.2641243522003811,
                0.34967045633704197,
                0.6219041739263703,
                0.6219041739263703,
                0.6219041739263703,
                1.9994172210368464,
            ],
        );
    }

    #[test]
    fn test_config() {
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-9, "breslow").unwrap();
        assert_eq!(config.cause_of_interest, 1);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            CauseSpecificCoxConfig::new(0, CensoringType::Censored, 100, 1e-9, "breslow").is_err()
        );
        assert!(
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 0, 1e-9, "breslow").is_err()
        );
        assert!(
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-9, "invalid").is_err()
        );
    }

    #[test]
    fn test_cause_specific_cox_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();

        let result = cause_specific_cox(x, 5, 2, time, cause, &config, None).unwrap();
        assert_eq!(result.coefficients.len(), 2);
        assert_eq!(result.n_events, 2);
        assert_eq!(result.n_competing, 2);
        assert_eq!(result.n_censored, 1);
    }

    #[test]
    fn test_cause_specific_cox_unweighted_matches_unit_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();

        let unweighted =
            cause_specific_cox(x.clone(), 5, 2, time.clone(), cause.clone(), &config, None)
                .unwrap();
        let unit_weighted =
            cause_specific_cox(x, 5, 2, time, cause, &config, Some(vec![1.0; 5])).unwrap();

        assert_result_close(&unweighted, &unit_weighted);
    }

    #[test]
    fn test_cause_specific_cox_all_keeps_requested_causes() {
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let time = vec![1.0, 2.0];
        let cause = vec![1, 0];

        let results = cause_specific_cox_all(x, 2, 2, time, cause, 3, None, 5, 1e-5).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].n_events, 1);
        assert_eq!(results[1].n_events, 0);
        assert_eq!(results[2].n_events, 0);
    }

    #[test]
    fn test_cause_specific_cox_all_unweighted_matches_unit_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];

        let unweighted = cause_specific_cox_all(
            x.clone(),
            5,
            2,
            time.clone(),
            cause.clone(),
            2,
            None,
            100,
            1e-5,
        )
        .unwrap();
        let unit_weighted =
            cause_specific_cox_all(x, 5, 2, time, cause, 2, Some(vec![1.0; 5]), 100, 1e-5).unwrap();

        assert_eq!(unweighted.len(), unit_weighted.len());
        for (left, right) in unweighted.iter().zip(&unit_weighted) {
            assert_result_close(left, right);
        }
    }

    #[test]
    fn test_cause_specific_cox_validates_public_inputs() {
        pyo3::Python::initialize();
        let config =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();

        let err = cause_specific_cox(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, -1],
            &config,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("cause must contain non-negative"));

        let err = cause_specific_cox(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, f64::INFINITY],
            vec![1, 0],
            &config,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("time contains non-finite"));

        let err = cause_specific_cox(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 0],
            &config,
            Some(vec![1.0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("weights must have length n_obs"));

        let err = cause_specific_cox_all(
            vec![1.0, 2.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 0],
            0,
            None,
            100,
            1e-5,
        )
        .unwrap_err();
        assert!(err.to_string().contains("max_cause must be >= 1"));
    }

    #[test]
    fn test_competing_censoring_type() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];

        let config_censored =
            CauseSpecificCoxConfig::new(1, CensoringType::Censored, 100, 1e-5, "breslow").unwrap();
        let config_competing =
            CauseSpecificCoxConfig::new(1, CensoringType::Competing, 100, 1e-5, "breslow").unwrap();

        let result_censored = cause_specific_cox(
            x.clone(),
            5,
            2,
            time.clone(),
            cause.clone(),
            &config_censored,
            None,
        )
        .unwrap();
        let result_competing =
            cause_specific_cox(x, 5, 2, time, cause, &config_competing, None).unwrap();

        assert_eq!(result_censored.n_events, result_competing.n_events);
    }
}
