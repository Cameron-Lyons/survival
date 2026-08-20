use crate::constants::{exp_clamped, same_time};
use crate::internal::validation::{validate_finite, validate_no_nan};
use crate::regression::cause_specific_cox_module::{
    CauseSpecificCoxConfig, CensoringType, cause_specific_cox_fit, validate_cause_specific_inputs,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(from_py_object)]
pub enum CorrelationType {
    Independent,
    SharedFrailty,
    CopulaBased,
}

#[pymethods]
impl CorrelationType {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "independent" => Ok(CorrelationType::Independent),
            "shared_frailty" | "sharedfrailty" | "frailty" => Ok(CorrelationType::SharedFrailty),
            "copula_based" | "copulabased" | "copula" => Ok(CorrelationType::CopulaBased),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unknown correlation type. Use 'independent', 'shared_frailty', or 'copula_based'",
            )),
        }
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct JointCompetingRisksConfig {
    #[pyo3(get, set)]
    pub num_causes: usize,
    #[pyo3(get, set)]
    pub correlation_structure: CorrelationType,
    #[pyo3(get, set)]
    pub frailty_variance: f64,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub estimate_correlation: bool,
}

#[pymethods]
impl JointCompetingRisksConfig {
    #[new]
    #[pyo3(signature = (
        num_causes=2,
        correlation_structure=CorrelationType::Independent,
        frailty_variance=1.0,
        max_iter=100,
        tol=1e-6,
        estimate_correlation=true
    ))]
    pub fn new(
        num_causes: usize,
        correlation_structure: CorrelationType,
        frailty_variance: f64,
        max_iter: usize,
        tol: f64,
        estimate_correlation: bool,
    ) -> PyResult<Self> {
        if num_causes < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_causes must be at least 2",
            ));
        }
        if !frailty_variance.is_finite() || frailty_variance <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "frailty_variance must be finite and positive",
            ));
        }
        if max_iter == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_iter must be positive",
            ));
        }
        if !tol.is_finite() || tol <= 0.0 {
            return Err(PyValueError::new_err("tol must be finite and positive"));
        }

        Ok(JointCompetingRisksConfig {
            num_causes,
            correlation_structure,
            frailty_variance,
            max_iter,
            tol,
            estimate_correlation,
        })
    }
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct CauseResult {
    #[pyo3(get)]
    pub cause: usize,
    #[pyo3(get)]
    pub coefficients: Vec<f64>,
    #[pyo3(get)]
    pub std_errors: Vec<f64>,
    #[pyo3(get)]
    pub hazard_ratios: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard_times: Vec<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub cumulative_baseline_hazard: Vec<f64>,
}

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct JointCompetingRisksResult {
    #[pyo3(get)]
    pub cause_specific_results: Vec<CauseResult>,
    #[pyo3(get)]
    pub subdistribution_results: Vec<CauseResult>,
    #[pyo3(get)]
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub frailty_variance: Option<f64>,
    #[pyo3(get)]
    pub log_likelihood: f64,
    #[pyo3(get)]
    pub aic: f64,
    #[pyo3(get)]
    pub bic: f64,
    #[pyo3(get)]
    pub n_events_by_cause: Vec<usize>,
    #[pyo3(get)]
    pub n_obs: usize,
    #[pyo3(get)]
    pub n_iter: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl JointCompetingRisksResult {
    fn __repr__(&self) -> String {
        format!(
            "JointCompetingRisksResult(n_causes={}, n_obs={}, converged={})",
            self.cause_specific_results.len(),
            self.n_obs,
            self.converged
        )
    }

    fn predict_cif(&self, x: Vec<f64>, n_obs: usize, cause_idx: usize) -> PyResult<Vec<Vec<f64>>> {
        if cause_idx >= self.cause_specific_results.len() {
            return Err(PyValueError::new_err("cause_idx out of range"));
        }

        let output_grid = &self.cause_specific_results[cause_idx].baseline_hazard_times;
        let grid = union_event_grid(&self.cause_specific_results);
        let aligned_baselines = self
            .cause_specific_results
            .iter()
            .map(|result| cumulative_hazard_on_grid(result, &grid))
            .collect::<Vec<_>>();
        let risk_scores = self.prediction_risk_scores(&x, n_obs)?;

        Ok(risk_scores
            .into_par_iter()
            .map(|row_scores| {
                let mut values = Vec::with_capacity(output_grid.len());
                let mut output_idx = 0;
                let mut previous_total_hazard = 0.0;
                let mut previous_cause_hazard = 0.0;
                let mut survival = 1.0;
                let mut cumulative_incidence = 0.0;

                for time_idx in 0..grid.len() {
                    let total_hazard = aligned_baselines
                        .iter()
                        .zip(&row_scores)
                        .map(|(hazards, &risk_score)| {
                            scaled_cumulative_hazard(hazards[time_idx], risk_score)
                        })
                        .fold(0.0, saturating_nonnegative_add);
                    let cause_hazard = scaled_cumulative_hazard(
                        aligned_baselines[cause_idx][time_idx],
                        row_scores[cause_idx],
                    );
                    let total_increment =
                        nonnegative_increment(total_hazard, previous_total_hazard);
                    let cause_increment =
                        nonnegative_increment(cause_hazard, previous_cause_hazard)
                            .min(total_increment);

                    if total_increment > 0.0 {
                        let event_probability = -(-total_increment).exp_m1();
                        cumulative_incidence += survival
                            * event_probability
                            * (cause_increment / total_increment).clamp(0.0, 1.0);
                        survival *= (-total_increment).exp();
                    }

                    if output_idx < output_grid.len()
                        && same_time(grid[time_idx], output_grid[output_idx])
                    {
                        values.push(cumulative_incidence.clamp(0.0, 1.0));
                        output_idx += 1;
                    }
                    previous_total_hazard = total_hazard;
                    previous_cause_hazard = cause_hazard;
                }

                debug_assert_eq!(values.len(), output_grid.len());
                values
            })
            .collect())
    }

    fn predict_overall_survival(&self, x: Vec<f64>, n_obs: usize) -> PyResult<Vec<Vec<f64>>> {
        let first_result = self
            .cause_specific_results
            .first()
            .ok_or_else(|| PyValueError::new_err("model has no cause-specific results"))?;
        let grid = &first_result.baseline_hazard_times;
        let aligned_baselines = self
            .cause_specific_results
            .iter()
            .map(|result| cumulative_hazard_on_grid(result, grid))
            .collect::<Vec<_>>();
        let risk_scores = self.prediction_risk_scores(&x, n_obs)?;

        Ok(risk_scores
            .into_par_iter()
            .map(|row_scores| {
                (0..grid.len())
                    .map(|time_idx| {
                        let total_hazard = aligned_baselines
                            .iter()
                            .zip(&row_scores)
                            .map(|(hazards, &risk_score)| {
                                scaled_cumulative_hazard(hazards[time_idx], risk_score)
                            })
                            .fold(0.0, saturating_nonnegative_add);
                        (-total_hazard).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect())
    }
}

impl JointCompetingRisksResult {
    fn prediction_risk_scores(&self, x: &[f64], n_obs: usize) -> PyResult<Vec<Vec<f64>>> {
        let first_result = self
            .cause_specific_results
            .first()
            .ok_or_else(|| PyValueError::new_err("model has no cause-specific results"))?;
        let n_vars = first_result.coefficients.len();
        if n_vars == 0 {
            return Err(PyValueError::new_err(
                "cannot predict with a model that has no coefficients",
            ));
        }
        if self
            .cause_specific_results
            .iter()
            .any(|result| result.coefficients.len() != n_vars)
        {
            return Err(PyValueError::new_err(
                "cause-specific coefficient dimensions are inconsistent",
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
                self.cause_specific_results
                    .iter()
                    .map(|result| {
                        let linear_predictor = row
                            .iter()
                            .zip(&result.coefficients)
                            .map(|(&value, &coefficient)| value * coefficient)
                            .sum::<f64>();
                        (!linear_predictor.is_nan()).then(|| exp_clamped(linear_predictor))
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| PyValueError::new_err("x produced an undefined linear predictor"))
    }
}

fn cumulative_hazard_on_grid(result: &CauseResult, grid: &[f64]) -> Vec<f64> {
    let mut result_idx = 0;
    let mut cumulative_hazard = 0.0;

    grid.iter()
        .map(|&grid_time| {
            while result_idx < result.baseline_hazard_times.len()
                && (result.baseline_hazard_times[result_idx] < grid_time
                    || same_time(result.baseline_hazard_times[result_idx], grid_time))
            {
                cumulative_hazard = result
                    .cumulative_baseline_hazard
                    .get(result_idx)
                    .copied()
                    .unwrap_or(cumulative_hazard);
                result_idx += 1;
            }
            cumulative_hazard
        })
        .collect()
}

fn union_event_grid(results: &[CauseResult]) -> Vec<f64> {
    let mut times = results
        .iter()
        .flat_map(|result| result.baseline_hazard_times.iter().copied())
        .collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|left, right| same_time(*left, *right));
    times
}

#[inline]
fn scaled_cumulative_hazard(baseline_hazard: f64, risk_score: f64) -> f64 {
    let value = baseline_hazard.max(0.0) * risk_score;
    if value.is_finite() { value } else { f64::MAX }
}

#[inline]
fn saturating_nonnegative_add(left: f64, right: f64) -> f64 {
    let value = left + right;
    if value.is_finite() { value } else { f64::MAX }
}

#[inline]
fn nonnegative_increment(current: f64, previous: f64) -> f64 {
    if current <= previous {
        0.0
    } else {
        let increment = current - previous;
        if increment.is_finite() {
            increment
        } else {
            f64::MAX
        }
    }
}

fn validate_joint_config(config: &JointCompetingRisksConfig) -> PyResult<()> {
    if config.num_causes < 2 {
        return Err(PyValueError::new_err("num_causes must be at least 2"));
    }
    if !config.frailty_variance.is_finite() || config.frailty_variance <= 0.0 {
        return Err(PyValueError::new_err(
            "frailty_variance must be finite and positive",
        ));
    }
    if config.max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err(PyValueError::new_err("tol must be finite and positive"));
    }
    Ok(())
}

fn validate_joint_inputs(
    x: &[f64],
    n_obs: usize,
    n_vars: usize,
    time: &[f64],
    cause: &[i32],
    weights: Option<&[f64]>,
    num_causes: usize,
) -> PyResult<()> {
    validate_cause_specific_inputs(x, n_obs, n_vars, time, cause, weights)?;
    for (idx, &value) in cause.iter().enumerate() {
        if value as usize > num_causes {
            return Err(PyValueError::new_err(format!(
                "cause values must be between 0 and num_causes; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (x, n_obs, n_vars, time, cause, config, weights=None))]
pub fn joint_competing_risks(
    x: Vec<f64>,
    n_obs: usize,
    n_vars: usize,
    time: Vec<f64>,
    cause: Vec<i32>,
    config: &JointCompetingRisksConfig,
    weights: Option<Vec<f64>>,
) -> PyResult<JointCompetingRisksResult> {
    validate_joint_config(config)?;
    validate_joint_inputs(
        &x,
        n_obs,
        n_vars,
        &time,
        &cause,
        weights.as_deref(),
        config.num_causes,
    )?;
    let weights = weights.as_deref();

    let n_events_by_cause: Vec<usize> = (1..=config.num_causes as i32)
        .map(|c| cause.iter().filter(|&&cc| cc == c).count())
        .collect();

    let mut total_loglik = 0.0;
    let mut total_n_iter = 0;
    let mut all_converged = true;

    let mut cause_specific_results = Vec::with_capacity(config.num_causes);

    for c in 1..=config.num_causes as i32 {
        let cause_config = CauseSpecificCoxConfig::new(
            c,
            CensoringType::Censored,
            config.max_iter,
            config.tol,
            "breslow",
        )?;
        let result =
            cause_specific_cox_fit(&x, n_obs, n_vars, &time, &cause, &cause_config, weights)?;

        total_loglik += result.log_likelihood;
        total_n_iter = total_n_iter.max(result.n_iter);
        all_converged = all_converged && result.converged;

        cause_specific_results.push(CauseResult {
            cause: c as usize,
            coefficients: result.coefficients,
            std_errors: result.std_errors,
            hazard_ratios: result.hazard_ratios,
            baseline_hazard_times: result.baseline_hazard_times,
            baseline_hazard: result.baseline_hazard,
            cumulative_baseline_hazard: result.cumulative_baseline_hazard,
        });
    }

    let subdistribution_results = cause_specific_results.clone();

    let correlation_matrix = match config.correlation_structure {
        CorrelationType::Independent => None,
        CorrelationType::SharedFrailty | CorrelationType::CopulaBased => {
            let mut corr = vec![vec![0.0; config.num_causes]; config.num_causes];
            for (i, row) in corr.iter_mut().enumerate().take(config.num_causes) {
                row[i] = 1.0;
            }
            Some(corr)
        }
    };

    let frailty_variance = match config.correlation_structure {
        CorrelationType::SharedFrailty => Some(config.frailty_variance),
        _ => None,
    };

    let n_params = n_vars * config.num_causes;
    let aic = -2.0 * total_loglik + 2.0 * n_params as f64;
    let bic = -2.0 * total_loglik + (n_params as f64) * (n_obs as f64).ln();

    Ok(JointCompetingRisksResult {
        cause_specific_results,
        subdistribution_results,
        correlation_matrix,
        frailty_variance,
        log_likelihood: total_loglik,
        aic,
        bic,
        n_events_by_cause,
        n_obs,
        n_iter: total_n_iter,
        converged: all_converged,
    })
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

    #[test]
    fn test_config() {
        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-6, true)
                .unwrap();
        assert_eq!(config.num_causes, 2);
    }

    #[test]
    fn prediction_grid_aligns_each_cause_by_event_time() {
        let first = CauseResult {
            cause: 1,
            coefficients: vec![0.0],
            std_errors: vec![1.0],
            hazard_ratios: vec![1.0],
            baseline_hazard_times: vec![1.0, 3.0],
            baseline_hazard: vec![0.1, 0.3],
            cumulative_baseline_hazard: vec![0.1, 0.4],
        };
        let second = CauseResult {
            cause: 2,
            coefficients: vec![0.0],
            std_errors: vec![1.0],
            hazard_ratios: vec![1.0],
            baseline_hazard_times: vec![2.0],
            baseline_hazard: vec![0.2],
            cumulative_baseline_hazard: vec![0.2],
        };

        let grid = union_event_grid(&[first.clone(), second.clone()]);
        assert_eq!(grid, vec![1.0, 2.0, 3.0]);
        assert_eq!(
            cumulative_hazard_on_grid(&first, &grid),
            vec![0.1, 0.1, 0.4]
        );
        assert_eq!(
            cumulative_hazard_on_grid(&second, &grid),
            vec![0.0, 0.2, 0.2]
        );
    }

    #[test]
    fn test_config_validation() {
        assert!(
            JointCompetingRisksConfig::new(1, CorrelationType::Independent, 1.0, 100, 1e-6, true)
                .is_err()
        );
        assert!(
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, -1.0, 100, 1e-6, true)
                .is_err()
        );
    }

    #[test]
    fn test_joint_competing_risks_basic() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];

        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-5, true)
                .unwrap();

        let result = joint_competing_risks(x, 5, 2, time, cause, &config, None).unwrap();

        assert_eq!(result.cause_specific_results.len(), 2);
        assert_eq!(result.n_events_by_cause.len(), 2);
        assert_eq!(result.n_obs, 5);
    }

    #[test]
    fn test_joint_competing_risks_unweighted_matches_unit_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5];
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cause = vec![1, 2, 0, 1, 2];
        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-5, true)
                .unwrap();

        let unweighted =
            joint_competing_risks(x.clone(), 5, 2, time.clone(), cause.clone(), &config, None)
                .unwrap();
        let unit_weighted =
            joint_competing_risks(x, 5, 2, time, cause, &config, Some(vec![1.0; 5])).unwrap();

        assert_eq!(
            unweighted.cause_specific_results.len(),
            unit_weighted.cause_specific_results.len()
        );
        assert_eq!(
            unweighted.n_events_by_cause,
            unit_weighted.n_events_by_cause
        );
        assert_eq!(unweighted.n_obs, unit_weighted.n_obs);
        assert_eq!(unweighted.n_iter, unit_weighted.n_iter);
        assert_eq!(unweighted.converged, unit_weighted.converged);
        assert_close(unweighted.log_likelihood, unit_weighted.log_likelihood);
        assert_close(unweighted.aic, unit_weighted.aic);
        assert_close(unweighted.bic, unit_weighted.bic);

        for (left, right) in unweighted
            .cause_specific_results
            .iter()
            .zip(&unit_weighted.cause_specific_results)
        {
            assert_eq!(left.cause, right.cause);
            assert_vec_close(&left.coefficients, &right.coefficients);
            assert_vec_close(&left.std_errors, &right.std_errors);
            assert_vec_close(&left.hazard_ratios, &right.hazard_ratios);
            assert_vec_close(&left.baseline_hazard_times, &right.baseline_hazard_times);
            assert_vec_close(&left.baseline_hazard, &right.baseline_hazard);
            assert_vec_close(
                &left.cumulative_baseline_hazard,
                &right.cumulative_baseline_hazard,
            );
        }
    }

    #[test]
    fn test_joint_competing_risks_rejects_bad_weights_length() {
        pyo3::Python::initialize();
        let config =
            JointCompetingRisksConfig::new(2, CorrelationType::Independent, 1.0, 100, 1e-5, true)
                .unwrap();

        let err = joint_competing_risks(
            vec![0.0, 1.0],
            2,
            1,
            vec![1.0, 2.0],
            vec![1, 2],
            &config,
            Some(vec![1.0]),
        )
        .unwrap_err();

        assert!(err.to_string().contains("weights must have length n_obs"));
    }
}
