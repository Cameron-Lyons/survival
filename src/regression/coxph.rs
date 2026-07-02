use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN, TIME_EPSILON};
use crate::internal::validation::validate_binary_i32;
use crate::regression::cox_optimizer::{CoxFit, Method as CoxMethod};
pub use crate::regression::coxph_model::{CoxPHModel, Subject};
use crate::regression::coxph_support::{ActiveRiskSet, CoxSweepRow, StratifiedBaselineLookup};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::BTreeMap;

fn scaled_hazard_increment(events: f64, scaled_risk_sum: f64, risk_scale: f64) -> f64 {
    if events > 0.0 && scaled_risk_sum > 0.0 {
        events / scaled_risk_sum * risk_scale
    } else {
        0.0
    }
}

fn scaled_efron_hazard_increment(
    events: f64,
    deaths: usize,
    scaled_risk_sum: f64,
    scaled_death_risk_sum: f64,
    risk_scale: f64,
) -> f64 {
    if events <= 0.0 || deaths == 0 || scaled_risk_sum <= 0.0 {
        return 0.0;
    }

    let step_weight = events / deaths as f64;
    let mut increment = 0.0;
    for step in 0..deaths {
        let fraction = step as f64 / deaths as f64;
        let denom = scaled_risk_sum - fraction * scaled_death_risk_sum;
        if denom > 0.0 {
            increment += step_weight / denom * risk_scale;
        }
    }
    increment
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct CoxPHFit {
    #[pyo3(get)]
    pub coefficients: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub means: Vec<f64>,
    #[pyo3(get)]
    pub score_vector: Vec<f64>,
    #[pyo3(get)]
    pub information_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub log_likelihood: Vec<f64>,
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
    pub strata: Vec<i32>,
    #[pyo3(get)]
    pub method: String,
    #[pyo3(get)]
    pub nocenter: Vec<f64>,
}

impl CoxPHFit {
    fn explicit_row_strata(&self) -> Option<&[i32]> {
        (self.strata.len() == self.event_times.len()).then_some(self.strata.as_slice())
    }

    fn unique_row_strata(&self) -> Vec<i32> {
        let Some(strata) = self.explicit_row_strata() else {
            return vec![0];
        };
        let mut unique = strata.to_vec();
        unique.sort_unstable();
        unique.dedup();
        unique
    }

    pub(crate) fn row_strata_cow(&self) -> Cow<'_, [i32]> {
        if let Some(strata) = self.explicit_row_strata() {
            Cow::Borrowed(strata)
        } else {
            Cow::Owned(vec![0; self.event_times.len()])
        }
    }

    fn survival_curve_for_shared_row(
        &self,
        beta: &[f64],
        row: &[f64],
        strata: &[i32],
        centered: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        if row.len() != beta.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariates must have {} columns",
                beta.len()
            )));
        }
        validate_finite_values("covariates[0]", row)?;

        let center = if centered && !self.linear_predictors.is_empty() {
            self.linear_predictors.iter().sum::<f64>() / self.linear_predictors.len() as f64
        } else {
            0.0
        };
        let (base_times, base_hazards, base_strata) =
            self.basehaz_with_strata_internal(centered)?;
        let baseline =
            StratifiedBaselineLookup::from_components(&base_times, &base_hazards, &base_strata);
        let times = baseline.times_for_strata(strata);
        let linear_predictor = row
            .iter()
            .zip(beta.iter())
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>();
        let risk_multiplier = (linear_predictor - center)
            .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
            .exp();
        let curves = strata
            .iter()
            .map(|&stratum| {
                times
                    .iter()
                    .map(|&time| {
                        let hazard = baseline.cumulative_hazard_at(stratum, time);
                        (-(hazard * risk_multiplier)).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();
        Ok((times, curves))
    }

    pub(crate) fn basehaz_with_strata_internal(
        &self,
        centered: bool,
    ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
        let n = self.event_times.len();
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "time must not be empty",
            ));
        }
        let row_strata = self.explicit_row_strata();
        let center = if centered && !self.linear_predictors.is_empty() {
            self.linear_predictors.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };

        let mut rows_by_stratum: BTreeMap<i32, Vec<CoxSweepRow>> = BTreeMap::new();
        for idx in 0..n {
            let stratum = row_strata.map_or(0, |strata| strata[idx]);
            rows_by_stratum
                .entry(stratum)
                .or_default()
                .push(CoxSweepRow {
                    original_idx: idx,
                    stop: self.event_times[idx],
                    entry: self
                        .entry_times
                        .as_ref()
                        .map_or(f64::NEG_INFINITY, |entry| entry[idx]),
                    risk: 0.0,
                    weight: self.weights[idx],
                    status: self.status[idx],
                });
        }

        let total_event_count = self.status.iter().filter(|&&status| status == 1).count();
        let mut out_times = Vec::with_capacity(total_event_count);
        let mut out_hazards = Vec::with_capacity(total_event_count);
        let mut out_strata = Vec::with_capacity(total_event_count);
        let use_entry_times = self.entry_times.is_some();
        let use_efron = self.method == "efron";

        for (stratum, mut rows) in rows_by_stratum {
            let stratum_event_count = rows.iter().filter(|row| row.status == 1).count();
            let mut event_times = Vec::with_capacity(stratum_event_count);
            let mut death_order = Vec::with_capacity(stratum_event_count);
            for (row_idx, row) in rows.iter().enumerate() {
                if row.status == 1 {
                    event_times.push(row.stop);
                    death_order.push(row_idx);
                }
            }
            event_times.sort_by(|a, b| a.total_cmp(b));
            event_times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);
            if event_times.is_empty() {
                continue;
            }

            let max_shifted_lp = rows
                .iter()
                .filter_map(|row| {
                    (row.weight > 0.0).then_some(self.linear_predictors[row.original_idx] - center)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            let risk_scale = if max_shifted_lp.is_finite() {
                (-max_shifted_lp).exp()
            } else {
                1.0
            };
            for row in rows.iter_mut() {
                row.risk = if row.weight == 0.0 || !max_shifted_lp.is_finite() {
                    0.0
                } else {
                    row.weight
                        * (self.linear_predictors[row.original_idx] - center - max_shifted_lp).exp()
                };
            }

            let mut active = ActiveRiskSet::new(&rows, use_entry_times);

            death_order.sort_by(|&lhs, &rhs| {
                rows[lhs]
                    .stop
                    .total_cmp(&rows[rhs].stop)
                    .then_with(|| lhs.cmp(&rhs))
            });
            let mut death_weight_prefix = Vec::with_capacity(death_order.len() + 1);
            let mut death_risk_prefix = Vec::with_capacity(death_order.len() + 1);
            death_weight_prefix.push(0.0);
            death_risk_prefix.push(0.0);
            for &row_idx in &death_order {
                death_weight_prefix.push(
                    death_weight_prefix.last().copied().unwrap_or(0.0) + rows[row_idx].weight,
                );
                death_risk_prefix
                    .push(death_risk_prefix.last().copied().unwrap_or(0.0) + rows[row_idx].risk);
            }

            let mut cumulative = 0.0;
            for event_time in event_times {
                active.advance_to(event_time, |_, _| {});

                let lower = death_order
                    .partition_point(|&row_idx| rows[row_idx].stop <= event_time - TIME_EPSILON);
                let upper = death_order
                    .partition_point(|&row_idx| rows[row_idx].stop < event_time + TIME_EPSILON);
                let deaths = upper - lower;
                let events = death_weight_prefix[upper] - death_weight_prefix[lower];
                if active.risk_sum > 0.0 {
                    if use_efron && deaths > 1 {
                        let death_risk_sum = death_risk_prefix[upper] - death_risk_prefix[lower];
                        cumulative += scaled_efron_hazard_increment(
                            events,
                            deaths,
                            active.risk_sum,
                            death_risk_sum,
                            risk_scale,
                        );
                    } else {
                        cumulative += scaled_hazard_increment(events, active.risk_sum, risk_scale);
                    }
                }
                out_times.push(event_time);
                out_hazards.push(cumulative);
                out_strata.push(stratum);
            }
        }

        Ok((out_times, out_hazards, out_strata))
    }
}

#[pymethods]
impl CoxPHFit {
    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        covariates
            .into_iter()
            .map(|row| {
                if row.len() != nvar {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "covariate row has {} columns but model expects {}",
                        row.len(),
                        nvar
                    )));
                }
                validate_finite_values("covariates row", &row)?;
                Ok(row
                    .iter()
                    .zip(beta.iter())
                    .map(|(value, coefficient)| value * coefficient)
                    .sum())
            })
            .collect()
    }

    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients
            .first()
            .map(|beta| {
                beta.iter()
                    .map(|value| value.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[pyo3(signature = (centered = true))]
    pub fn basehaz(&self, centered: bool) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let (times, hazards, _) = self.basehaz_with_strata_internal(centered)?;
        Ok((times, hazards))
    }

    #[pyo3(signature = (centered = true))]
    pub fn basehaz_with_strata(&self, centered: bool) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
        self.basehaz_with_strata_internal(centered)
    }

    #[pyo3(signature = (covariates = None, centered = true))]
    pub fn survival_curve(
        &self,
        covariates: Option<Vec<Vec<f64>>>,
        centered: bool,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let rows = match covariates {
            Some(rows) => rows,
            None => {
                let strata = self.unique_row_strata();
                return self.survival_curve_for_shared_row(beta, &self.means, &strata, centered);
            }
        };
        if rows.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariates must have {} columns",
                nvar
            )));
        }
        for (idx, row) in rows.iter().enumerate() {
            validate_finite_values(&format!("covariates[{}]", idx), row)?;
        }

        let center = if centered && !self.linear_predictors.is_empty() {
            self.linear_predictors.iter().sum::<f64>() / self.linear_predictors.len() as f64
        } else {
            0.0
        };
        let (times, hazards) = self.basehaz(centered)?;
        let curves = rows
            .iter()
            .map(|row| {
                let linear_predictor = row
                    .iter()
                    .zip(beta.iter())
                    .map(|(value, coefficient)| value * coefficient)
                    .sum::<f64>();
                let risk_multiplier = (linear_predictor - center)
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                hazards
                    .iter()
                    .map(|hazard| (-(hazard * risk_multiplier)).exp().clamp(0.0, 1.0))
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
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        if covariates.len() != strata.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "strata must have one entry per covariate row",
            ));
        }
        if covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariates must have {} columns",
                nvar
            )));
        }
        for (idx, row) in covariates.iter().enumerate() {
            validate_finite_values(&format!("covariates[{}]", idx), row)?;
        }

        let center = if centered && !self.linear_predictors.is_empty() {
            self.linear_predictors.iter().sum::<f64>() / self.linear_predictors.len() as f64
        } else {
            0.0
        };
        let (base_times, base_hazards, base_strata) =
            self.basehaz_with_strata_internal(centered)?;
        let baseline =
            StratifiedBaselineLookup::from_components(&base_times, &base_hazards, &base_strata);
        let mut requested_strata = strata.clone();
        requested_strata.sort_unstable();
        requested_strata.dedup();

        let times = baseline.times_for_strata(&requested_strata);

        let curves = covariates
            .iter()
            .zip(strata.iter())
            .map(|(row, &stratum)| {
                let linear_predictor = row
                    .iter()
                    .zip(beta.iter())
                    .map(|(value, coefficient)| value * coefficient)
                    .sum::<f64>();
                let risk_multiplier = (linear_predictor - center)
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                times
                    .iter()
                    .map(|&time| {
                        let hazard = baseline.cumulative_hazard_at(stratum, time);
                        (-(hazard * risk_multiplier)).exp().clamp(0.0, 1.0)
                    })
                    .collect()
            })
            .collect();
        Ok((times, curves))
    }

    pub fn expected_events(&self) -> PyResult<Vec<f64>> {
        self.expected_events_internal()
    }

    pub fn martingale_residuals(&self) -> PyResult<Vec<f64>> {
        let expected = self.expected_events_internal()?;
        Ok(self
            .status
            .iter()
            .zip(expected.iter())
            .map(|(&status, &expected)| status as f64 - expected)
            .collect())
    }

    pub fn deviance_residuals(&self) -> PyResult<Vec<f64>> {
        let expected = self.expected_events_internal()?;
        Ok(self
            .status
            .iter()
            .zip(expected.iter())
            .map(|(&status, &expected)| {
                let status = status as f64;
                let residual = status - expected;
                let log_term = if status > 0.0 {
                    status * expected.max(crate::constants::DIVISION_FLOOR).ln()
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
            .collect())
    }

    pub fn schoenfeld_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.schoenfeld_residuals_internal()
    }

    pub fn scaled_schoenfeld_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.scaled_schoenfeld_residuals_internal()
    }

    pub fn partial_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.partial_residuals_internal()
    }

    pub fn score_residuals(&self) -> PyResult<Vec<Vec<f64>>> {
        self.score_residuals_internal()
    }

    pub fn dfbeta(&self) -> PyResult<Vec<Vec<f64>>> {
        self.dfbeta_from_score_residuals(false)
    }

    pub fn dfbetas(&self) -> PyResult<Vec<Vec<f64>>> {
        self.dfbeta_from_score_residuals(true)
    }
}

fn parse_cox_method(method: Option<&str>) -> PyResult<CoxMethod> {
    let method_name = method.unwrap_or("efron").to_ascii_lowercase();
    match method_name.as_str() {
        "breslow" => Ok(CoxMethod::Breslow),
        "efron" => Ok(CoxMethod::Efron),
        "exact" => Ok(CoxMethod::Exact),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "method must be 'efron', 'breslow', or 'exact'",
        )),
    }
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} contains non-finite value at index {}",
                name, idx
            )));
        }
    }
    Ok(())
}

fn validate_case_weights(weights: &[f64]) -> PyResult<()> {
    validate_finite_values("weights", weights)?;
    for (idx, &value) in weights.iter().enumerate() {
        if value < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "weights must be non-negative; got {} at index {}",
                value, idx
            )));
        }
    }
    if weights.iter().all(|&value| value == 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights must include at least one positive value",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (time, status, covariates, strata=None, weights=None, offset=None, initial_beta=None, max_iter=None, eps=None, toler=None, method=None, entry_times=None, nocenter=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    strata: Option<Vec<i32>>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    initial_beta: Option<Vec<f64>>,
    max_iter: Option<usize>,
    eps: Option<f64>,
    toler: Option<f64>,
    method: Option<&str>,
    entry_times: Option<Vec<f64>>,
    nocenter: Option<Vec<f64>>,
) -> PyResult<CoxPHFit> {
    let n = time.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "time must not be empty",
        ));
    }
    if status.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "status has {} rows but time has {}",
            status.len(),
            n
        )));
    }
    if covariates.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "covariates has {} rows but time has {}",
            covariates.len(),
            n
        )));
    }
    let nvar = covariates.first().map_or(0, Vec::len);
    if covariates.iter().any(|row| row.len() != nvar) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covariates must be rectangular",
        ));
    }
    validate_finite_values("time", &time)?;
    validate_binary_i32(&status, "status")?;
    for (idx, row) in covariates.iter().enumerate() {
        validate_finite_values(&format!("covariates[{}]", idx), row)?;
    }

    let check_len = |name: &str, len: usize| -> PyResult<()> {
        if len != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{} has {} rows but time has {}",
                name, len, n
            )));
        }
        Ok(())
    };

    if let Some(values) = strata.as_ref() {
        check_len("strata", values.len())?;
    }
    if let Some(values) = weights.as_ref() {
        check_len("weights", values.len())?;
        validate_case_weights(values)?;
    }
    if let Some(values) = offset.as_ref() {
        check_len("offset", values.len())?;
        validate_finite_values("offset", values)?;
    }
    if let Some(values) = entry_times.as_ref() {
        check_len("entry_times", values.len())?;
        validate_finite_values("entry_times", values)?;
        for (idx, (&start, &stop)) in values.iter().zip(time.iter()).enumerate() {
            if start >= stop {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "entry_times[{}] must be less than time[{}]",
                    idx, idx
                )));
            }
        }
    }
    if let Some(values) = nocenter.as_ref() {
        validate_finite_values("nocenter", values)?;
    }
    if let Some(values) = initial_beta.as_ref()
        && values.len() != nvar
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "initial_beta has {} values but covariates has {} columns",
            values.len(),
            nvar
        )));
    }
    if let Some(values) = initial_beta.as_ref() {
        validate_finite_values("initial_beta", values)?;
    }
    if let Some(value) = eps
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "eps must be a finite positive value",
        ));
    }
    if let Some(value) = toler
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "toler must be a finite positive value",
        ));
    }

    let cox_method = parse_cox_method(method)?;
    let method_name = match cox_method {
        CoxMethod::Breslow => "breslow",
        CoxMethod::Efron => "efron",
        CoxMethod::Exact => "exact",
    }
    .to_string();
    let offset_vec = offset.unwrap_or_else(|| vec![0.0; n]);
    let weights_vec = weights.unwrap_or_else(|| vec![1.0; n]);
    let strata_values = strata.unwrap_or_else(|| vec![0; n]);
    let nocenter_values = nocenter.unwrap_or_default();
    let doscale: Vec<bool> = if nocenter_values.is_empty() {
        vec![true; nvar]
    } else {
        (0..nvar)
            .map(|col_idx| {
                !covariates
                    .iter()
                    .all(|row| nocenter_values.iter().any(|value| row[col_idx] == *value))
            })
            .collect()
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        strata_values[lhs]
            .cmp(&strata_values[rhs])
            .then_with(|| time[lhs].total_cmp(&time[rhs]))
            .then_with(|| lhs.cmp(&rhs))
    });
    let entry_times_ref = entry_times.as_deref();
    let mut sorted_time = Vec::with_capacity(n);
    let mut sorted_status = Vec::with_capacity(n);
    let mut sorted_entry_times = entry_times_ref.map(|_| Vec::with_capacity(n));
    let mut sorted_offset = Vec::with_capacity(n);
    let mut sorted_weights = Vec::with_capacity(n);
    let mut strata_boundaries = vec![0; n];
    let mut flat = Vec::with_capacity(n * nvar);
    for (sorted_idx, &idx) in order.iter().enumerate() {
        sorted_time.push(time[idx]);
        sorted_status.push(status[idx]);
        if let (Some(values), Some(sorted_values)) = (entry_times_ref, sorted_entry_times.as_mut())
        {
            sorted_values.push(values[idx]);
        }
        sorted_offset.push(offset_vec[idx]);
        sorted_weights.push(weights_vec[idx]);
        if sorted_idx + 1 == n || strata_values[order[sorted_idx + 1]] != strata_values[idx] {
            strata_boundaries[sorted_idx] = 1;
        }
        flat.extend(covariates[idx].iter().copied());
    }
    let covar = Array2::from_shape_vec((n, nvar), flat).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid covariate shape: {}", e))
    })?;
    let mut cox_fit = CoxFit::new_with_entry_times(
        Array1::from_vec(sorted_time),
        Array1::from_vec(sorted_status),
        covar,
        sorted_entry_times.map(Array1::from_vec),
        Array1::from_vec(strata_boundaries),
        Array1::from_vec(sorted_offset),
        Array1::from_vec(sorted_weights),
        cox_method,
        max_iter.unwrap_or(20),
        eps.unwrap_or(1e-5),
        toler.unwrap_or(1e-9),
        doscale,
        initial_beta.unwrap_or_else(|| vec![0.0; nvar]),
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Cox fit initialization failed: {}", e))
    })?;
    cox_fit
        .fit()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Cox fit failed: {}", e)))?;
    let (beta, means, score_vector, information, log_likelihood, score_test, flag, iterations) =
        cox_fit.results();
    let mut linear_predictors = Vec::with_capacity(n);
    let mut risk_scores = Vec::with_capacity(n);
    for (row, &offset) in covariates.iter().zip(offset_vec.iter()) {
        let linear_predictor = row
            .iter()
            .zip(beta.iter())
            .map(|(value, coefficient)| value * coefficient)
            .sum::<f64>()
            + offset;
        risk_scores.push(linear_predictor.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp());
        linear_predictors.push(linear_predictor);
    }
    let information_matrix = information
        .outer_iter()
        .map(|row| row.iter().copied().collect())
        .collect();

    Ok(CoxPHFit {
        coefficients: vec![beta],
        means,
        score_vector,
        information_matrix,
        log_likelihood: log_likelihood.to_vec(),
        score_test,
        convergence_flag: flag,
        iterations,
        risk_scores,
        event_times: time,
        status,
        linear_predictors,
        entry_times,
        weights: weights_vec,
        covariates,
        strata: strata_values,
        method: method_name,
        nocenter: nocenter_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_test_fit(method: &str) -> CoxPHFit {
        CoxPHFit {
            coefficients: vec![vec![0.4]],
            means: vec![0.0],
            score_vector: vec![],
            information_matrix: vec![],
            log_likelihood: vec![],
            score_test: 0.0,
            convergence_flag: 0,
            iterations: 0,
            risk_scores: vec![],
            event_times: vec![2.0, 4.0, 4.0, 3.0, 5.0, 6.0],
            status: vec![1, 1, 1, 1, 1, 0],
            linear_predictors: vec![0.1, -0.2, 0.4, 0.0, 0.3, -0.1],
            entry_times: Some(vec![0.0, 1.0, 3.5, 0.5, 4.0, 0.0]),
            weights: vec![1.0, 2.0, 1.25, 1.5, 0.5, 1.2],
            covariates: vec![
                vec![0.2],
                vec![1.4],
                vec![-0.3],
                vec![0.8],
                vec![1.1],
                vec![-0.7],
            ],
            strata: vec![1, 1, 1, 2, 2, 2],
            method: method.to_string(),
            nocenter: vec![-1.0, 0.0, 1.0],
        }
    }

    fn near_tied_test_fit(method: &str) -> CoxPHFit {
        let mut fit = baseline_test_fit(method);
        fit.event_times[2] += TIME_EPSILON / 2.0;
        fit
    }

    fn brute_force_basehaz(fit: &CoxPHFit, centered: bool) -> (Vec<f64>, Vec<f64>, Vec<i32>) {
        let n = fit.event_times.len();
        let row_strata = fit.row_strata_cow().into_owned();
        let center = if centered && !fit.linear_predictors.is_empty() {
            fit.linear_predictors.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let risk_scores: Vec<f64> = fit
            .linear_predictors
            .iter()
            .zip(fit.weights.iter())
            .map(|(&lp, &weight)| weight * (lp - center).clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp())
            .collect();

        let mut strata_values = row_strata.clone();
        strata_values.sort_unstable();
        strata_values.dedup();

        let mut out_times = Vec::new();
        let mut out_hazards = Vec::new();
        let mut out_strata = Vec::new();

        for stratum in strata_values {
            let mut event_times: Vec<f64> = fit
                .event_times
                .iter()
                .zip(fit.status.iter())
                .zip(row_strata.iter())
                .filter_map(|((&time, &status), &row_stratum)| {
                    (status == 1 && row_stratum == stratum).then_some(time)
                })
                .collect();
            event_times.sort_by(|a, b| a.total_cmp(b));
            event_times.dedup_by(|a, b| (*a - *b).abs() < TIME_EPSILON);

            let mut cumulative = 0.0;
            for event_time in event_times {
                let death_indices: Vec<usize> = (0..n)
                    .filter(|&idx| {
                        row_strata[idx] == stratum
                            && fit.status[idx] == 1
                            && (fit.event_times[idx] - event_time).abs() < TIME_EPSILON
                    })
                    .collect();
                let events = death_indices
                    .iter()
                    .map(|&idx| fit.weights[idx])
                    .sum::<f64>();
                let risk_sum = (0..n)
                    .filter(|&idx| {
                        row_strata[idx] == stratum
                            && fit.event_times[idx] >= event_time
                            && fit
                                .entry_times
                                .as_ref()
                                .is_none_or(|entry| entry[idx] < event_time)
                    })
                    .map(|idx| risk_scores[idx])
                    .sum::<f64>();
                if risk_sum > 0.0 {
                    if fit.method == "efron" && death_indices.len() > 1 {
                        let death_risk_sum = death_indices
                            .iter()
                            .map(|&idx| risk_scores[idx])
                            .sum::<f64>();
                        let step_weight = events / death_indices.len() as f64;
                        for step in 0..death_indices.len() {
                            let fraction = step as f64 / death_indices.len() as f64;
                            let denom = risk_sum - fraction * death_risk_sum;
                            if denom > 0.0 {
                                cumulative += step_weight / denom;
                            }
                        }
                    } else {
                        cumulative += events / risk_sum;
                    }
                }
                out_times.push(event_time);
                out_hazards.push(cumulative);
                out_strata.push(stratum);
            }
        }

        (out_times, out_hazards, out_strata)
    }

    fn assert_close_vec(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "index {}: actual={} expected={}",
                idx,
                actual,
                expected
            );
        }
    }

    fn assert_close_matrix(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate()
        {
            assert_eq!(actual_row.len(), expected_row.len());
            for (col_idx, (&actual, &expected)) in
                actual_row.iter().zip(expected_row.iter()).enumerate()
            {
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "row {}, col {}: actual={} expected={}",
                    row_idx,
                    col_idx,
                    actual,
                    expected
                );
            }
        }
    }

    fn sorted_fit_order(fit: &CoxPHFit) -> Vec<usize> {
        let mut order: Vec<usize> = (0..fit.event_times.len()).collect();
        order.sort_by(|&lhs, &rhs| {
            fit.strata[lhs]
                .cmp(&fit.strata[rhs])
                .then_with(|| fit.event_times[lhs].total_cmp(&fit.event_times[rhs]))
                .then_with(|| lhs.cmp(&rhs))
        });
        order
    }

    #[test]
    fn test_coxph_fit_basehaz_matches_bruteforce_for_strata_entry_and_ties() {
        for method in ["breslow", "efron"] {
            let fit = baseline_test_fit(method);
            for centered in [false, true] {
                let expected = brute_force_basehaz(&fit, centered);
                let actual = fit
                    .basehaz_with_strata_internal(centered)
                    .expect("baseline hazard should be computed");

                assert_close_vec(&actual.0, &expected.0);
                assert_close_vec(&actual.1, &expected.1);
                assert_eq!(actual.2, expected.2);
            }
        }
    }

    #[test]
    fn test_coxph_fit_basehaz_uses_scaled_risk_scores_for_large_linear_predictors() {
        let fit = CoxPHFit {
            coefficients: vec![vec![710.0]],
            means: vec![0.0],
            score_vector: vec![],
            information_matrix: vec![],
            log_likelihood: vec![],
            score_test: 0.0,
            convergence_flag: 0,
            iterations: 0,
            risk_scores: vec![],
            event_times: vec![1.0, 2.0, 3.0],
            status: vec![1, 1, 1],
            linear_predictors: vec![710.0, 709.0, 708.0],
            entry_times: None,
            weights: vec![1.0, 1.0, 1.0],
            covariates: vec![vec![1.0], vec![709.0 / 710.0], vec![708.0 / 710.0]],
            strata: vec![0, 0, 0],
            method: "breslow".to_string(),
            nocenter: Vec::new(),
        };

        let (times, hazards, strata) = fit.basehaz_with_strata_internal(false).unwrap();
        let expected_first = (-710.0_f64).exp() / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());

        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        assert_eq!(strata, vec![0, 0, 0]);
        assert!(hazards[0].is_finite());
        assert!(hazards[0] > 0.0);
        assert!((hazards[0] - expected_first).abs() <= expected_first * 1e-12);
        assert!(hazards[1] > hazards[0]);
        assert!(hazards[2] > hazards[1]);
    }

    #[test]
    fn test_coxph_missing_strata_matches_explicit_zero_strata() {
        for method in ["breslow", "efron"] {
            let mut explicit = baseline_test_fit(method);
            explicit.strata = vec![0; explicit.event_times.len()];
            let mut implicit = explicit.clone();
            implicit.strata.clear();

            for centered in [false, true] {
                let explicit_basehaz = explicit
                    .basehaz_with_strata_internal(centered)
                    .expect("explicit zero-strata baseline hazard should compute");
                let implicit_basehaz = implicit
                    .basehaz_with_strata_internal(centered)
                    .expect("implicit zero-strata baseline hazard should compute");
                assert_close_vec(&implicit_basehaz.0, &explicit_basehaz.0);
                assert_close_vec(&implicit_basehaz.1, &explicit_basehaz.1);
                assert_eq!(implicit_basehaz.2, explicit_basehaz.2);

                let explicit_survival = explicit
                    .survival_curve(None, centered)
                    .expect("explicit zero-strata survival curve should compute");
                let implicit_survival = implicit
                    .survival_curve(None, centered)
                    .expect("implicit zero-strata survival curve should compute");
                assert_close_vec(&implicit_survival.0, &explicit_survival.0);
                assert_close_matrix(&implicit_survival.1, &explicit_survival.1);
            }
        }
    }

    #[test]
    fn test_coxph_default_survival_curve_matches_explicit_strata_means() {
        for method in ["breslow", "efron"] {
            let fit = baseline_test_fit(method);
            let default = fit
                .survival_curve(None, true)
                .expect("default stratified survival curve should compute");
            let explicit = fit
                .survival_curve_with_strata(
                    vec![fit.means.clone(), fit.means.clone()],
                    vec![1, 2],
                    true,
                )
                .expect("explicit stratified survival curve should compute");

            assert_close_vec(&default.0, &explicit.0);
            assert_close_matrix(&default.1, &explicit.1);
        }
    }

    #[test]
    fn test_coxph_default_survival_curve_matches_explicit_single_stratum_mean() {
        let mut fit = baseline_test_fit("breslow");
        fit.strata = vec![1; fit.event_times.len()];

        let default = fit
            .survival_curve(None, true)
            .expect("default single-stratum survival curve should compute");
        let explicit = fit
            .survival_curve(Some(vec![fit.means.clone()]), true)
            .expect("explicit single-stratum survival curve should compute");

        assert_close_vec(&default.0, &explicit.0);
        assert_close_matrix(&default.1, &explicit.1);
    }

    #[test]
    fn test_coxph_default_survival_curve_validates_means() {
        let mut fit = baseline_test_fit("breslow");
        fit.means = vec![f64::NAN];

        assert!(fit.survival_curve(None, true).is_err());
    }

    #[test]
    fn test_coxph_deviance_residuals_reuse_expected_events() {
        let fit = baseline_test_fit("breslow");
        let expected: Vec<f64> = fit
            .martingale_residuals()
            .expect("martingale residuals should compute")
            .iter()
            .zip(fit.status.iter())
            .map(|(&residual, &status)| {
                let status = status as f64;
                let log_term = if status > 0.0 {
                    let expected = (status - residual).max(crate::constants::DIVISION_FLOOR);
                    status * expected.ln()
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
            .collect();

        let actual = fit
            .deviance_residuals()
            .expect("deviance residuals should compute");

        assert_close_vec(&actual, &expected);
    }

    #[test]
    fn test_coxph_schoenfeld_sweep_matches_scan_for_strata_entry_and_ties() {
        for method in ["breslow", "efron"] {
            let fit = baseline_test_fit(method);
            let order = sorted_fit_order(&fit);
            let entry_times = fit.entry_times.as_ref();
            let tie_method = fit.tie_method();
            let expected = fit.schoenfeld_residuals_by_scan(1, &order, entry_times, tie_method);
            let actual = fit.schoenfeld_residuals_sweep(1, &order, entry_times, tie_method);
            assert_close_matrix(&actual, &expected);
        }
    }

    #[test]
    fn test_coxph_counting_score_residual_sweep_matches_scan_for_strata_entry_and_ties() {
        for method in ["breslow", "efron"] {
            let fit = baseline_test_fit(method);
            let order = sorted_fit_order(&fit);
            let entry_times = fit.entry_times.as_ref().expect("test fit has entry times");
            let risk: Vec<f64> = fit
                .linear_predictors
                .iter()
                .zip(fit.weights.iter())
                .map(|(&lp, &weight)| lp.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp() * weight)
                .collect();
            let method_code = if method == "efron" { 1 } else { 0 };

            let expected = fit.score_residuals_counting_process_by_scan(
                1,
                method_code,
                &risk,
                &order,
                entry_times,
            );
            let actual = fit.score_residuals_counting_process_sweep(
                1,
                method_code,
                &risk,
                &order,
                entry_times,
            );

            assert_close_matrix(&actual, &expected);
        }
    }

    #[test]
    fn test_coxph_schoenfeld_residuals_treat_near_ties_as_ties() {
        for method in ["breslow", "efron", "exact"] {
            let expected = baseline_test_fit(method)
                .schoenfeld_residuals_internal()
                .expect("exact-tied Schoenfeld residuals should compute");
            let actual = near_tied_test_fit(method)
                .schoenfeld_residuals_internal()
                .expect("near-tied Schoenfeld residuals should compute");

            assert_close_matrix(&actual, &expected);
        }
    }

    #[test]
    fn test_coxph_score_residuals_treat_near_ties_as_ties() {
        for method in ["breslow", "efron", "exact"] {
            let expected = baseline_test_fit(method)
                .score_residuals_internal()
                .expect("exact-tied score residuals should compute");
            let actual = near_tied_test_fit(method)
                .score_residuals_internal()
                .expect("near-tied score residuals should compute");

            assert_close_matrix(&actual, &expected);
        }
    }

    #[test]
    fn test_coxph_exact_score_residuals_treat_right_censored_near_ties_as_ties() {
        let mut expected_fit = baseline_test_fit("exact");
        expected_fit.entry_times = None;
        let expected = expected_fit
            .score_residuals_internal()
            .expect("exact-tied right-censored score residuals should compute");

        let mut actual_fit = near_tied_test_fit("exact");
        actual_fit.entry_times = None;
        let actual = actual_fit
            .score_residuals_internal()
            .expect("near-tied right-censored score residuals should compute");

        assert_close_matrix(&actual, &expected);
    }
}
