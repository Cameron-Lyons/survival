use crate::constants::{
    DIVISION_FLOOR, EXP_CLAMP_MIN, TIME_EPSILON, exp_clamped, exp_clamped_ci,
    z_score_for_confidence,
};
use crate::internal::matrix::invert_matrix;
use crate::regression::cox_optimizer::{CoxFitBuilder, Method as CoxMethod};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;

struct CoxModelRiskSetCache {
    risk_sum: Vec<f64>,
    weighted_cov: Vec<f64>,
    weighted_cov_sq: Vec<f64>,
    weighted_outer: Vec<f64>,
}

fn value_error(message: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(message.into())
}

fn validate_finite_values(name: &str, values: &[f64]) -> PyResult<()> {
    for (idx, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return Err(value_error(format!("{name} contains NaN at index {idx}")));
        }
        if !value.is_finite() {
            return Err(value_error(format!(
                "{name} contains non-finite value at index {idx}"
            )));
        }
    }
    Ok(())
}

fn validate_binary_censoring(censoring: &[u8]) -> PyResult<()> {
    for (idx, &value) in censoring.iter().enumerate() {
        if value > 1 {
            return Err(value_error(format!(
                "censoring must contain only 0/1 values; got {value} at index {idx}"
            )));
        }
    }
    Ok(())
}

fn sorted_unique_times(values: &[f64]) -> Vec<f64> {
    let mut times = values.to_vec();
    times.sort_by(f64::total_cmp);
    times.dedup_by(|left, right| (*left - *right).abs() < TIME_EPSILON);
    times
}

fn validate_covariate_rows(
    covariates: &[Vec<f64>],
    expected_rows: usize,
    expected_cols: Option<usize>,
) -> PyResult<usize> {
    if covariates.len() != expected_rows {
        return Err(value_error(format!(
            "covariates row count mismatch: expected {expected_rows}, got {}",
            covariates.len()
        )));
    }

    let ncols = expected_cols.unwrap_or_else(|| covariates.first().map_or(0, Vec::len));
    for (row_idx, row) in covariates.iter().enumerate() {
        if row.len() != ncols {
            return Err(value_error(format!(
                "covariate row {row_idx} has {} columns but expected {ncols}",
                row.len()
            )));
        }
        validate_finite_values(&format!("covariate row {row_idx}"), row)?;
    }
    Ok(ncols)
}

#[derive(Clone)]
#[pyclass(from_py_object)]
pub struct Subject {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub covariates: Vec<f64>,
    #[pyo3(get, set)]
    pub is_case: bool,
    #[pyo3(get, set)]
    pub is_subcohort: bool,
    #[pyo3(get, set)]
    pub stratum: usize,
}
#[pymethods]
impl Subject {
    #[new]
    pub fn new(
        id: usize,
        covariates: Vec<f64>,
        is_case: bool,
        is_subcohort: bool,
        stratum: usize,
    ) -> Self {
        Self {
            id,
            covariates,
            is_case,
            is_subcohort,
            stratum,
        }
    }
}

#[pyclass]
pub struct CoxPHModel {
    coefficients: Array2<f64>,
    #[pyo3(get)]
    pub baseline_hazard: Vec<f64>,
    #[pyo3(get)]
    pub risk_scores: Vec<f64>,
    #[pyo3(get, set)]
    pub event_times: Vec<f64>,
    #[pyo3(get, set)]
    pub censoring: Vec<u8>,
    covariates: Array2<f64>,
    covariates_flat: Vec<f64>,
    n_covariates: usize,
    baseline_hazard_lookup_times: Vec<f64>,
    baseline_hazard_lookup_values: Vec<f64>,
}

impl CoxPHModel {
    fn invalidate_fit_cache(&mut self) {
        self.risk_scores.clear();
        self.baseline_hazard.clear();
        self.baseline_hazard_lookup_times.clear();
        self.baseline_hazard_lookup_values.clear();
    }

    fn rebuild_covariates_array(&mut self) -> PyResult<()> {
        let nrows = self.event_times.len();
        if self.censoring.len() != nrows {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "event_times and censoring lengths differ: {} vs {}",
                nrows,
                self.censoring.len()
            )));
        }

        let expected_len = nrows
            .checked_mul(self.n_covariates)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("covariate shape overflow"))?;

        if self.covariates_flat.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariate data mismatch: expected {} values for shape ({}, {}), got {}",
                expected_len,
                nrows,
                self.n_covariates,
                self.covariates_flat.len()
            )));
        }

        self.covariates =
            Array2::from_shape_vec((nrows, self.n_covariates), self.covariates_flat.clone())
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "failed to materialize covariate matrix: {}",
                        e
                    ))
                })?;
        Ok(())
    }

    #[inline]
    fn baseline_cumulative_hazard_at(&self, time: f64) -> f64 {
        if self.baseline_hazard_lookup_times.is_empty() {
            return 0.0;
        }
        let pos = self
            .baseline_hazard_lookup_times
            .partition_point(|&t| t <= time);
        if pos == 0 {
            0.0
        } else {
            self.baseline_hazard_lookup_values[pos - 1]
        }
    }

    fn validate_prediction_rows(&self, covariates: &[Vec<f64>]) -> PyResult<()> {
        validate_covariate_rows(
            covariates,
            covariates.len(),
            Some(self.coefficients.nrows()),
        )?;
        Ok(())
    }

    fn linear_predictor_for_row(&self, row: &[f64]) -> f64 {
        let ncoef = self.coefficients.nrows();
        let mut risk = 0.0;
        for (col_idx, &cov) in row.iter().enumerate().take(ncoef) {
            risk += self.coefficients[[col_idx, 0]] * cov;
        }
        risk
    }

    fn exp_risk_for_row(&self, row: &[f64]) -> f64 {
        exp_clamped(self.linear_predictor_for_row(row))
    }

    fn descending_time_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.event_times.len()).collect();
        indices.sort_by(|&lhs, &rhs| {
            self.event_times[rhs]
                .total_cmp(&self.event_times[lhs])
                .then_with(|| lhs.cmp(&rhs))
        });
        indices
    }

    fn risk_set_cache(
        &self,
        include_cov: bool,
        include_cov_sq: bool,
        include_outer: bool,
    ) -> CoxModelRiskSetCache {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        let track_cov = include_cov || include_cov_sq || include_outer;
        let mut cache = CoxModelRiskSetCache {
            risk_sum: vec![0.0; n],
            weighted_cov: if track_cov {
                vec![0.0; n * nvar]
            } else {
                Vec::new()
            },
            weighted_cov_sq: if include_cov_sq {
                vec![0.0; n * nvar]
            } else {
                Vec::new()
            },
            weighted_outer: if include_outer {
                vec![0.0; n * nvar * nvar]
            } else {
                Vec::new()
            },
        };
        if n == 0 {
            return cache;
        }

        let sorted_indices = self.descending_time_indices();
        let mut risk_sum = 0.0;
        let mut weighted_cov = if track_cov {
            vec![0.0; nvar]
        } else {
            Vec::new()
        };
        let mut weighted_cov_sq = if include_cov_sq {
            vec![0.0; nvar]
        } else {
            Vec::new()
        };
        let mut weighted_outer = if include_outer {
            vec![0.0; nvar * nvar]
        } else {
            Vec::new()
        };
        let mut pos = 0usize;

        while pos < sorted_indices.len() {
            let group_time = self.event_times[sorted_indices[pos]];
            let group_start = pos;
            while pos < sorted_indices.len()
                && (self.event_times[sorted_indices[pos]] - group_time).abs() < TIME_EPSILON
            {
                let idx = sorted_indices[pos];
                let risk = self.risk_scores.get(idx).copied().unwrap_or(1.0);
                risk_sum += risk;
                if track_cov {
                    for col_idx in 0..nvar {
                        let cov = self.covariates.get([idx, col_idx]).copied().unwrap_or(0.0);
                        weighted_cov[col_idx] += risk * cov;
                        if include_cov_sq {
                            weighted_cov_sq[col_idx] += risk * cov * cov;
                        }
                        if include_outer {
                            for inner_idx in 0..nvar {
                                let inner_cov = self
                                    .covariates
                                    .get([idx, inner_idx])
                                    .copied()
                                    .unwrap_or(0.0);
                                weighted_outer[col_idx * nvar + inner_idx] +=
                                    risk * cov * inner_cov;
                            }
                        }
                    }
                }
                pos += 1;
            }

            for &idx in &sorted_indices[group_start..pos] {
                cache.risk_sum[idx] = risk_sum;
                if track_cov {
                    let base = idx * nvar;
                    cache.weighted_cov[base..base + nvar].copy_from_slice(&weighted_cov);
                    if include_cov_sq {
                        cache.weighted_cov_sq[base..base + nvar].copy_from_slice(&weighted_cov_sq);
                    }
                    if include_outer {
                        let outer_base = idx * nvar * nvar;
                        cache.weighted_outer[outer_base..outer_base + nvar * nvar]
                            .copy_from_slice(&weighted_outer);
                    }
                }
            }
        }

        cache
    }
}
impl Default for CoxPHModel {
    fn default() -> Self {
        Self::new()
    }
}
#[pymethods]
impl CoxPHModel {
    #[new]
    pub fn new() -> Self {
        Self {
            coefficients: Array2::<f64>::zeros((0, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times: Vec::new(),
            censoring: Vec::new(),
            covariates: Array2::<f64>::zeros((0, 0)),
            covariates_flat: Vec::new(),
            n_covariates: 0,
            baseline_hazard_lookup_times: Vec::new(),
            baseline_hazard_lookup_values: Vec::new(),
        }
    }
    #[pyo3(signature = (covariates, event_times, censoring))]
    #[staticmethod]
    pub fn new_with_data(
        covariates: Vec<Vec<f64>>,
        event_times: Vec<f64>,
        censoring: Vec<u8>,
    ) -> PyResult<Self> {
        let nrows = covariates.len();
        if event_times.len() != nrows {
            return Err(value_error(format!(
                "event_times length mismatch: expected {nrows}, got {}",
                event_times.len()
            )));
        }
        if censoring.len() != nrows {
            return Err(value_error(format!(
                "censoring length mismatch: expected {nrows}, got {}",
                censoring.len()
            )));
        }
        validate_finite_values("event_times", &event_times)?;
        validate_binary_censoring(&censoring)?;
        let ncols = validate_covariate_rows(&covariates, nrows, None)?;
        let mut covariates_flat = Vec::with_capacity(nrows * ncols);
        for row in &covariates {
            covariates_flat.extend_from_slice(row);
        }
        let cov_array = Array2::from_shape_vec((nrows, ncols), covariates_flat.clone())
            .map_err(|e| value_error(format!("failed to materialize covariate matrix: {e}")))?;
        Ok(Self {
            coefficients: Array2::<f64>::zeros((ncols, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times,
            censoring,
            covariates: cov_array,
            covariates_flat,
            n_covariates: ncols,
            baseline_hazard_lookup_times: Vec::new(),
            baseline_hazard_lookup_values: Vec::new(),
        })
    }
    pub fn add_subject(&mut self, subject: &Subject) -> PyResult<()> {
        if self.n_covariates == 0 {
            self.n_covariates = subject.covariates.len();
        }
        if self.n_covariates != subject.covariates.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "covariate dimension mismatch: expected {}, got {}",
                self.n_covariates,
                subject.covariates.len()
            )));
        }
        validate_finite_values("subject covariates", &subject.covariates)?;
        self.covariates_flat.extend_from_slice(&subject.covariates);
        self.event_times.push(0.0);
        self.censoring.push(if subject.is_case { 1 } else { 0 });
        self.invalidate_fit_cache();
        Ok(())
    }
    #[pyo3(signature = (n_iters = 20))]
    pub fn fit(&mut self, n_iters: u16) -> PyResult<()> {
        if self.event_times.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot fit model: no data provided",
            ));
        }
        self.rebuild_covariates_array()?;
        let n = self.event_times.len();
        let nvar = self.n_covariates;
        if nvar == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot fit model: no covariates provided",
            ));
        }
        let time_array = Array1::from_vec(self.event_times.clone());
        let status_array: Array1<i32> =
            Array1::from_vec(self.censoring.iter().map(|&x| x as i32).collect());
        let strata = Array1::zeros(n);
        let initial_beta: Vec<f64> =
            if self.coefficients.nrows() == nvar && self.coefficients.ncols() > 0 {
                self.coefficients.column(0).to_vec()
            } else {
                vec![0.0; nvar]
            };
        let mut cox_fit = CoxFitBuilder::new(time_array, status_array, self.covariates.clone())
            .strata(strata)
            .method(CoxMethod::Breslow)
            .max_iter(n_iters as usize)
            .eps(1e-5)
            .toler(1e-9)
            .initial_beta(initial_beta)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Cox fit initialization failed: {}",
                    e
                ))
            })?;
        cox_fit.fit().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Cox fit failed: {}", e))
        })?;
        let (beta, _means, _u, _imat, _loglik, _sctest, _flag, _iter) = cox_fit.results();

        let mut risk_scores = Vec::with_capacity(n);
        for row in self.covariates.outer_iter() {
            let risk_score = row
                .iter()
                .zip(beta.iter())
                .map(|(&cov, &coef)| coef * cov)
                .sum::<f64>();
            risk_scores.push(exp_clamped(risk_score));
        }

        self.coefficients = Array2::from_shape_vec((nvar, 1), beta).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to materialize Cox coefficients: {e}"
            ))
        })?;
        self.risk_scores = risk_scores;
        self.calculate_baseline_hazard();
        Ok(())
    }
    fn calculate_baseline_hazard(&mut self) {
        let n = self.event_times.len();
        if n == 0 {
            self.baseline_hazard = Vec::new();
            self.baseline_hazard_lookup_times.clear();
            self.baseline_hazard_lookup_values.clear();
            return;
        }
        let risk_sets = self.risk_set_cache(false, false, false);
        let mut event_indices: Vec<usize> =
            (0..n).filter(|&idx| self.censoring[idx] == 1).collect();
        event_indices.sort_by(|&lhs, &rhs| {
            self.event_times[lhs]
                .total_cmp(&self.event_times[rhs])
                .then_with(|| lhs.cmp(&rhs))
        });
        let mut unique_times = Vec::with_capacity(event_indices.len());
        let mut baseline_hazard = Vec::with_capacity(event_indices.len());
        let mut cum_hazard = 0.0;
        let mut i = 0;
        while i < event_indices.len() {
            let idx = event_indices[i];
            let current_time = self.event_times[idx];
            let mut events = 0.0;
            while i < event_indices.len()
                && (self.event_times[event_indices[i]] - current_time).abs() < TIME_EPSILON
            {
                events += 1.0;
                i += 1;
            }
            let risk_sum = risk_sets.risk_sum[idx];
            if risk_sum > 0.0 {
                let hazard = events / risk_sum;
                cum_hazard += hazard;
            }
            unique_times.push(current_time);
            baseline_hazard.push(cum_hazard);
        }
        if baseline_hazard.is_empty() {
            self.baseline_hazard = vec![0.0; n];
            self.baseline_hazard_lookup_times.clear();
            self.baseline_hazard_lookup_values.clear();
        } else {
            self.baseline_hazard_lookup_times = unique_times;
            self.baseline_hazard_lookup_values = baseline_hazard;
            let mut full_baseline = vec![0.0; n];
            for (i, &t) in self.event_times.iter().enumerate() {
                full_baseline[i] = self.baseline_cumulative_hazard_at(t);
            }
            self.baseline_hazard = full_baseline;
        }
    }
    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.validate_prediction_rows(&covariates)?;
        Ok(covariates
            .par_iter()
            .map(|row| self.linear_predictor_for_row(row))
            .collect())
    }
    #[getter]
    pub fn get_coefficients(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(self.coefficients.ncols());
        for col in self.coefficients.columns() {
            result.push(col.iter().copied().collect());
        }
        result
    }
    pub fn brier_score(&self) -> f64 {
        let mut score = 0.0;
        let mut count = 0.0;
        let avg_risk = if self.baseline_hazard.is_empty() || self.risk_scores.is_empty() {
            None
        } else {
            Some(self.risk_scores.iter().sum::<f64>() / self.risk_scores.len() as f64)
        };
        for (time, &status) in self.event_times.iter().zip(self.censoring.iter()) {
            let pred = if let Some(avg_risk) = avg_risk {
                let baseline_haz = self.baseline_cumulative_hazard_at(*time);
                (-baseline_haz * avg_risk).exp()
            } else {
                0.5
            };
            score += (pred - status as f64).powi(2);
            count += 1.0;
        }
        if count > 0.0 { score / count } else { 0.0 }
    }
    pub fn survival_curve(
        &self,
        covariates: Vec<Vec<f64>>,
        time_points: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.validate_prediction_rows(&covariates)?;
        let times = time_points.unwrap_or_else(|| sorted_unique_times(&self.event_times));
        let baseline_hazards: Vec<f64> = times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        let survival_curves: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|row| {
                let risk_exp = self.exp_risk_for_row(row);
                baseline_hazards
                    .iter()
                    .map(|&bh| (-bh * risk_exp).exp())
                    .collect()
            })
            .collect();
        Ok((times, survival_curves))
    }
    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients
            .column(0)
            .iter()
            .map(|&beta| exp_clamped(beta))
            .collect()
    }
    #[pyo3(signature = (confidence_level = 0.95))]
    pub fn hazard_ratios_with_ci(&self, confidence_level: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let coefs = self.coefficients.column(0);
        let n = coefs.len();
        let z = z_score_for_confidence(confidence_level);
        let se = self.compute_standard_errors();
        let mut hr = Vec::with_capacity(n);
        let mut ci_lower = Vec::with_capacity(n);
        let mut ci_upper = Vec::with_capacity(n);
        for (i, &beta) in coefs.iter().enumerate() {
            let se_i = se.get(i).copied().unwrap_or(0.1);
            hr.push(exp_clamped(beta));
            let (lower, upper) = exp_clamped_ci(beta, se_i, z);
            ci_lower.push(lower);
            ci_upper.push(upper);
        }
        (hr, ci_lower, ci_upper)
    }
    fn compute_standard_errors(&self) -> Vec<f64> {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        if n == 0 || nvar == 0 {
            return vec![0.1; nvar];
        }
        let risk_sets = self.risk_set_cache(true, true, false);
        let fisher_diag = (0..n)
            .into_par_iter()
            .filter(|&i| self.censoring[i] == 1)
            .fold(
                || vec![0.0; nvar],
                |mut diag, i| {
                    let risk_set_sum = risk_sets.risk_sum[i];
                    if risk_set_sum <= 0.0 {
                        return diag;
                    }
                    let base = i * nvar;
                    for (k, diag_k) in diag.iter_mut().enumerate().take(nvar) {
                        let weighted_cov = risk_sets.weighted_cov[base + k];
                        let weighted_cov_sq = risk_sets.weighted_cov_sq[base + k];
                        let mean_cov = weighted_cov / risk_set_sum;
                        *diag_k += weighted_cov_sq / risk_set_sum - mean_cov * mean_cov;
                    }
                    diag
                },
            )
            .reduce(
                || vec![0.0; nvar],
                |mut total, diag| {
                    for (total_k, diag_k) in total.iter_mut().zip(diag) {
                        *total_k += diag_k;
                    }
                    total
                },
            );
        fisher_diag
            .iter()
            .map(|&f| if f > 0.0 { (1.0 / f).sqrt() } else { 0.1 })
            .collect()
    }
    pub fn log_likelihood(&self) -> f64 {
        if self.event_times.is_empty() || self.risk_scores.is_empty() {
            return 0.0;
        }
        let risk_sets = self.risk_set_cache(false, false, false);
        (0..self.event_times.len())
            .into_par_iter()
            .filter(|&i| self.censoring[i] == 1)
            .map(|i| {
                let risk_score_i = self.risk_scores.get(i).copied().unwrap_or(1.0).ln();
                let risk_set_sum = risk_sets.risk_sum[i];
                if risk_set_sum > 0.0 {
                    risk_score_i - risk_set_sum.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
    pub fn aic(&self) -> f64 {
        let k = self.coefficients.nrows() as f64;
        -2.0 * self.log_likelihood() + 2.0 * k
    }
    pub fn bic(&self) -> f64 {
        let k = self.coefficients.nrows() as f64;
        let n = self.event_times.len() as f64;
        -2.0 * self.log_likelihood() + k * n.ln()
    }
    pub fn cumulative_hazard(
        &self,
        covariates: Vec<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        self.validate_prediction_rows(&covariates)?;
        let unique_times = sorted_unique_times(&self.event_times);
        let baseline_hazards: Vec<f64> = unique_times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        let cumulative_hazards: Vec<Vec<f64>> = covariates
            .par_iter()
            .map(|row| {
                let risk_exp = self.exp_risk_for_row(row);
                baseline_hazards.iter().map(|&bh| bh * risk_exp).collect()
            })
            .collect();
        Ok((unique_times, cumulative_hazards))
    }
    #[pyo3(signature = (covariates, percentile = 0.5))]
    pub fn predicted_survival_time(
        &self,
        covariates: Vec<Vec<f64>>,
        percentile: f64,
    ) -> Vec<Option<f64>> {
        if self.validate_prediction_rows(&covariates).is_err() {
            return vec![];
        }
        let times = sorted_unique_times(&self.event_times);
        let baseline_hazards: Vec<f64> = times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        let target_survival = 1.0 - percentile;
        covariates
            .par_iter()
            .map(|row| {
                let risk_exp = self.exp_risk_for_row(row);
                let mut prev_surv = 1.0;
                for (i, (&time, &baseline_hazard)) in
                    times.iter().zip(baseline_hazards.iter()).enumerate()
                {
                    let survival = (-baseline_hazard * risk_exp).exp();
                    if survival <= target_survival {
                        if i == 0 {
                            return Some(time);
                        }
                        let t0 = times[i - 1];
                        let frac = (prev_surv - target_survival) / (prev_surv - survival);
                        return Some(t0 + frac * (time - t0));
                    }
                    prev_surv = survival;
                }
                None
            })
            .collect()
    }
    pub fn restricted_mean_survival_time(&self, covariates: Vec<Vec<f64>>, tau: f64) -> Vec<f64> {
        if self.validate_prediction_rows(&covariates).is_err() {
            return vec![];
        }
        let times = sorted_unique_times(&self.event_times);
        let baseline_hazards: Vec<f64> = times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        covariates
            .par_iter()
            .map(|row| {
                let risk_exp = self.exp_risk_for_row(row);
                let mut rmst = 0.0;
                let mut prev_time = 0.0;
                let mut prev_surv = 1.0;
                for (i, (&time, &baseline_hazard)) in
                    times.iter().zip(baseline_hazards.iter()).enumerate()
                {
                    if time > tau {
                        rmst += prev_surv * (tau - prev_time);
                        break;
                    }
                    rmst += prev_surv * (time - prev_time);
                    prev_time = time;
                    prev_surv = (-baseline_hazard * risk_exp).exp();
                    if i == times.len() - 1 {
                        rmst += prev_surv * (tau - time);
                    }
                }
                rmst
            })
            .collect()
    }
    pub fn martingale_residuals(&self) -> Vec<f64> {
        let n = self.event_times.len();
        let mut residuals = Vec::with_capacity(n);
        for i in 0..n {
            let status = self.censoring[i] as f64;
            let cum_haz = self.baseline_hazard.get(i).copied().unwrap_or(0.0)
                * self.risk_scores.get(i).copied().unwrap_or(1.0);
            residuals.push(status - cum_haz);
        }
        residuals
    }
    pub fn deviance_residuals(&self) -> Vec<f64> {
        let n = self.event_times.len();
        let mut residuals = Vec::with_capacity(n);
        for i in 0..n {
            let status = self.censoring[i] as f64;
            let cum_haz = self.baseline_hazard.get(i).copied().unwrap_or(0.0)
                * self.risk_scores.get(i).copied().unwrap_or(1.0);
            let martingale = status - cum_haz;
            let sign = if martingale >= 0.0 { 1.0 } else { -1.0 };
            let abs_term = -2.0
                * (martingale - status + status * (status - martingale).ln().max(EXP_CLAMP_MIN));
            residuals.push(sign * abs_term.abs().sqrt());
        }
        residuals
    }
    pub fn dfbeta(&self) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        if n == 0 || nvar == 0 {
            return vec![];
        }
        let risk_sets = self.risk_set_cache(true, false, false);
        (0..n)
            .into_par_iter()
            .map(|i| {
                let status = self.censoring[i] as f64;
                let risk_i = self.risk_scores.get(i).copied().unwrap_or(1.0);
                let cum_haz = self.baseline_hazard.get(i).copied().unwrap_or(0.0) * risk_i;
                let mart_i = status - cum_haz;
                let risk_sum = risk_sets.risk_sum[i];
                let base = i * nvar;
                (0..nvar)
                    .map(|k| {
                        let cov_ik = self.covariates.get([i, k]).copied().unwrap_or(0.0);
                        let weighted_mean = if risk_sum > 0.0 {
                            risk_sets.weighted_cov[base + k] / risk_sum
                        } else {
                            0.0
                        };
                        mart_i * (cov_ik - weighted_mean) / risk_i.max(DIVISION_FLOOR)
                    })
                    .collect()
            })
            .collect()
    }
    pub fn n_events(&self) -> usize {
        self.censoring.iter().filter(|&&c| c == 1).count()
    }
    pub fn vcov(&self) -> Vec<Vec<f64>> {
        let nvar = self.coefficients.nrows();
        if nvar == 0 {
            return vec![];
        }
        let fisher_info = self.compute_fisher_information();
        if fisher_info.is_empty() {
            return vec![vec![0.0; nvar]; nvar];
        }
        invert_matrix(&fisher_info).unwrap_or_else(|| vec![vec![0.0; nvar]; nvar])
    }
    pub fn std_errors(&self) -> Vec<f64> {
        self.compute_standard_errors()
    }
    fn compute_fisher_information(&self) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        if n == 0 || nvar == 0 {
            return vec![];
        }
        let risk_sets = self.risk_set_cache(true, false, true);
        let mut fisher = vec![vec![0.0; nvar]; nvar];
        for (i, &censor) in self.censoring.iter().enumerate() {
            if censor != 1 {
                continue;
            }
            let rs = risk_sets.risk_sum[i];
            if rs <= 0.0 {
                continue;
            }
            let wc_start = i * nvar;
            let wco_start = i * nvar * nvar;
            for (k, fisher_row) in fisher.iter_mut().enumerate().take(nvar) {
                let wc_k = risk_sets.weighted_cov[wc_start + k];
                for (l, fisher_value) in fisher_row.iter_mut().enumerate().take(nvar) {
                    let wc_l = risk_sets.weighted_cov[wc_start + l];
                    let wco_kl = risk_sets.weighted_outer[wco_start + k * nvar + l];
                    let info_kl = wco_kl / rs - (wc_k / rs) * (wc_l / rs);
                    *fisher_value += info_kl;
                }
            }
        }
        fisher
    }
    pub fn n_observations(&self) -> usize {
        self.event_times.len()
    }
    pub fn summary(&self) -> String {
        let nvar = self.coefficients.nrows();
        let n_obs = self.n_observations();
        let n_events = self.n_events();
        let loglik = self.log_likelihood();
        let aic = self.aic();
        let mut result = String::with_capacity(200 + 50 * nvar);
        result.push_str("Cox Proportional Hazards Model\n");
        result.push_str("================================\n");
        result.push_str(&format!("n={}, events={}\n\n", n_obs, n_events));
        result.push_str(&format!("Log-likelihood: {:.4}\n", loglik));
        result.push_str(&format!("AIC: {:.4}\n\n", aic));
        let hrs = self.hazard_ratios();
        let (_, ci_lower, ci_upper) = self.hazard_ratios_with_ci(0.95);
        result.push_str(&format!(
            "{:<10} {:>10} {:>10} {:>10}\n",
            "Variable", "HR", "CI_Lower", "CI_Upper"
        ));
        result.push_str(&format!("{:-<43}\n", ""));
        for i in 0..nvar {
            result.push_str(&format!(
                "var{:<7} {:>10.4} {:>10.4} {:>10.4}\n",
                i,
                hrs.get(i).copied().unwrap_or(f64::NAN),
                ci_lower.get(i).copied().unwrap_or(f64::NAN),
                ci_upper.get(i).copied().unwrap_or(f64::NAN)
            ));
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subject_new() {
        let subject = Subject::new(1, vec![1.0, 2.0], true, false, 0);
        assert_eq!(subject.id, 1);
        assert_eq!(subject.covariates, vec![1.0, 2.0]);
        assert!(subject.is_case);
        assert!(!subject.is_subcohort);
        assert_eq!(subject.stratum, 0);
    }

    #[test]
    fn test_coxph_model_default() {
        let model = CoxPHModel::new();
        assert!(model.baseline_hazard.is_empty());
        assert!(model.risk_scores.is_empty());
        assert!(model.event_times.is_empty());
    }

    #[test]
    fn test_coxph_model_getters() {
        let model = CoxPHModel::new();
        assert_eq!(model.n_observations(), 0);
        assert_eq!(model.n_events(), 0);
    }

    #[test]
    fn test_add_subject_tracks_covariate_buffer() {
        let mut model = CoxPHModel::new();
        let s1 = Subject::new(1, vec![1.0, 2.0], true, false, 0);
        let s2 = Subject::new(2, vec![3.0, 4.0], false, false, 0);
        let bad = Subject::new(3, vec![5.0], true, false, 0);

        model.add_subject(&s1).expect("first subject should append");
        model
            .add_subject(&s2)
            .expect("second subject should append");

        assert_eq!(model.n_observations(), 2);
        assert_eq!(model.n_covariates, 2);
        assert_eq!(model.covariates_flat, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(model.add_subject(&bad).is_err());
    }

    #[test]
    fn test_cumulative_hazard_uses_cached_baseline_lookup() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.event_times = vec![1.0, 2.0, 3.0];
        model.censoring = vec![1, 1, 1];
        model.baseline_hazard = vec![0.1, 0.2, 0.3];
        model.baseline_hazard_lookup_times = vec![1.0, 2.0, 3.0];
        model.baseline_hazard_lookup_values = vec![0.1, 0.2, 0.3];

        let (times, hazards) = model
            .cumulative_hazard(vec![vec![0.0]])
            .expect("cumulative hazard should succeed for valid covariates");
        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        assert_eq!(hazards.len(), 1);
        assert!((hazards[0][0] - 0.1).abs() < 1e-12);
        assert!((hazards[0][1] - 0.2).abs() < 1e-12);
        assert!((hazards[0][2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_risk_set_cache_groups_tied_times() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.covariates =
            Array2::from_shape_vec((4, 1), vec![0.0; 4]).expect("covariate shape is valid");
        model.n_covariates = 1;
        model.event_times = vec![1.0, 2.0, 2.0, 3.0];
        model.censoring = vec![1, 1, 1, 0];
        model.risk_scores = vec![2.0, 3.0, 5.0, 4.0];

        let cache = model.risk_set_cache(false, false, false);
        assert_eq!(cache.risk_sum, vec![14.0, 12.0, 12.0, 4.0]);

        let expected_loglik = 2.0_f64.ln() - 14.0_f64.ln() + 3.0_f64.ln() - 12.0_f64.ln()
            + 5.0_f64.ln()
            - 12.0_f64.ln();
        assert!((model.log_likelihood() - expected_loglik).abs() < 1e-12);

        model.calculate_baseline_hazard();
        assert_eq!(model.baseline_hazard_lookup_times, vec![1.0, 2.0]);
        assert!((model.baseline_hazard_lookup_values[0] - 1.0 / 14.0).abs() < 1e-12);
        assert!((model.baseline_hazard_lookup_values[1] - (1.0 / 14.0 + 2.0 / 12.0)).abs() < 1e-12);
    }

    #[test]
    fn test_default_prediction_times_are_total_ordered() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.covariates =
            Array2::from_shape_vec((4, 1), vec![0.0; 4]).expect("covariate shape is valid");
        model.n_covariates = 1;
        model.event_times = vec![3.0, 1.0, 2.0, 2.0 + TIME_EPSILON / 2.0];
        model.censoring = vec![0, 1, 1, 1];
        model.risk_scores = vec![4.0, 2.0, 3.0, 5.0];

        model.calculate_baseline_hazard();
        assert_eq!(model.baseline_hazard_lookup_times, vec![1.0, 2.0]);

        let (survival_times, survival) = model
            .survival_curve(vec![vec![0.0]], None)
            .expect("survival curve should use default event times");
        let (hazard_times, cumulative_hazard) = model
            .cumulative_hazard(vec![vec![0.0]])
            .expect("cumulative hazard should use default event times");

        assert_eq!(survival_times, vec![1.0, 2.0, 3.0]);
        assert_eq!(hazard_times, vec![1.0, 2.0, 3.0]);
        assert_eq!(survival.len(), 1);
        assert_eq!(cumulative_hazard.len(), 1);
    }

    #[test]
    fn test_predicted_survival_time_interpolates_from_baseline_hazard() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.event_times = vec![1.0, 2.0, 3.0];
        model.censoring = vec![1, 1, 1];
        model.baseline_hazard_lookup_times = vec![1.0, 2.0, 3.0];
        model.baseline_hazard_lookup_values = vec![0.1, 0.8, 1.2];

        let survival_at_1 = (-0.1_f64).exp();
        let survival_at_2 = (-0.8_f64).exp();
        let expected = 1.0 + (survival_at_1 - 0.5) / (survival_at_1 - survival_at_2) * (2.0 - 1.0);

        let predicted = model.predicted_survival_time(vec![vec![0.0]], 0.5);

        assert_eq!(predicted.len(), 1);
        assert!((predicted[0].expect("median should be predicted") - expected).abs() < 1e-12);
    }

    #[test]
    fn test_restricted_mean_survival_time_integrates_baseline_hazard() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.event_times = vec![1.0, 2.0, 3.0];
        model.censoring = vec![1, 1, 1];
        model.baseline_hazard_lookup_times = vec![1.0, 2.0, 3.0];
        model.baseline_hazard_lookup_values = vec![0.1, 0.8, 1.2];

        let expected = 1.0 + (-0.1_f64).exp() + (-0.8_f64).exp() * 0.5;

        let rmst = model.restricted_mean_survival_time(vec![vec![0.0]], 2.5);

        assert_eq!(rmst.len(), 1);
        assert!((rmst[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn test_deviance_residuals_match_martingale_formula() {
        let mut model = CoxPHModel::new();
        model.event_times = vec![1.0, 2.0, 3.0];
        model.censoring = vec![1, 0, 1];
        model.baseline_hazard = vec![0.2, 0.3, 0.4];
        model.risk_scores = vec![2.0, 1.5, 0.5];

        let expected: Vec<f64> = model
            .martingale_residuals()
            .iter()
            .zip(model.censoring.iter())
            .map(|(&martingale, &status)| {
                let status = status as f64;
                let sign = if martingale >= 0.0 { 1.0 } else { -1.0 };
                let abs_term = -2.0
                    * (martingale - status
                        + status * (status - martingale).ln().max(EXP_CLAMP_MIN));
                sign * abs_term.abs().sqrt()
            })
            .collect();

        let residuals = model.deviance_residuals();

        assert_eq!(residuals.len(), expected.len());
        for (actual, expected) in residuals.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dfbeta_computes_martingale_rows_directly() {
        let mut model = CoxPHModel::new();
        model.coefficients =
            Array2::from_shape_vec((1, 1), vec![0.0]).expect("coefficient shape is valid");
        model.covariates =
            Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("covariate shape is valid");
        model.event_times = vec![1.0, 2.0, 3.0];
        model.censoring = vec![1, 0, 1];
        model.baseline_hazard = vec![0.2, 0.3, 0.4];
        model.risk_scores = vec![2.0, 1.5, 0.5];

        let dfbeta = model.dfbeta();

        assert_eq!(dfbeta.len(), 3);
        assert!((dfbeta[0][0] + 0.1875).abs() < 1e-12);
        assert!((dfbeta[1][0] - 0.075).abs() < 1e-12);
        assert!(dfbeta[2][0].abs() < 1e-12);
    }

    #[test]
    fn test_new_with_data_rejects_ragged_and_invalid_inputs() {
        assert!(
            CoxPHModel::new_with_data(vec![vec![1.0], vec![]], vec![1.0, 2.0], vec![1, 0]).is_err()
        );
        assert!(CoxPHModel::new_with_data(vec![vec![1.0]], vec![f64::INFINITY], vec![1]).is_err());
        assert!(CoxPHModel::new_with_data(vec![vec![1.0]], vec![1.0], vec![2]).is_err());
        assert!(CoxPHModel::new_with_data(vec![vec![1.0]], vec![1.0, 2.0], vec![1]).is_err());
    }

    #[test]
    fn test_prediction_rows_are_validated() {
        let model = CoxPHModel::new_with_data(vec![vec![0.0, 1.0]], vec![1.0], vec![1])
            .expect("valid model data should construct");

        assert!(model.predict(vec![vec![1.0]]).is_err());
        assert!(model.predict(vec![vec![1.0, f64::NAN]]).is_err());
        assert!(model.cumulative_hazard(vec![vec![1.0]]).is_err());
    }

    #[test]
    fn test_large_coefficients_produce_finite_hazard_ratios() {
        let mut model = CoxPHModel::new();
        model.coefficients = Array2::from_shape_vec((2, 1), vec![1000.0, -1000.0])
            .expect("coefficient shape is valid");

        let hazard_ratios = model.hazard_ratios();
        assert!(hazard_ratios.iter().all(|value| value.is_finite()));
        assert!(hazard_ratios[0] > 0.0);
        assert!(hazard_ratios[1] > 0.0);

        let (hr, ci_lower, ci_upper) = model.hazard_ratios_with_ci(0.95);
        assert!(
            hr.iter()
                .chain(ci_lower.iter())
                .chain(ci_upper.iter())
                .all(|value| value.is_finite())
        );
    }
}
