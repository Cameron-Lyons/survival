use crate::constants::{EXP_CLAMP_MIN, z_score_for_confidence};
use crate::internal::matrix::invert_matrix;
use crate::regression::cox_optimizer::{CoxFitBuilder, Method as CoxMethod};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rayon::prelude::*;
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

    fn compute_exp_risk_scores(&self, covariates: &[Vec<f64>]) -> Vec<f64> {
        let ncoef = self.coefficients.nrows();
        covariates
            .par_iter()
            .map(|row| {
                let mut risk = 0.0;
                for col_idx in 0..ncoef {
                    let cov = row.get(col_idx).copied().unwrap_or(0.0);
                    risk += self.coefficients[[col_idx, 0]] * cov;
                }
                risk.exp()
            })
            .collect()
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
    ) -> Self {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut covariates_flat = Vec::with_capacity(nrows * ncols);
        for row in &covariates {
            for j in 0..ncols {
                covariates_flat.push(row.get(j).copied().unwrap_or(0.0));
            }
        }
        let cov_array = Array2::from_shape_vec((nrows, ncols), covariates_flat.clone())
            .expect("covariate shape and flat data length are consistent");
        Self {
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
        }
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
        let mut coefficients_array = Array2::<f64>::zeros((nvar, 1));
        for (idx, &beta_val) in beta.iter().enumerate() {
            coefficients_array[[idx, 0]] = beta_val;
        }
        self.coefficients = coefficients_array;
        self.risk_scores.clear();
        for row in self.covariates.outer_iter() {
            let risk_score = self.coefficients.column(0).dot(&row);
            self.risk_scores.push(risk_score.exp());
        }
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
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            self.event_times[i]
                .partial_cmp(&self.event_times[j])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| self.censoring[j].cmp(&self.censoring[i]))
        });
        let mut cumulative_risk = vec![0.0; n];
        let mut running_sum = 0.0;
        for i in (0..n).rev() {
            running_sum += self.risk_scores[indices[i]];
            cumulative_risk[i] = running_sum;
        }
        let n_events_estimate = self.censoring.iter().filter(|&&c| c == 1).count();
        let mut unique_times = Vec::with_capacity(n_events_estimate);
        let mut baseline_hazard = Vec::with_capacity(n_events_estimate);
        let mut cum_hazard = 0.0;
        let mut i = 0;
        while i < n {
            let idx = indices[i];
            if self.censoring[idx] == 0 {
                i += 1;
                continue;
            }
            let current_time = self.event_times[idx];
            let mut events = 0.0;
            let start_i = i;
            while i < n
                && (self.event_times[indices[i]] - current_time).abs()
                    < crate::constants::TIME_EPSILON
            {
                if self.censoring[indices[i]] == 1 {
                    events += 1.0;
                }
                i += 1;
            }
            let risk_sum = cumulative_risk[start_i];
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
    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        covariates
            .par_iter()
            .map(|row| {
                let mut risk_score = 0.0;
                for (col_idx, &val) in row.iter().enumerate().take(ncols) {
                    if col_idx < self.coefficients.nrows() {
                        risk_score += self.coefficients[[col_idx, 0]] * val;
                    }
                }
                risk_score
            })
            .collect()
    }
    #[getter]
    pub fn get_coefficients(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for col in self.coefficients.columns() {
            result.push(col.iter().copied().collect());
        }
        result
    }
    pub fn brier_score(&self) -> f64 {
        let mut score = 0.0;
        let mut count = 0.0;
        for (time, &status) in self.event_times.iter().zip(self.censoring.iter()) {
            let pred = self.predict_survival(*time);
            score += (pred - status as f64).powi(2);
            count += 1.0;
        }
        if count > 0.0 { score / count } else { 0.0 }
    }
    fn predict_survival(&self, time: f64) -> f64 {
        if self.baseline_hazard.is_empty() || self.risk_scores.is_empty() {
            return 0.5;
        }
        let baseline_haz = self.baseline_cumulative_hazard_at(time);
        let avg_risk = if !self.risk_scores.is_empty() {
            self.risk_scores.iter().sum::<f64>() / self.risk_scores.len() as f64
        } else {
            1.0
        };
        (-baseline_haz * avg_risk).exp()
    }
    pub fn survival_curve(
        &self,
        covariates: Vec<Vec<f64>>,
        time_points: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        let times = time_points.unwrap_or_else(|| {
            let mut t = self.event_times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t.dedup();
            t
        });
        let risk_scores = self.compute_exp_risk_scores(&covariates);
        let baseline_hazards: Vec<f64> = times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        let survival_curves: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk_exp| {
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
            .map(|&beta| beta.exp())
            .collect()
    }
    #[pyo3(signature = (confidence_level = 0.95))]
    pub fn hazard_ratios_with_ci(&self, confidence_level: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let coefs: Vec<f64> = self.coefficients.column(0).to_vec();
        let n = coefs.len();
        let z = z_score_for_confidence(confidence_level);
        let se = self.compute_standard_errors();
        let mut hr = Vec::with_capacity(n);
        let mut ci_lower = Vec::with_capacity(n);
        let mut ci_upper = Vec::with_capacity(n);
        for (i, &beta) in coefs.iter().enumerate() {
            let se_i = se.get(i).copied().unwrap_or(0.1);
            hr.push(beta.exp());
            ci_lower.push((beta - z * se_i).exp());
            ci_upper.push((beta + z * se_i).exp());
        }
        (hr, ci_lower, ci_upper)
    }
    fn compute_standard_errors(&self) -> Vec<f64> {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        if n == 0 || nvar == 0 {
            return vec![0.1; nvar];
        }
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| {
            self.event_times[j]
                .partial_cmp(&self.event_times[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumulative_risk = vec![0.0; n];
        let mut cumulative_weighted_cov = vec![0.0; n * nvar];
        let mut cumulative_weighted_cov_sq = vec![0.0; n * nvar];
        let mut running_risk = 0.0;
        let mut running_weighted_cov = vec![0.0; nvar];
        let mut running_weighted_cov_sq = vec![0.0; nvar];
        for (pos, &idx) in sorted_indices.iter().enumerate() {
            let risk_j = self.risk_scores.get(idx).copied().unwrap_or(1.0);
            running_risk += risk_j;
            for k in 0..nvar {
                let cov_jk = self.covariates.get([idx, k]).copied().unwrap_or(0.0);
                running_weighted_cov[k] += risk_j * cov_jk;
                running_weighted_cov_sq[k] += risk_j * cov_jk * cov_jk;
            }
            cumulative_risk[pos] = running_risk;
            let base = pos * nvar;
            cumulative_weighted_cov[base..base + nvar].copy_from_slice(&running_weighted_cov);
            cumulative_weighted_cov_sq[base..base + nvar].copy_from_slice(&running_weighted_cov_sq);
        }
        let mut index_to_pos = vec![0usize; n];
        for (pos, &idx) in sorted_indices.iter().enumerate() {
            index_to_pos[idx] = pos;
        }
        let event_indices: Vec<usize> = (0..n).filter(|&i| self.censoring[i] == 1).collect();
        let fisher_contributions: Vec<Vec<f64>> = event_indices
            .par_iter()
            .filter_map(|&i| {
                let pos = index_to_pos[i];
                let risk_set_sum = cumulative_risk[pos];
                if risk_set_sum <= 0.0 {
                    return None;
                }
                let base = pos * nvar;
                let contrib: Vec<f64> = (0..nvar)
                    .map(|k| {
                        let weighted_cov = cumulative_weighted_cov[base + k];
                        let weighted_cov_sq = cumulative_weighted_cov_sq[base + k];
                        let mean_cov = weighted_cov / risk_set_sum;
                        weighted_cov_sq / risk_set_sum - mean_cov * mean_cov
                    })
                    .collect();
                Some(contrib)
            })
            .collect();
        let mut fisher_diag = vec![0.0; nvar];
        for contrib in fisher_contributions {
            for (k, &val) in contrib.iter().enumerate() {
                fisher_diag[k] += val;
            }
        }
        fisher_diag
            .iter()
            .map(|&f| if f > 0.0 { (1.0 / f).sqrt() } else { 0.1 })
            .collect()
    }
    pub fn log_likelihood(&self) -> f64 {
        if self.event_times.is_empty() || self.risk_scores.is_empty() {
            return 0.0;
        }
        let n = self.event_times.len();
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| {
            self.event_times[j]
                .partial_cmp(&self.event_times[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumulative_risk = vec![0.0; n];
        let mut running_sum = 0.0;
        for (pos, &idx) in sorted_indices.iter().enumerate() {
            running_sum += self.risk_scores.get(idx).copied().unwrap_or(1.0);
            cumulative_risk[pos] = running_sum;
        }
        let mut index_to_pos = vec![0usize; n];
        for (pos, &idx) in sorted_indices.iter().enumerate() {
            index_to_pos[idx] = pos;
        }
        let event_indices: Vec<usize> = (0..n).filter(|&i| self.censoring[i] == 1).collect();
        event_indices
            .par_iter()
            .map(|&i| {
                let risk_score_i = self.risk_scores.get(i).copied().unwrap_or(1.0).ln();
                let pos = index_to_pos[i];
                let risk_set_sum = cumulative_risk[pos];
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
    pub fn cumulative_hazard(&self, covariates: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut unique_times: Vec<f64> = self.event_times.clone();
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_times.dedup();
        let risk_scores = self.compute_exp_risk_scores(&covariates);
        let baseline_hazards: Vec<f64> = unique_times
            .iter()
            .map(|&t| self.baseline_cumulative_hazard_at(t))
            .collect();
        let cumulative_hazards: Vec<Vec<f64>> = risk_scores
            .par_iter()
            .map(|&risk_exp| baseline_hazards.iter().map(|&bh| bh * risk_exp).collect())
            .collect();
        (unique_times, cumulative_hazards)
    }
    #[pyo3(signature = (covariates, percentile = 0.5))]
    pub fn predicted_survival_time(
        &self,
        covariates: Vec<Vec<f64>>,
        percentile: f64,
    ) -> Vec<Option<f64>> {
        let (times, survival_curves) = match self.survival_curve(covariates, None) {
            Ok(result) => result,
            Err(_) => return vec![],
        };
        let target_survival = 1.0 - percentile;
        survival_curves
            .iter()
            .map(|surv| {
                for (i, &s) in surv.iter().enumerate() {
                    if s <= target_survival {
                        if i == 0 {
                            return Some(times[0]);
                        }
                        let s0 = surv[i - 1];
                        let s1 = s;
                        let t0 = times[i - 1];
                        let t1 = times[i];
                        let frac = (s0 - target_survival) / (s0 - s1);
                        return Some(t0 + frac * (t1 - t0));
                    }
                }
                None
            })
            .collect()
    }
    pub fn restricted_mean_survival_time(&self, covariates: Vec<Vec<f64>>, tau: f64) -> Vec<f64> {
        let (times, survival_curves) = match self.survival_curve(covariates, None) {
            Ok(result) => result,
            Err(_) => return vec![],
        };
        survival_curves
            .iter()
            .map(|surv| {
                let mut rmst = 0.0;
                let mut prev_time = 0.0;
                let mut prev_surv = 1.0;
                for (i, &t) in times.iter().enumerate() {
                    if t > tau {
                        rmst += prev_surv * (tau - prev_time);
                        break;
                    }
                    rmst += prev_surv * (t - prev_time);
                    prev_time = t;
                    prev_surv = surv[i];
                    if i == times.len() - 1 {
                        rmst += prev_surv * (tau - t);
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
        let martingale = self.martingale_residuals();
        martingale
            .iter()
            .zip(self.censoring.iter())
            .map(|(&m, &d)| {
                let sign = if m >= 0.0 { 1.0 } else { -1.0 };
                let abs_term =
                    -2.0 * (m - d as f64 + d as f64 * (d as f64 - m).ln().max(EXP_CLAMP_MIN));
                sign * abs_term.abs().sqrt()
            })
            .collect()
    }
    pub fn dfbeta(&self) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let nvar = self.coefficients.nrows();
        if n == 0 || nvar == 0 {
            return vec![];
        }
        let martingale = self.martingale_residuals();
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mart_i = martingale[i];
                let risk_i = self.risk_scores.get(i).copied().unwrap_or(1.0);
                (0..nvar)
                    .map(|k| {
                        let cov_ik = self.covariates.get([i, k]).copied().unwrap_or(0.0);
                        let mut weighted_mean = 0.0;
                        let mut risk_sum = 0.0;
                        for j in 0..n {
                            if self.event_times[j] >= self.event_times[i] {
                                let risk_j = self.risk_scores.get(j).copied().unwrap_or(1.0);
                                let cov_jk = self.covariates.get([j, k]).copied().unwrap_or(0.0);
                                weighted_mean += risk_j * cov_jk;
                                risk_sum += risk_j;
                            }
                        }
                        if risk_sum > 0.0 {
                            weighted_mean /= risk_sum;
                        }
                        mart_i * (cov_ik - weighted_mean)
                            / risk_i.max(crate::constants::DIVISION_FLOOR)
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
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&i, &j| {
            self.event_times[j]
                .partial_cmp(&self.event_times[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut fisher = vec![vec![0.0; nvar]; nvar];
        let mut index_to_pos = vec![0usize; n];
        for (pos, &idx) in sorted_indices.iter().enumerate() {
            index_to_pos[idx] = pos;
        }
        let mut cumulative_risk_sum = vec![0.0; n];
        let mut cumulative_weighted_cov = vec![0.0; n * nvar];
        let mut cumulative_weighted_outer = vec![0.0; n * nvar * nvar];

        let mut risk_sum = 0.0;
        let mut weighted_cov = vec![0.0; nvar];
        let mut weighted_outer = vec![0.0; nvar * nvar];

        for (pos, &idx) in sorted_indices.iter().enumerate() {
            let risk_i = self.risk_scores.get(idx).copied().unwrap_or(1.0);
            risk_sum += risk_i;
            for k in 0..nvar {
                let cov_ik = self.covariates.get([idx, k]).copied().unwrap_or(0.0);
                weighted_cov[k] += risk_i * cov_ik;
                for l in 0..nvar {
                    let cov_il = self.covariates.get([idx, l]).copied().unwrap_or(0.0);
                    weighted_outer[k * nvar + l] += risk_i * cov_ik * cov_il;
                }
            }
            cumulative_risk_sum[pos] = risk_sum;
            cumulative_weighted_cov[pos * nvar..(pos + 1) * nvar].copy_from_slice(&weighted_cov);
            cumulative_weighted_outer[pos * nvar * nvar..(pos + 1) * nvar * nvar]
                .copy_from_slice(&weighted_outer);
        }
        for (i, &censor) in self.censoring.iter().enumerate() {
            if censor != 1 {
                continue;
            }
            let pos = index_to_pos[i];
            let rs = cumulative_risk_sum[pos];
            if rs <= 0.0 {
                continue;
            }
            let wc_start = pos * nvar;
            let wco_start = pos * nvar * nvar;
            for k in 0..nvar {
                let wc_k = cumulative_weighted_cov[wc_start + k];
                for l in 0..nvar {
                    let wc_l = cumulative_weighted_cov[wc_start + l];
                    let wco_kl = cumulative_weighted_outer[wco_start + k * nvar + l];
                    let info_kl = wco_kl / rs - (wc_k / rs) * (wc_l / rs);
                    fisher[k][l] += info_kl;
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

        let (times, hazards) = model.cumulative_hazard(vec![vec![0.0]]);
        assert_eq!(times, vec![1.0, 2.0, 3.0]);
        assert_eq!(hazards.len(), 1);
        assert!((hazards[0][0] - 0.1).abs() < 1e-12);
        assert!((hazards[0][1] - 0.2).abs() < 1e-12);
        assert!((hazards[0][2] - 0.3).abs() < 1e-12);
    }
}
