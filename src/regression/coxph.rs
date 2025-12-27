use crate::regression::coxfit6::{CoxFit, Method as CoxMethod};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
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
            coefficients: Array2::<f64>::zeros((1, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times: Vec::new(),
            censoring: Vec::new(),
            covariates: Array2::<f64>::zeros((1, 1)),
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
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (i, row) in covariates.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                cov_array[[i, j]] = val;
            }
        }
        Self {
            coefficients: Array2::<f64>::zeros((ncols, 1)),
            baseline_hazard: Vec::new(),
            risk_scores: Vec::new(),
            event_times,
            censoring,
            covariates: cov_array,
        }
    }

    pub fn add_subject(&mut self, subject: &Subject) {
        let n = self.event_times.len();
        let ncols = self.covariates.ncols();

        if ncols != subject.covariates.len() {
            return;
        }

        let mut new_covariates = Array2::<f64>::zeros((n + 1, ncols));
        for row_idx in 0..n {
            for col_idx in 0..ncols {
                new_covariates[[row_idx, col_idx]] = self.covariates[[row_idx, col_idx]];
            }
        }
        for col_idx in 0..ncols {
            new_covariates[[n, col_idx]] = subject.covariates[col_idx];
        }

        self.covariates = new_covariates;
        self.event_times.push(0.0);
        self.censoring.push(if subject.is_case { 1 } else { 0 });
    }

    #[pyo3(signature = (n_iters = 20))]
    pub fn fit(&mut self, n_iters: u16) {
        if self.event_times.is_empty() || self.covariates.nrows() == 0 {
            return;
        }

        let n = self.event_times.len();
        let nvar = self.covariates.ncols();

        if nvar == 0 {
            return;
        }

        let time_array = Array1::from_vec(self.event_times.clone());
        let status_array: Array1<i32> =
            Array1::from_vec(self.censoring.iter().map(|&x| x as i32).collect());
        let strata = Array1::zeros(n);
        let offset = Array1::zeros(n);
        let weights = Array1::from_elem(n, 1.0);

        let initial_beta: Vec<f64> =
            if self.coefficients.nrows() == nvar && self.coefficients.ncols() > 0 {
                self.coefficients.column(0).to_vec()
            } else {
                vec![0.0; nvar]
            };

        let mut cox_fit = match CoxFit::new(
            time_array,
            status_array,
            self.covariates.clone(),
            strata,
            offset,
            weights,
            CoxMethod::Breslow,
            n_iters as usize,
            1e-5,
            1e-9,
            vec![true; nvar],
            initial_beta,
        ) {
            Ok(fit) => fit,
            Err(_) => return,
        };

        if cox_fit.fit().is_err() {
            return;
        }

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
    }

    fn calculate_baseline_hazard(&mut self) {
        let n = self.event_times.len();
        if n == 0 {
            self.baseline_hazard = Vec::new();
            return;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            self.event_times[i]
                .partial_cmp(&self.event_times[j])
                .unwrap()
                .then_with(|| self.censoring[j].cmp(&self.censoring[i]))
        });

        let mut unique_times = Vec::new();
        let mut baseline_hazard = Vec::new();
        let mut cum_hazard = 0.0;

        for &idx in &indices {
            if self.censoring[idx] == 0 {
                continue;
            }

            let current_time = self.event_times[idx];

            let should_add = if unique_times.is_empty() {
                true
            } else {
                let last_time: f64 = unique_times[unique_times.len() - 1];
                (current_time - last_time).abs() > 1e-9
            };

            if should_add {
                let mut events = 0.0;
                let mut risk_sum = 0.0;

                for &j in &indices {
                    if self.event_times[j] >= current_time {
                        risk_sum += self.risk_scores[j];
                        if self.event_times[j] == current_time && self.censoring[j] == 1 {
                            events += 1.0;
                        }
                    }
                }

                if risk_sum > 0.0 {
                    let hazard = events / risk_sum;
                    cum_hazard += hazard;
                }

                unique_times.push(current_time);
                baseline_hazard.push(cum_hazard);
            }
        }

        if baseline_hazard.is_empty() {
            self.baseline_hazard = vec![0.0; n];
        } else {
            let mut full_baseline = vec![0.0; n];
            for (i, &t) in self.event_times.iter().enumerate() {
                let mut closest_hazard = 0.0;
                for (j, &ut) in unique_times.iter().enumerate() {
                    if ut <= t {
                        closest_hazard = baseline_hazard[j];
                    } else {
                        break;
                    }
                }
                full_baseline[i] = closest_hazard;
            }
            self.baseline_hazard = full_baseline;
        }
    }

    pub fn predict(&self, covariates: Vec<Vec<f64>>) -> Vec<f64> {
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (row_idx, row) in covariates.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                cov_array[[row_idx, col_idx]] = val;
            }
        }
        let mut risk_scores = Vec::new();
        for row in cov_array.outer_iter() {
            let risk_score = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk_score);
        }
        risk_scores
    }

    #[getter]
    pub fn get_coefficients(&self) -> Vec<Vec<f64>> {
        let mut result = Vec::new();
        for col in self.coefficients.columns() {
            result.push(col.iter().cloned().collect());
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

        let baseline_haz = self
            .baseline_hazard
            .iter()
            .zip(&self.event_times)
            .filter(|&(_, &et)| et <= time)
            .map(|(h, _)| *h)
            .next_back()
            .unwrap_or(0.0);

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
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (row_idx, row) in covariates.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                cov_array[[row_idx, col_idx]] = val;
            }
        }

        let times = time_points.unwrap_or_else(|| {
            let mut t = self.event_times.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t.dedup();
            t
        });

        let mut risk_scores = Vec::new();
        for row in cov_array.outer_iter() {
            let risk = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk.exp());
        }

        let mut cum_baseline_haz = Vec::new();
        let mut cum = 0.0;
        for &haz in &self.baseline_hazard {
            cum += haz;
            cum_baseline_haz.push(cum);
        }

        let mut survival_curves = Vec::new();
        for risk_exp in &risk_scores {
            let mut surv = Vec::new();
            for &t in &times {
                let baseline_haz = self
                    .baseline_hazard
                    .iter()
                    .zip(&self.event_times)
                    .filter(|&(_, et)| *et <= t)
                    .map(|(h, _)| *h)
                    .sum::<f64>();

                let s = (-baseline_haz * risk_exp).exp();
                surv.push(s);
            }
            survival_curves.push(surv);
        }

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

        let z = if confidence_level >= 0.99 {
            2.576
        } else if confidence_level >= 0.95 {
            1.96
        } else {
            1.645
        };

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

        let mut fisher_diag = vec![0.0; nvar];

        for i in 0..n {
            if self.censoring[i] == 0 {
                continue;
            }

            let risk_set_sum: f64 = (0..n)
                .filter(|&j| self.event_times[j] >= self.event_times[i])
                .map(|j| self.risk_scores.get(j).copied().unwrap_or(1.0))
                .sum();

            if risk_set_sum <= 0.0 {
                continue;
            }

            for (k, fisher_k) in fisher_diag.iter_mut().enumerate() {
                let mut weighted_cov = 0.0;
                let mut weighted_cov_sq = 0.0;

                for j in 0..n {
                    if self.event_times[j] >= self.event_times[i] {
                        let risk_j = self.risk_scores.get(j).copied().unwrap_or(1.0);
                        let cov_jk = self.covariates.get([j, k]).copied().unwrap_or(0.0);
                        weighted_cov += risk_j * cov_jk;
                        weighted_cov_sq += risk_j * cov_jk * cov_jk;
                    }
                }

                let mean_cov = weighted_cov / risk_set_sum;
                let var_cov = weighted_cov_sq / risk_set_sum - mean_cov * mean_cov;
                *fisher_k += var_cov;
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
        let mut loglik = 0.0;

        for i in 0..n {
            if self.censoring[i] == 0 {
                continue;
            }

            let risk_score_i = self.risk_scores.get(i).copied().unwrap_or(1.0).ln();

            let risk_set_sum: f64 = (0..n)
                .filter(|&j| self.event_times[j] >= self.event_times[i])
                .map(|j| self.risk_scores.get(j).copied().unwrap_or(1.0))
                .sum();

            if risk_set_sum > 0.0 {
                loglik += risk_score_i - risk_set_sum.ln();
            }
        }

        loglik
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
        let nrows = covariates.len();
        let ncols = if nrows > 0 { covariates[0].len() } else { 0 };
        let mut cov_array = Array2::<f64>::zeros((nrows, ncols));
        for (row_idx, row) in covariates.iter().enumerate() {
            for (col_idx, &val) in row.iter().enumerate() {
                cov_array[[row_idx, col_idx]] = val;
            }
        }

        let mut unique_times: Vec<f64> = self.event_times.clone();
        unique_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_times.dedup();

        let mut risk_scores = Vec::new();
        for row in cov_array.outer_iter() {
            let risk = self.coefficients.column(0).dot(&row);
            risk_scores.push(risk.exp());
        }

        let mut cumulative_hazards = Vec::new();
        for risk_exp in &risk_scores {
            let mut cum_haz = Vec::new();
            for &t in &unique_times {
                let baseline_haz = self
                    .baseline_hazard
                    .iter()
                    .zip(&self.event_times)
                    .filter(|&(_, et)| *et <= t)
                    .map(|(h, _)| *h)
                    .sum::<f64>();
                cum_haz.push(baseline_haz * risk_exp);
            }
            cumulative_hazards.push(cum_haz);
        }

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
                let abs_term = -2.0 * (m - d as f64 + d as f64 * (d as f64 - m).ln().max(-100.0));
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
        let mut dfbeta = vec![vec![0.0; nvar]; n];

        for (i, (&mart_i, dfbeta_row)) in martingale.iter().zip(dfbeta.iter_mut()).enumerate() {
            for (k, dfbeta_cell) in dfbeta_row.iter_mut().enumerate() {
                let cov_ik = self.covariates.get([i, k]).copied().unwrap_or(0.0);
                let risk_i = self.risk_scores.get(i).copied().unwrap_or(1.0);

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

                *dfbeta_cell = mart_i * (cov_ik - weighted_mean) / risk_i.max(1e-10);
            }
        }

        dfbeta
    }

    pub fn n_events(&self) -> usize {
        self.censoring.iter().filter(|&&c| c == 1).count()
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

        let mut result = String::new();
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
