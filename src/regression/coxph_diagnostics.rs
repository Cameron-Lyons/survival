use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN};
use crate::regression::cox_optimizer::{Method as CoxMethod, exact_tied_moments};
use crate::regression::coxph::CoxPHFit;
use crate::regression::coxph_support::{
    ActiveRiskSet, CoxSweepRow, StratifiedBaselineLookup, cumulative_step_at,
};
use crate::scoring::coxscore2::{CoxScoreData, CoxScoreParams, compute_cox_score_residuals};
use ndarray::Array2;
use pyo3::prelude::*;
use std::cmp::Ordering;

impl CoxPHFit {
    pub(crate) fn expected_events_internal(&self) -> PyResult<Vec<f64>> {
        let (times, hazards, hazard_strata) = self.basehaz_with_strata_internal(false)?;
        let baseline = StratifiedBaselineLookup::from_components(&times, &hazards, &hazard_strata);
        let entry_times = self.entry_times.as_deref();
        let row_strata = self.row_strata();

        Ok(self
            .event_times
            .iter()
            .enumerate()
            .map(|(idx, &stop)| {
                let start_hazard = entry_times
                    .map(|starts| baseline.cumulative_hazard_at(row_strata[idx], starts[idx]))
                    .unwrap_or(0.0);
                let stop_hazard = baseline.cumulative_hazard_at(row_strata[idx], stop);
                let interval_hazard = (stop_hazard - start_hazard).max(0.0);
                let risk_multiplier = self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp();
                interval_hazard * risk_multiplier
            })
            .collect())
    }

    pub(crate) fn tie_method(&self) -> CoxMethod {
        match self.method.as_str() {
            "exact" => CoxMethod::Exact,
            "efron" => CoxMethod::Efron,
            _ => CoxMethod::Breslow,
        }
    }

    fn exact_inclusion_weights(
        risk_indices: &[usize],
        deaths: usize,
        risk: &[f64],
    ) -> Option<Vec<(usize, f64)>> {
        if deaths == 0 || deaths > risk_indices.len() {
            return None;
        }
        let n_risk = risk_indices.len();
        let mut prefix = vec![vec![0.0; deaths + 1]; n_risk + 1];
        prefix[0][0] = 1.0;
        for pos in 0..n_risk {
            prefix[pos + 1] = prefix[pos].clone();
            let value = risk[risk_indices[pos]];
            for size in 1..=deaths.min(pos + 1) {
                prefix[pos + 1][size] += value * prefix[pos][size - 1];
            }
        }
        let mut suffix = vec![vec![0.0; deaths + 1]; n_risk + 1];
        suffix[n_risk][0] = 1.0;
        for pos in (0..n_risk).rev() {
            suffix[pos] = suffix[pos + 1].clone();
            let value = risk[risk_indices[pos]];
            for size in 1..=deaths.min(n_risk - pos) {
                suffix[pos][size] += value * suffix[pos + 1][size - 1];
            }
        }
        let denom = prefix[n_risk][deaths];
        if denom <= 0.0 {
            return None;
        }
        Some(
            (0..n_risk)
                .map(|pos| {
                    let mut excluded = 0.0;
                    for (left_size, prefix_value) in prefix[pos].iter().take(deaths).enumerate() {
                        let right_size = deaths - 1 - left_size;
                        excluded += *prefix_value * suffix[pos + 1][right_size];
                    }
                    (
                        risk_indices[pos],
                        risk[risk_indices[pos]] * excluded / denom,
                    )
                })
                .collect(),
        )
    }

    pub(crate) fn score_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let n = self.event_times.len();
        if nvar == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let tie_method = self.tie_method();
        let method = match tie_method {
            CoxMethod::Breslow => 0,
            CoxMethod::Efron => 1,
            CoxMethod::Exact => 2,
        };
        if self.covariates.len() != n
            || self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }
        if self.entry_times.is_some() {
            return self.score_residuals_counting_process(nvar, method);
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&lhs, &rhs| {
            self.strata[lhs]
                .cmp(&self.strata[rhs])
                .then_with(|| {
                    self.event_times[lhs]
                        .partial_cmp(&self.event_times[rhs])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| lhs.cmp(&rhs))
        });

        let mut y = Vec::with_capacity(2 * n);
        y.extend(order.iter().map(|&idx| self.event_times[idx]));
        y.extend(order.iter().map(|&idx| self.status[idx] as f64));
        let strata: Vec<i32> = order.iter().map(|&idx| self.strata[idx]).collect();
        let score: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
            })
            .collect();
        let weights: Vec<f64> = order.iter().map(|&idx| self.weights[idx]).collect();
        let mut covar = Vec::with_capacity(n * nvar);
        for &idx in &order {
            covar.extend(self.covariates[idx].iter().copied());
        }

        if matches!(tie_method, CoxMethod::Exact) {
            return Ok(self.score_residuals_exact_right_censored(nvar, &order, &score, &weights));
        }

        let flat = compute_cox_score_residuals(
            CoxScoreData {
                y: &y,
                strata: &strata,
                covar: &covar,
                score: &score,
                weights: &weights,
            },
            CoxScoreParams { method, n, nvar },
        );
        let mut residuals = vec![vec![0.0; nvar]; n];
        for (sorted_idx, &original_idx) in order.iter().enumerate() {
            for col_idx in 0..nvar {
                residuals[original_idx][col_idx] = flat[sorted_idx * nvar + col_idx];
            }
        }
        Ok(residuals)
    }

    fn score_residuals_exact_right_censored(
        &self,
        nvar: usize,
        order: &[usize],
        score: &[f64],
        weights: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let risk: Vec<f64> = order
            .iter()
            .enumerate()
            .map(|(sorted_idx, _)| score[sorted_idx] * weights[sorted_idx])
            .collect();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }
            let mut risk_indices: Vec<usize> = Vec::new();
            let mut time_pos = stratum_end;
            loop {
                let event_time = self.event_times[order[time_pos]];
                let mut time_start = time_pos;
                while time_start > stratum_start
                    && self.event_times[order[time_start - 1]] == event_time
                {
                    time_start -= 1;
                }
                for sorted_idx in time_start..=time_pos {
                    risk_indices.push(sorted_idx);
                }
                let deaths: Vec<usize> = (time_start..=time_pos)
                    .filter(|&idx| self.status[order[idx]] == 1)
                    .collect();
                if !deaths.is_empty() {
                    for &sorted_idx in &deaths {
                        let original_idx = order[sorted_idx];
                        let weight = self.weights[original_idx];
                        for (col_idx, residual) in
                            residuals[original_idx].iter_mut().enumerate().take(nvar)
                        {
                            *residual += weight * self.covariates[original_idx][col_idx];
                        }
                    }
                    if let Some(inclusion_weights) =
                        Self::exact_inclusion_weights(&risk_indices, deaths.len(), &risk)
                    {
                        for (sorted_idx, inclusion_weight) in inclusion_weights {
                            let original_idx = order[sorted_idx];
                            for (col_idx, residual) in
                                residuals[original_idx].iter_mut().enumerate().take(nvar)
                            {
                                *residual -=
                                    inclusion_weight * self.covariates[original_idx][col_idx];
                            }
                        }
                    }
                }
                if time_start == stratum_start {
                    break;
                }
                time_pos = time_start - 1;
            }
            stratum_start = stratum_end + 1;
        }
        residuals
    }

    fn score_residuals_counting_process(
        &self,
        nvar: usize,
        method: i32,
    ) -> PyResult<Vec<Vec<f64>>> {
        let n = self.event_times.len();
        let Some(entry_times) = self.entry_times.as_ref() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "counting-process score residuals require entry times",
            ));
        };
        if entry_times.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model entry times have inconsistent length",
            ));
        }

        let risk: Vec<f64> = (0..n)
            .map(|idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
                    * self.weights[idx]
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&lhs, &rhs| {
            self.strata[lhs]
                .cmp(&self.strata[rhs])
                .then_with(|| {
                    self.event_times[lhs]
                        .partial_cmp(&self.event_times[rhs])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| lhs.cmp(&rhs))
        });

        if method == 2 {
            return Ok(self.score_residuals_counting_process_by_scan(
                nvar,
                method,
                &risk,
                &order,
                entry_times,
            ));
        }

        Ok(self.score_residuals_counting_process_sweep(nvar, method, &risk, &order, entry_times))
    }

    pub(crate) fn score_residuals_counting_process_by_scan(
        &self,
        nvar: usize,
        method: i32,
        risk: &[f64],
        order: &[usize],
        entry_times: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }
            let stratum_indices = &order[stratum_start..=stratum_end];
            let mut time_start = stratum_start;
            while time_start <= stratum_end {
                let event_time = self.event_times[order[time_start]];
                let mut time_end = time_start;
                while time_end < stratum_end && self.event_times[order[time_end + 1]] == event_time
                {
                    time_end += 1;
                }

                let deaths: Vec<usize> = (time_start..=time_end)
                    .map(|idx| order[idx])
                    .filter(|&idx| self.status[idx] == 1)
                    .collect();
                if !deaths.is_empty() {
                    let mut is_death = vec![false; n];
                    for &idx in &deaths {
                        is_death[idx] = true;
                    }
                    let risk_indices: Vec<usize> = stratum_indices
                        .iter()
                        .copied()
                        .filter(|&idx| {
                            entry_times[idx] < event_time && self.event_times[idx] >= event_time
                        })
                        .collect();
                    let denom: f64 = risk_indices.iter().map(|&idx| risk[idx]).sum();
                    if denom > 0.0 {
                        for &idx in &deaths {
                            for (col_idx, residual) in
                                residuals[idx].iter_mut().enumerate().take(nvar)
                            {
                                *residual += self.weights[idx] * self.covariates[idx][col_idx];
                            }
                        }
                        if method == 2 {
                            if let Some(inclusion_weights) =
                                Self::exact_inclusion_weights(&risk_indices, deaths.len(), risk)
                            {
                                for (idx, inclusion_weight) in inclusion_weights {
                                    for (col_idx, residual) in
                                        residuals[idx].iter_mut().enumerate().take(nvar)
                                    {
                                        *residual -=
                                            inclusion_weight * self.covariates[idx][col_idx];
                                    }
                                }
                            }
                        } else if method == 0 || deaths.len() == 1 {
                            let deadwt: f64 = deaths.iter().map(|&idx| self.weights[idx]).sum();
                            for &idx in &risk_indices {
                                let factor = risk[idx] * deadwt / denom;
                                for (col_idx, residual) in
                                    residuals[idx].iter_mut().enumerate().take(nvar)
                                {
                                    *residual -= factor * self.covariates[idx][col_idx];
                                }
                            }
                        } else {
                            let death_count = deaths.len();
                            let deaths_f = death_count as f64;
                            let deadwt: f64 = deaths.iter().map(|&idx| self.weights[idx]).sum();
                            let weight_average = deadwt / deaths_f;
                            let death_risk: f64 = deaths.iter().map(|&idx| risk[idx]).sum();
                            for step in 0..death_count {
                                let fraction = step as f64 / deaths_f;
                                let step_denom = denom - fraction * death_risk;
                                if step_denom <= 0.0 {
                                    continue;
                                }
                                for &idx in &risk_indices {
                                    let multiplier =
                                        if is_death[idx] { 1.0 - fraction } else { 1.0 };
                                    let factor =
                                        weight_average * risk[idx] * multiplier / step_denom;
                                    for (col_idx, residual) in
                                        residuals[idx].iter_mut().enumerate().take(nvar)
                                    {
                                        *residual -= factor * self.covariates[idx][col_idx];
                                    }
                                }
                            }
                        }
                    }
                }

                if time_end == stratum_end {
                    break;
                }
                time_start = time_end + 1;
            }
            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn score_residuals_counting_process_sweep(
        &self,
        nvar: usize,
        method: i32,
        risk: &[f64],
        order: &[usize],
        entry_times: &[f64],
    ) -> Vec<Vec<f64>> {
        let n = self.event_times.len();
        let mut residuals = vec![vec![0.0; nvar]; n];
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }
            let stratum_order = &order[stratum_start..=stratum_end];
            let rows: Vec<CoxSweepRow> = stratum_order
                .iter()
                .map(|&idx| CoxSweepRow {
                    original_idx: idx,
                    stop: self.event_times[idx],
                    entry: entry_times[idx],
                    risk: risk[idx],
                    weight: self.weights[idx],
                    status: self.status[idx],
                })
                .collect();
            let mut active = ActiveRiskSet::new(&rows, true);
            let mut event_times = Vec::new();
            let mut cumulative_scalars = Vec::new();
            let mut cumulative_scalar = 0.0;
            let mut time_start = 0usize;
            while time_start < rows.len() {
                let event_time = rows[time_start].stop;
                let mut time_end = time_start;
                while time_end + 1 < rows.len() && rows[time_end + 1].stop == event_time {
                    time_end += 1;
                }

                active.advance_to(event_time, |_, _| {});

                let deaths: Vec<usize> = (time_start..=time_end)
                    .filter(|&idx| rows[idx].status == 1)
                    .collect();
                if !deaths.is_empty() && active.risk_sum > 0.0 {
                    for &row_idx in &deaths {
                        let original_idx = rows[row_idx].original_idx;
                        for (col_idx, residual) in
                            residuals[original_idx].iter_mut().enumerate().take(nvar)
                        {
                            *residual +=
                                rows[row_idx].weight * self.covariates[original_idx][col_idx];
                        }
                    }

                    let deadwt: f64 = deaths.iter().map(|&idx| rows[idx].weight).sum();
                    let (scalar, death_adjustment) = if method == 1 && deaths.len() > 1 {
                        let death_count = deaths.len();
                        let deaths_f = death_count as f64;
                        let weight_average = deadwt / deaths_f;
                        let death_risk: f64 = deaths.iter().map(|&idx| rows[idx].risk).sum();
                        let mut all_active_scalar = 0.0;
                        let mut death_scalar = 0.0;
                        for step in 0..death_count {
                            let fraction = step as f64 / deaths_f;
                            let step_denom = active.risk_sum - fraction * death_risk;
                            if step_denom <= 0.0 {
                                continue;
                            }
                            let step_scalar = weight_average / step_denom;
                            all_active_scalar += step_scalar;
                            death_scalar += step_scalar * (1.0 - fraction);
                        }
                        (all_active_scalar, all_active_scalar - death_scalar)
                    } else {
                        (deadwt / active.risk_sum, 0.0)
                    };

                    cumulative_scalar += scalar;
                    event_times.push(event_time);
                    cumulative_scalars.push(cumulative_scalar);

                    if death_adjustment != 0.0 {
                        for &row_idx in &deaths {
                            let original_idx = rows[row_idx].original_idx;
                            for (col_idx, residual) in
                                residuals[original_idx].iter_mut().enumerate().take(nvar)
                            {
                                *residual += rows[row_idx].risk
                                    * death_adjustment
                                    * self.covariates[original_idx][col_idx];
                            }
                        }
                    }
                }

                time_start = time_end + 1;
            }

            for row in &rows {
                let start_scalar = cumulative_step_at(&event_times, &cumulative_scalars, row.entry);
                let stop_scalar = cumulative_step_at(&event_times, &cumulative_scalars, row.stop);
                let active_scalar = stop_scalar - start_scalar;
                if active_scalar != 0.0 {
                    for (col_idx, residual) in residuals[row.original_idx]
                        .iter_mut()
                        .enumerate()
                        .take(nvar)
                    {
                        *residual -=
                            row.risk * active_scalar * self.covariates[row.original_idx][col_idx];
                    }
                }
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn dfbeta_from_score_residuals(&self, scaled: bool) -> PyResult<Vec<Vec<f64>>> {
        let score_residuals = self.score_residuals_internal()?;
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        if nvar == 0 {
            return Ok(score_residuals);
        }
        if self.information_matrix.len() != nvar
            || self.information_matrix.iter().any(|row| row.len() != nvar)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model information matrix does not match coefficient width",
            ));
        }
        let scale: Vec<f64> = if scaled {
            (0..nvar)
                .map(|idx| {
                    self.information_matrix[idx][idx]
                        .abs()
                        .sqrt()
                        .max(crate::constants::DIVISION_FLOOR)
                })
                .collect()
        } else {
            vec![1.0; nvar]
        };
        Ok(score_residuals
            .iter()
            .map(|row| {
                (0..nvar)
                    .map(|col_idx| {
                        let value = (0..nvar)
                            .map(|inner_idx| {
                                self.information_matrix[col_idx][inner_idx] * row[inner_idx]
                            })
                            .sum::<f64>();
                        value / scale[col_idx]
                    })
                    .collect()
            })
            .collect())
    }

    pub(crate) fn scaled_schoenfeld_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let schoenfeld = self.schoenfeld_residuals_internal()?;
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        if nvar == 0 || schoenfeld.is_empty() {
            return Ok(schoenfeld);
        }
        if self.information_matrix.len() != nvar
            || self.information_matrix.iter().any(|row| row.len() != nvar)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model information matrix does not match coefficient width",
            ));
        }
        if schoenfeld.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model Schoenfeld residuals do not match coefficient width",
            ));
        }

        let event_count = schoenfeld.len() as f64;
        Ok(schoenfeld
            .iter()
            .map(|row| {
                (0..nvar)
                    .map(|col_idx| {
                        beta[col_idx]
                            + event_count
                                * (0..nvar)
                                    .map(|inner_idx| {
                                        row[inner_idx] * self.information_matrix[inner_idx][col_idx]
                                    })
                                    .sum::<f64>()
                    })
                    .collect()
            })
            .collect())
    }

    pub(crate) fn schoenfeld_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let n = self.event_times.len();
        if nvar == 0 {
            return Ok(Vec::new());
        }
        if self.covariates.len() != n
            || self.status.len() != n
            || self.linear_predictors.len() != n
            || self.weights.len() != n
            || self.strata.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model diagnostic arrays have inconsistent lengths",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }
        let entry_times = self.entry_times.as_ref();
        if let Some(values) = entry_times
            && values.len() != n
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model entry times have inconsistent length",
            ));
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&lhs, &rhs| {
            self.strata[lhs]
                .cmp(&self.strata[rhs])
                .then_with(|| {
                    self.event_times[lhs]
                        .partial_cmp(&self.event_times[rhs])
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| lhs.cmp(&rhs))
        });

        let method = self.tie_method();
        if matches!(method, CoxMethod::Exact) {
            Ok(self.schoenfeld_residuals_by_scan(nvar, &order, entry_times, method))
        } else {
            Ok(self.schoenfeld_residuals_sweep(nvar, &order, entry_times, method))
        }
    }

    pub(crate) fn schoenfeld_residuals_by_scan(
        &self,
        nvar: usize,
        order: &[usize],
        entry_times: Option<&Vec<f64>>,
        method: CoxMethod,
    ) -> Vec<Vec<f64>> {
        let n = order.len();
        let sorted_time: Vec<f64> = order.iter().map(|&idx| self.event_times[idx]).collect();
        let sorted_start: Vec<f64> = order
            .iter()
            .map(|&idx| entry_times.map(|values| values[idx]).unwrap_or(0.0))
            .collect();
        let sorted_status: Vec<i32> = order.iter().map(|&idx| self.status[idx]).collect();
        let sorted_strata: Vec<i32> = order.iter().map(|&idx| self.strata[idx]).collect();
        let sorted_risk: Vec<f64> = order
            .iter()
            .map(|&idx| {
                self.linear_predictors[idx]
                    .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                    .exp()
                    * self.weights[idx]
            })
            .collect();
        let mut covar = Array2::<f64>::zeros((n, nvar));
        for (row_idx, &source_idx) in order.iter().enumerate() {
            for col_idx in 0..nvar {
                covar[(row_idx, col_idx)] = self.covariates[source_idx][col_idx];
            }
        }

        let mut residuals = Vec::new();
        let mut stratum_start = 0usize;
        while stratum_start < n {
            let stratum = sorted_strata[stratum_start];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < n && sorted_strata[stratum_end + 1] == stratum {
                stratum_end += 1;
            }

            let mut time_start = stratum_start;
            while time_start <= stratum_end {
                let event_time = sorted_time[time_start];
                let mut time_end = time_start;
                while time_end < stratum_end && sorted_time[time_end + 1] == event_time {
                    time_end += 1;
                }

                let death_indices: Vec<usize> = (time_start..=time_end)
                    .filter(|&idx| sorted_status[idx] == 1)
                    .collect();
                if !death_indices.is_empty() {
                    let risk_indices: Vec<usize> = (stratum_start..=stratum_end)
                        .filter(|&idx| {
                            sorted_start[idx] < event_time && sorted_time[idx] >= event_time
                        })
                        .collect();
                    let mut mean = vec![0.0; nvar];
                    if matches!(method, CoxMethod::Exact) && death_indices.len() > 1 {
                        let (denom, a, _cmat) = exact_tied_moments(
                            &risk_indices,
                            death_indices.len(),
                            &sorted_risk,
                            &covar,
                        );
                        if denom > 0.0 {
                            for col_idx in 0..nvar {
                                mean[col_idx] = a[col_idx] / denom / death_indices.len() as f64;
                            }
                        }
                    } else {
                        let mut denom = 0.0;
                        let mut a = vec![0.0; nvar];
                        let mut death_denom = 0.0;
                        let mut death_a = vec![0.0; nvar];
                        for &idx in &risk_indices {
                            let risk = sorted_risk[idx];
                            denom += risk;
                            for col_idx in 0..nvar {
                                let value = covar[(idx, col_idx)];
                                a[col_idx] += risk * value;
                                if sorted_time[idx] == event_time && sorted_status[idx] == 1 {
                                    death_a[col_idx] += risk * value;
                                }
                            }
                            if sorted_time[idx] == event_time && sorted_status[idx] == 1 {
                                death_denom += risk;
                            }
                        }
                        if matches!(method, CoxMethod::Efron) && death_indices.len() > 1 {
                            let deaths = death_indices.len() as f64;
                            for step in 0..death_indices.len() {
                                let fraction = step as f64 / deaths;
                                let step_denom = denom - fraction * death_denom;
                                if step_denom > 0.0 {
                                    for col_idx in 0..nvar {
                                        mean[col_idx] += (a[col_idx] - fraction * death_a[col_idx])
                                            / step_denom
                                            / deaths;
                                    }
                                }
                            }
                        } else if denom > 0.0 {
                            for col_idx in 0..nvar {
                                mean[col_idx] = a[col_idx] / denom;
                            }
                        }
                    }

                    for &idx in &death_indices {
                        residuals.push(
                            (0..nvar)
                                .map(|col_idx| covar[(idx, col_idx)] - mean[col_idx])
                                .collect(),
                        );
                    }
                }

                if time_end == stratum_end {
                    break;
                }
                time_start = time_end + 1;
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn schoenfeld_residuals_sweep(
        &self,
        nvar: usize,
        order: &[usize],
        entry_times: Option<&Vec<f64>>,
        method: CoxMethod,
    ) -> Vec<Vec<f64>> {
        let mut residuals = Vec::new();
        let use_entry_times = entry_times.is_some();
        let mut stratum_start = 0usize;
        while stratum_start < order.len() {
            let stratum = self.strata[order[stratum_start]];
            let mut stratum_end = stratum_start;
            while stratum_end + 1 < order.len() && self.strata[order[stratum_end + 1]] == stratum {
                stratum_end += 1;
            }

            let rows: Vec<CoxSweepRow> = order[stratum_start..=stratum_end]
                .iter()
                .map(|&idx| CoxSweepRow {
                    original_idx: idx,
                    stop: self.event_times[idx],
                    entry: entry_times.map_or(f64::NEG_INFINITY, |values| values[idx]),
                    risk: self.linear_predictors[idx]
                        .clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX)
                        .exp()
                        * self.weights[idx],
                    weight: self.weights[idx],
                    status: self.status[idx],
                })
                .collect();

            let mut active_cov = vec![0.0; nvar];
            if !use_entry_times {
                for row in &rows {
                    for (col_idx, value) in self.covariates[row.original_idx]
                        .iter()
                        .copied()
                        .enumerate()
                        .take(nvar)
                    {
                        active_cov[col_idx] += row.risk * value;
                    }
                }
            }
            let mut active = ActiveRiskSet::new(&rows, use_entry_times);

            let mut time_start = 0usize;
            while time_start < rows.len() {
                let event_time = rows[time_start].stop;
                let mut time_end = time_start;
                while time_end + 1 < rows.len() && rows[time_end + 1].stop == event_time {
                    time_end += 1;
                }

                active.advance_to(event_time, |row_idx, entered| {
                    let sign = if entered { 1.0 } else { -1.0 };
                    for (col_idx, value) in self.covariates[rows[row_idx].original_idx]
                        .iter()
                        .copied()
                        .enumerate()
                        .take(nvar)
                    {
                        active_cov[col_idx] += sign * rows[row_idx].risk * value;
                    }
                });

                let deaths: Vec<usize> = (time_start..=time_end)
                    .filter(|&idx| rows[idx].status == 1)
                    .collect();
                if !deaths.is_empty() && active.risk_sum > 0.0 {
                    let mut mean = vec![0.0; nvar];
                    if matches!(method, CoxMethod::Efron) && deaths.len() > 1 {
                        let mut death_risk = 0.0;
                        let mut death_cov = vec![0.0; nvar];
                        for &row_idx in &deaths {
                            death_risk += rows[row_idx].risk;
                            for (col_idx, value) in self.covariates[rows[row_idx].original_idx]
                                .iter()
                                .copied()
                                .enumerate()
                                .take(nvar)
                            {
                                death_cov[col_idx] += rows[row_idx].risk * value;
                            }
                        }
                        let death_count = deaths.len() as f64;
                        for step in 0..deaths.len() {
                            let fraction = step as f64 / death_count;
                            let step_denom = active.risk_sum - fraction * death_risk;
                            if step_denom > 0.0 {
                                for col_idx in 0..nvar {
                                    mean[col_idx] += (active_cov[col_idx]
                                        - fraction * death_cov[col_idx])
                                        / step_denom
                                        / death_count;
                                }
                            }
                        }
                    } else {
                        for col_idx in 0..nvar {
                            mean[col_idx] = active_cov[col_idx] / active.risk_sum;
                        }
                    }

                    for &row_idx in &deaths {
                        residuals.push(
                            self.covariates[rows[row_idx].original_idx]
                                .iter()
                                .zip(mean.iter())
                                .map(|(&value, &mean)| value - mean)
                                .collect(),
                        );
                    }
                }

                time_start = time_end + 1;
            }

            stratum_start = stratum_end + 1;
        }

        residuals
    }

    pub(crate) fn partial_residuals_internal(&self) -> PyResult<Vec<Vec<f64>>> {
        let beta = self.coefficients.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("model has no fitted coefficients")
        })?;
        let nvar = beta.len();
        let martingale = self.martingale_residuals()?;
        let n = martingale.len();
        if nvar == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        if self.covariates.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match residual length",
            ));
        }
        if self.covariates.iter().any(|row| row.len() != nvar) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fitted Cox model covariates do not match coefficient width",
            ));
        }

        Ok(self
            .covariates
            .iter()
            .zip(martingale.iter())
            .map(|(row, &residual)| {
                row.iter()
                    .zip(beta.iter())
                    .map(|(&value, &coefficient)| residual + value * coefficient)
                    .collect()
            })
            .collect())
    }
}
