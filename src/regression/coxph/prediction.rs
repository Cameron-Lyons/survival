use super::scaling::{
    PredictionRisks, PredictorOrigin, add_predictor_parts, relative_covariate_scale,
    stable_weighted_mean,
};
use super::{CoxNewData, CoxPHFit, CoxPrediction, CoxTermsPrediction, PredictReference};
use crate::core::strata_order::stratum_groups;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::step::find_interval;
use crate::internal::validation::validate_length;
use crate::regression::cox_optimizer::predictor_bound_needs_anchor;
use crate::regression::log_scale::LogScale;
use crate::surv_analysis::agsurv::{AgsurvCurve, IntegratedCurve};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

impl CoxPHFit {
    pub fn nvar(&self) -> usize {
        self.coefficients.len()
    }

    pub fn coefficients_or_zero(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|b| if b.is_nan() { 0.0 } else { *b })
            .collect()
    }

    pub(crate) fn check_prediction_dimensions(&self) -> SurvivalResult<()> {
        for (name, length) in [
            ("fit time", self.time.len()),
            ("fit status", self.status.len()),
            ("fit design rows", self.x.nrows()),
            ("fit weights", self.weights.len()),
            ("fit offset", self.offset.len()),
            ("fit linear predictors", self.linear_predictors.len()),
            ("fit residuals", self.residuals.len()),
            ("fit sorted rows", self.sorted.order.len()),
            ("fit stratum rows", self.sorted.stratum_index.len()),
        ] {
            validate_length(self.n, length, name)?;
        }
        if let Some(entry) = &self.entry {
            validate_length(self.n, entry.len(), "fit entry")?;
        }
        if let Some(strata) = &self.strata {
            validate_length(self.n, strata.len(), "fit strata")?;
        } else if self.sorted.nstrata() > 1 {
            return Err(SurvivalError::invalid_input(
                "a stratified fit requires its stored strata",
            ));
        }
        let p = self.nvar();
        validate_length(p, self.x.ncols(), "fit design columns")?;
        validate_length(p, self.means.len(), "fit means")?;
        if self.var.dim() != (p, p) {
            return Err(SurvivalError::invalid_input(format!(
                "fit variance must have shape ({p}, {p})"
            )));
        }
        if let Some(curve) = self.curves.get().and_then(|curves| curves.first()) {
            validate_length(p, curve.xbar.ncols(), "fit baseline columns")?;
        }
        Ok(())
    }

    pub(super) fn check_newdata(
        &self,
        newdata: &CoxNewData,
        allow_missing: bool,
    ) -> SurvivalResult<()> {
        newdata.validate(allow_missing)?;
        if newdata.x.ncols() != self.nvar() {
            return Err(SurvivalError::invalid_input(format!(
                "newdata has {} columns but the model has {}",
                newdata.x.ncols(),
                self.nvar()
            )));
        }
        if let Some(strata) = &newdata.strata
            && let Some(code) = strata
                .iter()
                .find(|code| self.sorted.position_of(**code).is_none())
        {
            return Err(SurvivalError::invalid_input(format!(
                "New data has a strata not found in the original model: {code}"
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn prediction_risks(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> Vec<f64> {
        self.prediction_log_risks_at(
            x,
            offset,
            &[PredictorOrigin::reference(self.offset_mean())],
            None,
        )
        .risks()
    }

    pub(super) fn prediction_log_risks_at(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
        origins: &[PredictorOrigin],
        strata: Option<&[i32]>,
    ) -> PredictionRisks {
        let coefficients = self.coefficients_or_zero();
        let coef = ArrayView1::from(&coefficients);
        let max_beta = coefficients.iter().map(|b| b.abs()).fold(0.0, f64::max);
        let mut centered = Array1::<f64>::zeros(self.nvar());
        let mut scalar = Vec::with_capacity(x.nrows());
        let mut scaled: Option<Vec<LogScale>> = None;
        for (i, row) in x.outer_iter().enumerate() {
            let origin_index = strata.map_or(0, |codes| {
                self.sorted.position_of(codes[i]).expect("strata checked")
            });
            let origin = &origins[origin_index];
            let anchor = origin.covariates.as_deref().unwrap_or(&self.means);
            let mut unsafe_covariate = false;
            for ((target, &value), &mean) in centered.iter_mut().zip(row).zip(anchor) {
                *target = value - mean;
                unsafe_covariate |= predictor_bound_needs_anchor(target.abs() * max_beta);
            }
            let value = offset.map_or(0.0, |values| values[i]);
            if origin.covariates.is_some()
                || value.abs() > 200.0
                || origin.offset.abs() > 200.0
                || unsafe_covariate
            {
                let cov = relative_covariate_scale(row, anchor, &coefficients);
                let value = add_predictor_parts(
                    add_predictor_parts(
                        LogScale::from_log(value),
                        LogScale::from_log(-origin.offset),
                    ),
                    cov,
                );
                scaled
                    .get_or_insert_with(|| {
                        let mut values = Vec::with_capacity(x.nrows());
                        values.extend(scalar.drain(..).map(LogScale::from_log));
                        values
                    })
                    .push(value);
            } else {
                let value = centered.dot(&coef) + value - origin.offset;
                if let Some(scaled) = &mut scaled {
                    scaled.push(LogScale::from_log(value));
                } else {
                    scalar.push(value);
                }
            }
        }
        scaled.map_or_else(|| PredictionRisks::Scalar(scalar), PredictionRisks::Scaled)
    }

    fn stratum_means(&self) -> Vec<Vec<f64>> {
        let nvar = self.nvar();
        let mut sums = vec![vec![0.0; nvar]; self.sorted.nstrata()];
        let mut totals = vec![0.0; self.sorted.nstrata()];
        for row in 0..self.n {
            let s = self.sorted.stratum_index[row];
            totals[s] += self.weights[row];
            for (col, sum) in sums[s].iter_mut().enumerate().take(nvar) {
                *sum += self.weights[row] * self.x[(row, col)];
            }
        }
        for (stratum, (sum, &total)) in sums.iter_mut().zip(&totals).enumerate() {
            if total.is_finite() && sum.iter().all(|value| value.is_finite()) {
                for value in sum.iter_mut() {
                    *value /= total;
                }
                continue;
            }
            // A finite centre can have an overflowing weighted sum. Keep the
            // usual row-order arithmetic above, rescaling only affected columns.
            let (start, end) = self.sorted.bounds[stratum];
            let rows = &self.sorted.order[start..end];
            let weights: Vec<f64> = rows.iter().map(|&row| self.weights[row]).collect();
            for (column, value) in sum.iter_mut().enumerate() {
                if total.is_finite() && value.is_finite() {
                    *value /= total;
                } else {
                    let values: Vec<f64> = rows.iter().map(|&row| self.x[(row, column)]).collect();
                    *value = stable_weighted_mean(&values, &weights);
                }
            }
        }
        sums
    }

    fn uses_training_x(&self, se_fit: bool, reference: PredictReference) -> bool {
        se_fit
            || (self.strata.is_some() && reference == PredictReference::Strata)
            || (reference == PredictReference::Zero && self.means.iter().any(|&m| m != 0.0))
    }

    fn prediction_rows(
        &self,
        newdata: Option<&CoxNewData>,
        reference: PredictReference,
        training_offset: bool,
    ) -> SurvivalResult<(Array2<f64>, Vec<f64>)> {
        let offset_mean = if training_offset {
            super::scaling::stable_mean(&self.offset)
        } else {
            0.0
        };
        let has_strata = self.strata.is_some();
        let (mut newx, offset, stratum_index): (Array2<f64>, Vec<f64>, Vec<usize>) = match newdata {
            None => (
                self.x.clone(),
                self.offset.iter().map(|o| o - offset_mean).collect(),
                self.sorted.stratum_index.clone(),
            ),
            Some(newdata) => {
                self.check_newdata(newdata, true)?;
                let m = newdata.nrows();
                let offset = newdata.offset.as_ref().map_or_else(
                    || vec![-offset_mean; m],
                    |o| o.iter().map(|v| v - offset_mean).collect(),
                );
                let stratum_index = match &newdata.strata {
                    Some(strata) => strata
                        .iter()
                        .map(|&code| self.sorted.position_of(code).expect("checked"))
                        .collect(),
                    None => {
                        if has_strata && reference == PredictReference::Strata {
                            return Err(SurvivalError::invalid_input(
                                "newdata must carry the strata for reference = 'strata'",
                            ));
                        }
                        vec![0; m]
                    }
                };
                (newdata.x.clone(), offset, stratum_index)
            }
        };
        if has_strata && reference == PredictReference::Strata {
            let xmeans = self.stratum_means();
            for (i, &s) in stratum_index.iter().enumerate() {
                for col in 0..self.nvar() {
                    newx[(i, col)] -= xmeans[s][col];
                }
            }
        } else if reference != PredictReference::Zero {
            for (col, &mean) in self.means.iter().enumerate() {
                newx.column_mut(col).mapv_inplace(|value| value - mean);
            }
        }
        Ok((newx, offset))
    }

    pub fn predict_lp(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
    ) -> SurvivalResult<CoxPrediction> {
        self.check_prediction_dimensions()?;
        let training_x = self.uses_training_x(se_fit, reference);
        if newdata.is_none() && !training_x {
            return Ok(CoxPrediction {
                fit: self.linear_predictors.clone(),
                se_fit: None,
            });
        }
        let (newx, offset) = self.prediction_rows(newdata, reference, training_x)?;
        let coef = self.coefficients_or_zero();
        let mut zero_anchor = None;
        let fit: Vec<f64> = newx
            .outer_iter()
            .zip(&offset)
            .map(|(row, o)| {
                let mut unsafe_scale = o.abs() > 200.0;
                let ordinary = row
                    .iter()
                    .zip(&coef)
                    .map(|(x, b)| {
                        let term = x * b;
                        unsafe_scale |= predictor_bound_needs_anchor(term.abs());
                        term
                    })
                    .sum::<f64>()
                    + o;
                if unsafe_scale {
                    let anchor = zero_anchor.get_or_insert_with(|| vec![0.0; self.nvar()]);
                    add_predictor_parts(
                        relative_covariate_scale(row, anchor, &coef),
                        LogScale::from_log(*o),
                    )
                    .log_value()
                } else {
                    ordinary
                }
            })
            .collect();
        let se_fit = se_fit.then(|| {
            newx.outer_iter()
                .map(|row| row.dot(&self.var.dot(&row)).sqrt())
                .collect()
        });
        Ok(CoxPrediction { fit, se_fit })
    }

    pub fn predict_risk(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
    ) -> SurvivalResult<CoxPrediction> {
        let lp = self.predict_lp(newdata, se_fit, reference)?;
        let fit: Vec<f64> = lp.fit.iter().map(|v| v.exp()).collect();
        let se_fit = lp
            .se_fit
            .map(|se| se.iter().zip(&fit).map(|(s, p)| s * p.sqrt()).collect());
        Ok(CoxPrediction { fit, se_fit })
    }

    pub fn predict_terms(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
        assign: &[Vec<usize>],
    ) -> SurvivalResult<CoxTermsPrediction> {
        self.predict_terms_inner(newdata, se_fit, reference, assign, None)
    }

    pub fn predict_terms_grouped(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
        assign: &[Vec<usize>],
        group: &[i32],
    ) -> SurvivalResult<CoxTermsPrediction> {
        self.predict_terms_inner(newdata, se_fit, reference, assign, Some(group))
    }

    fn predict_terms_inner(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
        assign: &[Vec<usize>],
        group: Option<&[i32]>,
    ) -> SurvivalResult<CoxTermsPrediction> {
        self.check_prediction_dimensions()?;
        validate_assign(assign, self.nvar())?;
        let nrows = newdata.map_or(self.n, CoxNewData::nrows);
        if group.is_some_and(|group| group.len() != nrows) {
            return Err(SurvivalError::invalid_input(
                "group must have one value per prediction row",
            ));
        }
        let groups = group.map(stratum_groups);
        let (newx, _) = self.prediction_rows(newdata, reference, true)?;
        let coef = self.coefficients_or_zero();
        let nterms = assign.len();
        let output_rows = groups.as_ref().map_or(newx.nrows(), Vec::len);
        let mut fit = vec![vec![0.0; nterms]; output_rows];
        let mut se = se_fit.then(|| vec![vec![0.0; nterms]; output_rows]);
        for (t, columns) in assign.iter().enumerate() {
            let evaluate = |row: ArrayView1<'_, f64>| {
                let value = columns.iter().map(|&c| row[c] * coef[c]).sum();
                let error = if se_fit {
                    let mut total = 0.0;
                    for &c1 in columns {
                        for &c2 in columns {
                            total += row[c1] * self.var[(c1, c2)] * row[c2];
                        }
                    }
                    total.sqrt()
                } else {
                    0.0
                };
                (value, error)
            };
            if let Some(groups) = &groups {
                for (i, (_, rows)) in groups.iter().enumerate() {
                    for &row in rows {
                        let (value, error) = evaluate(newx.row(row));
                        fit[i][t] += value;
                        if let Some(se) = se.as_mut() {
                            se[i][t] += error * error;
                        }
                    }
                    if let Some(se) = se.as_mut() {
                        se[i][t] = se[i][t].sqrt();
                    }
                }
            } else {
                for (i, row) in newx.outer_iter().enumerate() {
                    let (value, error) = evaluate(row);
                    fit[i][t] = value;
                    if let Some(se) = se.as_mut() {
                        se[i][t] = error;
                    }
                }
            }
        }
        Ok(CoxTermsPrediction {
            n_columns: nterms,
            fit,
            se_fit: se,
            constant: coef.iter().zip(&self.means).map(|(b, m)| b * m).sum(),
        })
    }

    pub fn predict_expected(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        self.check_prediction_dimensions()?;
        let (x, time, entry, mut fit) = match newdata {
            Some(newdata) => {
                self.check_newdata(newdata, true)?;
                if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
                    return Err(SurvivalError::invalid_input(
                        "newdata must carry the strata for expected predictions",
                    ));
                }
                let time = newdata.time.as_deref().ok_or_else(|| {
                    SurvivalError::invalid_input(
                        "newdata must contain the follow-up time for type = 'expected'",
                    )
                })?;
                if self.entry.is_some() && newdata.entry.is_none() {
                    return Err(SurvivalError::invalid_input(
                        "New data has a different survival type than the model",
                    ));
                }
                (
                    newdata.x.view(),
                    time,
                    newdata.entry.as_deref(),
                    Vec::with_capacity(time.len()),
                )
            }
            None => {
                let fit = self
                    .status
                    .iter()
                    .zip(&self.residuals)
                    .map(|(&status, &residual)| f64::from(status) - residual)
                    .collect();
                if !se_fit {
                    return Ok(CoxPrediction { fit, se_fit: None });
                }
                (
                    self.x.view(),
                    self.time.as_slice(),
                    self.entry.as_deref(),
                    fit,
                )
            }
        };
        let curves = self.baseline_curves()?;
        let log_risk = match newdata {
            Some(newdata) => self.prediction_log_risks_at(
                x,
                newdata.offset.as_deref(),
                &curves.origins,
                newdata.strata.as_deref(),
            ),
            None => self.training_prediction_risks(&curves.origins),
        };
        let logs = log_risk.needs_logs(curves.stable).then(|| curves.logs());
        let mut se = se_fit.then(|| Vec::with_capacity(time.len()));
        let mut scratch = if se_fit {
            vec![0.0; self.nvar()]
        } else {
            Vec::new()
        };
        for (i, row) in x.outer_iter().enumerate() {
            let stratum = match newdata {
                None => self.sorted.stratum_index[i],
                Some(newdata) => newdata.strata.as_ref().map_or(0, |strata| {
                    self.sorted.position_of(strata[i]).expect("strata checked")
                }),
            };
            let curve = &curves[stratum];
            let first = entry.map_or(0, |entry| find_interval(&curve.time, entry[i], false));
            let end = find_interval(&curve.time, time[i], false);
            if let Some(logs) = logs {
                let lp = log_risk.scale(i);
                if newdata.is_some() {
                    fit.push(logs[stratum].range_hazard(first, end, lp));
                }
                if let Some(se) = &mut se {
                    for ((value, &x), &mean) in scratch.iter_mut().zip(row).zip(&self.means) {
                        *value = x - mean;
                    }
                    se.push(logs[stratum].expected_se(first, end, lp, &scratch, &self.var));
                }
            } else {
                let risk = log_risk.scale(i).exp_at(0.0);
                if newdata.is_some() {
                    let hazard_at =
                        |index: usize| index.checked_sub(1).map_or(0.0, |j| curve.cumhaz[j]);
                    fit.push((hazard_at(end) - hazard_at(first)) * risk);
                }
                if let Some(se) = &mut se {
                    let integrated = curves.integrated(stratum);
                    let mut variance_at = |index| {
                        self.expected_variance_at(curve, integrated, index, row, &mut scratch)
                    };
                    se.push((variance_at(end) - variance_at(first)).sqrt() * risk);
                }
            }
        }
        Ok(CoxPrediction { fit, se_fit: se })
    }

    fn expected_variance_at(
        &self,
        curve: &AgsurvCurve,
        integrated: &IntegratedCurve,
        index: usize,
        row: ArrayView1<'_, f64>,
        derivative: &mut [f64],
    ) -> f64 {
        let Some(g) = index.checked_sub(1) else {
            return 0.0;
        };
        let chaz = curve.cumhaz[g];
        for (k, value) in derivative.iter_mut().enumerate() {
            *value = chaz * (row[k] - self.means[k]) - integrated.cum_xbar[(g, k)];
        }
        let mut quad = 0.0;
        for (i, &left) in derivative.iter().enumerate() {
            for (j, &right) in derivative.iter().enumerate() {
                quad += left * self.var[(i, j)] * right;
            }
        }
        integrated.cum_varhaz[g] + quad
    }

    pub fn predict_survival(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        let expected = self.predict_expected(newdata, se_fit)?;
        let fit: Vec<f64> = expected.fit.iter().map(|e| (-e).exp()).collect();
        let se_fit = expected.se_fit.map(|se| {
            se.iter()
                .zip(&fit)
                .map(|(s, p)| if *p == 0.0 { 0.0 } else { s * p })
                .collect()
        });
        Ok(CoxPrediction { fit, se_fit })
    }

    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }
}

pub(crate) fn validate_assign(assign: &[Vec<usize>], nvar: usize) -> SurvivalResult<()> {
    for (term, columns) in assign.iter().enumerate() {
        if let Some(column) = columns.iter().find(|&&c| c >= nvar) {
            return Err(SurvivalError::invalid_input(format!(
                "assign[{term}] refers to column {column}, but the model has {nvar}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn default_assign(nvar: usize) -> Vec<Vec<usize>> {
    (0..nvar).map(|c| vec![c]).collect()
}
