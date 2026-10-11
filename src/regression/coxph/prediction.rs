//! Linear predictors, term contributions, expected counts and prediction errors.

use super::{CoxNewData, CoxPHFit, CoxPrediction, CoxTermsPrediction, PredictReference};
use crate::core::strata_order::stratum_groups;
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::step::step_at;
use crate::internal::validation::validate_length;
use crate::surv_analysis::agsurv::{
    AgsurvCurve, IntegratedCurve, cum_xbar_at, cumhaz_at, integrate_curve,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

impl CoxPHFit {
    pub fn nvar(&self) -> usize {
        self.coefficients.len()
    }

    /// Coefficients with aliased (`NaN`) entries replaced by 0, R's
    /// `ifelse(is.na(coef), 0, coef)`.
    pub fn coefficients_or_zero(&self) -> Vec<f64> {
        self.coefficients
            .iter()
            .map(|b| if b.is_nan() { 0.0 } else { *b })
            .collect()
    }

    /// Fitted fields are publicly mutable in Rust. Validate dimensions before
    /// consuming them, including when privately cached baselines already exist.
    /// These checks take constant time in the number of training observations;
    /// fitted numeric values retain their method-specific missing-value behavior.
    pub(super) fn check_prediction_dimensions(&self) -> SurvivalResult<()> {
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

    /// Relative risks without an nrow-by-nvar centered matrix. Reusing one
    /// contiguous row preserves the same subtraction and dot product as the
    /// full curve path while bounding temporary covariate storage by nvar.
    pub(super) fn prediction_risks(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> Vec<f64> {
        let coefficients = self.coefficients_or_zero();
        let coef = ArrayView1::from(&coefficients);
        let offset_mean = self.offset_mean();
        let mut centered = Array1::<f64>::zeros(self.nvar());
        x.outer_iter()
            .enumerate()
            .map(|(i, row)| {
                for ((target, &value), &mean) in centered.iter_mut().zip(row).zip(&self.means) {
                    *target = value - mean;
                }
                (centered.dot(&coef) + offset.map_or(0.0, |values| values[i]) - offset_mean).exp()
            })
            .collect()
    }

    /// Weighted per-stratum column means (`predict.coxph`'s `xmeans`).
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
        for (sum, total) in sums.iter_mut().zip(&totals) {
            for value in sum.iter_mut() {
                *value /= total;
            }
        }
        sums
    }

    /// Whether `predict.coxph` reads back the training design and offset
    /// (its `use.x` branch): for `se.fit`, for a stratified fit centred
    /// within strata, or for `reference = "zero"` with non-zero means.
    fn uses_training_x(&self, se_fit: bool, reference: PredictReference) -> bool {
        se_fit
            || (self.strata.is_some() && reference == PredictReference::Strata)
            || (reference == PredictReference::Zero && self.means.iter().any(|&m| m != 0.0))
    }

    /// The design rows `predict.coxph` uses for `lp`, `risk` and `terms`:
    /// centred per the reference, plus the offset.  With `training_offset`
    /// the offset is centred at the mean training offset, as R's `offset -
    /// mean(offset)` does in its `use.x` branch; otherwise the training
    /// offset is 0 and a new offset is used as it is.
    fn prediction_rows(
        &self,
        newdata: Option<&CoxNewData>,
        reference: PredictReference,
        training_offset: bool,
    ) -> SurvivalResult<(Array2<f64>, Vec<f64>)> {
        let offset_mean = if training_offset {
            self.offset.iter().sum::<f64>() / self.n as f64
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

    /// `predict(type = "lp")` (and `"risk"` via [`Self::predict_risk`]).
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
        let fit: Vec<f64> = newx
            .outer_iter()
            .zip(&offset)
            .map(|(row, o)| row.iter().zip(&coef).map(|(x, b)| x * b).sum::<f64>() + o)
            .collect();
        let se_fit = se_fit.then(|| {
            newx.outer_iter()
                .map(|row| row.dot(&self.var.dot(&row)).sqrt())
                .collect()
        });
        Ok(CoxPrediction { fit, se_fit })
    }

    /// `predict(type = "risk")`: `exp(lp)` with R's Taylor-series standard error.
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

    /// `predict(type = "terms")`: `assign` lists the columns of each term.
    pub fn predict_terms(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
        reference: PredictReference,
        assign: &[Vec<usize>],
    ) -> SurvivalResult<CoxTermsPrediction> {
        self.predict_terms_inner(newdata, se_fit, reference, assign, None)
    }

    /// Term predictions summed by ascending integer group. Only grouped
    /// output rows are allocated; standard errors combine in quadrature.
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

    /// `predict(type = "expected")`: the expected number of events over each
    /// observation's follow-up.  `newdata` needs `time` (and `entry` for a
    /// counting-process fit). Multiple fitted strata require an explicit stratum
    /// for every new row.
    pub fn predict_expected(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        self.check_prediction_dimensions()?;
        let counting = self.entry.is_some();
        let Some(newdata) = newdata else {
            let fit: Vec<f64> = self
                .status
                .iter()
                .zip(&self.residuals)
                .map(|(&s, r)| f64::from(s) - r)
                .collect();
            if !se_fit {
                return Ok(CoxPrediction { fit, se_fit: None });
            }
            // predict.coxph's exp(linear.predictors), relative to the mean
            // offset as the baseline curves are (the standard error does not
            // depend on that scale)
            let offset_mean = self.offset_mean();
            let risk: Vec<f64> = self
                .linear_predictors
                .iter()
                .map(|lp| (lp - offset_mean).exp())
                .collect();
            let se = self.expected_se(
                self.x.view(),
                Some(&self.means),
                &risk,
                &self.sorted.stratum_index,
                self.entry.as_deref(),
                &self.time,
            )?;
            return Ok(CoxPrediction {
                fit,
                se_fit: Some(se),
            });
        };
        self.check_newdata(newdata, true)?;
        if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
            return Err(SurvivalError::invalid_input(
                "newdata must carry the strata for expected predictions",
            ));
        }
        let Some(new_time) = newdata.time.as_deref() else {
            return Err(SurvivalError::invalid_input(
                "newdata must contain the follow-up time for type = 'expected'",
            ));
        };
        if counting && newdata.entry.is_none() {
            return Err(SurvivalError::invalid_input(
                "New data has a different survival type than the model",
            ));
        }
        let risk2 = self.prediction_risks(newdata.x.view(), newdata.offset.as_deref());
        let curves = self.baseline_curves()?;
        let stratum_index: Vec<usize> = match &newdata.strata {
            Some(strata) => strata
                .iter()
                .map(|&code| self.sorted.position_of(code).expect("checked"))
                .collect(),
            None => vec![0; newdata.nrows()],
        };
        let fit: Vec<f64> = (0..newdata.nrows())
            .map(|i| {
                let curve = &curves[stratum_index[i]];
                let stop = cumhaz_at(curve, new_time[i]);
                let start = newdata
                    .entry
                    .as_ref()
                    .map_or(0.0, |entry| cumhaz_at(curve, entry[i]));
                (stop - start) * risk2[i]
            })
            .collect();
        let se = if se_fit {
            Some(self.expected_se(
                newdata.x.view(),
                Some(&self.means),
                &risk2,
                &stratum_index,
                newdata.entry.as_deref(),
                new_time,
            )?)
        } else {
            None
        };
        Ok(CoxPrediction { fit, se_fit: se })
    }

    /// Standard error of an expected count (`predict.coxph`, `type =
    /// "expected"`): `sqrt(varh + dt' V dt) * risk`, differenced over
    /// (entry, time] for counting-process data.  The covariate rows are
    /// `x - means` when `means` is given, `x` itself otherwise.
    #[allow(clippy::too_many_arguments)]
    fn expected_se(
        &self,
        x: ArrayView2<'_, f64>,
        means: Option<&[f64]>,
        risk: &[f64],
        stratum_index: &[usize],
        entry: Option<&[f64]>,
        time: &[f64],
    ) -> SurvivalResult<Vec<f64>> {
        let curves = self.baseline_curves()?;
        let integrated: Vec<IntegratedCurve> = curves.iter().map(integrate_curve).collect();
        let variance_at =
            |curve: &AgsurvCurve, integrated: &IntegratedCurve, t: f64, row: usize| {
                let chaz = cumhaz_at(curve, t);
                let varh = step_at(&curve.time, &integrated.cum_varhaz, t, 0.0);
                let xbar = cum_xbar_at(curve, integrated, t);
                let dt: Vec<f64> = (0..self.nvar())
                    .map(|k| chaz * (x[(row, k)] - means.map_or(0.0, |m| m[k])) - xbar[k])
                    .collect();
                let mut quad = 0.0;
                for (i, &left) in dt.iter().enumerate() {
                    for (j, &right) in dt.iter().enumerate() {
                        quad += left * self.var[(i, j)] * right;
                    }
                }
                varh + quad
            };
        Ok((0..time.len())
            .map(|i| {
                let s = stratum_index[i];
                let v2 = variance_at(&curves[s], &integrated[s], time[i], i);
                let v1 = entry.map_or(0.0, |entry| {
                    variance_at(&curves[s], &integrated[s], entry[i], i)
                });
                (v2 - v1).sqrt() * risk[i]
            })
            .collect())
    }

    /// `predict(type = "survival")`: `exp(-expected)`.
    pub fn predict_survival(
        &self,
        newdata: Option<&CoxNewData>,
        se_fit: bool,
    ) -> SurvivalResult<CoxPrediction> {
        let expected = self.predict_expected(newdata, se_fit)?;
        let fit: Vec<f64> = expected.fit.iter().map(|e| (-e).exp()).collect();
        let se_fit = expected
            .se_fit
            .map(|se| se.iter().zip(&fit).map(|(s, p)| s * p).collect());
        Ok(CoxPrediction { fit, se_fit })
    }

    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }
}

/// Checks that `assign` groups existing columns.  A term may have no
/// columns left (all of them aliased); its prediction is then 0, as in
/// `predict.coxph`.
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

/// One column per coefficient, the default term structure.
pub(crate) fn default_assign(nvar: usize) -> Vec<Vec<usize>> {
    (0..nvar).map(|c| vec![c]).collect()
}
