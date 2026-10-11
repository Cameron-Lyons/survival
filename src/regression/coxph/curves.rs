//! Cached baselines, survival curves and cohort expected survival.

use super::{Basehaz, CoxNewData, CoxPHFit, CoxSurvfitCurve, SurvfitOptions};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::step::find_interval;
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::TieMethod;
use crate::surv_analysis::agsurv::{
    AgsurvCurve, AgsurvData, CoxSurvType, IndividualInterval, agsurv_rows, expand_curve_validated,
    individual_curve_validated,
};
use ndarray::{Array2, ArrayView1, ArrayView2};
use std::borrow::Cow;
use std::collections::HashMap;

impl CoxPHFit {
    /// Survival-curve types matching the tie method (`survfit.coxph`:
    /// `ctype` 2 for Efron, 1 otherwise).
    fn default_survtype(&self) -> CoxSurvType {
        if self.method == TieMethod::Efron {
            CoxSurvType::Efron
        } else {
            CoxSurvType::Breslow
        }
    }

    /// Weighted mean of the offsets (`survfit.coxph`'s `offset.mean`).
    pub(super) fn offset_mean(&self) -> f64 {
        let total: f64 = self.weights.iter().sum();
        self.offset
            .iter()
            .zip(&self.weights)
            .map(|(o, w)| o * w)
            .sum::<f64>()
            / total
    }

    /// Per-stratum `agsurv` pieces at `x - means`, in the fit's stratum
    /// order, with `survfit.coxph`'s `risk = exp(X %*% beta + offset -
    /// xcenter)`: the risks relative to a subject at the means and the mean
    /// offset, so a large offset neither overflows nor rounds the baseline
    /// survival to 1.  `coxsurv.fit` uses `survtype` for the variance too.
    ///
    /// `start_time` keeps only the rows whose stop time is at or after it
    /// (`survfit.coxph`'s `keep <- Y[, ncol(Y) - 1] >= start.time`); a
    /// stratum it empties keeps an empty curve.
    fn compute_curves(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<Vec<AgsurvCurve>> {
        if let Some(t0) = start_time
            && !(0..self.n).any(|i| self.status[i] == 1 && self.time[i] >= t0)
        {
            return Err(SurvivalError::invalid_input(
                "start.time argument has removed all endpoints",
            ));
        }
        let offset_mean = self.offset_mean();
        let risk: Vec<f64> = self
            .linear_predictors
            .iter()
            .map(|lp| (lp - offset_mean).exp())
            .collect();
        let data = AgsurvData {
            start: self.entry.as_deref(),
            stop: &self.time,
            status: &self.status,
            x: self.x.view(),
            means: Some(&self.means),
            weights: &self.weights,
            risk: &risk,
        };
        self.sorted
            .bounds
            .iter()
            .map(|&(start, end)| {
                let rows = &self.sorted.order[start..end];
                match start_time {
                    None => agsurv_rows(&data, rows, survtype, survtype),
                    Some(t0) => {
                        let kept: Vec<usize> = rows
                            .iter()
                            .copied()
                            .filter(|&i| self.time[i] >= t0)
                            .collect();
                        agsurv_rows(&data, &kept, survtype, survtype)
                    }
                }
            })
            .collect()
    }

    /// The cached baseline curves for the fit's own hazard type.
    pub(crate) fn baseline_curves(&self) -> SurvivalResult<&[AgsurvCurve]> {
        self.check_prediction_dimensions()?;
        if let Some(curves) = self.curves.get() {
            return Ok(curves);
        }
        let curves = self.compute_curves(self.default_survtype(), None)?;
        Ok(self.curves.get_or_init(|| curves))
    }

    /// The curves of one hazard type from the rows kept by `start_time`:
    /// the cached ones for the fit's own type and every row.
    fn curves_for(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<Cow<'_, [AgsurvCurve]>> {
        self.check_prediction_dimensions()?;
        if survtype == self.default_survtype() && start_time.is_none() {
            Ok(Cow::Borrowed(self.baseline_curves()?))
        } else {
            Ok(Cow::Owned(self.compute_curves(survtype, start_time)?))
        }
    }

    /// Centred new covariate rows (`newx - means`) and their relative risks
    /// on the baseline curves' scale, `survfit.coxph`'s `risk2 = exp(x2 %*%
    /// beta + offset2 - xcenter)`.
    fn centered_newdata(&self, newdata: &CoxNewData) -> (Array2<f64>, Vec<f64>) {
        self.centered_rows(newdata.x.view(), newdata.offset.as_deref())
    }

    pub(super) fn centered_rows(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> (Array2<f64>, Vec<f64>) {
        let mut x2c = x.to_owned();
        for (col, &mean) in self.means.iter().enumerate() {
            x2c.column_mut(col).mapv_inplace(|value| value - mean);
        }
        let coef = self.coefficients_or_zero();
        let offset_mean = self.offset_mean();
        let risk2: Vec<f64> = x2c
            .outer_iter()
            .enumerate()
            .map(|(i, row)| {
                let offset2 = offset.map_or(0.0, |o| o[i]);
                (row.dot(&ArrayView1::from(&coef)) + offset2 - offset_mean).exp()
            })
            .collect();
        (x2c, risk2)
    }

    /// `basehaz(fit, centered)`.
    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        let curves = self.baseline_curves()?;
        // the curves are survfit(fit)'s, at x = means and the mean offset;
        // uncentred divides the offset sum(means * coef) back out.
        let scale = if centered {
            1.0
        } else {
            let center: f64 = self
                .means
                .iter()
                .zip(self.coefficients_or_zero())
                .map(|(m, b)| m * b)
                .sum();
            (-center).exp()
        };
        let mut time = Vec::new();
        let mut hazard = Vec::new();
        let mut strata = Vec::new();
        for (position, curve) in curves.iter().enumerate() {
            time.extend_from_slice(&curve.time);
            hazard.extend(curve.cumhaz.iter().map(|h| h * scale));
            strata.extend(std::iter::repeat_n(
                self.sorted.codes[position],
                curve.time.len(),
            ));
        }
        Ok(Basehaz {
            time,
            hazard,
            strata: self.strata.as_ref().map(|_| strata),
        })
    }

    /// `survfit(fit, newdata)`: one curve per stratum (all new rows as
    /// columns), or one curve per new row when `newdata` carries strata.
    /// Without `newdata` the curve is for a covariate row at the means.
    pub fn survfit(
        &self,
        newdata: Option<&CoxNewData>,
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        let ctype = options.ctype.unwrap_or(if self.method == TieMethod::Efron {
            2
        } else {
            1
        });
        let survtype = CoxSurvType::from_stype_ctype(options.stype, ctype)?;
        if let Some(newdata) = newdata {
            self.check_newdata(newdata, false)?;
        }
        let curves = self.curves_for(survtype, options.start_time)?;
        let (x2c, risk2) = match newdata {
            Some(newdata) => self.centered_newdata(newdata),
            // the curve at the means and the mean offset
            None => (Array2::zeros((1, self.nvar())), vec![1.0]),
        };
        let varmat = options.se_fit.then_some(&self.var);
        let mut result = Vec::new();
        let new_strata = newdata.and_then(|newdata| newdata.strata.as_deref());
        if let Some(new_strata) = new_strata {
            for (i, &code) in new_strata.iter().enumerate() {
                let position = self
                    .sorted
                    .position_of(code)
                    .expect("strata were checked against the fit");
                let expanded = expand_curve_validated(
                    &curves[position],
                    survtype,
                    x2c.row(i).insert_axis(ndarray::Axis(0)),
                    &risk2[i..=i],
                    varmat,
                )?;
                result.push(finish_curve(code, expanded, options.censor));
            }
        } else {
            for (position, curve) in curves.iter().enumerate() {
                let expanded = expand_curve_validated(curve, survtype, x2c.view(), &risk2, varmat)?;
                result.push(finish_curve(
                    self.sorted.codes[position],
                    expanded,
                    options.censor,
                ));
            }
        }
        Ok(result)
    }

    /// Survival probabilities at `times`, with one column per observation.
    /// Without `newdata`, predict for the training rows in their original order.
    /// Fits with multiple strata require a stratum for each new observation.
    ///
    /// Uses the same default estimate as [`Self::survfit`] (`stype = 2`,
    /// `ctype` matching the fitted tie method). Times may be unsorted or
    /// repeated; survival is 1 before the first time and stays at the last
    /// value beyond follow-up. Only the requested `times.len() * nrows`
    /// probabilities are allocated, rather than the full curves and hazards.
    pub fn predict_survival_at(
        &self,
        times: &[f64],
        newdata: Option<&CoxNewData>,
    ) -> SurvivalResult<Array2<f64>> {
        self.check_prediction_dimensions()?;
        validate_finite(times, "times")?;
        let (x, strata, offset) = match newdata {
            Some(newdata) => {
                self.check_newdata(newdata, false)?;
                if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
                    return Err(SurvivalError::invalid_input(
                        "newdata must carry the strata for survival predictions",
                    ));
                }
                (
                    newdata.x.view(),
                    newdata.strata.as_deref(),
                    newdata.offset.as_deref(),
                )
            }
            None => (
                self.x.view(),
                self.strata.as_deref(),
                Some(self.offset.as_slice()),
            ),
        };
        let mut result = Array2::ones((times.len(), x.nrows()));
        if times.is_empty() {
            return Ok(result);
        }
        let risk = self.prediction_risks(x, offset);
        let curves = self.baseline_curves()?;
        let mut rows_by_stratum = vec![Vec::new(); curves.len()];
        for row in 0..x.nrows() {
            let position = match strata {
                Some(codes) => self.sorted.position_of(codes[row]).ok_or_else(|| {
                    SurvivalError::invalid_input(format!(
                        "fit stratum {} is not a fitted stratum",
                        codes[row]
                    ))
                })?,
                None => 0,
            };
            rows_by_stratum[position].push(row);
        }
        for (curve, rows) in curves.iter().zip(rows_by_stratum) {
            if rows.is_empty() {
                continue;
            }
            for (i, &at) in times.iter().enumerate() {
                let index = find_interval(&curve.time, at, false);
                if index == 0 {
                    continue;
                }
                // Match expand_curve's exp(-H).powf(risk), including its
                // underflow behavior, rather than reassociating the exponent.
                let baseline = (-curve.cumhaz[index - 1]).exp();
                let mut output = result.row_mut(i);
                for &row in &rows {
                    output[row] = baseline.powf(risk[row]);
                }
            }
        }
        Ok(result)
    }

    /// Cohort expected survival, aggregating baseline hazards directly instead
    /// of materializing every subject's survival and cumulative-hazard curve.
    pub fn expected_survival(
        &self,
        newdata: &CoxNewData,
        group: &[usize],
        weights: &[f64],
        y: Option<&[f64]>,
        times: Option<&[f64]>,
        method: &str,
    ) -> SurvivalResult<crate::population::SurvExpResult> {
        use crate::population::{CoxExpectedBaseline, survexp_cox_prepared};
        self.check_prediction_dimensions()?;
        self.check_newdata(newdata, false)?;
        if newdata.strata.is_none() && self.sorted.nstrata() > 1 {
            return Err(SurvivalError::invalid_input(
                "newdata must carry the strata for expected survival",
            ));
        }
        let risk = self.prediction_risks(newdata.x.view(), newdata.offset.as_deref());
        let strata: Vec<usize> = (0..newdata.nrows())
            .map(|i| {
                newdata.strata.as_ref().map_or(0, |codes| {
                    self.sorted
                        .position_of(codes[i])
                        .expect("strata were checked")
                })
            })
            .collect();
        let baselines = self
            .baseline_curves()?
            .iter()
            .map(|curve| {
                let rows = || {
                    curve
                        .n_event
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &n)| (n > 0.0).then_some(i))
                };
                CoxExpectedBaseline {
                    time: rows().map(|i| curve.time[i]).collect(),
                    cumhaz: rows().map(|i| curve.cumhaz[i]).collect(),
                }
            })
            .collect::<Vec<_>>();
        survexp_cox_prepared(&baselines, &risk, &strata, group, weights, y, times, method)
    }

    /// `survfit(fit, newdata, id)`: one curve per subject whose covariates
    /// change over the (entry, time] intervals of `newdata`. As in R, omitting
    /// strata selects the first fitted stratum for an individual path.
    pub fn survfit_individual(
        &self,
        newdata: &CoxNewData,
        id: &[i32],
        options: SurvfitOptions,
    ) -> SurvivalResult<Vec<CoxSurvfitCurve>> {
        self.check_newdata(newdata, false)?;
        let (Some(entry), Some(time)) = (&newdata.entry, &newdata.time) else {
            return Err(SurvivalError::invalid_input(
                "Individual=TRUE is only valid for counting process data",
            ));
        };
        if id.len() != newdata.nrows() {
            return Err(SurvivalError::invalid_input(
                "id must have one value per newdata row",
            ));
        }
        let ctype = options.ctype.unwrap_or(if self.method == TieMethod::Efron {
            2
        } else {
            1
        });
        let survtype = CoxSurvType::from_stype_ctype(options.stype, ctype)?;
        let curves = self.curves_for(survtype, options.start_time)?;
        let (x2c, risk2) = self.centered_newdata(newdata);
        let varmat = options.se_fit.then_some(&self.var);
        let mut positions = HashMap::new();
        let mut rows: Vec<Vec<usize>> = Vec::new();
        for (i, &subject) in id.iter().enumerate() {
            let position = *positions.entry(subject).or_insert_with(|| {
                rows.push(Vec::new());
                rows.len() - 1
            });
            rows[position].push(i);
        }
        let mut result = Vec::with_capacity(rows.len());
        for subject_rows in rows {
            let intervals: Vec<IndividualInterval<'_>> = subject_rows
                .into_iter()
                .map(|i| IndividualInterval {
                    start: entry[i],
                    stop: time[i],
                    stratum: newdata.strata.as_ref().map_or(0, |s| {
                        self.sorted
                            .position_of(s[i])
                            .expect("strata were checked against the fit")
                    }),
                    x2: x2c.row(i).to_slice().expect("row is contiguous"),
                    risk2: risk2[i],
                })
                .collect();
            let curve = individual_curve_validated(&curves, survtype, &intervals, varmat)?;
            let stratum = intervals
                .first()
                .map_or(0, |interval| self.sorted.codes[interval.stratum]);
            result.push(finish_curve(stratum, curve, options.censor));
        }
        Ok(result)
    }
}

fn finish_curve(
    stratum: i32,
    curve: crate::surv_analysis::agsurv::CoxSurvCurve,
    censor: bool,
) -> CoxSurvfitCurve {
    let keep: Vec<usize> = (0..curve.time.len())
        .filter(|&g| censor || curve.n_event[g] > 0.0)
        .collect();
    let pick = |values: &[f64]| keep.iter().map(|&g| values[g]).collect::<Vec<_>>();
    let pick_rows = |matrix: &Array2<f64>| {
        keep.iter()
            .map(|&g| matrix.row(g).to_vec())
            .collect::<Vec<_>>()
    };
    CoxSurvfitCurve {
        stratum,
        n: curve.n,
        time: pick(&curve.time),
        n_risk: pick(&curve.n_risk),
        n_event: pick(&curve.n_event),
        n_censor: pick(&curve.n_censor),
        surv: pick_rows(&curve.surv),
        cumhaz: pick_rows(&curve.cumhaz),
        std_err: curve.std_err.as_ref().map(pick_rows),
    }
}
