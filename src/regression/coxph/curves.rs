use super::log_curves::CachedCurves;
use super::scaling::{PredictionRisks, PredictorOrigin};
use super::{Basehaz, CoxNewData, CoxPHFit, CoxSurvfitCurve, SurvfitOptions};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::step::find_interval;
use crate::internal::validation::validate_finite;
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::log_risk_sweep::needs_log_risk;
use crate::regression::log_scale::LogScale;
use crate::surv_analysis::agsurv::{
    AgsurvData, CoxSurvType, IndividualInterval, agsurv_rows_checked, agsurv_rows_log_kp,
    expand_curve_log_kp, expand_curve_validated, individual_curve_validated,
};
use ndarray::{Array2, ArrayView2};
use std::borrow::Cow;
use std::collections::HashMap;

impl CoxPHFit {
    fn default_survtype(&self) -> CoxSurvType {
        if self.method == TieMethod::Efron {
            CoxSurvType::Efron
        } else {
            CoxSurvType::Breslow
        }
    }

    pub(super) fn offset_mean(&self) -> f64 {
        super::scaling::stable_weighted_mean(&self.offset, &self.weights)
    }

    fn compute_curves(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<CachedCurves> {
        if let Some(t0) = start_time
            && !(0..self.n).any(|i| self.status[i] == 1 && self.time[i] >= t0)
        {
            return Err(SurvivalError::invalid_input(
                "start.time argument has removed all endpoints",
            ));
        }
        let offset_mean = self.offset_mean();
        let origins = self.curve_origins(offset_mean);
        // Without events every hazard and variance increment is zero. The
        // stored no-event LP has historical centering semantics; evaluating its
        // otherwise irrelevant risk can overflow when centering it a second time.
        let log_risk = if self.nevent == 0 {
            vec![0.0; self.n]
        } else {
            self.training_log_risks_at(&origins)
        };
        if self.has_curve_origins() || needs_log_risk(&log_risk, &self.weights) {
            let scales = if self.nevent == 0 {
                vec![LogScale::from_log(0.0); self.n]
            } else {
                self.training_risk_scales(&origins)
            };
            return self.compute_log_curves(survtype, start_time, &scales, &origins);
        }
        let kp_risks = (survtype == CoxSurvType::KalbfleischPrentice).then(|| log_risk.clone());
        let mut risk = log_risk;
        risk.iter_mut().for_each(|lp| *lp = lp.exp());
        let data = AgsurvData {
            start: self.entry.as_deref(),
            stop: &self.time,
            status: &self.status,
            x: self.x.view(),
            means: Some(&self.means),
            weights: &self.weights,
            risk: &risk,
        };
        let mut curves = Vec::with_capacity(self.sorted.bounds.len());
        let mut kp_logs = if kp_risks.is_some() {
            Vec::with_capacity(self.sorted.bounds.len())
        } else {
            Vec::new()
        };
        for &(start, end) in &self.sorted.bounds {
            let rows = &self.sorted.order[start..end];
            let kept: Vec<usize>;
            let rows = if let Some(t0) = start_time {
                kept = rows
                    .iter()
                    .copied()
                    .filter(|&i| self.time[i] >= t0)
                    .collect();
                &kept
            } else {
                rows
            };
            if let Some(log_risk) = &kp_risks {
                let Some((curve, increments)) = agsurv_rows_log_kp(&data, rows, log_risk)? else {
                    return self.compute_log_curves(
                        survtype,
                        start_time,
                        &self.training_risk_scales(&origins),
                        &origins,
                    );
                };
                curves.push(curve);
                kp_logs.push(increments);
            } else {
                let Some(curve) = agsurv_rows_checked(&data, rows, survtype, survtype)? else {
                    return self.compute_log_curves(
                        survtype,
                        start_time,
                        &self.training_risk_scales(&origins),
                        &origins,
                    );
                };
                curves.push(curve);
            }
        }
        Ok(if kp_risks.is_some() {
            CachedCurves::ordinary_kp(curves, origins, kp_logs)
        } else {
            CachedCurves::ordinary(curves, origins)
        })
    }

    pub(super) fn baseline_curves(&self) -> SurvivalResult<&CachedCurves> {
        self.check_prediction_dimensions()?;
        if let Some(curves) = self.curves.get() {
            return Ok(curves);
        }
        let curves = self.compute_curves(self.default_survtype(), None)?;
        Ok(self.curves.get_or_init(|| curves))
    }

    fn curves_for(
        &self,
        survtype: CoxSurvType,
        start_time: Option<f64>,
    ) -> SurvivalResult<Cow<'_, CachedCurves>> {
        self.check_prediction_dimensions()?;
        if survtype == self.default_survtype() && start_time.is_none() {
            Ok(Cow::Borrowed(self.baseline_curves()?))
        } else {
            Ok(Cow::Owned(self.compute_curves(survtype, start_time)?))
        }
    }

    fn centered_newdata(&self, newdata: &CoxNewData) -> (Array2<f64>, PredictionRisks) {
        self.centered_log_rows(newdata.x.view(), newdata.offset.as_deref())
    }

    #[cfg(test)]
    pub(super) fn centered_rows(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> (Array2<f64>, Vec<f64>) {
        let (x, risk) = self.centered_log_rows(x, offset);
        (x, risk.risks())
    }

    fn centered_log_rows(
        &self,
        x: ArrayView2<'_, f64>,
        offset: Option<&[f64]>,
    ) -> (Array2<f64>, PredictionRisks) {
        let mut centered = x.to_owned();
        for (col, &mean) in self.means.iter().enumerate() {
            centered.column_mut(col).mapv_inplace(|value| value - mean);
        }
        let risks = self.prediction_log_risks_at(
            x,
            offset,
            &[PredictorOrigin::reference(self.offset_mean())],
            None,
        );
        (centered, risks)
    }

    pub fn basehaz(&self, centered: bool) -> SurvivalResult<Basehaz> {
        let curves = self.baseline_curves()?;
        let offset_mean = curves.reference_offset;
        let log_scale = if centered {
            0.0
        } else {
            -self
                .means
                .iter()
                .zip(self.coefficients_or_zero())
                .map(|(m, b)| m * b)
                .sum::<f64>()
        };
        let scale = log_scale.exp();
        let stable = curves.stable || !(-200.0..=200.0).contains(&log_scale);
        let mut time = Vec::new();
        let mut hazard = Vec::new();
        let mut strata = Vec::new();
        for (position, curve) in curves.iter().enumerate() {
            time.extend_from_slice(&curve.time);
            if stable {
                let target = if centered {
                    Array2::from_shape_vec((1, self.nvar()), self.means.clone())
                        .expect("one baseline row")
                } else {
                    Array2::zeros((1, self.nvar()))
                };
                let at = self
                    .prediction_log_risks_at(
                        target.view(),
                        Some(&[offset_mean]),
                        &curves.origins[position..=position],
                        None,
                    )
                    .scale(0);
                hazard.extend(
                    curves.logs()[position]
                        .log_cumulative
                        .iter()
                        .map(|h| h.exp_at_scale(at)),
                );
            } else {
                hazard.extend(curve.cumhaz.iter().map(|h| h * scale));
            }
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
        let (x2c, log_risk) = match newdata {
            Some(newdata) => self.centered_newdata(newdata),
            None => (
                Array2::zeros((1, self.nvar())),
                PredictionRisks::Scalar(vec![0.0]),
            ),
        };
        let risk2 = log_risk.risks();
        let varmat = options.se_fit.then_some(&self.var);
        let offset_mean = curves.reference_offset;
        let stable = log_risk.needs_logs(curves.stable);
        let mut result = Vec::new();
        let new_strata = newdata.and_then(|newdata| newdata.strata.as_deref());
        if let Some(new_strata) = new_strata {
            for (i, &code) in new_strata.iter().enumerate() {
                let position = self
                    .sorted
                    .position_of(code)
                    .expect("strata were checked against the fit");
                let expanded = if stable {
                    let local_risk = self.prediction_log_risks_at(
                        newdata
                            .expect("new strata require newdata")
                            .x
                            .slice(ndarray::s![i..=i, ..]),
                        newdata.and_then(|d| d.offset.as_ref()).map(|o| &o[i..=i]),
                        &curves.origins[position..=position],
                        None,
                    );
                    curves.logs()[position].expand(
                        &curves[position],
                        x2c.row(i).insert_axis(ndarray::Axis(0)),
                        &local_risk.as_scales(),
                        varmat,
                    )
                } else if let Some(kp) = &curves.kp_logs {
                    expand_curve_log_kp(
                        &curves[position],
                        x2c.row(i).insert_axis(ndarray::Axis(0)),
                        &risk2[i..=i],
                        &[log_risk.scale(i).log_value()],
                        &kp[position],
                        varmat,
                    )?
                } else {
                    expand_curve_validated(
                        &curves[position],
                        survtype,
                        x2c.row(i).insert_axis(ndarray::Axis(0)),
                        &risk2[i..=i],
                        varmat,
                    )?
                };
                result.push(finish_curve(code, expanded, options.censor));
            }
        } else {
            for (position, curve) in curves.iter().enumerate() {
                let expanded = if stable {
                    let local_risk = newdata.map_or_else(
                        || {
                            let target =
                                Array2::from_shape_vec((1, self.nvar()), self.means.clone())
                                    .expect("one baseline row");
                            self.prediction_log_risks_at(
                                target.view(),
                                Some(&[offset_mean]),
                                &curves.origins[position..=position],
                                None,
                            )
                        },
                        |data| {
                            self.prediction_log_risks_at(
                                data.x.view(),
                                data.offset.as_deref(),
                                &curves.origins[position..=position],
                                None,
                            )
                        },
                    );
                    curves.logs()[position].expand(
                        curve,
                        x2c.view(),
                        &local_risk.as_scales(),
                        varmat,
                    )
                } else if let Some(kp) = &curves.kp_logs {
                    expand_curve_log_kp(
                        curve,
                        x2c.view(),
                        &risk2,
                        &log_risk.as_scalar(),
                        &kp[position],
                        varmat,
                    )?
                } else {
                    expand_curve_validated(curve, survtype, x2c.view(), &risk2, varmat)?
                };
                result.push(finish_curve(
                    self.sorted.codes[position],
                    expanded,
                    options.censor,
                ));
            }
        }
        Ok(result)
    }

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
        let log_risk = if newdata.is_none() && self.has_curve_origins() {
            self.training_prediction_risks(&curves.origins)
        } else {
            self.prediction_log_risks_at(x, offset, &curves.origins, strata)
        };
        let risk = log_risk.risks();
        let stable = log_risk.needs_logs(curves.stable);
        for (position, (curve, rows)) in curves.iter().zip(rows_by_stratum).enumerate() {
            if rows.is_empty() {
                continue;
            }
            for (i, &at) in times.iter().enumerate() {
                let index = find_interval(&curve.time, at, false);
                if index == 0 {
                    continue;
                }
                let baseline = (-curve.cumhaz[index - 1]).exp();
                let mut output = result.row_mut(i);
                for &row in &rows {
                    output[row] = if stable {
                        (-curves.logs()[position].log_cumulative[index - 1]
                            .exp_at_scale(log_risk.scale(row)))
                        .exp()
                    } else if baseline == 0.0 && curve.cumhaz[index - 1].is_finite() {
                        (-curve.cumhaz[index - 1] * risk[row]).exp()
                    } else {
                        baseline.powf(risk[row])
                    };
                }
            }
        }
        Ok(result)
    }

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
        let curves = self.baseline_curves()?;
        let log_risk = self.prediction_log_risks_at(
            newdata.x.view(),
            newdata.offset.as_deref(),
            &curves.origins,
            newdata.strata.as_deref(),
        );
        let risk = log_risk.risks();
        let strata: Vec<usize> = (0..newdata.nrows())
            .map(|i| {
                newdata.strata.as_ref().map_or(0, |codes| {
                    self.sorted
                        .position_of(codes[i])
                        .expect("strata were checked")
                })
            })
            .collect();
        if log_risk.needs_logs(curves.stable) {
            let events: Vec<Vec<usize>> = curves
                .iter()
                .map(|curve| {
                    curve
                        .n_event
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &n)| (n > 0.0).then_some(i))
                        .collect()
                })
                .collect();
            let baselines: Vec<CoxExpectedBaseline> = curves
                .iter()
                .zip(&events)
                .map(|(curve, events)| CoxExpectedBaseline {
                    time: events.iter().map(|&i| curve.time[i]).collect(),
                    cumhaz: vec![0.0; events.len()],
                })
                .collect();
            let evaluate = |row: usize, previous: usize, current: usize| {
                let s = strata[row];
                let first = if previous == 0 {
                    0
                } else {
                    events[s][previous - 1] + 1
                };
                let end = if current == 0 {
                    0
                } else {
                    events[s][current - 1] + 1
                };
                let curve = &curves.logs()[s];
                (
                    curve.range_hazard(0, end, log_risk.scale(row)),
                    curve.range_hazard(0, first, log_risk.scale(row)),
                    curve.range_log_hazard(first, end, log_risk.scale(row)),
                )
            };
            return crate::population::survexp_cox_evaluated(
                &baselines,
                &vec![1.0; risk.len()],
                &strata,
                group,
                weights,
                y,
                times,
                method,
                Some(&evaluate),
            );
        }
        let baselines = curves
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
        let (x2c, ordinary_log_risk) = self.centered_newdata(newdata);
        let log_risk = if curves.stable {
            self.prediction_log_risks_at(
                newdata.x.view(),
                newdata.offset.as_deref(),
                &curves.origins,
                newdata.strata.as_deref(),
            )
        } else {
            ordinary_log_risk
        };
        let risk2 = log_risk.risks();
        let stable = log_risk.needs_logs(curves.stable) || curves.kp_logs.is_some();
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
            let subject_log_risk: Vec<LogScale> =
                subject_rows.iter().map(|&i| log_risk.scale(i)).collect();
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
            let curve = if stable {
                super::log_curves::individual(&curves, &intervals, &subject_log_risk, varmat)
            } else {
                individual_curve_validated(&curves, survtype, &intervals, varmat)?
            };
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
