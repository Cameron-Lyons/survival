//! Residuals of a fitted Cox model: R survival's `residuals.coxph()`
//! (`R/residuals.coxph.R`) and the C kernels it calls.
//!
//! * martingale — `coxmart.c` / `agmart3.c`: `status - risk * (H(stop) -
//!   H(entry))`, with the Efron correction for a subject's own tied death;
//! * score — `coxscore2.c` / `agscore3.c`: `int (x - xbar(t)) dM_i(t)`;
//! * schoenfeld — `coxscho.c`: `x_k - xbar(t_k)` for each death;
//! * deviance, dfbeta, dfbetas, scaledsch and partial are the algebra of
//!   `residuals.coxph` on top of those, including the `weighted` and
//!   `collapse` arguments.
//!
//! All kernels are single backward sweeps over the sorted rows of each
//! stratum ([`StratumSweep`]) followed by a lookup of the accumulated hazard
//! at every row's entry and stop time, so a residual costs `O(n log n)`
//! rather than the `O(deaths x n)` scan of the older C code.

use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::{CoxPHFit, PredictReference, default_assign, validate_assign};
use crate::regression::coxph_support::{DeathTime, StratumSweep, step_value};
use ndarray::Array2;
use pyo3::prelude::*;

/// The residual types of `residuals.coxph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualType {
    Martingale,
    Deviance,
    Score,
    Schoenfeld,
    Dfbeta,
    Dfbetas,
    ScaledSchoenfeld,
    Partial,
}

impl ResidualType {
    pub fn parse(name: &str) -> SurvivalResult<Self> {
        match name {
            "martingale" => Ok(Self::Martingale),
            "deviance" => Ok(Self::Deviance),
            "score" => Ok(Self::Score),
            "schoenfeld" => Ok(Self::Schoenfeld),
            "dfbeta" => Ok(Self::Dfbeta),
            "dfbetas" => Ok(Self::Dfbetas),
            "scaledsch" => Ok(Self::ScaledSchoenfeld),
            "partial" => Ok(Self::Partial),
            other => Err(SurvivalError::invalid_input(format!(
                "unknown residual type '{other}'"
            ))),
        }
    }

    /// R's default for `weighted`: `TRUE` for dfbeta and dfbetas only.
    pub fn default_weighted(self) -> bool {
        matches!(self, Self::Dfbeta | Self::Dfbetas)
    }
}

/// Schoenfeld (or scaled Schoenfeld) residuals: one row per death, in
/// (stratum, time) order.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct SchoenfeldResiduals {
    /// Death times (the row names of R's matrix).
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Stratum code of each death (absent for an unstratified fit).
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
    /// Original row index of each death.
    #[pyo3(get)]
    pub rows: Vec<usize>,
    #[pyo3(get)]
    pub residuals: Vec<Vec<f64>>,
}

/// A residual vector or matrix, R's `residuals(fit, type)` value.
#[derive(Debug, Clone, PartialEq)]
pub enum Residuals {
    Vector(Vec<f64>),
    Matrix(Array2<f64>),
}

/// `exp(lp)` with `coxph.fit`'s overflow guard: near-infinite coefficients
/// are shifted so the largest score is representable.
fn risk_scores(lp: &[f64]) -> Vec<f64> {
    let log_max = f64::MAX.ln();
    let max_lp = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shift = if max_lp > log_max {
        log_max - (1.0 + max_lp)
    } else {
        0.0
    };
    lp.iter().map(|value| (value + shift).exp()).collect()
}

/// Per-death-time hazard pieces of one stratum, ascending in time.
struct HazardSteps {
    time: Vec<f64>,
    /// Hazard increment at each death time.
    hazard: Vec<f64>,
    /// Hazard increment seen by a death tied at the time (Efron).
    e_hazard: Vec<f64>,
    cumhaz: Vec<f64>,
    /// Cumulative `sum_j h_j xbar_j`, one vector per covariate.
    xhaz: Vec<Vec<f64>>,
}

/// One Efron (or Breslow) step of a death time: the `j`-th of `steps`.
struct HazardStep<'a> {
    xbar: &'a [f64],
    hazard: f64,
    /// `j / d`, the fraction of the tied deaths removed from the risk set.
    fraction: f64,
    index: usize,
    steps: usize,
}

impl CoxPHFit {
    fn stratum_sweep<'a>(
        &'a self,
        stratum: usize,
        risk: &'a [f64],
        second_moments: bool,
    ) -> StratumSweep<'a> {
        let (start, end) = self.sorted.bounds[stratum];
        StratumSweep {
            stop: &self.time,
            entry: self.entry.as_deref(),
            status: &self.status,
            x: self.x.view(),
            weights: &self.weights,
            risk,
            rows: &self.sorted.order[start..end],
            second_moments,
        }
    }

    /// Hazard increments of one stratum; `on_step` sees each Efron step of
    /// each death time (descending times).
    fn hazard_steps(
        &self,
        stratum: usize,
        risk: &[f64],
        mut on_step: impl FnMut(&DeathTime<'_>, &HazardStep<'_>),
    ) -> HazardSteps {
        let nvar = self.nvar();
        let efron = self.method == TieMethod::Efron;
        let mut time = Vec::new();
        let mut hazard = Vec::new();
        let mut e_hazard = Vec::new();
        let mut xbar_hazard: Vec<Vec<f64>> = Vec::new();
        let mut xbar = vec![0.0; nvar];
        self.stratum_sweep(stratum, risk, false)
            .for_each_death_time(|death| {
                let d = death.ndead();
                let steps = if efron && d > 1 { d } else { 1 };
                let wtsum = death.tied.weight / steps as f64;
                let mut total = 0.0;
                let mut e_total = 0.0;
                let mut xh = vec![0.0; nvar];
                for j in 0..steps {
                    let fraction = if steps == 1 { 0.0 } else { j as f64 / d as f64 };
                    let denom = death.risk_set.denom - fraction * death.tied.denom;
                    let h = wtsum / denom;
                    total += h;
                    e_total += h * (1.0 - fraction);
                    for i in 0..nvar {
                        xbar[i] = (death.risk_set.a[i] - fraction * death.tied.a[i]) / denom;
                        xh[i] += xbar[i] * h;
                    }
                    on_step(
                        death,
                        &HazardStep {
                            xbar: &xbar,
                            hazard: h,
                            fraction,
                            index: j,
                            steps,
                        },
                    );
                }
                time.push(death.time);
                hazard.push(total);
                e_hazard.push(e_total);
                xbar_hazard.push(xh);
            });
        time.reverse();
        hazard.reverse();
        e_hazard.reverse();
        xbar_hazard.reverse();
        let mut cumhaz = hazard.clone();
        let mut running = 0.0;
        for value in cumhaz.iter_mut() {
            running += *value;
            *value = running;
        }
        let mut xhaz = vec![Vec::with_capacity(time.len()); nvar];
        let mut running = vec![0.0; nvar];
        for xh in &xbar_hazard {
            for i in 0..nvar {
                running[i] += xh[i];
                xhaz[i].push(running[i]);
            }
        }
        HazardSteps {
            time,
            hazard,
            e_hazard,
            cumhaz,
            xhaz,
        }
    }

    /// Increment of a per-time step series over a row's (entry, stop].
    fn interval_increment(&self, times: &[f64], values: &[f64], row: usize) -> f64 {
        let at_stop = step_value(times, values, self.time[row]);
        let at_entry = self
            .entry
            .as_ref()
            .map_or(0.0, |entry| step_value(times, values, entry[row]));
        at_stop - at_entry
    }
}

/// Martingale residuals at the linear predictors `lp` (`coxmart.c`,
/// `agmart3.c`; the Breslow form for the exact method, as `coxmart2.c`).
pub(crate) fn martingale_residuals(fit: &CoxPHFit, lp: &[f64]) -> Vec<f64> {
    let risk = risk_scores(lp);
    let mut resid: Vec<f64> = fit.status.iter().map(|&s| f64::from(s)).collect();
    for stratum in 0..fit.sorted.nstrata() {
        let steps = fit.hazard_steps(stratum, &risk, |_, _| {});
        let (start, end) = fit.sorted.bounds[stratum];
        for &row in &fit.sorted.order[start..end] {
            resid[row] -= risk[row] * fit.interval_increment(&steps.time, &steps.cumhaz, row);
            if fit.status[row] == 1 {
                // A death only experiences the Efron share of its own time.
                let g = steps.time.partition_point(|&t| t < fit.time[row]);
                resid[row] += risk[row] * (steps.hazard[g] - steps.e_hazard[g]);
            }
        }
    }
    resid
}

/// Score residuals (`n x nvar`) at the linear predictors `lp`
/// (`coxscore2.c`, `agscore3.c`).
pub(crate) fn score_residuals(fit: &CoxPHFit, lp: &[f64]) -> SurvivalResult<Array2<f64>> {
    if fit.method == TieMethod::Exact {
        return Err(SurvivalError::invalid_input(
            "score residuals are not available for the exact method",
        ));
    }
    let nvar = fit.nvar();
    let risk = risk_scores(lp);
    let mut resid = Array2::zeros((fit.n, nvar));
    for stratum in 0..fit.sorted.nstrata() {
        let steps = fit.hazard_steps(stratum, &risk, |death, step| {
            // The deaths' own contribution: (x - xbar) dN, spread over the
            // Efron steps with the partial hazard they experience.
            let d = death.ndead() as f64;
            for &row in death.deaths {
                for i in 0..nvar {
                    let centered = fit.x[(row, i)] - step.xbar[i];
                    resid[(row, i)] += if step.steps == 1 {
                        centered
                    } else {
                        centered / d + centered * risk[row] * step.hazard * step.fraction
                    };
                }
            }
        });
        let (start, end) = fit.sorted.bounds[stratum];
        for &row in &fit.sorted.order[start..end] {
            let dh = fit.interval_increment(&steps.time, &steps.cumhaz, row);
            for i in 0..nvar {
                let dxh = fit.interval_increment(&steps.time, &steps.xhaz[i], row);
                resid[(row, i)] -= risk[row] * (fit.x[(row, i)] * dh - dxh);
            }
        }
    }
    Ok(resid)
}

/// Schoenfeld residuals of the deaths (`coxscho.c`), in (stratum, time)
/// order; `weighted` multiplies each row by its case weight.
pub(crate) fn schoenfeld_residuals(
    fit: &CoxPHFit,
    weighted: bool,
) -> SurvivalResult<SchoenfeldResiduals> {
    if fit.method == TieMethod::Exact {
        return Err(SurvivalError::invalid_input(
            "schoenfeld residuals are not available for the exact method",
        ));
    }
    let nvar = fit.nvar();
    let risk = risk_scores(&fit.linear_predictors);
    let mut time = Vec::new();
    let mut strata = Vec::new();
    let mut rows = Vec::new();
    let mut residuals = Vec::new();
    for stratum in 0..fit.sorted.nstrata() {
        let mut per_stratum: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut mean = vec![0.0; nvar];
        fit.hazard_steps(stratum, &risk, |death, step| {
            // Efron: the mean of the step means, one row per death.
            if step.index == 0 {
                mean.fill(0.0);
            }
            for (value, xbar) in mean.iter_mut().zip(step.xbar) {
                *value += xbar / step.steps as f64;
            }
            if step.index + 1 == step.steps {
                for &row in death.deaths.iter().rev() {
                    let scale = if weighted { fit.weights[row] } else { 1.0 };
                    per_stratum.push((
                        row,
                        (0..nvar)
                            .map(|i| (fit.x[(row, i)] - mean[i]) * scale)
                            .collect(),
                    ));
                }
            }
        });
        per_stratum.reverse();
        for (row, values) in per_stratum {
            time.push(fit.time[row]);
            strata.push(fit.sorted.codes[stratum]);
            rows.push(row);
            residuals.push(values);
        }
    }
    Ok(SchoenfeldResiduals {
        time,
        strata: fit.strata.as_ref().map(|_| strata),
        rows,
        residuals,
    })
}

/// Sorted unique cluster codes and each row's position among them.
fn cluster_groups(collapse: &[i32]) -> (usize, Vec<usize>) {
    let mut codes = collapse.to_vec();
    codes.sort_unstable();
    codes.dedup();
    let positions = collapse
        .iter()
        .map(|code| codes.binary_search(code).expect("code is present"))
        .collect();
    (codes.len(), positions)
}

/// `residuals.coxph`'s finishing steps for a matrix: multiply the rows by
/// the case weights (`weighted`) and sum them by cluster (`collapse`,
/// `rowsum` in ascending code order).
pub(crate) fn collapse_rows(
    rows: &Array2<f64>,
    weights: Option<&[f64]>,
    collapse: Option<&[i32]>,
) -> Array2<f64> {
    let mut weighted = rows.clone();
    if let Some(weights) = weights {
        for (i, mut row) in weighted.outer_iter_mut().enumerate() {
            row.mapv_inplace(|value| value * weights[i]);
        }
    }
    let Some(collapse) = collapse else {
        return weighted;
    };
    let (ngroups, positions) = cluster_groups(collapse);
    let mut collapsed = Array2::zeros((ngroups, rows.ncols()));
    for (i, row) in weighted.outer_iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            collapsed[(positions[i], j)] += value;
        }
    }
    collapsed
}

fn collapse_vector(values: &[f64], weights: Option<&[f64]>, collapse: Option<&[i32]>) -> Vec<f64> {
    let column = Array2::from_shape_vec((values.len(), 1), values.to_vec())
        .expect("a column vector always has a valid shape");
    collapse_rows(&column, weights, collapse).column(0).to_vec()
}

impl CoxPHFit {
    fn check_collapse(&self, collapse: Option<&[i32]>) -> SurvivalResult<()> {
        if let Some(collapse) = collapse
            && collapse.len() != self.n
        {
            return Err(SurvivalError::invalid_input("Wrong length for 'collapse'"));
        }
        Ok(())
    }

    /// Score residuals at `lp` times the model variance, weighted and
    /// collapsed as requested (`residuals(type = "dfbeta")`).
    pub(crate) fn dfbeta_matrix(
        &self,
        lp: &[f64],
        weighted: bool,
        collapse: Option<&[i32]>,
    ) -> SurvivalResult<Array2<f64>> {
        let vv = self.naive_var.as_ref().unwrap_or(&self.var);
        let dfbeta = score_residuals(self, lp)?.dot(vv);
        Ok(collapse_rows(
            &dfbeta,
            weighted.then_some(self.weights.as_slice()),
            collapse,
        ))
    }

    /// `residuals(fit, type, weighted, collapse)`.  `assign` (the columns
    /// of each term) only matters for `partial`; `weighted` defaults per
    /// type as in R.
    pub fn residuals(
        &self,
        kind: ResidualType,
        weighted: Option<bool>,
        collapse: Option<&[i32]>,
        assign: Option<&[Vec<usize>]>,
    ) -> SurvivalResult<Residuals> {
        self.check_collapse(collapse)?;
        let weighted = weighted.unwrap_or(kind.default_weighted());
        let weights = weighted.then_some(self.weights.as_slice());
        match kind {
            ResidualType::Martingale => Ok(Residuals::Vector(collapse_vector(
                &self.residuals,
                weights,
                collapse,
            ))),
            ResidualType::Deviance => {
                let rr = collapse_vector(&self.residuals, weights, collapse);
                let status: Vec<f64> = self.status.iter().map(|&s| f64::from(s)).collect();
                let status = collapse_vector(&status, None, collapse);
                Ok(Residuals::Vector(
                    rr.iter()
                        .zip(&status)
                        .map(|(&r, &s)| {
                            let inner = r + if s == 0.0 { 0.0 } else { s * (s - r).ln() };
                            r.signum() * (-2.0 * inner).sqrt()
                        })
                        .collect(),
                ))
            }
            ResidualType::Score => Ok(Residuals::Matrix(collapse_rows(
                &score_residuals(self, &self.linear_predictors)?,
                weights,
                collapse,
            ))),
            ResidualType::Dfbeta => Ok(Residuals::Matrix(self.dfbeta_matrix(
                &self.linear_predictors,
                weighted,
                collapse,
            )?)),
            ResidualType::Dfbetas => {
                let vv = self.naive_var.as_ref().unwrap_or(&self.var);
                let mut dfbetas =
                    self.dfbeta_matrix(&self.linear_predictors, weighted, collapse)?;
                for j in 0..self.nvar() {
                    let scale = 1.0 / vv[(j, j)].sqrt();
                    dfbetas.column_mut(j).mapv_inplace(|value| value * scale);
                }
                Ok(Residuals::Matrix(dfbetas))
            }
            ResidualType::Schoenfeld => {
                let schoenfeld = schoenfeld_residuals(self, weighted)?;
                Ok(Residuals::Matrix(rows_matrix(
                    &schoenfeld.residuals,
                    self.nvar(),
                )))
            }
            ResidualType::ScaledSchoenfeld => {
                let scaled = self.scaled_schoenfeld_residuals(weighted)?;
                Ok(Residuals::Matrix(rows_matrix(
                    &scaled.residuals,
                    self.nvar(),
                )))
            }
            ResidualType::Partial => {
                let default = default_assign(self.nvar());
                let assign = assign.unwrap_or(&default);
                validate_assign(assign, self.nvar())?;
                let terms = self.predict_terms(None, false, PredictReference::Sample, assign)?;
                let mut partial = Array2::zeros((self.n, assign.len()));
                for i in 0..self.n {
                    let scale = if weighted { self.weights[i] } else { 1.0 };
                    for t in 0..assign.len() {
                        partial[(i, t)] = self.residuals[i] * scale + terms.fit[i][t];
                    }
                }
                Ok(Residuals::Matrix(collapse_rows(&partial, None, collapse)))
            }
        }
    }

    /// `residuals(type = "scaledsch")`: `rr %*% vv * ndead + coef`.
    pub fn scaled_schoenfeld_residuals(
        &self,
        weighted: bool,
    ) -> SurvivalResult<SchoenfeldResiduals> {
        let mut schoenfeld = schoenfeld_residuals(self, weighted)?;
        let vv = self.naive_var.as_ref().unwrap_or(&self.var);
        let coef = self.coefficients_or_zero();
        let ndead = schoenfeld.residuals.len() as f64;
        for row in schoenfeld.residuals.iter_mut() {
            let scaled: Vec<f64> = (0..self.nvar())
                .map(|j| {
                    row.iter()
                        .enumerate()
                        .map(|(i, &value)| value * vv[(i, j)])
                        .sum::<f64>()
                        * ndead
                        + coef[j]
                })
                .collect();
            *row = scaled;
        }
        Ok(schoenfeld)
    }
}

fn rows_matrix(rows: &[Vec<f64>], ncols: usize) -> Array2<f64> {
    Array2::from_shape_vec(
        (rows.len(), ncols),
        rows.iter().flatten().copied().collect(),
    )
    .expect("rows have the coefficient width")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxph::{CoxphData, CoxphOptions};

    fn fit(method: TieMethod, entry: Option<Vec<f64>>) -> CoxPHFit {
        let weighted = method != TieMethod::Exact;
        let time = vec![2.0, 2.0, 3.0, 4.0, 4.0, 3.0, 5.0, 5.0];
        let status = vec![1, 1, 0, 1, 0, 1, 1, 0];
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                -1.2, 0.5, 0.4, -1.0, 1.1, 0.3, -0.3, 1.2, 0.8, -0.7, 1.7, 0.9, -0.9, 0.1, 0.2,
                -1.3,
            ],
        )
        .unwrap();
        let weights = vec![1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9, 1.3];
        let strata = vec![0, 0, 0, 0, 0, 1, 1, 1];
        let data = CoxphData::try_new(
            time,
            entry,
            status,
            x,
            weighted.then_some(weights),
            Some(strata),
            None,
        )
        .unwrap();
        let options = CoxphOptions {
            method,
            init: Some(vec![0.2, -0.1]),
            iter_max: 0,
            ..CoxphOptions::default()
        };
        CoxPHFit::fit(data, options).unwrap()
    }

    fn assert_close_rows(actual: &Array2<f64>, expected: &[Vec<f64>]) {
        assert_eq!(actual.nrows(), expected.len());
        for (row, expected) in actual.outer_iter().zip(expected) {
            for (a, e) in row.iter().zip(expected) {
                assert!((a - e).abs() < 1e-10, "{a} != {e}");
            }
        }
    }

    #[test]
    fn counting_score_residuals_match_r_for_weights_strata_and_ties() {
        // R: coxph(Surv(start, stop, status) ~ x1 + x2 + strata(g), weights,
        //    init = c(0.2, -0.1), iter.max = 0); residuals(type = "score")
        let entry = Some(vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0]);
        let cases = [
            (
                TieMethod::Breslow,
                vec![
                    vec![-0.778427039030269, 0.283533537794141],
                    vec![0.091186357321952, -0.342237690541851],
                    vec![-0.650093055328823, -0.190343304961445],
                    vec![-0.0420910220554235, -0.132051648232679],
                    vec![-0.469472843288164, 0.810907638406828],
                    vec![0.709746822901141, 0.66611561001089],
                    vec![-0.143052423889649, 0.568213016404157],
                    vec![-0.0432781142437781, 0.608556079386341],
                ],
            ),
            (
                TieMethod::Efron,
                vec![
                    vec![-0.890022262887945, 0.233804575353994],
                    vec![0.0973290025651707, -0.50524720360612],
                    vec![-0.741057387549249, -0.122553172255785],
                    vec![0.0221464326595451, -0.166914303193565],
                    vec![-0.469472843288164, 0.810907638406828],
                    vec![0.709746822901141, 0.66611561001089],
                    vec![-0.143052423889649, 0.568213016404157],
                    vec![-0.0432781142437781, 0.608556079386341],
                ],
            ),
        ];
        for (method, expected) in cases {
            let model = fit(method, entry.clone());
            let actual = score_residuals(&model, &model.linear_predictors).unwrap();
            assert_close_rows(&actual, &expected);
        }
    }

    #[test]
    fn martingale_residuals_sum_to_zero_and_match_expected_counts() {
        for method in [TieMethod::Breslow, TieMethod::Efron, TieMethod::Exact] {
            let model = fit(method, None);
            let total: f64 = model
                .residuals
                .iter()
                .zip(&model.weights)
                .map(|(r, w)| r * w)
                .sum();
            assert!(total.abs() < 1e-10, "{method:?}: weighted sum {total}");
            let expected = model.predict_expected(None, false).unwrap();
            for (i, e) in expected.fit.iter().enumerate() {
                assert!((f64::from(model.status[i]) - model.residuals[i] - e).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn deviance_residuals_follow_r_formula_and_collapse() {
        let model = fit(TieMethod::Efron, None);
        let Residuals::Vector(deviance) = model
            .residuals(ResidualType::Deviance, None, None, None)
            .unwrap()
        else {
            panic!("vector residuals")
        };
        for (i, d) in deviance.iter().enumerate() {
            let r = model.residuals[i];
            let s = f64::from(model.status[i]);
            let expected =
                r.signum() * (-2.0 * (r + if s == 0.0 { 0.0 } else { s * (s - r).ln() })).sqrt();
            assert!((d - expected).abs() < 1e-12);
        }
        let collapse = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let Residuals::Vector(collapsed) = model
            .residuals(ResidualType::Martingale, None, Some(&collapse), None)
            .unwrap()
        else {
            panic!("vector residuals")
        };
        assert_eq!(collapsed.len(), 4);
        assert!((collapsed[0] - (model.residuals[0] + model.residuals[1])).abs() < 1e-12);
    }

    #[test]
    fn schoenfeld_rows_are_deaths_in_stratum_time_order() {
        let model = fit(TieMethod::Efron, None);
        let schoenfeld = schoenfeld_residuals(&model, false).unwrap();
        assert_eq!(schoenfeld.time, vec![2.0, 2.0, 4.0, 3.0, 5.0]);
        assert_eq!(schoenfeld.rows, vec![0, 1, 3, 5, 6]);
        assert_eq!(schoenfeld.strata, Some(vec![0, 0, 0, 1, 1]));
        // Within a stratum the Schoenfeld residuals sum to the score.
        let scaled = model.scaled_schoenfeld_residuals(false).unwrap();
        assert_eq!(scaled.residuals.len(), 5);
        let weighted = schoenfeld_residuals(&model, true).unwrap();
        assert!((weighted.residuals[1][0] - schoenfeld.residuals[1][0] * 1.5).abs() < 1e-12);
    }

    #[test]
    fn dfbeta_variants_share_the_score_residuals() {
        let model = fit(TieMethod::Breslow, None);
        let Residuals::Matrix(dfbeta) = model
            .residuals(ResidualType::Dfbeta, None, None, None)
            .unwrap()
        else {
            panic!("matrix residuals")
        };
        let Residuals::Matrix(dfbetas) = model
            .residuals(ResidualType::Dfbetas, None, None, None)
            .unwrap()
        else {
            panic!("matrix residuals")
        };
        for i in 0..8 {
            for j in 0..2 {
                assert!(
                    (dfbetas[(i, j)] - dfbeta[(i, j)] / model.var[(j, j)].sqrt()).abs() < 1e-12
                );
            }
        }
        let Residuals::Matrix(partial) = model
            .residuals(ResidualType::Partial, None, None, None)
            .unwrap()
        else {
            panic!("matrix residuals")
        };
        assert_eq!(partial.dim(), (8, 2));
        assert!(matches!(
            model.residuals(ResidualType::Score, None, Some(&[0, 1]), None),
            Err(SurvivalError::InvalidInput(_))
        ));
        let exact = fit(TieMethod::Exact, None);
        assert!(
            exact
                .residuals(ResidualType::Score, None, None, None)
                .is_err()
        );
    }
}
