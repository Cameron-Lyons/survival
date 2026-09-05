//! Joint Cox ridge fitting with R's effective-df outer controller.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::coxph::{CoxPHFit, coxph_penalized_fit};
use super::coxph_penalty::{CoxPenaltyDiagnostics, validate_penalty};
use crate::constants::COX_MAX_ITER;

/// Applied penalties and controller proposals are deliberately separate: R's
/// controller proposes another theta even when its current fit has converged.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CoxRidgeSelection {
    #[pyo3(get)]
    pub fitted_theta: Vec<f64>,
    #[pyo3(get)]
    pub proposed_theta: Vec<f64>,
    #[pyo3(get)]
    pub done: Vec<bool>,
    #[pyo3(get)]
    pub histories: Vec<Vec<Vec<f64>>>,
    #[pyo3(get)]
    pub halves: Vec<usize>,
    #[pyo3(get)]
    pub outer_iterations: usize,
    #[pyo3(get)]
    pub inner_iterations: usize,
    #[pyo3(get)]
    pub inner_failures: Vec<usize>,
    #[pyo3(get)]
    pub penalty: Vec<f64>,
    #[pyo3(get)]
    pub initial_loglik: f64,
}

#[derive(Clone, Debug)]
struct DfController {
    target: f64,
    tolerance: f64,
    history: Vec<[f64; 2]>,
    proposed_theta: f64,
    done: bool,
    half: usize,
}

impl DfController {
    fn new(width: usize, target: f64, tolerance: f64) -> Self {
        Self {
            target,
            tolerance,
            // This is R's synthetic reference, not an unpenalized fit.
            history: vec![[0.0, width as f64]],
            proposed_theta: 1.0,
            done: false,
            half: 0,
        }
    }

    fn update(&mut self, applied_theta: f64, observed_df: f64) {
        self.history.push([applied_theta, observed_df]);
        if self.history.len() == 2 {
            let [x0, y0] = self.history[0];
            self.proposed_theta =
                x0 + (applied_theta - x0) * (self.target - y0) / (observed_df - y0);
            if self.target > observed_df {
                self.proposed_theta *= 1.5;
            }
            // The first completed fit never terminates R's df search.
            self.done = false;
            self.half = 0;
            self.recover_undefined_proposal(applied_theta);
            return;
        }

        self.done = (observed_df - self.target).abs() < self.tolerance;
        let (proposal, half) = self.interpolate();
        self.proposed_theta = proposal;
        self.half = half;
        self.recover_undefined_proposal(applied_theta);
    }

    fn recover_undefined_proposal(&mut self, applied_theta: f64) {
        if self.done || (self.proposed_theta.is_finite() && self.proposed_theta >= 0.0) {
            return;
        }
        // A tiny penalty can round its df to the synthetic theta-zero df,
        // making R's shifted-power interpolation undefined. Bisect the
        // narrowest observed bracket instead of inventing a df perturbation.
        // Completed full-df controllers retain R's unused NaN proposal.
        let mut points = self.history.clone();
        points.retain(|point| point[0].is_finite() && point[1].is_finite());
        points.sort_by(|left, right| left[0].total_cmp(&right[0]));
        let mut midpoint = applied_theta;
        let mut narrowest = f64::INFINITY;
        for pair in points.windows(2) {
            let [left, right] = [pair[0], pair[1]];
            let width = right[0] - left[0];
            let brackets_target = (left[1] <= self.target && self.target <= right[1])
                || (right[1] <= self.target && self.target <= left[1]);
            if width > 0.0 && width < narrowest && brackets_target {
                midpoint = left[0] + width / 2.0;
                narrowest = width;
            }
        }
        self.proposed_theta = midpoint;
        self.half = 0;
    }

    /// Direct translation of survival::frailty.controldf. Sorting only affects
    /// interpolation; public history stays in evaluation order, with duplicates.
    fn interpolate(&self) -> (f64, usize) {
        let n = self.history.len();
        let last_error = self.history[n - 1][1] - self.target;
        let previous_error = self.history[n - 2][1] - self.target;
        let doing_well = (last_error / previous_error).abs() <= 0.6;
        let mut points = self.history.clone();
        points.sort_by(|left, right| left[0].total_cmp(&right[0]));
        let direction = if (self.history[0][0] - self.history[1][0])
            * (self.history[0][1] - self.history[1][1])
            > 0.0
        {
            1.0
        } else {
            -1.0
        };
        let target = direction * self.target;
        for point in &mut points {
            point[1] *= direction;
        }

        let mut base;
        if points.iter().all(|point| point[1] > target) {
            base = 0;
        } else if points.iter().all(|point| point[1] < target) {
            base = n - 3;
        } else {
            let Some(bracket) = points.iter().rposition(|point| point[1] <= target) else {
                return (f64::NAN, 0);
            };
            base = bracket;
            let Some(next) = points.get(base + 1) else {
                return (f64::NAN, 0);
            };
            if !doing_well && self.half < 2 {
                return ((points[base][0] + next[0]) / 2.0, self.half + 1);
            }
            if base + 1 == n - 1 || (base > 0 && target - points[base][1] < next[1] - target) {
                base = base.saturating_sub(1);
            }
        }

        let Some(second) = points.get(base + 2) else {
            return (f64::NAN, 0);
        };
        let origin = points[base];
        let first = points[base + 1];
        let x1 = (first[0] - origin[0]).ln();
        let x2 = (second[0] - origin[0]).ln();
        let y1 = (first[1] - origin[1]).ln();
        let y2 = (second[1] - origin[1]).ln();
        let power = (y2 - y1) / (x2 - x1);
        let intercept = y1 - power * x1;
        (
            origin[0] + (((target - origin[1]).ln() - intercept) / power).exp(),
            0,
        )
    }
}

fn closest_theta(previous: &[Vec<f64>], proposed: &[f64]) -> usize {
    let mut nearest = 0;
    let mut nearest_distance = f64::INFINITY;
    for (index, theta) in previous.iter().enumerate() {
        let distance = theta
            .iter()
            .zip(proposed)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>();
        // Strict comparison retains R's earliest fit when distances tie.
        if distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }
    nearest
}

#[allow(clippy::too_many_arguments)]
fn validate_selection(
    scales: &[f64],
    groups: &[Vec<usize>],
    theta: &[Option<f64>],
    df: &[Option<f64>],
    tolerances: &[f64],
    nvar: usize,
    outer_max: usize,
) -> PyResult<()> {
    validate_penalty(scales, groups, nvar)?;
    if theta.len() != groups.len() || df.len() != groups.len() || tolerances.len() != groups.len() {
        return Err(PyValueError::new_err(
            "theta, df and df_tolerances must have one value per term group",
        ));
    }
    if outer_max == 0 {
        return Err(PyValueError::new_err("outer_max_iter must be positive"));
    }
    for (index, group) in groups.iter().enumerate() {
        if !tolerances[index].is_finite() || tolerances[index] <= 0.0 {
            return Err(PyValueError::new_err(
                "df_tolerances must be finite and positive",
            ));
        }
        match (theta[index], df[index]) {
            (Some(value), None) if value.is_finite() && value >= 0.0 => {}
            (None, Some(value))
                if value.is_finite() && value >= 0.0 && value <= group.len() as f64 => {}
            (Some(_), None) => {
                return Err(PyValueError::new_err(
                    "fixed theta must be finite and non-negative",
                ));
            }
            (None, Some(_)) => {
                return Err(PyValueError::new_err(
                    "df must be finite and between zero and the number of columns in its term",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "supply exactly one of theta or df for each term group",
                ));
            }
        }
    }
    Ok(())
}

/// Fit all Cox terms jointly while selecting requested ridge effective dfs.
///
/// `penalty_scales` contains frozen sample variances or unit scales. Groups
/// partition all columns; ordinary terms use zero scales and fixed theta zero.
/// Each group has exactly one fixed theta or df target. Selected groups start
/// at theta one and use R's absolute-df interpolation controller.
#[pyfunction]
#[pyo3(signature = (time, status, covariates, penalty_scales, term_groups, theta, df, df_tolerances, strata=None, weights=None, offset=None, initial_beta=None, max_iter=None, eps=None, toler=None, method=None, entry_times=None, nocenter=None, outer_max_iter=None))]
#[allow(clippy::too_many_arguments)]
pub fn coxph_ridge_fit(
    time: Vec<f64>,
    status: Vec<i32>,
    covariates: Vec<Vec<f64>>,
    penalty_scales: Vec<f64>,
    term_groups: Vec<Vec<usize>>,
    theta: Vec<Option<f64>>,
    df: Vec<Option<f64>>,
    df_tolerances: Vec<f64>,
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
    outer_max_iter: Option<usize>,
) -> PyResult<(CoxPHFit, CoxPenaltyDiagnostics, CoxRidgeSelection)> {
    let outer_max = outer_max_iter.unwrap_or(10);
    let nvar = covariates.first().map_or(0, Vec::len);
    validate_selection(
        &penalty_scales,
        &term_groups,
        &theta,
        &df,
        &df_tolerances,
        nvar,
        outer_max,
    )?;
    let mut controllers: Vec<Option<DfController>> = df
        .iter()
        .enumerate()
        .map(|(index, target)| {
            target.map(|target| {
                DfController::new(term_groups[index].len(), target, df_tolerances[index])
            })
        })
        .collect();
    let mut applied_theta: Vec<f64> = theta.iter().map(|value| value.unwrap_or(1.0)).collect();
    let mut next_beta = initial_beta.clone();
    let mut fitted_thetas = Vec::new();
    let mut fitted_betas = Vec::new();
    let mut inner_iterations = 0usize;
    let mut inner_failures = Vec::new();
    let inner_max = max_iter.unwrap_or(COX_MAX_ITER);
    let mut first_penalty = 0.0;
    let mut initial_loglik = 0.0;
    let mut initial_unpenalized_loglik = 0.0;

    for outer in 1..=outer_max {
        let mut diagonal = vec![0.0; nvar];
        for (index, group) in term_groups.iter().enumerate() {
            for &column in group {
                diagonal[column] = penalty_scales[column] * applied_theta[index];
            }
        }
        let (mut fit, diagnostics) = coxph_penalized_fit(
            time.clone(),
            status.clone(),
            covariates.clone(),
            diagonal,
            term_groups.clone(),
            strata.clone(),
            weights.clone(),
            offset.clone(),
            next_beta,
            max_iter,
            eps,
            toler,
            method,
            entry_times.clone(),
            nocenter.clone(),
        )?;
        inner_iterations = inner_iterations.saturating_add(fit.iterations);
        if inner_max > 1 && fit.iterations >= inner_max {
            inner_failures.push(outer);
        }
        if outer == 1 {
            first_penalty = diagnostics.penalty;
            initial_unpenalized_loglik = fit.log_likelihood[0];
            let starting_penalty = initial_beta.as_ref().map_or(0.0, |beta| {
                0.5 * beta
                    .iter()
                    .zip(&diagnostics.penalty_diagonal)
                    .map(|(&value, &penalty)| penalty * value * value)
                    .sum::<f64>()
            });
            initial_loglik = initial_unpenalized_loglik - starting_penalty;
        }
        for (index, controller) in controllers.iter_mut().enumerate() {
            if let Some(controller) = controller {
                controller.update(applied_theta[index], diagnostics.term_df[index]);
            }
        }
        let proposed_theta: Vec<f64> = controllers
            .iter()
            .enumerate()
            .map(|(index, controller)| {
                controller
                    .as_ref()
                    .map_or(applied_theta[index], |value| value.proposed_theta)
            })
            .collect();
        let done: Vec<bool> = controllers
            .iter()
            .map(|controller| controller.as_ref().is_none_or(|value| value.done))
            .collect();
        if done.iter().all(|value| *value) || outer == outer_max {
            fit.log_likelihood[0] = initial_unpenalized_loglik;
            let selection = CoxRidgeSelection {
                fitted_theta: applied_theta,
                proposed_theta,
                done,
                histories: controllers
                    .iter()
                    .map(|controller| {
                        controller.as_ref().map_or_else(Vec::new, |value| {
                            value.history.iter().map(|row| row.to_vec()).collect()
                        })
                    })
                    .collect(),
                halves: controllers
                    .iter()
                    .map(|controller| controller.as_ref().map_or(0, |value| value.half))
                    .collect(),
                outer_iterations: outer,
                inner_iterations,
                inner_failures,
                penalty: vec![first_penalty, diagnostics.penalty],
                initial_loglik,
            };
            return Ok((fit, diagnostics, selection));
        }

        fitted_thetas.push(applied_theta.clone());
        fitted_betas.push(fit.coefficients.first().cloned().unwrap_or_default());
        for (index, proposal) in proposed_theta.into_iter().enumerate() {
            // Full-df solutions can propose NaN even when done. Keep that
            // proposal in metadata, but never feed nonfinite curvature into
            // the next joint fit while other terms are still being selected.
            if proposal.is_finite() && proposal >= 0.0 {
                applied_theta[index] = proposal;
            }
        }
        next_beta = Some(fitted_betas[closest_theta(&fitted_thetas, &applied_theta)].clone());
    }
    unreachable!("positive outer iteration limit always produces a fit")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> (Vec<f64>, Vec<i32>, Vec<Vec<f64>>) {
        (
            vec![1., 2., 2., 3., 4., 4., 5., 6., 7., 8.],
            vec![1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            vec![0.2, 0.5, 0.7, 0.1, 0.4, 0.8, 0.3, 0.9, 0.6, 1.2]
                .into_iter()
                .map(|value| vec![value])
                .collect(),
        )
    }

    fn selected(
        target: f64,
        outer: Option<usize>,
    ) -> (CoxPHFit, CoxPenaltyDiagnostics, CoxRidgeSelection) {
        let (time, status, rows) = data();
        coxph_ridge_fit(
            time,
            status,
            rows,
            vec![0.115_666_666_666_666_65],
            vec![vec![0]],
            vec![None],
            vec![Some(target)],
            vec![0.1],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("efron"),
            None,
            None,
            outer,
        )
        .unwrap()
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 2e-7,
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn default_df_selection_matches_r_history_and_fitted_theta() {
        let (fit, diagnostics, selection) = selected(0.5, None);
        close(fit.coefficients[0][0], -1.141_821_064_216_129);
        close(diagnostics.term_df[0], 0.529_127_040_632_734_6);
        close(selection.fitted_theta[0], 3.916_475_343_464_142);
        close(selection.proposed_theta[0], 4.380_724_630_302_534);
        assert_eq!(selection.outer_iterations, 3);
        assert_eq!(selection.inner_iterations, 10);
        assert_eq!(selection.done, vec![true]);
        assert_eq!(selection.histories[0].len(), 4);
        for (actual, expected) in selection.histories[0].iter().zip([
            [0.0, 1.0],
            [1.0, 0.795_907_297_380_806_9],
            [2.449_867_112_264_794, 0.632_516_106_479_660_6],
            [3.916_475_343_464_142, 0.529_127_040_632_734_6],
        ]) {
            close(actual[0], expected[0]);
            close(actual[1], expected[1]);
        }
        close(selection.penalty[0], 0.184_831_196_696_176_4);
        close(selection.penalty[1], 0.295_304_266_963_401_1);
        close(selection.initial_loglik, -10.855_917_331_026_16);
        close(fit.log_likelihood[0], -10.855_917_331_026_16);
        close(fit.log_likelihood[1], -9.915_334_056_208_56);
    }

    #[test]
    fn outer_limit_returns_actual_df_and_unfinished_controller() {
        let (_, diagnostics, selection) = selected(0.5, Some(1));
        close(selection.fitted_theta[0], 1.0);
        close(diagnostics.term_df[0], 0.795_907_297_380_806_9);
        assert_eq!(selection.done, vec![false]);
        assert_eq!(selection.outer_iterations, 1);
        assert_eq!(selection.histories[0].len(), 2);
    }

    #[test]
    fn full_df_keeps_zero_applied_theta_and_nan_unused_proposal() {
        let (fit, diagnostics, selection) = selected(1.0, None);
        close(fit.coefficients[0][0], -2.270_855_866_832_257);
        close(diagnostics.term_df[0], 1.0);
        close(selection.fitted_theta[0], 0.0);
        assert!(selection.proposed_theta[0].is_nan());
        assert_eq!(selection.done, vec![true]);
        assert_eq!(selection.outer_iterations, 2);
    }

    #[test]
    fn zero_df_uses_absolute_tolerance_instead_of_forcing_zero_coefficients() {
        let (fit, diagnostics, selection) = selected(0.0, None);
        close(fit.coefficients[0][0], -0.134_345_280_961_494_1);
        close(diagnostics.term_df[0], 0.067_401_637_428_618_07);
        close(selection.fitted_theta[0], 68.613_813_966_568_12);
        assert_eq!(selection.done, vec![true]);
    }

    #[test]
    fn warm_start_uses_raw_theta_distance_and_earliest_ties() {
        let previous = vec![vec![1.0, 3.0], vec![10.0, 30.0], vec![2.0, 4.0]];
        assert_eq!(closest_theta(&previous, &[1.5, 3.5]), 0);
        assert_eq!(closest_theta(&previous, &[9.0, 29.0]), 1);
        assert_eq!(closest_theta(&previous, &[2.1, 4.1]), 2);
    }

    #[test]
    fn initialized_selection_preserves_first_likelihood_and_zero_iteration_behavior() {
        for inner_max in [0, 20] {
            let (time, status, rows) = data();
            let z = [0.3, 1.2, 0.4, 0.8, 1.1, 0.6, 0.2, 0.7, 1.4, 0.9];
            let rows = rows.iter().zip(z).map(|(row, z)| vec![z, row[0]]).collect();
            let (fit, diagnostics, selection) = coxph_ridge_fit(
                time,
                status,
                rows,
                vec![0.0, 0.115_666_666_666_666_65],
                vec![vec![0], vec![1]],
                vec![Some(0.0), None],
                vec![None, Some(0.5)],
                vec![0.1, 0.001],
                None,
                Some(vec![1., 2., 1., 0.5, 1.5, 1., 2., 1., 1., 1.]),
                None,
                Some(vec![0.4, -0.2]),
                Some(inner_max),
                None,
                None,
                Some("efron"),
                None,
                None,
                None,
            )
            .unwrap();
            close(selection.initial_loglik, -16.929_716_971_787_3);
            close(fit.log_likelihood[0], -16.927_403_638_453_97);
            assert_eq!(selection.outer_iterations, 5);
            assert_eq!(selection.done, vec![true, true]);
            assert!(selection.histories[0].is_empty());
            assert!(selection.inner_failures.is_empty());
            if inner_max == 0 {
                assert_eq!(fit.coefficients[0], vec![0.4, -0.2]);
                assert_eq!(selection.inner_iterations, 0);
                close(selection.fitted_theta[1], 6.231_566_988_025_204);
                close(diagnostics.term_df[0], 0.992_483_151_895_664_8);
                close(diagnostics.term_df[1], 0.500_139_846_480_098);
                close(selection.penalty[0], 0.002_313_333_333_333_333);
                close(selection.penalty[1], 0.014_415_691_632_298_31);
                close(fit.log_likelihood[1], -16.927_403_638_453_97);
            } else {
                assert_eq!(selection.inner_iterations, 15);
                close(selection.fitted_theta[1], 4.941_830_553_174_727);
                close(fit.coefficients[0][0], -0.324_393_123_196_665);
                close(fit.coefficients[0][1], -1.071_022_774_863_569);
                close(diagnostics.term_df[0], 0.973_789_408_494_018_8);
                close(diagnostics.term_df[1], 0.500_155_069_714_574_7);
                close(selection.penalty[0], 0.196_286_737_174_522_5);
                close(selection.penalty[1], 0.327_841_166_680_114_6);
                close(fit.log_likelihood[1], -15.562_841_407_507_22);
            }
        }
    }

    #[test]
    fn full_df_group_keeps_finite_fit_while_other_group_continues() {
        let (time, status, rows) = data();
        let z = [0.3, 1.2, 0.4, 0.8, 1.1, 0.6, 0.2, 0.7, 1.4, 0.9];
        let rows = rows.iter().zip(z).map(|(row, z)| vec![row[0], z]).collect();
        let (fit, diagnostics, selection) = coxph_ridge_fit(
            time,
            status,
            rows,
            vec![0.115_666_666_666_666_65, 0.16],
            vec![vec![0], vec![1]],
            vec![None, None],
            vec![Some(1.0), Some(0.5)],
            vec![0.001; 2],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("efron"),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(selection.outer_iterations > 2);
        assert!(selection.fitted_theta.iter().all(|value| value.is_finite()));
        assert!(fit.coefficients[0].iter().all(|value| value.is_finite()));
        assert!(diagnostics.term_df.iter().all(|value| value.is_finite()));
        close(selection.fitted_theta[0], 0.0);
    }

    #[test]
    fn rounded_duplicate_df_uses_finite_bracket_without_perturbing_history() {
        let mut controller = DfController::new(1, 0.6, 0.1);
        for [theta, df] in [
            [1.0, 0.002_068_457_877_769_405_4],
            [0.601_243_647_158_423_7, 0.003_435_519_184_101_211],
            [0.300_621_823_579_211_86, 0.006_847_163_894_840_784],
            [0.150_310_911_789_605_93, 0.013_599_673_009_645_217],
            [2.240_219_127_018_992e-41, 1.0],
            [0.075_155_455_894_802_96, 0.026_827_394_307_682_103],
            [0.037_577_727_947_401_48, 0.052_219_016_279_364_41],
        ] {
            controller.update(theta, df);
        }
        assert!(!controller.done);
        assert_eq!(controller.proposed_theta, 0.018_788_863_973_700_74);
        assert_eq!(controller.history[0][1], 1.0);
        assert_eq!(controller.history[5][1], 1.0);
    }

    #[test]
    fn undefined_proposal_without_bracket_keeps_actual_theta_and_unfinished_state() {
        let mut controller = DfController::new(1, 0.5, 0.1);
        controller.update(1.0, 1.0);
        assert_eq!(controller.proposed_theta, 1.0);
        assert!(!controller.done);
    }
}
