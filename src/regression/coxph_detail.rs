//! `coxph.detail()`: the per-death-time pieces of a Cox fit, a port of R
//! survival's `R/coxph.detail.R` and `src/coxdetail.c`.
//!
//! For every unique death time (within stratum, ascending) it returns the
//! number of deaths and of subjects at risk, the hazard increment and its
//! variance, the weighted risk-set mean of each covariate, the score vector
//! contribution and the information-matrix contribution, so that
//! `colSums(score)` and `apply(imat, 1:2, sum)` are the fit's score and
//! information at the fitted coefficients.  One backward sweep per stratum
//! ([`StratumSweep`]) replaces `coxdetail.c`'s `O(deaths x n)` rescan; its
//! running risk set restarts from zero whenever it empties.

use crate::core::risk_sweep::StratumSweep;
use crate::error::{SurvivalError, SurvivalResult};
use crate::regression::cox_optimizer::TieMethod;
use crate::regression::coxph::CoxPHFit;
use pyo3::prelude::*;

/// `coxph.detail(fit)`: one entry per unique death time.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct CoxphDetail {
    #[pyo3(get)]
    pub time: Vec<f64>,
    /// Number of deaths (unweighted).
    #[pyo3(get)]
    pub nevent: Vec<usize>,
    /// Number of subjects at risk (unweighted).
    #[pyo3(get)]
    pub nrisk: Vec<usize>,
    /// Hazard increment.
    #[pyo3(get)]
    pub hazard: Vec<f64>,
    /// Increment of the variance of the cumulative hazard.
    #[pyo3(get)]
    pub varhaz: Vec<f64>,
    /// Weighted risk-set size `sum w * exp(lp)` (`wtrisk`).
    #[pyo3(get)]
    pub wtrisk: Vec<f64>,
    /// Weighted number of deaths (`nevent.wt`).
    #[pyo3(get)]
    pub nevent_wt: Vec<f64>,
    /// Weighted risk-set means of the covariates (`ntime x nvar`).
    #[pyo3(get)]
    pub means: Vec<Vec<f64>>,
    /// Score-vector contribution of each death time (`ntime x nvar`).
    #[pyo3(get)]
    pub score: Vec<Vec<f64>>,
    /// Information-matrix contribution of each death time
    /// (`ntime x nvar x nvar`).
    #[pyo3(get)]
    pub imat: Vec<Vec<Vec<f64>>>,
    /// Stratum code of each death time (absent for an unstratified fit).
    #[pyo3(get)]
    pub strata: Option<Vec<i32>>,
    /// `riskmat = TRUE`: `n x ntime`, 1 when the row is at risk at the
    /// death time (rows in the fit's order).
    #[pyo3(get)]
    pub riskmat: Option<Vec<Vec<i32>>>,
}

/// Port of `coxph.detail` / `coxdetail.c`.
#[allow(clippy::needless_range_loop)]
pub fn coxph_detail(fit: &CoxPHFit, riskmat: bool) -> SurvivalResult<CoxphDetail> {
    if fit.method == TieMethod::Exact {
        return Err(SurvivalError::invalid_input(
            "Detailed output is not available for the exact method",
        ));
    }
    let nvar = fit.nvar();
    let efron = fit.method == TieMethod::Efron;
    // coxdetail.c centres the covariates at the fit's means (0 for the
    // nocenter columns) and adds them back to the reported means, so the
    // second moments do not cancel for covariates with a large mean.
    let mut x = fit.x.clone();
    for (mut column, &center) in x.columns_mut().into_iter().zip(&fit.means) {
        column -= center;
    }
    // Risk scores `exp(lp)` as in coxdetail.c, except that a stratum whose
    // largest linear predictor lies beyond +-200 (agfit4.c's recentring
    // threshold) is shifted by it so `exp` neither overflows nor underflows;
    // the outputs built on the denominator are scaled back.
    let mut risk = vec![0.0; fit.n];
    let shift: Vec<f64> = fit
        .sorted
        .bounds
        .iter()
        .map(|&(start, end)| {
            let rows = &fit.sorted.order[start..end];
            let top = rows
                .iter()
                .map(|&row| fit.linear_predictors[row])
                .fold(f64::NEG_INFINITY, f64::max);
            let shift = if top.abs() > 200.0 { top } else { 0.0 };
            for &row in rows {
                risk[row] = (fit.linear_predictors[row] - shift).exp();
            }
            shift
        })
        .collect();
    let mut time = Vec::new();
    let mut nevent = Vec::new();
    let mut nrisk = Vec::new();
    let mut hazard = Vec::new();
    let mut varhaz = Vec::new();
    let mut wtrisk = Vec::new();
    let mut nevent_wt = Vec::new();
    let mut means = Vec::new();
    let mut score = Vec::new();
    let mut imat = Vec::new();
    let mut strata = Vec::new();
    for stratum in 0..fit.sorted.nstrata() {
        let (start, end) = fit.sorted.bounds[stratum];
        let sweep = StratumSweep {
            stop: &fit.time,
            entry: fit.entry.as_deref(),
            status: &fit.status,
            x: x.view(),
            weights: &fit.weights,
            risk: &risk,
            rows: &fit.sorted.order[start..end],
            second_moments: true,
        };
        let risk_scale = shift[stratum].exp();
        let first = time.len();
        sweep.for_each_death_time(|death| {
            let d = death.ndead();
            let d_f = d as f64;
            let meanwt = death.tied.weight / d_f;
            let mut haz = 0.0;
            let mut var = 0.0;
            let mut mean = vec![0.0; nvar];
            let mut u: Vec<f64> = (0..nvar)
                .map(|i| {
                    death
                        .deaths
                        .iter()
                        .map(|&row| fit.weights[row] * x[(row, i)])
                        .sum()
                })
                .collect();
            let mut info = vec![vec![0.0; nvar]; nvar];
            for j in 0..d {
                // Breslow keeps the full risk set for every tied death.
                let step = if efron { j } else { 0 };
                let d2 = death.efron_denom(step);
                haz += meanwt / d2;
                var += meanwt * meanwt / (d2 * d2);
                let xbar: Vec<f64> = (0..nvar).map(|i| death.efron_a(step, i) / d2).collect();
                for i in 0..nvar {
                    mean[i] += (fit.means[i] + xbar[i]) / d_f;
                    u[i] -= meanwt * xbar[i];
                    for k in 0..=i {
                        let value = meanwt
                            * (death.efron_cmat(step, i, k) - xbar[i] * death.efron_a(step, k))
                            / d2;
                        info[i][k] += value;
                        if k < i {
                            info[k][i] += value;
                        }
                    }
                }
            }
            time.push(death.time);
            nevent.push(d);
            nrisk.push(death.risk_set.count);
            hazard.push(haz / risk_scale);
            varhaz.push(var / risk_scale / risk_scale);
            wtrisk.push(death.risk_set.denom * risk_scale);
            nevent_wt.push(death.tied.weight);
            means.push(mean);
            score.push(u);
            imat.push(info);
            strata.push(fit.sorted.codes[stratum]);
        });
        // The sweep runs backwards; R lists death times ascending.
        time[first..].reverse();
        nevent[first..].reverse();
        nrisk[first..].reverse();
        hazard[first..].reverse();
        varhaz[first..].reverse();
        wtrisk[first..].reverse();
        nevent_wt[first..].reverse();
        means[first..].reverse();
        score[first..].reverse();
        imat[first..].reverse();
    }
    let riskmat = riskmat.then(|| {
        let mut matrix = vec![vec![0i32; time.len()]; fit.n];
        for (g, &t) in time.iter().enumerate() {
            let code = strata[g];
            for row in 0..fit.n {
                let same_stratum = fit.strata.as_ref().is_none_or(|s| s[row] == code);
                let entered = fit.entry.as_ref().is_none_or(|entry| entry[row] < t);
                if same_stratum && entered && fit.time[row] >= t {
                    matrix[row][g] = 1;
                }
            }
        }
        matrix
    });
    Ok(CoxphDetail {
        time,
        nevent,
        nrisk,
        hazard,
        varhaz,
        wtrisk,
        nevent_wt,
        means,
        score,
        imat,
        strata: fit.strata.as_ref().map(|_| strata),
        riskmat,
    })
}

/// `coxph.detail(fit, riskmat)` on a fitted model.
#[pyfunction(name = "coxph_detail")]
#[pyo3(signature = (fit, riskmat = false))]
pub fn coxph_detail_py(fit: &CoxPHFit, riskmat: bool) -> PyResult<CoxphDetail> {
    Ok(coxph_detail(fit, riskmat)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::coxph::{CoxphData, CoxphOptions};
    use ndarray::Array2;

    fn fitted(method: TieMethod) -> CoxPHFit {
        let time = vec![1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0];
        let status = vec![1, 1, 1, 0, 1, 1, 0, 1];
        let x = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.2, 0.4, 0.16, 0.8, 0.62, 0.2, -0.07, 1.0, 0.95, 1.4, 0.61, 0.6, 0.49, 1.2,
                0.68,
            ],
        )
        .unwrap();
        let weights = vec![1.0, 2.0, 1.0, 1.5, 1.0, 0.5, 1.0, 1.0];
        let data = CoxphData::try_new(time, None, status, x, Some(weights), None, None).unwrap();
        let options = CoxphOptions {
            method,
            ..CoxphOptions::default()
        };
        CoxPHFit::fit(data, options).unwrap()
    }

    #[test]
    fn detail_sums_reproduce_the_score_and_information() {
        for method in [TieMethod::Breslow, TieMethod::Efron] {
            let fit = fitted(method);
            let detail = coxph_detail(&fit, true).unwrap();
            assert_eq!(detail.time, vec![1.0, 2.0, 4.0, 6.0]);
            assert_eq!(detail.nevent, vec![1, 2, 2, 1]);
            assert_eq!(detail.nrisk, vec![8, 7, 4, 1]);
            // The score contributions sum to the fitted score (zero at the
            // solution) and the information contributions to the inverse
            // of the variance.
            for i in 0..2 {
                let total: f64 = detail.score.iter().map(|row| row[i]).sum();
                assert!((total - fit.first[i]).abs() < 1e-8, "{method:?}: {total}");
            }
            let mut info = [[0.0; 2]; 2];
            for block in &detail.imat {
                for i in 0..2 {
                    for k in 0..2 {
                        info[i][k] += block[i][k];
                    }
                }
            }
            let det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
            let inverse = [
                [info[1][1] / det, -info[0][1] / det],
                [-info[1][0] / det, info[0][0] / det],
            ];
            for (i, row) in inverse.iter().enumerate() {
                for (k, value) in row.iter().enumerate() {
                    assert!((value - fit.var[(i, k)]).abs() < 1e-8);
                }
            }
            // Hazard increments integrate to the Breslow/Efron baseline.
            let basehaz = fit.basehaz(true).unwrap();
            let mut cumulative = 0.0;
            for (g, h) in detail.hazard.iter().enumerate() {
                cumulative += h;
                let position = basehaz
                    .time
                    .iter()
                    .position(|&t| t == detail.time[g])
                    .unwrap();
                assert!((basehaz.hazard[position] - cumulative).abs() < 1e-12);
            }
            let riskmat = detail.riskmat.unwrap();
            assert_eq!(riskmat[3], vec![1, 1, 0, 0]);
            assert_eq!(detail.wtrisk.len(), 4);
            assert_eq!(detail.nevent_wt[1], 3.0);
        }
    }

    /// 11 (start, stop] rows whose risk scores span many orders of
    /// magnitude, at R's coefficients (`iter.max = 0`), with `shift` added
    /// to the first covariate.
    fn counting_fit_at_r_coefficients(shift: f64) -> CoxPHFit {
        let time = vec![15.0, 4.0, 25.0, 8.0, 18.0, 6.0, 9.0, 14.0, 11.0, 64.0, 19.0];
        let entry = vec![9.0, 2.0, 5.0, 1.0, 14.0, 3.0, 7.0, 6.0, 3.0, 39.0, 2.0];
        let status = vec![1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1];
        let x1 = [
            -3.397587783734524,
            -4.001456275604981,
            1.2485501179741572,
            -1.922243349085923,
            -0.6191833991904704,
            0.9541844726560894,
            0.9722226187833938,
            -2.083451024212451,
            -0.557594058080056,
            -6.582305367737362,
            0.9175209607015593,
        ];
        let x2 = [
            -1.2399609396099633,
            -0.9612195837228918,
            -0.36802848351666784,
            -0.29505277170923994,
            -0.432759419947449,
            3.636570321763567,
            -1.5241526403027048,
            1.3906982107759198,
            1.1836246992905952,
            2.3943458650099303,
            -1.2770581196683075,
        ];
        let x = Array2::from_shape_fn((11, 2), |(i, j)| if j == 0 { x1[i] + shift } else { x2[i] });
        let data = CoxphData::try_new(time, Some(entry), status, x, None, None, None).unwrap();
        let options = CoxphOptions {
            init: Some(vec![-2.729510672, 2.391046607]),
            iter_max: 0,
            ..CoxphOptions::default()
        };
        CoxPHFit::fit(data, options).unwrap()
    }

    fn assert_close(actual: f64, expected: f64, rtol: f64) {
        assert!(
            (actual - expected).abs() <= rtol * expected.abs(),
            "{actual} vs {expected}"
        );
    }

    #[test]
    fn detail_of_counting_data_keeps_small_risk_sets_exact() {
        // R 3.8-12: coxph.detail(coxph(Surv(st, t, s) ~ x1 + x2, init =
        // c(-2.729510672, 2.391046607), iter.max = 0, timefix = FALSE)).
        let detail = coxph_detail(&counting_fit_at_r_coefficients(0.0), false).unwrap();
        assert_eq!(
            detail.time,
            vec![4.0, 6.0, 14.0, 15.0, 18.0, 19.0, 25.0, 64.0]
        );
        let hazard = [
            0.0117538131278798,
            0.118360346872988,
            0.00829441948728503,
            0.131628388009271,
            37.3485109236605,
            4126.13694309023,
            5284.90450618274,
            3.73036555845107e-09,
        ];
        let mean_x1 = [
            -3.57205179183823,
            0.322744313035974,
            -2.16596152280981,
            -3.38773850910569,
            -0.60293423955203,
            1.17596870246578,
            1.24855011797416,
            -6.58230536773736,
        ];
        let wtrisk = [
            85.0787730858182,
            8.44877550986816,
            120.562988348124,
            7.59714538120428,
            0.0267748291770983,
            0.000242357443243525,
            0.000189218177704084,
            268070242.535486,
        ];
        for g in 0..8 {
            assert_close(detail.hazard[g], hazard[g], 1e-12);
            assert_close(detail.means[g][0], mean_x1[g], 1e-12);
            assert_close(detail.wtrisk[g], wtrisk[g], 1e-12);
        }
        assert_close(detail.varhaz[5], 17025006.073134, 1e-12);
    }

    #[test]
    fn detail_centres_the_covariates_at_the_fit_means() {
        // R 3.8-12 on the same data with x1 + 1e6 (coxdetail.c subtracts
        // the fit's means before summing).
        let detail = coxph_detail(&counting_fit_at_r_coefficients(1e6), false).unwrap();
        let imat = &detail.imat[4];
        assert_close(imat[0][0], 0.029075474044559, 1e-8);
        assert_close(imat[1][0], -0.00170079031707362, 1e-8);
        assert_close(imat[0][1], -0.00170079031707362, 1e-8);
        assert_close(imat[1][1], 0.00144288193816428, 1e-8);
        assert_close(detail.means[4][0], 999999.397065761, 1e-14);
        let total: f64 = detail.imat.iter().flatten().flatten().sum();
        assert_close(total, 14.3540251466645, 1e-8);
    }

    #[test]
    fn exact_fits_have_no_detail() {
        let fit = {
            let data = CoxphData::try_new(
                vec![1.0, 2.0, 3.0],
                None,
                vec![1, 1, 0],
                Array2::from_shape_vec((3, 1), vec![0.1, 0.5, 0.9]).unwrap(),
                None,
                None,
                None,
            )
            .unwrap();
            CoxPHFit::fit(
                data,
                CoxphOptions {
                    method: TieMethod::Exact,
                    ..CoxphOptions::default()
                },
            )
            .unwrap()
        };
        assert!(coxph_detail(&fit, false).is_err());
    }
}
