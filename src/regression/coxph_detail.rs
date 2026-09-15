//! `coxph.detail()`: the per-death-time pieces of a Cox fit, a port of R
//! survival's `R/coxph.detail.R` and `src/coxdetail.c`.
//!
//! For every unique death time (within stratum, ascending) it returns the
//! number of deaths and of subjects at risk, the hazard increment and its
//! variance, the weighted risk-set mean of each covariate, the score vector
//! contribution and the information-matrix contribution, so that
//! `colSums(score)` and `apply(imat, 1:2, sum)` are the fit's score and
//! information at the fitted coefficients.  One backward sweep per stratum
//! ([`StratumSweep`]) replaces `coxdetail.c`'s `O(deaths x n)` rescan.

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
    let risk: Vec<f64> = fit.linear_predictors.iter().map(|lp| lp.exp()).collect();
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
            x: fit.x.view(),
            weights: &fit.weights,
            risk: &risk,
            rows: &fit.sorted.order[start..end],
            second_moments: true,
        };
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
                        .map(|&row| fit.weights[row] * fit.x[(row, i)])
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
                    mean[i] += xbar[i] / d_f;
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
            hazard.push(haz);
            varhaz.push(var);
            wtrisk.push(death.risk_set.denom);
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
