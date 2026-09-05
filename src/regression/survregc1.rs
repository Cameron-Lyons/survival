use crate::internal::aft::AftDistribution;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

type SurvregDerivatives = [f64; 6];

const SURVREG_PARALLEL_THRESHOLD: usize = 10_000;

#[derive(Clone, Copy)]
pub(crate) enum SurvivalDist {
    ExtremeValue,
    Logistic,
    Gaussian,
    Weibull,
    LogNormal,
    LogLogistic,
    StudentT(f64),
}
impl SurvivalDist {
    fn family(self) -> AftDistribution {
        match self {
            Self::ExtremeValue | Self::Weibull => AftDistribution::Extreme,
            Self::Logistic | Self::LogLogistic => AftDistribution::Logistic,
            Self::Gaussian | Self::LogNormal => AftDistribution::Gaussian,
            Self::StudentT(df) => AftDistribution::from_key("t", Some(df)),
        }
    }
}

pub(crate) struct SurvivalLikelihood {
    pub loglik: f64,
    pub u: Array1<f64>,
    pub imat: Array2<f64>,
    pub jj: Array2<f64>,
    pub fdiag: Array1<f64>,
    pub jdiag: Array1<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct SurvregDimensions {
    pub nvar: usize,
    pub nstrat: usize,
    pub nf: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Derivatives {
    pub dg: f64,
    pub ddg: f64,
    pub dsig: f64,
    pub ddsig: f64,
    pub dsg: f64,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn survregc1(
    n: usize,
    nvar: usize,
    nstrat: usize,
    whichcase: bool,
    beta: &ArrayView1<f64>,
    dist: SurvivalDist,
    strat: &ArrayView1<i32>,
    offset: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    interval_widths: Option<&ArrayView1<f64>>,
    status: &ArrayView1<i32>,
    wt: &ArrayView1<f64>,
    covar: &ArrayView2<f64>,
    nf: usize,
    frail: &ArrayView1<i32>,
) -> Result<SurvivalLikelihood, Box<dyn std::error::Error>> {
    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nf;

    if n < SURVREG_PARALLEL_THRESHOLD || whichcase {
        return survregc1_sequential(
            n,
            nvar,
            nstrat,
            whichcase,
            beta,
            dist,
            strat,
            offset,
            time1,
            interval_widths,
            status,
            wt,
            covar,
            nf,
            frail,
        );
    }

    let interval_widths_slice = match interval_widths {
        Some(t) => Some(
            t.as_slice()
                .ok_or_else(|| "interval_widths array must be contiguous in memory".to_string())?,
        ),
        None => None,
    };

    let family = dist.family();
    let scales: Vec<f64> = (0..nstrat.max(1))
        .map(|stratum| beta[nvar + nf + stratum].exp())
        .collect();
    type PersonResult = (usize, usize, usize, f64, SurvregDerivatives);
    let partial_results: Result<Vec<PersonResult>, Box<dyn std::error::Error + Send + Sync>> = (0
        ..n)
        .into_par_iter()
        .map(|person| {
            if !matches!(status[person], 0..=3) {
                return Err("Invalid status value".into());
            }
            if wt[person] == 0.0 {
                return Ok((person, 0, 0, 0.0, [0.0; 6]));
            }
            let strata_idx = if nstrat > 1 {
                (strat[person] - 1) as usize
            } else {
                0
            };
            let sigma = scales[strata_idx];
            let mut eta = offset[person];
            for i in 0..nvar {
                eta += beta[i + nf] * covar[[i, person]];
            }
            let fgrp = if nf > 0 {
                (frail[person] - 1) as usize
            } else {
                0
            };
            if nf > 0 {
                eta += beta[fgrp];
            }
            let z = (time1[person] - eta) / sigma;
            let row = likelihood_row(
                family,
                z,
                sigma,
                status[person],
                interval_widths_slice.map(|widths| widths[person]),
            )?;
            Ok((person, fgrp, strata_idx, wt[person], row))
        })
        .collect();

    let partial_results =
        partial_results.map_err(|e| Box::<dyn std::error::Error>::from(e.to_string()))?;

    let mut result = SurvivalLikelihood {
        loglik: 0.0,
        u: Array1::zeros(nvar3),
        imat: Array2::zeros((nvar2, nvar3)),
        jj: Array2::zeros((nvar2, nvar3)),
        fdiag: Array1::zeros(nf),
        jdiag: Array1::zeros(nf),
    };

    let dims = SurvregDimensions { nvar, nstrat, nf };
    for (person, fgrp, strata_idx, w, [g, dg, ddg, dsig, ddsig, dsg]) in partial_results {
        if w == 0.0 {
            continue;
        }
        result.loglik += g * w;
        let derivs = Derivatives {
            dg,
            ddg,
            dsig,
            ddsig,
            dsg,
        };
        update_derivatives(
            &mut result,
            person,
            fgrp,
            strata_idx,
            dims,
            covar,
            w,
            derivs,
        );
    }

    symmetrize_matrix(&mut result.imat);
    symmetrize_matrix(&mut result.jj);

    Ok(result)
}

fn symmetrize_matrix(mat: &mut Array2<f64>) {
    let n = mat.nrows().min(mat.ncols());
    for i in 0..n {
        for j in 0..i {
            let val = mat[[i, j]];
            mat[[j, i]] = val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn survregc1_sequential(
    n: usize,
    nvar: usize,
    nstrat: usize,
    whichcase: bool,
    beta: &ArrayView1<f64>,
    dist: SurvivalDist,
    strat: &ArrayView1<i32>,
    offset: &ArrayView1<f64>,
    time1: &ArrayView1<f64>,
    interval_widths: Option<&ArrayView1<f64>>,
    status: &ArrayView1<i32>,
    wt: &ArrayView1<f64>,
    covar: &ArrayView2<f64>,
    nf: usize,
    frail: &ArrayView1<i32>,
) -> Result<SurvivalLikelihood, Box<dyn std::error::Error>> {
    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nf;
    let dims = SurvregDimensions { nvar, nstrat, nf };
    let mut result = SurvivalLikelihood {
        loglik: 0.0,
        u: Array1::zeros(nvar3),
        imat: Array2::zeros((nvar2, nvar3)),
        jj: Array2::zeros((nvar2, nvar3)),
        fdiag: Array1::zeros(nf),
        jdiag: Array1::zeros(nf),
    };
    let family = dist.family();
    let scales: Vec<f64> = (0..nstrat.max(1))
        .map(|stratum| beta[nvar + nf + stratum].exp())
        .collect();
    for person in 0..n {
        if !matches!(status[person], 0..=3) {
            return Err("Invalid status value".into());
        }
        if wt[person] == 0.0 {
            continue;
        }
        let strata = if nstrat > 1 {
            (strat[person] - 1) as usize
        } else {
            0
        };
        let sigma = scales[strata];
        let mut eta = offset[person];
        for i in 0..nvar {
            eta += beta[i + nf] * covar[[i, person]];
        }
        let fgrp = if nf > 0 {
            (frail[person] - 1) as usize
        } else {
            0
        };
        if nf > 0 {
            eta += beta[fgrp];
        }
        let z = (time1[person] - eta) / sigma;
        let [g, dg, ddg, dsig, ddsig, dsg] = likelihood_row(
            family,
            z,
            sigma,
            status[person],
            interval_widths.map(|widths| widths[person]),
        )?;
        result.loglik += g * wt[person];
        if whichcase {
            continue;
        }
        let w = wt[person];
        let derivs = Derivatives {
            dg,
            ddg,
            dsig,
            ddsig,
            dsg,
        };
        update_derivatives(&mut result, person, fgrp, strata, dims, covar, w, derivs);
    }

    symmetrize_matrix(&mut result.imat);
    symmetrize_matrix(&mut result.jj);

    Ok(result)
}
/// Derivatives with respect to location and log(scale). Interval widths are
/// transformed once from the original response endpoints before optimization.
#[inline]
fn likelihood_row(
    family: AftDistribution,
    z: f64,
    sigma: f64,
    status: i32,
    interval_width: Option<f64>,
) -> Result<SurvregDerivatives, &'static str> {
    match status {
        0..=2 => Ok(family.single(z, sigma, status)),
        3 => {
            let width =
                interval_width.ok_or("Missing interval widths for interval censored data")?;
            Ok(family.interval_from_response_width(z, width, sigma))
        }
        _ => Err("Invalid status value"),
    }
}

#[allow(clippy::too_many_arguments)]
fn update_derivatives(
    res: &mut SurvivalLikelihood,
    person: usize,
    fgrp: usize,
    strata: usize,
    dims: SurvregDimensions,
    covar: &ArrayView2<f64>,
    w: f64,
    derivs: Derivatives,
) {
    let Derivatives {
        dg,
        ddg,
        dsig,
        ddsig,
        dsg,
    } = derivs;
    let SurvregDimensions {
        nvar, nstrat, nf, ..
    } = dims;

    if nf > 0 {
        res.u[fgrp] += dg * w;
        res.fdiag[fgrp] -= ddg * w;
        res.jdiag[fgrp] += dg.powi(2) * w;
    }
    for i in 0..nvar {
        let cov_i = covar[[i, person]];
        let temp = dg * cov_i * w;
        res.u[i + nf] += temp;
        for j in 0..=i {
            let cov_j = covar[[j, person]];
            res.imat[[i, j + nf]] -= cov_i * cov_j * ddg * w;
            res.jj[[i, j + nf]] += temp * cov_j * dg;
        }
        if nf > 0 {
            res.imat[[i, fgrp]] -= cov_i * ddg * w;
            res.jj[[i, fgrp]] += temp * dg;
        }
    }
    if nstrat > 0 {
        let k = strata + nvar;
        res.u[k + nf] += dsig * w;
        for i in 0..nvar {
            let cov_i = covar[[i, person]];
            res.imat[[k, i + nf]] -= dsg * cov_i * w;
            res.jj[[k, i + nf]] += dsig * cov_i * dg * w;
        }
        res.imat[[k, k + nf]] -= ddsig * w;
        res.jj[[k, k + nf]] += dsig.powi(2) * w;
        if nf > 0 {
            res.imat[[k, fgrp]] -= dsg * w;
            res.jj[[k, fgrp]] += dsig * dg * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_survival_dist_variants() {
        let variants = [
            SurvivalDist::ExtremeValue,
            SurvivalDist::Logistic,
            SurvivalDist::Gaussian,
            SurvivalDist::Weibull,
            SurvivalDist::LogNormal,
            SurvivalDist::LogLogistic,
            SurvivalDist::StudentT(4.0),
        ];
        assert_eq!(variants.len(), 7);
    }

    #[test]
    fn test_symmetrize_matrix() {
        let mut mat = Array2::zeros((3, 3));
        mat[[0, 0]] = 1.0;
        mat[[1, 0]] = 2.0;
        mat[[1, 1]] = 3.0;
        mat[[2, 0]] = 4.0;
        mat[[2, 1]] = 5.0;
        mat[[2, 2]] = 6.0;

        symmetrize_matrix(&mut mat);

        assert!((mat[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((mat[[0, 2]] - 4.0).abs() < 1e-10);
        assert!((mat[[1, 2]] - 5.0).abs() < 1e-10);
        assert!((mat[[1, 0]] - mat[[0, 1]]).abs() < 1e-10);
        assert!((mat[[2, 0]] - mat[[0, 2]]).abs() < 1e-10);
        assert!((mat[[2, 1]] - mat[[1, 2]]).abs() < 1e-10);
    }

    #[test]
    fn test_symmetrize_matrix_empty() {
        let mut mat = Array2::zeros((0, 0));
        symmetrize_matrix(&mut mat);
        assert_eq!(mat.nrows(), 0);
    }

    #[test]
    fn test_symmetrize_matrix_single() {
        let mut mat = Array2::from_elem((1, 1), 5.0);
        symmetrize_matrix(&mut mat);
        assert!((mat[[0, 0]] - 5.0).abs() < 1e-10);
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn contribution(
        status: i32,
        y: f64,
        upper: Option<f64>,
        eta: f64,
        log_sigma: f64,
        dist: SurvivalDist,
    ) -> SurvregDerivatives {
        let sigma = log_sigma.exp();
        likelihood_row(
            dist.family(),
            (y - eta) / sigma,
            sigma,
            status,
            upper.map(|upper| upper - y),
        )
        .unwrap()
    }

    fn assert_derivatives_match_finite_difference(
        status: i32,
        y: f64,
        upper: Option<f64>,
        eta: f64,
        log_sigma: f64,
        dist: SurvivalDist,
    ) {
        let h = 1e-5;
        let h2 = 1e-4;
        let [g, dg, ddg, dsig, ddsig, dsg] = contribution(status, y, upper, eta, log_sigma, dist);

        let g_eta_plus = contribution(status, y, upper, eta + h, log_sigma, dist)[0];
        let g_eta_minus = contribution(status, y, upper, eta - h, log_sigma, dist)[0];
        let eta_score = (g_eta_plus - g_eta_minus) / (2.0 * h);
        let eta_hessian = (g_eta_plus - 2.0 * g + g_eta_minus) / (h * h);

        let g_sigma_plus = contribution(status, y, upper, eta, log_sigma + h, dist)[0];
        let g_sigma_minus = contribution(status, y, upper, eta, log_sigma - h, dist)[0];
        let sigma_score = (g_sigma_plus - g_sigma_minus) / (2.0 * h);
        let sigma_hessian = (g_sigma_plus - 2.0 * g + g_sigma_minus) / (h * h);

        let cross = (contribution(status, y, upper, eta + h2, log_sigma + h2, dist)[0]
            - contribution(status, y, upper, eta + h2, log_sigma - h2, dist)[0]
            - contribution(status, y, upper, eta - h2, log_sigma + h2, dist)[0]
            + contribution(status, y, upper, eta - h2, log_sigma - h2, dist)[0])
            / (4.0 * h2 * h2);

        assert_close(dg, eta_score, 1e-5);
        assert_close(ddg, eta_hessian, 1e-4);
        assert_close(dsig, sigma_score, 1e-5);
        assert_close(ddsig, sigma_hessian, 1e-4);
        assert_close(dsg, cross, 1e-4);
    }

    #[test]
    fn likelihood_density_at_the_location_matches_distribution_references() {
        for (dist, log_density, curvature) in [
            (SurvivalDist::Weibull, -1.0, -1.0),
            (
                SurvivalDist::LogNormal,
                -0.5 * std::f64::consts::TAU.ln(),
                -1.0,
            ),
            (SurvivalDist::LogLogistic, -4.0_f64.ln(), -0.5),
            (SurvivalDist::StudentT(4.0), 0.375_f64.ln(), -1.25),
        ] {
            let expected = [log_density, 0.0, curvature, -1.0, 0.0, 0.0];
            let row = contribution(1, 0.0, None, 0.0, 0.0, dist);
            for (actual, expected) in row.into_iter().zip(expected) {
                assert_close(actual, expected, 1e-12);
            }
        }
    }

    #[test]
    fn likelihood_derivatives_match_finite_difference_for_every_censoring_type() {
        for dist in [
            SurvivalDist::Weibull,
            SurvivalDist::LogNormal,
            SurvivalDist::LogLogistic,
            SurvivalDist::StudentT(4.0),
        ] {
            for status in 0..=3 {
                assert_derivatives_match_finite_difference(
                    status,
                    0.4,
                    Some(1.1),
                    -0.2,
                    0.15,
                    dist,
                );
            }
        }
    }

    #[test]
    fn test_survregc1_basic() {
        let n = 5;
        let nvar = 1;
        let nstrat = 1;
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let strat = Array1::from_vec(vec![1i32; n]);
        let offset = Array1::from_vec(vec![0.0; n]);
        let time1 = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        let status = Array1::from_vec(vec![1i32, 1, 1, 1, 1]);
        let wt = Array1::from_vec(vec![1.0; n]);
        let covar = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let frail = Array1::from_vec(vec![0i32; n]);

        let result = survregc1(
            n,
            nvar,
            nstrat,
            false,
            &beta.view(),
            SurvivalDist::Weibull,
            &strat.view(),
            &offset.view(),
            &time1.view(),
            None,
            &status.view(),
            &wt.view(),
            &covar.view(),
            0,
            &frail.view(),
        );

        assert!(result.is_ok());
        let lik = result.unwrap();
        assert!(lik.loglik.is_finite());
        assert_eq!(lik.u.len(), nvar + nstrat);
        assert_eq!(lik.imat.nrows(), nvar + nstrat);
    }

    #[test]
    fn parallel_likelihood_matches_sequential_at_threshold() {
        let n = SURVREG_PARALLEL_THRESHOLD;
        let nvar = 2;
        let nstrat = 2;
        let beta = Array1::from_vec(vec![0.1, -0.2, 0.05, -0.1]);
        let strat = Array1::from_iter((0..n).map(|person| (person % 2 + 1) as i32));
        let offset = Array1::from_vec((0..n).map(|person| (person % 7) as f64 * 0.01).collect());
        let time1 = Array1::from_vec(
            (0..n)
                .map(|person| 0.2 + (person % 101) as f64 * 0.01)
                .collect(),
        );
        let status = Array1::from_vec((0..n).map(|person| (person % 4) as i32).collect());
        let weights = Array1::from_vec(
            (0..n)
                .map(|person| 0.75 + (person % 5) as f64 * 0.05)
                .collect(),
        );
        let covariates = Array2::from_shape_fn((nvar, n), |(column, person)| {
            (person % (17 + column)) as f64 * 0.02 - 0.1 * column as f64
        });
        let frailty = Array1::from_vec(vec![0i32; n]);

        let interval_widths =
            Array1::from_iter((0..n).map(|person| if person % 3 == 0 { 1e-8 } else { 0.2 }));
        for dist in [
            SurvivalDist::Weibull,
            SurvivalDist::LogNormal,
            SurvivalDist::LogLogistic,
            SurvivalDist::StudentT(4.0),
        ] {
            let parallel = survregc1(
                n,
                nvar,
                nstrat,
                false,
                &beta.view(),
                dist,
                &strat.view(),
                &offset.view(),
                &time1.view(),
                Some(&interval_widths.view()),
                &status.view(),
                &weights.view(),
                &covariates.view(),
                0,
                &frailty.view(),
            )
            .unwrap();
            let sequential = survregc1_sequential(
                n,
                nvar,
                nstrat,
                false,
                &beta.view(),
                dist,
                &strat.view(),
                &offset.view(),
                &time1.view(),
                Some(&interval_widths.view()),
                &status.view(),
                &weights.view(),
                &covariates.view(),
                0,
                &frailty.view(),
            )
            .unwrap();

            assert_close(parallel.loglik, sequential.loglik, 1e-10);
            for (actual, expected) in parallel.u.iter().zip(sequential.u.iter()) {
                assert_close(*actual, *expected, 1e-10);
            }
            for (actual, expected) in parallel.imat.iter().zip(sequential.imat.iter()) {
                assert_close(*actual, *expected, 1e-10);
            }
            for (actual, expected) in parallel.jj.iter().zip(sequential.jj.iter()) {
                assert_close(*actual, *expected, 1e-10);
            }
        }
    }

    #[test]
    fn zero_weight_overflowing_responses_do_not_affect_likelihood() {
        for n in [2, SURVREG_PARALLEL_THRESHOLD] {
            let beta = Array1::from_vec(vec![0.0, 0.0]);
            let strata = Array1::from_elem(n, 1);
            let offsets = Array1::zeros(n);
            let mut times = Array1::zeros(n);
            times[n - 1] = 1e200;
            let status = Array1::from_elem(n, 1);
            let mut weights = Array1::ones(n);
            weights[n - 1] = 0.0;
            let rows = Array2::ones((1, n));
            let frailty = Array1::zeros(n);
            let evaluate = |count| {
                survregc1(
                    count,
                    1,
                    1,
                    false,
                    &beta.view(),
                    SurvivalDist::Gaussian,
                    &strata.view(),
                    &offsets.view(),
                    &times.view(),
                    None,
                    &status.view(),
                    &weights.view(),
                    &rows.view(),
                    0,
                    &frailty.view(),
                )
                .unwrap()
            };
            let actual = evaluate(n);
            let expected = evaluate(n - 1);
            assert_eq!(actual.loglik, expected.loglik);
            assert_eq!(actual.u, expected.u);
            assert_eq!(actual.imat, expected.imat);
            assert_eq!(actual.jj, expected.jj);
        }
    }

    #[test]
    fn test_survregc1_with_censoring() {
        let n = 6;
        let nvar = 1;
        let nstrat = 1;
        let beta = Array1::from_vec(vec![0.5, 0.0]);
        let strat = Array1::from_vec(vec![1i32; n]);
        let offset = Array1::from_vec(vec![0.0; n]);
        let time1 = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
        let status = Array1::from_vec(vec![1i32, 0, 1, 0, 1, 0]);
        let wt = Array1::from_vec(vec![1.0; n]);
        let covar = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let frail = Array1::from_vec(vec![0i32; n]);

        let result = survregc1(
            n,
            nvar,
            nstrat,
            false,
            &beta.view(),
            SurvivalDist::Weibull,
            &strat.view(),
            &offset.view(),
            &time1.view(),
            None,
            &status.view(),
            &wt.view(),
            &covar.view(),
            0,
            &frail.view(),
        );

        assert!(result.is_ok());
        let lik = result.unwrap();
        assert!(lik.loglik.is_finite());
    }

    #[test]
    fn test_survregc1_lognormal() {
        let n = 5;
        let nvar = 1;
        let nstrat = 1;
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let strat = Array1::from_vec(vec![1i32; n]);
        let offset = Array1::from_vec(vec![0.0; n]);
        let time1 = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        let status = Array1::from_vec(vec![1i32, 1, 1, 1, 1]);
        let wt = Array1::from_vec(vec![1.0; n]);
        let covar = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let frail = Array1::from_vec(vec![0i32; n]);

        let result = survregc1(
            n,
            nvar,
            nstrat,
            false,
            &beta.view(),
            SurvivalDist::LogNormal,
            &strat.view(),
            &offset.view(),
            &time1.view(),
            None,
            &status.view(),
            &wt.view(),
            &covar.view(),
            0,
            &frail.view(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_survregc1_loglogistic() {
        let n = 5;
        let nvar = 1;
        let nstrat = 1;
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let strat = Array1::from_vec(vec![1i32; n]);
        let offset = Array1::from_vec(vec![0.0; n]);
        let time1 = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
        let status = Array1::from_vec(vec![1i32, 1, 1, 1, 1]);
        let wt = Array1::from_vec(vec![1.0; n]);
        let covar = Array2::from_shape_vec((nvar, n), vec![1.0; n]).unwrap();
        let frail = Array1::from_vec(vec![0i32; n]);

        let result = survregc1(
            n,
            nvar,
            nstrat,
            false,
            &beta.view(),
            SurvivalDist::LogLogistic,
            &strat.view(),
            &offset.view(),
            &time1.view(),
            None,
            &status.view(),
            &wt.view(),
            &covar.view(),
            0,
            &frail.view(),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_survival_likelihood_fields() {
        let lik = SurvivalLikelihood {
            loglik: -10.0,
            u: Array1::zeros(2),
            imat: Array2::zeros((2, 2)),
            jj: Array2::zeros((2, 2)),
            fdiag: Array1::zeros(0),
            jdiag: Array1::zeros(0),
        };
        assert!((lik.loglik - (-10.0)).abs() < 1e-10);
        assert_eq!(lik.u.len(), 2);
        assert_eq!(lik.imat.nrows(), 2);
    }
}
