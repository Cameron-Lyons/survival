//! The derivative matrix, the dfbeta residuals and the robust variance against
//! copies of their row-by-row definitions: every value must agree bit for bit.

use super::*;
use crate::core::strata_order::rowsum;
use crate::internal::matrix::matrix_from_rows;
use crate::internal::rng::Rng;
use crate::regression::parametric_survival::{
    FittingResponse, SurvregControl, SurvregData, derfun, robust_variance, survreg_fit,
};
use crate::regression::survreg_kernels::reference::{
    builtin_density, builtin_distributions, residuals, same_bits,
};
use ndarray::Array2;

fn reference_deriv_from_density(
    dmat: SurvregDensity,
    dmat2: SurvregDensity,
    z: f64,
    z2: f64,
    status: i32,
    sigma: f64,
) -> [f64; 6] {
    let dtemp = dmat.distribution_kernel()[3];
    let dtemp2 = dmat2.distribution_kernel()[3];
    let (tdenom, numerator_dg, numerator_ddg) = match status {
        0 => (dmat.survival, -dmat.pdf, -dtemp),
        1 => (1.0, dmat.score, dmat.curvature),
        2 => (dmat.cdf, dmat.pdf, dtemp),
        _ => (
            if z > 0.0 {
                dmat.survival - dmat2.survival
            } else {
                dmat2.cdf - dmat.cdf
            },
            dmat2.pdf - dmat.pdf,
            dtemp2 - dtemp,
        ),
    };
    let g = if status == 1 {
        (dmat.pdf / sigma).ln()
    } else {
        tdenom.ln()
    };
    let tdenom = 1.0 / tdenom;
    let dg = -(tdenom / sigma) * numerator_dg;
    let ddg = (tdenom / (sigma * sigma)) * numerator_ddg;
    let (ds, dds, dsg) = if status < 3 {
        (dg * sigma * z, ddg * (sigma * z).powi(2), ddg * sigma * z)
    } else {
        (
            tdenom * (z * dmat.pdf - z2 * dmat2.pdf),
            tdenom * (z2 * z2 * dtemp2 - z * z * dtemp),
            tdenom * (z2 * dtemp2 - z * dtemp) / sigma,
        )
    };
    [
        g,
        dg,
        ddg - dg * dg,
        if status == 1 { ds - 1.0 } else { ds },
        dds - ds * (1.0 + ds),
        dsg - dg * (1.0 + ds),
    ]
}

fn reference_deriv(
    distribution: &SurvregDistribution,
    y1: f64,
    y2: f64,
    status: i32,
    eta: f64,
    sigma: f64,
) -> [f64; 6] {
    let z = (y1 - eta) / sigma;
    let dmat = builtin_density(distribution, z);
    let (z2, dmat2) = if status == 3 {
        let z2 = (y2 - eta) / sigma;
        (z2, builtin_density(distribution, z2))
    } else {
        (0.0, dmat)
    };
    reference_deriv_from_density(dmat, dmat2, z, z2, status, sigma)
}

fn assert_same(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(same_bits(*a, *e), "{context} [{index}]: {a} != {e}");
    }
}

fn assert_same_rows(actual: &[Vec<f64>], expected: &[Vec<f64>], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (a, e) in actual.iter().zip(expected) {
        assert_same(a, e, context);
    }
}

#[test]
fn derivatives_match_the_reference_bit_for_bit() {
    let mut rng = Rng::with_seed(5);
    let offsets = residuals(1500, 6);
    for distribution in builtin_distributions() {
        for (index, &offset) in offsets.iter().enumerate() {
            let status = (index % 4) as i32;
            let sigma = 0.2 + 2.0 * rng.f64();
            let eta = 2.0 * rng.f64() - 1.0;
            let y1 = eta + offset * sigma;
            let y2 = if index % 3 == 0 {
                y1
            } else {
                y1 + 3.0 * rng.f64()
            };
            let context = format!("{} status {status} at {offset}", distribution.name);
            let expected = reference_deriv(&distribution, y1, y2, status, eta, sigma);
            let (actual, location) = with_builtin_kernel!(&distribution, kernel => (
                survreg_deriv(kernel, y1, y2, status, eta, sigma),
                location_deriv(kernel, y1, y2, status, eta, sigma),
            ));
            assert_same(&actual, &expected, &context);
            assert_same(&[location.0, location.1], &expected[1..3], &context);
        }
    }
}

#[test]
fn derfun_matches_the_reference_bit_for_bit() {
    let mut rng = Rng::with_seed(9);
    let n = 400;
    let y1: Vec<f64> = (0..n).map(|_| 6.0 * rng.f64() - 3.0).collect();
    let y2: Vec<f64> = y1
        .iter()
        .enumerate()
        .map(|(i, y)| if i % 5 == 0 { *y } else { y + rng.f64() })
        .collect();
    let status: Vec<i32> = (0..n).map(|i| (i % 4) as i32).collect();
    let eta: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
    let strata: Vec<usize> = (0..n).map(|i| i % 3).collect();
    let sigmas = [0.7, 1.3, 2.1];
    let response = FittingResponse {
        time: y1.clone(),
        time2: Some(y2.clone()),
        status: status.clone(),
        y1: y1.clone(),
        y2: y2.clone(),
        logcorrect: 0.0,
    };
    for distribution in builtin_distributions() {
        let (dg, ddg) = derfun(&distribution, &response, &eta, |i| sigmas[strata[i]]).unwrap();
        for i in 0..n {
            let expected = reference_deriv(
                &distribution,
                y1[i],
                y2[i],
                status[i],
                eta[i],
                sigmas[strata[i]],
            );
            assert_same(&[dg[i], ddg[i]], &expected[1..3], &distribution.name);
        }
    }
}

fn reference_row_times_matrix(score: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
    (0..score.len())
        .map(|col| {
            score
                .iter()
                .zip(matrix)
                .map(|(s, row)| s * row[col])
                .sum::<f64>()
        })
        .collect()
}

fn reference_score_row(
    x: ArrayView1<'_, f64>,
    eta_part: f64,
    scale_part: f64,
    stratum: usize,
    nstrata: usize,
    rsigma: bool,
) -> Vec<f64> {
    let mut row: Vec<f64> = x.iter().map(|value| eta_part * value).collect();
    if rsigma {
        let start = row.len();
        row.resize(start + nstrata, 0.0);
        row[start + stratum] = scale_part;
    }
    row
}

fn reference_collapse(rows: Vec<Vec<f64>>, collapse: &[usize]) -> Vec<Vec<f64>> {
    let codes: Vec<i32> = collapse
        .iter()
        .map(|&code| i32::try_from(code).unwrap())
        .collect();
    let rows = matrix_from_rows(&rows, "residuals").unwrap();
    matrix_rows(&rowsum(rows.view(), &codes))
}

/// `residuals(fit, "dfbeta", rsigma, collapse, weighted)` built row by row.
fn reference_dfbeta(
    fit: &SurvregFit,
    rsigma: bool,
    collapse: Option<&[usize]>,
    weighted: bool,
) -> Vec<Vec<f64>> {
    let nvar = fit.nvar();
    let nstrata = fit.nstrata();
    let rsigma = rsigma && fit.variance_matrix.len() != nvar;
    let vv = fit
        .naive_variance_matrix
        .as_deref()
        .unwrap_or(&fit.variance_matrix);
    let distribution = &fit.distribution;
    let (y1, y2) = distribution
        .transformed_endpoints(&fit.time, fit.time2.as_deref(), &fit.status)
        .unwrap();
    let mut rows: Vec<Vec<f64>> = (0..fit.n)
        .map(|i| {
            let sigma = fit.scale[fit.strata[i]];
            let deriv = reference_deriv(
                distribution,
                y1[i],
                y2[i],
                fit.status[i],
                fit.linear_predictors[i],
                sigma,
            );
            let score = reference_score_row(
                fit.covariates.row(i),
                deriv[1],
                deriv[3],
                fit.strata[i],
                nstrata,
                rsigma,
            );
            reference_row_times_matrix(&score, vv)
        })
        .collect();
    if weighted && let Some(weights) = &fit.weights {
        for (row, &w) in rows.iter_mut().zip(weights) {
            row.iter_mut().for_each(|value| *value *= w);
        }
    }
    match collapse {
        Some(groups) => reference_collapse(rows, groups),
        None => rows,
    }
}

fn reference_robust_variance(fit: &SurvregFit, cluster: Option<&[usize]>) -> Vec<Vec<f64>> {
    let width = fit.coefficients.len();
    let mut sandwich = vec![vec![0.0; width]; width];
    for row in &reference_dfbeta(fit, true, cluster, true) {
        for j in 0..width {
            for k in 0..width {
                sandwich[j][k] += row[j] * row[k];
            }
        }
    }
    sandwich
}

struct FitCase {
    distribution: &'static str,
    weights: bool,
    nstrata: usize,
    /// Codes are `rank * spread + 3`: a spread of 1 gives zero-based codes,
    /// a large one codes wider than the number of rows.
    spread: usize,
    clusters: usize,
}

fn sample_fit(case: &FitCase, seed: u64) -> (SurvregFit, Vec<usize>) {
    let mut rng = Rng::with_seed(seed);
    let n = 300;
    let x = Array2::from_shape_fn(
        (n, 3),
        |(_, j)| {
            if j == 0 { 1.0 } else { 2.0 * rng.f64() - 1.0 }
        },
    );
    let mut time = Vec::with_capacity(n);
    let mut time2 = Vec::with_capacity(n);
    let mut status = Vec::with_capacity(n);
    for i in 0..n {
        let t = (0.5 * x[[i, 1]] - 0.3 * x[[i, 2]] + rng.f64() - 0.5).exp();
        time.push(t);
        time2.push(t * (1.0 + rng.f64()));
        status.push(match i % 6 {
            0 | 1 => 1,
            2 | 3 => 0,
            4 => 2,
            _ => 3,
        });
    }
    let weights = case
        .weights
        .then(|| (0..n).map(|_| 0.2 + 2.0 * rng.f64()).collect());
    let strata = (case.nstrata > 1).then(|| (0..n).map(|i| i % case.nstrata).collect());
    let cluster: Vec<usize> = (0..n)
        .map(|_| (rng.f64() * case.clusters as f64) as usize * case.spread + 3)
        .collect();
    let data =
        SurvregData::try_new(time, status, x, Some(time2), weights, None, strata, None).unwrap();
    let distribution = SurvregDistribution::from_name(case.distribution, None).unwrap();
    let fit = survreg_fit(
        &data,
        &distribution,
        None,
        0.0,
        &SurvregControl::default(),
        false,
    )
    .unwrap();
    (fit, cluster)
}

#[test]
fn robust_variance_and_dfbeta_match_the_reference_bit_for_bit() {
    let cases = [
        FitCase {
            distribution: "weibull",
            weights: false,
            nstrata: 1,
            spread: 1,
            clusters: 60,
        },
        FitCase {
            distribution: "lognormal",
            weights: true,
            nstrata: 3,
            spread: 1000,
            clusters: 40,
        },
        FitCase {
            distribution: "loglogistic",
            weights: true,
            nstrata: 1,
            spread: 1,
            clusters: 300,
        },
        FitCase {
            distribution: "exponential",
            weights: true,
            nstrata: 1,
            spread: 7,
            clusters: 25,
        },
    ];
    for (index, case) in cases.iter().enumerate() {
        let (fit, cluster) = sample_fit(case, 40 + index as u64);
        let context = case.distribution;
        for cluster in [None, Some(cluster.as_slice())] {
            assert_same_rows(
                &robust_variance(&fit, cluster).unwrap(),
                &reference_robust_variance(&fit, cluster),
                context,
            );
            for (rsigma, weighted) in [(true, true), (false, true), (true, false)] {
                let actual = fit
                    .residuals(SurvregResidType::Dfbeta, rsigma, cluster, weighted)
                    .unwrap()
                    .values;
                assert_same_rows(
                    &actual,
                    &reference_dfbeta(&fit, rsigma, cluster, weighted),
                    context,
                );
            }
        }
    }
}

#[test]
fn collapse_codes_beyond_32_bits_are_rejected_by_every_path() {
    let case = FitCase {
        distribution: "weibull",
        weights: false,
        nstrata: 1,
        spread: 1,
        clusters: 10,
    };
    let (fit, mut cluster) = sample_fit(&case, 3);
    cluster[7] = i32::MAX as usize + 1;
    let message = "collapse codes must fit in 32 bits";
    let residuals = fit
        .residuals(SurvregResidType::Dfbeta, true, Some(&cluster), true)
        .unwrap_err();
    assert!(residuals.to_string().contains(message), "{residuals}");
    let robust = robust_variance(&fit, Some(&cluster)).unwrap_err();
    assert!(robust.to_string().contains(message), "{robust}");
    let short = robust_variance(&fit, Some(&cluster[1..])).unwrap_err();
    assert!(short.to_string().contains("collapse"), "{short}");
}
