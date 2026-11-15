//! The derivative matrix against a copy of its definition over the reference
//! density formulas: every value must agree bit for bit.

use super::*;
use crate::internal::rng::Rng;
use crate::regression::parametric_survival::{FittingResponse, derfun};
use crate::regression::survreg_kernels::reference::{
    builtin_density, builtin_distributions, residuals, same_bits,
};

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
