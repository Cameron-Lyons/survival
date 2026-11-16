//! The builtin `survreg` formulas behind a per-call match on the family, kept
//! as an independent reference: the [`BuiltinKernel`](super::BuiltinKernel)
//! implementations and the sweeps compiled from them must agree with these bit
//! for bit.

use crate::internal::dist::{dnorm, dt, erf, erfc, pnorm, pt};
use crate::regression::survreg_distributions::{
    SurvregDensity, SurvregDistribution, SurvregFamily,
};
use std::f64::consts::SQRT_2;

const SPI: f64 = 2.506628274631001;
const KERNEL_CLAMP: f64 = 200.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelCase {
    Density,
    Distribution,
}

fn t_tails(z: f64, df: f64) -> (f64, f64) {
    let v = pt(-z.abs(), df, true, false);
    let other = 0.5 - v + 0.5;
    if z > 0.0 { (other, v) } else { (v, other) }
}

fn t_density(z: f64, df: f64) -> [f64; 3] {
    let denom = df + z * z;
    [
        dt(z, df, false),
        -(df + 1.0) * z / denom,
        (df + 1.0) * (z * z * (df + 3.0) / denom - 1.0) / denom,
    ]
}

pub(crate) fn builtin_density(distribution: &SurvregDistribution, z: f64) -> SurvregDensity {
    match distribution.family {
        SurvregFamily::Custom => unreachable!("custom density uses batches"),
        SurvregFamily::ExtremeValue => {
            let w = z.exp();
            let ww = (-w).exp();
            SurvregDensity {
                cdf: 1.0 - ww,
                survival: ww,
                pdf: w * ww,
                score: 1.0 - w,
                curvature: w * (w - 3.0) + 1.0,
            }
        }
        SurvregFamily::Logistic => {
            let w = z.exp();
            let denom = 1.0 + w;
            SurvregDensity {
                cdf: w / denom,
                survival: 1.0 / denom,
                pdf: w / (denom * denom),
                score: (1.0 - w) / denom,
                curvature: (w * (w - 4.0) + 1.0) / (denom * denom),
            }
        }
        SurvregFamily::Gaussian => SurvregDensity {
            cdf: pnorm(z, true, false),
            survival: pnorm(-z, true, false),
            pdf: dnorm(z, false),
            score: -z,
            curvature: z * z - 1.0,
        },
        SurvregFamily::T => {
            let df = distribution.df();
            let (cdf, survival) = t_tails(z, df);
            let [pdf, score, curvature] = t_density(z, df);
            SurvregDensity {
                cdf,
                survival,
                pdf,
                score,
                curvature,
            }
        }
    }
}

pub(crate) fn kernel(distribution: &SurvregDistribution, z: f64, case: KernelCase) -> [f64; 4] {
    match distribution.family {
        SurvregFamily::Custom => unreachable!("custom density uses batches"),
        SurvregFamily::ExtremeValue => {
            let w = z.clamp(-KERNEL_CLAMP, KERNEL_CLAMP).exp();
            let temp = (-w).exp();
            match case {
                KernelCase::Density => [0.0, w * temp, 1.0 - w, w * (w - 3.0) + 1.0],
                KernelCase::Distribution => [1.0 - temp, temp, w * temp, w * temp * (1.0 - w)],
            }
        }
        SurvregFamily::Logistic => {
            let (w, sign, positive) = if z > 0.0 {
                ((-z).exp(), -1.0, true)
            } else {
                (z.exp(), 1.0, false)
            };
            let temp = 1.0 + w;
            match case {
                KernelCase::Density => [
                    0.0,
                    w / (temp * temp),
                    sign * (1.0 - w) / temp,
                    (w * w - 4.0 * w + 1.0) / (temp * temp),
                ],
                KernelCase::Distribution => {
                    let (cdf, survival) = if positive {
                        (1.0 / temp, w / temp)
                    } else {
                        (w / temp, 1.0 / temp)
                    };
                    let pdf = w / (temp * temp);
                    [cdf, survival, pdf, sign * pdf * (1.0 - w) / temp]
                }
            }
        }
        SurvregFamily::Gaussian => {
            let f = (-z * z / 2.0).exp() / SPI;
            match case {
                KernelCase::Density => [0.0, f, -z, z * z - 1.0],
                KernelCase::Distribution => {
                    let (cdf, survival) = if z > 0.0 {
                        ((1.0 + erf(z / SQRT_2)) / 2.0, erfc(z / SQRT_2) / 2.0)
                    } else {
                        (erfc(-z / SQRT_2) / 2.0, (1.0 + erf(-z / SQRT_2)) / 2.0)
                    };
                    [cdf, survival, f, -z * f]
                }
            }
        }
        SurvregFamily::T => {
            let df = distribution.df();
            let [pdf, score, curvature] = t_density(z, df);
            match case {
                KernelCase::Density => [0.0, pdf, score, curvature],
                KernelCase::Distribution => {
                    let (cdf, survival) = t_tails(z, df);
                    [cdf, survival, pdf, pdf * score]
                }
            }
        }
    }
}

/// Bitwise equality, with every NaN equal to every other (a NaN's payload is
/// not part of any result).
pub(crate) fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The builtin families, the t with its default and a non-integer df.
pub(crate) fn builtin_distributions() -> Vec<SurvregDistribution> {
    vec![
        SurvregDistribution::from_name("extreme", None).unwrap(),
        SurvregDistribution::from_name("logistic", None).unwrap(),
        SurvregDistribution::from_name("gaussian", None).unwrap(),
        SurvregDistribution::from_name("t", None).unwrap(),
        SurvregDistribution::from_name("t", Some(&[7.5])).unwrap(),
    ]
}

/// Standardized residuals covering the branch points and tails of every
/// family (the extreme-value clamp at 200, underflow past 700, infinities and
/// NaN) followed by `random` draws spread over many magnitudes.
pub(crate) fn residuals(random: usize, seed: u64) -> Vec<f64> {
    let mut values = vec![
        0.0,
        -0.0,
        1e-300,
        -1e-300,
        0.5,
        -0.5,
        1.0,
        -1.0,
        37.5,
        -37.5,
        199.9,
        -199.9,
        200.0,
        -200.0,
        201.0,
        -201.0,
        710.0,
        -710.0,
        800.0,
        -800.0,
        1e6,
        -1e6,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut rng = crate::internal::rng::Rng::with_seed(seed);
    values.extend((0..random).map(|_| {
        let magnitude = 10f64.powf(rng.f64() * 6.0 - 3.0);
        if rng.bool() { magnitude } else { -magnitude }
    }));
    values
}
