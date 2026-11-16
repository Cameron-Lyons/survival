use crate::internal::dist::{dnorm, dt, erf, erfc, pnorm, pt};
use crate::regression::survreg_distributions::SurvregDensity;
use std::f64::consts::SQRT_2;

pub(crate) const SPI: f64 = 2.506628274631001;
const KERNEL_CLAMP: f64 = 200.0;

/// The formulas of one builtin `survreg` family at a standardized residual `z`.
///
/// Each family is its own type so that a likelihood sweep can be compiled once
/// per family (see [`with_builtin_kernel`]): the formulas then inline into the
/// loop instead of being reached through a match on the family for every
/// observation, and the optimizer can drop the lanes a row does not read.
pub(crate) trait BuiltinKernel: Copy {
    /// `survreg6`'s kernel for an exact observation: `[0, f, f'/f, f''/f]`.
    fn exact(self, z: f64) -> [f64; 4];

    /// `survreg6`'s kernel for a censored observation: `[F, 1 - F, f, f']`.
    fn censored(self, z: f64) -> [f64; 4];

    /// One row of R's `density` component of `survreg.distributions`.
    ///
    /// These are R's own formulas, which differ from the `survreg6` kernels in
    /// their last bits (the extreme value is not clamped, the logistic is not
    /// split by the sign of `z`, the Gaussian tails come from `pnorm`), so the
    /// two are kept apart.
    fn density(self, z: f64) -> SurvregDensity;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExtremeValue;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Logistic;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Gaussian;

#[derive(Debug, Clone, Copy)]
pub(crate) struct StudentT {
    pub(crate) df: f64,
}

/// Evaluates `$body` with `$kernel` bound to the [`BuiltinKernel`] of the
/// family of `$distribution` (a `&SurvregDistribution` without callbacks), so
/// that `$body` is compiled once for each family.
macro_rules! with_builtin_kernel {
    ($distribution:expr, $kernel:ident => $body:expr) => {{
        use $crate::regression::survreg_distributions::SurvregFamily;
        use $crate::regression::survreg_kernels::{ExtremeValue, Gaussian, Logistic, StudentT};
        let distribution = $distribution;
        match distribution.family {
            SurvregFamily::ExtremeValue => {
                let $kernel = ExtremeValue;
                $body
            }
            SurvregFamily::Logistic => {
                let $kernel = Logistic;
                $body
            }
            SurvregFamily::Gaussian => {
                let $kernel = Gaussian;
                $body
            }
            SurvregFamily::T => {
                let $kernel = StudentT {
                    df: distribution.df(),
                };
                $body
            }
            SurvregFamily::Custom => unreachable!("custom density uses batches"),
        }
    }};
}

pub(crate) use with_builtin_kernel;

#[inline(always)]
fn extreme_value_terms(z: f64) -> (f64, f64) {
    let w = z.clamp(-KERNEL_CLAMP, KERNEL_CLAMP).exp();
    (w, (-w).exp())
}

impl BuiltinKernel for ExtremeValue {
    #[inline(always)]
    fn exact(self, z: f64) -> [f64; 4] {
        let (w, temp) = extreme_value_terms(z);
        [0.0, w * temp, 1.0 - w, w * (w - 3.0) + 1.0]
    }

    #[inline(always)]
    fn censored(self, z: f64) -> [f64; 4] {
        let (w, temp) = extreme_value_terms(z);
        [1.0 - temp, temp, w * temp, w * temp * (1.0 - w)]
    }

    #[inline(always)]
    fn density(self, z: f64) -> SurvregDensity {
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
}

/// `exp(-|z|)`, the sign of the score and whether `z > 0`.
#[inline(always)]
fn logistic_terms(z: f64) -> (f64, f64, bool) {
    if z > 0.0 {
        ((-z).exp(), -1.0, true)
    } else {
        (z.exp(), 1.0, false)
    }
}

impl BuiltinKernel for Logistic {
    #[inline(always)]
    fn exact(self, z: f64) -> [f64; 4] {
        let (w, sign, _) = logistic_terms(z);
        let temp = 1.0 + w;
        [
            0.0,
            w / (temp * temp),
            sign * (1.0 - w) / temp,
            (w * w - 4.0 * w + 1.0) / (temp * temp),
        ]
    }

    #[inline(always)]
    fn censored(self, z: f64) -> [f64; 4] {
        let (w, sign, positive) = logistic_terms(z);
        let temp = 1.0 + w;
        let (cdf, survival) = if positive {
            (1.0 / temp, w / temp)
        } else {
            (w / temp, 1.0 / temp)
        };
        let pdf = w / (temp * temp);
        [cdf, survival, pdf, sign * pdf * (1.0 - w) / temp]
    }

    #[inline(always)]
    fn density(self, z: f64) -> SurvregDensity {
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
}

#[inline(always)]
fn gaussian_pdf(z: f64) -> f64 {
    (-z * z / 2.0).exp() / SPI
}

impl BuiltinKernel for Gaussian {
    #[inline(always)]
    fn exact(self, z: f64) -> [f64; 4] {
        [0.0, gaussian_pdf(z), -z, z * z - 1.0]
    }

    #[inline(always)]
    fn censored(self, z: f64) -> [f64; 4] {
        let f = gaussian_pdf(z);
        let (cdf, survival) = if z > 0.0 {
            ((1.0 + erf(z / SQRT_2)) / 2.0, erfc(z / SQRT_2) / 2.0)
        } else {
            (erfc(-z / SQRT_2) / 2.0, (1.0 + erf(-z / SQRT_2)) / 2.0)
        };
        [cdf, survival, f, -z * f]
    }

    #[inline(always)]
    fn density(self, z: f64) -> SurvregDensity {
        SurvregDensity {
            cdf: pnorm(z, true, false),
            survival: pnorm(-z, true, false),
            pdf: dnorm(z, false),
            score: -z,
            curvature: z * z - 1.0,
        }
    }
}

/// The two tails of the t distribution from one `pt` call: `(F, 1 - F)`.
fn t_tails(z: f64, df: f64) -> (f64, f64) {
    let v = pt(-z.abs(), df, true, false);
    let other = 0.5 - v + 0.5;
    if z > 0.0 { (other, v) } else { (v, other) }
}

/// `[f, f'/f, f''/f]` of the t distribution.
fn t_density(z: f64, df: f64) -> [f64; 3] {
    let denom = df + z * z;
    [
        dt(z, df, false),
        -(df + 1.0) * z / denom,
        (df + 1.0) * (z * z * (df + 3.0) / denom - 1.0) / denom,
    ]
}

impl BuiltinKernel for StudentT {
    #[inline(always)]
    fn exact(self, z: f64) -> [f64; 4] {
        let [pdf, score, curvature] = t_density(z, self.df);
        [0.0, pdf, score, curvature]
    }

    #[inline(always)]
    fn censored(self, z: f64) -> [f64; 4] {
        let [pdf, score, _] = t_density(z, self.df);
        let (cdf, survival) = t_tails(z, self.df);
        [cdf, survival, pdf, pdf * score]
    }

    #[inline(always)]
    fn density(self, z: f64) -> SurvregDensity {
        let (cdf, survival) = t_tails(z, self.df);
        let [pdf, score, curvature] = t_density(z, self.df);
        SurvregDensity {
            cdf,
            survival,
            pdf,
            score,
            curvature,
        }
    }
}

#[cfg(test)]
pub(crate) mod reference;

#[cfg(test)]
mod tests {
    use super::reference::{self, KernelCase, builtin_distributions, residuals, same_bits};
    use super::*;

    fn assert_same(actual: &[f64], expected: &[f64], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(same_bits(*a, *e), "{context}: {actual:?} != {expected:?}");
        }
    }

    fn density_values(d: SurvregDensity) -> [f64; 5] {
        [d.cdf, d.survival, d.pdf, d.score, d.curvature]
    }

    #[test]
    fn kernels_reproduce_the_reference_formulas_bit_for_bit() {
        for distribution in builtin_distributions() {
            for z in residuals(4000, 11) {
                let context = format!("{} at {z}", distribution.name);
                let (exact, censored, density) = with_builtin_kernel!(&distribution, kernel => {
                    (kernel.exact(z), kernel.censored(z), kernel.density(z))
                });
                assert_same(
                    &exact,
                    &reference::kernel(&distribution, z, KernelCase::Density),
                    &context,
                );
                assert_same(
                    &censored,
                    &reference::kernel(&distribution, z, KernelCase::Distribution),
                    &context,
                );
                assert_same(
                    &density_values(density),
                    &density_values(reference::builtin_density(&distribution, z)),
                    &context,
                );
                assert_same(
                    &density_values(distribution.builtin_density(z)),
                    &density_values(density),
                    &context,
                );
            }
        }
    }
}
