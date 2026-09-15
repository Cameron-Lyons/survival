//! The `survreg` log-likelihood kernel: a port of `src/survregc1.c` from the
//! CRAN `survival` package.  Given the current parameter vector it returns
//! the log-likelihood, the score vector `u`, the observed information
//! `imat` and the outer-product approximation `JJ` that `survreg6.c` falls
//! back on when `imat` is not positive definite.
//!
//! `survregc2.c`, the variant that evaluates a user-written density through
//! an R callback, differs from `survregc1.c` only in where the density
//! summary comes from; [`SurvregDistribution::kernel`] hides that difference
//! (the `t` family goes through the five-column `density` exactly as the
//! callback does), so one sweep serves every distribution.  Sparse frailty
//! terms (`nf > 0`, used only by `survreg7.c`) are not supported.

use crate::regression::survreg_distributions::{KernelCase, SurvregDistribution};
use ndarray::{Array2, ArrayView2};

/// `#define SMALL -200`: the log-likelihood contribution used when an
/// observation lands off the probability scale (a wildly wrong `beta`);
/// the value only has to trigger step halving in the caller.
const SMALL: f64 = -200.0;

/// Everything the kernel needs besides the parameter vector: the response on
/// the fitting scale, the design matrix, weights, offset and strata.
pub(crate) struct SurvregKernel<'a> {
    /// Transformed lower time (`y[, 1]`).
    pub y1: &'a [f64],
    /// Transformed upper time (`y[, 2]`), read only where `status == 3`.
    pub y2: &'a [f64],
    /// Censoring code per observation: 0 right, 1 exact, 2 left, 3 interval.
    pub status: &'a [i32],
    /// `n x nvar` design matrix.
    pub covariates: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    pub offset: &'a [f64],
    /// Zero-based stratum of each observation (all zero without strata).
    pub strata: &'a [usize],
    /// Number of `log(scale)` parameters being estimated: 0 for a fixed
    /// scale, 1 without strata, the number of strata otherwise.
    pub nstrat: usize,
    pub distribution: &'a SurvregDistribution,
}

/// The kernel output for one parameter vector.
pub(crate) struct SurvregLikelihood {
    pub loglik: f64,
    /// Score vector, `nvar + nstrat` long.
    pub u: Vec<f64>,
    /// Observed information (negative Hessian), `nvar2 x nvar2`.
    pub imat: Array2<f64>,
    /// Sum of squared score contributions, `nvar2 x nvar2`.
    pub jj: Array2<f64>,
}

/// One observation's log-likelihood and its derivatives with respect to the
/// linear predictor `eta` and `log(sigma)`: `g`, `dg = dg/d eta`,
/// `ddg = d2g/d eta2`, `dsig = dg/d log sigma`, `ddsig`, `dsg` (cross).
#[derive(Debug, Clone, Copy)]
struct Contribution {
    g: f64,
    dg: f64,
    ddg: f64,
    dsig: f64,
    ddsig: f64,
    dsg: f64,
}

impl Contribution {
    /// The "off the probability scale" fallback of the C code.
    fn small(dg: f64, ddg: f64) -> Self {
        Self {
            g: SMALL,
            dg,
            ddg,
            dsig: 0.0,
            ddsig: 0.0,
            dsg: 0.0,
        }
    }

    /// `case 1` of `survregc1.c`: an exact observation.
    fn exact(distribution: &SurvregDistribution, z: f64, sz: f64, sigma: f64) -> Self {
        let funs = distribution.kernel(z, KernelCase::Density);
        if funs[1] <= 0.0 {
            return Self::small(-z / sigma, -1.0 / sigma);
        }
        let g = funs[1].ln() - sigma.ln();
        let temp = funs[2] / sigma;
        let temp2 = funs[3] / (sigma * sigma);
        let dg = -temp;
        let dsig = -temp * sz;
        Self {
            g,
            dg,
            ddg: temp2 - dg * dg,
            dsg: sz * temp2 - dg * (dsig + 1.0),
            ddsig: sz * sz * temp2 - dsig * (1.0 + dsig),
            dsig: dsig - 1.0,
        }
    }

    /// `case 0`: right censored.
    fn right_censored(distribution: &SurvregDistribution, z: f64, sz: f64, sigma: f64) -> Self {
        let funs = distribution.kernel(z, KernelCase::Distribution);
        if funs[1] <= 0.0 {
            return Self::small(z / sigma, 0.0);
        }
        let temp = -funs[2] / (funs[1] * sigma);
        let temp2 = -funs[3] / (funs[1] * sigma * sigma);
        Self::censored(funs[1].ln(), temp, temp2, sz)
    }

    /// `case 2`: left censored.
    fn left_censored(distribution: &SurvregDistribution, z: f64, sz: f64, sigma: f64) -> Self {
        let funs = distribution.kernel(z, KernelCase::Distribution);
        if funs[0] <= 0.0 {
            return Self::small(-z / sigma, 0.0);
        }
        let temp = funs[2] / (funs[0] * sigma);
        let temp2 = funs[3] / (funs[0] * sigma * sigma);
        Self::censored(funs[0].ln(), temp, temp2, sz)
    }

    /// The derivative block shared by the two one-sided censoring cases.
    fn censored(g: f64, temp: f64, temp2: f64, sz: f64) -> Self {
        let dg = -temp;
        let dsig = -temp * sz;
        Self {
            g,
            dg,
            ddg: temp2 - dg * dg,
            dsig,
            ddsig: sz * sz * temp2 - dsig * (1.0 + dsig),
            dsg: sz * temp2 - dg * (dsig + 1.0),
        }
    }

    /// `case 3`: interval censored between `z` and `zu`.
    fn interval_censored(distribution: &SurvregDistribution, z: f64, zu: f64, sigma: f64) -> Self {
        let funs = distribution.kernel(z, KernelCase::Distribution);
        let ufun = distribution.kernel(zu, KernelCase::Distribution);
        // Differencing the tail that is small on both ends stops round-off.
        let temp = if z > 0.0 {
            funs[1] - ufun[1]
        } else {
            ufun[0] - funs[0]
        };
        if temp <= 0.0 {
            return Self::small(1.0, 0.0);
        }
        let sig2 = 1.0 / (sigma * sigma);
        let dg = -(ufun[2] - funs[2]) / (temp * sigma);
        let dsig = (z * funs[2] - zu * ufun[2]) / temp;
        Self {
            g: temp.ln(),
            dg,
            ddg: (ufun[3] - funs[3]) * sig2 / temp - dg * dg,
            dsig,
            ddsig: (zu * zu * ufun[3] - z * z * funs[3]) / temp - dsig * (1.0 + dsig),
            dsg: (zu * ufun[3] - z * funs[3]) / (temp * sigma) - dg * (dsig + 1.0),
        }
    }
}

impl SurvregKernel<'_> {
    pub(crate) fn n(&self) -> usize {
        self.y1.len()
    }

    pub(crate) fn nvar(&self) -> usize {
        self.covariates.ncols()
    }

    /// Number of parameters iterated over (`nvar + nstrat`).
    pub(crate) fn nvar2(&self) -> usize {
        self.nvar() + self.nstrat
    }

    /// Log-likelihood, score and information at `beta` (`whichcase = 0`;
    /// the log-likelihood-only `whichcase = 1` call of the C code is
    /// always followed by a full evaluation at the same point, so it is not
    /// reproduced).
    ///
    /// `beta` holds `nvar` coefficients followed by the `log(scale)` values:
    /// one per stratum when they are estimated, or the fixed `log(scale)`
    /// tacked on at position `nvar` when `nstrat == 0`.
    pub(crate) fn evaluate(&self, beta: &[f64]) -> SurvregLikelihood {
        let n = self.n();
        let nvar = self.nvar();
        let nvar2 = self.nvar2();
        debug_assert!(beta.len() > nvar, "beta must carry a log(scale)");
        let mut result = SurvregLikelihood {
            loglik: 0.0,
            u: vec![0.0; nvar2],
            imat: Array2::zeros((nvar2, nvar2)),
            jj: Array2::zeros((nvar2, nvar2)),
        };

        for person in 0..n {
            let stratum = if self.nstrat > 1 {
                self.strata[person]
            } else {
                0
            };
            let sigma = beta[nvar + stratum].exp();
            let row = self.covariates.row(person);
            let eta = self.offset[person]
                + row
                    .iter()
                    .zip(&beta[..nvar])
                    .map(|(x, b)| x * b)
                    .sum::<f64>();
            let sz = self.y1[person] - eta;
            let z = sz / sigma;
            let contribution = match self.status[person] {
                1 => Contribution::exact(self.distribution, z, sz, sigma),
                0 => Contribution::right_censored(self.distribution, z, sz, sigma),
                2 => Contribution::left_censored(self.distribution, z, sz, sigma),
                _ => {
                    let zu = (self.y2[person] - eta) / sigma;
                    Contribution::interval_censored(self.distribution, z, zu, sigma)
                }
            };
            let w = self.weights[person];
            result.loglik += contribution.g * w;

            let Contribution {
                dg,
                ddg,
                dsig,
                ddsig,
                dsg,
                ..
            } = contribution;
            for i in 0..nvar {
                let temp = dg * row[i] * w;
                result.u[i] += temp;
                for j in 0..=i {
                    result.imat[[i, j]] -= row[i] * row[j] * ddg * w;
                    result.jj[[i, j]] += temp * row[j] * dg;
                }
            }
            if self.nstrat != 0 {
                let k = stratum + nvar;
                result.u[k] += w * dsig;
                for i in 0..nvar {
                    result.imat[[k, i]] -= dsg * row[i] * w;
                    result.jj[[k, i]] += dsig * row[i] * dg * w;
                }
                result.imat[[k, k]] -= ddsig * w;
                result.jj[[k, k]] += dsig * dsig * w;
            }
        }

        symmetrize_lower(&mut result.imat);
        symmetrize_lower(&mut result.jj);
        result
    }
}

/// Copies the strictly lower triangle into the upper one.
fn symmetrize_lower(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..i {
            matrix[[j, i]] = matrix[[i, j]];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::survreg_distributions::{SurvregFamily, SurvregTransform};
    use ndarray::Array2;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn contribution(
        distribution: &SurvregDistribution,
        status: i32,
        y: f64,
        upper: f64,
        eta: f64,
        log_sigma: f64,
    ) -> Contribution {
        let sigma = log_sigma.exp();
        let sz = y - eta;
        let z = sz / sigma;
        match status {
            0 => Contribution::right_censored(distribution, z, sz, sigma),
            1 => Contribution::exact(distribution, z, sz, sigma),
            2 => Contribution::left_censored(distribution, z, sz, sigma),
            _ => Contribution::interval_censored(distribution, z, (upper - eta) / sigma, sigma),
        }
    }

    fn assert_derivatives_match_finite_difference(
        distribution: &SurvregDistribution,
        status: i32,
        y: f64,
        upper: f64,
        eta: f64,
        log_sigma: f64,
    ) {
        let h = 1e-5;
        let h2 = 1e-4;
        let c = contribution(distribution, status, y, upper, eta, log_sigma);
        let g = |eta: f64, log_sigma: f64| {
            contribution(distribution, status, y, upper, eta, log_sigma).g
        };

        let eta_score = (g(eta + h, log_sigma) - g(eta - h, log_sigma)) / (2.0 * h);
        let eta_hessian = (g(eta + h, log_sigma) - 2.0 * c.g + g(eta - h, log_sigma)) / (h * h);
        let sigma_score = (g(eta, log_sigma + h) - g(eta, log_sigma - h)) / (2.0 * h);
        let sigma_hessian = (g(eta, log_sigma + h) - 2.0 * c.g + g(eta, log_sigma - h)) / (h * h);
        let cross = (g(eta + h2, log_sigma + h2)
            - g(eta + h2, log_sigma - h2)
            - g(eta - h2, log_sigma + h2)
            + g(eta - h2, log_sigma - h2))
            / (4.0 * h2 * h2);

        assert_close(c.dg, eta_score, 1e-5);
        assert_close(c.ddg, eta_hessian, 1e-4);
        assert_close(c.dsig, sigma_score, 1e-5);
        assert_close(c.ddsig, sigma_hessian, 1e-4);
        assert_close(c.dsg, cross, 1e-4);
    }

    #[test]
    fn derivatives_match_finite_differences_for_every_family_and_status() {
        for family in [
            SurvregFamily::ExtremeValue,
            SurvregFamily::Logistic,
            SurvregFamily::Gaussian,
            SurvregFamily::T,
        ] {
            let distribution =
                SurvregDistribution::custom("test", family, SurvregTransform::Identity, None, None);
            for status in 0..4 {
                assert_derivatives_match_finite_difference(
                    &distribution,
                    status,
                    0.4,
                    1.1,
                    -0.2,
                    0.15,
                );
                assert_derivatives_match_finite_difference(
                    &distribution,
                    status,
                    -0.7,
                    0.3,
                    0.5,
                    -0.4,
                );
            }
        }
    }

    #[test]
    fn off_scale_observations_use_the_small_fallback() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        // An interval of zero width has probability zero.
        let c = Contribution::interval_censored(&weibull, 0.3, 0.3, 1.0);
        assert_eq!(c.g, SMALL);
        assert_eq!(c.dg, 1.0);
        // Far in the upper tail the extreme value survival is exactly zero.
        let c = Contribution::right_censored(&weibull, 800.0, 800.0, 1.0);
        assert_eq!(c.g, SMALL);
    }

    #[test]
    fn kernel_accumulates_score_and_information() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let y1 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        let status = [1, 0, 1, 0, 1, 0];
        let covariates = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.4, 1.0, 0.5, 1.0, 0.6],
        )
        .unwrap();
        let weights = [1.0; 6];
        let offset = [0.0; 6];
        let strata = [0; 6];
        let kernel = SurvregKernel {
            y1: &y1,
            y2: &y1,
            status: &status,
            covariates: covariates.view(),
            weights: &weights,
            offset: &offset,
            strata: &strata,
            nstrat: 1,
            distribution: &weibull,
        };
        let beta = [0.5, 0.2, -0.1];
        let lik = kernel.evaluate(&beta);
        assert_eq!(lik.u.len(), 3);
        assert_eq!(lik.imat.shape(), &[3, 3]);
        // The information matrix is symmetric and, at these values, positive.
        for i in 0..3 {
            for j in 0..3 {
                assert_close(lik.imat[[i, j]], lik.imat[[j, i]], 0.0);
                assert_close(lik.jj[[i, j]], lik.jj[[j, i]], 0.0);
            }
            assert!(lik.imat[[i, i]] > 0.0);
        }
        // Score = sum of weighted per-observation gradients: check the
        // log(scale) entry against a finite difference of the loglik.
        let h = 1e-6;
        let mut up = beta;
        up[2] += h;
        let mut down = beta;
        down[2] -= h;
        let fd = (kernel.evaluate(&up).loglik - kernel.evaluate(&down).loglik) / (2.0 * h);
        assert_close(lik.u[2], fd, 1e-6);
    }

    #[test]
    fn fixed_scale_kernel_reads_the_trailing_log_scale() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let y1 = [0.0, 0.5, 1.0];
        let status = [1, 1, 1];
        let covariates = Array2::from_shape_vec((3, 1), vec![1.0; 3]).unwrap();
        let weights = [1.0; 3];
        let offset = [0.0; 3];
        let strata = [0; 3];
        let kernel = SurvregKernel {
            y1: &y1,
            y2: &y1,
            status: &status,
            covariates: covariates.view(),
            weights: &weights,
            offset: &offset,
            strata: &strata,
            nstrat: 0,
            distribution: &weibull,
        };
        let lik = kernel.evaluate(&[0.3, 0.0]);
        assert_eq!(lik.u.len(), 1);
        assert_eq!(lik.imat.shape(), &[1, 1]);
        assert!(lik.loglik.is_finite());
    }
}
