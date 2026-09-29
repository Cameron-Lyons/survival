//! The `survreg` log-likelihood kernel: a port of `src/survregc1.c` from the
//! CRAN `survival` package.  Given the current parameter vector it returns
//! the log-likelihood, the score vector `u` and the observed information
//! `imat`, and on request the outer-product approximation `JJ` that
//! `survreg6.c` falls back on when `imat` is not positive definite.
//!
//! `survregc2.c`, the variant that evaluates a user-written density through
//! an R callback, differs from `survregc1.c` only in where the density
//! summary comes from. [`SurvregDensitySource`] supplies one batch of densities
//! for custom implementations; built-ins keep their direct scalar kernels.
//! Errors propagate through Newton steps, Fisher fallbacks and penalized line
//! searches. (survregc2.c also
//! indexes the linear predictor and the stratum scales of a model with a
//! sparse term without the `nf` frailty offset; the shared sweep indexes
//! them as survregc1.c does.)
//!
//! [`SurvregKernel::evaluate`] is the sweep of `survreg6.c`; the penalised
//! `survreg7.c` calls [`SurvregKernel::evaluate_blocks`], which adds the
//! sparse frailty term (`nf > 0`: a diagonal block of `nf` group effects)
//! and keeps the C code's layout of the information, and
//! [`SurvregKernel::loglik_at`], its log-likelihood-only `whichcase = 1`.

use crate::error::SurvivalResult;
use crate::regression::survreg_density::{SurvregDensitySource, check_density_batch};
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
    /// `n x nvar` design matrix; a view that is not in standard (row major)
    /// layout is copied into one per evaluation.
    pub covariates: ArrayView2<'a, f64>,
    pub weights: &'a [f64],
    pub offset: &'a [f64],
    /// Zero-based stratum of each observation (all zero without strata).
    pub strata: &'a [usize],
    /// Number of `log(scale)` parameters being estimated: 0 for a fixed
    /// scale, 1 without strata, the number of strata otherwise.
    pub nstrat: usize,
    pub distribution: &'a dyn SurvregDensitySource,
}

/// The kernel output for one parameter vector.
pub(crate) struct SurvregLikelihood {
    pub loglik: f64,
    /// Score vector, `nvar + nstrat` long.
    pub u: Vec<f64>,
    /// Observed information (negative Hessian), `nvar2 x nvar2`.
    pub imat: Array2<f64>,
    /// `JJ`, when the evaluation was asked for it.
    pub jj: Option<Array2<f64>>,
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
        Self::exact_from(funs, z, sz, sigma)
    }

    fn exact_from(funs: [f64; 4], z: f64, sz: f64, sigma: f64) -> Self {
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
        Self::right_from(funs, z, sz, sigma)
    }

    fn right_from(funs: [f64; 4], z: f64, sz: f64, sigma: f64) -> Self {
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
        Self::left_from(funs, z, sz, sigma)
    }

    fn left_from(funs: [f64; 4], z: f64, sz: f64, sigma: f64) -> Self {
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
        Self::interval_from(funs, ufun, z, zu, sigma)
    }

    fn interval_from(funs: [f64; 4], ufun: [f64; 4], z: f64, zu: f64, sigma: f64) -> Self {
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

    /// Adds the observation's weighted score to `u` and its information to
    /// the lower triangle of `imat`, and the outer product of its score to
    /// the lower triangle of `jj` when that is given; the matrices are
    /// `nvar2 x nvar2` (`nvar2 = u.len()`), row major.  `x` holds its
    /// covariates and `scale` is the index of its `log(scale)` parameter,
    /// `None` for a fixed scale.
    fn accumulate(
        &self,
        x: &[f64],
        scale: Option<usize>,
        w: f64,
        u: &mut [f64],
        imat: &mut [f64],
        mut jj: Option<&mut [f64]>,
    ) {
        let nvar2 = u.len();
        for (i, &xi) in x.iter().enumerate() {
            let temp = self.dg * xi * w;
            u[i] += temp;
            let lower = i * nvar2..i * nvar2 + i + 1;
            // With JJ, one loop updates both matrices, as in survregc1.c.
            match jj.as_deref_mut() {
                None => {
                    for (m, &xj) in imat[lower].iter_mut().zip(x) {
                        *m -= xi * xj * self.ddg * w;
                    }
                }
                Some(jj) => {
                    for ((m, q), &xj) in imat[lower.clone()].iter_mut().zip(&mut jj[lower]).zip(x) {
                        *m -= xi * xj * self.ddg * w;
                        *q += temp * xj * self.dg;
                    }
                }
            }
        }
        if let Some(k) = scale {
            let row = k * nvar2..k * nvar2 + x.len();
            u[k] += w * self.dsig;
            for (m, &xi) in imat[row.clone()].iter_mut().zip(x) {
                *m -= self.dsg * xi * w;
            }
            imat[k * nvar2 + k] -= self.ddsig * w;
            if let Some(jj) = jj {
                for (q, &xi) in jj[row].iter_mut().zip(x) {
                    *q += self.dsig * xi * self.dg * w;
                }
                jj[k * nvar2 + k] += self.dsig * self.dsig * w;
            }
        }
    }
}

/// The sparse term of survregc1.c (`nf > 0`): the 0-based group of each
/// row and the number of groups.
pub(crate) struct SparseFrailty<'a> {
    pub group: &'a [usize],
    pub nf: usize,
}

/// survregc1.c's `whichcase = 0` output in survreg7's layout, reused across
/// the evaluations of one fit.  With `nvar2 = nvar + nstrat` dense
/// parameters, `hmat` and `jj` have `nvar2` rows and `nf + nvar2` columns
/// (C's `imat[i][j]`, row `i` a dense parameter, the frailty columns first)
/// and only their lower triangle is filled.
pub(crate) struct BlockLikelihood {
    /// The `beta` this evaluation was made at (before `survpenal` recentres
    /// it).
    pub at: Vec<f64>,
    pub loglik: f64,
    /// Score: frailties, `x` coefficients, `log(scale)`s.
    pub u: Vec<f64>,
    /// The information (R's `imat`).
    pub hmat: Array2<f64>,
    /// The sparse diagonal of the information.
    pub fdiag: Vec<f64>,
    /// `JJ` and its sparse diagonal, valid only when `has_jj`.
    pub jj: Array2<f64>,
    pub jdiag: Vec<f64>,
    pub has_jj: bool,
}

impl BlockLikelihood {
    pub(crate) fn new(nf: usize, nvar2: usize) -> Self {
        Self {
            at: Vec::new(),
            loglik: 0.0,
            u: vec![0.0; nf + nvar2],
            hmat: Array2::zeros((nvar2, nf + nvar2)),
            fdiag: vec![0.0; nf],
            jj: Array2::zeros((nvar2, nf + nvar2)),
            jdiag: vec![0.0; nf],
            has_jj: false,
        }
    }
}

/// Density callbacks are evaluated once, before accumulation. Built-ins keep
/// their allocation-free, case-specific scalar evaluation.
enum Contributions<'a> {
    Builtin(&'a SurvregDistribution),
    Batch(Vec<Contribution>),
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
    /// reproduced), and `JJ` when `with_jj` is set.
    ///
    /// `beta` holds `nvar` coefficients followed by the `log(scale)` values:
    /// one per stratum when they are estimated, or the fixed `log(scale)`
    /// tacked on at position `nvar` when `nstrat == 0`.
    pub(crate) fn evaluate(
        &self,
        beta: &[f64],
        with_jj: bool,
    ) -> SurvivalResult<SurvregLikelihood> {
        let nvar = self.nvar();
        let nvar2 = self.nvar2();
        debug_assert!(beta.len() > nvar, "beta must carry a log(scale)");
        let design = self.covariates.as_standard_layout();
        let design = design.as_slice().expect("standard layout");
        let contributions = self.prepare_contributions(beta, None, design)?;
        let mut loglik = 0.0;
        let mut u = vec![0.0; nvar2];
        let mut imat = vec![0.0; nvar2 * nvar2];
        let mut jj = with_jj.then(|| vec![0.0; nvar2 * nvar2]);

        for person in 0..self.n() {
            let stratum = if self.nstrat > 1 {
                self.strata[person]
            } else {
                0
            };
            let sigma = beta[nvar + stratum].exp();
            let x = &design[person * nvar..(person + 1) * nvar];
            let contribution = match &contributions {
                Contributions::Builtin(distribution) => {
                    let eta = self.offset[person]
                        + x.iter().zip(&beta[..nvar]).map(|(x, b)| x * b).sum::<f64>();
                    self.contribution(distribution, person, eta, sigma)
                }
                Contributions::Batch(rows) => rows[person],
            };
            let w = self.weights[person];
            loglik += contribution.g * w;
            let scale = (self.nstrat != 0).then_some(nvar + stratum);
            contribution.accumulate(x, scale, w, &mut u, &mut imat, jj.as_deref_mut());
        }

        Ok(SurvregLikelihood {
            loglik,
            u,
            imat: symmetric_from_lower(nvar2, imat),
            jj: jj.map(|jj| symmetric_from_lower(nvar2, jj)),
        })
    }

    /// Row `person`'s log-likelihood and derivatives at the linear
    /// predictor `eta` and scale `sigma`: the four censoring cases.
    fn contribution(
        &self,
        distribution: &SurvregDistribution,
        person: usize,
        eta: f64,
        sigma: f64,
    ) -> Contribution {
        let sz = self.y1[person] - eta;
        let z = sz / sigma;
        match self.status[person] {
            1 => Contribution::exact(distribution, z, sz, sigma),
            0 => Contribution::right_censored(distribution, z, sz, sigma),
            2 => Contribution::left_censored(distribution, z, sz, sigma),
            _ => {
                let zu = (self.y2[person] - eta) / sigma;
                Contribution::interval_censored(distribution, z, zu, sigma)
            }
        }
    }

    /// `survregc2` packs all lower endpoints first, then the interval upper
    /// endpoints. Offset, scale stratum and sparse effects are applied before
    /// the one callback invocation. Its errors stop the optimizer immediately.
    fn prepare_contributions(
        &self,
        beta: &[f64],
        frailty: Option<&SparseFrailty<'_>>,
        design: &[f64],
    ) -> SurvivalResult<Contributions<'_>> {
        if let Some(distribution) = self.distribution.builtin() {
            return Ok(Contributions::Builtin(distribution));
        }
        let n = self.n();
        let nvar = self.nvar();
        let mut z = vec![0.0; n + self.status.iter().filter(|&&s| s == 3).count()];
        let mut upper_index = n;
        let mut scales = Vec::with_capacity(n);
        for person in 0..n {
            let x = &design[person * nvar..(person + 1) * nvar];
            let (eta, sigma) = self.block_eta_sigma(person, x, beta, frailty);
            z[person] = (self.y1[person] - eta) / sigma;
            scales.push(sigma);
            if self.status[person] == 3 {
                z[upper_index] = (self.y2[person] - eta) / sigma;
                upper_index += 1;
            }
        }
        let values = self.distribution.density_batch(&z)?;
        check_density_batch(&values, z.len())?;
        let mut upper_index = n;
        let mut rows = Vec::with_capacity(n);
        for person in 0..n {
            let d = values[person];
            let sigma = scales[person];
            let sz = z[person] * sigma;
            let row = match self.status[person] {
                1 => Contribution::exact_from(d.density_kernel(), z[person], sz, sigma),
                0 => Contribution::right_from(d.distribution_kernel(), z[person], sz, sigma),
                2 => Contribution::left_from(d.distribution_kernel(), z[person], sz, sigma),
                _ => {
                    let row = Contribution::interval_from(
                        d.distribution_kernel(),
                        values[upper_index].distribution_kernel(),
                        z[person],
                        z[upper_index],
                        sigma,
                    );
                    upper_index += 1;
                    row
                }
            };
            rows.push(row);
        }
        Ok(Contributions::Batch(rows))
    }

    /// The linear predictor and scale of row `person` for survreg7's
    /// `beta` (frailties, `x` coefficients, `log(scale)`s), in survregc1.c's
    /// order: `sum(x * beta)`, then the offset, then the frailty.
    fn block_eta_sigma(
        &self,
        person: usize,
        x: &[f64],
        beta: &[f64],
        frailty: Option<&SparseFrailty<'_>>,
    ) -> (f64, f64) {
        let nf = frailty.map_or(0, |f| f.nf);
        let nvar = x.len();
        let stratum = if self.nstrat > 1 {
            self.strata[person]
        } else {
            0
        };
        let mut eta = 0.0;
        for (xi, b) in x.iter().zip(&beta[nf..nf + nvar]) {
            eta += b * xi;
        }
        eta += self.offset[person];
        if let Some(frailty) = frailty {
            eta += beta[frailty.group[person]];
        }
        (eta, beta[nf + nvar + stratum].exp())
    }

    /// survregc1.c with `whichcase = 0` for survreg7: the log-likelihood,
    /// score and information at `beta` (`nf` frailties, `nvar`
    /// coefficients, then the `log(scale)`s, or the fixed `log(scale)` when
    /// `nstrat == 0`) into `out`, and `JJ` when `with_jj` (it costs about
    /// 40% of the sweep). `O(n nvar2^2)` time; built-ins allocate no workspace,
    /// while callbacks use linear endpoint, density and contribution buffers.
    pub(crate) fn evaluate_blocks(
        &self,
        beta: &[f64],
        frailty: Option<&SparseFrailty<'_>>,
        with_jj: bool,
        out: &mut BlockLikelihood,
    ) -> SurvivalResult<()> {
        let nvar = self.nvar();
        let nf = frailty.map_or(0, |f| f.nf);
        let width = nf + self.nvar2();
        let design = self.covariates.as_standard_layout();
        let design = design.as_slice().expect("standard layout");
        let contributions = self.prepare_contributions(beta, frailty, design)?;
        out.at.clear();
        out.at.extend_from_slice(beta);
        out.u.fill(0.0);
        out.fdiag.fill(0.0);
        out.hmat.fill(0.0);
        if with_jj {
            out.jdiag.fill(0.0);
            out.jj.fill(0.0);
        }
        out.has_jj = with_jj;
        let u = &mut out.u;
        let fdiag = &mut out.fdiag;
        let jdiag = &mut out.jdiag;
        let hmat = out.hmat.as_slice_mut().expect("standard layout");
        let jj = out.jj.as_slice_mut().expect("standard layout");
        let mut loglik = 0.0;
        for person in 0..self.n() {
            let x = &design[person * nvar..(person + 1) * nvar];
            let c = match &contributions {
                Contributions::Builtin(distribution) => {
                    let (eta, sigma) = self.block_eta_sigma(person, x, beta, frailty);
                    self.contribution(distribution, person, eta, sigma)
                }
                Contributions::Batch(rows) => rows[person],
            };
            let w = self.weights[person];
            loglik += c.g * w;
            let group = frailty.map(|f| f.group[person]);
            if let Some(g) = group {
                u[g] += c.dg * w;
                fdiag[g] -= c.ddg * w;
                if with_jj {
                    jdiag[g] += c.dg * c.dg * w;
                }
            }
            for (i, &xi) in x.iter().enumerate() {
                let temp = c.dg * xi * w;
                u[nf + i] += temp;
                let row = i * width;
                for (j, &xj) in x[..=i].iter().enumerate() {
                    hmat[row + nf + j] -= xi * xj * c.ddg * w;
                    if with_jj {
                        jj[row + nf + j] += temp * xj * c.dg;
                    }
                }
                if let Some(g) = group {
                    hmat[row + g] -= xi * c.ddg * w;
                    if with_jj {
                        jj[row + g] += temp * c.dg;
                    }
                }
            }
            if self.nstrat != 0 {
                let stratum = if self.nstrat > 1 {
                    self.strata[person]
                } else {
                    0
                };
                let k = stratum + nvar;
                let row = k * width;
                u[nf + k] += w * c.dsig;
                for (i, &xi) in x.iter().enumerate() {
                    hmat[row + nf + i] -= c.dsg * xi * w;
                    if with_jj {
                        jj[row + nf + i] += c.dsig * xi * c.dg * w;
                    }
                }
                hmat[row + nf + k] -= c.ddsig * w;
                if with_jj {
                    jj[row + nf + k] += c.dsig * c.dsig * w;
                }
                if let Some(g) = group {
                    hmat[row + g] -= c.dsg * w;
                    if with_jj {
                        jj[row + g] += c.dsig * c.dg * w;
                    }
                }
            }
        }
        out.loglik = loglik;
        Ok(())
    }

    /// survregc1.c with `whichcase = 1`: the log-likelihood alone at
    /// `beta`, bit for bit the one [`Self::evaluate_blocks`] returns there.
    /// `O(n nvar)`.
    pub(crate) fn loglik_at(
        &self,
        beta: &[f64],
        frailty: Option<&SparseFrailty<'_>>,
    ) -> SurvivalResult<f64> {
        let nvar = self.nvar();
        let design = self.covariates.as_standard_layout();
        let design = design.as_slice().expect("standard layout");
        let contributions = self.prepare_contributions(beta, frailty, design)?;
        let mut loglik = 0.0;
        for person in 0..self.n() {
            let x = &design[person * nvar..(person + 1) * nvar];
            let c = match &contributions {
                Contributions::Builtin(distribution) => {
                    let (eta, sigma) = self.block_eta_sigma(person, x, beta, frailty);
                    self.contribution(distribution, person, eta, sigma)
                }
                Contributions::Batch(rows) => rows[person],
            };
            loglik += c.g * self.weights[person];
        }
        Ok(loglik)
    }

    /// `JJ` at `beta`: the sum of the squared score contributions, the
    /// Fisher-scoring information `survreg6.c` uses when `imat` is not
    /// positive definite.  `survregc1.c` accumulates it in every call, about
    /// 40% of the `O(n p^2)` work, so the fit asks for it only while it
    /// steps with it and otherwise evaluates `beta` again to get it.
    pub(crate) fn jj(&self, beta: &[f64]) -> SurvivalResult<Array2<f64>> {
        Ok(self.evaluate(beta, true)?.jj.expect("evaluated with JJ"))
    }
}

/// The symmetric `n x n` matrix whose lower triangle is that of `lower`
/// (row major).
fn symmetric_from_lower(n: usize, lower: Vec<f64>) -> Array2<f64> {
    let mut matrix = Array2::from_shape_vec((n, n), lower).expect("an n x n matrix");
    for i in 0..n {
        for j in 0..i {
            matrix[[j, i]] = matrix[[i, j]];
        }
    }
    matrix
}

#[cfg(test)]
mod callback_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::survreg_distributions::{SurvregFamily, SurvregTransform};
    use ndarray::{Array2, ArrayView2};

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
        let lik = kernel.evaluate(&beta, false).unwrap();
        let jj = kernel.jj(&beta).unwrap();
        assert_eq!(lik.u.len(), 3);
        assert_eq!(lik.imat.shape(), &[3, 3]);
        assert_eq!(jj.shape(), &[3, 3]);
        // The information matrix is symmetric and, at these values, positive.
        for i in 0..3 {
            for j in 0..3 {
                assert_close(lik.imat[[i, j]], lik.imat[[j, i]], 0.0);
                assert_close(jj[[i, j]], jj[[j, i]], 0.0);
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
        let fd = (kernel.evaluate(&up, false).unwrap().loglik
            - kernel.evaluate(&down, false).unwrap().loglik)
            / (2.0 * h);
        assert_close(lik.u[2], fd, 1e-6);
    }

    #[test]
    fn jj_is_the_sum_of_squared_score_contributions() {
        let t = SurvregDistribution::from_name("t", None).unwrap();
        let y1 = [0.0, 0.5, 1.0, 1.5, 2.0];
        let y2 = [0.0, 0.5, 1.4, 1.5, 2.0];
        let status = [1, 0, 3, 2, 1];
        let covariates = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.1, 1.0, -0.4, 1.0, 0.3, 1.0, 0.8, 1.0, 0.5],
        )
        .unwrap();
        let ones = [1.0; 5];
        let zeros = [0.0; 5];
        let strata = [0, 1, 0, 1, 1];
        let kernel = |rows: std::ops::Range<usize>| SurvregKernel {
            y1: &y1[rows.clone()],
            y2: &y2[rows.clone()],
            status: &status[rows.clone()],
            covariates: covariates.slice(ndarray::s![rows.clone(), ..]),
            weights: &ones[rows.clone()],
            offset: &zeros[rows.clone()],
            strata: &strata[rows],
            nstrat: 2,
            distribution: &t,
        };
        let beta = [0.4, -0.3, 0.1, -0.2];
        let jj = kernel(0..5).jj(&beta).unwrap();
        let mut expected = Array2::<f64>::zeros((4, 4));
        for person in 0..5 {
            let u = kernel(person..person + 1).evaluate(&beta, false).unwrap().u;
            for i in 0..4 {
                for j in 0..4 {
                    expected[[i, j]] += u[i] * u[j];
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                assert_close(jj[[i, j]], expected[[i, j]], 1e-14);
            }
        }
        // Accumulating JJ leaves the log-likelihood, score and information
        // as they are without it.
        let without = kernel(0..5).evaluate(&beta, false).unwrap();
        let with = kernel(0..5).evaluate(&beta, true).unwrap();
        assert!(without.jj.is_none());
        assert_eq!(with.loglik, without.loglik);
        assert_eq!(with.u, without.u);
        assert_eq!(with.imat, without.imat);
    }

    #[test]
    fn a_column_major_design_gives_the_same_likelihood() {
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let y1 = [0.2, 0.9, 1.3, 0.4];
        let status = [1, 0, 1, 2];
        let rows = [1.0, 0.5, 1.0, -0.2, 1.0, 1.1, 1.0, 0.3];
        let row_major = Array2::from_shape_vec((4, 2), rows.to_vec()).unwrap();
        let column_major = row_major.t().to_owned();
        let ones = [1.0; 4];
        let zeros = [0.0; 4];
        let strata = [0; 4];
        let kernel = |covariates| SurvregKernel {
            y1: &y1,
            y2: &y1,
            status: &status,
            covariates,
            weights: &ones,
            offset: &zeros,
            strata: &strata,
            nstrat: 1,
            distribution: &weibull,
        };
        let beta = [0.3, 0.2, -0.1];
        let expected = kernel(row_major.view()).evaluate(&beta, true).unwrap();
        let got = kernel(column_major.t()).evaluate(&beta, true).unwrap();
        assert_eq!(got.loglik, expected.loglik);
        assert_eq!(got.u, expected.u);
        assert_eq!(got.imat, expected.imat);
        assert_eq!(got.jj, expected.jj);
    }

    /// Rows with all four censoring codes, two strata and three groups.
    struct BlockData {
        y1: Vec<f64>,
        y2: Vec<f64>,
        status: Vec<i32>,
        x: Array2<f64>,
        weights: Vec<f64>,
        offset: Vec<f64>,
        strata: Vec<usize>,
        group: Vec<usize>,
    }

    fn block_data() -> BlockData {
        let n = 12;
        BlockData {
            y1: (0..n).map(|i| 0.3 + 0.2 * i as f64).collect(),
            y2: (0..n).map(|i| 0.9 + 0.2 * i as f64).collect(),
            status: (0..n).map(|i| (i % 4) as i32).collect(),
            x: Array2::from_shape_fn((n, 2), |(i, j)| {
                if j == 0 {
                    1.0
                } else {
                    ((i * 7) % 5) as f64 * 0.3 - 0.5
                }
            }),
            weights: (0..n).map(|i| 1.0 + (i % 3) as f64 * 0.5).collect(),
            offset: (0..n).map(|i| (i % 2) as f64 * 0.1).collect(),
            strata: (0..n).map(|i| i % 2).collect(),
            group: (0..n).map(|i| (i * 5) % 3).collect(),
        }
    }

    fn block_kernel<'a>(
        data: &'a BlockData,
        x: ArrayView2<'a, f64>,
        nstrat: usize,
        distribution: &'a SurvregDistribution,
    ) -> SurvregKernel<'a> {
        SurvregKernel {
            y1: &data.y1,
            y2: &data.y2,
            status: &data.status,
            covariates: x,
            weights: &data.weights,
            offset: &data.offset,
            strata: &data.strata,
            nstrat,
            distribution,
        }
    }

    #[test]
    fn blocks_without_a_frailty_are_the_survreg6_sweep() {
        let data = block_data();
        let weibull = SurvregDistribution::from_name("weibull", None).unwrap();
        let kernel = block_kernel(&data, data.x.view(), 2, &weibull);
        let beta = [0.4, 0.3, -0.2, 0.1];
        let full = kernel.evaluate(&beta, true).unwrap();
        let mut blocks = BlockLikelihood::new(0, 4);
        kernel
            .evaluate_blocks(&beta, None, true, &mut blocks)
            .unwrap();
        assert_eq!(blocks.loglik, full.loglik);
        assert_eq!(blocks.u, full.u);
        assert_eq!(blocks.at, beta);
        let jj = full.jj.unwrap();
        for i in 0..4 {
            for j in 0..=i {
                assert_eq!(blocks.hmat[[i, j]], full.imat[[i, j]]);
                assert_eq!(blocks.jj[[i, j]], jj[[i, j]]);
            }
        }
    }

    #[test]
    fn loglik_at_is_the_loglik_of_the_full_evaluation() {
        let data = block_data();
        let frailty = SparseFrailty {
            group: &data.group,
            nf: 3,
        };
        for name in ["weibull", "loglogistic", "lognormal", "t"] {
            let distribution = SurvregDistribution::from_name(name, None).unwrap();
            for (nstrat, sparse) in [(2, true), (1, false), (0, true)] {
                let kernel = block_kernel(&data, data.x.view(), nstrat, &distribution);
                let frailty = sparse.then_some(&frailty);
                let nf = if sparse { 3 } else { 0 };
                let mut beta = vec![0.05, -0.1, 0.08][..nf].to_vec();
                beta.extend([0.4, 0.3, -0.2, 0.1][..2 + nstrat.max(1)].iter());
                let mut blocks = BlockLikelihood::new(nf, 2 + nstrat);
                kernel
                    .evaluate_blocks(&beta, frailty, false, &mut blocks)
                    .unwrap();
                let loglik = kernel.loglik_at(&beta, frailty).unwrap();
                assert_eq!(loglik.to_bits(), blocks.loglik.to_bits(), "{name} {nstrat}");
            }
        }
    }

    #[test]
    fn a_sparse_frailty_is_its_indicator_columns() {
        // The sparse term's blocks equal those of the same model with the
        // groups as dense indicator columns placed first.
        let data = block_data();
        let lognormal = SurvregDistribution::from_name("lognormal", None).unwrap();
        let frailty = SparseFrailty {
            group: &data.group,
            nf: 3,
        };
        let sparse_kernel = block_kernel(&data, data.x.view(), 2, &lognormal);
        let beta = [0.05, -0.1, 0.08, 0.4, 0.3, -0.2, 0.1];
        let mut sparse = BlockLikelihood::new(3, 4);
        sparse_kernel
            .evaluate_blocks(&beta, Some(&frailty), true, &mut sparse)
            .unwrap();

        let dense_x = Array2::from_shape_fn((12, 5), |(i, j)| {
            if j < 3 {
                f64::from(u8::from(data.group[i] == j))
            } else {
                data.x[[i, j - 3]]
            }
        });
        let dense_kernel = block_kernel(&data, dense_x.view(), 2, &lognormal);
        let dense = dense_kernel.evaluate(&beta, true).unwrap();
        let dense_jj = dense.jj.unwrap();
        let close = |a: f64, b: f64| assert_close(a, b, 1e-12 * b.abs().max(1.0));
        close(sparse.loglik, dense.loglik);
        for i in 0..7 {
            close(sparse.u[i], dense.u[i]);
        }
        for g in 0..3 {
            close(sparse.fdiag[g], dense.imat[[g, g]]);
            close(sparse.jdiag[g], dense_jj[[g, g]]);
        }
        for i in 0..4 {
            for j in 0..3 + i + 1 {
                close(sparse.hmat[[i, j]], dense.imat[[i + 3, j]]);
                close(sparse.jj[[i, j]], dense_jj[[i + 3, j]]);
            }
        }
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
        let lik = kernel.evaluate(&[0.3, 0.0], false).unwrap();
        assert_eq!(lik.u.len(), 1);
        assert_eq!(lik.imat.shape(), &[1, 1]);
        assert!(lik.loglik.is_finite());
    }
}
