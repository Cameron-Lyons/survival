use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN, PARALLEL_THRESHOLD_MEDIUM};
use crate::internal::statistical::{erf, erfc, ln_gamma};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::fmt;

type SurvregDerivatives = (f64, f64, f64, f64, f64, f64);
type DistributionEval = [f64; 4];

const SMALL: f64 = -200.0;
const SPI: f64 = 2.506628274631001;
const ROOT_2: f64 = std::f64::consts::SQRT_2;

#[derive(Debug)]
pub(crate) enum DistributionError {
    InvalidCase { case: i32, distribution: String },
}

impl fmt::Display for DistributionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistributionError::InvalidCase { case, distribution } => write!(
                f,
                "Invalid case {} for {} distribution. Valid cases are 1 (density) and 2 (CDF)",
                case, distribution
            ),
        }
    }
}

impl std::error::Error for DistributionError {}
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
    time2: Option<&ArrayView1<f64>>,
    status: &ArrayView1<i32>,
    wt: &ArrayView1<f64>,
    covar: &ArrayView2<f64>,
    nf: usize,
    frail: &ArrayView1<i32>,
) -> Result<SurvivalLikelihood, Box<dyn std::error::Error>> {
    let nvar2 = nvar + nstrat;
    let nvar3 = nvar2 + nf;

    if n < PARALLEL_THRESHOLD_MEDIUM || whichcase {
        return survregc1_sequential(
            n, nvar, nstrat, whichcase, beta, dist, strat, offset, time1, time2, status, wt, covar,
            nf, frail,
        );
    }

    let time2_slice = match time2 {
        Some(t) => Some(
            t.as_slice()
                .ok_or_else(|| "time2 array must be contiguous in memory".to_string())?,
        ),
        None => None,
    };

    type PersonResult = (usize, usize, usize, f64, f64, f64, SurvregDerivatives);
    let partial_results: Result<Vec<PersonResult>, Box<dyn std::error::Error + Send + Sync>> = (0
        ..n)
        .into_par_iter()
        .map(|person| {
            let strata_idx = if nstrat > 1 {
                (strat[person] - 1) as usize
            } else {
                0
            };
            let sigma = if nstrat > 1 {
                beta[nvar + nf + strata_idx].exp()
            } else {
                beta[nvar + nf].exp()
            };

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

            let sz = time1[person] - eta;
            let z = sz / sigma;

            let derivs: SurvregDerivatives = match status[person] {
                1 => compute_exact(z, sz, sigma, dist),
                0 => compute_right_censored(z, sz, sigma, dist),
                2 => compute_left_censored(z, sz, sigma, dist),
                3 => {
                    let time2_val = time2_slice.ok_or_else(|| {
                        Box::<dyn std::error::Error + Send + Sync>::from(
                            "Missing time2 for interval censored data",
                        )
                    })?[person];
                    compute_interval_censored(z, time2_val, eta, sigma, dist)
                }
                _ => Err("Invalid status value".into()),
            }
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

            Ok((person, fgrp, strata_idx, sigma, sz, wt[person], derivs))
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
    for (person, fgrp, strata_idx, _sigma, _sz, w, (g, dg, ddg, dsig, ddsig, dsg)) in
        partial_results
    {
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
    time2: Option<&ArrayView1<f64>>,
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
    let mut sigma;
    let mut _sig2;
    let mut strata = 0;
    for person in 0..n {
        if nstrat > 1 {
            strata = (strat[person] - 1) as usize;
            sigma = beta[nvar + nf + strata].exp();
        } else {
            sigma = beta[nvar + nf].exp();
        }
        _sig2 = 1.0 / (sigma * sigma);
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
        let sz = time1[person] - eta;
        let z = sz / sigma;
        let (g, dg, ddg, dsig, ddsig, dsg) = match status[person] {
            1 => compute_exact(z, sz, sigma, dist),
            0 => compute_right_censored(z, sz, sigma, dist),
            2 => compute_left_censored(z, sz, sigma, dist),
            3 => {
                let time2_val = time2
                    .ok_or_else(|| "Missing time2 for interval censored data".to_string())?[person];
                compute_interval_censored(z, time2_val, eta, sigma, dist)
            }
            _ => return Err("Invalid status value".into()),
        }?;
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
#[inline]
fn compute_exact(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let funs = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 1)?,
        SurvivalDist::Logistic | SurvivalDist::LogLogistic => logistic_d(z, 1)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 1)?,
        SurvivalDist::StudentT(df) => student_t_d(z, 1, df)?,
    };
    if funs[1] <= 0.0 {
        Ok((SMALL, -z / sigma, -1.0 / sigma, 0.0, 0.0, 0.0))
    } else {
        let g = funs[1].ln() - sigma.ln();
        let temp = funs[2] / sigma;
        let temp2 = funs[3] / (sigma * sigma);
        let dg = -temp;
        let dsig = -temp * sz;
        let ddg = temp2 - dg.powi(2);
        let dsg = sz * temp2 - dg * (dsig + 1.0);
        let ddsig = sz.powi(2) * temp2 - dsig * (1.0 + dsig);
        Ok((g, dg, ddg, dsig - 1.0, ddsig, dsg))
    }
}
#[inline]
fn compute_right_censored(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let funs = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic | SurvivalDist::LogLogistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
        SurvivalDist::StudentT(df) => student_t_d(z, 2, df)?,
    };
    if funs[1] <= 0.0 {
        Ok((SMALL, z / sigma, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = funs[1].ln();
        let temp = -funs[2] / (funs[1] * sigma);
        let temp2 = -funs[3] / (funs[1] * sigma * sigma);
        let dg = -temp;
        let dsig = -temp * sz;
        let ddg = temp2 - dg.powi(2);
        let dsg = sz * temp2 - dg * (dsig + 1.0);
        let ddsig = sz.powi(2) * temp2 - dsig * (1.0 + dsig);
        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}
#[inline]
fn compute_left_censored(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let funs = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic | SurvivalDist::LogLogistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
        SurvivalDist::StudentT(df) => student_t_d(z, 2, df)?,
    };
    if funs[0] <= 0.0 {
        Ok((SMALL, -z / sigma, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = funs[0].ln();
        let temp = funs[2] / (funs[0] * sigma);
        let temp2 = funs[3] / (funs[0] * sigma * sigma);
        let dg = -temp;
        let dsig = -temp * sz;
        let ddg = temp2 - dg.powi(2);
        let dsg = sz * temp2 - dg * (dsig + 1.0);
        let ddsig = sz.powi(2) * temp2 - dsig * (1.0 + dsig);
        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}
#[inline]
fn compute_interval_censored(
    z: f64,
    time2: f64,
    eta: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let sz2 = time2 - eta;
    let z2 = sz2 / sigma;
    let funs = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic | SurvivalDist::LogLogistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
        SurvivalDist::StudentT(df) => student_t_d(z, 2, df)?,
    };
    let ufun = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z2, 2)?,
        SurvivalDist::Logistic | SurvivalDist::LogLogistic => logistic_d(z2, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z2, 2)?,
        SurvivalDist::StudentT(df) => student_t_d(z2, 2, df)?,
    };
    let diff = if z > 0.0 {
        funs[1] - ufun[1]
    } else {
        ufun[0] - funs[0]
    };
    if diff <= 0.0 {
        Ok((SMALL, 1.0, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = diff.ln();
        let dg = -(ufun[2] - funs[2]) / (diff * sigma);
        let ddg = (ufun[3] - funs[3]) / (diff * sigma * sigma) - dg.powi(2);
        let dsig = (z * funs[2] - z2 * ufun[2]) / diff;
        let ddsig = (z2 * z2 * ufun[3] - z * z * funs[3]) / diff - dsig * (1.0 + dsig);
        let dsg = (z2 * ufun[3] - z * funs[3]) / (diff * sigma) - dg * (dsig + 1.0);
        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}
#[inline]
fn logistic_d(z: f64, case: i32) -> Result<DistributionEval, DistributionError> {
    let mut ans = [0.0; 4];
    let (w, sign, ii) = if z > 0.0 {
        ((-z).exp(), -1.0, 0usize)
    } else {
        (z.exp(), 1.0, 1usize)
    };
    let temp = 1.0 + w;
    match case {
        1 => {
            ans[1] = w / temp.powi(2);
            ans[2] = sign * (1.0 - w) / temp;
            ans[3] = (w.powi(2) - 4.0 * w + 1.0) / temp.powi(2);
            Ok(ans)
        }
        2 => {
            ans[1 - ii] = w / temp;
            ans[ii] = 1.0 / temp;
            ans[2] = w / temp.powi(2);
            ans[3] = sign * ans[2] * (1.0 - w) / temp;
            Ok(ans)
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "logistic".to_string(),
        }),
    }
}
#[inline]
fn gauss_d(z: f64, case: i32) -> Result<DistributionEval, DistributionError> {
    let mut ans = [0.0; 4];
    let f = (-z.powi(2) / 2.0).exp() / SPI;
    match case {
        1 => {
            ans[1] = f;
            ans[2] = -z;
            ans[3] = z.powi(2) - 1.0;
            Ok(ans)
        }
        2 => {
            if z > 0.0 {
                ans[0] = (1.0 + erf(z / ROOT_2)) / 2.0;
                ans[1] = erfc(z / ROOT_2) / 2.0;
            } else {
                ans[1] = (1.0 + erf(-z / ROOT_2)) / 2.0;
                ans[0] = erfc(-z / ROOT_2) / 2.0;
            }
            ans[2] = f;
            ans[3] = -z * f;
            Ok(ans)
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "Gaussian".to_string(),
        }),
    }
}
#[inline]
fn exvalue_d(z: f64, case: i32) -> Result<DistributionEval, DistributionError> {
    let mut ans = [0.0; 4];
    let w = z.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp();
    let temp = (-w).exp();
    match case {
        1 => {
            ans[1] = w * temp;
            ans[2] = 1.0 - w;
            ans[3] = w * (w - 3.0) + 1.0;
            Ok(ans)
        }
        2 => {
            ans[0] = 1.0 - temp;
            ans[1] = temp;
            ans[2] = w * temp;
            ans[3] = w * temp * (1.0 - w);
            Ok(ans)
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "extreme value".to_string(),
        }),
    }
}

#[inline]
fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b
    }
}

#[inline]
fn beta_continued_fraction(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-12 {
            break;
        }
    }
    h
}

#[inline]
fn student_t_pdf(z: f64, df: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_nan() { f64::NAN } else { 0.0 };
    }
    let log_coef = ln_gamma((df + 1.0) / 2.0)
        - ln_gamma(df / 2.0)
        - 0.5 * (df.ln() + std::f64::consts::PI.ln());
    (log_coef - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln()).exp()
}

#[inline]
fn student_t_cdf(z: f64, df: f64) -> f64 {
    if z == f64::INFINITY {
        return 1.0;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if z == 0.0 {
        return 0.5;
    }
    let x = df / (df + z * z);
    let ibeta = regularized_beta(df / 2.0, 0.5, x);
    if z > 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

#[inline]
fn student_t_d(z: f64, case: i32, df: f64) -> Result<DistributionEval, DistributionError> {
    let mut ans = [0.0; 4];
    let f = student_t_pdf(z, df);
    let denom = df + z * z;
    let score = -((df + 1.0) * z) / denom;
    match case {
        1 => {
            ans[1] = f;
            ans[2] = score;
            ans[3] = ((df + 1.0) * (df + 3.0) * z * z / denom.powi(2)) - ((df + 1.0) / denom);
            Ok(ans)
        }
        2 => {
            ans[0] = student_t_cdf(z, df);
            ans[1] = 1.0 - ans[0];
            ans[2] = f;
            ans[3] = f * score;
            Ok(ans)
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "Student-t".to_string(),
        }),
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
        let sz = y - eta;
        let z = sz / sigma;
        match status {
            0 => compute_right_censored(z, sz, sigma, dist).unwrap(),
            1 => compute_exact(z, sz, sigma, dist).unwrap(),
            2 => compute_left_censored(z, sz, sigma, dist).unwrap(),
            3 => compute_interval_censored(z, upper.unwrap(), eta, sigma, dist).unwrap(),
            _ => unreachable!(),
        }
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
        let (g, dg, ddg, dsig, ddsig, dsg) = contribution(status, y, upper, eta, log_sigma, dist);

        let g_eta_plus = contribution(status, y, upper, eta + h, log_sigma, dist).0;
        let g_eta_minus = contribution(status, y, upper, eta - h, log_sigma, dist).0;
        let eta_score = (g_eta_plus - g_eta_minus) / (2.0 * h);
        let eta_hessian = (g_eta_plus - 2.0 * g + g_eta_minus) / (h * h);

        let g_sigma_plus = contribution(status, y, upper, eta, log_sigma + h, dist).0;
        let g_sigma_minus = contribution(status, y, upper, eta, log_sigma - h, dist).0;
        let sigma_score = (g_sigma_plus - g_sigma_minus) / (2.0 * h);
        let sigma_hessian = (g_sigma_plus - 2.0 * g + g_sigma_minus) / (h * h);

        let cross = (contribution(status, y, upper, eta + h2, log_sigma + h2, dist).0
            - contribution(status, y, upper, eta + h2, log_sigma - h2, dist).0
            - contribution(status, y, upper, eta - h2, log_sigma + h2, dist).0
            + contribution(status, y, upper, eta - h2, log_sigma - h2, dist).0)
            / (4.0 * h2 * h2);

        assert_close(dg, eta_score, 1e-5);
        assert_close(ddg, eta_hessian, 1e-4);
        assert_close(dsig, sigma_score, 1e-5);
        assert_close(ddsig, sigma_hessian, 1e-4);
        assert_close(dsg, cross, 1e-4);
    }

    #[test]
    fn test_exvalue_d_density() {
        let result = exvalue_d(0.0, 1);
        assert!(result.is_ok());
        let funs = result.unwrap();
        assert!(funs[1] > 0.0);
        assert!((funs[2] - 0.0).abs() < 1e-10);
        assert!(funs[3].is_finite());
    }

    #[test]
    fn test_exvalue_d_survival() {
        let result = exvalue_d(0.0, 2);
        assert!(result.is_ok());
        let funs = result.unwrap();
        assert_close(funs[0], 1.0 - (-1.0f64).exp(), 1e-12);
        assert_close(funs[1], (-1.0f64).exp(), 1e-12);
        assert!(funs[2] > 0.0);
    }

    #[test]
    fn test_gauss_d_density() {
        let result = gauss_d(0.0, 1);
        assert!(result.is_ok());
        let funs = result.unwrap();
        assert!(funs[1] > 0.0);
        assert!((funs[2] - 0.0).abs() < 1e-10);
        assert!(funs[3].is_finite());
    }

    #[test]
    fn test_gauss_d_survival() {
        let result = gauss_d(0.0, 2);
        assert!(result.is_ok());
        let funs = result.unwrap();
        assert!((funs[0] - 0.5).abs() < 0.01);
        assert!((funs[1] - 0.5).abs() < 0.01);
        assert!(funs[2] > 0.0);
    }

    #[test]
    fn test_logistic_d_density() {
        let result = logistic_d(0.0, 1);
        assert!(result.is_ok());
        let funs = result.unwrap();
        assert!(funs[1] > 0.0);
        assert!((funs[2] - 0.0).abs() < 1e-10);
        assert!(funs[3].is_finite());
    }

    #[test]
    fn test_logistic_d_survival() {
        let result = logistic_d(2.0, 2);
        assert!(result.is_ok());
        let funs = result.unwrap();
        let cdf = 1.0 / (1.0 + (-2.0f64).exp());
        assert_close(funs[0], cdf, 1e-12);
        assert_close(funs[1], 1.0 - cdf, 1e-12);
        assert_close(funs[2], cdf * (1.0 - cdf), 1e-12);
    }

    #[test]
    fn test_compute_exact_weibull() {
        let result = compute_exact(0.0, 0.0, 1.0, SurvivalDist::Weibull);
        assert!(result.is_ok());
        let (g, dg, ddg, dsig, ddsig, dsg) = result.unwrap();
        assert!(g.is_finite());
        assert!(dg.is_finite());
        assert!(ddg.is_finite());
        assert!(dsig.is_finite());
        assert!(ddsig.is_finite());
        assert!(dsg.is_finite());
    }

    #[test]
    fn test_compute_exact_lognormal() {
        let result = compute_exact(0.0, 0.0, 1.0, SurvivalDist::LogNormal);
        assert!(result.is_ok());
        let (g, dg, ddg, dsig, ddsig, dsg) = result.unwrap();
        assert!(g.is_finite());
        assert!(dg.is_finite());
        assert!(ddg.is_finite());
        assert!(dsig.is_finite());
        assert!(ddsig.is_finite());
        assert!(dsg.is_finite());
    }

    #[test]
    fn test_compute_exact_loglogistic() {
        let result = compute_exact(0.0, 0.0, 1.0, SurvivalDist::LogLogistic);
        assert!(result.is_ok());
        let (g, dg, ddg, dsig, ddsig, dsg) = result.unwrap();
        assert!(g.is_finite());
        assert!(dg.is_finite());
        assert!(ddg.is_finite());
        assert!(dsig.is_finite());
        assert!(ddsig.is_finite());
        assert!(dsg.is_finite());
    }

    #[test]
    fn test_compute_right_censored_weibull() {
        let result = compute_right_censored(0.0, 0.0, 1.0, SurvivalDist::Weibull);
        assert!(result.is_ok());
        let (g, dg, ddg, dsig, ddsig, dsg) = result.unwrap();
        assert!(g.is_finite());
        assert_close(g, -1.0, 1e-12);
        assert!(dg.is_finite());
        assert!(ddg.is_finite());
        assert!(dsig.is_finite());
        assert!(ddsig.is_finite());
        assert!(dsg.is_finite());
    }

    #[test]
    fn test_compute_right_censored_lognormal() {
        let result = compute_right_censored(0.0, 0.0, 1.0, SurvivalDist::LogNormal);
        assert!(result.is_ok());
        let (g, _dg, _ddg, _dsig, _ddsig, _dsg) = result.unwrap();
        assert!(g.is_finite());
    }

    #[test]
    fn test_compute_right_censored_loglogistic() {
        let result = compute_right_censored(0.0, 0.0, 1.0, SurvivalDist::LogLogistic);
        assert!(result.is_ok());
        let (g, _dg, _ddg, _dsig, _ddsig, _dsg) = result.unwrap();
        assert!(g.is_finite());
    }

    #[test]
    fn test_compute_left_censored_weibull() {
        let result = compute_left_censored(0.0, 0.0, 1.0, SurvivalDist::Weibull);
        assert!(result.is_ok());
        let (g, _dg, _ddg, _dsig, _ddsig, _dsg) = result.unwrap();
        assert_close(g, (1.0 - (-1.0f64).exp()).ln(), 1e-12);
    }

    #[test]
    fn test_compute_interval_censored_weibull() {
        let upper = 2.0f64.ln();
        let result = compute_interval_censored(0.0, upper, 0.0, 1.0, SurvivalDist::Weibull);
        assert!(result.is_ok());
        let (g, _dg, _ddg, _dsig, _ddsig, _dsg) = result.unwrap();
        assert_close(g, ((-1.0f64).exp() - (-2.0f64).exp()).ln(), 1e-12);
    }

    #[test]
    fn test_censored_derivatives_match_finite_difference() {
        let dist = SurvivalDist::Weibull;
        assert_derivatives_match_finite_difference(0, 0.4, None, -0.2, 0.15, dist);
        assert_derivatives_match_finite_difference(2, 0.4, None, -0.2, 0.15, dist);
        assert_derivatives_match_finite_difference(3, 0.4, Some(1.1), -0.2, 0.15, dist);
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
