use crate::constants::{EXP_CLAMP_MAX, EXP_CLAMP_MIN, PARALLEL_THRESHOLD_MEDIUM};
use crate::utilities::statistical::{erf, erfc};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::fmt;

type SurvregDerivatives = (f64, f64, f64, f64, f64, f64);

const SMALL: f64 = -200.0;
const SPI: f64 = 2.506628274631001;
const ROOT_2: f64 = std::f64::consts::SQRT_2;

#[derive(Debug)]
pub enum DistributionError {
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
pub enum SurvivalDist {
    ExtremeValue,
    Logistic,
    Gaussian,
    Weibull,
    LogNormal,
}
pub struct SurvivalLikelihood {
    pub loglik: f64,
    pub u: Array1<f64>,
    pub imat: Array2<f64>,
    pub jj: Array2<f64>,
    pub fdiag: Array1<f64>,
    pub jdiag: Array1<f64>,
}

#[derive(Clone, Copy)]
pub struct SurvregDimensions {
    pub nvar: usize,
    pub nstrat: usize,
    pub nf: usize,
}

#[derive(Clone, Copy)]
pub struct Derivatives {
    pub dg: f64,
    pub ddg: f64,
    pub dsig: f64,
    pub ddsig: f64,
    pub dsg: f64,
}
#[allow(clippy::too_many_arguments)]
pub fn survregc1(
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
                    compute_interval_censored(z, sz, time2_val, eta, sigma, dist)
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

    Ok(result)
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
                compute_interval_censored(z, sz, time2_val, eta, sigma, dist)
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
    Ok(result)
}
#[inline]
fn compute_exact(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let (f, df, ddf) = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 1)?,
        SurvivalDist::Logistic => logistic_d(z, 1)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 1)?,
    };
    if f <= 0.0 {
        Ok((SMALL, -z / sigma, -1.0 / sigma, 0.0, 0.0, 0.0))
    } else {
        let g = f.ln() - sigma.ln();
        let temp = df / sigma;
        let temp2 = ddf / (sigma * sigma);
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
    let (f, df, _ddf) = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
    };
    if f <= 0.0 || f >= 1.0 {
        Ok((SMALL, 0.0, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = f.ln();
        let temp = df / (f * sigma);
        let dg = temp;
        let dsig = temp * sz;
        let ddg = -dg.powi(2);
        let dsg = -sz * dg.powi(2);
        let ddsig = -sz.powi(2) * dg.powi(2);
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
    let (f, df, _ddf) = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
    };
    if f <= 0.0 || f >= 1.0 {
        Ok((SMALL, 0.0, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = (1.0 - f).ln();
        let temp = -df / ((1.0 - f) * sigma);
        let dg = temp;
        let dsig = temp * sz;
        let ddg = -dg.powi(2);
        let dsg = -sz * dg.powi(2);
        let ddsig = -sz.powi(2) * dg.powi(2);
        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}
#[inline]
fn compute_interval_censored(
    z: f64,
    sz: f64,
    time2: f64,
    eta: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<SurvregDerivatives, Box<dyn std::error::Error>> {
    let sz2 = time2 - eta;
    let z2 = sz2 / sigma;
    let (f1, df1, _ddf1) = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z, 2)?,
        SurvivalDist::Logistic => logistic_d(z, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z, 2)?,
    };
    let (f2, df2, _ddf2) = match dist {
        SurvivalDist::ExtremeValue | SurvivalDist::Weibull => exvalue_d(z2, 2)?,
        SurvivalDist::Logistic => logistic_d(z2, 2)?,
        SurvivalDist::Gaussian | SurvivalDist::LogNormal => gauss_d(z2, 2)?,
    };
    let diff = f2 - f1;
    if diff <= 0.0 {
        Ok((SMALL, 0.0, 0.0, 0.0, 0.0, 0.0))
    } else {
        let g = diff.ln();
        let temp1 = df1 / (diff * sigma);
        let temp2 = df2 / (diff * sigma);
        let dg = temp2 - temp1;
        let dsig = (temp2 * sz2 - temp1 * sz) / sigma;
        let ddg = -(dg.powi(2));
        let dsg = -(sz * temp1.powi(2) + sz2 * temp2.powi(2)) / sigma;
        let ddsig = -(sz.powi(2) * temp1.powi(2) + sz2.powi(2) * temp2.powi(2)) / (sigma * sigma);
        Ok((g, dg, ddg, dsig, ddsig, dsg))
    }
}
#[inline]
fn logistic_d(z: f64, case: i32) -> Result<(f64, f64, f64), DistributionError> {
    let (w, sign) = if z > 0.0 {
        ((-z).exp(), -1.0)
    } else {
        (z.exp(), 1.0)
    };
    let temp = 1.0 + w;
    match case {
        1 => {
            let f = w / temp.powi(2);
            let df = sign * (1.0 - w) / temp;
            let ddf = (w.powi(2) - 4.0 * w + 1.0) / temp.powi(2);
            Ok((f, df, ddf))
        }
        2 => {
            let f = w / temp;
            let df = w / temp.powi(2);
            let ddf = sign * df * (1.0 - w) / temp;
            Ok((f, df, ddf))
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "logistic".to_string(),
        }),
    }
}
#[inline]
fn gauss_d(z: f64, case: i32) -> Result<(f64, f64, f64), DistributionError> {
    let f = (-z.powi(2) / 2.0).exp() / SPI;
    match case {
        1 => Ok((f, -z, z.powi(2) - 1.0)),
        2 => {
            let (f0, f1) = if z > 0.0 {
                ((1.0 + erf(z / ROOT_2)) / 2.0, erfc(z / ROOT_2) / 2.0)
            } else {
                (erfc(-z / ROOT_2) / 2.0, (1.0 + erf(-z / ROOT_2)) / 2.0)
            };
            Ok((f0, f1, -z * f))
        }
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "Gaussian".to_string(),
        }),
    }
}
#[inline]
fn exvalue_d(z: f64, case: i32) -> Result<(f64, f64, f64), DistributionError> {
    let w = z.clamp(EXP_CLAMP_MIN, EXP_CLAMP_MAX).exp();
    let temp = (-w).exp();
    match case {
        1 => Ok((w * temp, 1.0 - w, w * (w - 3.0) + 1.0)),
        2 => Ok((1.0 - temp, temp, w * temp * (1.0 - w))),
        _ => Err(DistributionError::InvalidCase {
            case,
            distribution: "extreme value".to_string(),
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
