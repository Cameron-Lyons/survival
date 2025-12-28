#![allow(dead_code)]
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;
const SMALL: f64 = -200.0;
const SPI: f64 = 2.506628274631001;
const ROOT_2: f64 = std::f64::consts::SQRT_2;
#[derive(Error, Debug)]
pub enum DistributionError {
    #[error(
        "Invalid case {case} for {distribution} distribution. Valid cases are 1 (density) and 2 (CDF)"
    )]
    InvalidCase { case: i32, distribution: String },
}
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
        update_derivatives(
            &mut result,
            person,
            fgrp,
            nf,
            nvar,
            nstrat,
            strata,
            covar,
            w,
            dg,
            ddg,
            dsig,
            ddsig,
            dsg,
            sigma,
            sz,
        );
    }
    Ok(result)
}
#[allow(clippy::type_complexity)]
fn compute_exact(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<(f64, f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
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
#[allow(clippy::type_complexity)]
fn compute_right_censored(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<(f64, f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
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
#[allow(clippy::type_complexity)]
fn compute_left_censored(
    z: f64,
    sz: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<(f64, f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
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
#[allow(clippy::type_complexity)]
fn compute_interval_censored(
    z: f64,
    sz: f64,
    time2: f64,
    eta: f64,
    sigma: f64,
    dist: SurvivalDist,
) -> Result<(f64, f64, f64, f64, f64, f64), Box<dyn std::error::Error>> {
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
fn exvalue_d(z: f64, case: i32) -> Result<(f64, f64, f64), DistributionError> {
    let w = z.clamp(-100.0, 100.0).exp();
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
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}
fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn update_derivatives(
    res: &mut SurvivalLikelihood,
    person: usize,
    fgrp: usize,
    nf: usize,
    nvar: usize,
    nstrat: usize,
    strata: usize,
    covar: &ArrayView2<f64>,
    w: f64,
    dg: f64,
    ddg: f64,
    dsig: f64,
    ddsig: f64,
    dsg: f64,
    _sigma: f64,
    _sz: f64,
) {
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
