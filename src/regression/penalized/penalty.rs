use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::lgammafn;
use ndarray::Array2;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use std::sync::Arc;

#[doc = " What a penalty function hands back for one call, R's `coxlist1`/`coxlist2`."]
#[doc = ""]
#[doc = " Lengths are the penalty function's responsibility, exactly as in R, where"]
#[doc = " `coxpenal.fit` checks them before the C code reads the buffers: `coef` and"]
#[doc = " `first` have one entry per penalised coefficient, `second` holds either the"]
#[doc = " diagonal (`p` entries) or the full matrix (`p * p` entries, column-major),"]
#[doc = " and `flag` has one entry (`which == 1`) or one per coefficient."]
#[doc = ""]
#[doc = " The signs are the C code's: `first` and `penalty` are the negatives of"]
#[doc = " the term's derivative and penalty (`coxpenal.fit` negates the `pfun`"]
#[doc = " values so that the kernel can add them to the score and the log"]
#[doc = " likelihood), and `coef` holds the coefficients after any recentring."]
#[pyclass(module = "survival._survival", frozen, get_all, skip_from_py_object)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoxPenaltyTerms {
    #[doc = " Coefficients after any recentring the penalty function applied."]
    pub coef: Vec<f64>,
    #[doc = " First derivative of the penalty."]
    pub first: Vec<f64>,
    #[doc = " Second derivative of the penalty, diagonal or full."]
    pub second: Vec<f64>,
    #[doc = " The penalty's contribution to the log-likelihood."]
    pub penalty: f64,
    #[doc = " \"Force this term to zero\" flags."]
    pub flag: Vec<bool>,
}

crate::internal::pickle::picklable!(CoxPenaltyTerms);

impl CoxPenaltyTerms {
    pub(crate) fn zeros(p: usize, second_len: usize, flags: usize) -> Self {
        Self {
            coef: vec![0.0; p],
            first: vec![0.0; p],
            second: vec![0.0; second_len],
            penalty: 0.0,
            flag: vec![false; flags],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PenaltyValue {
    pub coef: Vec<f64>,
    pub first: Vec<f64>,
    pub second: Vec<f64>,
    pub penalty: f64,
    pub flag: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PsplineMethod {
    Fixed(f64),
    Df(f64),
    Aic,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrailtyFamily {
    Gamma,
    Gaussian,
    T(f64),
}

impl FrailtyFamily {
    pub fn parse(name: &str, tdf: f64) -> SurvivalResult<Self> {
        let families = [Self::Gamma, Self::Gaussian, Self::T(tdf)];
        let exact = families.into_iter().find(|family| family.r_name() == name);
        let partial = || {
            let mut prefixed = families
                .into_iter()
                .filter(|family| !name.is_empty() && family.r_name().starts_with(name));
            match (prefixed.next(), prefixed.next()) {
                (Some(family), None) => Some(family),
                _ => None,
            }
        };
        exact.or_else(partial).ok_or_else(|| {
            SurvivalError::invalid_input(format!("Function 'frailty.{name}' not found"))
        })
    }

    pub fn r_name(self) -> &'static str {
        match self {
            Self::Gamma => "gamma",
            Self::Gaussian => "gaussian",
            Self::T(_) => "t",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrailtyMethod {
    Em,
    Reml,
    Aic,
    Df,
    Fixed,
}

impl FrailtyMethod {
    fn parse(name: &str, allowed: &[Self]) -> SurvivalResult<Self> {
        let lower = name.to_ascii_lowercase();
        let candidates: Vec<Self> = allowed
            .iter()
            .copied()
            .filter(|method| !lower.is_empty() && method.r_name().starts_with(&lower))
            .collect();
        match candidates.as_slice() {
            [method] => Ok(*method),
            _ => Err(SurvivalError::invalid_input(format!(
                "'arg' should be one of {}",
                allowed
                    .iter()
                    .map(|method| format!("\"{}\"", method.r_name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }

    pub fn r_name(self) -> &'static str {
        match self {
            Self::Em => "em",
            Self::Reml => "reml",
            Self::Aic => "aic",
            Self::Df => "df",
            Self::Fixed => "fixed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RidgePenalty {
    pub theta: Option<f64>,
    pub df: Option<f64>,
    pub eps: f64,
    pub scale: bool,
    pub scale_values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PsplinePenalty {
    pub method: PsplineMethod,
    pub nterm: usize,
    pub intercept: bool,
    pub eps: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrailtyPenalty {
    pub distribution: FrailtyFamily,
    pub method: FrailtyMethod,
    pub theta: Option<f64>,
    pub df: Option<f64>,
    pub eps: Option<f64>,
    pub caic: bool,
    pub sparse: bool,
    pub init: Option<Vec<f64>>,
    pub n: Option<usize>,
}

#[cfg(feature = "python")]
#[derive(Debug, Clone)]
pub struct CallbackPenalty {
    pub fexpr: Arc<Py<PyAny>>,
    pub diag: bool,
    pub sparse: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PenaltyTerm {
    Ridge(RidgePenalty),
    Pspline(PsplinePenalty),
    Frailty(FrailtyPenalty),
    #[cfg(feature = "python")]
    #[serde(skip)]
    Callback(CallbackPenalty),
    #[cfg(feature = "python")]
    #[serde(skip)]
    Controlled(super::callbacks::ControlledPenalty),
}

impl PenaltyTerm {
    pub fn ridge(
        theta: Option<f64>,
        df: Option<f64>,
        eps: f64,
        scale: bool,
        scale_values: Option<Vec<f64>>,
    ) -> SurvivalResult<Self> {
        if theta.is_some() && df.is_some() {
            return Err(SurvivalError::invalid_input(
                "Only one of df or theta can be specified",
            ));
        }
        Ok(Self::Ridge(RidgePenalty {
            theta,
            df,
            eps,
            scale,
            scale_values,
        }))
    }

    pub fn pspline(
        df: f64,
        theta: Option<f64>,
        nterm: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        intercept: bool,
    ) -> SurvivalResult<Self> {
        if let Some(name) = method
            && !matches!(name, "aic" | "df" | "fixed")
        {
            return Err(SurvivalError::invalid_input(format!(
                "pspline method must be 'df', 'aic' or 'fixed', got '{name}'"
            )));
        }
        let (method, nterm, eps) = if let Some(theta) = theta {
            if theta <= 0.0 || theta >= 1.0 {
                return Err(SurvivalError::invalid_input("Invalid value for theta"));
            }
            (
                PsplineMethod::Fixed(theta),
                nterm.unwrap_or(2.5 * df),
                eps.unwrap_or(0.1),
            )
        } else if df == 0.0 || method == Some("aic") {
            (PsplineMethod::Aic, 15.0, eps.unwrap_or(1e-5))
        } else {
            if df <= 1.0 {
                return Err(SurvivalError::invalid_input("Too few degrees of freedom"));
            }
            let nterm = nterm.unwrap_or(2.5 * df);
            if df > nterm {
                return Err(SurvivalError::invalid_input(format!(
                    "`nterm' too small for df={df}"
                )));
            }
            (PsplineMethod::Df(df), nterm, eps.unwrap_or(0.1))
        };
        let nterm = nterm.round();
        if nterm < 3.0 {
            return Err(SurvivalError::invalid_input("Too few basis functions"));
        }
        Ok(Self::Pspline(PsplinePenalty {
            method,
            nterm: nterm as usize,
            intercept,
            eps,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn frailty(
        distribution: FrailtyFamily,
        sparse: bool,
        theta: Option<f64>,
        df: Option<f64>,
        eps: Option<f64>,
        method: Option<&str>,
        caic: bool,
        init: Option<Vec<f64>>,
        n: Option<usize>,
    ) -> SurvivalResult<Self> {
        let allowed: &[FrailtyMethod] = match distribution {
            FrailtyFamily::Gamma => &[
                FrailtyMethod::Em,
                FrailtyMethod::Aic,
                FrailtyMethod::Df,
                FrailtyMethod::Fixed,
            ],
            FrailtyFamily::Gaussian => &[
                FrailtyMethod::Reml,
                FrailtyMethod::Aic,
                FrailtyMethod::Df,
                FrailtyMethod::Fixed,
            ],
            FrailtyFamily::T(_) => &[FrailtyMethod::Aic, FrailtyMethod::Df, FrailtyMethod::Fixed],
        };
        if let FrailtyFamily::T(tdf) = distribution
            && tdf <= 2.0
        {
            return Err(SurvivalError::invalid_input(
                "Cannot have df <3 for the t-frailty",
            ));
        }
        let method = match method {
            Some(name) => FrailtyMethod::parse(name, allowed)?,
            None => {
                if theta.is_some() {
                    if df.is_some() {
                        return Err(SurvivalError::invalid_input(
                            "Cannot give both a df and theta argument",
                        ));
                    }
                    FrailtyMethod::Fixed
                } else if let Some(df) = df {
                    if df == 0.0 && distribution != FrailtyFamily::Gamma {
                        FrailtyMethod::Aic
                    } else {
                        FrailtyMethod::Df
                    }
                } else {
                    allowed[0]
                }
            }
        };
        if method == FrailtyMethod::Df && df.is_none() {
            return Err(SurvivalError::invalid_input(
                "Method = df but no df argument",
            ));
        }
        if method == FrailtyMethod::Fixed && theta.is_none() {
            return Err(SurvivalError::invalid_input(
                "Method= fixed but no theta argument",
            ));
        }
        if method != FrailtyMethod::Df && df.is_some() && distribution == FrailtyFamily::Gamma {
            return Err(SurvivalError::invalid_input(
                "Method is not df, but have a df argument",
            ));
        }
        if method != FrailtyMethod::Fixed && theta.is_some() {
            return Err(SurvivalError::invalid_input(
                "Method is not 'fixed', but have a theta argument",
            ));
        }
        if distribution == FrailtyFamily::Gaussian && theta.is_some() && df.is_some() {
            return Err(SurvivalError::invalid_input(
                "Cannot give both a df and theta argument",
            ));
        }
        let eps = match distribution {
            FrailtyFamily::Gaussian => eps,
            _ => Some(eps.unwrap_or(if method == FrailtyMethod::Df {
                0.1
            } else {
                1e-5
            })),
        };
        Ok(Self::Frailty(FrailtyPenalty {
            distribution,
            method,
            theta,
            df,
            eps,
            caic,
            sparse,
            init,
            n,
        }))
    }

    pub fn is_sparse(&self) -> bool {
        match self {
            Self::Ridge(_) | Self::Pspline(_) => false,
            Self::Frailty(term) => term.sparse,
            #[cfg(feature = "python")]
            Self::Callback(term) => term.sparse,
            #[cfg(feature = "python")]
            Self::Controlled(term) => term.sparse,
        }
    }

    pub fn is_diagonal(&self) -> bool {
        match self {
            Self::Ridge(_) | Self::Frailty(_) => true,
            Self::Pspline(_) => false,
            #[cfg(feature = "python")]
            Self::Callback(term) => term.diag,
            #[cfg(feature = "python")]
            Self::Controlled(term) => term.diag,
        }
    }

    pub fn r_name(&self) -> &'static str {
        match self {
            Self::Ridge(_) => "ridge",
            Self::Pspline(_) => "pspline",
            Self::Frailty(_) => "frailty",
            #[cfg(feature = "python")]
            Self::Callback(_) | Self::Controlled(_) => "callback",
        }
    }

    pub(crate) fn pparm(&self, x: &Array2<f64>) -> SurvivalResult<Pparm> {
        Ok(match self {
            Self::Ridge(term) if term.scale => match &term.scale_values {
                Some(values) if values.len() != x.ncols() => {
                    return Err(SurvivalError::invalid_input(format!(
                        "ridge scale_values has {} entries for {} columns",
                        values.len(),
                        x.ncols()
                    )));
                }
                Some(values) => Pparm::Scale(values.clone()),
                None => Pparm::Scale(
                    x.columns()
                        .into_iter()
                        .map(|column| {
                            let n = column.len() as f64;
                            let mean = column.sum() / n;
                            column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
                        })
                        .collect(),
                ),
            },
            Self::Pspline(term) => Pparm::Dmat(difference_penalty(x.ncols(), term.intercept)),
            _ => Pparm::None,
        })
    }

    #[cfg_attr(not(feature = "python"), expect(unused_variables))]
    pub(crate) fn evaluate(
        &self,
        coef: &[f64],
        theta: f64,
        pparm: &Pparm,
        which: i32,
    ) -> SurvivalResult<PenaltyValue> {
        let p = coef.len();
        let unflagged = |coef: Vec<f64>, first: Vec<f64>, second: Vec<f64>, penalty: f64| {
            Ok(PenaltyValue {
                coef,
                first,
                second,
                penalty,
                flag: false,
            })
        };
        let flagged = |coef: &[f64], penalty: f64| {
            Ok(PenaltyValue {
                coef: coef.to_vec(),
                first: Vec::new(),
                second: Vec::new(),
                penalty,
                flag: true,
            })
        };
        match self {
            Self::Ridge(term) => match pparm {
                Pparm::Scale(scale) if term.scale => unflagged(
                    coef.to_vec(),
                    coef.iter().zip(scale).map(|(b, s)| theta * b * s).collect(),
                    scale.iter().map(|s| theta * s).collect(),
                    coef.iter().zip(scale).map(|(b, s)| b * b * s).sum::<f64>() * theta / 2.0,
                ),
                _ => unflagged(
                    coef.to_vec(),
                    coef.iter().map(|b| theta * b).collect(),
                    vec![theta],
                    coef.iter().map(|b| b * b).sum::<f64>() * theta / 2.0,
                ),
            },
            Self::Pspline(_) => {
                let Pparm::Dmat(dmat) = pparm else {
                    return Err(SurvivalError::computation(
                        "pspline term without its penalty matrix",
                    ));
                };
                if theta >= 1.0 {
                    return flagged(coef, 100.0 * (1.0 - theta));
                }
                let lambda = if theta <= 0.0 {
                    0.0
                } else {
                    theta / (1.0 - theta)
                };
                let dcoef: Vec<f64> = (0..p)
                    .map(|i| (0..p).map(|j| dmat[(i, j)] * coef[j]).sum())
                    .collect();
                let quad: f64 = coef.iter().zip(&dcoef).map(|(b, d)| b * d).sum();
                let second = (0..p)
                    .flat_map(|j| (0..p).map(move |i| (i, j)))
                    .map(|(i, j)| dmat[(i, j)] * lambda)
                    .collect();
                unflagged(
                    coef.to_vec(),
                    dcoef.iter().map(|d| d * lambda).collect(),
                    second,
                    quad * lambda / 2.0,
                )
            }
            Self::Frailty(term) => {
                if theta == 0.0 {
                    return flagged(coef, 0.0);
                }
                match term.distribution {
                    FrailtyFamily::Gamma => {
                        let recenter = (coef.iter().map(|b| b.exp()).sum::<f64>() / p as f64).ln();
                        let centred: Vec<f64> = coef.iter().map(|b| b - recenter).collect();
                        let nu = 1.0 / theta;
                        unflagged(
                            centred.clone(),
                            centred.iter().map(|b| (b.exp() - 1.0) * nu).collect(),
                            centred.iter().map(|b| b.exp() * nu).collect(),
                            -centred.iter().sum::<f64>() * nu,
                        )
                    }
                    FrailtyFamily::Gaussian => {
                        let recenter = coef.iter().sum::<f64>() / p as f64;
                        let centred: Vec<f64> = coef.iter().map(|b| b - recenter).collect();
                        unflagged(
                            centred.clone(),
                            centred.iter().map(|b| b / theta).collect(),
                            vec![1.0 / theta; p],
                            0.5 * centred
                                .iter()
                                .map(|b| b * b / theta + (2.0 * std::f64::consts::PI * theta).ln())
                                .sum::<f64>(),
                        )
                    }
                    FrailtyFamily::T(tdf) => {
                        let sig = theta * (tdf - 2.0) / tdf;
                        let denom = tdf * sig;
                        let (mut num, mut den) = (0.0, 0.0);
                        for &b in coef {
                            let temp = 1.0 + b * b / denom;
                            num += b / temp;
                            den += 1.0 / temp - (2.0 / denom) * b * b / (temp * temp);
                        }
                        let recenter = num / den;
                        let centred: Vec<f64> = coef.iter().map(|b| b - recenter).collect();
                        let constant = (tdf + 1.0) / denom;
                        let temp: Vec<f64> = centred.iter().map(|b| 1.0 + b * b / denom).collect();
                        let penalty = temp
                            .iter()
                            .map(|t| {
                                0.5 * (std::f64::consts::PI * denom).ln()
                                    + ((tdf + 1.0) / 2.0) * t.ln()
                                    + lgammafn(tdf / 2.0)
                                    - lgammafn((tdf + 1.0) / 2.0)
                            })
                            .sum();
                        unflagged(
                            centred.to_vec(),
                            centred
                                .iter()
                                .zip(&temp)
                                .map(|(b, t)| constant * b / t)
                                .collect(),
                            centred
                                .iter()
                                .zip(&temp)
                                .map(|(b, t)| {
                                    constant * (1.0 / t - (2.0 / denom) * b * b / (t * t))
                                })
                                .collect(),
                            penalty,
                        )
                    }
                }
            }
            #[cfg(feature = "python")]
            Self::Controlled(_) => unreachable!("controlled penalties use fit-local term state"),
            #[cfg(feature = "python")]
            Self::Callback(term) => {
                let terms = Python::attach(|py| {
                    crate::pybridge::cox_py_callback::evaluate_penalty(
                        term.fexpr.bind(py),
                        which,
                        coef,
                    )
                })
                .map_err(|err| SurvivalError::computation(err.to_string()))?;
                let flag = terms.flag.iter().any(|f| *f);
                Ok(PenaltyValue {
                    coef: terms.coef,
                    first: terms.first.iter().map(|v| -v).collect(),
                    second: terms.second,
                    penalty: -terms.penalty,
                    flag,
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Pparm {
    None,
    Scale(Vec<f64>),
    Dmat(Array2<f64>),
}

fn difference_penalty(ncols: usize, intercept: bool) -> Array2<f64> {
    let nvar = if intercept { ncols } else { ncols + 1 };
    let mut dmat = Array2::zeros((nvar, nvar));
    if nvar >= 3 {
        for r in 0..nvar - 2 {
            let weights = [1.0, -2.0, 1.0];
            for (i, wi) in weights.iter().enumerate() {
                for (j, wj) in weights.iter().enumerate() {
                    dmat[(r + i, r + j)] += wi * wj;
                }
            }
        }
    }
    if intercept {
        dmat
    } else {
        dmat.slice(ndarray::s![1.., 1..]).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(term: &PenaltyTerm, coef: &[f64], theta: f64, x: &Array2<f64>) -> PenaltyValue {
        term.evaluate(coef, theta, &term.pparm(x).unwrap(), 2)
            .unwrap()
    }

    #[test]
    fn ridge_penalty_matches_r() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 8.0, 4.0, 16.0]).unwrap();
        let scaled = PenaltyTerm::ridge(Some(2.0), None, 0.1, true, None).unwrap();
        let out = value(&scaled, &[0.5, -1.0], 2.0, &x);
        let vars = [5.0 / 3.0, 115.0 / 3.0];
        assert!((out.penalty - (0.25 * vars[0] + 1.0 * vars[1])).abs() < 1e-12);
        assert!((out.first[0] - 2.0 * 0.5 * vars[0]).abs() < 1e-12);
        assert!((out.second[1] - 2.0 * vars[1]).abs() < 1e-12);
        assert!(!out.flag);

        let plain = PenaltyTerm::ridge(Some(2.0), None, 0.1, false, None).unwrap();
        let out = value(&plain, &[0.5, -1.0], 2.0, &x);
        assert!((out.penalty - 1.25).abs() < 1e-12);
        assert_eq!(out.first, vec![1.0, -2.0]);
        assert_eq!(out.second, vec![2.0]);
        assert!(PenaltyTerm::ridge(Some(1.0), Some(2.0), 0.1, true, None).is_err());

        let framed = PenaltyTerm::ridge(Some(2.0), None, 0.1, true, Some(vec![3.0, 0.5])).unwrap();
        assert_eq!(framed.pparm(&x).unwrap(), Pparm::Scale(vec![3.0, 0.5]));
        let short = PenaltyTerm::ridge(Some(2.0), None, 0.1, true, Some(vec![3.0])).unwrap();
        assert!(short.pparm(&x).is_err());
    }

    #[test]
    fn pspline_penalty_uses_the_difference_matrix() {
        let term = PenaltyTerm::pspline(4.0, Some(0.5), None, None, None, false).unwrap();
        let PenaltyTerm::Pspline(inner) = &term else {
            panic!("pspline")
        };
        assert_eq!(inner.nterm, 10);
        let x = Array2::zeros((3, 4));
        let Pparm::Dmat(dmat) = term.pparm(&x).unwrap() else {
            panic!("dmat")
        };
        assert_eq!(dmat.shape(), &[4, 4]);
        assert_eq!(dmat[(0, 0)], 5.0);
        assert_eq!(dmat[(0, 1)], -4.0);
        assert_eq!(dmat[(0, 2)], 1.0);
        assert_eq!(dmat[(3, 3)], 1.0);
        let out = value(&term, &[1.0, 0.0, 0.0, 0.0], 0.5, &x);
        assert!((out.penalty - 2.5).abs() < 1e-12);
        assert_eq!(out.first[0], 5.0);
        assert_eq!(out.second.len(), 16);
        let flagged = value(&term, &[1.0, 0.0, 0.0, 0.0], 1.0, &x);
        assert!(flagged.flag);
        assert_eq!(flagged.penalty, 0.0);

        assert!(PenaltyTerm::pspline(1.0, None, None, None, None, false).is_err());
        assert!(PenaltyTerm::pspline(4.0, Some(1.5), None, None, None, false).is_err());
        assert!(PenaltyTerm::pspline(6.0, None, Some(5.0), None, None, false).is_err());
        let aic = PenaltyTerm::pspline(0.0, None, None, None, None, false).unwrap();
        let PenaltyTerm::Pspline(inner) = &aic else {
            panic!("pspline")
        };
        assert_eq!(inner.method, PsplineMethod::Aic);
        assert_eq!(inner.nterm, 15);
        assert_eq!(inner.eps, 1e-5);
    }

    #[test]
    fn frailty_penalties_recentre_and_match_r() {
        let x = Array2::zeros((3, 3));
        let gamma = PenaltyTerm::frailty(
            FrailtyFamily::Gamma,
            true,
            Some(0.5),
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
        let out = value(&gamma, &[0.1, -0.2, 0.3], 0.5, &x);
        let recenter = ((0.1f64.exp() + (-0.2f64).exp() + 0.3f64.exp()) / 3.0).ln();
        assert!((out.coef[0] - (0.1 - recenter)).abs() < 1e-12);
        assert!((out.first[1] - ((-0.2 - recenter).exp() - 1.0) * 2.0).abs() < 1e-12);
        assert!((out.penalty + out.coef.iter().sum::<f64>() * 2.0).abs() < 1e-12);
        assert!(value(&gamma, &[0.1, -0.2, 0.3], 0.0, &x).flag);

        let gauss = PenaltyTerm::frailty(
            FrailtyFamily::Gaussian,
            true,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
        let PenaltyTerm::Frailty(inner) = &gauss else {
            panic!("frailty")
        };
        assert_eq!(inner.method, FrailtyMethod::Reml);
        assert_eq!(inner.eps, None);
        let out = value(&gauss, &[1.0, 2.0, 3.0], 2.0, &x);
        assert_eq!(out.coef, vec![-1.0, 0.0, 1.0]);
        assert_eq!(out.second, vec![0.5; 3]);
        let expected = 0.5 * (2.0 / 2.0 + 3.0 * (4.0 * std::f64::consts::PI).ln());
        assert!((out.penalty - expected).abs() < 1e-12);

        let t = PenaltyTerm::frailty(
            FrailtyFamily::T(5.0),
            true,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
        let PenaltyTerm::Frailty(inner) = &t else {
            panic!("frailty")
        };
        assert_eq!(inner.method, FrailtyMethod::Aic);
        assert_eq!(inner.eps, Some(1e-5));
        let out = value(&t, &[0.0, 0.0, 0.0], 1.0, &x);
        let sig = 0.6;
        let expected =
            3.0 * (0.5 * (std::f64::consts::PI * 5.0 * sig).ln() + lgammafn(2.5) - lgammafn(3.0));
        assert!((out.penalty - expected).abs() < 1e-12);
        assert!((out.second[0] - 6.0 / (5.0 * sig)).abs() < 1e-12);

        assert!(
            PenaltyTerm::frailty(
                FrailtyFamily::T(2.0),
                true,
                None,
                None,
                None,
                None,
                false,
                None,
                None
            )
            .is_err()
        );
        assert!(
            PenaltyTerm::frailty(
                FrailtyFamily::Gamma,
                true,
                Some(1.0),
                Some(2.0),
                None,
                None,
                false,
                None,
                None
            )
            .is_err()
        );
        assert!(
            PenaltyTerm::frailty(
                FrailtyFamily::Gamma,
                true,
                None,
                None,
                None,
                Some("df"),
                false,
                None,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn frailty_distributions_are_pmatched() {
        for (name, family) in [
            ("gamma", "gamma"),
            ("gam", "gamma"),
            ("gau", "gaussian"),
            ("t", "t"),
        ] {
            assert_eq!(FrailtyFamily::parse(name, 5.0).unwrap().r_name(), family);
        }
        assert_eq!(
            FrailtyFamily::parse("t", 7.0).unwrap(),
            FrailtyFamily::T(7.0)
        );
        for name in ["g", "ga", "Gamma", "", "weibull"] {
            let message = FrailtyFamily::parse(name, 5.0).unwrap_err().to_string();
            assert!(message.contains(&format!("Function 'frailty.{name}' not found")));
        }
    }
}
