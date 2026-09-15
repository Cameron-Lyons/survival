//! The penalty terms of a penalised Cox model: R survival's `ridge()`
//! (`R/ridge.R`), `pspline()` (the penalty side of `R/pspline.R`; the basis
//! is `core::pspline_basis`) and `frailty()` (`R/frailty.R`,
//! `R/frailty.gamma.R`, `R/frailty.gaussian.R`, `R/frailty.t.R`).
//!
//! Each R term is a `coxph.penalty` object carrying a `pfun` (the penalty
//! and its first two derivatives at the current coefficients, evaluated once
//! per Newton iteration), a `cfun` (the outer-loop rule that chooses the
//! next smoothing parameter `theta`, see [`super::control`]), the `cparm`
//! and `pparm` those functions read, and the `diag`/`sparse` flags telling
//! `coxpenal.fit` how the second derivative is stored.  [`PenaltyTerm`]
//! holds that information as data; [`PenaltyTerm::evaluate`] is the `pfun`.
//!
//! With the `python` feature a user-defined penalty can be supplied as a
//! Python callable ([`PenaltyTerm::Callback`]) following the contract of
//! `cox_callback`: it returns the C-level `coxlist` (R's `cox_Rcallback.c`)
//! for its coefficients.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::dist::lgammafn;
use ndarray::Array2;
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::sync::Arc;

/// What a penalty function hands back for one call, R's `coxlist1`/`coxlist2`.
///
/// Lengths are the penalty function's responsibility, exactly as in R, where
/// `coxpenal.fit` checks them before the C code reads the buffers: `coef` and
/// `first` have one entry per penalised coefficient, `second` holds either the
/// diagonal (`p` entries) or the full matrix (`p * p` entries, column-major),
/// and `flag` has one entry (`which == 1`) or one per coefficient.
///
/// The signs are the C code's: `first` and `penalty` are the negatives of
/// the term's derivative and penalty (`coxpenal.fit` negates the `pfun`
/// values so that the kernel can add them to the score and the log
/// likelihood), and `coef` holds the coefficients after any recentring.
#[pyclass(frozen, get_all, skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub struct CoxPenaltyTerms {
    /// Coefficients after any recentring the penalty function applied.
    pub coef: Vec<f64>,
    /// First derivative of the penalty.
    pub first: Vec<f64>,
    /// Second derivative of the penalty, diagonal or full.
    pub second: Vec<f64>,
    /// The penalty's contribution to the log-likelihood.
    pub penalty: f64,
    /// "Force this term to zero" flags.
    pub flag: Vec<bool>,
}

impl CoxPenaltyTerms {
    /// R's initial `coxlist1`/`coxlist2`: zeros, nothing flagged.
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

/// The value of a term's `pfun` at one set of coefficients (R's
/// `list(recenter, first, second, penalty, flag)`), on the R side of the
/// sign convention: `penalty` is subtracted from the log likelihood and
/// `first` is its gradient.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PenaltyValue {
    /// The coefficients after the term recentred them (`coef - recenter`),
    /// or the input coefficients when the term does not recentre.
    pub coef: Vec<f64>,
    /// Gradient of the penalty; unused (and stale in R) when `flag` is set.
    pub first: Vec<f64>,
    /// The diagonal (length `p`, or a single value R recycles) or the full
    /// column-major matrix (`p * p`) of second derivatives.
    pub second: Vec<f64>,
    pub penalty: f64,
    /// `TRUE` forces the term's coefficients to zero this iteration (a
    /// frailty with `theta = 0`, a `pspline` with `theta >= 1`).
    pub flag: bool,
}

/// How `pspline()` chooses its smoothing parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PsplineMethod {
    /// `theta` given: no calibration (`0 < theta < 1`).
    Fixed(f64),
    /// Calibrate `theta` to the target degrees of freedom
    /// (`frailty.controldf`).
    Df(f64),
    /// Minimise the AIC (`df = 0` or `method = "aic"`, `frailty.controlaic`).
    Aic,
}

/// The distribution of a `frailty()` term (R's `distribution` argument).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrailtyFamily {
    Gamma,
    Gaussian,
    /// Student t with `tdf` degrees of freedom (`frailty.t(tdf = )`).
    T(f64),
}

impl FrailtyFamily {
    /// `frailty(distribution = )` with R's partial matching of `"gamma"`,
    /// `"gaussian"` and `"t"`.
    pub fn parse(name: &str, tdf: f64) -> SurvivalResult<Self> {
        let lower = name.to_ascii_lowercase();
        let matches = |full: &str| !lower.is_empty() && full.starts_with(&lower);
        if matches("gamma") {
            Ok(Self::Gamma)
        } else if matches("gaussian") {
            Ok(Self::Gaussian)
        } else if matches("t") {
            Ok(Self::T(tdf))
        } else {
            Err(SurvivalError::invalid_input(format!(
                "Function 'frailty.{name}' not found"
            )))
        }
    }
}

/// How a `frailty()` term chooses its variance `theta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrailtyMethod {
    /// Gamma only: the EM-equivalent profile likelihood (`frailty.controlgam`).
    Em,
    /// Gaussian only: REML (`frailty.controlgauss`).
    Reml,
    /// Minimise the (corrected) AIC (`frailty.controlaic`).
    Aic,
    /// Calibrate to a target degrees of freedom (`frailty.controldf`).
    Df,
    /// `theta` given.
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

/// `ridge(..., theta, df = nvar/2, eps = .1, scale = TRUE)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RidgePenalty {
    /// A fixed `theta`; `None` calibrates to `df`.
    pub theta: Option<f64>,
    /// Target degrees of freedom; `None` is R's default `nvar / 2`.
    pub df: Option<f64>,
    pub eps: f64,
    /// Scale the penalty of each column by its variance (`pparm`; R takes
    /// the variance over the model frame before `na.action`, this port over
    /// the rows fitted).
    pub scale: bool,
}

/// The penalty side of `pspline(x, df = 4, theta, nterm = 2.5 * df, eps = .1,
/// method, intercept = FALSE, ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PsplinePenalty {
    pub method: PsplineMethod,
    /// Number of interior knot intervals; with the degree it sets the
    /// basis size (`nterm + degree`, minus the dropped intercept column).
    pub nterm: usize,
    /// Whether the basis keeps its first column (R's `intercept = TRUE`).
    pub intercept: bool,
    pub eps: f64,
}

/// `frailty(x, distribution, sparse, theta, df, eps, method, tdf, ...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct FrailtyPenalty {
    pub distribution: FrailtyFamily,
    pub method: FrailtyMethod,
    /// The variance of the random effect (`method = "fixed"`).
    pub theta: Option<f64>,
    /// Target degrees of freedom (`method = "df"`); the search starts from
    /// `3 * df / n` (R's `guess`, with `n` the model-frame length).
    pub df: Option<f64>,
    /// Convergence tolerance of the outer iteration; `None` is R's fallback
    /// to `sqrt(coxph.control()$eps)` (the `eps2` entry `coxpenal.fit`
    /// appends to every `cparm`, reached through partial matching when the
    /// term itself sets no `eps`, as `frailty.gaussian` never does).
    pub eps: Option<f64>,
    /// Use the corrected AIC for `method = "aic"` (`caic = TRUE`).
    pub caic: bool,
    /// Sparse handling: the term is a single column of group codes fitted
    /// through the frailty kernel; otherwise its columns are the indicator
    /// matrix of the groups.
    pub sparse: bool,
    /// Starting values of `theta` for the `em` and `reml` searches (R's
    /// `init` passed through `...`).
    pub init: Option<Vec<f64>>,
}

/// A user-defined penalty evaluated in Python (the `cox_callback` contract):
/// `fexpr(coef, which=which)` returns the C-level `coxlist` for the term's
/// coefficients.  `theta` is the callable's own business, so the outer loop
/// treats the term as fixed.
#[cfg(feature = "python")]
#[derive(Debug, Clone)]
pub struct CallbackPenalty {
    pub fexpr: Arc<Py<PyAny>>,
    /// `second` is the diagonal (`true`) or the full matrix.
    pub diag: bool,
    pub sparse: bool,
}

/// One penalised term of the model: which R penalty it is and its arguments.
#[derive(Debug, Clone)]
pub enum PenaltyTerm {
    Ridge(RidgePenalty),
    Pspline(PsplinePenalty),
    Frailty(FrailtyPenalty),
    #[cfg(feature = "python")]
    Callback(CallbackPenalty),
}

impl PenaltyTerm {
    /// `ridge(..., theta, df, eps, scale)`: exactly one of `theta` and `df`
    /// may be given.
    pub fn ridge(
        theta: Option<f64>,
        df: Option<f64>,
        eps: f64,
        scale: bool,
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
        }))
    }

    /// `pspline(df, theta, nterm, eps, method)` argument resolution: `theta`
    /// fixes the smoothing parameter, `df = 0` or `method = "aic"` switches
    /// to the AIC search (with `nterm = 15` and `eps = 1e-5` unless given),
    /// otherwise `theta` is calibrated to `df` (`nterm = round(2.5 * df)`).
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

    /// `frailty(x, distribution, sparse, theta, df, eps, method, tdf, caic,
    /// init)`: the argument checks of `frailty.gamma`, `frailty.gaussian`
    /// and `frailty.t`.  `method = None` is R's missing `method`: `theta`
    /// gives `fixed`, `df` gives `df` (`aic` for `df = 0` outside the gamma
    /// family), else the family's default.
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
        // The R defaults of `eps`: 1e-5 for the gamma and t families, 0.1
        // for their `df` method when eps was not given; frailty.gaussian
        // has no eps argument at all.
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
        }))
    }

    /// The term is fitted through the sparse frailty kernel.
    pub fn is_sparse(&self) -> bool {
        match self {
            Self::Ridge(_) | Self::Pspline(_) => false,
            Self::Frailty(term) => term.sparse,
            #[cfg(feature = "python")]
            Self::Callback(term) => term.sparse,
        }
    }

    /// R's `diag` attribute: the second derivative is a diagonal.
    pub fn is_diagonal(&self) -> bool {
        match self {
            Self::Ridge(_) | Self::Frailty(_) => true,
            Self::Pspline(_) => false,
            #[cfg(feature = "python")]
            Self::Callback(term) => term.diag,
        }
    }

    /// The R function the term comes from.
    pub fn r_name(&self) -> &'static str {
        match self {
            Self::Ridge(_) => "ridge",
            Self::Pspline(_) => "pspline",
            Self::Frailty(_) => "frailty",
            #[cfg(feature = "python")]
            Self::Callback(_) => "callback",
        }
    }

    /// The term's `pparm`, computed from its design columns (`x`, `n x p`):
    /// the column variances of a scaled `ridge`, the difference-penalty
    /// matrix of a `pspline`.
    pub(crate) fn pparm(&self, x: &Array2<f64>) -> Pparm {
        match self {
            Self::Ridge(term) if term.scale => Pparm::Scale(
                x.columns()
                    .into_iter()
                    .map(|column| {
                        // R's var(): the n - 1 denominator.
                        let n = column.len() as f64;
                        let mean = column.sum() / n;
                        column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
                    })
                    .collect(),
            ),
            Self::Pspline(term) => Pparm::Dmat(difference_penalty(x.ncols(), term.intercept)),
            _ => Pparm::None,
        }
    }

    /// The term's `pfun` at `coef` for the smoothing parameter `theta`
    /// (R's `n.eff` argument is unused by every built-in penalty); `which`
    /// (1 sparse, 2 dense) is handed to a callback penalty.
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
                // R's c(dmat * lambda): column-major.
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
                        // Scale constant squared of the density, then one
                        // Newton-Raphson step towards the centring MLE.
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
            Self::Callback(term) => {
                let terms = Python::attach(|py| {
                    crate::pybridge::cox_py_callback::evaluate_penalty(
                        term.fexpr.bind(py),
                        which,
                        coef,
                    )
                })
                .map_err(|err| SurvivalError::computation(err.to_string()))?;
                // The callback speaks the C-level convention; the
                // composition in `coxpenal.fit` re-applies the negation.
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

/// A term's `pparm`: the extra argument of its `pfun`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Pparm {
    None,
    /// `ridge(scale = TRUE)`: the variance of each column.
    Scale(Vec<f64>),
    /// `pspline`: the second-difference penalty matrix `D'D`.
    Dmat(Array2<f64>),
}

/// `pspline`'s penalty matrix for `ncols` design columns: `D'D` for the
/// second-difference operator over the full basis (`ncols + 1` functions
/// when the intercept column was dropped), with the intercept row and
/// column removed.
fn difference_penalty(ncols: usize, intercept: bool) -> Array2<f64> {
    let nvar = if intercept { ncols } else { ncols + 1 };
    let mut dmat = Array2::zeros((nvar, nvar));
    if nvar >= 3 {
        // Row r of D is (1, -2, 1) at columns r, r + 1, r + 2.
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
        term.evaluate(coef, theta, &term.pparm(x), 2).unwrap()
    }

    #[test]
    fn ridge_penalty_matches_r() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 4.0, 3.0, 8.0, 4.0, 16.0]).unwrap();
        let scaled = PenaltyTerm::ridge(Some(2.0), None, 0.1, true).unwrap();
        let out = value(&scaled, &[0.5, -1.0], 2.0, &x);
        let vars = [5.0 / 3.0, 115.0 / 3.0];
        assert!((out.penalty - (0.25 * vars[0] + 1.0 * vars[1])).abs() < 1e-12);
        assert!((out.first[0] - 2.0 * 0.5 * vars[0]).abs() < 1e-12);
        assert!((out.second[1] - 2.0 * vars[1]).abs() < 1e-12);
        assert!(!out.flag);

        let plain = PenaltyTerm::ridge(Some(2.0), None, 0.1, false).unwrap();
        let out = value(&plain, &[0.5, -1.0], 2.0, &x);
        assert!((out.penalty - 1.25).abs() < 1e-12);
        assert_eq!(out.first, vec![1.0, -2.0]);
        assert_eq!(out.second, vec![2.0]);
        assert!(PenaltyTerm::ridge(Some(1.0), Some(2.0), 0.1, true).is_err());
    }

    #[test]
    fn pspline_penalty_uses_the_difference_matrix() {
        let term = PenaltyTerm::pspline(4.0, Some(0.5), None, None, None, false).unwrap();
        let PenaltyTerm::Pspline(inner) = &term else {
            panic!("pspline")
        };
        assert_eq!(inner.nterm, 10);
        let x = Array2::zeros((3, 4));
        let Pparm::Dmat(dmat) = term.pparm(&x) else {
            panic!("dmat")
        };
        // Full 5 x 5 D'D with the first row/column dropped.
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
                None
            )
            .is_err()
        );
        assert!(FrailtyFamily::parse("gau", 5.0).is_ok());
        assert!(FrailtyFamily::parse("weibull", 5.0).is_err());
    }
}
