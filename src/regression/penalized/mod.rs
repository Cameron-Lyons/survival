//! The penalty machinery shared by `coxpenal.fit` and `survpenal.fit`.
//!
//! R survival fits `ridge()`, `pspline()` and `frailty()` terms in Cox
//! models (`R/coxpenal.fit.R`) and parametric models (`R/survpenal.fit.R`)
//! with the same objects and the same outer loop: [`penalty`] holds the
//! terms and their `pfun`s, [`control`] the `cfun`s that choose each term's
//! smoothing parameter, [`df`] the degrees-of-freedom computation
//! (`coxpenal.df`), [`cholesky3`] the sparse Cholesky routines of the C
//! kernels and [`terms`] the pieces of the outer loop.

pub(crate) mod cholesky3;
pub(crate) mod control;
pub(crate) mod df;
pub(crate) mod penalty;
pub(crate) mod terms;

#[cfg(feature = "python")]
pub use penalty::CallbackPenalty;
pub use penalty::{
    CoxPenaltyTerms, FrailtyFamily, FrailtyMethod, FrailtyPenalty, PenaltyTerm, PsplineMethod,
    PsplinePenalty, RidgePenalty,
};
pub use terms::{ModelTerm, PenaltyHistory};
