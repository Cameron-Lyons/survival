//! R's `concordance` (survival 3.8-12): the concordance statistic with its
//! influence-based and Cox-model variances, for right-censored and
//! (start, stop] data, several predictors, strata, case weights, time
//! weights and clustering.
//!
//! `kernels` ports the C sweeps (`concordance3.c`, `concordance5.c`,
//! `fastkm.c`) and the `btree` rank tree; `fit` ports the R-level
//! `concordancefit` that drives them.

pub mod fit;
pub mod kernels;

pub use fit::{
    ConcordanceCounts, ConcordanceFit, ConcordanceOptions, ConcordanceRanks, TimeWeight,
    concordancefit,
};
