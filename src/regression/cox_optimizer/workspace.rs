use super::gram::{GRAM_MIN_VARIABLES, GramWorkspace};
use crate::core::risk_sweep::RiskSetSums;

pub(super) struct EvaluationWorkspace {
    pub eta: Vec<f64>,
    pub risk: Vec<f64>,
    pub risk_set: RiskSetSums,
    pub tied: RiskSetSums,
    /// The Gram evaluation's storage, for right-censored fits with at least
    /// [`GRAM_MIN_VARIABLES`] covariates.
    pub gram: Option<Box<GramWorkspace>>,
}

impl EvaluationWorkspace {
    pub(super) fn new(n: usize, nvar: usize, counting: bool) -> Self {
        Self {
            eta: vec![0.0; n],
            risk: Vec::new(),
            risk_set: RiskSetSums::zeros(if counting { 0 } else { nvar }, true),
            tied: RiskSetSums::zeros(nvar, true),
            gram: (!counting && nvar >= GRAM_MIN_VARIABLES)
                .then(|| Box::new(GramWorkspace::new(n, nvar))),
        }
    }
}
