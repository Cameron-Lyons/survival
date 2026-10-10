//! Reusable buffers for Breslow and Efron likelihood evaluations.
//!
//! Their lifetime follows the fit, while the evaluator resets sums and
//! overwrites predictors on every call, including rejected Newton steps.

use crate::core::risk_sweep::RiskSetSums;

pub(super) struct EvaluationWorkspace {
    pub eta: Vec<f64>,
    /// Precomputed weighted risks for the counting-process fast path.
    /// Allocated only when that path is first used; the right-censored
    /// sweep computes each risk when its row joins.
    pub risk: Vec<f64>,
    /// Right-censored running sums. Counting-process evaluations reuse
    /// the fitter's recentred sums for both the fast and general paths.
    pub risk_set: RiskSetSums,
    pub tied: RiskSetSums,
}

impl EvaluationWorkspace {
    pub(super) fn new(n: usize, nvar: usize, counting: bool) -> Self {
        Self {
            eta: vec![0.0; n],
            risk: Vec::new(),
            risk_set: RiskSetSums::zeros(if counting { 0 } else { nvar }, true),
            tied: RiskSetSums::zeros(nvar, true),
        }
    }
}
