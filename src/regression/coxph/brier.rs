//! Brier scores of a Cox model's own survival predictions, without the matrix of
//! predictions that the general kernel takes.

use super::{CoxNewData, CoxPHFit};
use crate::error::SurvivalResult;
use crate::internal::validation::validate_finite;
use crate::validation::BrierResult;
use crate::validation::brier::{BrierOutcomes, brier_of_rows};

impl CoxPHFit {
    /// [`crate::validation::brier`] with `phat` the fit's event probabilities,
    /// `1 - predict_survival_at(outcomes.times, newdata)`. Every row is evaluated and
    /// scored in turn, so the times-by-subjects matrix is never stored and the
    /// result's `phat` is empty. Errors are those of the prediction, then the score.
    pub(crate) fn brier_scores(
        &self,
        outcomes: BrierOutcomes<'_>,
        newdata: Option<&CoxNewData>,
    ) -> SurvivalResult<BrierResult> {
        let times = outcomes.times;
        self.check_prediction_dimensions()?;
        validate_finite(times, "times")?;
        let design = self.survival_design(newdata)?;
        let survival = if times.is_empty() {
            None
        } else {
            Some(self.survival_at_times(&design)?)
        };
        brier_of_rows(outcomes, design.x.nrows(), |i, row| {
            let survival = survival
                .as_ref()
                .expect("rows are filled only for requested times");
            survival.fill(times[i], row);
            for value in row.iter_mut() {
                *value = 1.0 - *value;
            }
        })
    }
}
