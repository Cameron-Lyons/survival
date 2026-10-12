mod bindings;
mod curves;
mod fitting;
mod prediction;
mod types;

pub use bindings::coxph_fit;
pub use types::{
    Basehaz, CoxNewData, CoxPHFit, CoxPrediction, CoxSurvfitCurve, CoxTermsPrediction, CoxphData,
    CoxphOptions, PredictReference, SurvfitOptions,
};

pub(crate) use bindings::newdata_from_python;
#[cfg(test)]
pub(crate) use fitting::linear_predictor_concordance;
pub(crate) use fitting::{
    FittedCox, add_offset_mean, centre_offset, fit_cox_engine, nocenter_columns,
};
pub(crate) use prediction::{default_assign, validate_assign};
pub(crate) use types::SortedRows;

#[cfg(test)]
mod input_tests;
#[cfg(test)]
mod tests;
