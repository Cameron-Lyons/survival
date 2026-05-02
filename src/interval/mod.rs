pub(crate) mod interval_censoring;

// Public facade exports
pub use interval_censoring::{
    IntervalCensoredResult, IntervalDistribution, TurnbullResult, interval_censored_regression,
    npmle_interval, turnbull_estimator,
};
