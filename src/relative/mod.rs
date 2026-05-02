pub(crate) mod net_survival;
pub(crate) mod relative_survival;

// Public facade exports
pub use net_survival::{
    NetSurvivalMethod, NetSurvivalResult, crude_probability_of_death, net_survival,
};
pub use relative_survival::{
    ExcessHazardModelResult, RelativeSurvivalResult, excess_hazard_regression, relative_survival,
};
