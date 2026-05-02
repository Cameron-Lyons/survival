#[path = "net_survival.rs"]
pub(crate) mod net_survival_module;
#[path = "relative_survival.rs"]
pub(crate) mod relative_survival_module;

// Public facade exports
pub use net_survival_module::{
    NetSurvivalMethod, NetSurvivalResult, crude_probability_of_death, net_survival,
};
pub use relative_survival_module::{
    ExcessHazardModelResult, RelativeSurvivalResult, excess_hazard_regression, relative_survival,
};
