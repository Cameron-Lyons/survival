pub(crate) mod basic;
pub(crate) mod common;
#[path = "concordance1.rs"]
pub(crate) mod concordance1_module;
pub(crate) mod concordance3;
pub(crate) mod concordance5;

// Public facade exports
pub use basic::concordance as compute_concordance;
pub use concordance1_module::{concordance1, perform_concordance1_calculation};
pub use concordance3::perform_concordance3_calculation;
pub use concordance5::perform_concordance_calculation;
