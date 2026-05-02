pub(crate) mod core;
pub(crate) mod warranty;

// Public facade exports
pub use core::{
    ReliabilityResult, ReliabilityScale, conditional_reliability, failure_probability,
    hazard_to_reliability, mean_residual_life, reliability, reliability_inverse,
};
pub use warranty::{
    ReliabilityGrowthResult, RenewalResult, WarrantyConfig, WarrantyResult, reliability_growth,
    renewal_analysis, warranty_analysis,
};
