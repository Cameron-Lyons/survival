pub(crate) mod pyears_summary;
pub(crate) mod ratetable;
#[path = "survexp.rs"]
pub(crate) mod survexp_module;
#[path = "survexp_us.rs"]
pub(crate) mod survexp_us_module;

// Public facade exports
pub use pyears_summary::{PyearsCell, PyearsSummary, pyears_by_cell, pyears_ci, summary_pyears};
pub use ratetable::{
    DimType, RateDimension, RateTable, RatetableDateResult, create_simple_ratetable, days_to_date,
    is_ratetable, ratetable_date,
};
pub use survexp_module::{SurvExpResult, survexp, survexp_individual};
pub use survexp_us_module::{
    ExpectedSurvivalResult, compute_expected_survival, survexp_mn, survexp_us, survexp_usr,
};
