//! Population rate tables and the routines of R's `survival` package built
//! on them: `ratetable`/`is.ratetable`/`ratetableDate`, `match.ratetable`,
//! the census tables `survexp.us`/`survexp.usr`/`survexp.mn`, `pyears`
//! with `summary.pyears`, and `survexp` with `survexp.fit`.

pub(crate) mod match_ratetable;
pub(crate) mod pyears;
pub(crate) mod pyears_summary;
pub(crate) mod pystep;
pub(crate) mod ratetable;
pub(crate) mod ratetable_data;
pub(crate) mod survexp;
pub(crate) mod survexp_cox;
pub(crate) mod survexp_fit;

pub use match_ratetable::{
    MatchRatetableResult, RatetableColumn, align_us_year_axis, match_ratetable, match_ratetable_py,
};
pub use pyears::{
    PyearsCategories, PyearsExpect, PyearsFollowup, PyearsRatetable, PyearsResult, pyears,
    pyears_py,
};
pub use pyears_summary::{PyearsSummary, PyearsSummaryOptions, summary_pyears, summary_pyears_py};
pub use pystep::{PystepResult, PystepTable, pystep};
pub use ratetable::{
    CalendarDate, DimType, RateTable, RatetableCheck, calendar_to_days, days_to_date, is_leap_year,
    is_ratetable, ratetable_date, ratetable_problems, start_of_year,
};
pub use ratetable_data::{
    survexp_mn, survexp_mn_table, survexp_us, survexp_us_table, survexp_usr, survexp_usr_table,
};
pub use survexp::{SurvExpResult, SurvexpInput, SurvexpMethod, survexp, survexp_py};
pub use survexp_cox::{survexp_cox, survexp_cox_py};
pub use survexp_fit::{SurvexpFit, survexp_fit};
