//! R's `ratetable` object (`?ratetable`, `R/ratetable.R`,
//! `R/is.ratetable.R`, `R/summary.ratetable.R`, `R/ratetableDate.R`).
//!
//! A rate table is a multi-way array of hazard rates together with, per
//! dimension, a name (`dimid`), labels (`dimnames`), a type code and, for
//! non-factor dimensions, the lower cutpoint of every category.  Types follow
//! `?ratetable`: 1 = factor, 2 = continuous (age in days), 3 = date, 4 = the
//! calendar-year axis of the US census tables (see [`DimType`]).  Date
//! cutpoints are stored the way `ratetableDate` returns them: days since
//! 1970-01-01, which is how R's `Date` class counts.  Rates are stored in
//! R's column-major order (the first index varies fastest).

use crate::error::{SurvivalError, SurvivalResult};
use ndarray::Array2;
use pyo3::prelude::*;
use std::fmt;

/// R's `type` attribute of a `ratetable` dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int, from_py_object)]
pub enum DimType {
    /// Type 1: a categorical dimension, indexed by level (`sex`, `race`).
    Factor = 1,
    /// Type 2: a continuous, time-advancing dimension, usually age in days.
    Continuous = 2,
    /// Type 3: a calendar-date dimension (days since 1970-01-01).
    Date = 3,
    /// Type 4: the calendar-year axis of the US census tables, whose rates
    /// change on the subject's birthday rather than on January 1st.
    UsYear = 4,
}

impl DimType {
    /// R's integer type code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// Parse R's integer type code.
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            1 => Some(Self::Factor),
            2 => Some(Self::Continuous),
            3 => Some(Self::Date),
            4 => Some(Self::UsYear),
            _ => None,
        }
    }

    /// Whether the dimension advances with follow-up time.
    pub fn is_time_based(self) -> bool {
        self != Self::Factor
    }
}

/// Outcome of R's `is.ratetable(x, verbose = TRUE)`: an empty `messages`
/// list means the structure is a valid rate table.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct RatetableCheck {
    #[pyo3(get)]
    pub valid: bool,
    #[pyo3(get)]
    pub messages: Vec<String>,
}

/// A rate table with R's attributes.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct RateTable {
    /// `dim(x)`.
    #[pyo3(get)]
    pub dims: Vec<usize>,
    /// `names(dimnames(x))` (R's `dimid`), one per dimension.
    #[pyo3(get)]
    pub dimid: Vec<String>,
    /// `dimnames(x)`, one label vector per dimension.
    #[pyo3(get)]
    pub dimnames: Vec<Vec<String>>,
    /// `attr(x, "cutpoints")`: lower bounds per dimension, `None` for factor
    /// dimensions (R `NULL`).  Dates are days since 1970-01-01.
    #[pyo3(get)]
    pub cutpoints: Vec<Option<Vec<f64>>>,
    /// `attr(x, "type")` per dimension.
    #[pyo3(get)]
    pub types: Vec<DimType>,
    /// The hazard rates in column-major order, `dims.iter().product()` long.
    #[pyo3(get)]
    pub rates: Vec<f64>,
}

impl fmt::Display for RateTable {
    /// The text of R's `summary.ratetable` (`R/summary.ratetable.R`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, " Rate table with {} dimensions:", self.dims.len())?;
        for (d, name) in self.dimid.iter().enumerate() {
            let cuts = self.cutpoints[d].as_deref().unwrap_or_default();
            let first = cuts.first().copied().unwrap_or(f64::NAN);
            let last = cuts.last().copied().unwrap_or(f64::NAN);
            match self.types[d] {
                DimType::Factor => {
                    writeln!(f, "\t{name} has levels of: {}", self.dimnames[d].join(" "))?;
                }
                DimType::Continuous => writeln!(
                    f,
                    "\t{name} ranges from {first} to {last}; with {} categories",
                    self.dims[d]
                )?,
                DimType::Date | DimType::UsYear => writeln!(
                    f,
                    "\t{name} ranges from {} to {}; with {} categories",
                    days_to_date(first),
                    days_to_date(last),
                    self.dims[d]
                )?,
            }
        }
        Ok(())
    }
}

/// R's `is.ratetable(x, verbose = TRUE)` structural checks
/// (`R/is.ratetable.R`), reported as its messages; an empty list is R's
/// `TRUE`.  `types` are R's integer codes so that malformed codes can be
/// reported rather than rejected up front.
pub fn ratetable_problems(
    dims: &[usize],
    dimid: &[String],
    dimnames: &[Vec<String>],
    cutpoints: &[Option<Vec<f64>>],
    types: &[i64],
    n_rates: usize,
) -> Vec<String> {
    let mut msg = Vec::new();
    let nd = dims.len();
    let expected: usize = dims.iter().product();
    if nd == 0 {
        msg.push("missing attribute: dim".to_string());
    }
    if n_rates != expected {
        msg.push("length of the data does not match prod(dim)".to_string());
    }
    if dimnames.len() != nd {
        msg.push("wrong length for dimnames".to_string());
    }
    if dimid.len() != nd {
        msg.push("wrong length for dimid, or dimnames do not have names".to_string());
    }
    if dimid.iter().any(|id| id.is_empty()) {
        msg.push("one of the dimnames identifiers is blank".to_string());
    }
    if cutpoints.len() != nd {
        msg.push("wrong length for cutpoints".to_string());
    }
    if types.iter().any(|&t| DimType::from_code(t).is_none()) {
        msg.push("type attribute must be 1, 2, 3, or 4".to_string());
    }
    if types.len() != nd {
        msg.push("wrong length for type attribute".to_string());
    }
    if types.iter().filter(|&&t| t == 4).count() > 1 {
        msg.push("two dimenesions idenitied as US ratetable years".to_string());
    }
    for i in 0..nd.min(types.len()) {
        let n = dims[i];
        let one_based = i + 1;
        if let Some(labels) = dimnames.get(i)
            && labels.len() != n
        {
            msg.push(format!("dimname {one_based} is the wrong length"));
        }
        let Some(dim_type) = DimType::from_code(types[i]) else {
            continue;
        };
        let cuts = cutpoints.get(i).map(Option::as_deref).unwrap_or_default();
        if dim_type.is_time_based() {
            match cuts {
                Some(values) if values.len() == n => {
                    if values.iter().any(|v| !v.is_finite()) {
                        msg.push(format!("cutpoints {one_based} must be finite"));
                    } else if values.windows(2).any(|w| w[1] <= w[0]) {
                        msg.push(format!("unsorted cutpoints for dimension {one_based}"));
                    }
                }
                _ => msg.push(format!("wrong length for cutpoints {one_based}")),
            }
        } else if cuts.is_some() {
            msg.push(format!(
                "attribute type[{one_based}] is continuous; cutpoint should be null"
            ));
        }
    }
    msg
}

impl RateTable {
    /// Build and validate a rate table from R's attributes.
    pub fn try_new(
        dims: Vec<usize>,
        dimid: Vec<String>,
        dimnames: Vec<Vec<String>>,
        cutpoints: Vec<Option<Vec<f64>>>,
        types: Vec<DimType>,
        rates: Vec<f64>,
    ) -> SurvivalResult<Self> {
        let codes: Vec<i64> = types.iter().map(|t| i64::from(t.code())).collect();
        let problems =
            ratetable_problems(&dims, &dimid, &dimnames, &cutpoints, &codes, rates.len());
        if !problems.is_empty() {
            return Err(SurvivalError::invalid_input(format!(
                "not a valid ratetable: {}",
                problems.join("; ")
            )));
        }
        if rates.iter().any(|r| !r.is_finite() || *r < 0.0) {
            return Err(SurvivalError::invalid_input(
                "ratetable rates must be finite and non-negative",
            ));
        }
        Ok(Self {
            dims,
            dimid,
            dimnames,
            cutpoints,
            types,
            rates,
        })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Zero-based position of `label` along dimension `dim` (R's
    /// `match(label, dimnames(x)[[dim]])`).
    pub fn level(&self, dim: usize, label: &str) -> Option<usize> {
        self.dimnames.get(dim)?.iter().position(|l| l == label)
    }

    /// The rate at a zero-based multi-index, e.g. `[50, 0, 60]` for
    /// `survexp.us["50", "male", "2000"]`.
    pub fn rate(&self, index: &[usize]) -> Option<f64> {
        if index.len() != self.dims.len() {
            return None;
        }
        let mut offset = 0;
        let mut stride = 1;
        for (&i, &n) in index.iter().zip(&self.dims) {
            if i >= n {
                return None;
            }
            offset += i * stride;
            stride *= n;
        }
        self.rates.get(offset).copied()
    }

    /// Position of the type-4 (US calendar year) dimension, if any.
    pub fn us_year_dimension(&self) -> Option<usize> {
        self.types.iter().position(|t| *t == DimType::UsYear)
    }

    /// R's `rfac <- 1*(atts$type == 1)`: the factor flags handed to the C
    /// person-years code.
    pub fn factor_flags(&self) -> Vec<i32> {
        self.types
            .iter()
            .map(|t| i32::from(*t == DimType::Factor))
            .collect()
    }

    /// The per-dimension cutpoints as ragged slices, empty for factor
    /// dimensions, in the layout `pystep` expects.
    pub fn cut_slices(&self) -> Vec<&[f64]> {
        self.cutpoints
            .iter()
            .map(|c| c.as_deref().unwrap_or_default())
            .collect()
    }

    /// Check a matrix of starting positions in the table (one row per
    /// subject, one column per dimension in the table's order, the `R` of
    /// `match.ratetable`): every entry is finite and, on a factor
    /// dimension, an integer level subscript between 1 and the number of
    /// levels, the invariants `match.ratetable` establishes and the C
    /// person-years code relies on when it indexes the rate array.
    pub fn validate_positions(&self, positions: &Array2<f64>) -> SurvivalResult<()> {
        if positions.ncols() != self.ndim() {
            return Err(SurvivalError::invalid_input(format!(
                "ratetable positions must have one column per dimension: {} expected, got {}",
                self.ndim(),
                positions.ncols()
            )));
        }
        for (dim, column) in positions.columns().into_iter().enumerate() {
            let dimid = &self.dimid[dim];
            if column.iter().any(|v| v.is_nan()) {
                return Err(SurvivalError::invalid_input(format!(
                    "The variable {dimid} contains missing values"
                )));
            }
            if column.iter().any(|v| v.is_infinite()) {
                return Err(SurvivalError::invalid_input(format!(
                    "The variable {dimid} must be finite"
                )));
            }
            if self.types[dim] == DimType::Factor {
                let n_levels = self.dims[dim] as f64;
                if column
                    .iter()
                    .any(|&v| v.fract() != 0.0 || v <= 0.0 || v > n_levels)
                {
                    return Err(SurvivalError::invalid_input(format!(
                        "The variable {dimid} is out of range"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl RateTable {
    /// Construct from R's attributes; `types` are R's integer codes 1-4.
    #[new]
    #[pyo3(signature = (dims, dimid, dimnames, cutpoints, types, rates))]
    fn new(
        dims: Vec<usize>,
        dimid: Vec<String>,
        dimnames: Vec<Vec<String>>,
        cutpoints: Vec<Option<Vec<f64>>>,
        types: Vec<i64>,
        rates: Vec<f64>,
    ) -> PyResult<Self> {
        let types = types
            .iter()
            .map(|&code| {
                DimType::from_code(code).ok_or_else(|| {
                    SurvivalError::invalid_input("type attribute must be 1, 2, 3, or 4")
                })
            })
            .collect::<SurvivalResult<Vec<_>>>()?;
        Ok(Self::try_new(
            dims, dimid, dimnames, cutpoints, types, rates,
        )?)
    }

    /// R's integer type codes, one per dimension.
    fn type_codes(&self) -> Vec<i64> {
        self.types.iter().map(|t| i64::from(t.code())).collect()
    }

    /// The text of R's `summary.ratetable`.
    fn __str__(&self) -> String {
        self.to_string()
    }

    /// The rate at a zero-based multi-index (`None` when out of range).
    #[pyo3(name = "rate")]
    fn rate_py(&self, index: Vec<usize>) -> Option<f64> {
        self.rate(&index)
    }
}

/// R's `is.ratetable(x, verbose = TRUE)` on raw attributes (`types` are
/// R's integer codes).  The `valid` flag is R's non-verbose result.
#[pyfunction]
#[pyo3(signature = (dims, dimid, dimnames, cutpoints, types, n_rates))]
pub fn is_ratetable(
    dims: Vec<usize>,
    dimid: Vec<String>,
    dimnames: Vec<Vec<String>>,
    cutpoints: Vec<Option<Vec<f64>>>,
    types: Vec<i64>,
    n_rates: usize,
) -> RatetableCheck {
    let messages = ratetable_problems(&dims, &dimid, &dimnames, &cutpoints, &types, n_rates);
    RatetableCheck {
        valid: messages.is_empty(),
        messages,
    }
}

/// A proleptic Gregorian calendar date, the components of an R `Date`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(from_py_object)]
pub struct CalendarDate {
    #[pyo3(get)]
    pub year: i32,
    #[pyo3(get)]
    pub month: u32,
    #[pyo3(get)]
    pub day: u32,
}

impl fmt::Display for CalendarDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

const DAYS_BEFORE_MONTH: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
/// Days from 0000-03-01 to 1970-01-01 in the proleptic Gregorian calendar.
const EPOCH_DAY_NUMBER: i64 = 719_468;
/// Days from 0001-01-01 to 1970-01-01.
const EPOCH_FROM_YEAR_ONE: i64 = 719_162;

/// Whether `year` is a leap year in the proleptic Gregorian calendar.
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in a month.
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
    }
}

/// Days from 1970-01-01 to `year-month-day`, i.e. `as.numeric(as.Date(...))`.
fn day_number(year: i32, month: u32, day: u32) -> i64 {
    let previous_year = i64::from(year) - 1;
    let days_before_year = 365 * previous_year + previous_year.div_euclid(4)
        - previous_year.div_euclid(100)
        + previous_year.div_euclid(400);
    let leap_day = i64::from(month > 2 && is_leap_year(year));
    days_before_year + DAYS_BEFORE_MONTH[(month - 1) as usize] + i64::from(day - 1) + leap_day
        - EPOCH_FROM_YEAR_ONE
}

/// Days since 1970-01-01 of a calendar date.
pub fn calendar_to_days(date: CalendarDate) -> SurvivalResult<i64> {
    if !(1..=12).contains(&date.month) {
        return Err(SurvivalError::invalid_input(
            "month must be between 1 and 12",
        ));
    }
    if date.day < 1 || date.day > days_in_month(date.year, date.month) {
        return Err(SurvivalError::invalid_input(
            "day is invalid for the given month and year",
        ));
    }
    Ok(day_number(date.year, date.month, date.day))
}

/// R's `ratetableDate` for a calendar date: the number of days since
/// 1970-01-01, which is how `Date` objects are stored (`R/ratetableDate.R`).
/// Numeric values pass through `ratetableDate` unchanged, so callers holding
/// day counts need no conversion.
#[pyfunction]
#[pyo3(signature = (year, month=1, day=1))]
pub fn ratetable_date(year: i32, month: u32, day: u32) -> PyResult<f64> {
    Ok(calendar_to_days(CalendarDate { year, month, day })? as f64)
}

/// The calendar date `days` after 1970-01-01 (`as.Date(days, origin =
/// "1970-01-01")`); fractional days are truncated towards negative infinity
/// as R's `Date` printing does.
#[pyfunction]
pub fn days_to_date(days: f64) -> CalendarDate {
    // Howard Hinnant's civil-from-days algorithm on a March-based year.
    let z = days.floor() as i64 + EPOCH_DAY_NUMBER;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = (yoe + era * 400 + i64::from(month <= 2)) as i32;
    CalendarDate { year, month, day }
}

/// R's `as.Date(paste0(format(bdate, "%Y"), "-01-01"))`: January 1st of the
/// year containing day `days`, as days since 1970-01-01.
pub fn start_of_year(days: f64) -> f64 {
    day_number(days_to_date(days).year, 1, 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_table() -> RateTable {
        RateTable::try_new(
            vec![2, 2],
            vec!["age".into(), "sex".into()],
            vec![
                vec!["0".into(), "1".into()],
                vec!["male".into(), "female".into()],
            ],
            vec![Some(vec![0.0, 365.25]), None],
            vec![DimType::Continuous, DimType::Factor],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn rates_are_column_major_and_levels_match_by_label() {
        let table = toy_table();
        assert_eq!(table.rate(&[1, 0]), Some(2.0));
        assert_eq!(table.rate(&[0, 1]), Some(3.0));
        assert_eq!(table.rate(&[2, 0]), None);
        assert_eq!(table.rate(&[0]), None);
        assert_eq!(table.level(1, "female"), Some(1));
        assert_eq!(table.level(2, "female"), None);
        assert_eq!(table.factor_flags(), vec![0, 1]);
        assert_eq!(table.cut_slices()[1], &[] as &[f64]);
        assert!(table.to_string().contains("sex has levels of: male female"));
        assert_eq!(DimType::from_code(4), Some(DimType::UsYear));
        assert_eq!(DimType::UsYear.code(), 4);
    }

    #[test]
    fn positions_must_be_finite_with_factor_codes_in_range() {
        let table = toy_table();
        let err = |rows: &[[f64; 2]]| {
            table
                .validate_positions(
                    &Array2::from_shape_vec(
                        (rows.len(), 2),
                        rows.iter().flatten().copied().collect(),
                    )
                    .unwrap(),
                )
                .unwrap_err()
                .to_string()
        };
        assert!(
            table
                .validate_positions(&ndarray::arr2(&[[-5.0, 1.0], [800.0, 2.0]]))
                .is_ok()
        );
        assert!(err(&[[0.0, 0.0]]).contains("sex is out of range"));
        assert!(err(&[[0.0, 3.0]]).contains("sex is out of range"));
        assert!(err(&[[0.0, 1.5]]).contains("sex is out of range"));
        assert!(err(&[[f64::NAN, 1.0]]).contains("age contains missing values"));
        assert!(err(&[[f64::INFINITY, 1.0]]).contains("age must be finite"));
        assert!(
            table
                .validate_positions(&ndarray::arr2(&[[0.0]]))
                .unwrap_err()
                .to_string()
                .contains("one column per dimension")
        );
    }

    #[test]
    fn structural_checks_follow_is_ratetable() {
        let ok = ratetable_problems(
            &[2, 2],
            &["age".into(), "sex".into()],
            &[vec!["0".into(), "1".into()], vec!["m".into(), "f".into()]],
            &[Some(vec![0.0, 1.0]), None],
            &[2, 1],
            4,
        );
        assert!(ok.is_empty());

        let bad = ratetable_problems(
            &[2, 2],
            &["age".into(), "".into()],
            &[vec!["0".into()], vec!["m".into(), "f".into()]],
            &[Some(vec![1.0, 0.0]), Some(vec![0.0])],
            &[2, 5, 4],
            3,
        );
        assert!(bad.contains(&"length of the data does not match prod(dim)".to_string()));
        assert!(bad.contains(&"one of the dimnames identifiers is blank".to_string()));
        assert!(bad.contains(&"type attribute must be 1, 2, 3, or 4".to_string()));
        assert!(bad.contains(&"wrong length for type attribute".to_string()));
        assert!(bad.contains(&"dimname 1 is the wrong length".to_string()));
        assert!(bad.contains(&"unsorted cutpoints for dimension 1".to_string()));

        let two_us_years = ratetable_problems(
            &[1, 1],
            &["a".into(), "b".into()],
            &[vec!["x".into()], vec!["y".into()]],
            &[Some(vec![0.0]), Some(vec![0.0])],
            &[4, 4],
            1,
        );
        assert!(
            two_us_years
                .iter()
                .any(|m| m.contains("US ratetable years"))
        );

        let factor_with_cuts = ratetable_problems(
            &[1],
            &["a".into()],
            &[vec!["x".into()]],
            &[Some(vec![0.0])],
            &[1],
            1,
        );
        assert_eq!(
            factor_with_cuts,
            vec!["attribute type[1] is continuous; cutpoint should be null"]
        );

        assert!(
            RateTable::try_new(
                vec![1],
                vec!["a".into()],
                vec![vec!["x".into()]],
                vec![Some(vec![0.0])],
                vec![DimType::Continuous],
                vec![-1.0],
            )
            .is_err()
        );
        let check = is_ratetable(vec![], vec![], vec![], vec![], vec![], 0);
        assert!(!check.valid);
    }

    #[test]
    fn ratetable_date_counts_days_from_1970() {
        assert_eq!(ratetable_date(1970, 1, 1).unwrap(), 0.0);
        assert_eq!(ratetable_date(1960, 1, 1).unwrap(), -3653.0);
        assert_eq!(ratetable_date(1990, 6, 15).unwrap(), 7470.0);
        assert_eq!(ratetable_date(2000, 2, 29).unwrap(), 11016.0);
        assert_eq!(ratetable_date(2020, 12, 31).unwrap(), 18627.0);
        assert_eq!(ratetable_date(1900, 3, 1).unwrap(), -25508.0);
        assert!(ratetable_date(2001, 2, 29).is_err());
        assert!(ratetable_date(2001, 13, 1).is_err());
    }

    #[test]
    fn days_to_date_inverts_ratetable_date() {
        for (year, month, day) in [
            (1970, 1, 1),
            (1960, 1, 1),
            (1990, 6, 15),
            (2000, 2, 29),
            (2020, 12, 31),
            (1900, 3, 1),
            (1899, 12, 31),
            (2100, 2, 28),
        ] {
            let days = ratetable_date(year, month, day).unwrap();
            assert_eq!(days_to_date(days), CalendarDate { year, month, day });
        }
        assert_eq!(days_to_date(7470.5).to_string(), "1990-06-15");
        assert_eq!(days_to_date(-0.5).to_string(), "1969-12-31");
        assert_eq!(start_of_year(7470.0), ratetable_date(1990, 1, 1).unwrap());
        assert_eq!(start_of_year(-1.0), ratetable_date(1969, 1, 1).unwrap());
    }
}
