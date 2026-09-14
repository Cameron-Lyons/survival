//! Exact copies of the census rate tables shipped with R's `survival`
//! package: `survexp.us`, `survexp.usr` and `survexp.mn` (survival 3.8-11).
//!
//! Each table is stored as a TSV under `data/` holding R's `dim`, `dimnames`,
//! `type` and `cutpoints` attributes plus the daily hazard rates in R's
//! column-major storage order (first index varies fastest). Rates are the
//! per-day hazards R stores (`-log(1 - q) / 365.25` for a one-year death
//! probability `q`), age cutpoints are in days, and calendar-year cutpoints
//! are R `Date`s stored as days since 1970-01-01. Every double is written as
//! the shortest decimal that a correctly rounded parser reads back as the
//! identical double, so the rates are bit-exact copies of R's.
//!
//! Type codes follow `?ratetable`: 1 = factor, 2 = continuous (age in
//! days), 3 = date, 4 = the calendar-year date axis of the US census tables.
//! R 3.8 tables carry no `dimid` attribute; as in R, `names(dimnames)` is
//! used instead.

use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::OnceLock;

const SURVEXP_US_TSV: &str = include_str!("data/survexp_us.tsv");
const SURVEXP_USR_TSV: &str = include_str!("data/survexp_usr.tsv");
const SURVEXP_MN_TSV: &str = include_str!("data/survexp_mn.tsv");

/// An R `ratetable` array with its attributes, exactly as exported from R.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRateTable {
    /// R name of the table, e.g. `survexp.us`.
    pub name: &'static str,
    /// `dim(x)`.
    pub dims: Vec<usize>,
    /// `names(dimnames(x))` (R's `dimid`), one per dimension.
    pub dimid: Vec<String>,
    /// `dimnames(x)`, one label vector per dimension.
    pub dimnames: Vec<Vec<String>>,
    /// `attr(x, "cutpoints")`: lower bounds per dimension, `None` for factor
    /// dimensions (R `NULL`). Dates are days since 1970-01-01.
    pub cutpoints: Vec<Option<Vec<f64>>>,
    /// `attr(x, "type")`: R type code (1-4) per dimension.
    pub types: Vec<u8>,
    /// The daily hazard rates in column-major order, `dims.iter().product()`
    /// long.
    pub rates: Vec<f64>,
}

impl RawRateTable {
    fn parse(name: &'static str, text: &str) -> SurvivalResult<Self> {
        let err = |message: String| SurvivalError::invalid_input(format!("{name}: {message}"));
        let mut dims = Vec::new();
        let mut dimid = Vec::new();
        let mut types = Vec::new();
        let mut dimnames: Vec<(String, Vec<String>)> = Vec::new();
        let mut cutpoints: Vec<(String, Option<Vec<f64>>)> = Vec::new();
        let mut rates = Vec::new();

        for (line_no, line) in text.lines().enumerate() {
            let mut fields = line.split('\t');
            let key = fields.next().unwrap_or_default();
            let line_err = |message: String| err(format!("line {}: {message}", line_no + 1));
            match key {
                "dim" => {
                    dims = fields
                        .map(|f| f.parse::<usize>().map_err(|e| line_err(e.to_string())))
                        .collect::<SurvivalResult<_>>()?;
                    rates.reserve(dims.iter().product());
                }
                "dimid" => dimid = fields.map(str::to_string).collect(),
                "type" => {
                    types = fields
                        .map(|f| f.parse::<u8>().map_err(|e| line_err(e.to_string())))
                        .collect::<SurvivalResult<_>>()?;
                }
                "dimnames" => {
                    let id = fields.next().unwrap_or_default().to_string();
                    dimnames.push((id, fields.map(str::to_string).collect()));
                }
                "cutpoints" => {
                    let id = fields.next().unwrap_or_default().to_string();
                    let values: Vec<f64> = fields
                        .map(|f| f.parse::<f64>().map_err(|e| line_err(e.to_string())))
                        .collect::<SurvivalResult<_>>()?;
                    cutpoints.push((id, (!values.is_empty()).then_some(values)));
                }
                "rate" => {
                    let value = fields
                        .next()
                        .ok_or_else(|| line_err("missing rate".to_string()))?
                        .parse::<f64>()
                        .map_err(|e| line_err(e.to_string()))?;
                    rates.push(value);
                }
                other => return Err(line_err(format!("unknown key {other:?}"))),
            }
        }

        let ndim = dims.len();
        if ndim == 0 {
            return Err(err("missing dim line".to_string()));
        }
        if dimid.len() != ndim || types.len() != ndim {
            return Err(err("dimid/type length does not match dim".to_string()));
        }
        if types.iter().any(|&t| !(1..=4).contains(&t)) {
            return Err(err("type codes must be 1, 2, 3 or 4".to_string()));
        }
        let ordered = |entries: Vec<(String, Option<Vec<f64>>)>| -> SurvivalResult<Vec<_>> {
            if entries.len() != ndim {
                return Err(err("one cutpoints line per dimension required".to_string()));
            }
            dimid
                .iter()
                .map(|id| {
                    entries
                        .iter()
                        .find(|(name, _)| name == id)
                        .map(|(_, values)| values.clone())
                        .ok_or_else(|| err(format!("missing cutpoints for dimension {id:?}")))
                })
                .collect()
        };
        let dimnames: Vec<Vec<String>> = dimid
            .iter()
            .map(|id| {
                dimnames
                    .iter()
                    .find(|(name, _)| name == id)
                    .map(|(_, labels)| labels.clone())
                    .ok_or_else(|| err(format!("missing dimnames for dimension {id:?}")))
            })
            .collect::<SurvivalResult<_>>()?;
        let cutpoints = ordered(cutpoints)?;

        for (d, (&n, ty)) in dims.iter().zip(&types).enumerate() {
            if dimnames[d].len() != n {
                return Err(err(format!(
                    "dimnames[{d}] has {} labels, dim is {n}",
                    dimnames[d].len()
                )));
            }
            match (&cutpoints[d], *ty) {
                (None, 1) => {}
                (Some(values), 2..=4) if values.len() == n => {
                    if values.windows(2).any(|w| w[1] <= w[0]) {
                        return Err(err(format!("cutpoints[{d}] must be strictly increasing")));
                    }
                }
                _ => {
                    return Err(err(format!(
                        "cutpoints[{d}] do not match type {ty} with dim {n}"
                    )));
                }
            }
        }
        let expected: usize = dims.iter().product();
        if rates.len() != expected {
            return Err(err(format!(
                "expected {expected} rates, found {}",
                rates.len()
            )));
        }
        if rates.iter().any(|r| !r.is_finite() || *r < 0.0) {
            return Err(err("rates must be finite and non-negative".to_string()));
        }

        Ok(Self {
            name,
            dims,
            dimid,
            dimnames,
            cutpoints,
            types,
            rates,
        })
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

    /// Dictionary keyed by R's attribute names: `name`, `dim`, `dimid`,
    /// `dimnames`, `cutpoints` (`None` for factor dimensions), `type`, and
    /// `rates` (column-major).
    fn to_pydict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("name", self.name)?;
        dict.set_item("dim", &self.dims)?;
        dict.set_item("dimid", &self.dimid)?;
        dict.set_item("dimnames", &self.dimnames)?;
        dict.set_item("cutpoints", &self.cutpoints)?;
        // `Vec<u8>` would become Python `bytes`; R's type codes are integers.
        let types: Vec<i32> = self.types.iter().map(|&t| i32::from(t)).collect();
        dict.set_item("type", types)?;
        dict.set_item("rates", &self.rates)?;
        Ok(dict.into())
    }
}

fn load(
    cell: &'static OnceLock<RawRateTable>,
    name: &'static str,
    text: &str,
) -> &'static RawRateTable {
    cell.get_or_init(|| {
        RawRateTable::parse(name, text)
            .unwrap_or_else(|e| panic!("embedded rate table is malformed: {e}"))
    })
}

/// R's `survexp.us`: US total population daily hazards by single year of
/// age (0-109), sex, and calendar year 1940-2020.
pub fn survexp_us_raw() -> &'static RawRateTable {
    static TABLE: OnceLock<RawRateTable> = OnceLock::new();
    load(&TABLE, "survexp.us", SURVEXP_US_TSV)
}

/// R's `survexp.usr`: US daily hazards by age, sex, race (white/black), and
/// calendar year 1940-2020.
pub fn survexp_usr_raw() -> &'static RawRateTable {
    static TABLE: OnceLock<RawRateTable> = OnceLock::new();
    load(&TABLE, "survexp.usr", SURVEXP_USR_TSV)
}

/// R's `survexp.mn`: Minnesota daily hazards by age, sex, and calendar year
/// 1970-2020.
pub fn survexp_mn_raw() -> &'static RawRateTable {
    static TABLE: OnceLock<RawRateTable> = OnceLock::new();
    load(&TABLE, "survexp.mn", SURVEXP_MN_TSV)
}

/// R's `survexp.us` as a dictionary of its array and attributes.
#[pyfunction]
pub fn survexp_us_table(py: Python<'_>) -> PyResult<Py<PyDict>> {
    survexp_us_raw().to_pydict(py)
}

/// R's `survexp.usr` as a dictionary of its array and attributes.
#[pyfunction]
pub fn survexp_usr_table(py: Python<'_>) -> PyResult<Py<PyDict>> {
    survexp_usr_raw().to_pydict(py)
}

/// R's `survexp.mn` as a dictionary of its array and attributes.
#[pyfunction]
pub fn survexp_mn_table(py: Python<'_>) -> PyResult<Py<PyDict>> {
    survexp_mn_raw().to_pydict(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAYS_PER_YEAR: f64 = 365.25;

    /// Order-dependent hash of the IEEE-754 bit patterns of the rates
    /// (`(h * 31 + low32) mod 2^32`, then the same with the high word),
    /// computed in R over `as.vector(table)`; it detects a rate that is off
    /// by a single ulp.
    fn bits_hash(values: &[f64]) -> u64 {
        values.iter().fold(0, |h, v| {
            let bits = v.to_bits();
            let h = (h * 31 + (bits & 0xffff_ffff)) % (1 << 32);
            (h * 31 + (bits >> 32)) % (1 << 32)
        })
    }

    fn lookup(table: &RawRateTable, labels: &[&str]) -> f64 {
        let index: Vec<usize> = labels
            .iter()
            .enumerate()
            .map(|(d, label)| {
                table
                    .level(d, label)
                    .unwrap_or_else(|| panic!("{}: no level {label:?} on axis {d}", table.name))
            })
            .collect();
        table.rate(&index).unwrap()
    }

    fn check_common_shape(table: &RawRateTable, years: (&str, &str), n_years: usize) {
        assert_eq!(table.dims[0], 110);
        assert_eq!(table.dims[1], 2);
        assert_eq!(*table.dims.last().unwrap(), n_years);
        assert_eq!(table.rates.len(), table.dims.iter().product::<usize>());
        assert_eq!(table.dimid[0], "age");
        assert_eq!(table.dimid[1], "sex");
        assert_eq!(table.dimid.last().unwrap(), "year");
        assert_eq!(table.types[0], 2);
        assert_eq!(table.types[1], 1);
        assert_eq!(*table.types.last().unwrap(), 4);
        assert_eq!(table.dimnames[1], ["male", "female"]);
        assert_eq!(table.dimnames[0][0], "0");
        assert_eq!(table.dimnames[0][109], "109");
        let year_names = table.dimnames.last().unwrap();
        assert_eq!(year_names[0], years.0);
        assert_eq!(year_names[n_years - 1], years.1);

        let age = table.cutpoints[0].as_ref().unwrap();
        assert_eq!(age.len(), 110);
        for (i, &cut) in age.iter().enumerate() {
            assert_eq!(cut, i as f64 * DAYS_PER_YEAR);
        }
        assert_eq!(table.cutpoints[1], None);
        let year = table.cutpoints.last().unwrap().as_ref().unwrap();
        assert_eq!(year.len(), n_years);
        // 2020-01-01 is 18262 days after 1970-01-01.
        assert_eq!(*year.last().unwrap(), 18262.0);
        assert!(table.rates.iter().all(|r| *r > 0.0 && *r < 0.01));
    }

    #[test]
    fn survexp_us_matches_r() {
        let table = survexp_us_raw();
        assert_eq!(table.dims, [110, 2, 81]);
        check_common_shape(table, ("1940", "2020"), 81);
        // 1940-01-01 as days since 1970-01-01.
        assert_eq!(table.cutpoints[2].as_ref().unwrap()[0], -10958.0);
        assert_eq!(
            lookup(table, &["50", "male", "2000"]),
            1.5264926305778617e-05
        );
        assert_eq!(table.rates[0], 0.00014730102934525853);
        assert_eq!(table.rate(&[0, 0, 0]), Some(0.00014730102934525853));
        assert_eq!(table.rate(&[109, 1, 80]), Some(0.002866561356372693));
        assert_eq!(table.rate(&[50, 0, 60]), Some(table.rates[13250]));
        assert_eq!(
            bits_hash(&table.rates),
            745_321_604,
            "rates are not bit-exact"
        );
        assert_eq!(table.rate(&[110, 0, 0]), None);
        assert_eq!(table.rate(&[0, 0]), None);
    }

    #[test]
    fn survexp_usr_matches_r() {
        let table = survexp_usr_raw();
        assert_eq!(table.dims, [110, 2, 2, 81]);
        assert_eq!(table.dimid, ["age", "sex", "race", "year"]);
        assert_eq!(table.types, [2, 1, 1, 4]);
        assert_eq!(table.dimnames[2], ["white", "black"]);
        assert_eq!(table.cutpoints[2], None);
        check_common_shape(table, ("1940", "2020"), 81);
        assert_eq!(
            lookup(table, &["50", "female", "black", "2000"]),
            1.6614306369902047e-05
        );
        assert_eq!(table.rate(&[0, 0, 0, 0]), Some(0.00013502067777128467));
        assert_eq!(table.rate(&[109, 1, 1, 80]), Some(0.0022555625142435176));
        assert_eq!(
            bits_hash(&table.rates),
            1_345_443_196,
            "rates are not bit-exact"
        );
    }

    #[test]
    fn survexp_mn_matches_r() {
        let table = survexp_mn_raw();
        assert_eq!(table.dims, [110, 2, 51]);
        check_common_shape(table, ("1970", "2020"), 51);
        assert_eq!(table.cutpoints[2].as_ref().unwrap()[0], 0.0);
        assert_eq!(
            lookup(table, &["50", "male", "2000"]),
            1.1783374031503476e-05
        );
        assert_eq!(table.rate(&[0, 0, 0]), Some(5.461365587870247e-05));
        assert_eq!(table.rate(&[109, 1, 50]), Some(0.0029458008384485026));
        assert_eq!(
            bits_hash(&table.rates),
            3_951_088_110,
            "rates are not bit-exact"
        );
    }

    #[test]
    fn loaders_are_cached() {
        assert!(std::ptr::eq(survexp_us_raw(), survexp_us_raw()));
    }

    #[test]
    fn parse_rejects_inconsistent_tables() {
        let good = "dim\t2\t2\ndimid\tage\tsex\ntype\t2\t1\ndimnames\tage\t0\t1\ndimnames\tsex\tmale\tfemale\ncutpoints\tage\t0\t365.25\ncutpoints\tsex\nrate\t1\nrate\t2\nrate\t3\nrate\t4\n";
        let table = RawRateTable::parse("toy", good).unwrap();
        assert_eq!(table.rate(&[1, 0]), Some(2.0));
        assert_eq!(table.rate(&[0, 1]), Some(3.0));
        assert_eq!(table.level(1, "female"), Some(1));
        assert_eq!(table.level(2, "female"), None);

        let too_few_rates = good.trim_end_matches("rate\t4\n");
        assert!(RawRateTable::parse("toy", too_few_rates).is_err());
        let factor_with_cutpoints = good.replace("cutpoints\tsex\n", "cutpoints\tsex\t0\t1\n");
        assert!(RawRateTable::parse("toy", &factor_with_cutpoints).is_err());
        let bad_type = good.replace("type\t2\t1", "type\t2\t5");
        assert!(RawRateTable::parse("toy", &bad_type).is_err());
        let decreasing = good.replace("cutpoints\tage\t0\t365.25", "cutpoints\tage\t365.25\t0");
        assert!(RawRateTable::parse("toy", &decreasing).is_err());
        assert!(RawRateTable::parse("toy", "").is_err());
    }
}
