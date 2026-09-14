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
//! R 3.8 tables carry no `dimid` attribute; as in R, `names(dimnames)` is
//! used instead.

use super::ratetable::{DimType, RateTable};
use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use std::sync::OnceLock;

const SURVEXP_US_TSV: &str = include_str!("data/survexp_us.tsv");
const SURVEXP_USR_TSV: &str = include_str!("data/survexp_usr.tsv");
const SURVEXP_MN_TSV: &str = include_str!("data/survexp_mn.tsv");

/// Parse one of the embedded TSV rate tables.
fn parse(name: &str, text: &str) -> SurvivalResult<RateTable> {
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
                    .map(|f| {
                        f.parse::<i64>()
                            .ok()
                            .and_then(DimType::from_code)
                            .ok_or_else(|| line_err("type codes must be 1, 2, 3 or 4".into()))
                    })
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

    // Reorder the per-dimension lines to the `dimid` order.
    fn by_dimid<T: Clone>(
        dimid: &[String],
        entries: &[(String, T)],
        what: &str,
        err: &impl Fn(String) -> SurvivalError,
    ) -> SurvivalResult<Vec<T>> {
        if entries.len() != dimid.len() {
            return Err(err(format!("one {what} line per dimension required")));
        }
        dimid
            .iter()
            .map(|id| {
                entries
                    .iter()
                    .find(|(name, _)| name == id)
                    .map(|(_, value)| value.clone())
                    .ok_or_else(|| err(format!("missing {what} for dimension {id:?}")))
            })
            .collect()
    }
    let dimnames = by_dimid(&dimid, &dimnames, "dimnames", &err)?;
    let cutpoints = by_dimid(&dimid, &cutpoints, "cutpoints", &err)?;
    RateTable::try_new(dims, dimid, dimnames, cutpoints, types, rates)
        .map_err(|e| err(e.to_string()))
}

fn load(cell: &'static OnceLock<RateTable>, name: &'static str, text: &str) -> &'static RateTable {
    cell.get_or_init(|| {
        parse(name, text).unwrap_or_else(|e| panic!("embedded rate table is malformed: {e}"))
    })
}

/// R's `survexp.us`: US total population daily hazards by single year of
/// age (0-109), sex, and calendar year 1940-2020.
pub fn survexp_us_table() -> &'static RateTable {
    static TABLE: OnceLock<RateTable> = OnceLock::new();
    load(&TABLE, "survexp.us", SURVEXP_US_TSV)
}

/// R's `survexp.usr`: US daily hazards by age, sex, race (white/black), and
/// calendar year 1940-2020.
pub fn survexp_usr_table() -> &'static RateTable {
    static TABLE: OnceLock<RateTable> = OnceLock::new();
    load(&TABLE, "survexp.usr", SURVEXP_USR_TSV)
}

/// R's `survexp.mn`: Minnesota daily hazards by age, sex, and calendar year
/// 1970-2020.
pub fn survexp_mn_table() -> &'static RateTable {
    static TABLE: OnceLock<RateTable> = OnceLock::new();
    load(&TABLE, "survexp.mn", SURVEXP_MN_TSV)
}

/// R's `survexp.us` rate table.
#[pyfunction]
pub fn survexp_us() -> RateTable {
    survexp_us_table().clone()
}

/// R's `survexp.usr` rate table.
#[pyfunction]
pub fn survexp_usr() -> RateTable {
    survexp_usr_table().clone()
}

/// R's `survexp.mn` rate table.
#[pyfunction]
pub fn survexp_mn() -> RateTable {
    survexp_mn_table().clone()
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

    fn lookup(table: &RateTable, labels: &[&str]) -> f64 {
        let index: Vec<usize> = labels
            .iter()
            .enumerate()
            .map(|(d, label)| {
                table
                    .level(d, label)
                    .unwrap_or_else(|| panic!("no level {label:?} on axis {d}"))
            })
            .collect();
        table.rate(&index).unwrap()
    }

    fn check_common_shape(table: &RateTable, years: (&str, &str), n_years: usize) {
        assert_eq!(table.dims[0], 110);
        assert_eq!(table.dims[1], 2);
        assert_eq!(*table.dims.last().unwrap(), n_years);
        assert_eq!(table.rates.len(), table.dims.iter().product::<usize>());
        assert_eq!(table.dimid[0], "age");
        assert_eq!(table.dimid[1], "sex");
        assert_eq!(table.dimid.last().unwrap(), "year");
        assert_eq!(table.types[0], DimType::Continuous);
        assert_eq!(table.types[1], DimType::Factor);
        assert_eq!(*table.types.last().unwrap(), DimType::UsYear);
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
        let table = survexp_us_table();
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
        assert_eq!(survexp_us(), *table);
    }

    #[test]
    fn survexp_usr_matches_r() {
        let table = survexp_usr_table();
        assert_eq!(table.dims, [110, 2, 2, 81]);
        assert_eq!(table.dimid, ["age", "sex", "race", "year"]);
        assert_eq!(
            table.types,
            [
                DimType::Continuous,
                DimType::Factor,
                DimType::Factor,
                DimType::UsYear
            ]
        );
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
        assert_eq!(survexp_usr().dims, table.dims);
    }

    #[test]
    fn survexp_mn_matches_r() {
        let table = survexp_mn_table();
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
        assert_eq!(survexp_mn().dims, table.dims);
    }

    #[test]
    fn loaders_are_cached() {
        assert!(std::ptr::eq(survexp_us_table(), survexp_us_table()));
    }

    #[test]
    fn parse_rejects_inconsistent_tables() {
        let good = "dim\t2\t2\ndimid\tage\tsex\ntype\t2\t1\ndimnames\tage\t0\t1\ndimnames\tsex\tmale\tfemale\ncutpoints\tage\t0\t365.25\ncutpoints\tsex\nrate\t1\nrate\t2\nrate\t3\nrate\t4\n";
        let table = parse("toy", good).unwrap();
        assert_eq!(table.rate(&[1, 0]), Some(2.0));
        assert_eq!(table.rate(&[0, 1]), Some(3.0));
        assert_eq!(table.level(1, "female"), Some(1));

        let too_few_rates = good.trim_end_matches("rate\t4\n");
        assert!(parse("toy", too_few_rates).is_err());
        let factor_with_cutpoints = good.replace("cutpoints\tsex\n", "cutpoints\tsex\t0\t1\n");
        assert!(parse("toy", &factor_with_cutpoints).is_err());
        let bad_type = good.replace("type\t2\t1", "type\t2\t5");
        assert!(parse("toy", &bad_type).is_err());
        let decreasing = good.replace("cutpoints\tage\t0\t365.25", "cutpoints\tage\t365.25\t0");
        assert!(parse("toy", &decreasing).is_err());
        let missing_cutpoints = good.replace("cutpoints\tsex\n", "");
        assert!(parse("toy", &missing_cutpoints).is_err());
        assert!(parse("toy", "").is_err());
    }
}
