//! R's `match.ratetable` (`R/match.ratetable.R`): map the variables named
//! in `rmap` onto the dimensions of a rate table, producing the matrix `R`
//! of per-subject starting positions that `pyears` and `survexp` feed to
//! the C person-years code.

use super::ratetable::{DimType, RateTable, start_of_year};
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::Array2;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// One variable of the `rmap` list: either numeric (a continuous value, a
/// date already converted with `ratetableDate`, or an integer factor code)
/// or a vector of labels matched against the dimension's `dimnames`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum RatetableColumn {
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Numeric(Vec<f64>),
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Labels(Vec<String>),
}

impl RatetableColumn {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Labels(values) => values.len(),
        }
    }
}

/// The result of `match.ratetable`: `r` has one row per subject and one
/// column per rate-table dimension (in the table's order); factor columns
/// hold R's one-based level subscripts, the others the numeric values.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct MatchRatetableResult {
    #[pyo3(get)]
    pub r: Vec<Vec<f64>>,
    /// The table's cutpoints after `ratetableDate` (already day counts).
    #[pyo3(get)]
    pub cutpoints: Vec<Option<Vec<f64>>>,
}

/// R's `charmatch(casefold(x), casefold(table))`: an exact match wins, a
/// unique prefix match is accepted, several prefix matches give `Some(0)`
/// (R's 0) and none gives `None` (R's `NA`).  Returns one-based positions.
struct LevelMatcher {
    exact: HashMap<String, usize>,
    folded: Vec<String>,
}

impl LevelMatcher {
    fn new(levels: &[String]) -> Self {
        let folded: Vec<String> = levels.iter().map(|s| s.to_lowercase()).collect();
        let mut exact = HashMap::with_capacity(levels.len());
        for (i, label) in folded.iter().enumerate() {
            // R's charmatch rejects duplicate exact matches, including case folding.
            exact
                .entry(label.clone())
                .and_modify(|code| *code = 0)
                .or_insert(i + 1);
        }
        Self { exact, folded }
    }

    fn find(&self, value: &str) -> Option<usize> {
        let value = value.to_lowercase();
        if let Some(&code) = self.exact.get(&value) {
            return Some(code);
        }
        let mut found = None;
        for (i, candidate) in self.folded.iter().enumerate() {
            if candidate.starts_with(&value) {
                if found.is_some() {
                    return Some(0);
                }
                found = Some(i + 1);
            }
        }
        found
    }
}

fn visit_labels(
    dimid: &str,
    levels: &[String],
    labels: &[String],
    mut emit: impl FnMut(usize, usize),
) -> SurvivalResult<()> {
    let matcher = LevelMatcher::new(levels);
    // Sex/race dimensions usually have two labels: a tiny linear cache avoids
    // hashing every observation. Promote once there are more distinct labels.
    let mut small: Vec<(&str, usize)> = Vec::new();
    let mut codes: HashMap<&str, usize> = HashMap::new();
    for (row, label) in labels.iter().enumerate() {
        let cached = if codes.is_empty() {
            small
                .iter()
                .find_map(|&(seen, code)| (seen == label).then_some(code))
        } else {
            codes.get(label.as_str()).copied()
        };
        let code = match cached {
            Some(code) => code,
            None => {
                let code = match matcher.find(label) {
                    None => {
                        return Err(SurvivalError::invalid_input(format!(
                            "Levels do not match for ratetable() variable {dimid}"
                        )));
                    }
                    Some(0) => {
                        return Err(SurvivalError::invalid_input(format!(
                            "Non-unique ratetable match for variable {dimid}"
                        )));
                    }
                    Some(code) => code,
                };
                if codes.is_empty() && small.len() < 4 {
                    small.push((label.as_str(), code));
                } else {
                    if codes.is_empty() {
                        codes.extend(small.drain(..));
                    }
                    codes.insert(label.as_str(), code);
                }
                code
            }
        };
        emit(row, code);
    }
    Ok(())
}

/// Match all declared factor levels, including levels absent from observations.
/// The dimension is zero-based; returned level positions are one-based, as in R.
pub fn match_levels(
    table: &RateTable,
    dimension: usize,
    labels: &[String],
) -> SurvivalResult<Vec<usize>> {
    table.validate()?;
    if dimension >= table.ndim() {
        return Err(SurvivalError::invalid_input(
            "rate-table dimension out of range",
        ));
    }
    let name = &table.dimid[dimension];
    if table.types[dimension] != DimType::Factor {
        return Err(SurvivalError::invalid_input(format!(
            "for this ratetable, {name} must be a continuous variable"
        )));
    }
    let mut codes = Vec::with_capacity(labels.len());
    visit_labels(name, &table.dimnames[dimension], labels, |_, code| {
        codes.push(code)
    })?;
    Ok(codes)
}

/// Match user variables onto a rate table's dimensions.
///
/// `names` are the `rmap` variable names and `columns` their values; every
/// `dimid` of the table must appear exactly once.
pub fn match_ratetable(
    table: &RateTable,
    names: &[String],
    columns: &[RatetableColumn],
) -> SurvivalResult<Array2<f64>> {
    table.validate()?;
    if names.len() != columns.len() {
        return Err(SurvivalError::invalid_input(
            "rmap names and columns must have the same length",
        ));
    }
    let n = columns.first().map_or(0, RatetableColumn::len);
    if columns.iter().any(|c| c.len() != n) {
        return Err(SurvivalError::invalid_input(
            "all rmap variables must have the same length",
        ));
    }
    let mut ord = Vec::with_capacity(table.ndim());
    for id in &table.dimid {
        let positions: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, name)| *name == id)
            .map(|(i, _)| i)
            .collect();
        match positions.as_slice() {
            [] => {
                return Err(SurvivalError::invalid_input(format!(
                    "Argument '{id}' needed by the ratetable was not found in the data"
                )));
            }
            [single] => ord.push(*single),
            _ => {
                return Err(SurvivalError::invalid_input(
                    "A ratetable argument appears twice in the data",
                ));
            }
        }
    }
    if ord.iter().collect::<HashSet<_>>().len() != ord.len() {
        return Err(SurvivalError::invalid_input(
            "A ratetable argument appears twice in the data",
        ));
    }

    let mut r = Array2::<f64>::zeros((n, table.ndim()));
    for (dim, &column) in ord.iter().enumerate() {
        let dimid = &table.dimid[dim];
        let dim_type = table.types[dim];
        let levels = &table.dimnames[dim];
        match &columns[column] {
            RatetableColumn::Labels(labels) => {
                if dim_type != DimType::Factor {
                    return Err(SurvivalError::invalid_input(format!(
                        "for this ratetable, {dimid} must be a continuous variable"
                    )));
                }
                visit_labels(dimid, levels, labels, |row, code| {
                    r[[row, dim]] = code as f64;
                })?;
            }
            RatetableColumn::Numeric(values) => {
                for (row, &value) in values.iter().enumerate() {
                    r[[row, dim]] = value;
                }
            }
        }
    }
    // Numeric factor codes must be level subscripts, and nothing missing.
    table.validate_positions_validated(&r)?;
    Ok(r)
}

/// The birthday adjustment R applies to type-4 (US calendar year) axes in
/// `pyears` and `survexp.fit`: someone born in June 1945 keeps the 1945
/// rate until their next birthday, while the table's cutpoints fall on
/// January 1st, so their entry date on the year axis is moved back by the
/// offset between their birthday and the start of that year.  `r` is the
/// matrix from [`match_ratetable`] and is adjusted in place.
pub fn align_us_year_axis(table: &RateTable, r: &mut Array2<f64>) -> SurvivalResult<()> {
    table.validate()?;
    align_us_year_axis_validated(table, r)
}

/// Alignment for a table already checked by the enclosing public boundary.
pub(crate) fn align_us_year_axis_validated(
    table: &RateTable,
    r: &mut Array2<f64>,
) -> SurvivalResult<()> {
    if table.us_year_dimension().is_none() {
        return Ok(());
    }
    // R looks the columns up by name, not by type.
    let (Some(age), Some(year)) = (
        table.dimid.iter().position(|id| id == "age"),
        table.dimid.iter().position(|id| id == "year"),
    ) else {
        return Err(SurvivalError::invalid_input(
            "ratetable does not have expected shape",
        ));
    };
    if age >= r.ncols() || year >= r.ncols() {
        return Err(SurvivalError::invalid_input(
            "ratetable positions do not contain the age and year columns",
        ));
    }
    for mut row in r.rows_mut() {
        let birth_date = row[year] - row[age];
        let offset = birth_date - start_of_year(birth_date)?;
        row[year] -= offset;
    }
    Ok(())
}

/// Python entry point of [`match_ratetable`].
#[pyfunction(name = "match_ratetable")]
#[pyo3(signature = (ratetable, names, columns))]
pub fn match_ratetable_py(
    py: Python<'_>,
    ratetable: &RateTable,
    names: Vec<String>,
    columns: Vec<RatetableColumn>,
) -> PyResult<MatchRatetableResult> {
    let r = py.detach(|| match_ratetable(ratetable, &names, &columns))?;
    Ok(MatchRatetableResult {
        r: r.rows().into_iter().map(|row| row.to_vec()).collect(),
        cutpoints: ratetable.cutpoints.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::population::ratetable_data::survexp_usr_table;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn labels_match_case_insensitively_and_by_unique_prefix() {
        let table = survexp_usr_table();
        let r = match_ratetable(
            table,
            &strings(&["year", "race", "sex", "age"]),
            &[
                RatetableColumn::Numeric(vec![9496.0, 9526.0]),
                RatetableColumn::Labels(strings(&["White", "b"])),
                RatetableColumn::Numeric(vec![1.0, 2.0]),
                RatetableColumn::Numeric(vec![27028.5, 24837.0]),
            ],
        )
        .unwrap();
        assert_eq!(r.shape(), [2, 4]);
        assert_eq!(r.row(0).to_vec(), vec![27028.5, 1.0, 1.0, 9496.0]);
        assert_eq!(r.row(1).to_vec(), vec![24837.0, 2.0, 2.0, 9526.0]);
        assert_eq!(
            LevelMatcher::new(&strings(&["male", "female"])).find("m"),
            Some(1)
        );
        assert_eq!(
            LevelMatcher::new(&strings(&["male", "female"])).find("e"),
            None
        );
        assert_eq!(
            LevelMatcher::new(&strings(&["ab", "ac"])).find("a"),
            Some(0)
        );
    }

    #[test]
    fn exact_matches_win_but_duplicates_are_ambiguous() {
        let matcher = LevelMatcher::new(&strings(&["a", "ab", "Male", "MALE", "女性"]));
        assert_eq!(matcher.find("a"), Some(1));
        assert_eq!(matcher.find("ma"), Some(0));
        assert_eq!(matcher.find("male"), Some(0));
        assert_eq!(matcher.find("女"), Some(5));
        assert_eq!(matcher.find(""), Some(0));
    }

    #[test]
    fn declared_levels_are_validated_independently_of_observations() {
        let table = survexp_usr_table();
        assert_eq!(
            table.match_levels(1, &strings(&["F", "m"])).unwrap(),
            vec![2, 1]
        );
        assert!(
            table
                .match_levels(1, &strings(&["male", "unknown"]))
                .unwrap_err()
                .to_string()
                .contains("Levels do not match")
        );
        assert!(
            table
                .match_levels(0, &strings(&["1"]))
                .unwrap_err()
                .to_string()
                .contains("continuous variable")
        );
        assert!(table.match_levels(4, &[]).is_err());
        assert_eq!(table.match_levels(1, &[]).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn duplicate_dimension_identifiers_cannot_reuse_the_same_input() {
        let table = RateTable::try_new(
            vec![1, 1],
            strings(&["group", "group"]),
            vec![strings(&["a"]); 2],
            vec![None; 2],
            vec![DimType::Factor; 2],
            vec![0.1],
        )
        .unwrap();
        let error = match_ratetable(
            &table,
            &strings(&["group"]),
            &[RatetableColumn::Numeric(vec![1.0])],
        )
        .unwrap_err();
        assert!(error.to_string().contains("appears twice"));
    }

    #[test]
    fn us_year_axis_moves_entry_back_to_the_birthday() {
        let table = survexp_usr_table();
        // Born 1945-06-15 (day -8966), entered on 1996-01-01 (day 9496).
        let age = 9496.0 - (-8966.0);
        let mut r = ndarray::arr2(&[[age, 1.0, 1.0, 9496.0]]);
        align_us_year_axis(table, &mut r).unwrap();
        // 1945-01-01 is day -9131; the birthday offset is 165 days.
        assert_eq!(r[[0, 3]], 9496.0 - 165.0);
        assert_eq!(r[[0, 0]], age);

        let mut plain = RateTable::try_new(
            vec![1],
            vec!["year".into()],
            vec![vec!["x".into()]],
            vec![Some(vec![0.0])],
            vec![DimType::UsYear],
            vec![0.1],
        )
        .unwrap();
        let mut r = ndarray::arr2(&[[1.0]]);
        assert!(align_us_year_axis(&plain, &mut r).is_err());
        plain.types[0] = DimType::Date;
        assert!(align_us_year_axis(&plain, &mut r).is_ok());
    }

    #[test]
    fn us_year_axis_rejects_unrepresentable_birth_dates() {
        let table = survexp_usr_table();
        for age in [f64::MAX, -f64::MAX] {
            let mut positions = ndarray::arr2(&[[age, 1.0, 1.0, 0.0]]);
            assert!(align_us_year_axis(table, &mut positions).is_err());
        }
    }

    #[test]
    fn errors_follow_r() {
        let table = survexp_usr_table();
        let err = |names: &[&str], columns: Vec<RatetableColumn>| {
            match_ratetable(table, &strings(names), &columns)
                .unwrap_err()
                .to_string()
        };
        let one = || RatetableColumn::Numeric(vec![1.0]);
        assert!(
            err(&["age", "sex", "year"], vec![one(), one(), one()])
                .contains("'race' needed by the ratetable was not found")
        );
        assert!(
            err(
                &["age", "sex", "race", "year", "age"],
                vec![one(), one(), one(), one(), one()]
            )
            .contains("appears twice")
        );
        assert!(
            err(
                &["age", "sex", "race", "year"],
                vec![
                    one(),
                    one(),
                    one(),
                    RatetableColumn::Labels(strings(&["x"]))
                ]
            )
            .contains("year must be a continuous variable")
        );
        assert!(
            err(
                &["age", "sex", "race", "year"],
                vec![
                    one(),
                    one(),
                    RatetableColumn::Labels(strings(&["green"])),
                    one()
                ]
            )
            .contains("Levels do not match")
        );
        assert!(
            err(
                &["age", "sex", "race", "year"],
                vec![one(), RatetableColumn::Numeric(vec![3.0]), one(), one()]
            )
            .contains("sex is out of range")
        );
        assert!(
            err(
                &["age", "sex", "race", "year"],
                vec![
                    one(),
                    one(),
                    one(),
                    RatetableColumn::Numeric(vec![1.0, 2.0])
                ]
            )
            .contains("same length")
        );
    }
}
