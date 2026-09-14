//! R's `match.ratetable` (`R/match.ratetable.R`): map the variables named
//! in `rmap` onto the dimensions of a rate table, producing the matrix `R`
//! of per-subject starting positions that `pyears` and `survexp` feed to
//! the C person-years code.

use super::ratetable::{DimType, RateTable, start_of_year};
use crate::error::{SurvivalError, SurvivalResult};
use ndarray::Array2;
use pyo3::prelude::*;

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
fn charmatch_fold(value: &str, table: &[String]) -> Option<usize> {
    let value = value.to_lowercase();
    let mut partial = Vec::new();
    for (i, candidate) in table.iter().enumerate() {
        let candidate = candidate.to_lowercase();
        if candidate == value {
            return Some(i + 1);
        }
        if candidate.starts_with(&value) {
            partial.push(i + 1);
        }
    }
    match partial.as_slice() {
        [] => None,
        [single] => Some(*single),
        _ => Some(0),
    }
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
                // R matches the factor levels once and indexes by level.
                let mut codes: Vec<(String, usize)> = Vec::new();
                for (row, label) in labels.iter().enumerate() {
                    let code = match codes.iter().find(|(seen, _)| seen == label) {
                        Some((_, code)) => *code,
                        None => {
                            let code = match charmatch_fold(label, levels) {
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
                            codes.push((label.clone(), code));
                            code
                        }
                    };
                    r[[row, dim]] = code as f64;
                }
            }
            RatetableColumn::Numeric(values) => {
                for (row, &value) in values.iter().enumerate() {
                    r[[row, dim]] = value;
                }
            }
        }
    }
    // Numeric factor codes must be level subscripts, and nothing missing.
    table.validate_positions(&r)?;
    Ok(r)
}

/// The birthday adjustment R applies to type-4 (US calendar year) axes in
/// `pyears` and `survexp.fit`: someone born in June 1945 keeps the 1945
/// rate until their next birthday, while the table's cutpoints fall on
/// January 1st, so their entry date on the year axis is moved back by the
/// offset between their birthday and the start of that year.  `r` is the
/// matrix from [`match_ratetable`] and is adjusted in place.
pub fn align_us_year_axis(table: &RateTable, r: &mut Array2<f64>) -> SurvivalResult<()> {
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
    for mut row in r.rows_mut() {
        let birth_date = row[year] - row[age];
        let offset = birth_date - start_of_year(birth_date);
        row[year] -= offset;
    }
    Ok(())
}

/// Python entry point of [`match_ratetable`].
#[pyfunction(name = "match_ratetable")]
#[pyo3(signature = (ratetable, names, columns))]
pub fn match_ratetable_py(
    ratetable: &RateTable,
    names: Vec<String>,
    columns: Vec<RatetableColumn>,
) -> PyResult<MatchRatetableResult> {
    let r = match_ratetable(ratetable, &names, &columns)?;
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
        assert_eq!(charmatch_fold("m", &strings(&["male", "female"])), Some(1));
        assert_eq!(charmatch_fold("e", &strings(&["male", "female"])), None);
        assert_eq!(charmatch_fold("a", &strings(&["ab", "ac"])), Some(0));
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
