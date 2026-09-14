//! R's `neardate` (`R/neardate.R`): for every row of one data set, the
//! row of a second data set with the same identifier and the closest date
//! on or after (`best = "after"`) or on or before (`best = "prior"`) it.

use super::id_value::{IdValue, SubjectId};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::collections::HashMap;

/// R's `best` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeardateBest {
    /// The closest date on or after the query date.
    After,
    /// The closest date on or before the query date.
    Prior,
}

impl NeardateBest {
    /// `match.arg(best)`: a unique prefix of `"after"` or `"prior"`.
    pub fn parse(value: &str) -> SurvivalResult<Self> {
        match value {
            v if !v.is_empty() && "after".starts_with(v) => Ok(Self::After),
            v if !v.is_empty() && "prior".starts_with(v) => Ok(Self::Prior),
            _ => Err(SurvivalError::invalid_input(
                "best must be 'after' or 'prior'",
            )),
        }
    }
}

/// Zero-based row of data set 2 matched to each row of data set 1, `None`
/// where there is no match (R's `nomatch`).  Missing dates (`NaN`) in
/// either set never match.  Among tied dates R keeps the first row for
/// `after` and the last row for `prior`.
pub fn neardate<I: SubjectId>(
    id1: &[I],
    y1: &[f64],
    id2: &[I],
    y2: &[f64],
    best: NeardateBest,
) -> SurvivalResult<Vec<Option<usize>>> {
    validate_length(id1.len(), y1.len(), "y1")?;
    validate_length(id2.len(), y2.len(), "y2")?;

    // Rows of data set 2 with a usable date, grouped by id and sorted by
    // date (stable, so ties keep data order).
    let mut by_id: HashMap<I::Key, Vec<(f64, usize)>> = HashMap::new();
    for (row, (id, &date)) in id2.iter().zip(y2).enumerate() {
        if !date.is_nan() {
            by_id.entry(id.key()).or_default().push((date, row));
        }
    }
    if by_id.is_empty() {
        return Err(SurvivalError::invalid_input(
            "No valid entries in data set 2",
        ));
    }
    let query_keys: Vec<I::Key> = id1.iter().map(SubjectId::key).collect();
    if !query_keys.iter().any(|key| by_id.contains_key(key)) {
        return Err(SurvivalError::invalid_input(
            "No valid entries in data set 2",
        ));
    }
    for rows in by_id.values_mut() {
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    Ok(query_keys
        .iter()
        .zip(y1)
        .map(|(key, &date)| {
            if date.is_nan() {
                return None;
            }
            let rows = by_id.get(key)?;
            match best {
                NeardateBest::After => {
                    let position = rows.partition_point(|(d, _)| *d < date);
                    rows.get(position).map(|(_, row)| *row)
                }
                NeardateBest::Prior => {
                    let position = rows.partition_point(|(d, _)| *d <= date);
                    position.checked_sub(1).map(|p| rows[p].1)
                }
            }
        })
        .collect())
}

/// Python entry point of [`neardate`].
#[pyfunction(name = "neardate")]
#[pyo3(signature = (id1, y1, id2, y2, best="after"))]
pub fn neardate_py(
    id1: Vec<IdValue>,
    y1: Vec<f64>,
    id2: Vec<IdValue>,
    y2: Vec<f64>,
    best: &str,
) -> PyResult<Vec<Option<usize>>> {
    Ok(neardate(&id1, &y1, &id2, &y2, NeardateBest::parse(best)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example from `?neardate`, with R's one-based answers.
    fn doc_example() -> (Vec<i64>, Vec<f64>, Vec<i64>, Vec<f64>) {
        (
            vec![1, 1, 2, 2, 2, 3, 4, 4, 5],
            vec![10.0, 20.0, 5.0, 15.0, 25.0, 30.0, 4.0, 12.0, 7.0],
            vec![1, 1, 1, 2, 2, 3, 4, 4, 4, 6],
            vec![8.0, 12.0, 22.0, 4.0, 26.0, 30.0, 3.0, 11.0, 13.0, 1.0],
        )
    }

    fn one_based(indices: &[Option<usize>]) -> Vec<Option<usize>> {
        indices.iter().map(|i| i.map(|v| v + 1)).collect()
    }

    #[test]
    fn matches_the_documentation_example() {
        let (id1, y1, id2, y2) = doc_example();
        let after = neardate(&id1, &y1, &id2, &y2, NeardateBest::After).unwrap();
        assert_eq!(
            one_based(&after),
            vec![
                Some(2),
                Some(3),
                Some(5),
                Some(5),
                Some(5),
                Some(6),
                Some(8),
                Some(9),
                None
            ]
        );
        let prior = neardate(&id1, &y1, &id2, &y2, NeardateBest::Prior).unwrap();
        assert_eq!(
            one_based(&prior),
            vec![
                Some(1),
                Some(2),
                Some(4),
                Some(4),
                Some(4),
                Some(6),
                Some(7),
                Some(8),
                None
            ]
        );
    }

    #[test]
    fn ties_keep_the_first_row_after_and_the_last_row_prior() {
        let after = neardate(
            &[1i64],
            &[10.0],
            &[1, 1],
            &[10.0, 10.0],
            NeardateBest::After,
        )
        .unwrap();
        assert_eq!(after, vec![Some(0)]);
        let prior = neardate(
            &[1i64],
            &[10.0],
            &[1, 1],
            &[10.0, 10.0],
            NeardateBest::Prior,
        )
        .unwrap();
        assert_eq!(prior, vec![Some(1)]);
    }

    #[test]
    fn missing_dates_never_match_and_prefixes_are_accepted() {
        let after = neardate(
            &["a", "a", "a"],
            &[f64::NAN, 2.0, f64::INFINITY],
            &["a", "a", "a"],
            &[1.0, f64::NAN, f64::INFINITY],
            NeardateBest::After,
        )
        .unwrap();
        assert_eq!(after, vec![None, Some(2), Some(2)]);
        assert_eq!(NeardateBest::parse("a").unwrap(), NeardateBest::After);
        assert_eq!(NeardateBest::parse("pr").unwrap(), NeardateBest::Prior);
        assert!(NeardateBest::parse("").is_err());
        assert!(NeardateBest::parse("closest").is_err());
    }

    #[test]
    fn requires_a_usable_reference_row() {
        assert!(neardate(&[1i64], &[1.0], &[1], &[f64::NAN], NeardateBest::After).is_err());
        assert!(neardate(&[1i64], &[1.0], &[2], &[1.0], NeardateBest::After).is_err());
        assert!(neardate(&[1i64], &[], &[1], &[1.0], NeardateBest::After).is_err());
    }
}
