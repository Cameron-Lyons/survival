use super::id_value::{IdValue, SubjectId};
use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeardateBest {
    After,
    Prior,
}

impl NeardateBest {
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

pub fn neardate<I: SubjectId>(
    id1: &[I],
    y1: &[f64],
    id2: &[I],
    y2: &[f64],
    best: NeardateBest,
) -> SurvivalResult<Vec<Option<usize>>> {
    validate_length(id1.len(), y1.len(), "y1")?;
    validate_length(id2.len(), y2.len(), "y2")?;

    let query_keys: Vec<I::Key> = id1.iter().map(SubjectId::key).collect();
    // A small query set can avoid indexing unrelated reference subjects. For
    // comparable input sizes, keep the single-pass reference index: preparing
    // a query index adds hashing and string clones with little work to skip.
    let restrict_reference = id1.len() <= id2.len() / 16;
    let mut by_id: HashMap<I::Key, Vec<(f64, usize)>> = HashMap::new();
    if restrict_reference {
        for key in &query_keys {
            by_id.entry(key.clone()).or_default();
        }
    }
    for (row, (id, &date)) in id2.iter().zip(y2).enumerate() {
        if !date.is_nan() {
            if restrict_reference {
                if let Some(rows) = by_id.get_mut(&id.key()) {
                    rows.push((date, row));
                }
            } else {
                by_id.entry(id.key()).or_default().push((date, row));
            }
        }
    }
    if !query_keys
        .iter()
        .any(|key| by_id.get(key).is_some_and(|rows| !rows.is_empty()))
    {
        return Err(SurvivalError::invalid_input(
            "No valid entries in data set 2",
        ));
    }
    for rows in by_id.values_mut() {
        // NaNs were removed above. Numeric comparison keeps -0 and +0 tied;
        // stable sorting preserves the first/last reference row chosen below.
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

#[doc = " Subject ids from Python: an int64 array is read without per-element"]
#[doc = " conversion, anything else as int, float or str labels."]
pub(crate) enum NeardateIds {
    Ints(Vec<i64>),
    Labels(Vec<IdValue>),
}

impl NeardateIds {
    fn into_labels(self) -> Vec<IdValue> {
        match self {
            Self::Ints(values) => values.into_iter().map(IdValue::Int).collect(),
            Self::Labels(values) => values,
        }
    }
}

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for NeardateIds {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        use numpy::PyReadonlyArray1;

        if let Ok(array) = obj.extract::<PyReadonlyArray1<'py, i64>>() {
            return Ok(Self::Ints(array.as_array().to_vec()));
        }
        Ok(Self::Labels(obj.extract()?))
    }
}

#[doc = " Python entry point of [`neardate`]."]
#[pyfunction(name = "neardate")]
#[pyo3(signature = (id1, y1, id2, y2, best="after"))]
pub(crate) fn neardate_py(
    py: Python<'_>,
    id1: NeardateIds,
    y1: crate::internal::numpy_utils::FloatVec,
    id2: NeardateIds,
    y2: crate::internal::numpy_utils::FloatVec,
    best: &str,
) -> PyResult<Vec<Option<usize>>> {
    let best = NeardateBest::parse(best)?;
    Ok(py.detach(|| match (id1, id2) {
        (NeardateIds::Ints(id1), NeardateIds::Ints(id2)) => neardate(&id1, &y1, &id2, &y2, best),
        (id1, id2) => neardate(&id1.into_labels(), &y1, &id2.into_labels(), &y2, best),
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn signed_zero_dates_keep_the_reference_row_tie_order() {
        // survival::neardate(c(1, 1), c(1, 1), c(0, -0), c(0, -0))
        // returns c(1, 1), or c(2, 2) with best = "prior".
        for dates in [[0.0, -0.0], [-0.0, 0.0]] {
            for (best, expected) in [
                (NeardateBest::After, vec![Some(0); 2]),
                (NeardateBest::Prior, vec![Some(1); 2]),
            ] {
                assert_eq!(
                    neardate(&[1i64; 2], &[0.0, -0.0], &[1; 2], &dates, best).unwrap(),
                    expected,
                );
            }
        }
    }

    #[test]
    fn matches_linear_search_with_unqueried_subjects_and_missing_dates() {
        let id1 = ["b", "a", "b", "absent", "b", "a", "a"];
        let y1 = [
            1.0,
            0.0,
            f64::NAN,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            2.0,
        ];
        let id2 = ["unused", "b", "a", "unused", "b", "a", "a", "b", "a"];
        let y2 = [
            1.0,
            1.0,
            0.0,
            2.0,
            1.0,
            -0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ];
        for best in [NeardateBest::After, NeardateBest::Prior] {
            let expected: Vec<_> = id1
                .iter()
                .zip(y1)
                .map(|(id, date)| {
                    let mut found = None;
                    for (row, (candidate_id, &candidate_date)) in id2.iter().zip(&y2).enumerate() {
                        if id != candidate_id {
                            continue;
                        }
                        let is_better = match best {
                            NeardateBest::After => {
                                candidate_date >= date
                                    && found.is_none_or(|previous| candidate_date < y2[previous])
                            }
                            NeardateBest::Prior => {
                                candidate_date <= date
                                    && found.is_none_or(|previous| candidate_date >= y2[previous])
                            }
                        };
                        if is_better {
                            found = Some(row);
                        }
                    }
                    found
                })
                .collect();
            // Extra unrelated rows exercise the restricted reference index.
            for extra in [0, 128] {
                let mut references = id2.to_vec();
                references.extend(std::iter::repeat_n("unused", extra));
                let mut dates = y2.to_vec();
                dates.extend(std::iter::repeat_n(1.0, extra));
                for repeats in [1, 3] {
                    assert_eq!(
                        neardate(
                            &id1.repeat(repeats),
                            &y1.repeat(repeats),
                            &references,
                            &dates,
                            best,
                        )
                        .unwrap(),
                        expected.repeat(repeats),
                    );
                }
            }
        }
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
        assert!(
            neardate(
                &[1i64; 3],
                &[1.0; 3],
                &[1, 2],
                &[f64::NAN, 1.0],
                NeardateBest::After,
            )
            .is_err()
        );
    }
}
