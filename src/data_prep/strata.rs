//! R's `strata` (`R/strata.R`): combine one or more categorical variables
//! into a single factor whose levels are the observed combinations, in
//! the lexicographic order of the component levels, labelled
//! `name=level` (or the bare level with `shortlabel`) joined by `sep`.
//!
//! Each variable arrives as zero-based level codes (`None` for `NA`) with
//! its level labels, so the caller decides the level order the way R's
//! `factor()` would for that data type.

use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// One variable of a `strata(...)` call.
#[derive(Debug, Clone, PartialEq)]
pub struct StrataVariable {
    /// The argument name (what was typed, or the name of a named argument).
    pub name: String,
    /// The level labels, in level order.
    pub levels: Vec<String>,
    /// Zero-based level code of each observation, `None` for `NA`.
    pub codes: Vec<Option<usize>>,
}

/// The strata factor.
#[derive(Debug, Clone, PartialEq)]
#[pyclass(from_py_object)]
pub struct StrataResult {
    /// Zero-based stratum of each observation, `None` where any component
    /// is missing (and `na_group` is off).
    #[pyo3(get)]
    pub codes: Vec<Option<usize>>,
    /// Label of each observed stratum, in level order.
    #[pyo3(get)]
    pub levels: Vec<String>,
    /// Number of observations in each stratum.
    #[pyo3(get)]
    pub counts: Vec<usize>,
}

/// R's `format()` of a character vector: pad every string to the width of
/// the longest.
fn pad_to_common_width(labels: &mut [String]) {
    let width = labels.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    for label in labels.iter_mut() {
        let short = width - label.chars().count();
        label.extend(std::iter::repeat_n(' ', short));
    }
}

/// Build the strata factor.
pub fn strata(
    variables: &[StrataVariable],
    na_group: bool,
    shortlabel: bool,
    sep: &str,
) -> SurvivalResult<StrataResult> {
    let Some(first) = variables.first() else {
        return Err(SurvivalError::invalid_input(
            "strata needs at least one variable",
        ));
    };
    let n = first.codes.len();
    if variables.iter().any(|v| v.codes.len() != n) {
        return Err(SurvivalError::invalid_input(
            "all arguments must be the same length",
        ));
    }
    // Per-variable labels (with an "NA" level when requested) and codes.
    let mut labels: Vec<Vec<String>> = Vec::with_capacity(variables.len());
    let mut codes: Vec<Vec<Option<usize>>> = Vec::with_capacity(variables.len());
    for (k, variable) in variables.iter().enumerate() {
        if variable
            .codes
            .iter()
            .flatten()
            .any(|&code| code >= variable.levels.len())
        {
            return Err(SurvivalError::invalid_input(format!(
                "strata codes of {} exceed its level count",
                variable.name
            )));
        }
        let mut level_labels: Vec<String> = if shortlabel {
            variable.levels.clone()
        } else {
            variable
                .levels
                .iter()
                .map(|level| format!("{}={}", variable.name, level))
                .collect()
        };
        let mut level_codes = variable.codes.clone();
        if na_group && level_codes.iter().any(Option::is_none) {
            let na_code = variable.levels.len();
            level_labels.push(if shortlabel {
                "NA".to_string()
            } else {
                format!("{}=NA", variable.name)
            });
            for code in level_codes.iter_mut() {
                code.get_or_insert(na_code);
            }
        }
        // R formats every variable but the first to a common width.
        if !shortlabel && k > 0 {
            pad_to_common_width(&mut level_labels);
        }
        labels.push(level_labels);
        codes.push(level_codes);
    }

    // The combined level number is the radix code of the tuple, so the
    // observed combinations sorted as tuples are R's `sort(unique(levs))`.
    let mut observed: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut tuples: Vec<Option<Vec<usize>>> = Vec::with_capacity(n);
    for row in 0..n {
        let tuple: Option<Vec<usize>> = codes.iter().map(|c| c[row]).collect();
        if let Some(tuple) = &tuple {
            *observed.entry(tuple.clone()).or_insert(0) += 1;
        }
        tuples.push(tuple);
    }
    let mut levels = Vec::with_capacity(observed.len());
    let mut counts = Vec::with_capacity(observed.len());
    let mut position: BTreeMap<&Vec<usize>, usize> = BTreeMap::new();
    for (index, (tuple, count)) in observed.iter().enumerate() {
        let label: Vec<&str> = tuple
            .iter()
            .enumerate()
            .map(|(k, &code)| labels[k][code].as_str())
            .collect();
        levels.push(label.join(sep));
        counts.push(*count);
        position.insert(tuple, index);
    }
    let codes = tuples
        .iter()
        .map(|tuple| tuple.as_ref().map(|t| position[t]))
        .collect();
    Ok(StrataResult {
        codes,
        levels,
        counts,
    })
}

/// Python entry point of [`strata`]: parallel lists of names, level labels
/// and per-observation codes.
#[pyfunction(name = "strata")]
#[pyo3(signature = (names, levels, codes, na_group=false, shortlabel=false, sep=", "))]
pub fn strata_py(
    names: Vec<String>,
    levels: Vec<Vec<String>>,
    codes: Vec<Vec<Option<usize>>>,
    na_group: bool,
    shortlabel: bool,
    sep: &str,
) -> PyResult<StrataResult> {
    if names.len() != levels.len() || names.len() != codes.len() {
        return Err(SurvivalError::invalid_input(
            "names, levels and codes must have one entry per variable",
        )
        .into());
    }
    let variables: Vec<StrataVariable> = names
        .into_iter()
        .zip(levels)
        .zip(codes)
        .map(|((name, levels), codes)| StrataVariable {
            name,
            levels,
            codes,
        })
        .collect();
    Ok(strata(&variables, na_group, shortlabel, sep)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable(name: &str, levels: &[&str], codes: &[Option<usize>]) -> StrataVariable {
        StrataVariable {
            name: name.to_string(),
            levels: levels.iter().map(|s| s.to_string()).collect(),
            codes: codes.to_vec(),
        }
    }

    #[test]
    fn single_variable_levels_are_named_and_counted() {
        let result = strata(
            &[variable(
                "x",
                &["1", "2", "3"],
                &[Some(1), Some(0), Some(1), Some(2), Some(0)],
            )],
            false,
            false,
            ", ",
        )
        .unwrap();
        assert_eq!(
            result.codes,
            vec![Some(1), Some(0), Some(1), Some(2), Some(0)]
        );
        assert_eq!(result.levels, vec!["x=1", "x=2", "x=3"]);
        assert_eq!(result.counts, vec![2, 2, 1]);
    }

    #[test]
    fn combinations_are_ordered_lexicographically_and_unused_ones_dropped() {
        let result = strata(
            &[
                variable("a", &["1", "2"], &[Some(1), Some(0), Some(1), None]),
                variable("b", &["x", "yy"], &[Some(1), Some(0), Some(0), Some(0)]),
            ],
            false,
            false,
            ", ",
        )
        .unwrap();
        assert_eq!(result.codes, vec![Some(2), Some(0), Some(1), None]);
        // R pads the labels of later variables to a common width.
        assert_eq!(result.levels, vec!["a=1, b=x ", "a=2, b=x ", "a=2, b=yy"]);
        assert_eq!(result.counts, vec![1, 1, 1]);

        let short = strata(
            &[
                variable("a", &["1", "2"], &[Some(1), Some(0)]),
                variable("b", &["x", "yy"], &[Some(1), Some(0)]),
            ],
            false,
            true,
            "/",
        )
        .unwrap();
        assert_eq!(short.levels, vec!["1/x", "2/yy"]);
    }

    #[test]
    fn na_group_adds_a_level() {
        let result = strata(
            &[variable("g", &["a"], &[Some(0), None, None])],
            true,
            false,
            ", ",
        )
        .unwrap();
        assert_eq!(result.codes, vec![Some(0), Some(1), Some(1)]);
        assert_eq!(result.levels, vec!["g=a", "g=NA"]);
        assert_eq!(result.counts, vec![1, 2]);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(strata(&[], false, false, ", ").is_err());
        assert!(
            strata(
                &[
                    variable("a", &["1"], &[Some(0), Some(0)]),
                    variable("b", &["1"], &[Some(0)])
                ],
                false,
                false,
                ", "
            )
            .is_err()
        );
        assert!(strata(&[variable("a", &["1"], &[Some(1)])], false, false, ", ").is_err());
    }
}
