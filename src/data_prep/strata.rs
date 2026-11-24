use crate::error::{SurvivalError, SurvivalResult};
use pyo3::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct StrataVariable {
    pub name: String,
    pub levels: Vec<String>,
    pub codes: Vec<Option<usize>>,
}

#[doc = " The strata factor."]
#[derive(Debug, Clone, PartialEq)]
#[pyclass(module = "survival._survival", from_py_object)]
pub struct StrataResult {
    #[doc = " Zero-based stratum of each observation, `None` where any component"]
    #[doc = " is missing (and `na_group` is off)."]
    #[pyo3(get)]
    pub codes: Vec<Option<usize>>,
    #[doc = " Label of each observed stratum, in level order."]
    #[pyo3(get)]
    pub levels: Vec<String>,
    #[doc = " Number of observations in each stratum."]
    #[pyo3(get)]
    pub counts: Vec<usize>,
}

fn pad_to_common_width(labels: &mut [String]) {
    let width = labels.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    for label in labels.iter_mut() {
        let short = width - label.chars().count();
        label.extend(std::iter::repeat_n(' ', short));
    }
}

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
        if !shortlabel && k > 0 {
            pad_to_common_width(&mut level_labels);
        }
        labels.push(level_labels);
        codes.push(level_codes);
    }

    const FLAT_TABLE_LIMIT: usize = 65_536;
    if labels.len() == 1 {
        let level_labels = &labels[0];
        let col = &codes[0];
        let mut raw_counts = vec![0usize; level_labels.len()];
        for &entry in col {
            if let Some(code) = entry {
                raw_counts[code] += 1;
            }
        }
        let mut remap = vec![0usize; level_labels.len()];
        let mut levels = Vec::new();
        let mut counts = Vec::new();
        for (old, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                remap[old] = levels.len();
                levels.push(level_labels[old].clone());
                counts.push(count);
            }
        }
        let out_codes = col.iter().map(|entry| entry.map(|c| remap[c])).collect();
        return Ok(StrataResult {
            codes: out_codes,
            levels,
            counts,
        });
    }

    let total_combos = labels
        .iter()
        .try_fold(1usize, |acc, l| acc.checked_mul(l.len()))
        .filter(|&prod| prod > 0 && prod <= FLAT_TABLE_LIMIT);
    if let Some(prod) = total_combos {
        let dims: Vec<usize> = labels.iter().map(Vec::len).collect();
        let mut flat_idx: Vec<Option<usize>> = Vec::with_capacity(n);
        let mut raw_counts = vec![0usize; prod];
        for row in 0..n {
            let mut idx = 0usize;
            let mut valid = true;
            for (k, col) in codes.iter().enumerate() {
                if let Some(c) = col[row] {
                    idx = idx * dims[k] + c;
                } else {
                    valid = false;
                    break;
                }
            }
            if valid {
                raw_counts[idx] += 1;
                flat_idx.push(Some(idx));
            } else {
                flat_idx.push(None);
            }
        }
        let mut remap = vec![0usize; prod];
        let mut levels = Vec::new();
        let mut counts = Vec::new();
        let mut parts = vec![""; labels.len()];
        for (idx, &count) in raw_counts.iter().enumerate() {
            if count > 0 {
                remap[idx] = levels.len();
                let mut rem = idx;
                for k in (0..labels.len()).rev() {
                    let c = rem % dims[k];
                    rem /= dims[k];
                    parts[k] = labels[k][c].as_str();
                }
                levels.push(parts.join(sep));
                counts.push(count);
            }
        }
        let out_codes = flat_idx
            .into_iter()
            .map(|entry| entry.map(|idx| remap[idx]))
            .collect();
        return Ok(StrataResult {
            codes: out_codes,
            levels,
            counts,
        });
    }

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

#[cfg(feature = "python")]
fn extract_strata_codes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Option<usize>>> {
    use numpy::{PyArrayMethods, PyUntypedArray};
    if obj.is_exact_instance_of::<PyUntypedArray>() {
        if let Ok(arr) = obj.cast::<numpy::PyArray1<i64>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice
                    .iter()
                    .map(|&v| if v < 0 { None } else { Some(v as usize) })
                    .collect());
            }
        }
        if let Ok(arr) = obj.cast::<numpy::PyArray1<i32>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice
                    .iter()
                    .map(|&v| if v < 0 { None } else { Some(v as usize) })
                    .collect());
            }
        }
        if let Ok(arr) = obj.cast::<numpy::PyArray1<usize>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice.iter().map(|&v| Some(v)).collect());
            }
        }
        if let Ok(arr) = obj.cast::<numpy::PyArray1<u32>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice.iter().map(|&v| Some(v as usize)).collect());
            }
        }
        if let Ok(arr) = obj.cast::<numpy::PyArray1<i16>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice
                    .iter()
                    .map(|&v| if v < 0 { None } else { Some(v as usize) })
                    .collect());
            }
        }
        if let Ok(arr) = obj.cast::<numpy::PyArray1<i8>>() {
            let ro = arr.readonly();
            if let Ok(slice) = ro.as_slice() {
                return Ok(slice
                    .iter()
                    .map(|&v| if v < 0 { None } else { Some(v as usize) })
                    .collect());
            }
        }
    }
    obj.extract::<Vec<Option<usize>>>()
}

#[cfg(not(feature = "python"))]
fn extract_strata_codes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Option<usize>>> {
    obj.extract::<Vec<Option<usize>>>()
}

#[doc = " Python entry point of [`strata`]: parallel lists of names, level labels"]
#[doc = " and per-observation codes."]
#[pyfunction(name = "strata")]
#[pyo3(signature = (names, levels, codes, na_group=false, shortlabel=false, sep=", "))]
pub(crate) fn strata_py(
    py: Python<'_>,
    names: Vec<String>,
    levels: Vec<Vec<String>>,
    codes: Vec<Bound<'_, PyAny>>,
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
    let parsed_codes: Vec<Vec<Option<usize>>> = codes
        .iter()
        .map(extract_strata_codes)
        .collect::<PyResult<_>>()?;
    let variables: Vec<StrataVariable> = names
        .into_iter()
        .zip(levels)
        .zip(parsed_codes)
        .map(|((name, levels), codes)| StrataVariable {
            name,
            levels,
            codes,
        })
        .collect();
    Ok(py.detach(|| strata(&variables, na_group, shortlabel, sep))?)
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
