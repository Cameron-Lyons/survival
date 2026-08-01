use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::collections::HashMap;

#[derive(Debug, Clone)]
#[pyclass(from_py_object)]
pub struct StrataResult {
    #[pyo3(get)]
    pub strata: Vec<i32>,
    #[pyo3(get)]
    pub levels: Vec<String>,
    #[pyo3(get)]
    pub counts: Vec<usize>,
    #[pyo3(get)]
    pub n_strata: usize,
}

fn strata_internal<T, F>(variables: &[Vec<T>], format_label: F) -> Result<StrataResult, String>
where
    T: Clone + Eq + Ord,
    F: Fn(&[T]) -> String,
{
    if variables.is_empty() {
        return Ok(StrataResult {
            strata: vec![],
            levels: vec![],
            counts: vec![],
            n_strata: 0,
        });
    }

    let n = variables[0].len();
    for (i, var) in variables.iter().enumerate() {
        if var.len() != n {
            return Err(format!(
                "Variable {} has length {} but expected {}",
                i,
                var.len(),
                n
            ));
        }
    }

    if n == 0 {
        return Ok(StrataResult {
            strata: vec![],
            levels: vec![],
            counts: vec![],
            n_strata: 0,
        });
    }

    let mut row_keys = Vec::with_capacity(n);
    let mut strata_map: BTreeMap<Vec<T>, i32> = BTreeMap::new();

    for row in 0..n {
        let key: Vec<T> = variables.iter().map(|var| var[row].clone()).collect();
        row_keys.push(key.clone());
        strata_map.entry(key).or_insert(0);
    }

    let mut levels = Vec::with_capacity(strata_map.len());
    for (stratum_id, (key, value)) in strata_map.iter_mut().enumerate() {
        *value = stratum_id as i32;
        levels.push(format_label(key));
    }

    let n_strata = strata_map.len();
    let mut strata = Vec::with_capacity(n);
    for key in &row_keys {
        let Some(&stratum_id) = strata_map.get(key) else {
            return Err("internal strata key missing from level map".to_string());
        };
        strata.push(stratum_id);
    }
    let mut counts = vec![0usize; n_strata];
    for &s in &strata {
        counts[s as usize] += 1;
    }

    Ok(StrataResult {
        strata,
        levels,
        counts,
        n_strata,
    })
}

#[pyfunction]
pub fn strata(variables: Vec<Vec<i64>>) -> PyResult<StrataResult> {
    strata_internal(&variables, |key| {
        key.iter()
            .enumerate()
            .map(|(j, v)| format!("v{}={}", j + 1, v))
            .collect::<Vec<_>>()
            .join(", ")
    })
    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

#[pyfunction]
pub fn strata_str(variables: Vec<Vec<String>>) -> PyResult<StrataResult> {
    strata_internal(&variables, |key| key.join(", "))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

type CompactStrataResult = (Vec<Option<usize>>, Vec<Vec<usize>>, Vec<usize>);

fn validate_strata_codes(
    variables: &[Vec<Option<usize>>],
    level_counts: &[usize],
) -> Result<usize, String> {
    if variables.len() != level_counts.len() {
        return Err(format!(
            "level_counts length ({}) must match variable count ({})",
            level_counts.len(),
            variables.len()
        ));
    }
    let n = variables.first().map_or(0, Vec::len);
    for (column_idx, (variable, &level_count)) in variables.iter().zip(level_counts).enumerate() {
        if variable.len() != n {
            return Err(format!(
                "Variable {} has length {} but expected {}",
                column_idx,
                variable.len(),
                n
            ));
        }
        if let Some((row_idx, part)) = variable.iter().enumerate().find_map(|(row_idx, part)| {
            part.filter(|&part| part >= level_count)
                .map(|part| (row_idx, part))
        }) {
            return Err(format!(
                "strata code {} at variable {} row {} exceeds level count {}",
                part, column_idx, row_idx, level_count
            ));
        }
    }
    Ok(n)
}

fn compact_strata_radix(
    variables: &[Vec<Option<usize>>],
    level_counts: &[usize],
    n: usize,
) -> CompactStrataResult {
    let mut raw_codes = vec![None; n];
    let mut observed = BTreeMap::<usize, (usize, usize)>::new();
    for row_idx in 0..n {
        let mut raw_code = 0usize;
        let mut complete = true;
        for (variable, &level_count) in variables.iter().zip(level_counts) {
            let Some(part) = variable[row_idx] else {
                complete = false;
                break;
            };
            raw_code = raw_code * level_count + part;
        }
        if !complete {
            continue;
        }
        raw_codes[row_idx] = Some(raw_code);
        observed
            .entry(raw_code)
            .and_modify(|(_, count)| *count += 1)
            .or_insert((row_idx, 1));
    }

    let mut compact_by_raw = HashMap::with_capacity(observed.len());
    let mut observed_parts = Vec::with_capacity(observed.len());
    let mut counts = Vec::with_capacity(observed.len());
    for (compact_idx, (raw_code, (row_idx, count))) in observed.into_iter().enumerate() {
        compact_by_raw.insert(raw_code, compact_idx + 1);
        observed_parts.push(
            variables
                .iter()
                .map(|variable| variable[row_idx].expect("complete strata row"))
                .collect(),
        );
        counts.push(count);
    }
    let codes = raw_codes
        .into_iter()
        .map(|raw_code| raw_code.map(|value| compact_by_raw[&value]))
        .collect();
    (codes, observed_parts, counts)
}

fn compact_strata_lexicographic(variables: &[Vec<Option<usize>>], n: usize) -> CompactStrataResult {
    let mut complete_rows = (0..n)
        .filter(|&row_idx| variables.iter().all(|variable| variable[row_idx].is_some()))
        .collect::<Vec<_>>();
    complete_rows.sort_unstable_by(|&left, &right| {
        variables
            .iter()
            .map(|variable| {
                variable[left]
                    .expect("complete strata row")
                    .cmp(&variable[right].expect("complete strata row"))
            })
            .find(|ordering| !ordering.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut codes = vec![None; n];
    let mut observed_parts = Vec::new();
    let mut counts = Vec::new();
    let mut previous_row = None;
    for row_idx in complete_rows {
        let starts_group = previous_row.is_none_or(|previous| {
            variables
                .iter()
                .any(|variable| variable[previous] != variable[row_idx])
        });
        if starts_group {
            observed_parts.push(
                variables
                    .iter()
                    .map(|variable| variable[row_idx].expect("complete strata row"))
                    .collect(),
            );
            counts.push(0);
        }
        let compact_code = observed_parts.len();
        codes[row_idx] = Some(compact_code);
        counts[compact_code - 1] += 1;
        previous_row = Some(row_idx);
    }
    (codes, observed_parts, counts)
}

fn compact_strata_codes(
    variables: &[Vec<Option<usize>>],
    level_counts: &[usize],
) -> Result<CompactStrataResult, String> {
    let n = validate_strata_codes(variables, level_counts)?;
    let radix_fits = level_counts.iter().try_fold(1usize, |product, &levels| {
        product.checked_mul(levels.max(1))
    });
    Ok(if radix_fits.is_some() {
        compact_strata_radix(variables, level_counts, n)
    } else {
        compact_strata_lexicographic(variables, n)
    })
}

#[pyfunction]
pub fn strata_compact(
    py: Python<'_>,
    variables: Vec<Vec<Option<usize>>>,
    level_counts: Vec<usize>,
) -> PyResult<CompactStrataResult> {
    py.detach(move || compact_strata_codes(&variables, &level_counts))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strata_single_var() {
        let vars = vec![vec![2, 1, 2, 3, 1]];
        let result = strata(vars).unwrap();
        assert_eq!(result.n_strata, 3);
        assert_eq!(result.strata, vec![1, 0, 1, 2, 0]);
        assert_eq!(result.levels, vec!["v1=1", "v1=2", "v1=3"]);
        assert_eq!(result.counts, vec![2, 2, 1]);
    }

    #[test]
    fn test_strata_two_vars() {
        let vars = vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2]];
        let result = strata(vars).unwrap();
        assert_eq!(result.n_strata, 4);
        assert_eq!(
            result.levels,
            vec!["v1=1, v2=1", "v1=1, v2=2", "v1=2, v2=1", "v1=2, v2=2"]
        );
        assert_eq!(result.counts, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_strata_string_levels_are_sorted() {
        let vars = vec![vec!["b".to_string(), "a".to_string(), "b".to_string()]];
        let result = strata_str(vars).unwrap();

        assert_eq!(result.strata, vec![1, 0, 1]);
        assert_eq!(result.levels, vec!["a", "b"]);
        assert_eq!(result.counts, vec![1, 2]);
    }

    #[test]
    fn test_strata_empty() {
        let vars: Vec<Vec<i64>> = vec![];
        let result = strata(vars).unwrap();
        assert_eq!(result.n_strata, 0);
    }

    #[test]
    fn test_strata_length_mismatch() {
        let vars = vec![vec![1, 2, 3], vec![1, 2]];
        assert!(strata(vars).is_err());
    }

    #[test]
    fn test_strata_compact_matches_r_style_codes() {
        let variables = vec![
            vec![Some(1), Some(0), Some(1), None],
            vec![Some(1), Some(0), Some(0), Some(0)],
        ];
        let (codes, parts, counts) = compact_strata_codes(&variables, &[2, 2]).unwrap();

        assert_eq!(codes, vec![Some(3), Some(1), Some(2), None]);
        assert_eq!(parts, vec![vec![0, 0], vec![1, 0], vec![1, 1]]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_strata_compact_uses_lexicographic_overflow_fallback() {
        let variables = vec![
            vec![Some(1), Some(0), Some(1)],
            vec![Some(0), Some(1), Some(0)],
        ];
        let (codes, parts, counts) =
            compact_strata_codes(&variables, &[usize::MAX, usize::MAX]).unwrap();

        assert_eq!(codes, vec![Some(2), Some(1), Some(2)]);
        assert_eq!(parts, vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(counts, vec![1, 2]);
    }

    #[test]
    fn test_strata_compact_validates_codes_and_shapes() {
        let err = compact_strata_codes(&[vec![Some(0)], vec![]], &[1, 1]).unwrap_err();
        assert!(err.contains("length 0 but expected 1"));

        let err = compact_strata_codes(&[vec![Some(1)]], &[1]).unwrap_err();
        assert!(err.contains("exceeds level count 1"));

        let err = compact_strata_codes(&[vec![None]], &[]).unwrap_err();
        assert!(err.contains("must match variable count"));
    }
}
