use pyo3::prelude::*;
use std::collections::BTreeMap;

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
}
