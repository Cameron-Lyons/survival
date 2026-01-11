use pyo3::prelude::*;
use std::collections::HashMap;

/// Result of stratification variable creation
#[derive(Debug, Clone)]
#[pyclass]
pub struct StrataResult {
    /// Strata codes for each observation (0-indexed)
    #[pyo3(get)]
    pub strata: Vec<i32>,
    /// Labels for each stratum level
    #[pyo3(get)]
    pub levels: Vec<String>,
    /// Count of observations in each stratum
    #[pyo3(get)]
    pub counts: Vec<usize>,
    /// Number of strata
    #[pyo3(get)]
    pub n_strata: usize,
}

/// Create stratification variable from one or more factors.
///
/// This function creates a combined stratification variable from one or more
/// input vectors. In survival analysis, stratification allows fitting separate
/// baseline hazards for each stratum while assuming common covariate effects.
///
/// # Arguments
/// * `variables` - Vector of vectors, each representing a stratifying variable
///
/// # Returns
/// * `StrataResult` containing combined strata codes and level information
#[pyfunction]
pub fn strata(variables: Vec<Vec<i64>>) -> PyResult<StrataResult> {
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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Variable {} has length {} but expected {}",
                i,
                var.len(),
                n
            )));
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

    let mut strata_map: HashMap<Vec<i64>, i32> = HashMap::new();
    let mut strata = Vec::with_capacity(n);
    let mut levels = Vec::new();
    let mut current_stratum_id = 0i32;

    for i in 0..n {
        let key: Vec<i64> = variables.iter().map(|var| var[i]).collect();
        let stratum_id = *strata_map.entry(key.clone()).or_insert_with(|| {
            let id = current_stratum_id;
            let label = key
                .iter()
                .enumerate()
                .map(|(j, v)| format!("v{}={}", j + 1, v))
                .collect::<Vec<_>>()
                .join(", ");
            levels.push(label);
            current_stratum_id += 1;
            id
        });
        strata.push(stratum_id);
    }

    let n_strata = strata_map.len();

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

/// Create stratification variable from string factors
#[pyfunction]
pub fn strata_str(variables: Vec<Vec<String>>) -> PyResult<StrataResult> {
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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Variable {} has length {} but expected {}",
                i,
                var.len(),
                n
            )));
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

    let mut strata_map: HashMap<Vec<String>, i32> = HashMap::new();
    let mut strata = Vec::with_capacity(n);
    let mut levels = Vec::new();
    let mut current_stratum_id = 0i32;

    for i in 0..n {
        let key: Vec<String> = variables.iter().map(|var| var[i].clone()).collect();
        let stratum_id = *strata_map.entry(key.clone()).or_insert_with(|| {
            let id = current_stratum_id;
            let label = key.join(", ");
            levels.push(label);
            current_stratum_id += 1;
            id
        });
        strata.push(stratum_id);
    }

    let n_strata = strata_map.len();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strata_single_var() {
        let vars = vec![vec![1, 1, 2, 2, 3]];
        let result = strata(vars).unwrap();
        assert_eq!(result.n_strata, 3);
        assert_eq!(result.strata, vec![0, 0, 1, 1, 2]);
        assert_eq!(result.counts, vec![2, 2, 1]);
    }

    #[test]
    fn test_strata_two_vars() {
        let vars = vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2]];
        let result = strata(vars).unwrap();
        assert_eq!(result.n_strata, 4);
        // Each combination is unique
        assert_eq!(result.counts, vec![1, 1, 1, 1]);
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
