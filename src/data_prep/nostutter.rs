use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum NumericValue {
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Integer(i128),
    #[cfg_attr(feature = "python", pyo3(transparent))]
    Float(f64),
}

impl NumericValue {
    fn integral_value(self) -> Option<i128> {
        match self {
            Self::Integer(value) => Some(value),
            Self::Float(value)
                if value.is_finite()
                    && value.fract() == 0.0
                    && value >= i128::MIN as f64
                    && value < i128::MAX as f64 =>
            {
                let integer = value as i128;
                ((integer as f64) == value).then_some(integer)
            }
            Self::Float(_) => None,
        }
    }

    fn is_nan(self) -> bool {
        matches!(self, Self::Float(value) if value.is_nan())
    }

    fn normalized_float_bits(self) -> u64 {
        match self {
            Self::Float(value) if value.is_nan() => f64::NAN.to_bits(),
            Self::Float(0.0) => 0,
            Self::Float(value) => value.to_bits(),
            Self::Integer(_) => unreachable!("integer values use their exact hash"),
        }
    }
}

impl PartialEq for NumericValue {
    fn eq(&self, other: &Self) -> bool {
        match (self.integral_value(), other.integral_value()) {
            (Some(left), Some(right)) => left == right,
            (None, None) => self.normalized_float_bits() == other.normalized_float_bits(),
            _ => false,
        }
    }
}

impl Eq for NumericValue {}

impl Hash for NumericValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(value) = self.integral_value() {
            0u8.hash(state);
            value.hash(state);
        } else {
            1u8.hash(state);
            self.normalized_float_bits().hash(state);
        }
    }
}

fn replacement_mask<I, S>(
    id: &[I],
    state: &[Option<S>],
    censor: &S,
    single: bool,
) -> PyResult<Vec<bool>>
where
    I: PartialEq,
    S: Eq + Hash,
{
    if state.len() != id.len() {
        return Err(PyValueError::new_err(
            "state must have the same length as id",
        ));
    }

    let mut replacements = vec![false; id.len()];
    let mut used: HashSet<&S> = HashSet::new();
    let mut current = censor;
    let mut previous_id = None;

    for (row, (subject, value)) in id.iter().zip(state).enumerate() {
        if previous_id != Some(subject) {
            used.clear();
            current = value.as_ref().unwrap_or(censor);
            if single && value.is_some() && current != censor {
                used.insert(current);
            }
        } else if let Some(value) = value {
            if value == current || (single && used.contains(value)) {
                replacements[row] = true;
            } else if value != censor {
                current = value;
                if single {
                    used.insert(value);
                }
            }
        }
        previous_id = Some(subject);
    }

    Ok(replacements)
}

fn numeric_state_keys(state: &[Option<NumericValue>]) -> Vec<Option<NumericValue>> {
    state
        .iter()
        .map(|value| value.and_then(|value| (!value.is_nan()).then_some(value)))
        .collect()
}

fn numeric_replacements<I: PartialEq>(
    id: Vec<I>,
    state: Vec<Option<NumericValue>>,
    censor: NumericValue,
    single: bool,
) -> PyResult<Vec<bool>> {
    let keys = numeric_state_keys(&state);
    replacement_mask(&id, &keys, &censor, single)
}

#[pyfunction]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_replacements(
    id: Vec<usize>,
    state: Vec<Option<usize>>,
    censor: usize,
    single: bool,
) -> PyResult<Vec<bool>> {
    replacement_mask(&id, &state, &censor, single)
}

#[pyfunction]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_numeric_numeric(
    py: Python<'_>,
    id: Vec<NumericValue>,
    state: Vec<Option<NumericValue>>,
    censor: NumericValue,
    single: bool,
) -> PyResult<Vec<bool>> {
    if id.iter().any(|value| value.is_nan()) {
        return Err(PyValueError::new_err("id must not contain missing values"));
    }
    py.detach(move || numeric_replacements(id, state, censor, single))
}

#[pyfunction]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_numeric_str(
    py: Python<'_>,
    id: Vec<NumericValue>,
    state: Vec<Option<String>>,
    censor: String,
    single: bool,
) -> PyResult<Vec<bool>> {
    if id.iter().any(|value| value.is_nan()) {
        return Err(PyValueError::new_err("id must not contain missing values"));
    }
    py.detach(move || replacement_mask(&id, &state, &censor, single))
}

#[pyfunction]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_str_numeric(
    py: Python<'_>,
    id: Vec<String>,
    state: Vec<Option<NumericValue>>,
    censor: NumericValue,
    single: bool,
) -> PyResult<Vec<bool>> {
    py.detach(move || numeric_replacements(id, state, censor, single))
}

#[pyfunction]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_str_str(
    py: Python<'_>,
    id: Vec<String>,
    state: Vec<Option<String>>,
    censor: String,
    single: bool,
) -> PyResult<Vec<bool>> {
    py.detach(move || replacement_mask(&id, &state, &censor, single))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_adjacent_repeated_states_within_subjects() {
        let result = nostutter_replacements(
            vec![1, 1, 1, 2, 2],
            vec![Some(0), Some(1), Some(1), Some(1), Some(1)],
            0,
            false,
        )
        .unwrap();

        assert_eq!(result, vec![false, false, true, false, true]);
    }

    #[test]
    fn single_mode_suppresses_states_already_used_by_each_subject() {
        let result = nostutter_replacements(
            vec![1, 1, 1, 1, 2, 2, 2],
            vec![
                Some(1),
                Some(2),
                Some(1),
                Some(3),
                Some(1),
                Some(1),
                Some(2),
            ],
            0,
            true,
        )
        .unwrap();

        assert_eq!(result, vec![false, false, true, false, false, true, false]);
    }

    #[test]
    fn censor_and_missing_states_do_not_replace_the_current_state() {
        let result = nostutter_replacements(
            vec![1, 1, 1, 1, 1],
            vec![None, Some(1), None, Some(0), Some(1)],
            0,
            true,
        )
        .unwrap();

        assert_eq!(result, vec![false, false, false, false, true]);
    }

    #[test]
    fn numeric_values_preserve_large_integers_and_signed_zero_ties() {
        let result = numeric_replacements(
            vec![1, 1, 1, 1],
            vec![
                Some(NumericValue::Float(-0.0)),
                Some(NumericValue::Integer(0)),
                Some(NumericValue::Integer(9_007_199_254_740_992)),
                Some(NumericValue::Integer(9_007_199_254_740_993)),
            ],
            NumericValue::Integer(9),
            false,
        )
        .unwrap();

        assert_eq!(result, vec![false, true, false, false]);
    }

    #[test]
    fn string_values_return_the_reference_replacement_mask() {
        let result = replacement_mask(
            &["a", "a", "a"],
            &[
                Some("x".to_owned()),
                Some("x".to_owned()),
                Some("y".to_owned()),
            ],
            &"censor".to_owned(),
            false,
        )
        .unwrap();

        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(nostutter_replacements(vec![1], vec![], 0, false).is_err());
    }
}
