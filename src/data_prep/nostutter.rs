//! R's `nostutter` (`R/xtras.R`): in timeline data, turn a repeat of the
//! subject's current state into "censored", so that a visit at which
//! nothing changed is not counted as a new transition.
//!
//! The result is the set of rows whose state should be replaced by the
//! censoring code; the caller applies it to values of any type.  R's
//! `single = TRUE` variant (each state may be entered once per subject)
//! is provided although R's own implementation of it does not run.

use super::id_value::{IdValue, SubjectId};
use crate::error::SurvivalResult;
use crate::internal::validation::validate_length;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Rows (as a mask) whose state is a stutter of the current state.
/// `state[i]` is `None` for a missing value, which neither replaces nor
/// changes the current state.
pub fn nostutter<I: SubjectId, S: SubjectId>(
    id: &[I],
    state: &[Option<S>],
    censor: &S,
    single: bool,
) -> SurvivalResult<Vec<bool>> {
    validate_length(id.len(), state.len(), "state")?;
    let censor = censor.key();
    let mut replace = vec![false; id.len()];
    let mut used: HashSet<S::Key> = HashSet::new();
    let mut current = censor.clone();
    let mut previous: Option<I::Key> = None;

    for (row, (subject, value)) in id.iter().zip(state).enumerate() {
        let subject = subject.key();
        let value = value.as_ref().map(SubjectId::key);
        if previous.as_ref() != Some(&subject) {
            used.clear();
            current = value.clone().unwrap_or_else(|| censor.clone());
            if single && current != censor {
                used.insert(current.clone());
            }
        } else if let Some(value) = value {
            if value == current || (single && used.contains(&value)) {
                replace[row] = true;
            } else if value != censor {
                if single {
                    used.insert(value.clone());
                }
                current = value;
            }
        }
        previous = Some(subject);
    }
    Ok(replace)
}

/// Python entry point of [`nostutter`].
#[pyfunction(name = "nostutter")]
#[pyo3(signature = (id, state, censor, single=false))]
pub fn nostutter_py(
    id: Vec<IdValue>,
    state: Vec<Option<IdValue>>,
    censor: IdValue,
    single: bool,
) -> PyResult<Vec<bool>> {
    if id.iter().any(IdValue::is_missing) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "id must not contain missing values",
        ));
    }
    let state: Vec<Option<IdValue>> = state
        .into_iter()
        .map(|value| value.filter(|v| !v.is_missing()))
        .collect();
    Ok(nostutter(&id, &state, &censor, single)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_adjacent_repeated_states_within_subjects() {
        let result = nostutter(
            &[1i64, 1, 1, 2, 2],
            &[Some(0i64), Some(1), Some(1), Some(1), Some(1)],
            &0,
            false,
        )
        .unwrap();
        assert_eq!(result, vec![false, false, true, false, true]);
    }

    #[test]
    fn single_mode_suppresses_states_already_used_by_each_subject() {
        let result = nostutter(
            &[1i64, 1, 1, 1, 2, 2, 2],
            &[
                Some(1i64),
                Some(2),
                Some(1),
                Some(3),
                Some(1),
                Some(1),
                Some(2),
            ],
            &0,
            true,
        )
        .unwrap();
        assert_eq!(result, vec![false, false, true, false, false, true, false]);
    }

    #[test]
    fn censor_and_missing_states_do_not_change_the_current_state() {
        let result = nostutter(
            &["a", "a", "a", "a", "a"],
            &[None, Some("x"), None, Some("censor"), Some("x")],
            &"censor",
            true,
        )
        .unwrap();
        assert_eq!(result, vec![false, false, false, false, true]);
    }

    #[test]
    fn validates_parallel_inputs() {
        assert!(nostutter(&[1i64], &[] as &[Option<i64>], &0, false).is_err());
    }
}
