//! Flag observations that are never at risk at an event time: the port of
//! `norisk` (`src/norisk.c`), a helper R's `survival` ships for the
//! accumulation routines (dropping such rows improves their accuracy).
//! Ported as written, including the index the C code carries over from the
//! previous iteration when it stores the running death count.

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{
    PermutationIndexError, validate_binary_i32, validate_finite, validate_length,
    validate_zero_based_i32_permutation,
};
use pyo3::prelude::*;

fn validate_sort_indices(values: &[i32], n: usize, name: &str) -> SurvivalResult<()> {
    match validate_zero_based_i32_permutation(values, n) {
        Ok(()) => Ok(()),
        Err(PermutationIndexError::Negative { position, value }) => {
            Err(SurvivalError::invalid_input(format!(
                "{name} index out of bounds at position {position}: {value}"
            )))
        }
        Err(PermutationIndexError::OutOfBounds { position, value }) => {
            Err(SurvivalError::invalid_input(format!(
                "{name} index out of bounds at position {position}: {value}"
            )))
        }
        Err(PermutationIndexError::Duplicate { position, value }) => {
            Err(SurvivalError::invalid_input(format!(
                "{name} must be a permutation of 0..{n}; duplicate index {value} at position {position}"
            )))
        }
    }
}

/// `strata` is either a 0/1 end-of-stratum marker per observation or the
/// strictly increasing positions at which a new stratum starts.
fn validate_strata_boundaries(values: &[i32], n: usize) -> SurvivalResult<()> {
    if values.len() == n && values.iter().all(|&value| value == 0 || value == 1) {
        return Ok(());
    }
    let mut previous = None;
    for (idx, &value) in values.iter().enumerate() {
        if value < 0 || value as usize > n {
            return Err(SurvivalError::invalid_input(format!(
                "strata values must be between 0 and {n}; got {value} at index {idx}"
            )));
        }
        if let Some(previous_value) = previous
            && value <= previous_value
        {
            return Err(SurvivalError::invalid_input(format!(
                "strata values must be strictly increasing; got {value} after {previous_value} at index {idx}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

/// Port of `norisk`: `sort1` orders the observations by start time and
/// `sort2` by stop time, both descending within strata as the caller of
/// the C routine arranges; `strata` gives the first index of each stratum.
/// Returns one flag per observation.
pub fn norisk_flags(
    time1: &[f64],
    time2: &[f64],
    status: &[i32],
    sort1: &[i32],
    sort2: &[i32],
    strata: &[i32],
) -> SurvivalResult<Vec<i32>> {
    let n = time1.len();
    validate_length(n, time2.len(), "time2")?;
    validate_length(n, status.len(), "status")?;
    validate_length(n, sort1.len(), "sort1")?;
    validate_length(n, sort2.len(), "sort2")?;
    validate_finite(time1, "time1")?;
    validate_finite(time2, "time2")?;
    validate_binary_i32(status, "status")?;
    validate_sort_indices(sort1, n, "sort1")?;
    validate_sort_indices(sort2, n, "sort2")?;
    validate_strata_boundaries(strata, n)?;

    let mut notused = vec![0; n];
    let mut ndeath = 0;
    let mut istrat = 0;
    let mut j = 0;
    let mut p1 = sort1.first().map_or(0, |&value| value as usize);
    for (i, &sort2_i) in sort2.iter().enumerate() {
        let p2 = sort2_i as usize;
        let dtime = time2[p2];
        if strata
            .get(istrat)
            .is_some_and(|&boundary| boundary as usize == i)
        {
            // first obs of a new stratum: finish off the old one
            while j < i {
                p1 = sort1[j] as usize;
                notused[p1] = i32::from(ndeath > notused[p1]);
                j += 1;
            }
            ndeath = 0;
            istrat += 1;
        } else {
            while j < i && time1[sort1[j] as usize] >= dtime {
                p1 = sort1[j] as usize;
                notused[p1] = i32::from(ndeath > notused[p1]);
                j += 1;
            }
        }
        ndeath += status[p2];
        notused[p1] = ndeath;
    }
    while j < n {
        let p = sort2[j] as usize;
        notused[p] = i32::from(ndeath > notused[p]);
        j += 1;
    }
    Ok(notused)
}

/// Python binding of [`norisk_flags`].  The Rust name differs from the
/// Python one because `#[pyfunction]` defines a module named after the
/// function, which would clash with this file's module when re-exported.
#[pyfunction(name = "norisk")]
pub fn norisk_py(
    time1: Vec<f64>,
    time2: Vec<f64>,
    status: Vec<i32>,
    sort1: Vec<i32>,
    sort2: Vec<i32>,
    strata: Vec<i32>,
) -> PyResult<Vec<i32>> {
    Ok(norisk_flags(
        &time1, &time2, &status, &sort1, &sort2, &strata,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norisk_rejects_malformed_inputs() {
        let err = norisk_flags(&[0.0, 1.0], &[1.0], &[1, 0], &[0, 1], &[0, 1], &[]).unwrap_err();
        assert!(err.to_string().contains("time2"));
        let err = norisk_flags(&[0.0], &[1.0], &[1], &[-1], &[0], &[]).unwrap_err();
        assert!(err.to_string().contains("sort1 index out of bounds"));
        let err =
            norisk_flags(&[0.0, 1.0], &[1.0, 2.0], &[1, 0], &[0, 0], &[0, 1], &[]).unwrap_err();
        assert!(err.to_string().contains("sort1 must be a permutation"));
        let err = norisk_flags(&[0.0], &[1.0], &[2], &[0], &[0], &[]).unwrap_err();
        assert!(err.to_string().contains("status must contain only 0/1"));
        let err = norisk_flags(
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0],
            &[1, 0, 1],
            &[0, 1, 2],
            &[0, 1, 2],
            &[2, 1],
        )
        .unwrap_err();
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn norisk_flags_observations_outside_every_event_risk_set() {
        // sorted by decreasing time: (3,4+], (2,3], (0,1]: the last interval
        // ends before the only death at 3 and is never at risk for it
        let time1 = [3.0, 2.0, 0.0];
        let time2 = [4.0, 3.0, 1.0];
        let status = [0, 1, 0];
        let result = norisk_flags(&time1, &time2, &status, &[0, 1, 2], &[0, 1, 2], &[0]).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[2], 1);
        let markers =
            norisk_flags(&time1, &time2, &status, &[0, 1, 2], &[0, 1, 2], &[1, 0, 0]).unwrap();
        assert_eq!(markers.len(), 3);
    }
}
