//! Canonical input validation helpers.
//!
//! Every check reports a structured [`ValidationError`]; callers convert it
//! into [`crate::SurvivalError`] or `PyErr` (`ValueError`) with `?`. Field
//! names are `&str` so callers may pass either literals or formatted names
//! (for example `"survival_predictions row 3"`); the name is only copied into
//! the error on the failure path.

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use std::fmt;

/// Structured description of an input validation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    LengthMismatch {
        expected: usize,
        got: usize,
        name: String,
    },
    Empty {
        name: String,
    },
    Negative {
        name: String,
        index: usize,
        value: f64,
    },
    NonPositive {
        name: String,
        index: usize,
        value: f64,
    },
    NaN {
        name: String,
        index: usize,
    },
    NonFinite {
        name: String,
        index: usize,
        value: f64,
    },
    NonBinary {
        name: String,
        index: usize,
        value: String,
    },
    /// A value outside the probability interval. `open` distinguishes the
    /// open interval `(0, 1)` from the closed `[0, 1]`.
    OutOfRange {
        name: String,
        index: usize,
        value: f64,
        open: bool,
    },
    /// `index` is the first position whose value is smaller than its
    /// predecessor.
    Unsorted {
        name: String,
        index: usize,
    },
    /// A flattened matrix whose length disagrees with its declared shape.
    ShapeMismatch {
        name: String,
        n_rows: usize,
        n_cols: usize,
        got: usize,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                expected,
                got,
                name,
            } => write!(f, "{name} length mismatch: expected {expected}, got {got}"),
            Self::Empty { name } => write!(f, "{name} cannot be empty"),
            Self::Negative { name, index, value } => {
                write!(f, "{name} contains negative value {value} at index {index}")
            }
            Self::NonPositive { name, index, value } => write!(
                f,
                "{name} must contain positive values; got {value} at index {index}"
            ),
            Self::NaN { name, index } => write!(f, "{name} contains NaN at index {index}"),
            Self::NonFinite { name, index, value } => write!(
                f,
                "{name} contains non-finite value {value} at index {index}"
            ),
            Self::NonBinary { name, index, value } => write!(
                f,
                "{name} values must be 0 or 1; {name} must contain only 0/1 values; got {value} at index {index}"
            ),
            Self::OutOfRange {
                name,
                index,
                value,
                open,
            } => {
                let qualifier = if *open { "strictly " } else { "" };
                write!(
                    f,
                    "{name} must contain probabilities {qualifier}between 0 and 1; got {value} at index {index}"
                )
            }
            Self::Unsorted { name, index } => write!(
                f,
                "{name} must be sorted in non-decreasing order; value at index {index} is smaller than its predecessor"
            ),
            Self::ShapeMismatch {
                name,
                n_rows,
                n_cols,
                got,
            } => match n_rows.checked_mul(*n_cols) {
                Some(expected) => write!(
                    f,
                    "{name} must have {n_rows} x {n_cols} = {expected} entries, got {got}"
                ),
                None => write!(f, "{name} shape {n_rows} x {n_cols} overflows usize"),
            },
        }
    }
}

impl std::error::Error for ValidationError {}

impl From<ValidationError> for PyErr {
    fn from(err: ValidationError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

/// Which probability interval [`validate_probability`] enforces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbabilityBounds {
    /// `[0, 1]`: predicted risks, survival probabilities.
    Closed,
    /// `(0, 1)`: confidence levels, quantile positions.
    Open,
}

pub(crate) fn validate_length(
    expected: usize,
    got: usize,
    name: &str,
) -> Result<(), ValidationError> {
    if expected != got {
        return Err(ValidationError::LengthMismatch {
            expected,
            got,
            name: name.to_string(),
        });
    }
    Ok(())
}

/// Checks that every `(name, len)` pair has the same length as the first one.
///
/// The first entry is the reference; the error names the first entry that
/// disagrees with it.
pub(crate) fn validate_equal_len(lengths: &[(&str, usize)]) -> Result<(), ValidationError> {
    let Some(&(_, expected)) = lengths.first() else {
        return Ok(());
    };
    for &(name, got) in &lengths[1..] {
        validate_length(expected, got, name)?;
    }
    Ok(())
}

pub(crate) fn validate_non_empty<T>(slice: &[T], name: &str) -> Result<(), ValidationError> {
    if slice.is_empty() {
        return Err(ValidationError::Empty {
            name: name.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn validate_finite(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if !value.is_finite() {
            return Err(ValidationError::NonFinite {
                name: name.to_string(),
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Rejects `NaN` only (infinities pass).
///
/// Almost every caller pairs this with [`validate_finite`], which already
/// rejects `NaN`; there it is redundant and only survives because tests
/// assert on the `"{name} contains NaN"` wording. It is the right check only
/// where an infinite value is legitimate (`surv2data` timelines). New code
/// should call [`validate_finite`] alone unless infinities are allowed.
pub(crate) fn validate_no_nan(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value.is_nan() {
            return Err(ValidationError::NaN {
                name: name.to_string(),
                index,
            });
        }
    }
    Ok(())
}

fn nan_error(name: &str, index: usize) -> ValidationError {
    ValidationError::NaN {
        name: name.to_string(),
        index,
    }
}

/// Every value must be `>= 0`; `NaN` is reported as [`ValidationError::NaN`].
/// Infinite values pass; pair with [`validate_finite`] when they must not.
pub(crate) fn validate_non_negative(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value.is_nan() {
            return Err(nan_error(name, index));
        }
        if value < 0.0 {
            return Err(ValidationError::Negative {
                name: name.to_string(),
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Every value must be `> 0`; `NaN` is reported as [`ValidationError::NaN`].
/// Infinite values pass; pair with [`validate_finite`] when they must not.
pub(crate) fn validate_positive(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value.is_nan() {
            return Err(nan_error(name, index));
        }
        if value <= 0.0 {
            return Err(ValidationError::NonPositive {
                name: name.to_string(),
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Every value must lie in `[0, 1]` (`Closed`) or `(0, 1)` (`Open`); `NaN` is
/// reported as [`ValidationError::NaN`].
pub(crate) fn validate_probability(
    slice: &[f64],
    name: &str,
    bounds: ProbabilityBounds,
) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value.is_nan() {
            return Err(nan_error(name, index));
        }
        let inside = match bounds {
            ProbabilityBounds::Closed => (0.0..=1.0).contains(&value),
            ProbabilityBounds::Open => value > 0.0 && value < 1.0,
        };
        if !inside {
            return Err(ValidationError::OutOfRange {
                name: name.to_string(),
                index,
                value,
                open: bounds == ProbabilityBounds::Open,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_binary_i32(slice: &[i32], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(ValidationError::NonBinary {
                name: name.to_string(),
                index,
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_binary_f64(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value != 0.0 && value != 1.0 {
            return Err(ValidationError::NonBinary {
                name: name.to_string(),
                index,
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

/// Values must be non-decreasing. `NaN` compares false and therefore passes;
/// pair with [`validate_finite`] when the data may contain `NaN`.
// Canonical helper for routines that require pre-sorted times (survfit,
// pyears); callers currently open-code the check and are expected to migrate.
#[allow(dead_code)]
pub(crate) fn validate_sorted(slice: &[f64], name: &str) -> Result<(), ValidationError> {
    for (index, pair) in slice.windows(2).enumerate() {
        if pair[1] < pair[0] {
            return Err(ValidationError::Unsorted {
                name: name.to_string(),
                index: index + 1,
            });
        }
    }
    Ok(())
}

/// A flattened matrix must hold exactly `n_rows * n_cols` entries.
pub(crate) fn validate_matrix_shape(
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
    name: &str,
) -> Result<(), ValidationError> {
    if n_rows.checked_mul(n_cols) != Some(values.len()) {
        return Err(ValidationError::ShapeMismatch {
            name: name.to_string(),
            n_rows,
            n_cols,
            got: values.len(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermutationIndexError {
    Negative { position: usize, value: i32 },
    OutOfBounds { position: usize, value: String },
    Duplicate { position: usize, value: usize },
}

fn mark_permutation_index(
    seen: &mut [bool],
    position: usize,
    zero_based_value: usize,
    display_value: usize,
) -> Result<(), PermutationIndexError> {
    if zero_based_value >= seen.len() {
        return Err(PermutationIndexError::OutOfBounds {
            position,
            value: display_value.to_string(),
        });
    }
    if seen[zero_based_value] {
        return Err(PermutationIndexError::Duplicate {
            position,
            value: display_value,
        });
    }
    seen[zero_based_value] = true;
    Ok(())
}

pub(crate) fn validate_zero_based_usize_permutation(
    values: &[usize],
    n: usize,
) -> Result<(), PermutationIndexError> {
    let mut seen = vec![false; n];
    for (position, &value) in values.iter().enumerate() {
        mark_permutation_index(&mut seen, position, value, value)?;
    }
    Ok(())
}

pub(crate) fn validate_zero_based_i32_permutation(
    values: &[i32],
    n: usize,
) -> Result<(), PermutationIndexError> {
    let mut seen = vec![false; n];
    for (position, &value) in values.iter().enumerate() {
        if value < 0 {
            return Err(PermutationIndexError::Negative { position, value });
        }
        let value = value as usize;
        mark_permutation_index(&mut seen, position, value, value)?;
    }
    Ok(())
}

pub(crate) fn validate_one_based_i32_permutation(
    values: &[i32],
    n: usize,
) -> Result<Vec<usize>, PermutationIndexError> {
    let mut seen = vec![false; n];
    let mut normalized = Vec::with_capacity(values.len());
    for (position, &raw_value) in values.iter().enumerate() {
        if raw_value < 1 {
            return Err(PermutationIndexError::OutOfBounds {
                position,
                value: raw_value.to_string(),
            });
        }
        let zero_based_value = raw_value as usize - 1;
        mark_permutation_index(&mut seen, position, zero_based_value, raw_value as usize)?;
        normalized.push(zero_based_value);
    }
    Ok(normalized)
}

pub(crate) fn validate_non_overlapping_intervals_i32(
    id: &[i32],
    start: &[f64],
    stop: &[f64],
    epsilon: f64,
) -> Result<(), PyErr> {
    let mut order: Vec<usize> = (0..id.len()).collect();
    order.sort_by(|&a, &b| {
        id[a]
            .cmp(&id[b])
            .then_with(|| start[a].total_cmp(&start[b]))
            .then_with(|| stop[a].total_cmp(&stop[b]))
            .then_with(|| a.cmp(&b))
    });

    for pair in order.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if id[previous] == id[current] && start[current] < stop[previous] - epsilon {
            return Err(PyValueError::new_err(format!(
                "intervals must not overlap within id; id {} rows {} and {} overlap",
                id[current],
                previous + 1,
                current + 1
            )));
        }
    }

    Ok(())
}

/// Finite values in `[0, 1]`, reported as `PyErr` for binding-level callers.
pub(crate) fn validate_probability_slice(values: &[f64], name: &str) -> Result<(), PyErr> {
    validate_finite(values, name)?;
    validate_probability(values, name, ProbabilityBounds::Closed)?;
    Ok(())
}

/// Finite values `> 0`, reported as `PyErr` for binding-level callers.
pub(crate) fn validate_positive_finite_slice(values: &[f64], name: &str) -> Result<(), PyErr> {
    validate_finite(values, name)?;
    validate_positive(values, name)?;
    Ok(())
}

pub(crate) fn validate_confidence_level(confidence_level: f64) -> Result<(), PyErr> {
    if !confidence_level.is_finite() || confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(PyValueError::new_err(
            "confidence_level must be a finite value between 0 and 1",
        ));
    }
    Ok(())
}

pub(crate) fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_helpers_report_the_offending_name() {
        assert!(validate_length(3, 3, "status").is_ok());
        assert_eq!(
            validate_length(3, 2, "status").unwrap_err(),
            ValidationError::LengthMismatch {
                expected: 3,
                got: 2,
                name: "status".to_string(),
            }
        );

        assert!(validate_equal_len(&[]).is_ok());
        assert!(validate_equal_len(&[("time", 4), ("status", 4), ("weights", 4)]).is_ok());
        let err = validate_equal_len(&[("time", 4), ("status", 4), ("weights", 3)]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "weights length mismatch: expected 4, got 3"
        );
    }

    #[test]
    fn non_empty_rejects_empty_slices() {
        assert!(validate_non_empty(&[1.0], "time").is_ok());
        assert_eq!(
            validate_non_empty::<f64>(&[], "time")
                .unwrap_err()
                .to_string(),
            "time cannot be empty"
        );
    }

    #[test]
    fn finite_and_nan_checks_report_index_and_value() {
        assert!(validate_finite(&[0.0, -1.5, 2.0], "x").is_ok());
        let err = validate_finite(&[0.0, f64::NEG_INFINITY], "x").unwrap_err();
        assert_eq!(
            err,
            ValidationError::NonFinite {
                name: "x".to_string(),
                index: 1,
                value: f64::NEG_INFINITY,
            }
        );
        assert_eq!(
            err.to_string(),
            "x contains non-finite value -inf at index 1"
        );

        let nan = validate_finite(&[f64::NAN], "x").unwrap_err();
        assert!(matches!(nan, ValidationError::NonFinite { index: 0, .. }));

        assert!(validate_no_nan(&[f64::INFINITY], "x").is_ok());
        assert_eq!(
            validate_no_nan(&[1.0, f64::NAN], "x")
                .unwrap_err()
                .to_string(),
            "x contains NaN at index 1"
        );
    }

    #[test]
    fn sign_checks_distinguish_negative_zero_and_nan() {
        assert!(validate_non_negative(&[0.0, 1.0, f64::INFINITY], "w").is_ok());
        assert_eq!(
            validate_non_negative(&[1.0, -0.5], "w").unwrap_err(),
            ValidationError::Negative {
                name: "w".to_string(),
                index: 1,
                value: -0.5,
            }
        );
        assert!(matches!(
            validate_non_negative(&[f64::NAN], "w").unwrap_err(),
            ValidationError::NaN { index: 0, .. }
        ));

        assert!(validate_positive(&[0.1, 2.0], "t").is_ok());
        assert_eq!(
            validate_positive(&[1.0, 0.0], "t").unwrap_err().to_string(),
            "t must contain positive values; got 0 at index 1"
        );
        assert!(matches!(
            validate_positive(&[f64::NAN], "t").unwrap_err(),
            ValidationError::NaN { index: 0, .. }
        ));
    }

    #[test]
    fn probability_checks_honor_open_and_closed_bounds() {
        assert!(validate_probability(&[0.0, 0.5, 1.0], "p", ProbabilityBounds::Closed).is_ok());
        assert!(validate_probability(&[0.0], "p", ProbabilityBounds::Open).is_err());
        assert!(validate_probability(&[1.0], "p", ProbabilityBounds::Open).is_err());
        assert!(validate_probability(&[0.5], "p", ProbabilityBounds::Open).is_ok());

        let err = validate_probability(&[0.5, 1.5], "p", ProbabilityBounds::Closed).unwrap_err();
        assert_eq!(
            err,
            ValidationError::OutOfRange {
                name: "p".to_string(),
                index: 1,
                value: 1.5,
                open: false,
            }
        );
        assert_eq!(
            err.to_string(),
            "p must contain probabilities between 0 and 1; got 1.5 at index 1"
        );
        assert_eq!(
            validate_probability(&[1.0], "p", ProbabilityBounds::Open)
                .unwrap_err()
                .to_string(),
            "p must contain probabilities strictly between 0 and 1; got 1 at index 0"
        );
        assert!(matches!(
            validate_probability(&[f64::NAN], "p", ProbabilityBounds::Closed).unwrap_err(),
            ValidationError::NaN { .. }
        ));

        assert!(validate_probability_slice(&[0.2], "p").is_ok());
        assert!(validate_probability_slice(&[f64::NAN], "p").is_err());
        assert!(validate_probability_slice(&[-0.1], "p").is_err());
        assert!(validate_positive_finite_slice(&[0.2], "se").is_ok());
        assert!(validate_positive_finite_slice(&[f64::INFINITY], "se").is_err());
        assert!(validate_positive_finite_slice(&[0.0], "se").is_err());
    }

    #[test]
    fn binary_checks_accept_only_zero_and_one() {
        assert!(validate_binary_i32(&[0, 1, 1], "status").is_ok());
        let err = validate_binary_i32(&[0, 2], "status").unwrap_err();
        assert_eq!(
            err,
            ValidationError::NonBinary {
                name: "status".to_string(),
                index: 1,
                value: "2".to_string(),
            }
        );
        assert!(err.to_string().contains("status values must be 0 or 1"));
        assert!(
            err.to_string()
                .contains("status must contain only 0/1 values")
        );

        assert!(validate_binary_f64(&[0.0, 1.0], "status").is_ok());
        assert!(validate_binary_f64(&[0.5], "status").is_err());
    }

    #[test]
    fn sorted_check_reports_first_descent() {
        assert!(validate_sorted(&[], "t").is_ok());
        assert!(validate_sorted(&[1.0, 1.0, 2.0], "t").is_ok());
        assert_eq!(
            validate_sorted(&[1.0, 3.0, 2.0], "t").unwrap_err(),
            ValidationError::Unsorted {
                name: "t".to_string(),
                index: 2,
            }
        );
    }

    #[test]
    fn matrix_shape_check_uses_row_times_column_count() {
        assert!(validate_matrix_shape(&[1.0; 6], 2, 3, "x").is_ok());
        let err = validate_matrix_shape(&[1.0; 5], 2, 3, "x").unwrap_err();
        assert_eq!(
            err,
            ValidationError::ShapeMismatch {
                name: "x".to_string(),
                n_rows: 2,
                n_cols: 3,
                got: 5,
            }
        );
        assert_eq!(err.to_string(), "x must have 2 x 3 = 6 entries, got 5");
        assert!(validate_matrix_shape(&[], usize::MAX, 2, "x").is_err());
    }

    #[test]
    fn errors_convert_to_value_errors() {
        crate::tests::common::initialize_python();
        let err: PyErr = ValidationError::Empty {
            name: "time".to_string(),
        }
        .into();
        assert!(err.to_string().contains("time cannot be empty"));
    }

    #[test]
    fn zero_based_usize_permutation_rejects_invalid_indices() {
        assert!(validate_zero_based_usize_permutation(&[2, 0, 1], 3).is_ok());

        let out_of_bounds = validate_zero_based_usize_permutation(&[0, 3, 1], 3)
            .expect_err("out-of-bounds index should fail");
        assert_eq!(
            out_of_bounds,
            PermutationIndexError::OutOfBounds {
                position: 1,
                value: "3".to_string(),
            }
        );

        let duplicate = validate_zero_based_usize_permutation(&[0, 0, 1], 3)
            .expect_err("duplicate index should fail");
        assert_eq!(
            duplicate,
            PermutationIndexError::Duplicate {
                position: 1,
                value: 0,
            }
        );
    }

    #[test]
    fn zero_based_i32_permutation_rejects_invalid_indices() {
        assert!(validate_zero_based_i32_permutation(&[2, 0, 1], 3).is_ok());

        let negative = validate_zero_based_i32_permutation(&[0, -1, 1], 3)
            .expect_err("negative index should fail");
        assert_eq!(
            negative,
            PermutationIndexError::Negative {
                position: 1,
                value: -1,
            }
        );

        let out_of_bounds = validate_zero_based_i32_permutation(&[0, 3, 1], 3)
            .expect_err("out-of-bounds index should fail");
        assert_eq!(
            out_of_bounds,
            PermutationIndexError::OutOfBounds {
                position: 1,
                value: "3".to_string(),
            }
        );
    }

    #[test]
    fn one_based_i32_permutation_normalizes_valid_indices() {
        assert_eq!(
            validate_one_based_i32_permutation(&[3, 1, 2], 3).unwrap(),
            vec![2, 0, 1]
        );

        let zero = validate_one_based_i32_permutation(&[1, 0, 3], 3)
            .expect_err("zero one-based index should fail");
        assert_eq!(
            zero,
            PermutationIndexError::OutOfBounds {
                position: 1,
                value: "0".to_string(),
            }
        );

        let duplicate = validate_one_based_i32_permutation(&[1, 1, 3], 3)
            .expect_err("duplicate one-based index should fail");
        assert_eq!(
            duplicate,
            PermutationIndexError::Duplicate {
                position: 1,
                value: 1,
            }
        );
    }
}
