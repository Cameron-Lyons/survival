use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::fmt;

#[derive(Debug, Clone)]
pub(crate) enum MatrixError {
    SingularMatrix,
    EmptyMatrix,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::SingularMatrix => {
                write!(
                    f,
                    "matrix is singular or nearly singular and cannot be solved"
                )
            }
            MatrixError::EmptyMatrix => {
                write!(f, "matrix is empty when non-empty matrix was expected")
            }
        }
    }
}

impl std::error::Error for MatrixError {}

impl From<MatrixError> for PyErr {
    fn from(err: MatrixError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

#[derive(Debug)]
pub enum ValidationError {
    LengthMismatch {
        expected: usize,
        got: usize,
        field: &'static str,
    },
    EmptyInput {
        field: &'static str,
    },
    NegativeValue {
        field: &'static str,
        index: usize,
        value: f64,
    },
    NaNValue {
        field: &'static str,
        index: usize,
    },
    NonFiniteValue {
        field: &'static str,
        index: usize,
        value: f64,
    },
    NonBinaryValue {
        field: &'static str,
        index: usize,
        value: String,
    },
}
impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::LengthMismatch {
                expected,
                got,
                field,
            } => write!(
                f,
                "{} length mismatch: expected {}, got {}",
                field, expected, got
            ),
            ValidationError::EmptyInput { field } => write!(f, "{} cannot be empty", field),
            ValidationError::NegativeValue {
                field,
                index,
                value,
            } => write!(
                f,
                "{} contains negative value {} at index {}",
                field, value, index
            ),
            ValidationError::NaNValue { field, index } => {
                write!(f, "{} contains NaN at index {}", field, index)
            }
            ValidationError::NonFiniteValue {
                field,
                index,
                value,
            } => write!(
                f,
                "{} contains non-finite value {} at index {}",
                field, value, index
            ),
            ValidationError::NonBinaryValue {
                field,
                index,
                value,
            } => write!(
                f,
                "{} values must be 0 or 1; {} must contain only 0/1 values; got {} at index {}",
                field, field, value, index
            ),
        }
    }
}
impl std::error::Error for ValidationError {}
impl From<ValidationError> for PyErr {
    fn from(err: ValidationError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
pub(crate) fn validate_length(
    expected: usize,
    got: usize,
    field: &'static str,
) -> Result<(), ValidationError> {
    if expected != got {
        return Err(ValidationError::LengthMismatch {
            expected,
            got,
            field,
        });
    }
    Ok(())
}
pub(crate) fn validate_non_empty<T>(
    slice: &[T],
    field: &'static str,
) -> Result<(), ValidationError> {
    if slice.is_empty() {
        return Err(ValidationError::EmptyInput { field });
    }
    Ok(())
}
pub(crate) fn validate_non_negative(
    slice: &[f64],
    field: &'static str,
) -> Result<(), ValidationError> {
    for (i, &val) in slice.iter().enumerate() {
        if val < 0.0 {
            return Err(ValidationError::NegativeValue {
                field,
                index: i,
                value: val,
            });
        }
    }
    Ok(())
}
pub(crate) fn validate_no_nan(slice: &[f64], field: &'static str) -> Result<(), ValidationError> {
    for (i, &val) in slice.iter().enumerate() {
        if val.is_nan() {
            return Err(ValidationError::NaNValue { field, index: i });
        }
    }
    Ok(())
}
pub(crate) fn validate_finite(slice: &[f64], field: &'static str) -> Result<(), ValidationError> {
    for (i, &val) in slice.iter().enumerate() {
        if !val.is_finite() {
            return Err(ValidationError::NonFiniteValue {
                field,
                index: i,
                value: val,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_binary_i32(
    slice: &[i32],
    field: &'static str,
) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value != 0 && value != 1 {
            return Err(ValidationError::NonBinaryValue {
                field,
                index,
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_binary_f64(
    slice: &[f64],
    field: &'static str,
) -> Result<(), ValidationError> {
    for (index, &value) in slice.iter().enumerate() {
        if value != 0.0 && value != 1.0 {
            return Err(ValidationError::NonBinaryValue {
                field,
                index,
                value: value.to_string(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PermutationIndexError {
    Negative {
        position: usize,
        value: i32,
    },
    OutOfBounds {
        position: usize,
        value: String,
    },
    Duplicate {
        position: usize,
        value: usize,
    },
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

pub(crate) fn validate_probability_slice(values: &[f64], field: &str) -> Result<(), PyErr> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{field} contains non-finite value {value} at index {index}"
            )));
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(PyValueError::new_err(format!(
                "{field} must contain probabilities between 0 and 1; got {value} at index {index}"
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_positive_finite_slice(values: &[f64], field: &str) -> Result<(), PyErr> {
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{field} contains non-finite value {value} at index {index}"
            )));
        }
        if value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "{field} must contain positive values; got {value} at index {index}"
            )));
        }
    }
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
