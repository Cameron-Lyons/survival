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
