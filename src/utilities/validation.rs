use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use std::fmt;
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
    InvalidValue {
        field: &'static str,
        message: String,
    },
    NegativeValue {
        field: &'static str,
        index: usize,
        value: f64,
    },
    InvalidStatus {
        index: usize,
        value: f64,
    },
    NaNValue {
        field: &'static str,
        index: usize,
    },
    InfiniteValue {
        field: &'static str,
        index: usize,
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
            ValidationError::InvalidValue { field, message } => {
                write!(f, "invalid value for {}: {}", field, message)
            }
            ValidationError::NegativeValue {
                field,
                index,
                value,
            } => write!(
                f,
                "{} contains negative value {} at index {}",
                field, value, index
            ),
            ValidationError::InvalidStatus { index, value } => {
                write!(f, "status must be 0 or 1, got {} at index {}", value, index)
            }
            ValidationError::NaNValue { field, index } => {
                write!(f, "{} contains NaN at index {}", field, index)
            }
            ValidationError::InfiniteValue { field, index } => {
                write!(f, "{} contains infinite value at index {}", field, index)
            }
        }
    }
}
impl std::error::Error for ValidationError {}
impl From<ValidationError> for PyErr {
    fn from(err: ValidationError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
pub fn validate_length(
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
pub fn validate_non_empty<T>(slice: &[T], field: &'static str) -> Result<(), ValidationError> {
    if slice.is_empty() {
        return Err(ValidationError::EmptyInput { field });
    }
    Ok(())
}
pub fn validate_non_negative(slice: &[f64], field: &'static str) -> Result<(), ValidationError> {
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
pub fn validate_no_nan(slice: &[f64], field: &'static str) -> Result<(), ValidationError> {
    for (i, &val) in slice.iter().enumerate() {
        if val.is_nan() {
            return Err(ValidationError::NaNValue { field, index: i });
        }
    }
    Ok(())
}
pub fn clamp_probability(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}
