use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SurvivalError {
    InvalidInput(String),
    Computation(String),
}

impl SurvivalError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub fn computation(message: impl Into<String>) -> Self {
        Self::Computation(message.into())
    }
}

impl fmt::Display for SurvivalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Computation(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SurvivalError {}

pub type SurvivalResult<T> = Result<T, SurvivalError>;

impl From<crate::internal::validation::ValidationError> for SurvivalError {
    fn from(err: crate::internal::validation::ValidationError) -> Self {
        Self::invalid_input(err.to_string())
    }
}

impl From<SurvivalError> for pyo3::PyErr {
    fn from(err: SurvivalError) -> Self {
        match err {
            SurvivalError::InvalidInput(message) => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            SurvivalError::Computation(message) => {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            }
        }
    }
}
