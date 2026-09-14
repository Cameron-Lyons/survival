use std::fmt;

pub use crate::internal::validation::ValidationError;

/// Crate-wide error type.
///
/// Variants are grouped by how R's `survival` package would report the same
/// condition: argument problems become `stop()` calls (mapped to Python
/// `ValueError`), while numerical failures such as singular information
/// matrices or exhausted iteration budgets are runtime failures (mapped to
/// Python `RuntimeError`).
#[derive(Debug, Clone, PartialEq)]
pub enum SurvivalError {
    /// A caller-supplied argument is malformed in a way that has no
    /// structured [`ValidationError`] representation.
    InvalidInput(String),
    /// A structured input validation failure (see
    /// `crate::internal::validation`).
    Validation(ValidationError),
    /// A linear system or matrix is singular. `context` names the operation
    /// (for example `"information matrix"`), `columns` lists the zero-based
    /// columns found to be redundant when that is known (R's `coxph` reports
    /// those coefficients as `NA`).
    Singular {
        context: String,
        columns: Vec<usize>,
    },
    /// A numerical routine failed for a reason other than singularity.
    Computation(String),
    /// An iterative fit reached its iteration cap without converging. Wording
    /// follows R's `coxph`/`survreg`: "Ran out of iterations and did not
    /// converge".
    NotConverged { iterations: usize },
}

impl SurvivalError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    pub fn computation(message: impl Into<String>) -> Self {
        Self::Computation(message.into())
    }

    /// A singular system with no column-level diagnosis.
    pub fn singular(context: impl Into<String>) -> Self {
        Self::Singular {
            context: context.into(),
            columns: Vec::new(),
        }
    }

    /// A singular system whose redundant (zero-based) columns are known.
    pub fn singular_columns(context: impl Into<String>, columns: Vec<usize>) -> Self {
        Self::Singular {
            context: context.into(),
            columns,
        }
    }

    pub fn not_converged(iterations: usize) -> Self {
        Self::NotConverged { iterations }
    }
}

impl fmt::Display for SurvivalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Computation(message) => f.write_str(message),
            Self::Validation(err) => err.fmt(f),
            Self::Singular { context, columns } => {
                write!(f, "{context}: system is computationally singular")?;
                if !columns.is_empty() {
                    write!(f, " (redundant column")?;
                    if columns.len() > 1 {
                        f.write_str("s")?;
                    }
                    f.write_str(" ")?;
                    for (position, column) in columns.iter().enumerate() {
                        if position > 0 {
                            f.write_str(", ")?;
                        }
                        write!(f, "{column}")?;
                    }
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::NotConverged { iterations } => write!(
                f,
                "Ran out of iterations and did not converge (after {iterations} iterations)"
            ),
        }
    }
}

impl std::error::Error for SurvivalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(err) => Some(err),
            _ => None,
        }
    }
}

pub type SurvivalResult<T> = Result<T, SurvivalError>;

impl From<ValidationError> for SurvivalError {
    fn from(err: ValidationError) -> Self {
        Self::Validation(err)
    }
}

impl From<SurvivalError> for pyo3::PyErr {
    fn from(err: SurvivalError) -> Self {
        let message = err.to_string();
        match err {
            SurvivalError::InvalidInput(_) | SurvivalError::Validation(_) => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            SurvivalError::Singular { .. }
            | SurvivalError::Computation(_)
            | SurvivalError::NotConverged { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_errors_wrap_structurally_and_keep_their_message() {
        let err: SurvivalError = ValidationError::NonFinite {
            name: "time".to_string(),
            index: 2,
            value: f64::INFINITY,
        }
        .into();
        assert!(matches!(err, SurvivalError::Validation(_)));
        assert_eq!(
            err.to_string(),
            "time contains non-finite value inf at index 2"
        );
    }

    #[test]
    fn singular_errors_list_redundant_columns() {
        assert_eq!(
            SurvivalError::singular("information matrix").to_string(),
            "information matrix: system is computationally singular"
        );
        assert_eq!(
            SurvivalError::singular_columns("information matrix", vec![1]).to_string(),
            "information matrix: system is computationally singular (redundant column 1)"
        );
        assert_eq!(
            SurvivalError::singular_columns("X'X", vec![0, 2]).to_string(),
            "X'X: system is computationally singular (redundant columns 0, 2)"
        );
    }

    #[test]
    fn not_converged_uses_r_wording() {
        assert_eq!(
            SurvivalError::not_converged(20).to_string(),
            "Ran out of iterations and did not converge (after 20 iterations)"
        );
    }

    #[test]
    fn python_mapping_splits_value_and_runtime_errors() {
        crate::tests::common::initialize_python();
        let value_err: pyo3::PyErr = SurvivalError::invalid_input("bad input").into();
        assert!(value_err.to_string().contains("bad input"));
        let runtime_err: pyo3::PyErr = SurvivalError::singular("x").into();
        assert!(runtime_err.to_string().contains("singular"));
    }
}
