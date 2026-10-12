use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub mod prelude {
    pub use crate::{Bound, Py, PyAny, PyErr, PyRefMut, PyResult, Python};
    pub use survival_pyo3_macros_shim::{pyclass, pyfunction, pymethods};
}

pub mod types {
    pub use crate::PyDict;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyErrKind {
    IndexError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
}

impl PyErrKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::IndexError => "IndexError",
            Self::NotImplementedError => "NotImplementedError",
            Self::RuntimeError => "RuntimeError",
            Self::TypeError => "TypeError",
            Self::ValueError => "ValueError",
        }
    }
}

pub trait ExceptionClass {
    const KIND: PyErrKind;
}

pub mod exceptions {
    use super::{ExceptionClass, PyErr, PyErrKind};

    macro_rules! define_exception {
        ($name:ident, $kind:ident) => {
            pub struct $name;

            impl ExceptionClass for $name {
                const KIND: PyErrKind = PyErrKind::$kind;
            }

            impl $name {
                pub fn new_err(message: impl Into<String>) -> PyErr {
                    PyErr::new::<Self, _>(message)
                }
            }
        };
    }

    define_exception!(PyIndexError, IndexError);
    define_exception!(PyNotImplementedError, NotImplementedError);
    define_exception!(PyRuntimeError, RuntimeError);
    define_exception!(PyTypeError, TypeError);
    define_exception!(PyValueError, ValueError);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyErr {
    kind: PyErrKind,
    message: String,
}

impl PyErr {
    pub fn new<E, M>(message: M) -> Self
    where
        E: ExceptionClass,
        M: Into<String>,
    {
        Self {
            kind: E::KIND,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> PyErrKind {
        self.kind
    }

    pub fn is_instance_of<E: ExceptionClass>(&self, _py: Python<'_>) -> bool {
        self.kind == E::KIND
    }
}

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for PyErr {}

pub type PyResult<T> = Result<T, PyErr>;

pub struct PyRefMut<'py, T: ?Sized> {
    inner: &'py mut T,
}

impl<T: ?Sized> Deref for PyRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: ?Sized> DerefMut for PyRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Python<'py> {
    _marker: PhantomData<&'py ()>,
}

impl Python<'_> {
    pub fn attach<F, R>(f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        f(Python {
            _marker: PhantomData,
        })
    }

    pub fn initialize() {}

    pub fn detach<F, T>(self, f: F) -> T
    where
        F: Send + FnOnce() -> T,
        T: Send,
    {
        f()
    }
}

#[derive(Debug)]
pub struct Py<T: ?Sized> {
    bound: Bound<'static, T>,
}

impl<T: ?Sized> Clone for Py<T> {
    fn clone(&self) -> Self {
        Self {
            bound: self.bound.clone(),
        }
    }
}

impl<T: ?Sized> Py<T> {
    fn placeholder() -> Self {
        Self {
            bound: Bound {
                _marker: PhantomData,
            },
        }
    }

    pub fn bind<'py>(&self, _py: Python<'py>) -> &Bound<'py, T> {
        &self.bound
    }
}

impl<T> Py<T> {
    pub fn new(_py: Python<'_>, _value: T) -> PyResult<Self> {
        Ok(Self::placeholder())
    }
}

impl Py<PyAny> {
    pub fn call<A>(
        &self,
        _py: Python<'_>,
        _args: A,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Err(exceptions::PyRuntimeError::new_err(
            "Python callbacks are unavailable without the `python` feature",
        ))
    }

    pub fn extract<T>(&self, _py: Python<'_>) -> PyResult<T> {
        Err(exceptions::PyTypeError::new_err(
            "Python extraction is unavailable without the `python` feature",
        ))
    }
}

impl<T: ?Sized> From<Bound<'_, T>> for Py<T> {
    fn from(_value: Bound<'_, T>) -> Self {
        Self::placeholder()
    }
}

impl From<Py<PyDict>> for Py<PyAny> {
    fn from(_value: Py<PyDict>) -> Self {
        Self::placeholder()
    }
}

impl From<Bound<'_, PyDict>> for Py<PyAny> {
    fn from(_value: Bound<'_, PyDict>) -> Self {
        Self::placeholder()
    }
}

type BoundMarker<'py, T> = PhantomData<(&'py (), fn() -> Box<T>)>;

#[derive(Debug)]
pub struct Bound<'py, T: ?Sized> {
    _marker: BoundMarker<'py, T>,
}

impl<T: ?Sized> Clone for Bound<'_, T> {
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<'py, T: ?Sized> Bound<'py, T> {
    pub fn py(&self) -> Python<'py> {
        Python {
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct PyAny;

#[derive(Debug)]
pub struct PyDict;

impl PyDict {
    pub fn new(_py: Python<'_>) -> Bound<'_, PyDict> {
        Bound {
            _marker: PhantomData,
        }
    }
}

impl Bound<'_, PyDict> {
    pub fn set_item<K, V>(&self, _key: K, _value: V) -> PyResult<()> {
        Ok(())
    }
}

impl Bound<'_, PyAny> {
    pub fn extract<T>(&self) -> PyResult<T> {
        Err(exceptions::PyTypeError::new_err(
            "Python extraction is unavailable without the `python` feature",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
    use super::{Py, PyAny, PyDict, PyErr, PyErrKind, Python};

    #[test]
    fn errors_keep_their_class_and_print_like_pyo3() {
        let err = PyErr::new::<PyValueError, _>("time must be positive");
        assert_eq!(err.kind(), PyErrKind::ValueError);
        assert_eq!(err.to_string(), "ValueError: time must be positive");

        let err = PyRuntimeError::new_err(String::from("singular"));
        assert_eq!(err.kind(), PyErrKind::RuntimeError);
        assert_eq!(err.to_string(), "RuntimeError: singular");
        Python::attach(|py| {
            assert!(err.is_instance_of::<PyRuntimeError>(py));
            assert!(!err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn placeholders_fail_loudly_instead_of_inventing_values() {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            assert!(dict.set_item("k", 1).is_ok());
            let any: Py<PyAny> = Py::<PyDict>::from(dict).into();
            let err = any.extract::<f64>(py).unwrap_err();
            assert_eq!(err.kind(), PyErrKind::TypeError);
            let err = any.call(py, (), None).unwrap_err();
            assert_eq!(err.kind(), PyErrKind::RuntimeError);
        });
    }
}
