//! Stand-in for the `pyo3` crate in Rust-only builds (`--no-default-features`).
//!
//! # Contract
//!
//! `lib.rs` declares `extern crate self as pyo3;` when the `python` feature is
//! off, so every `pyo3::...` path inside the crate resolves to the items this
//! module re-exports through the crate root (`crate::PyErr`,
//! `crate::exceptions`, `crate::prelude`, ...). The shim exists so domain code
//! annotated with `#[pyclass]`/`#[pyfunction]` compiles and its Rust tests run
//! without an interpreter. It is deliberately small and mirrors PyO3's
//! signatures exactly, so code that compiles here also compiles against PyO3:
//!
//! - the attribute macros come from `survival-pyo3-macros-shim`; they are
//!   no-ops that strip `#[pyo3(...)]`, `#[new]`, `#[getter]`, `#[setter]` and
//!   `#[staticmethod]` helper attributes;
//! - [`PyErr`] records the exception class ([`PyErrKind`]) and the message, and
//!   its `Display` prints `"<Class>: <message>"` exactly as PyO3 does, so error
//!   assertions behave identically in both builds;
//! - [`Python`] is `Copy`, [`Bound`] and [`Py`] are `Clone` but not `Copy`,
//!   [`Py::bind`] hands out a reference and [`Python::detach`] requires `Send`
//!   closures, matching PyO3's ownership rules;
//! - [`PyAny`] and [`PyDict`] are inert placeholders: constructing a dict and
//!   `set_item` succeed but store nothing, and every operation that would need
//!   an interpreter (`call`, `extract`) fails with `RuntimeError`/`TypeError`.
//!   Code that must inspect or build Python objects belongs behind
//!   `#[cfg(feature = "python")]`, or better, behind a typed `#[pyclass]`.
//!
//! Anything not listed here is unsupported on purpose: add an item only when a
//! Rust-only build needs it, and copy PyO3's signature when you do.

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

/// The Python exception class a [`PyErr`] would raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyErrKind {
    IndexError,
    NotImplementedError,
    RuntimeError,
    TypeError,
    ValueError,
}

impl PyErrKind {
    /// The Python class name, as PyO3's `Display` for `PyErr` prints it.
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

/// Marker trait for the exception classes in [`exceptions`]; plays the role of
/// PyO3's `PyTypeInfo` bound on `PyErr::new`.
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

    /// The exception class this error would raise in Python.
    pub fn kind(&self) -> PyErrKind {
        self.kind
    }

    /// Mirrors `PyErr::is_instance_of::<E>(py)`; exact class match only, since
    /// the shim has no class hierarchy.
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

/// Mutable borrow of a `#[pyclass]` value. Only PyO3 can construct one, so
/// `#[pymethods]` builders taking `PyRefMut<Self>` compile but are not
/// callable from Rust-only tests.
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

    /// Runs `f` "without the GIL"; the `Send` bounds stand in for PyO3's
    /// `Ungil`, which is `Send` on stable toolchains.
    pub fn detach<F, T>(self, f: F) -> T
    where
        F: Send + FnOnce() -> T,
        T: Send,
    {
        f()
    }
}

/// Interpreter-independent handle. Holds no value: `Py::new` drops what it is
/// given, so a `Py<T>` built in a Rust-only test cannot be inspected.
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

/// Covariant in `'py` without requiring `T: 'py`, so `Py<T>` can hold a
/// `Bound<'static, T>` and lend it out for any `'py`.
type BoundMarker<'py, T> = PhantomData<(&'py (), fn() -> Box<T>)>;

/// Interpreter-bound handle; `Clone` but, like PyO3's, not `Copy`.
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
    /// Accepted and discarded: the shim dict stores nothing.
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
