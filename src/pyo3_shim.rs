use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

pub mod prelude {
    pub use crate::{Bound, Py, PyAny, PyErr, PyModule, PyRefMut, PyResult, Python};
    pub use survival_pyo3_macros_shim::{pyclass, pyfunction, pymethods, pymodule};
}

pub mod types {
    pub use crate::{PyDict, PyList};
}

pub mod exceptions {
    use crate::PyErr;

    macro_rules! define_exception {
        ($name:ident) => {
            pub struct $name;

            impl $name {
                pub fn new_err(message: impl Into<String>) -> PyErr {
                    PyErr::new::<Self, _>(message)
                }
            }
        };
    }

    define_exception!(PyIndexError);
    define_exception!(PyKeyError);
    define_exception!(PyNotImplementedError);
    define_exception!(PyRuntimeError);
    define_exception!(PyTypeError);
    define_exception!(PyValueError);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyErr {
    message: String,
}

impl PyErr {
    pub fn new<E, S>(message: S) -> Self
    where
        S: Into<String>,
    {
        let _ = std::any::type_name::<E>();
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
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

#[derive(Clone, Copy, Debug, Default)]
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
}

impl<'py> Python<'py> {
    pub fn detach<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }
}

#[derive(Debug)]
pub struct Py<T: ?Sized> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Clone for Py<T> {
    fn clone(&self) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Py<T> {
    pub fn new(_py: Python<'_>, _value: T) -> PyResult<Self> {
        Ok(Self {
            _marker: PhantomData,
        })
    }
}

impl<T: ?Sized> Py<T> {
    pub fn bind<'py>(&self, _py: Python<'py>) -> Bound<'py, T> {
        Bound {
            _marker: PhantomData,
        }
    }
}

impl Py<PyAny> {
    pub fn call<A>(
        &self,
        _py: Python<'_>,
        _args: A,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Err(PyErr::new::<exceptions::PyRuntimeError, _>(
            "Python callbacks are unavailable without the `python` feature",
        ))
    }

    pub fn extract<T>(&self, _py: Python<'_>) -> PyResult<T> {
        Err(PyErr::new::<exceptions::PyTypeError, _>(
            "Python extraction is unavailable without the `python` feature",
        ))
    }
}

impl<T: ?Sized> From<Bound<'_, T>> for Py<T> {
    fn from(_value: Bound<'_, T>) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl From<Py<PyDict>> for Py<PyAny> {
    fn from(_value: Py<PyDict>) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl From<Bound<'_, PyDict>> for Py<PyAny> {
    fn from(_value: Bound<'_, PyDict>) -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct Bound<'py, T: ?Sized> {
    _marker: PhantomData<&'py T>,
}

impl<T: ?Sized> Clone for Bound<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Bound<'_, T> {}

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

#[derive(Debug)]
pub struct PyList;

#[derive(Debug)]
pub struct PyModule;

#[derive(Debug)]
pub struct PyType;

impl PyDict {
    pub fn new<'py>(_py: Python<'py>) -> Bound<'py, PyDict> {
        Bound {
            _marker: PhantomData,
        }
    }
}

impl PyList {
    pub fn new<'py, I>(_py: Python<'py>, _iter: I) -> PyResult<Bound<'py, PyList>>
    where
        I: IntoIterator,
    {
        Ok(Bound {
            _marker: PhantomData,
        })
    }
}

impl<'py> Bound<'py, PyDict> {
    pub fn keys(&self) -> Bound<'py, PyList> {
        Bound {
            _marker: PhantomData,
        }
    }

    pub fn set_item<K, V>(&self, _key: K, _value: V) -> PyResult<()> {
        Ok(())
    }

    pub fn get_item<K>(&self, _key: K) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(None)
    }
}

impl<'py> Bound<'py, PyList> {
    pub fn as_any(&self) -> Bound<'py, PyAny> {
        Bound {
            _marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        0
    }

    pub fn is_empty(&self) -> bool {
        true
    }

    pub fn iter(&self) -> std::iter::Empty<Bound<'py, PyAny>> {
        std::iter::empty()
    }
}

impl<'py> Bound<'py, PyAny> {
    pub fn call<A, K>(&self, _args: A, _kwargs: Option<&K>) -> PyResult<Bound<'py, PyAny>> {
        Err(PyErr::new::<exceptions::PyRuntimeError, _>(
            "Python callbacks are unavailable without the `python` feature",
        ))
    }

    pub fn call0(&self) -> PyResult<Bound<'py, PyAny>> {
        Err(PyErr::new::<exceptions::PyRuntimeError, _>(
            "Python objects are unavailable without the `python` feature",
        ))
    }

    pub fn cast<T>(&self) -> PyResult<Bound<'py, T>> {
        Err(PyErr::new::<exceptions::PyTypeError, _>(
            "Python casts are unavailable without the `python` feature",
        ))
    }

    pub fn extract<T>(&self) -> PyResult<T> {
        Err(PyErr::new::<exceptions::PyTypeError, _>(
            "Python extraction is unavailable without the `python` feature",
        ))
    }

    pub fn getattr(&self, _name: &str) -> PyResult<Bound<'py, PyAny>> {
        Err(PyErr::new::<exceptions::PyTypeError, _>(
            "Python attributes are unavailable without the `python` feature",
        ))
    }

    pub fn get_type(&self) -> Bound<'py, PyType> {
        Bound {
            _marker: PhantomData,
        }
    }
}

impl Bound<'_, PyType> {
    pub fn name(&self) -> PyResult<String> {
        Ok("unavailable".to_string())
    }
}

impl Bound<'_, PyModule> {
    pub fn add_function<T>(&self, _function: T) -> PyResult<()> {
        Ok(())
    }

    pub fn add_class<T>(&self) -> PyResult<()> {
        Ok(())
    }
}

#[macro_export]
macro_rules! wrap_pyfunction {
    ($function:path, $module:expr) => {{
        let _ = &$module;
        Ok($function)
    }};
}
