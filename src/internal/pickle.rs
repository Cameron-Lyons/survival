//! Pickle support for the result classes that `survival.r` keeps.
//!
//! A class's `__reduce__` hands `pickle` (and `copy`) the module-level
//! `_survival._unpickle`, the class and the object encoded with serde and
//! bincode; `_unpickle` decodes it back.  The encoding follows the Rust
//! structs, so a pickle is only readable by the package version that wrote
//! it, like the pickles of most compiled extensions.

use ndarray::{Array, Dimension};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;

#[cfg(feature = "python")]
use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyBytes, PyType},
};

/// The `__reduce__` value of a result class: `(_unpickle, (cls, state))`.
#[cfg(feature = "python")]
pub(crate) type Reduced<'py> = (Bound<'py, PyAny>, (Bound<'py, PyType>, Bound<'py, PyBytes>));

/// `__reduce__` of the result class `T`: `value` encoded for [`decode`].
#[cfg(feature = "python")]
pub(crate) fn reduce<'py, T>(py: Python<'py>, value: &T) -> PyResult<Reduced<'py>>
where
    T: PyTypeInfo + Serialize,
{
    let state = bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let unpickle = py.import("survival._survival")?.getattr("_unpickle")?;
    Ok((unpickle, (T::type_object(py), PyBytes::new(py, &state))))
}

/// Decodes the state [`reduce`] wrote for a `T`.
#[cfg(feature = "python")]
pub(crate) fn decode<T>(py: Python<'_>, state: &[u8]) -> PyResult<T>
where
    T: serde::de::DeserializeOwned + PyTypeInfo,
{
    match bincode::serde::decode_from_slice(state, bincode::config::standard()) {
        Ok((value, read)) if read == state.len() => Ok(value),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "not a pickled {} of this version of survival",
            T::type_object(py).name()?
        ))),
    }
}

/// Serde adapter for an array, owned or behind an `Arc`, that keeps its
/// memory order: a column-major array (the influence matrices, laid out as
/// R's) comes back column-major, anything else in standard order.
pub(crate) mod memory_order {
    use super::*;

    pub(crate) fn serialize<A, D, S>(array: &A, serializer: S) -> Result<S::Ok, S::Error>
    where
        A: Borrow<Array<f64, D>>,
        D: Dimension + Serialize,
        S: Serializer,
    {
        let array = array.borrow();
        if !array.is_standard_layout() && array.t().is_standard_layout() {
            (true, array.t()).serialize(serializer)
        } else {
            (false, array.view()).serialize(serializer)
        }
    }

    pub(crate) fn deserialize<'de, A, D, De>(deserializer: De) -> Result<A, De::Error>
    where
        A: From<Array<f64, D>>,
        D: Dimension + Deserialize<'de>,
        De: Deserializer<'de>,
    {
        let (column_major, array): (bool, Array<f64, D>) = Deserialize::deserialize(deserializer)?;
        Ok(A::from(if column_major {
            array.reversed_axes()
        } else {
            array
        }))
    }

    /// The same for an optional array.
    pub(crate) mod option {
        use super::*;

        pub(crate) fn serialize<D, S>(
            array: &Option<Array<f64, D>>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            D: Dimension + Serialize,
            S: Serializer,
        {
            #[derive(Serialize)]
            struct Wrapped<'a, D: Dimension + Serialize>(
                #[serde(with = "super")] &'a Array<f64, D>,
            );
            array.as_ref().map(Wrapped).serialize(serializer)
        }

        pub(crate) fn deserialize<'de, D, De>(
            deserializer: De,
        ) -> Result<Option<Array<f64, D>>, De::Error>
        where
            D: Dimension + Deserialize<'de>,
            De: Deserializer<'de>,
        {
            #[derive(Deserialize)]
            #[serde(bound = "D: Dimension + Deserialize<'de>")]
            struct Wrapped<D: Dimension>(#[serde(with = "super")] Array<f64, D>);
            let array: Option<Wrapped<D>> = Deserialize::deserialize(deserializer)?;
            Ok(array.map(|Wrapped(array)| array))
        }
    }
}

/// A `#[pymethods]` block holding only `__reduce__`, for the classes that
/// have no other Python methods; the others define the same `__reduce__`
/// in their own block.
macro_rules! picklable {
    ($($class:ty),+ $(,)?) => {$(
        #[cfg(feature = "python")]
        #[pyo3::pymethods]
        impl $class {
            fn __reduce__<'py>(
                &self,
                py: pyo3::Python<'py>,
            ) -> pyo3::PyResult<$crate::internal::pickle::Reduced<'py>> {
                $crate::internal::pickle::reduce(py, self)
            }
        }
    )+};
}
pub(crate) use picklable;

#[cfg(test)]
mod tests {
    use super::memory_order;
    use ndarray::{Array2, Array3, ShapeBuilder};
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    #[derive(Serialize, Deserialize)]
    struct Arrays {
        #[serde(with = "memory_order")]
        column_major: Arc<Array3<f64>>,
        #[serde(with = "memory_order")]
        row_major: Array2<f64>,
        #[serde(with = "memory_order::option")]
        optional: Option<Array2<f64>>,
        #[serde(with = "memory_order::option")]
        absent: Option<Array2<f64>>,
    }

    #[test]
    fn arrays_keep_their_values_and_memory_order() {
        let column_major = Array3::from_shape_vec((2, 3, 2).f(), (0..12).map(f64::from).collect())
            .expect("12 values");
        let row_major =
            Array2::from_shape_vec((2, 3), (0..6).map(f64::from).collect()).expect("6 values");
        let arrays = Arrays {
            column_major: Arc::new(column_major.clone()),
            row_major: row_major.clone(),
            optional: Some(
                row_major
                    .t()
                    .as_standard_layout()
                    .into_owned()
                    .reversed_axes(),
            ),
            absent: None,
        };
        let json = serde_json::to_string(&arrays).expect("serializes");
        let back: Arrays = serde_json::from_str(&json).expect("deserializes");

        assert_eq!(*back.column_major, column_major);
        assert!(back.column_major.t().is_standard_layout());
        assert_eq!(back.row_major, row_major);
        assert!(back.row_major.is_standard_layout());
        let optional = back.optional.expect("present");
        assert_eq!(optional, row_major);
        assert!(optional.t().is_standard_layout());
        assert!(back.absent.is_none());
    }
}
