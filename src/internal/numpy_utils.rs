//! Boundary input and output types for `#[pyfunction]` signatures.
//!
//! [`FloatVec`], [`IntVec`], [`BoolVec`] and [`FloatMatrix`] accept, without a
//! Python-side `.tolist()`: NumPy arrays of any dtype and memory layout,
//! pandas/polars columns (anything with `__array__`) and plain sequences. Each
//! converts exactly once into the owned container core code consumes
//! (`Vec<f64>`, `Vec<i32>`, `Vec<bool>`, `Array2<f64>`). Returning one from a
//! `#[pyfunction]` hands the caller a NumPy array without copying
//! (`IntoPyArray`), and a `#[pyo3(get)]` field of these types reads back as a
//! NumPy array.
//!
//! Without the `python` feature the same newtypes exist as plain wrappers, so
//! core signatures can name them in both builds.

use std::ops::Deref;

use ndarray::Array2;

use crate::error::SurvivalResult;
use crate::internal::validation::{ValidationError, validate_matrix_shape};

/// One-dimensional `float64` input.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FloatVec(pub Vec<f64>);

/// One-dimensional `int32` input. Integer and boolean dtypes convert directly;
/// floating values are accepted only when integral and within `i32` range.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IntVec(pub Vec<i32>);

/// One-dimensional boolean input. Boolean dtypes convert directly; numeric
/// values are accepted only when they are exactly `0` or `1`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoolVec(pub Vec<bool>);

macro_rules! vector_newtype {
    ($name:ident, $elem:ty) => {
        impl $name {
            pub fn into_inner(self) -> Vec<$elem> {
                self.0
            }
        }

        impl Deref for $name {
            type Target = [$elem];

            fn deref(&self) -> &[$elem] {
                &self.0
            }
        }

        impl From<Vec<$elem>> for $name {
            fn from(values: Vec<$elem>) -> Self {
                Self(values)
            }
        }

        impl From<$name> for Vec<$elem> {
            fn from(values: $name) -> Self {
                values.0
            }
        }
    };
}

vector_newtype!(FloatVec, f64);
vector_newtype!(IntVec, i32);
vector_newtype!(BoolVec, bool);

/// Two-dimensional `float64` input (`n` rows by `p` columns), always held in
/// row-major (C) layout so `as_slice()` on the inner array never fails.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatMatrix(Array2<f64>);

impl FloatMatrix {
    /// Wraps an array, copying it into row-major layout when it is not
    /// already stored that way.
    pub fn new(values: Array2<f64>) -> Self {
        if values.is_standard_layout() {
            Self(values)
        } else {
            let (n, p) = values.dim();
            Self(
                Array2::from_shape_vec((n, p), values.iter().copied().collect())
                    .expect("row-major copy of an (n, p) array has n * p entries"),
            )
        }
    }

    /// Builds the matrix from equally long rows.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> SurvivalResult<Self> {
        let ncol = rows.first().map_or(0, Vec::len);
        for (index, row) in rows.iter().enumerate() {
            if row.len() != ncol {
                return Err(ValidationError::LengthMismatch {
                    expected: ncol,
                    got: row.len(),
                    name: format!("row {index}"),
                }
                .into());
            }
        }
        let nrow = rows.len();
        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Ok(Self(
            Array2::from_shape_vec((nrow, ncol), flat).expect("rows validated to share a length"),
        ))
    }

    /// Builds the matrix from a row-major flat buffer with `ncol` columns.
    pub fn from_flat(values: Vec<f64>, ncol: usize) -> SurvivalResult<Self> {
        let nrow = if ncol == 0 { 0 } else { values.len() / ncol };
        validate_matrix_shape(&values, nrow, ncol, "x")?;
        Ok(Self(
            Array2::from_shape_vec((nrow, ncol), values).expect("shape validated against length"),
        ))
    }

    pub fn nrow(&self) -> usize {
        self.0.nrows()
    }

    pub fn ncol(&self) -> usize {
        self.0.ncols()
    }

    pub fn into_inner(self) -> Array2<f64> {
        self.0
    }

    /// The row-major entries; never fails because of the layout invariant.
    pub fn as_flat(&self) -> &[f64] {
        self.0
            .as_slice()
            .expect("FloatMatrix is kept in row-major layout")
    }

    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        self.0.rows().into_iter().map(|row| row.to_vec()).collect()
    }
}

impl Deref for FloatMatrix {
    type Target = Array2<f64>;

    fn deref(&self) -> &Array2<f64> {
        &self.0
    }
}

impl From<Array2<f64>> for FloatMatrix {
    fn from(values: Array2<f64>) -> Self {
        Self::new(values)
    }
}

impl From<FloatMatrix> for Array2<f64> {
    fn from(values: FloatMatrix) -> Self {
        values.0
    }
}

#[cfg(feature = "python")]
mod python {
    //! `FromPyObject`/`IntoPyObject` for the boundary types.
    //!
    //! Extraction order, cheapest first: a NumPy array of the exact dtype is
    //! read through a read-only view (any strides); a `list`/`tuple` goes
    //! through PyO3's sequence extraction; everything else (other dtypes,
    //! pandas/polars columns, `array.array`, `range`, ...) is normalised with
    //! `numpy.asarray(obj, dtype=...)` and read as a NumPy array.

    use ndarray::Array2;
    use numpy::{
        Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArray,
        PyUntypedArrayMethods, ToPyArray, get_array_module,
    };
    use pyo3::Borrowed;
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList, PyTuple};
    use std::convert::Infallible;

    use super::{BoolVec, FloatMatrix, FloatVec, IntVec};

    fn type_error(obj: &Bound<'_, PyAny>, expected: &str, detail: Option<PyErr>) -> PyErr {
        let type_name = obj
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let detail = detail.map(|err| format!(": {err}")).unwrap_or_default();
        PyTypeError::new_err(format!(
            "cannot convert '{type_name}' to {expected}{detail}"
        ))
    }

    fn is_plain_sequence(obj: &Bound<'_, PyAny>) -> bool {
        obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>()
    }

    /// `numpy.asarray(obj, dtype=T)`.
    fn asarray<'py, T: Element>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = obj.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", numpy::dtype::<T>(py))?;
        get_array_module(py)?
            .getattr("asarray")?
            .call((obj,), Some(&kwargs))
    }

    /// Copies a one-dimensional array out in logical order, whatever its strides.
    fn read_1d<T: Element + Copy>(array: &Bound<'_, PyArray1<T>>) -> Vec<T> {
        let view = array.readonly();
        match view.as_slice() {
            Ok(slice) => slice.to_vec(),
            Err(_) => view.as_array().iter().copied().collect(),
        }
    }

    fn wrong_ndim(obj: &Bound<'_, PyAny>, expected: &str, ndim: usize) -> PyErr {
        type_error(
            obj,
            expected,
            Some(PyValueError::new_err(format!("got {ndim} dimension(s)"))),
        )
    }

    /// Any array-like as `float64` values, via `numpy.asarray` when needed.
    /// `expected` names the target type in error messages.
    fn float_values(obj: &Bound<'_, PyAny>, expected: &str) -> PyResult<Vec<f64>> {
        if let Ok(array) = obj.cast::<PyArray1<f64>>() {
            return Ok(read_1d(array));
        }
        if is_plain_sequence(obj) {
            return obj
                .extract::<Vec<f64>>()
                .map_err(|err| type_error(obj, expected, Some(err)));
        }
        let converted = asarray::<f64>(obj).map_err(|err| type_error(obj, expected, Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            1 => Ok(read_1d(converted.cast::<PyArray1<f64>>()?)),
            ndim => Err(wrong_ndim(obj, "a 1-dimensional array", ndim)),
        }
    }

    fn narrow_i64(obj: &Bound<'_, PyAny>, values: Vec<i64>) -> PyResult<Vec<i32>> {
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                i32::try_from(value).map_err(|_| {
                    type_error(
                        obj,
                        "an int32 array",
                        Some(PyValueError::new_err(format!(
                            "value {value} at index {index} is out of range"
                        ))),
                    )
                })
            })
            .collect()
    }

    fn integral_f64(obj: &Bound<'_, PyAny>, values: Vec<f64>) -> PyResult<Vec<i32>> {
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                if value.fract() == 0.0
                    && value >= f64::from(i32::MIN)
                    && value <= f64::from(i32::MAX)
                {
                    Ok(value as i32)
                } else {
                    Err(type_error(
                        obj,
                        "an int32 array",
                        Some(PyValueError::new_err(format!(
                            "value {value} at index {index} is not an int32"
                        ))),
                    ))
                }
            })
            .collect()
    }

    fn int_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        if let Ok(array) = obj.cast::<PyArray1<i32>>() {
            return Ok(read_1d(array));
        }
        if let Ok(array) = obj.cast::<PyArray1<i64>>() {
            return narrow_i64(obj, read_1d(array));
        }
        if let Ok(array) = obj.cast::<PyArray1<bool>>() {
            return Ok(read_1d(array).into_iter().map(i32::from).collect());
        }
        if is_plain_sequence(obj)
            && let Ok(values) = obj.extract::<Vec<i64>>()
        {
            return narrow_i64(obj, values);
        }
        integral_f64(obj, float_values(obj, "an int32 array")?)
    }

    fn bool_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        if let Ok(array) = obj.cast::<PyArray1<bool>>() {
            return Ok(read_1d(array));
        }
        if is_plain_sequence(obj)
            && let Ok(values) = obj.extract::<Vec<bool>>()
        {
            return Ok(values);
        }
        int_values(obj)?
            .into_iter()
            .enumerate()
            .map(|(index, value)| match value {
                0 => Ok(false),
                1 => Ok(true),
                other => Err(type_error(
                    obj,
                    "a boolean array",
                    Some(PyValueError::new_err(format!(
                        "value {other} at index {index} is not 0 or 1"
                    ))),
                )),
            })
            .collect()
    }

    /// Copies a two-dimensional array into a row-major `Array2`, whatever its
    /// layout. `as_slice` is only the memory order for C-contiguous arrays
    /// (NumPy also reports Fortran-contiguous arrays as contiguous), so every
    /// other layout is walked in logical order.
    fn read_2d(array: &Bound<'_, PyArray2<f64>>) -> Array2<f64> {
        let view = array.readonly();
        let (n, p) = view.as_array().dim();
        let flat = match view.as_slice() {
            Ok(slice) if array.is_c_contiguous() => slice.to_vec(),
            _ => view.as_array().iter().copied().collect(),
        };
        Array2::from_shape_vec((n, p), flat).expect("(n, p) view yields n * p entries")
    }

    fn matrix_values(obj: &Bound<'_, PyAny>) -> PyResult<FloatMatrix> {
        if let Ok(array) = obj.cast::<PyArray2<f64>>() {
            return Ok(FloatMatrix(read_2d(array)));
        }
        if is_plain_sequence(obj) {
            if let Ok(rows) = obj.extract::<Vec<Vec<f64>>>() {
                return FloatMatrix::from_rows(rows).map_err(PyErr::from);
            }
            return float_values(obj, "a float matrix").map(column_matrix);
        }
        let converted =
            asarray::<f64>(obj).map_err(|err| type_error(obj, "a float matrix", Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            2 => Ok(FloatMatrix(read_2d(converted.cast::<PyArray2<f64>>()?))),
            1 => Ok(column_matrix(read_1d(converted.cast::<PyArray1<f64>>()?))),
            ndim => Err(wrong_ndim(obj, "a float matrix", ndim)),
        }
    }

    /// A vector is a single-column matrix, as `as.matrix()` treats it in R.
    fn column_matrix(values: Vec<f64>) -> FloatMatrix {
        let n = values.len();
        FloatMatrix(Array2::from_shape_vec((n, 1), values).expect("n values fill an (n, 1) array"))
    }

    impl<'py> FromPyObject<'_, 'py> for FloatVec {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            float_values(&obj, "a float array").map(FloatVec)
        }
    }

    impl<'py> FromPyObject<'_, 'py> for IntVec {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            int_values(&obj).map(IntVec)
        }
    }

    impl<'py> FromPyObject<'_, 'py> for BoolVec {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            bool_values(&obj).map(BoolVec)
        }
    }

    impl<'py> FromPyObject<'_, 'py> for FloatMatrix {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            matrix_values(&obj)
        }
    }

    macro_rules! into_pyarray1 {
        ($name:ident, $elem:ty) => {
            impl<'py> IntoPyObject<'py> for $name {
                type Target = PyArray1<$elem>;
                type Output = Bound<'py, PyArray1<$elem>>;
                type Error = Infallible;

                /// Moves the buffer into a NumPy array without copying.
                fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
                    Ok(self.0.into_pyarray(py))
                }
            }

            impl<'py> IntoPyObject<'py> for &$name {
                type Target = PyArray1<$elem>;
                type Output = Bound<'py, PyArray1<$elem>>;
                type Error = Infallible;

                /// Copies the buffer; this is what a `#[pyo3(get)]` field uses.
                fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
                    Ok(self.0.to_pyarray(py))
                }
            }
        };
    }

    into_pyarray1!(FloatVec, f64);
    into_pyarray1!(IntVec, i32);
    into_pyarray1!(BoolVec, bool);

    impl<'py> IntoPyObject<'py> for FloatMatrix {
        type Target = PyArray2<f64>;
        type Output = Bound<'py, PyArray2<f64>>;
        type Error = Infallible;

        /// Moves the buffer into a NumPy array without copying.
        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
            Ok(self.0.into_pyarray(py))
        }
    }

    impl<'py> IntoPyObject<'py> for &FloatMatrix {
        type Target = PyArray2<f64>;
        type Output = Bound<'py, PyArray2<f64>>;
        type Error = Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
            Ok(self.0.to_pyarray(py))
        }
    }
}

#[cfg(feature = "python")]
use pyo3::types::PyAnyMethods;

/// Extracts `float64` values from any array-like; prefer a [`FloatVec`]
/// parameter, which does the same conversion in the signature.
#[cfg(feature = "python")]
pub(crate) fn extract_vec_f64(obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Vec<f64>> {
    obj.extract::<FloatVec>().map(FloatVec::into_inner)
}

/// Extracts `int32` values from any array-like; prefer an [`IntVec`] parameter.
#[cfg(feature = "python")]
pub(crate) fn extract_vec_i32(obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Vec<i32>> {
    obj.extract::<IntVec>().map(IntVec::into_inner)
}

/// Extracts a matrix as rows from any 2-D array-like; prefer a
/// [`FloatMatrix`] parameter, which keeps the data in an `Array2`.
#[cfg(feature = "python")]
pub(crate) fn extract_matrix_f64(
    obj: &pyo3::Bound<'_, pyo3::PyAny>,
) -> pyo3::PyResult<Vec<Vec<f64>>> {
    obj.extract::<FloatMatrix>()
        .map(|matrix: FloatMatrix| matrix.to_rows())
}

#[cfg(feature = "python")]
pub(crate) fn extract_optional_vec_f64(
    obj: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> pyo3::PyResult<Option<Vec<f64>>> {
    obj.map(extract_vec_f64).transpose()
}

#[cfg(not(feature = "python"))]
fn unavailable<T>(what: &str) -> pyo3::PyResult<T> {
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{what} Python inputs require the `python` feature"
    )))
}

#[cfg(not(feature = "python"))]
pub(crate) fn extract_vec_f64(_obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Vec<f64>> {
    unavailable("array-like")
}

#[cfg(not(feature = "python"))]
pub(crate) fn extract_vec_i32(_obj: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Vec<i32>> {
    unavailable("array-like")
}

#[cfg(not(feature = "python"))]
pub(crate) fn extract_matrix_f64(
    _obj: &pyo3::Bound<'_, pyo3::PyAny>,
) -> pyo3::PyResult<Vec<Vec<f64>>> {
    unavailable("matrix-like")
}

#[cfg(not(feature = "python"))]
pub(crate) fn extract_optional_vec_f64(
    _obj: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> pyo3::PyResult<Option<Vec<f64>>> {
    unavailable("array-like")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn matrix_constructors_validate_and_normalise_layout() {
        let matrix = FloatMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!((matrix.nrow(), matrix.ncol()), (2, 2));
        assert_eq!(matrix.as_flat(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.to_rows(), vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let err = FloatMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert_eq!(err.to_string(), "row 1 length mismatch: expected 2, got 1");

        let flat = FloatMatrix::from_flat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3).unwrap();
        assert_eq!(*flat, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(FloatMatrix::from_flat(vec![1.0, 2.0, 3.0], 2).is_err());
        assert_eq!(FloatMatrix::from_flat(vec![], 0).unwrap().nrow(), 0);

        let fortran = array![[1.0, 2.0], [3.0, 4.0]].reversed_axes();
        assert!(!fortran.is_standard_layout());
        let normalised = FloatMatrix::new(fortran);
        assert_eq!(normalised.as_flat(), &[1.0, 3.0, 2.0, 4.0]);
        assert_eq!(normalised[[1, 0]], 2.0);
    }

    #[test]
    fn vector_newtypes_deref_and_convert() {
        let values = FloatVec::from(vec![1.0, 2.0]);
        assert_eq!(values.len(), 2);
        assert_eq!(values.iter().sum::<f64>(), 3.0);
        let inner: Vec<f64> = values.into();
        assert_eq!(inner, vec![1.0, 2.0]);
        assert_eq!(IntVec(vec![1, 2]).into_inner(), vec![1, 2]);
        assert_eq!(&*BoolVec(vec![true]), &[true]);
    }
}

#[cfg(all(test, feature = "python"))]
mod python_tests {
    //! Need an interpreter with NumPy: `PYO3_PYTHON` must point at one.

    use super::*;
    use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use std::ffi::CString;

    fn eval<'py>(py: Python<'py>, expr: &str) -> Bound<'py, PyAny> {
        let globals = PyDict::new(py);
        globals
            .set_item("np", py.import("numpy").expect("numpy is installed"))
            .unwrap();
        let code = CString::new(expr).unwrap();
        py.eval(&code, Some(&globals), None)
            .unwrap_or_else(|err| panic!("{expr}: {err}"))
    }

    fn floats(py: Python<'_>, expr: &str) -> PyResult<Vec<f64>> {
        eval(py, expr)
            .extract::<FloatVec>()
            .map(FloatVec::into_inner)
    }

    fn ints(py: Python<'_>, expr: &str) -> PyResult<Vec<i32>> {
        eval(py, expr).extract::<IntVec>().map(IntVec::into_inner)
    }

    fn bools(py: Python<'_>, expr: &str) -> PyResult<Vec<bool>> {
        eval(py, expr).extract::<BoolVec>().map(BoolVec::into_inner)
    }

    fn matrix(py: Python<'_>, expr: &str) -> PyResult<FloatMatrix> {
        eval(py, expr).extract::<FloatMatrix>()
    }

    #[test]
    fn float_vec_accepts_arrays_of_any_dtype_layout_or_sequence() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(floats(py, "np.array([1.0, 2.5])").unwrap(), [1.0, 2.5]);
            assert_eq!(floats(py, "np.arange(6.0)[::2]").unwrap(), [0.0, 2.0, 4.0]);
            assert_eq!(
                floats(py, "np.array([1, 2], dtype='int64')").unwrap(),
                [1.0, 2.0]
            );
            assert_eq!(
                floats(py, "np.array([1, 2], dtype='float32')").unwrap(),
                [1.0, 2.0]
            );
            assert_eq!(floats(py, "np.array([True, False])").unwrap(), [1.0, 0.0]);
            assert_eq!(floats(py, "[1, 2.5, True]").unwrap(), [1.0, 2.5, 1.0]);
            assert_eq!(floats(py, "(3.0,)").unwrap(), [3.0]);
            assert_eq!(floats(py, "range(3)").unwrap(), [0.0, 1.0, 2.0]);
            assert_eq!(floats(py, "np.array([])").unwrap(), Vec::<f64>::new());

            let err = floats(py, "np.zeros((2, 2))").unwrap_err();
            assert!(err.to_string().contains("1-dimensional"), "{err}");
            let err = floats(py, "'abc'").unwrap_err();
            assert!(err.to_string().contains("cannot convert 'str'"), "{err}");
            let err = floats(py, "[1.0, 'x']").unwrap_err();
            assert!(err.to_string().contains("cannot convert 'list'"), "{err}");
            let err = floats(py, "None").unwrap_err();
            assert!(err.to_string().contains("NoneType"), "{err}");
        });
    }

    #[test]
    fn int_vec_narrows_checked_and_accepts_integral_floats() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(ints(py, "np.array([1, 0], dtype='int32')").unwrap(), [1, 0]);
            assert_eq!(
                ints(py, "np.array([1, 0], dtype='int64')[::-1]").unwrap(),
                [0, 1]
            );
            assert_eq!(ints(py, "np.array([True, False])").unwrap(), [1, 0]);
            assert_eq!(ints(py, "np.array([1.0, 0.0])").unwrap(), [1, 0]);
            assert_eq!(ints(py, "np.array([3, 4], dtype='uint8')").unwrap(), [3, 4]);
            assert_eq!(ints(py, "[1, 0, True]").unwrap(), [1, 0, 1]);
            assert_eq!(ints(py, "[1.0, 2.0]").unwrap(), [1, 2]);

            let err = ints(py, "np.array([2**40])").unwrap_err();
            assert!(err.to_string().contains("out of range"), "{err}");
            let err = ints(py, "np.array([0.5])").unwrap_err();
            assert!(err.to_string().contains("not an int32"), "{err}");
            let err = ints(py, "[0.5]").unwrap_err();
            assert!(err.to_string().contains("not an int32"), "{err}");
        });
    }

    #[test]
    fn bool_vec_accepts_booleans_and_zero_one() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(bools(py, "np.array([True, False])").unwrap(), [true, false]);
            assert_eq!(bools(py, "[True, False]").unwrap(), [true, false]);
            assert_eq!(bools(py, "np.array([1, 0])").unwrap(), [true, false]);
            assert_eq!(bools(py, "[1.0, 0]").unwrap(), [true, false]);
            let err = bools(py, "[2]").unwrap_err();
            assert!(err.to_string().contains("not 0 or 1"), "{err}");
        });
    }

    #[test]
    fn float_matrix_accepts_any_layout_rows_or_a_vector() {
        Python::initialize();
        Python::attach(|py| {
            let c_order = matrix(py, "np.array([[1.0, 2.0], [3.0, 4.0]])").unwrap();
            assert_eq!(c_order.as_flat(), &[1.0, 2.0, 3.0, 4.0]);

            let f_order = matrix(py, "np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])").unwrap();
            assert_eq!(f_order.as_flat(), &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!((f_order.nrow(), f_order.ncol()), (2, 2));

            let strided = matrix(py, "np.arange(12.0).reshape(3, 4)[::2, 1::2]").unwrap();
            assert_eq!(strided.to_rows(), vec![vec![1.0, 3.0], vec![9.0, 11.0]]);

            let ints = matrix(py, "np.array([[1, 2], [3, 4]])").unwrap();
            assert_eq!(ints.as_flat(), &[1.0, 2.0, 3.0, 4.0]);

            let rows = matrix(py, "[[1, 2, 3], [4, 5, 6]]").unwrap();
            assert_eq!((rows.nrow(), rows.ncol()), (2, 3));

            let column = matrix(py, "[1.0, 2.0, 3.0]").unwrap();
            assert_eq!((column.nrow(), column.ncol()), (3, 1));
            let column = matrix(py, "np.array([1.0, 2.0])").unwrap();
            assert_eq!((column.nrow(), column.ncol()), (2, 1));

            let err = matrix(py, "[[1.0, 2.0], [3.0]]").unwrap_err();
            assert!(err.to_string().contains("row 1 length mismatch"), "{err}");
            let err = matrix(py, "np.zeros((2, 2, 2))").unwrap_err();
            assert!(err.to_string().contains("3 dimension"), "{err}");
        });
    }

    #[test]
    fn outputs_become_numpy_arrays() {
        Python::initialize();
        Python::attach(|py| {
            let out = FloatVec(vec![1.0, 2.0]).into_pyobject(py).unwrap();
            assert_eq!(out.readonly().as_slice().unwrap(), &[1.0, 2.0]);
            let by_ref = (&IntVec(vec![3, 4])).into_pyobject(py).unwrap();
            assert_eq!(by_ref.readonly().as_slice().unwrap(), &[3, 4]);
            let flags = (&BoolVec(vec![true])).into_pyobject(py).unwrap();
            assert_eq!(flags.readonly().as_slice().unwrap(), &[true]);

            let matrix = FloatMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
            let out: Bound<'_, PyArray2<f64>> = matrix.into_pyobject(py).unwrap();
            assert_eq!(out.shape(), &[2, 2]);
            assert_eq!(out.readonly().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

            let err = out.into_any().extract::<FloatVec>().unwrap_err();
            assert!(err.to_string().contains("1-dimensional"), "{err}");
            let empty: Bound<'_, PyArray1<f64>> = FloatVec(vec![]).into_pyobject(py).unwrap();
            assert!(empty.is_empty());
        });
    }

    #[test]
    fn legacy_extractors_delegate_to_the_typed_inputs() {
        Python::initialize();
        Python::attach(|py| {
            let arr = eval(py, "np.array([1, 2], dtype='int64')");
            assert_eq!(extract_vec_f64(&arr).unwrap(), [1.0, 2.0]);
            assert_eq!(extract_vec_i32(&arr).unwrap(), [1, 2]);
            assert_eq!(extract_optional_vec_f64(None).unwrap(), None);
            let m = eval(py, "np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])");
            assert_eq!(
                extract_matrix_f64(&m).unwrap(),
                vec![vec![1.0, 2.0], vec![3.0, 4.0]]
            );
        });
    }
}
