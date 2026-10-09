//! Boundary input and output types for `#[pyfunction]` signatures.
//!
//! [`FloatVec`], [`IntVec`], [`BoolVec`], [`FloatMatrix`] and [`FloatArray3`] accept, without a
//! Python-side `.tolist()`: NumPy arrays of any dtype and memory layout,
//! pandas/polars columns (anything with `__array__`) and plain sequences. Each
//! converts exactly once into the owned container core code consumes
//! (`Vec<f64>`, `Vec<i32>`, `Vec<bool>`, `Array2<f64>`, `Array3<f64>`). Returning one from a
//! `#[pyfunction]` hands the caller a NumPy array without copying
//! (`IntoPyArray`), and a `#[pyo3(get)]` field of these types reads back as a
//! NumPy array.
//!
//! Without the `python` feature the same newtypes exist as plain wrappers, so
//! core signatures can name them in both builds.

use std::ops::Deref;

use ndarray::{Array2, Array3};

use crate::error::{SurvivalError, SurvivalResult};
use crate::internal::validation::{ValidationError, validate_matrix_shape};

/// One-dimensional `float64` input.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FloatVec(pub Vec<f64>);

/// Matrix input for kernels that consume nested rows. Unlike [`FloatMatrix`],
/// list inputs keep their row buffers instead of flattening and rebuilding
/// them. The receiving kernel validates rectangular shape and dimensions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FloatRows(pub Vec<Vec<f64>>);

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
vector_newtype!(FloatRows, Vec<f64>);
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

    /// The array, checked to be `nrow x ncol`.  An empty input (`[]`) stands
    /// for zero rows of any width, or for any number of rows of zero width.
    pub fn into_shape(self, nrow: usize, ncol: usize, name: &str) -> SurvivalResult<Array2<f64>> {
        if self.0.dim() == (nrow, ncol) {
            Ok(self.0)
        } else if (nrow == 0 && self.0.is_empty()) || (ncol == 0 && self.0.nrows() == 0) {
            Ok(Array2::zeros((nrow, ncol)))
        } else {
            Err(SurvivalError::invalid_input(format!(
                "{name} must be {nrow} x {ncol}, got {} x {}",
                self.nrow(),
                self.ncol()
            )))
        }
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

/// Three-dimensional `float64` input, owned in row-major (C) layout.
/// NumPy storage is copied before a receiving kernel releases the GIL.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatArray3(Array3<f64>);

impl FloatArray3 {
    pub fn new(values: Array3<f64>) -> Self {
        if values.is_standard_layout() {
            Self(values)
        } else {
            Self(
                Array3::from_shape_vec(values.dim(), values.iter().copied().collect())
                    .expect("logical array traversal preserves the three-dimensional shape"),
            )
        }
    }

    pub fn into_inner(self) -> Array3<f64> {
        self.0
    }

    pub fn as_flat(&self) -> &[f64] {
        self.0
            .as_slice()
            .expect("FloatArray3 is kept in row-major layout")
    }
}

impl Deref for FloatArray3 {
    type Target = Array3<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Array3<f64>> for FloatArray3 {
    fn from(values: Array3<f64>) -> Self {
        Self::new(values)
    }
}

impl From<FloatArray3> for Array3<f64> {
    fn from(values: FloatArray3) -> Self {
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

    use ndarray::{Array2, Array3};
    use numpy::{
        Element, IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods,
        PyUntypedArray, PyUntypedArrayMethods, ToPyArray, get_array_module,
    };
    use pyo3::Borrowed;
    use pyo3::exceptions::{PyTypeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};
    use std::convert::Infallible;

    use super::{BoolVec, FloatArray3, FloatMatrix, FloatRows, FloatVec, IntVec};

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

    /// NumPy permits unaligned buffers and strides. Normalise those before
    /// creating Rust references; an owned copy is then made by the caller.
    fn aligned_array<'py, T: Element, D: ndarray::Dimension>(
        array: &Bound<'py, PyArray<T, D>>,
    ) -> PyResult<Bound<'py, PyArray<T, D>>> {
        if array.is_aligned() || array.is_empty() {
            return Ok(array.clone());
        }
        let kwargs = PyDict::new(array.py());
        kwargs.set_item("requirements", ["A", "E"])?;
        let aligned = array
            .py()
            .import("numpy")?
            .getattr("require")?
            .call((array,), Some(&kwargs))?
            .cast_into::<PyArray<T, D>>()?;
        if !aligned.is_aligned() {
            return Err(PyValueError::new_err("NumPy array storage must be aligned"));
        }
        Ok(aligned)
    }

    /// Copies a one-dimensional array out in logical order, whatever its strides.
    fn read_1d<T: Element + Copy>(array: &Bound<'_, PyArray1<T>>) -> PyResult<Vec<T>> {
        if array.is_empty() {
            return Ok(Vec::new());
        }
        let array = aligned_array(array)?;
        let view = array.readonly();
        Ok(match view.as_slice() {
            Ok(slice) => slice.to_vec(),
            Err(_) => view.as_array().iter().copied().collect(),
        })
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
            return read_1d(array);
        }
        if is_plain_sequence(obj) {
            return obj
                .extract::<Vec<f64>>()
                .map_err(|err| type_error(obj, expected, Some(err)));
        }
        let converted = asarray::<f64>(obj).map_err(|err| type_error(obj, expected, Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            1 => read_1d(converted.cast::<PyArray1<f64>>()?),
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
            return read_1d(array);
        }
        if let Ok(array) = obj.cast::<PyArray1<i64>>() {
            return narrow_i64(obj, read_1d(array)?);
        }
        if obj.cast::<PyArray1<bool>>().is_ok() {
            // NumPy bool storage can contain any nonzero byte (for example
            // frombuffer([2, 255])). Rust bool only permits bytes 0 and 1,
            // so convert numerically without borrowing that storage as bool.
            let converted = asarray::<i32>(obj)?;
            return read_1d(converted.cast::<PyArray1<i32>>()?);
        }
        if is_plain_sequence(obj)
            && let Ok(values) = obj.extract::<Vec<i64>>()
        {
            return narrow_i64(obj, values);
        }
        integral_f64(obj, float_values(obj, "an int32 array")?)
    }

    fn bool_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        if obj.cast::<PyArray1<bool>>().is_ok() {
            let converted = asarray::<u8>(obj)?;
            return Ok(read_1d(converted.cast::<PyArray1<u8>>()?)?
                .into_iter()
                .map(|value| value != 0)
                .collect());
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
    fn read_2d(array: &Bound<'_, PyArray2<f64>>) -> PyResult<Array2<f64>> {
        let array = aligned_array(array)?;
        let shape = array.shape();
        let (n, p) = (shape[0], shape[1]);
        if array.is_empty() {
            return Ok(Array2::zeros((n, p)));
        }
        let view = array.readonly();
        let flat = match view.as_slice() {
            Ok(slice) if array.is_c_contiguous() => slice.to_vec(),
            _ => view.as_array().iter().copied().collect(),
        };
        Ok(Array2::from_shape_vec((n, p), flat).expect("(n, p) view yields n * p entries"))
    }

    fn matrix_values(obj: &Bound<'_, PyAny>) -> PyResult<FloatMatrix> {
        if let Ok(array) = obj.cast::<PyArray2<f64>>() {
            return read_2d(array).map(FloatMatrix);
        }
        if is_plain_sequence(obj) {
            if let Some(matrix) = plain_matrix(obj) {
                return Ok(matrix);
            }
            if let Ok(rows) = obj.extract::<Vec<Vec<f64>>>() {
                return FloatMatrix::from_rows(rows).map_err(PyErr::from);
            }
            return float_values(obj, "a float matrix").map(column_matrix);
        }
        let converted =
            asarray::<f64>(obj).map_err(|err| type_error(obj, "a float matrix", Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            2 => read_2d(converted.cast::<PyArray2<f64>>()?).map(FloatMatrix),
            1 => read_1d(converted.cast::<PyArray1<f64>>()?).map(column_matrix),
            ndim => Err(wrong_ndim(obj, "a float matrix", ndim)),
        }
    }

    /// Copy a three-dimensional NumPy array in logical order, normalising
    /// unaligned storage before constructing Rust references to its values.
    fn read_3d(array: &Bound<'_, PyArray3<f64>>) -> PyResult<FloatArray3> {
        let shape = array.shape();
        let shape = (shape[0], shape[1], shape[2]);
        if array.is_empty() {
            return Ok(FloatArray3(Array3::zeros(shape)));
        }
        let array = aligned_array(array)?;
        let view = array.readonly();
        let flat = match view.as_slice() {
            Ok(slice) if array.is_c_contiguous() => slice.to_vec(),
            _ => view.as_array().iter().copied().collect(),
        };
        Array3::from_shape_vec(shape, flat)
            .map(FloatArray3)
            .map_err(|err| PyValueError::new_err(format!("3-dimensional float array: {err}")))
    }

    fn array3_values(obj: &Bound<'_, PyAny>) -> PyResult<FloatArray3> {
        if let Ok(array) = obj.cast::<PyArray3<f64>>() {
            return read_3d(array);
        }
        if is_plain_sequence(obj) {
            if let Some(array) = plain_array3(obj) {
                return Ok(array);
            }
            return sequence_array3(obj);
        }
        let converted = asarray::<f64>(obj)
            .map_err(|err| type_error(obj, "a 3-dimensional float array", Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            3 => read_3d(converted.cast::<PyArray3<f64>>()?),
            ndim => Err(wrong_ndim(obj, "a 3-dimensional float array", ndim)),
        }
    }

    /// Exact lists/tuples of Python floats have no conversion callbacks, so
    /// their innermost rows can be read without allocating Python iterators.
    fn plain_array3(obj: &Bound<'_, PyAny>) -> Option<FloatArray3> {
        let exact_sequence = |value: &Bound<'_, PyAny>| {
            value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>()
        };
        if !exact_sequence(obj) {
            return None;
        }
        let n_time = obj.len().ok()?;
        let (n_data, n_state) = if n_time == 0 {
            (0, 0)
        } else {
            let first = obj.get_item(0).ok()?;
            if !exact_sequence(&first) {
                return None;
            }
            let n_data = first.len().ok()?;
            let n_state = if n_data == 0 {
                0
            } else {
                let first = first.get_item(0).ok()?;
                if !exact_sequence(&first) {
                    return None;
                }
                first.len().ok()?
            };
            (n_data, n_state)
        };
        let size = n_time.checked_mul(n_data)?.checked_mul(n_state)?;
        if size > isize::MAX as usize / std::mem::size_of::<f64>() {
            return None;
        }
        let mut values = Vec::with_capacity(size);
        for row in obj.try_iter().ok()? {
            let row = row.ok()?;
            if !exact_sequence(&row) || row.len().ok()? != n_data {
                return None;
            }
            for states in row.try_iter().ok()? {
                let states = states.ok()?;
                if !exact_sequence(&states) || states.len().ok()? != n_state {
                    return None;
                }
                if let Ok(states) = states.cast::<PyList>() {
                    for value in states.iter() {
                        values.push(value.cast::<PyFloat>().ok()?.value());
                    }
                } else {
                    for value in states.cast::<PyTuple>().ok()?.iter() {
                        values.push(value.cast::<PyFloat>().ok()?.value());
                    }
                }
            }
        }
        Array3::from_shape_vec((n_time, n_data, n_state), values)
            .ok()
            .map(FloatArray3)
    }

    /// Flatten nested rows directly into their final buffer. This keeps lists,
    /// tuples, array-valued rows and sequence subclasses compatible without
    /// allocating an intermediate vector for every data/state row.
    fn sequence_array3(obj: &Bound<'_, PyAny>) -> PyResult<FloatArray3> {
        let length = |value: &Bound<'_, PyAny>, context: &str| {
            value
                .len()
                .map_err(|err| type_error(value, context, Some(err)))
        };
        let n_time = obj.len()?;
        let n_data = if n_time == 0 {
            0
        } else {
            length(&obj.get_item(0)?, "a 3-dimensional float array time row")?
        };
        let n_state = if n_data == 0 {
            0
        } else {
            length(
                &obj.get_item(0)?.get_item(0)?,
                "a 3-dimensional float array state row",
            )?
        };
        let size = n_time
            .checked_mul(n_data)
            .and_then(|size| size.checked_mul(n_state))
            .filter(|&size| size <= isize::MAX as usize / std::mem::size_of::<f64>())
            .ok_or_else(|| PyValueError::new_err("3-dimensional float array shape is too large"))?;
        let mut values = Vec::with_capacity(size);
        let mut times_read = 0;
        for (t, row) in obj.try_iter()?.enumerate() {
            let row = row?;
            times_read += 1;
            let count = length(&row, "a 3-dimensional float array time row")?;
            if count != n_data {
                return Err(PyValueError::new_err(format!(
                    "array time {t} length mismatch: expected {n_data}, got {count}"
                )));
            }
            let mut data_read = 0;
            for (j, states) in row.try_iter()?.enumerate() {
                let states = states?;
                data_read += 1;
                let count = length(&states, "a 3-dimensional float array state row")?;
                if count != n_state {
                    return Err(PyValueError::new_err(format!(
                        "array time {t} data {j} length mismatch: expected {n_state}, got {count}"
                    )));
                }
                let first_value = values.len();
                for value in states.try_iter()? {
                    let value = value?;
                    let number = if let Ok(number) = value.cast::<PyFloat>() {
                        number.value()
                    } else {
                        value
                            .extract::<f64>()
                            .map_err(|err| type_error(&value, "a float array value", Some(err)))?
                    };
                    values.push(number);
                }
                let count = values.len() - first_value;
                if count != n_state {
                    return Err(PyValueError::new_err(format!(
                        "array time {t} data {j} length mismatch: expected {n_state}, got {count}"
                    )));
                }
            }
            if data_read != n_data {
                return Err(PyValueError::new_err(format!(
                    "array time {t} length mismatch: expected {n_data}, got {data_read}"
                )));
            }
        }
        if times_read != n_time {
            return Err(PyValueError::new_err(format!(
                "array time margin length mismatch: expected {n_time}, got {times_read}"
            )));
        }
        Array3::from_shape_vec((n_time, n_data, n_state), values)
            .map(FloatArray3)
            .map_err(|err| PyValueError::new_err(format!("3-dimensional float array: {err}")))
    }

    /// Lists/tuples of float rows go directly into one row-major allocation.
    /// Keep the general extractor for sequence subclasses, array-valued rows,
    /// other scalar types, vectors and invalid inputs. Reading only PyFloat
    /// objects cannot invoke custom conversion methods before falling back.
    fn plain_matrix(obj: &Bound<'_, PyAny>) -> Option<FloatMatrix> {
        let exact_sequence = |value: &Bound<'_, PyAny>| {
            value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>()
        };
        if !exact_sequence(obj) {
            return None;
        }
        let nrow = obj.len().ok()?;
        let ncol = if nrow == 0 {
            0
        } else {
            let first = obj.get_item(0).ok()?;
            if !exact_sequence(&first) {
                return None;
            }
            first.len().ok()?
        };
        let mut values = Vec::with_capacity(nrow.checked_mul(ncol)?);
        for row in obj.try_iter().ok()? {
            let row = row.ok()?;
            if !exact_sequence(&row) || row.len().ok()? != ncol {
                return None;
            }
            if let Ok(row) = row.cast::<PyList>() {
                for value in row.iter() {
                    values.push(value.cast::<PyFloat>().ok()?.value());
                }
            } else {
                for value in row.cast::<PyTuple>().ok()?.iter() {
                    values.push(value.cast::<PyFloat>().ok()?.value());
                }
            }
        }
        Array2::from_shape_vec((nrow, ncol), values)
            .ok()
            .map(FloatMatrix)
    }

    fn read_rows(array: &Bound<'_, PyArray2<f64>>) -> PyResult<FloatRows> {
        let array = aligned_array(array)?;
        if array.is_empty() {
            return Ok(FloatRows(vec![Vec::new(); array.shape()[0]]));
        }
        let view = array.readonly();
        Ok(FloatRows(
            view.as_array()
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        ))
    }

    fn row_values(obj: &Bound<'_, PyAny>) -> PyResult<FloatRows> {
        if let Ok(array) = obj.cast::<PyArray2<f64>>() {
            return read_rows(array);
        }
        let column = |values: Vec<f64>| FloatRows(values.into_iter().map(|x| vec![x]).collect());
        if is_plain_sequence(obj) {
            if let Ok(rows) = obj.extract::<Vec<Vec<f64>>>() {
                return Ok(FloatRows(rows));
            }
            return float_values(obj, "a float matrix").map(column);
        }
        let converted =
            asarray::<f64>(obj).map_err(|err| type_error(obj, "a float matrix", Some(err)))?;
        match converted.cast::<PyUntypedArray>()?.ndim() {
            2 => read_rows(converted.cast::<PyArray2<f64>>()?),
            1 => Ok(column(read_1d(converted.cast::<PyArray1<f64>>()?)?)),
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

    impl<'py> FromPyObject<'_, 'py> for FloatArray3 {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            array3_values(&obj)
        }
    }

    impl<'py> FromPyObject<'_, 'py> for FloatRows {
        type Error = PyErr;

        fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
            row_values(&obj)
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

    impl<'py> IntoPyObject<'py> for FloatArray3 {
        type Target = PyArray3<f64>;
        type Output = Bound<'py, PyArray3<f64>>;
        type Error = Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
            Ok(self.0.into_pyarray(py))
        }
    }

    impl<'py> IntoPyObject<'py> for &FloatArray3 {
        type Target = PyArray3<f64>;
        type Output = Bound<'py, PyArray3<f64>>;
        type Error = Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Infallible> {
            Ok(self.0.to_pyarray(py))
        }
    }
}

#[cfg(feature = "python")]
use pyo3::types::PyAnyMethods;

/// Owned prediction matrices, without constructing Python scalar/row lists.
#[cfg(feature = "python")]
pub(crate) fn prediction_matrix_arrays<'py>(
    py: pyo3::Python<'py>,
    fit: &[Vec<f64>],
    se: Option<&[Vec<f64>]>,
    columns: usize,
) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    use crate::internal::matrix::matrix_from_rows_with_columns;
    use crate::internal::validation::validate_length;
    use numpy::IntoPyArray;
    use pyo3::types::PyDictMethods;

    let (fit, se) = py.detach(|| -> crate::error::SurvivalResult<_> {
        if let Some(se) = se {
            validate_length(fit.len(), se.len(), "prediction errors")?;
        }
        Ok((
            matrix_from_rows_with_columns(fit, columns, "prediction")?,
            se.map(|se| matrix_from_rows_with_columns(se, columns, "prediction errors"))
                .transpose()?,
        ))
    })?;
    let result = pyo3::types::PyDict::new(py);
    result.set_item("fit", fit.into_pyarray(py))?;
    result.set_item("se_fit", se.map(|se| se.into_pyarray(py)))?;
    Ok(result)
}

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

#[cfg(feature = "python")]
pub(crate) fn extract_optional_vec_f64(
    obj: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
) -> pyo3::PyResult<Option<Vec<f64>>> {
    obj.map(extract_vec_f64).transpose()
}

/// A read-only NumPy view of `array` whose base object is `owner`: how a
/// `#[pyclass]` getter hands Python a large array the class holds without
/// copying it.  NumPy refuses to make the view writeable again, since it
/// does not own the data.
///
/// # Safety
///
/// `array` must be owned by `owner` (directly or through an `Arc` it
/// holds), and must not be mutated or reallocated while `owner` is alive;
/// a `frozen` class that never hands out `&mut` access to it guarantees
/// both.
#[cfg(feature = "python")]
pub(crate) unsafe fn readonly_view<'py, D: ndarray::Dimension>(
    array: &ndarray::Array<f64, D>,
    owner: &pyo3::Bound<'py, pyo3::PyAny>,
) -> pyo3::Bound<'py, numpy::PyArray<f64, D>> {
    use numpy::{PyArray, PyArrayMethods};
    // SAFETY: the caller guarantees that `owner`, which becomes the view's
    // base object, keeps `array` alive and unchanged for the view's lifetime.
    let view = unsafe { PyArray::borrow_from_array(array, owner.clone()) };
    view.readwrite().make_nonwriteable();
    view
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

        assert_eq!(matrix.clone().into_shape(2, 2, "x").unwrap().dim(), (2, 2));
        let err = matrix.into_shape(2, 3, "x").unwrap_err();
        assert_eq!(err.to_string(), "x must be 2 x 3, got 2 x 2");
        let empty = FloatMatrix::from_rows(Vec::new()).unwrap();
        assert_eq!(empty.clone().into_shape(0, 3, "x").unwrap().dim(), (0, 3));
        assert_eq!(empty.into_shape(4, 0, "x").unwrap().dim(), (4, 0));

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
    fn array3_constructor_preserves_shape_and_normalises_layout() {
        let original = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| (100 * i + 10 * j + k) as f64);
        let transposed = original.permuted_axes([2, 0, 1]);
        assert!(!transposed.is_standard_layout());
        let expected: Vec<f64> = transposed.iter().copied().collect();
        let owned = FloatArray3::new(transposed);
        assert_eq!(owned.dim(), (4, 2, 3));
        assert_eq!(owned.as_flat(), expected);
        assert_eq!(owned[[3, 1, 2]], 123.0);
        assert!(owned.into_inner().is_standard_layout());
        assert_eq!(FloatArray3::new(Array3::zeros((0, 3, 4))).dim(), (0, 3, 4));
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
    use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PySlice};
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
    fn unaligned_vectors_matrices_and_rows_are_copied_safely() {
        Python::initialize();
        Python::attach(|py| {
            let input = eval(
                py,
                "np.ndarray((3,), dtype='float64', buffer=bytearray(25), offset=1)",
            );
            for (index, value) in [1.25, -2.5, 3.75].into_iter().enumerate() {
                input.set_item(index, value).unwrap();
            }
            assert!(!input.cast::<PyArray1<f64>>().unwrap().is_aligned());
            let owned = input.extract::<FloatVec>().unwrap();
            let strided = input.get_item(PySlice::new(py, 0, 3, 2)).unwrap();
            assert_eq!(*strided.extract::<FloatVec>().unwrap(), [1.25, 3.75]);
            input.set_item(0, -100.0).unwrap();
            py.detach(|| assert_eq!(*owned, [1.25, -2.5, 3.75]));

            for (dtype, bytes) in [("int32", 13), ("int64", 25)] {
                let input = eval(
                    py,
                    &format!(
                        "np.ndarray((3,), dtype='{dtype}', buffer=bytearray({bytes}), offset=1)"
                    ),
                );
                for (index, value) in [1, -2, 3].into_iter().enumerate() {
                    input.set_item(index, value).unwrap();
                }
                assert_eq!(*input.extract::<IntVec>().unwrap(), [1, -2, 3]);
                for (index, value) in [1, 0, 1].into_iter().enumerate() {
                    input.set_item(index, value).unwrap();
                }
                assert_eq!(*input.extract::<BoolVec>().unwrap(), [true, false, true]);
            }
            let matrix = eval(
                py,
                "np.ndarray((2, 3), dtype='float64', buffer=bytearray(49), offset=1)",
            );
            for i in 0..2 {
                for j in 0..3 {
                    matrix.set_item((i, j), (3 * i + j + 1) as f64).unwrap();
                }
            }
            let owned = matrix.extract::<FloatMatrix>().unwrap();
            let rows = matrix.extract::<FloatRows>().unwrap();
            assert_eq!(owned.as_flat(), &[1., 2., 3., 4., 5., 6.]);
            assert_eq!(*rows, vec![vec![1., 2., 3.], vec![4., 5., 6.]]);
            let strided = matrix
                .get_item((PySlice::new(py, 0, 2, 1), PySlice::new(py, 0, 3, 2)))
                .unwrap();
            assert_eq!(
                strided.extract::<FloatMatrix>().unwrap().as_flat(),
                &[1., 3., 4., 6.]
            );
            matrix.set_item((0, 0), -100.0).unwrap();
            py.detach(|| {
                assert_eq!(owned[[0, 0]], 1.0);
                assert_eq!(rows[0][0], 1.0);
            });

            for expr in [
                "np.ndarray((0,), dtype='float64', buffer=bytearray(1), offset=1)",
                "np.ndarray((0,), dtype='float64', buffer=bytearray(1), offset=1)[::-1]",
            ] {
                assert!(eval(py, expr).extract::<FloatVec>().unwrap().is_empty());
            }
            for expr in [
                "np.ndarray((0,), dtype='int32', buffer=bytearray(1), offset=1)[::-1]",
                "np.ndarray((0,), dtype='int64', buffer=bytearray(1), offset=1)[::-1]",
            ] {
                assert!(eval(py, expr).extract::<IntVec>().unwrap().is_empty());
                assert!(eval(py, expr).extract::<BoolVec>().unwrap().is_empty());
            }
            for (expr, shape) in [
                (
                    "np.ndarray((0, 3), dtype='float64', buffer=bytearray(1), offset=1)",
                    (0, 3),
                ),
                (
                    "np.ndarray((2, 0), dtype='float64', buffer=bytearray(1), offset=1)[:, ::-1]",
                    (2, 0),
                ),
            ] {
                let input = eval(py, expr);
                assert_eq!(input.extract::<FloatMatrix>().unwrap().dim(), shape);
                assert_eq!(input.extract::<FloatRows>().unwrap().len(), shape.0);
            }
        });
    }

    #[test]
    fn numpy_boolean_storage_is_normalised_without_rust_bool_views() {
        Python::initialize();
        Python::attach(|py| {
            for expr in [
                "np.frombuffer(bytearray([2, 0, 255]), dtype=bool)",
                "np.frombuffer(bytearray([2, 0, 255]), dtype=bool)[::-1]",
                "np.frombuffer(bytearray([0, 2, 0, 255]), dtype=bool)[1:]",
            ] {
                let input = eval(py, expr);
                assert_eq!(*input.extract::<BoolVec>().unwrap(), [true, false, true]);
                assert_eq!(*input.extract::<IntVec>().unwrap(), [1, 0, 1]);
            }
            assert!(
                eval(py, "np.frombuffer(bytearray([2]), dtype=bool)[:0]")
                    .extract::<BoolVec>()
                    .unwrap()
                    .is_empty()
            );
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
            for expr in [
                "[[1.0, 2.0], (3.0, 4.0)]",
                "((1, 2), (3, 4))",
                "[(1, 2), [3, 4]]",
                "[np.array([1., 2.]), [3, 4]]",
                "type('Rows', (list,), {})([[1, 2], [3, 4]])",
            ] {
                assert_eq!(matrix(py, expr).unwrap().as_flat(), &[1., 2., 3., 4.]);
            }
            assert_eq!(matrix(py, "[]").unwrap().dim(), (0, 0));
            assert_eq!(matrix(py, "[[], ()]").unwrap().dim(), (2, 0));

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
    fn float_array3_accepts_numpy_layouts_dtypes_and_nested_sequences() {
        Python::initialize();
        Python::attach(|py| {
            for expr in [
                "np.arange(24.).reshape(2, 3, 4)",
                "np.asfortranarray(np.arange(24.).reshape(2, 3, 4))",
                "np.arange(48.).reshape(2, 3, 8)[:, :, ::2]",
                "np.arange(24.).reshape(2, 3, 4)[::-1, :, ::-1]",
                "np.arange(24.).reshape(4, 3, 2).transpose(2, 1, 0)",
                "np.broadcast_to(np.arange(4.), (2, 3, 4))",
                "np.arange(24, dtype='int32').reshape(2, 3, 4)",
                "np.arange(24, dtype='float32').reshape(2, 3, 4)",
                "np.arange(24, dtype='>f8').reshape(2, 3, 4)",
                "np.arange(24).astype(bool).reshape(2, 3, 4)",
                "np.arange(24).astype(object).reshape(2, 3, 4)",
                "np.ndarray((2, 3, 4), dtype='float64', buffer=bytearray(193), offset=1)",
                "[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]",
                "(((1, 2), (3, 4)), ((5, 6), (7, 8)))",
                "[[np.array([1, 2]), (3., 4.)], [range(5, 7), [7, 8]]]",
                "type('Rows', (list,), {})([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])",
            ] {
                let array = eval(py, expr).extract::<FloatArray3>().unwrap();
                let expected: Vec<f64> = eval(
                    py,
                    &format!("np.asarray({expr}, dtype='float64').ravel(order='C').tolist()"),
                )
                .extract()
                .unwrap();
                assert_eq!(array.as_flat(), expected, "{expr}");
                assert!(array.is_standard_layout(), "{expr}");
            }
            for (expr, shape) in [
                ("[]", (0, 0, 0)),
                ("[[], ()]", (2, 0, 0)),
                ("[[[], ()]]", (1, 2, 0)),
                ("np.empty((0, 3, 4))", (0, 3, 4)),
                ("np.empty((2, 0, 4))", (2, 0, 4)),
                ("np.empty((2, 3, 0))", (2, 3, 0)),
            ] {
                assert_eq!(
                    eval(py, expr).extract::<FloatArray3>().unwrap().dim(),
                    shape
                );
            }
            for (expr, message) in [
                ("np.ones((2, 3))", "got 2 dimension"),
                ("np.ones((2, 3, 4, 1))", "got 4 dimension"),
                ("[[[1, 2]], [[3, 4], [5, 6]]]", "time 1 length mismatch"),
                ("[[[1, 2], [3]]]", "data 1 length mismatch"),
                ("[[1, 2]]", "3-dimensional float array state row"),
                ("[[[object()]]]", "float array value"),
                (
                    "type('Huge', (list,), {'__len__': lambda self: 2**60})([[[1.]]])",
                    "shape is too large",
                ),
            ] {
                let err = eval(py, expr).extract::<FloatArray3>().unwrap_err();
                assert!(err.to_string().contains(message), "{expr}: {err}");
            }
        });
    }

    #[test]
    fn float_array3_owns_values_before_detaching_and_converts_outputs() {
        Python::initialize();
        Python::attach(|py| {
            let input = eval(py, "np.arange(24.).reshape(2, 3, 4)");
            let owned = input.extract::<FloatArray3>().unwrap();
            input.set_item((0, 0, 0), -100.0).unwrap();
            let owned = py.detach(|| {
                assert_eq!(owned[[0, 0, 0]], 0.0);
                owned
            });
            let output: Bound<'_, PyArray3<f64>> = owned.into_pyobject(py).unwrap();
            assert_eq!(output.shape(), &[2, 3, 4]);
            assert_eq!(output.readonly().as_slice().unwrap()[0], 0.0);
        });
    }

    #[test]
    fn float_rows_preserves_layout_without_flattening_list_buffers() {
        Python::initialize();
        Python::attach(|py| {
            for expr in [
                "[[1.0, 2.0], [3.0, 4.0]]",
                "((1, 2), (3, 4))",
                "np.array([[1.0, 2.0], [3.0, 4.0]])",
                "np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])",
                "np.array([[1., 99., 2.], [3., 99., 4.]])[:, ::2]",
                "np.array([[1, 2], [3, 4]], dtype='int32')",
                "np.array([[1, 2], [3, 4]], dtype='float32')",
            ] {
                let rows = eval(py, expr).extract::<FloatRows>().unwrap();
                assert_eq!(*rows, vec![vec![1., 2.], vec![3., 4.]], "{expr}");
            }
            for expr in ["[1, 2]", "np.array([1, 2])"] {
                let rows = eval(py, expr).extract::<FloatRows>().unwrap();
                assert_eq!(*rows, vec![vec![1.], vec![2.]]);
            }
            for expr in ["[]", "np.empty((0, 3))"] {
                assert!(eval(py, expr).extract::<FloatRows>().unwrap().is_empty());
            }
            assert_eq!(
                *eval(py, "np.empty((2, 0))").extract::<FloatRows>().unwrap(),
                vec![Vec::<f64>::new(), vec![]]
            );
            let err = eval(py, "np.zeros((2, 2, 2))")
                .extract::<FloatRows>()
                .unwrap_err();
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
        });
    }
}
