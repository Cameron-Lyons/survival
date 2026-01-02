use numpy::{PyArray1, PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

#[allow(clippy::collapsible_if)]
pub fn extract_vec_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(arr.as_slice()?.to_vec());
    }
    if let Ok(list) = obj.extract::<Vec<f64>>() {
        return Ok(list);
    }
    if let Ok(values) = obj.getattr("values") {
        if let Ok(arr) = values.extract::<PyReadonlyArray1<'_, f64>>() {
            return Ok(arr.as_slice()?.to_vec());
        }
    }
    if let Ok(to_numpy) = obj.getattr("to_numpy") {
        if let Ok(arr_obj) = to_numpy.call0() {
            if let Ok(arr) = arr_obj.extract::<PyReadonlyArray1<'_, f64>>() {
                return Ok(arr.as_slice()?.to_vec());
            }
        }
    }
    let type_name = obj
        .get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Cannot convert '{}' to float array. Expected: numpy array, pandas Series, polars Series, or list of floats. \
         Tip: For pandas/polars, ensure the column contains numeric data.",
        type_name
    )))
}

#[allow(clippy::collapsible_if)]
pub fn extract_vec_i32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        return Ok(arr.as_slice()?.to_vec());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(arr.as_slice()?.iter().map(|&x| x as i32).collect());
    }
    if let Ok(list) = obj.extract::<Vec<i32>>() {
        return Ok(list);
    }
    if let Ok(list) = obj.extract::<Vec<i64>>() {
        return Ok(list.into_iter().map(|x| x as i32).collect());
    }
    if let Ok(values) = obj.getattr("values") {
        if let Ok(arr) = values.extract::<PyReadonlyArray1<'_, i32>>() {
            return Ok(arr.as_slice()?.to_vec());
        }
        if let Ok(arr) = values.extract::<PyReadonlyArray1<'_, i64>>() {
            return Ok(arr.as_slice()?.iter().map(|&x| x as i32).collect());
        }
    }
    if let Ok(to_numpy) = obj.getattr("to_numpy") {
        if let Ok(arr_obj) = to_numpy.call0() {
            if let Ok(arr) = arr_obj.extract::<PyReadonlyArray1<'_, i32>>() {
                return Ok(arr.as_slice()?.to_vec());
            }
            if let Ok(arr) = arr_obj.extract::<PyReadonlyArray1<'_, i64>>() {
                return Ok(arr.as_slice()?.iter().map(|&x| x as i32).collect());
            }
        }
    }
    let type_name = obj
        .get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Cannot convert '{}' to integer array. Expected: numpy array (int32/int64), pandas Series, polars Series, or list of integers. \
         Tip: For status/group columns, ensure values are integers (0, 1, etc.).",
        type_name
    )))
}

pub fn extract_optional_vec_f64(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<f64>>> {
    match obj {
        Some(o) => Ok(Some(extract_vec_f64(o)?)),
        None => Ok(None),
    }
}

pub fn extract_optional_vec_i32(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<i32>>> {
    match obj {
        Some(o) => Ok(Some(extract_vec_i32(o)?)),
        None => Ok(None),
    }
}

#[allow(dead_code)]
pub fn array_f64_to_vec(arr: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    arr.as_slice().unwrap().to_vec()
}

#[allow(dead_code)]
pub fn array_i32_to_vec(arr: PyReadonlyArray1<'_, i32>) -> Vec<i32> {
    arr.as_slice().unwrap().to_vec()
}

#[allow(dead_code)]
pub fn array_i64_to_vec(arr: PyReadonlyArray1<'_, i64>) -> Vec<i64> {
    arr.as_slice().unwrap().to_vec()
}

#[allow(dead_code)]
pub fn vec_to_array_f64<'py>(py: Python<'py>, vec: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, vec)
}

#[allow(dead_code)]
pub fn vec_to_array_i32<'py>(py: Python<'py>, vec: Vec<i32>) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_vec(py, vec)
}

#[allow(dead_code, clippy::collapsible_if)]
pub fn extract_2d_vec_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(list) = obj.extract::<Bound<'_, PyList>>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list.iter() {
            result.push(extract_vec_f64(&item)?);
        }
        return Ok(result);
    }
    if let Ok(list) = obj.extract::<Vec<Vec<f64>>>() {
        return Ok(list);
    }
    if let Ok(values) = obj.getattr("values") {
        if let Ok(arr) = values.extract::<numpy::PyReadonlyArray2<'_, f64>>() {
            let shape = arr.shape();
            let slice = arr.as_slice()?;
            let mut result = Vec::with_capacity(shape[0]);
            for i in 0..shape[0] {
                let row: Vec<f64> = slice[i * shape[1]..(i + 1) * shape[1]].to_vec();
                result.push(row);
            }
            return Ok(result);
        }
    }
    let type_name = obj
        .get_type()
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "Cannot convert '{}' to 2D float array. Expected: 2D numpy array, pandas DataFrame, or list of lists. \
         Tip: For covariates, provide as [[x1_obs1, x2_obs1], [x1_obs2, x2_obs2], ...].",
        type_name
    )))
}
