use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

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
