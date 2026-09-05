use pyo3::prelude::*;

#[inline]
pub(crate) fn ensure_positive_usize(name: &str, value: usize) -> PyResult<()> {
    if value == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be positive"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ensure_positive_f64(name: &str, value: f64) -> PyResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be positive and finite"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ensure_positive_f32(name: &str, value: f64) -> PyResult<()> {
    let narrowed = value as f32;
    if !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be positive and finite as f32"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ensure_nonnegative_f32(name: &str, value: f64) -> PyResult<()> {
    if value < 0.0 || !(value as f32).is_finite() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be nonnegative and finite as f32"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ensure_positive_unit_interval(name: &str, value: f64) -> PyResult<()> {
    if !(value > 0.0 && value <= 1.0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be in (0, 1]"
        )));
    }
    Ok(())
}

pub(crate) fn ensure_vec_capacity<T>(name: &str, count: usize) -> PyResult<()> {
    let item_size = size_of::<T>();
    if item_size != 0 && count > isize::MAX as usize / item_size {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} is too large for its storage"
        )));
    }
    Ok(())
}

pub(crate) fn ensure_matrix_capacity<T>(name: &str, rows: usize, cols: usize) -> PyResult<()> {
    let count = rows.checked_mul(cols).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{name} dimensions overflow usize"))
    })?;
    ensure_vec_capacity::<T>(name, count)
}

#[inline]
pub(crate) fn ensure_open_unit_interval(name: &str, value: f64) -> PyResult<()> {
    if !(0.0..1.0).contains(&value) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be in [0, 1)"
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn ensure_closed_unit_interval(name: &str, value: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&value) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be in [0, 1]"
        )));
    }
    Ok(())
}
