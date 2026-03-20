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
    if value <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{name} must be positive"
        )));
    }
    Ok(())
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
