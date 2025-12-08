#![allow(dead_code)]
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub fn cox_callback(
    which: i32,
    coef: &mut [f64],
    first: &mut [f64],
    second: &mut [f64],
    penalty: &mut [f64],
    flag: &mut [i32],
    fexpr: &PyAny,
) -> PyResult<()> {
    Python::attach(|py| {
        // Convert &PyAny to Bound<PyAny> for PyO3 0.27 API
        // Use Py::from_borrowed_ptr with raw pointer from PyNativeType
        let bound_fexpr = unsafe { 
            Py::from_borrowed_ptr(py, fexpr as *const _ as *mut _)
        }.into_bound(py);

        // Create coefficient list using PyO3 0.27 API
        let coef_vec: Vec<f64> = coef.iter().copied().collect();
        let coef_list = PyList::new(py, &coef_vec)?;

        // Create kwargs dictionary using PyO3 0.27 API
        let kwargs = PyDict::new(py);
        kwargs.set_item("which", which)?;

        // Call the Python callable using PyO3 0.27 Bound API with positional args and kwargs
        let result = bound_fexpr.call((coef_list.as_any(),), Some(&kwargs))?;
        
        // Downcast result to PyDict using PyO3 0.27 API (cast instead of downcast)
        let dict = result.cast::<PyDict>()?;

        macro_rules! extract_values {
            ($key:expr, $rust_slice:expr, $pytype:ty) => {
                let item = dict
                    .get_item($key)?
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Missing key: {}",
                            $key
                        ))
                    })?;
                let py_values = item.cast::<PyList>()?;

                if py_values.len() != $rust_slice.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid length for {}",
                        $key
                    )));
                }

                for (i, item) in py_values.iter().enumerate() {
                    $rust_slice[i] = item.extract::<$pytype>()?;
                }
            };
        }

        extract_values!("coef", coef, f64);
        extract_values!("first", first, f64);
        extract_values!("second", second, f64);
        extract_values!("penalty", penalty, f64);

        let flag_item = dict
            .get_item("flag")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key: flag"))?;
        let py_flags = flag_item.cast::<PyList>()?;

        if py_flags.len() != flag.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid length for flag",
            ));
        }

        for (i, item) in py_flags.iter().enumerate() {
            flag[i] = match item.extract::<bool>() {
                Ok(b) => b as i32,
                Err(_) => item.extract::<i32>()?,
            };
        }

        Ok(())
    })
}
