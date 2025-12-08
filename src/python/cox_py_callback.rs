#![allow(dead_code)]
use pyo3::prelude::*;

pub fn cox_callback(
    _which: i32,
    _coef: &mut [f64],
    _first: &mut [f64],
    _second: &mut [f64],
    _penalty: &mut [f64],
    _flag: &mut [i32],
    _fexpr: &PyAny,
) -> PyResult<()> {
    // TODO: Fix PyO3 0.27 API compatibility - this function is currently not used
    // The issue is calling a Python callable from &PyAny in PyO3 0.27
    // Possible solutions:
    // 1. Change function signature to accept Bound<PyAny> or PyObject
    // 2. Use proper PyO3 0.27 API for calling callables
    Python::attach(|_py| {
        /* Temporarily disabled - needs PyO3 0.27 API fix
        let coef_vec: Vec<f64> = coef.iter().copied().collect();
        let coef_list = PyList::new(py, &coef_vec);
        let kwargs = PyDict::new(py);
        kwargs.set_item("which", which)?;

        // Call the Python function - use unsafe conversion (PyO3 0.27 compatibility)
        let py_obj = unsafe { PyObject::from_borrowed_ptr(py, fexpr.as_ptr()) };
        let result = py_obj.call(py, (coef_list,), Some(kwargs))?;
        let dict = result.downcast::<PyDict>()?;

        macro_rules! extract_values {
            ($key:expr, $rust_slice:expr, $pytype:ty) => {
                let py_values = dict
                    .get_item($key)?
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                            "Missing key: {}",
                            $key
                        ))
                    })?
                    .downcast::<PyList>()?;

                if py_values.len()? != $rust_slice.len() {
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

        let py_flags = dict
            .get_item("flag")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing key: flag"))?
            .downcast::<PyList>()?;

        if py_flags.len()? != flag.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid length for flag",
            ));
        }

        for (i, item) in py_flags.iter().enumerate() {
            flag[i] = if let Ok(b) = item.extract::<bool>() {
                b as i32
            } else {
                item.extract::<i32>()?
            };
        }
        */

        Ok(())
    })
}
