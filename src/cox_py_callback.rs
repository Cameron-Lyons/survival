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
    Python::with_gil(|py| {
        let coef_list = PyList::new(py, coef);
        let kwargs = PyDict::new(py);
        kwargs.set_item("which", which)?;

        let result = fexpr.call((coef_list,), Some(kwargs))?;
        let dict: &PyDict = result.downcast()?;

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

        Ok(())
    })
}
